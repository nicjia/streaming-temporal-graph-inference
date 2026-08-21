"""
Temporal Graph Attention Network (TGAT).

Attention over a node's *past*, with the elapsed time since each past
interaction folded into the attention computation itself rather than bolted on
afterwards. Two events one minute apart and two events one month apart produce
different attention weights because the time encoding feeding the query/key
product differs, not because of any hand-written decay term.

A note on PyTorch Geometric: this uses plain nn.Modules rather than PyG's
MessagePassing. PyG's sparse-scatter formulation assumes one static edge_index
per forward pass, while TGAT needs a *different* neighbourhood per (node, time)
query -- the same node queried at two times has two different histories. Fixed
fan-out sampling gives dense (batch, K, dim) tensors, which map onto batched
matmuls and keep the temporal masking explicit and auditable. PyG remains the
right tool for the static-graph layers around this.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoding import TimeEncode


class TemporalAttentionLayer(nn.Module):
    """
    One round of multi-head attention from a node onto its sampled history.

    The query is built from the target node *at the query time*; keys and
    values are built from each sampled neighbour together with an encoding of
    how long ago that interaction happened. Because the time encoding is part
    of the key, the dot product q . k is a function of both semantic similarity
    and elapsed time -- the "time-decaying attention" is learned in that
    product rather than imposed.

    Args:
        node_dim: Width of the incoming node representations.
        time_dim: Width of the time encoding.
        out_dim: Width of the output representation.
        num_heads: Attention heads. out_dim must be divisible by this.
        dropout: Applied to attention weights and to the output projection.
    """

    def __init__(self, node_dim, time_dim, out_dim, num_heads=2, dropout=0.1,
                 relation_dim=0):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by "
                             f"num_heads ({num_heads})")

        self.node_dim = node_dim
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        # Scaled dot-product: without the 1/sqrt(d) the logits grow with width
        # and softmax saturates into a near-one-hot, killing the gradient.
        self.scale = self.head_dim ** -0.5

        # The relation encoding joins the key and value but not the query: the
        # query is the target node asking "what happened to me", which has no
        # relation of its own. Putting it in the key is what lets the attention
        # logit depend on the *kind* of interaction as well as who and when --
        # a sanction and a summit between the same pair at the same lag can now
        # receive different weight, which the model previously could not express.
        self.relation_dim = relation_dim
        query_dim = node_dim + time_dim
        key_dim = node_dim + time_dim + relation_dim

        self.q_proj = nn.Linear(query_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(key_dim, out_dim, bias=False)

        # Concatenating the target's own representation back in before the
        # merge is what makes this a graph *convolution* rather than pure
        # pooling: a node keeps its own signal even when its history is
        # uninformative.
        self.merger = nn.Sequential(
            nn.Linear(out_dim + node_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Residual only type-checks when the widths agree; otherwise project.
        self.residual = (nn.Identity() if node_dim == out_dim
                         else nn.Linear(node_dim, out_dim, bias=False))

    def forward(self, target_h, target_time_enc, neighbor_h, neighbor_time_enc, mask,
                neighbor_relation_enc=None):
        """
        Args:
            target_h: (B, node_dim) representation of each query node.
            target_time_enc: (B, time_dim), the encoding of delta_t = 0.
            neighbor_h: (B, K, node_dim).
            neighbor_time_enc: (B, K, time_dim), encoding t_query - t_event.
            mask: (B, K) bool, True where the slot holds a real neighbour.

        Returns:
            (B, out_dim)
        """
        batch, num_neighbors, _ = neighbor_h.shape

        query_in = torch.cat([target_h, target_time_enc], dim=-1)
        kv_parts = [neighbor_h, neighbor_time_enc]
        if self.relation_dim:
            if neighbor_relation_enc is None:
                raise ValueError("layer was built with relation_dim > 0 but no "
                                 "relation encoding was supplied")
            kv_parts.append(neighbor_relation_enc)
        kv_in = torch.cat(kv_parts, dim=-1)

        # (B, H, 1, dh) and (B, H, K, dh)
        q = self.q_proj(query_in).view(batch, self.num_heads, 1, self.head_dim)
        k = self.k_proj(kv_in).view(batch, num_neighbors, self.num_heads,
                                    self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(batch, num_neighbors, self.num_heads,
                                    self.head_dim).transpose(1, 2)

        scores = (q * k).sum(dim=-1) * self.scale          # (B, H, K)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # A node with no admissible history has every slot masked, so the row
        # is all -inf and softmax would return NaN and poison the backward pass.
        # Neutralise those rows first, then zero their context afterwards --
        # "no history" has to be a representable state, not a crash, because
        # every node looks like this on its first event.
        empty = ~mask.any(dim=-1)                           # (B,)
        scores = torch.where(empty[:, None, None], torch.zeros_like(scores), scores)

        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = (weights.unsqueeze(-1) * v).sum(dim=2)    # (B, H, dh)
        context = context.reshape(batch, self.out_dim)
        context = torch.where(empty[:, None], torch.zeros_like(context), context)

        merged = self.merger(torch.cat([context, target_h], dim=-1))
        return self.norm(self.residual(target_h) + self.out_dropout(merged))


class TGAT(nn.Module):
    """
    Stacked temporal attention: embeds a (node, time) pair by recursively
    attending over history.

    Layer 1 embeds each sampled neighbour *at the time that neighbour was
    interacted with* -- not at the query time. That is the whole point of the
    recursion: when actor A is queried at time t and its history contains B at
    time t' < t, what A should see is B as B was at t', built from B's own
    history before t'. Embedding B at t would let information from (t', t) flow
    backwards into a prediction about t.

    Cost is K^L node-time queries per root, so K and num_layers are the two
    dials that matter. K=20, L=2 is the paper's setting.

    Args:
        num_nodes: Vertex count; sizes the learned embedding table.
        sampler: A PCSRTemporalSampler over the C++ graph.
        node_dim: Width of node representations throughout.
        time_dim: Width of the time encoding. Must be even.
        num_layers: Attention rounds (= hops of history).
        num_neighbors: Fan-out K sampled per node per layer.
        num_heads, dropout: Passed to each attention layer.
        strategy: "recent" or "uniform" neighbour selection.
        node_features: Optional (num_nodes, node_dim) float array to use
            instead of a learned table. GDELT actors carry no features, so the
            default learns them.
    """

    def __init__(self, num_nodes, sampler, node_dim=64, time_dim=64, num_layers=2,
                 num_neighbors=20, num_heads=2, dropout=0.1, strategy="recent",
                 node_features=None, num_relations=0, relation_dim=16):
        super().__init__()
        self.num_nodes = num_nodes
        self.sampler = sampler
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.strategy = strategy

        if node_features is None:
            self.node_embedding = nn.Embedding(num_nodes, node_dim)
            nn.init.normal_(self.node_embedding.weight, std=0.1)
        else:
            features = torch.as_tensor(np.asarray(node_features), dtype=torch.float32)
            if features.shape != (num_nodes, node_dim):
                raise ValueError(f"node_features must be ({num_nodes}, {node_dim}), "
                                 f"got {tuple(features.shape)}")
            self.node_embedding = nn.Embedding.from_pretrained(features, freeze=True)

        # One shared time encoder across layers: elapsed time means the same
        # thing at every hop, so learning a separate spectrum per layer would
        # just split the gradient signal.
        self.time_encoder = TimeEncode(time_dim)

        # Relation types are a small unordered vocabulary (GDELT's 20 event root
        # codes, or 4 quad classes), so a learned embedding table -- not a
        # one-hot, and certainly not the integer code as a scalar, which would
        # assert that code 7 sits between 6 and 8.
        self.num_relations = num_relations
        self.relation_dim = relation_dim if num_relations else 0
        self.relation_embedding = (nn.Embedding(num_relations, relation_dim)
                                   if num_relations else None)
        if self.relation_embedding is not None:
            nn.init.normal_(self.relation_embedding.weight, std=0.1)

        self.layers = nn.ModuleList([
            TemporalAttentionLayer(node_dim, time_dim, node_dim, num_heads, dropout,
                                   relation_dim=self.relation_dim)
            for _ in range(num_layers)
        ])

        self._rng = np.random.default_rng(0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, node_ids, timestamps):
        """
        Args:
            node_ids: (B,) integer vertex ids (array or tensor).
            timestamps: (B,) query times, same units as the stored edges.

        Returns:
            (B, node_dim) embeddings of each node as of its query time.
        """
        nodes = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        times = np.asarray(timestamps, dtype=np.int64).reshape(-1)
        return self._embed(nodes, times, self.num_layers)

    def _embed(self, nodes, times, depth):
        node_tensor = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        h = self.node_embedding(node_tensor)

        if depth == 0:
            return h

        sampled = self.sampler.sample(
            nodes, times, self.num_neighbors,
            strategy=self.strategy,
            rng=self._rng if self.strategy == "uniform" else None,
            with_relations=self.num_relations > 0,
        )
        if self.num_relations:
            neighbor_ids, neighbor_times, mask, neighbor_relations = sampled
        else:
            neighbor_ids, neighbor_times, mask = sampled
            neighbor_relations = None

        batch = nodes.shape[0]
        flat_ids = neighbor_ids.reshape(-1).astype(np.int64)
        flat_times = neighbor_times.reshape(-1)

        # Recurse: each neighbour is embedded as of its own interaction time.
        neighbor_h = self._embed(flat_ids, flat_times, depth - 1)
        neighbor_h = neighbor_h.view(batch, self.num_neighbors, self.node_dim)

        # delta_t is computed in int64 before hitting float. GDELT timestamps
        # are ~1.7e9 seconds; float32 has 24 bits of mantissa, so subtracting
        # two of them after conversion loses everything below ~128 s -- which is
        # smaller than the gaps this model is supposed to distinguish.
        delta_t = times[:, None] - neighbor_times
        delta_t = np.where(mask, delta_t, 0)

        delta_tensor = torch.as_tensor(delta_t, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        neighbor_time_enc = self.time_encoder(delta_tensor)
        target_time_enc = self.time_encoder(
            torch.zeros(batch, dtype=torch.float32, device=self.device))

        relation_enc = None
        if self.num_relations:
            codes = np.clip(neighbor_relations.astype(np.int64), 0,
                            self.num_relations - 1)
            relation_enc = self.relation_embedding(
                torch.as_tensor(codes, dtype=torch.long, device=self.device))

        return self.layers[depth - 1](h, target_time_enc, neighbor_h,
                                      neighbor_time_enc, mask_tensor, relation_enc)


class LinkPredictor(nn.Module):
    """
    Scores an ordered pair of embeddings as a logit for "this edge forms".

    Concatenation rather than a dot product: the graph is directed, and a dot
    product is symmetric, so it could not distinguish "A sanctions B" from
    "B sanctions A".
    """

    def __init__(self, node_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or node_dim
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_h, dst_h):
        return self.net(torch.cat([src_h, dst_h], dim=-1)).squeeze(-1)


class TGATLinkModel(nn.Module):
    """
    TGAT encoder plus link predictor, trained by contrast against corrupted
    destinations.

    Training signal: for each real event (u, v, t), score (u, v, t) against
    (u, v_negative, t). There is no "non-edge" ground truth in a stream -- the
    absence of a GDELT report is not evidence of no interaction -- so the
    objective is ranking observed events above sampled ones rather than
    classifying edges as present or absent.
    """

    def __init__(self, num_nodes, sampler, **kwargs):
        super().__init__()
        predictor_dropout = kwargs.pop("predictor_dropout", 0.1)
        self.encoder = TGAT(num_nodes, sampler, **kwargs)
        self.predictor = LinkPredictor(self.encoder.node_dim, dropout=predictor_dropout)
        self.num_nodes = num_nodes

    def score(self, src, dst, times):
        """Logits for the given (src, dst, time) triples."""
        # One encoder pass over the concatenation, not two passes: the second
        # hop dominates the cost and batching the roots halves it.
        both_nodes = np.concatenate([np.asarray(src, dtype=np.int64),
                                     np.asarray(dst, dtype=np.int64)])
        both_times = np.concatenate([np.asarray(times, dtype=np.int64),
                                     np.asarray(times, dtype=np.int64)])
        embeddings = self.encoder(both_nodes, both_times)
        src_h, dst_h = embeddings.chunk(2, dim=0)
        return self.predictor(src_h, dst_h)

    def loss(self, src, dst, times, negative_dst, negative_src=None):
        """
        Binary cross-entropy over one positive and one or two negatives per event.

        `negative_dst` corrupts the destination, which trains the ranking task:
        given this source at this time, which target is real? That is the right
        objective for link prediction and it is all you need if ranking is all
        you want.

        `negative_src` corrupts the source, and it matters as soon as you want
        to *compare scores across sources* -- for instance to ask which country
        the model expects to be involved in the most new conflict. With
        destination-only negatives, the positive and the negative share a
        source, so nothing in the gradient ever asks the model to distinguish a
        busy source from a quiet one; sigma(score(u, .)) ends up uncalibrated
        between different u and a cross-sectional signal built from it is
        noise. Supplying source negatives fixes that, at the cost of one more
        embedding batch.

        All embeddings come from a single encoder call. The source embeddings
        are shared between the positive and the destination-negative pair, so
        computing them separately would waste the expensive half of the pass.
        """
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        negative_dst = np.asarray(negative_dst, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)

        groups = [src, dst, negative_dst]
        if negative_src is not None:
            groups.append(np.asarray(negative_src, dtype=np.int64))

        nodes = np.concatenate(groups)
        query_times = np.concatenate([times] * len(groups))
        embeddings = self.encoder(nodes, query_times)
        chunks = embeddings.chunk(len(groups), dim=0)

        src_h, dst_h, neg_dst_h = chunks[0], chunks[1], chunks[2]

        pos_logits = self.predictor(src_h, dst_h)
        neg_logits = self.predictor(src_h, neg_dst_h)

        all_negatives = [neg_logits]
        if negative_src is not None:
            all_negatives.append(self.predictor(chunks[3], dst_h))

        negative_logits = torch.cat(all_negatives)
        logits = torch.cat([pos_logits, negative_logits])
        labels = torch.cat([torch.ones_like(pos_logits),
                            torch.zeros_like(negative_logits)])
        return F.binary_cross_entropy_with_logits(logits, labels), pos_logits, neg_logits
