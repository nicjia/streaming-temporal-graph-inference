"""
Conflict-intensity forecasting on top of the TGAT encoder.

Why this exists, since it is not part of the original TGAT paper:

The link predictor is trained to *rank* partners -- given this country at this
time, which counterpart is real? That is the right objective for link
prediction, and the walk-forward evaluation uses it. It is the wrong thing to
build a cross-sectional trading signal from, and measurably so. Summing
sigma(score(c, v)) over partners to approximate "how much conflict is coming
for c" produced a panel whose variance was 98% a constant per-country offset:
the model had learned which bloc each country sits in, which is true, static,
and useless for deciding what to trade this week.

The problem is not the encoder, it is the head. Ranking losses are invariant to
anything that shifts all of one country's scores together, so nothing ever asks
the model to get a country's *level* of activity right over time. So here the
model is trained directly on the quantity the strategy needs: how many conflict
events will involve this country over the next few days. Same encoder, same
history, a supervised target instead of a derived one.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tgat import TGAT

CONFLICT_CLASSES = (3, 4)


class IntensityHead(nn.Module):
    """Maps a node embedding to a log event rate."""

    def __init__(self, node_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or node_dim
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings):
        return self.net(embeddings).squeeze(-1)


class TGATIntensityModel(nn.Module):
    """
    Predicts each country's conflict event count over the next `horizon_days`.

    The head emits a log rate and the loss is Poisson negative log-likelihood
    rather than MSE on the raw count. Event counts are non-negative, skewed, and
    have variance that grows with the mean; squared error on that target lets a
    handful of crisis days dominate every gradient, and nothing stops the model
    predicting a negative number of events. Poisson is the distribution the
    target actually comes from.
    """

    NUM_RECENCY_FEATURES = 4

    def __init__(self, num_nodes, sampler, use_recency_features=True, **kwargs):
        super().__init__()
        head_dropout = kwargs.pop("head_dropout", 0.1)
        self.encoder = TGAT(num_nodes, sampler, **kwargs)
        self.use_recency_features = use_recency_features

        # The attention embedding carries *who*; the recency features carry
        # *how often*. See PCSRTemporalSampler.recency_features for why the
        # second cannot come out of the first.
        width = self.encoder.node_dim
        if use_recency_features:
            width += self.NUM_RECENCY_FEATURES
        self.head = IntensityHead(width, dropout=head_dropout)

    def forward(self, node_ids, timestamps):
        """Returns the predicted log rate for each (node, time) query."""
        embeddings = self.encoder(node_ids, timestamps)

        if not self.use_recency_features:
            return self.head(embeddings)

        features = self.encoder.sampler.recency_features(
            np.asarray(node_ids, dtype=np.int64),
            np.asarray(timestamps, dtype=np.int64),
            self.encoder.num_neighbors)
        features = torch.as_tensor(features, dtype=torch.float32,
                                   device=embeddings.device)
        return self.head(torch.cat([embeddings, features], dim=-1))

    def loss(self, node_ids, timestamps, targets):
        log_rate = self.forward(node_ids, timestamps)
        target_tensor = torch.as_tensor(np.asarray(targets, dtype=np.float32),
                                        dtype=torch.float32,
                                        device=log_rate.device)
        return F.poisson_nll_loss(log_rate, target_tensor, log_input=True,
                                  full=False, reduction="mean")

    @torch.no_grad()
    def predict(self, node_ids, timestamps):
        self.eval()
        return self.forward(node_ids, timestamps).cpu().numpy()


def build_intensity_targets(events, countries, country_ids, dates,
                            horizon_days=5, cutoff_hour=0, max_ts=None):
    """
    Training examples for the intensity head.

    One example per (country, date): the query is that country as of the start
    of the date, and the target is the number of conflict events involving it
    over the following `horizon_days`.

    Dates whose forward window would run past `max_ts` are dropped. That bound
    is the fold's training cut-off, and dropping them is not optional -- keeping
    a date whose target is computed from events after the cut-off would train
    the model on the very future the walk-forward split exists to hide.

    Returns:
        (node_ids, timestamps, targets) as parallel arrays.
    """
    conflict = events[events["quad_class"].isin(CONFLICT_CLASSES)]

    index = {country: position for position, country in enumerate(countries)}
    day = pd.to_datetime(conflict["ts"], unit="s").dt.normalize()

    counts = {}
    for column in ("src", "dst"):
        grouped = conflict.groupby([day, conflict[column]]).size()
        for (event_day, country), value in grouped.items():
            if country in index:
                key = (event_day, country)
                counts[key] = counts.get(key, 0) + value

    node_ids, timestamps, targets = [], [], []
    horizon = pd.Timedelta(days=horizon_days)

    for date in dates:
        date = pd.Timestamp(date).normalize()
        cutoff = int((date + pd.Timedelta(hours=cutoff_hour)).timestamp())
        window_end = date + horizon
        if max_ts is not None and window_end.timestamp() > max_ts:
            continue

        window_days = pd.date_range(date, window_end, freq="D", inclusive="left")
        for country in countries:
            total = sum(counts.get((d, country), 0) for d in window_days)
            node_ids.append(country_ids[index[country]])
            timestamps.append(cutoff)
            targets.append(total)

    return (np.asarray(node_ids, dtype=np.int64),
            np.asarray(timestamps, dtype=np.int64),
            np.asarray(targets, dtype=np.float32))


def train_intensity(model, node_ids, timestamps, targets, epochs=3, batch_size=256,
                    learning_rate=1e-3, seed=0, quiet=False):
    """Fit the intensity head (and encoder) on prepared examples."""
    if len(node_ids) < batch_size:
        batch_size = max(16, len(node_ids) // 4)
    if len(node_ids) < 32:
        return []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    model.train()

    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(node_ids))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if len(batch) < 8:
                continue
            optimizer.zero_grad()
            loss = model.loss(node_ids[batch], timestamps[batch], targets[batch])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch_losses:
            losses.extend(epoch_losses)
            if not quiet:
                print(f"    intensity epoch {epoch + 1}/{epochs}  "
                      f"poisson nll {np.mean(epoch_losses):.4f}")
    return losses
