"""
Direct return forecasting on top of the TGAT encoder.

The intensity head (intensity.py) predicts how much conflict is coming and
leaves the mapping from conflict to positions to the strategy layer. This head
skips that step and predicts the tradable quantity itself: the forward return of
the instrument attached to each country.

Two decisions here do most of the work, and both are about making the target
learnable rather than about the network.

**Predict relative return, not absolute.** The dominant component of any equity
return is the market move that day, which is common to every name and is not
something a geopolitical event graph can forecast. Left in the target it is pure
variance -- the model spends its capacity failing to predict beta. So the label
is cross-sectionally demeaned: the return of a country's ETF minus the
equal-weight return of the universe that day. That is also exactly what a
dollar-neutral book earns, so the training target and the backtest agree.

**Standardise and winsorise.** Daily cross-sectional return dispersion is not
stationary; a crisis week has several times the spread of a quiet one. Without
scaling, a handful of days dominate the gradient. The target is divided by a
*trailing* cross-sectional dispersion (never the contemporaneous one, which
would leak) and clipped.

Be clear-eyed about what this head is up against. Cross-sectional daily equity
returns are close to noise: an information coefficient of 0.03 is a real signal
in production. With a few thousand training examples, the honest prior is that
this finds nothing, and the pipeline reports it against baselines and a random
control so that outcome is visible rather than flattering.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tgat import TGAT


class ReturnForecastModel(nn.Module):
    """
    TGAT encoder plus a linear head over the node embedding.

    Args:
        task: "regression" predicts the standardised relative return and trains
            with Huber loss; "classification" predicts P(outperforms) and trains
            with BCE. Huber rather than plain MSE because return targets keep
            fat tails even after winsorising, and squared error hands those tails
            the whole gradient.
        use_recency_features: Append the sampler's rate features to the
            embedding, as the intensity head does. Attention cannot represent
            event rate (see PCSRTemporalSampler.recency_features), and rate is
            plausibly the part of the graph state that matters here.
    """

    NUM_RECENCY_FEATURES = 4

    def __init__(self, num_nodes, sampler, task="regression",
                 use_recency_features=True, head_hidden=None, head_dropout=0.1,
                 **kwargs):
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be regression or classification, got {task!r}")

        self.task = task
        self.encoder = TGAT(num_nodes, sampler, **kwargs)
        self.use_recency_features = use_recency_features

        width = self.encoder.node_dim
        if use_recency_features:
            width += self.NUM_RECENCY_FEATURES
        hidden = head_hidden or width

        self.head = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )

    def _features(self, node_ids, timestamps):
        embeddings = self.encoder(node_ids, timestamps)
        if not self.use_recency_features:
            return embeddings

        recency = self.encoder.sampler.recency_features(
            np.asarray(node_ids, dtype=np.int64),
            np.asarray(timestamps, dtype=np.int64),
            self.encoder.num_neighbors)
        recency = torch.as_tensor(recency, dtype=torch.float32,
                                  device=embeddings.device)
        return torch.cat([embeddings, recency], dim=-1)

    def forward(self, node_ids, timestamps):
        """
        Raw head output.

        For regression this is the predicted standardised relative return; for
        classification it is a logit, so apply a sigmoid to read it as a
        probability.
        """
        return self.head(self._features(node_ids, timestamps)).squeeze(-1)

    def loss(self, node_ids, timestamps, targets):
        prediction = self.forward(node_ids, timestamps)
        target_tensor = torch.as_tensor(np.asarray(targets, dtype=np.float32),
                                        dtype=torch.float32,
                                        device=prediction.device)
        if self.task == "regression":
            return F.huber_loss(prediction, target_tensor, delta=1.0)
        return F.binary_cross_entropy_with_logits(prediction,
                                                  (target_tensor > 0).float())

    @torch.no_grad()
    def predict(self, node_ids, timestamps):
        """Signal values: predicted relative return, or P(outperform) - 0.5."""
        self.eval()
        raw = self.forward(node_ids, timestamps).cpu().numpy()
        if self.task == "classification":
            return 1.0 / (1.0 + np.exp(-raw)) - 0.5
        return raw


def build_return_targets(prices, universe, countries, country_ids, dates,
                         horizon_days=1, execution_lag_days=1, cutoff_hour=0,
                         demean=True, dispersion_window=60, clip=3.0,
                         max_ts=None):
    """
    Forward relative-return labels, aligned to how the backtest actually trades.

    For a signal dated d the book is entered at the close of d + lag and held
    for `horizon_days`, so the label is

        close(d + lag + horizon) / close(d + lag) - 1

    cross-sectionally demeaned, then divided by a trailing dispersion estimate.

    The `max_ts` bound drops any date whose forward window ends after the
    training cut-off. This is the single most important line in the function:
    without it the model trains on returns from inside its own test window and
    every downstream number becomes fiction.

    Returns:
        (node_ids, timestamps, targets, frame) where frame carries the
        per-(date, country) labels for inspection.
    """
    wide = prices.copy()
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.pivot_table(index="date", columns="ticker", values="close",
                            aggfunc="last").sort_index()

    tickers = {c: universe.ticker(c) for c in countries}
    tickers = {c: t for c, t in tickers.items() if t is not None and t in wide.columns}
    if not tickers:
        raise ValueError("no country in `countries` maps to a ticker with prices")

    usable = [c for c in countries if c in tickers]
    panel = wide[[tickers[c] for c in usable]]
    panel.columns = usable

    # Return over the holding period, indexed by the date the position opens.
    horizon_return = panel.shift(-horizon_days) / panel - 1.0

    # Move it back onto the signal date: a signal at d opens at d + lag.
    label = horizon_return.shift(-execution_lag_days)

    if demean:
        label = label.sub(label.mean(axis=1), axis=0)

    if dispersion_window:
        # Trailing dispersion only. Using the contemporaneous cross-sectional
        # standard deviation would scale each day by information from that day.
        dispersion = (label.std(axis=1, ddof=0)
                      .rolling(dispersion_window, min_periods=10)
                      .mean()
                      .shift(1))
        label = label.div(dispersion.replace(0.0, np.nan), axis=0)

    if clip:
        label = label.clip(-clip, clip)

    index = {country: position for position, country in enumerate(usable)}
    horizon_seconds = (horizon_days + execution_lag_days + 1) * 86400

    node_ids, timestamps, targets, rows = [], [], [], []
    for date in dates:
        date = pd.Timestamp(date).normalize()
        if date not in label.index:
            continue
        cutoff = int((date + pd.Timedelta(hours=cutoff_hour)).timestamp())
        if max_ts is not None and cutoff + horizon_seconds > max_ts:
            continue

        row = label.loc[date]
        for country in usable:
            value = row.get(country, np.nan)
            if not np.isfinite(value):
                continue
            node_ids.append(country_ids[countries.index(country)]
                            if isinstance(country_ids, list)
                            else country_ids[index[country]])
            timestamps.append(cutoff)
            targets.append(float(value))
            rows.append((date.date().isoformat(), country, float(value)))

    frame = pd.DataFrame(rows, columns=["date", "country", "target"])
    return (np.asarray(node_ids, dtype=np.int64),
            np.asarray(timestamps, dtype=np.int64),
            np.asarray(targets, dtype=np.float32),
            frame)


def train_forecast(model, node_ids, timestamps, targets, epochs=20,
                   batch_size=256, learning_rate=1e-3, weight_decay=1e-4,
                   seed=0, quiet=False):
    """
    Fit the return head.

    Weight decay is on by default here and not on the intensity head. The
    intensity target is a real, strongly autocorrelated signal; a return target
    is mostly noise, and an unregularised network will happily memorise which
    country did well during the training window and carry that into the test
    window as a spurious tilt.
    """
    if len(node_ids) < batch_size:
        batch_size = max(16, len(node_ids) // 4)
    if len(node_ids) < 32:
        return []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
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
                print(f"    forecast epoch {epoch + 1}/{epochs}  "
                      f"loss {np.mean(epoch_losses):.4f}")
    return losses


def forecast_signal(model, countries, country_ids, dates, cutoff_hour=0,
                    batch_size=512):
    """Per-country predicted relative return over `dates`."""
    node_ids, timestamps, index = [], [], []
    for date in dates:
        cutoff = int((pd.Timestamp(date).normalize() +
                      pd.Timedelta(hours=cutoff_hour)).timestamp())
        for country, node_id in zip(countries, country_ids):
            node_ids.append(node_id)
            timestamps.append(cutoff)
            index.append((pd.Timestamp(date).date().isoformat(), country))

    node_ids = np.asarray(node_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.int64)

    predictions = []
    for start in range(0, len(node_ids), batch_size):
        stop = start + batch_size
        predictions.append(model.predict(node_ids[start:stop], timestamps[start:stop]))
    predictions = np.concatenate(predictions) if predictions else np.array([])

    frame = pd.DataFrame(index, columns=["date", "country"])
    frame["signal"] = predictions
    return frame


def information_coefficient(predictions, realised):
    """
    Mean daily cross-sectional rank correlation between prediction and outcome.

    The standard way to judge a cross-sectional forecast, and far more
    informative than a loss value: it says whether the ordering is right, which
    is all a relative-value book needs. Rank correlation rather than Pearson so
    one crisis name cannot carry the number.
    """
    merged = predictions.merge(realised, on=["date", "country"], suffixes=("_p", "_r"))
    coefficients = []
    for _, group in merged.groupby("date"):
        if len(group) < 4:
            continue
        p = group["signal"].rank()
        r = group["target"].rank()
        if p.std() > 0 and r.std() > 0:
            coefficients.append(np.corrcoef(p, r)[0, 1])
    if not coefficients:
        return float("nan"), 0
    return float(np.mean(coefficients)), len(coefficients)
