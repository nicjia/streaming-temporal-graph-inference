"""
Continuous-time positional encoding via Bochner's theorem.

Discrete positional encodings assume events arrive on a grid. Streaming events
do not: the gap between two GDELT reports can be 40 seconds or 40 hours, and a
model that bins them into "steps" throws away exactly the signal we care about.
Bochner's theorem gives the continuous replacement.
"""

import numpy as np
import torch
import torch.nn as nn


class TimeEncode(nn.Module):
    r"""
    Maps a scalar time difference to a vector, so that dot products between
    encodings approximate a translation-invariant kernel over time.

    Bochner's theorem states that a continuous, translation-invariant,
    positive-definite kernel on the reals is the Fourier transform of a
    non-negative measure:

        K(t1, t2) = psi(t1 - t2) = E_{omega ~ p} [ exp(i * omega * (t1 - t2)) ]

    Drawing d frequencies from p and taking the real part turns that
    expectation into a finite-dimensional inner product:

        Phi(t) = sqrt(1/d) * [cos(w_1 t), sin(w_1 t), ..., cos(w_d t), sin(w_d t)]
        <Phi(t1), Phi(t2)> ~= psi(t1 - t2)

    Both halves are required. With cosines alone the encoding is even -- it
    cannot tell +delta from -delta -- and the inner product stops being a
    faithful estimate of the kernel. Pairing each cosine with its sine is what
    makes the map a rotation by angle (w * t), so the inner product of two
    encodings collapses to a function of the difference alone.

    Rather than sampling p once and freezing it, the frequencies are
    nn.Parameters: the model learns which timescales matter for the task, which
    is equivalent to learning the kernel's spectral density.

    Args:
        dimension: Output width. Must be even; d = dimension // 2 frequencies
            each contribute a (cos, sin) pair.
        max_timescale_decades: The initial frequencies span 10^0 down to
            10^-max_timescale_decades radians per time unit. With timestamps in
            seconds the default covers roughly one second to one century, so
            some component of the encoding is sensitive at whatever resolution
            the data actually varies on.
    """

    def __init__(self, dimension, max_timescale_decades=9.0):
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError(f"dimension must be even (got {dimension}); "
                             "each frequency contributes a cos and a sin")

        self.dimension = dimension
        self.num_frequencies = dimension // 2

        # Log-spaced init, not Xavier. Xavier would draw every frequency from
        # one narrow band, so the whole encoding would resolve a single
        # timescale and be blind to the others. A geometric ladder guarantees
        # coverage across decades from the first step, and gradient descent
        # only has to refine it.
        frequencies = 1.0 / np.logspace(0.0, max_timescale_decades,
                                        self.num_frequencies, dtype=np.float64)
        self.frequencies = nn.Parameter(torch.tensor(frequencies, dtype=torch.float32))

        # Monte-Carlo normalisation from the theorem: with 1/sqrt(d) scaling,
        # <Phi(t), Phi(t)> == 1 and the inner product is an unbiased estimate of
        # the kernel rather than something that grows with the width.
        self.register_buffer("scale", torch.tensor(1.0 / np.sqrt(self.num_frequencies),
                                                   dtype=torch.float32))

    def forward(self, delta_t):
        """
        Args:
            delta_t: Time differences, any shape (..., ). Typically
                (batch, num_neighbors) holding t_query - t_event.

        Returns:
            Tensor of shape (..., dimension).
        """
        # (..., 1) * (num_frequencies,) broadcasts to (..., num_frequencies).
        phase = delta_t.unsqueeze(-1).float() * self.frequencies

        # Interleaving vs concatenating is irrelevant to the maths -- the next
        # layer is a learned linear map either way -- so concatenate, which is
        # one fewer kernel launch.
        encoded = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return encoded * self.scale
