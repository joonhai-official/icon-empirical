# core/noise_channel.py
#
# Additive Gaussian noise channel used to perturb layer outputs before
# mutual information estimation.
#
# Formula
# -------
#   Z_tilde = Z + sigma * RMS(Z) * epsilon,   epsilon ~ N(0, I)
#   RMS(Z)  = sqrt( mean(Z^2) )               global scalar over batch x dim
#
# Scaling by RMS(Z) makes the noise level relative to the signal amplitude,
# so kappa stays comparable across architectures with very different output
# magnitudes.  sigma is the only free parameter the user controls.
#
# Reproducibility
# ---------------
# If seed is given, a Generator is created on first call and reused.
# Call reset() between independent measurements to re-seed cleanly.

import torch
import torch.nn as nn
from typing import Optional


class NoiseChannel(nn.Module):
    """Additive Gaussian noise channel: Z̃ = Z + σ·RMS(Z)·ε,  ε ~ N(0,I).

    Scaling noise by RMS(Z) keeps kappa comparable across architectures
    with different output magnitudes.
    """

    def __init__(self, sigma: float = 0.1, seed: Optional[int] = None):
        """sigma: noise level; seed: RNG seed for reproducibility."""
        super().__init__()
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.seed  = seed
        self._gen: Optional[torch.Generator] = None

    def rms(self, Z: torch.Tensor) -> torch.Tensor:
        """RMS(Z) = sqrt(mean(Z²)) — global scalar over [N, d].

        Clamped at 1e-8 to prevent division-by-zero in the caller.
        """
        # clamp prevents division by zero in the caller if Z is all-zero
        return Z.pow(2).mean().sqrt().clamp(min=1e-8)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z must be 2-D [N, d].  Pooling to 2-D happens in kappa.py before
        this is called, so we can enforce the shape strictly here.
        """
        if self.sigma == 0.0:
            return Z

        if Z.dim() != 2:
            raise ValueError(
                f"NoiseChannel expects [N, d] input, got shape {tuple(Z.shape)}"
            )

        # lazy Generator creation — waits until we know the device
        if self.seed is not None and self._gen is None:
            self._gen = torch.Generator(device=Z.device)
            self._gen.manual_seed(self.seed)

        noise_std = self.sigma * self.rms(Z)
        eps = torch.randn(Z.shape, device=Z.device, dtype=Z.dtype,
                          generator=self._gen)
        return Z + noise_std * eps

    def reset(self) -> None:
        """Drop the cached Generator so the next call re-initialises from seed."""
        self._gen = None
