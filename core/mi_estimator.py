# core/mi_estimator.py
#
# InfoNCE mutual information estimator.
#
# Theory
# ------
# InfoNCE gives a lower bound on MI:
#   I(X; Z) >= log(N) - CE(logits, labels)
# where logits[i,j] = <f(x_i), g(z_j)> / temperature
# and labels = [0, 1, 2, ..., N-1]  (diagonal positives).
#
# The bound tightens as the critic (f, g) improves.  The upper limit is
# log(batch_size), so kappa saturates at log(512) / d_z with N=512.
#
# Stability tweak
# ---------------
# Instead of returning the single best MI seen during training (which
# fluctuates), we average the estimates from the last 100 steps.  This
# trades a tiny downward bias for substantially lower variance, which
# matters when comparing architectures that differ by small amounts.
#
# Reproducibility
# ---------------
# seed is passed to the critic's weight initialisation and the per-step
# randperm, so two calls with the same inputs and seed return the same value.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import batch size constant to keep in sync with kappa.py saturation check.
try:
    from .config import INFONCE_BATCH as _INFONCE_BATCH
except ImportError:
    _INFONCE_BATCH = 512


class _Critic(nn.Module):
    """
    Two-layer MLP that projects inputs into a shared embedding space.
    Outputs are L2-normalised so the dot product is a cosine similarity,
    which keeps the logit scale stable regardless of embedding magnitude.
    """
    def __init__(self, d_in: int, d_embed: int, d_hidden: int):
        """Two-layer MLP projector with L2-normalised output."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project x and L2-normalise so dot product = cosine similarity."""
        return F.normalize(self.net(x), dim=-1)


class InfoNCE(nn.Module):
    """InfoNCE mutual information estimator.

    Trains two MLP critics f(x), g(z) to maximise the InfoNCE lower bound
    I(X;Z) >= log(N) - CE(logits, diag_labels).
    MI is averaged over the last 100 training steps for stability.
    Returns estimated MI in nats; upper bound = log(batch_size).
    """

    def __init__(
        self,
        d_x:         int,
        d_z:         int,
        hidden:      int   = 256,
        temperature: float = 0.1,
        lr:          float = 1e-3,
        steps:       int   = 2000,
        seed:        int   = 0,
    ):
        """Initialise two MLP critics and a shared Adam optimiser."""
        super().__init__()
        self.temperature = temperature
        self.steps       = steps
        self.seed        = seed

        # separate critics for X and Z; shared embedding dimension = hidden
        self.critic_x = _Critic(d_x, hidden, hidden)
        self.critic_z = _Critic(d_z, hidden, hidden)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def estimate(self, x: torch.Tensor, z: torch.Tensor) -> float:
        """
        Estimate MI(X; Z) in nats.

        x: [N, d_x]   must already be on the correct device
        z: [N, d_z]   must already be on the correct device

        Returns a non-negative float.  The hard upper bound is log(batch_size).
        """
        if x.shape[0] != z.shape[0]:
            raise ValueError(
                f"batch size mismatch: x has {x.shape[0]}, z has {z.shape[0]}"
            )
        if x.shape[0] < 2:
            raise ValueError("need at least 2 samples for InfoNCE")

        # Re-seeding here controls the randperm sequence; critic weights
        # are already set (they were initialised when build_estimator was
        # called after torch.manual_seed in kappa.py).
        torch.manual_seed(self.seed)

        self.train()
        N          = x.shape[0]
        batch_size = min(_INFONCE_BATCH, N)
        log_N      = math.log(batch_size)

        # Average the last 100 estimates rather than returning the single
        # best, which fluctuates.  Requires est_steps >= 101.
        if self.steps < 101:
            raise ValueError(f"est_steps must be >= 101, got {self.steps}")
        tail: list = []

        for step in range(self.steps):
            idx    = torch.randperm(N, device=x.device)[:batch_size]
            hx     = self.critic_x(x[idx])              # [B, hidden]
            hz     = self.critic_z(z[idx])              # [B, hidden]
            logits = torch.matmul(hx, hz.T) / self.temperature   # [B, B]
            labels = torch.arange(batch_size, device=x.device)
            loss   = F.cross_entropy(logits, labels)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if step >= self.steps - 100:
                tail.append(log_N - loss.item())

        # average over tail; floor at 0 because MI >= 0 by definition
        mi = sum(tail) / len(tail) if tail else 0.0
        return max(0.0, float(mi))


def build_estimator(
    d_x:         int,
    d_z:         int,
    device:      torch.device,
    hidden:      int   = 256,
    steps:       int   = 2000,
    lr:          float = 1e-3,
    temperature: float = 0.1,
    seed:        int   = 0,
) -> InfoNCE:
    """Create and return a fresh InfoNCE estimator placed on device.

    A new estimator must be created per measure_kappa call;
    reusing a trained critic across conditions would corrupt the estimate.
    """
    return InfoNCE(d_x, d_z, hidden, temperature, lr, steps, seed).to(device)
