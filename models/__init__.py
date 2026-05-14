# models/__init__.py
#
# All eight Icon_Empirical architectures in one file.
#
# Common interface
# ----------------
# Every model exposes:
#   forward(x, taps=None) -> logits  [B, n_classes]
#   taps dict keys: "input", "L{i}_ffn_out", "ffn_out"
#                   (GRU: layer hidden states; parallel: "B{i}_ffn_out")
#
# The taps dict is the bridge between training and kappa measurement.
# After a forward pass with taps={}, every intermediate representation
# is available for layerwise kappa computation.
#
# Width / depth conventions
# -------------------------
#   MLP (Mixer)  : width = d_model, depth = n_blocks (each block = token+channel mix)
#   CNN          : width = base channels, depth = total residual blocks
#   GRU          : width = hidden_size, depth = num_layers (capped at 8)
#   Transformer  : width = d_model, depth = n_layers
#   Serial / Parallel: width = d_model, depth = n_blocks / n_branches
#   Boltzmann    : width = n_hidden, depth = 1 (single visible→hidden layer)
#
# d_z unification
# ---------------
# All architectures produce ffn_out of shape [B, width] (d_z = width).
# CNN adds a Linear(width*4, width) projection after GAP so its d_z
# matches all other architectures, enabling direct C_arch comparison.
#
# Activation
# ----------
# All architectures accept activation in {"relu", "gelu", "tanh"}.
# For GRU, activation applies to the input embedding only (GRU gates are fixed).
# For Boltzmann, activation is unused (sigmoid is fixed by the RBM formulation).
#
# Reproducibility
# ---------------
# Every weight matrix is orthogonally initialised with a seed derived from
# the model-level seed.  Two models built with the same
# (arch, width, depth, activation, seed) are bitwise identical.
#
# Transformer design notes
# ------------------------
# - Learnable positional embedding [1, 64, width] added after patch embedding.
# - Standard Pre-LN formulation (no 1/sqrt(2) rescaling).
# - head_dim = 64 (single head for width < 64).
# - Bidirectional attention (no causal mask) — encoder style.
#
# GRU design notes
# ----------------
# - Input: 4x4 patch tokenization [B, 64, 48] (matches MLP/Transformer).
# - Classification: mean-pool over all 64 patch output states.
# - Depth capped at 8 layers (OOM prevention). depth=8,16,32 are identical.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_DIM = 64    # default attention head dimension


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _make_act(name: str) -> nn.Module:
    # Return a new instance each call so each layer owns its own module.
    # This avoids sharing the same nn.Module across ModuleList entries,
    # which can cause unexpected behaviour during model inspection.
    if name == "relu":  return nn.ReLU()
    if name == "gelu":  return nn.GELU()
    if name == "tanh":  return nn.Tanh()
    raise ValueError(f"activation must be one of ['relu','gelu','tanh'], got '{name}'")


def _ortho(layer: nn.Linear, seed: int) -> None:
    """Orthogonal init with a fixed seed for reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    nn.init.orthogonal_(layer.weight, generator=g)


def _pool_seq(h: torch.Tensor) -> torch.Tensor:
    """[B, T, D] -> [B, D]  mean over sequence."""
    return h.mean(dim=1)


def _pool_spatial(h: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, C]  global average pool."""
    return h.mean(dim=[2, 3])


def _n_heads(width: int) -> int:
    """Number of attention heads.  Single head for sub-64 widths."""
    return max(1, width // HEAD_DIM)


def _head_dim(width: int) -> int:
    """Per-head dimension.  Equals width when width < HEAD_DIM."""
    return width if width < HEAD_DIM else HEAD_DIM


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLPModel(nn.Module):
    """
    MLP-Mixer style architecture: patch-embed then alternating token-mixing
    and channel-mixing MLP blocks with residual connections.

    Each block:
      1. Token-mixing : LN -> transpose -> Linear(tokens,tokens) -> act
                        -> Linear(tokens,tokens) -> transpose -> residual
      2. Channel-mixing: LN -> Linear(w, 4w) -> act -> Linear(4w, w) -> residual

    Token count: 8x8 = 64 (4x4 patches from 32x32 input).
    Token-mixing allows spatial information exchange across all positions.
    Channel-mixing transforms each token's feature vector independently.

    Taps: L{i}_ffn_out after each block's channel-mixing residual.
    ffn_out = mean-pooled final representation.
    """

    N_TOKENS = 64   # 8x8 spatial grid

    def __init__(
        self,
        width:      int   = 256,
        depth:      int   = 4,
        activation: str   = "relu",
        n_classes:  int   = 10,
        seed:       int   = 0,
    ):
        """Build MLP-Mixer with orthogonal weight initialisation."""
        super().__init__()
        self._width   = width
        self._depth   = depth
        self._n_cls   = n_classes

        self.stem = nn.Conv2d(3, width, kernel_size=4, stride=4, bias=False)
        g = torch.Generator()
        g.manual_seed(seed)
        nn.init.orthogonal_(self.stem.weight.view(width, -1), generator=g)

        # Token-mixing: operates over the token dimension (64 tokens)
        self.tok_norms = nn.ModuleList()
        self.tok_fc1s  = nn.ModuleList()
        self.tok_acts  = nn.ModuleList()
        self.tok_fc2s  = nn.ModuleList()

        # Channel-mixing: operates over the channel dimension (width)
        self.ch_norms = nn.ModuleList()
        self.ch_fc1s  = nn.ModuleList()
        self.ch_acts  = nn.ModuleList()
        self.ch_fc2s  = nn.ModuleList()

        for i in range(depth):
            # token-mixing MLPs (hidden = N_TOKENS for standard Mixer)
            self.tok_norms.append(nn.LayerNorm(width))
            tf1 = nn.Linear(self.N_TOKENS, self.N_TOKENS, bias=False)
            tf2 = nn.Linear(self.N_TOKENS, self.N_TOKENS, bias=False)
            _ortho(tf1, seed + i * 4 + 200)
            _ortho(tf2, seed + i * 4 + 201)
            self.tok_fc1s.append(tf1)
            self.tok_acts.append(_make_act(activation))
            self.tok_fc2s.append(tf2)

            # channel-mixing MLPs (4x expansion)
            self.ch_norms.append(nn.LayerNorm(width))
            cf1 = nn.Linear(width, width * 4, bias=False)
            cf2 = nn.Linear(width * 4, width, bias=False)
            _ortho(cf1, seed + i * 4 + 202)
            _ortho(cf2, seed + i * 4 + 203)
            self.ch_fc1s.append(cf1)
            self.ch_acts.append(_make_act(activation))
            self.ch_fc2s.append(cf2)

        self.head = nn.Linear(width, n_classes)

    def forward(
        self, x: torch.Tensor, taps: Optional[Dict] = None
    ) -> torch.Tensor:
        """Forward pass; populates taps dict if provided."""
        h = self.stem(x)                          # [B, width, 8, 8]
        h = h.flatten(2).transpose(1, 2)          # [B, 64, width]

        if taps is not None:
            taps["input"] = h.detach()

        for i in range(self._depth):
            # token-mixing: mix information across spatial positions
            t = self.tok_norms[i](h)              # [B, 64, width]
            t = t.transpose(1, 2)                 # [B, width, 64]
            t = self.tok_fc2s[i](self.tok_acts[i](self.tok_fc1s[i](t)))
            h = h + t.transpose(1, 2)             # residual [B, 64, width]

            # channel-mixing: transform each token's features
            c  = self.ch_norms[i](h)
            c  = self.ch_fc2s[i](self.ch_acts[i](self.ch_fc1s[i](c)))
            h  = h + c

            if taps is not None:
                taps[f"L{i}_ffn_out"] = h.detach()

        z = _pool_seq(h)                          # [B, width]
        if taps is not None:
            taps["ffn_out"] = z.detach()
        return self.head(z)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """
    Standard residual block: conv3x3 -> BN -> act -> conv3x3 -> BN -> residual add -> act.
    Input and output channel count are the same (no downsampling here).
    """

    def __init__(self, ch: int, activation: str):
        super().__init__()
        self.c1  = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1  = nn.BatchNorm2d(ch)
        self.c2  = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2  = nn.BatchNorm2d(ch)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(h + x)


class CNNModel(nn.Module):
    """
    Three-stage ResNet-style network.

    Stage layout:
      stem -> stage1 (width) -> down -> stage2 (width*2) -> down -> stage3 (width*4) -> GAP

    'depth' is the total number of residual blocks distributed across 3 stages.
    Distribution: stage1 gets ceil(depth/3), stage2 gets ceil((depth-n1)/2),
    stage3 gets the remainder.  Every depth value in [1,2,4,8,16,32] produces
    a distinct block count so depth comparisons are meaningful.

    Downsampling between stages uses stride-2 1x1 conv + BN (He et al. 2016).
    Taps: spatial-averaged output after each residual block.
    """

    def __init__(
        self,
        width:      int = 64,
        depth:      int = 6,
        activation: str = "relu",
        n_classes:  int = 10,
        seed:       int = 0,
    ):
        """Build 3-stage ResNet with projection to d_z=width."""
        super().__init__()
        self._width = width
        self._depth = depth
        self._n_cls = n_classes

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            _make_act(activation),
        )

        # Distribute depth blocks across 3 stages so every value of depth
        # produces a different network.  depth=1 -> [1,0,0], depth=4 -> [2,1,1].
        n1 = (depth + 2) // 3           # ceil(depth/3)
        n2 = (depth - n1 + 1) // 2      # ceil((depth-n1)/2)
        n3 = max(0, depth - n1 - n2)
        n1, n2, n3 = max(1, n1), max(0, n2), max(0, n3)

        self.s1    = nn.ModuleList([_ResBlock(width,     activation) for _ in range(n1)])
        self.s2    = nn.ModuleList([_ResBlock(width * 2, activation) for _ in range(n2)])
        self.s3    = nn.ModuleList([_ResBlock(width * 4, activation) for _ in range(n3)])
        # Standard ResNet downsampling: stride-2 1x1 conv + BN (He et al. 2016).
        self.down1 = nn.Sequential(
            nn.Conv2d(width,     width * 2, 1, stride=2, bias=False),
            nn.BatchNorm2d(width * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 1, stride=2, bias=False),
            nn.BatchNorm2d(width * 4),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        # Project from width*4 down to width so d_z = width for all architectures.
        # Without this, CNN's ffn_out has d_z=width*4 while all other archs
        # have d_z=width, making C_arch values incomparable on the same scale.
        self.proj = nn.Linear(width * 4, width, bias=False)
        _ortho(self.proj, seed + 999)   # explicit seed for reproducibility
        self.head = nn.Linear(width, n_classes)

    def forward(
        self, x: torch.Tensor, taps: Optional[Dict] = None
    ) -> torch.Tensor:
        """Forward pass; L-taps are spatial-pooled per-stage outputs."""
        h = self.stem(x)
        if taps is not None:
            taps["input"] = _pool_spatial(h).detach()

        idx = 0
        for blk in self.s1:
            h = blk(h)
            if taps is not None:
                taps[f"L{idx}_ffn_out"] = _pool_spatial(h).detach()
            idx += 1
        h = self.down1(h)

        for blk in self.s2:
            h = blk(h)
            if taps is not None:
                taps[f"L{idx}_ffn_out"] = _pool_spatial(h).detach()
            idx += 1
        h = self.down2(h)

        for blk in self.s3:
            h = blk(h)
            if taps is not None:
                taps[f"L{idx}_ffn_out"] = _pool_spatial(h).detach()
            idx += 1

        z = self.proj(self.gap(h).flatten(1))   # [B, width]
        if taps is not None:
            taps["ffn_out"] = z.detach()
        return self.head(z)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    """
    Reshape CIFAR image into a sequence then run a stacked GRU.

    Input reshaping: [B, 3, 32, 32] -> [B, 64, 48]
    64 non-overlapping 4x4 patches (8x8 grid), each carrying 3*4*4=48 values.
    Patch tokenisation matches MLP and Transformer, making cross-arch comparisons
    more principled than row-wise scanning (which biases horizontal structure).

    Activation is applied to the input projection only; GRU internal gates
    use their standard sigmoid/tanh and are not configurable.

    num_layers is capped at 8 to avoid OOM on large widths.  Conditions
    with depth > 8 still run but use 8 GRU layers, so depth=8, 16, 32
    produce identical GRU architectures.  This contributes to GRU appearing
    as depth_invariant in the Depth Law analysis — the cap should be noted
    in the paper alongside any depth_invariant classification of GRU.

    Taps: input tap = embedded patches [B, 64, width] (d_z=width, pool→mean).
    Layer taps L{l}_ffn_out = h_n[l] (final hidden of layer l).
    ffn_out = mean of all patch outputs from final layer.
    """

    def __init__(
        self,
        width:      int = 256,
        depth:      int = 2,
        activation: str = "relu",
        n_classes:  int = 10,
        seed:       int = 0,
    ):
        """Build stacked GRU on 4×4 patch sequence; depth capped at 8."""
        super().__init__()
        self._width     = width
        self._depth     = depth
        self._n_cls     = n_classes
        self._gru_layers = min(depth, 8)

        self.embed = nn.Linear(48, width, bias=False)  # 3*4*4 patch features
        _ortho(self.embed, seed)
        self.act = _make_act(activation)
        self.gru = nn.GRU(width, width, self._gru_layers, batch_first=True)
        self.head = nn.Linear(width, n_classes)
        self._init_gru()

    def _init_gru(self) -> None:
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)

    def forward(
        self, x: torch.Tensor, taps: Optional[Dict] = None
    ) -> torch.Tensor:
        """Tokenise → embed → GRU → mean-pool; L-taps = h_n[layer]."""
        B   = x.shape[0]
        # [B,3,32,32] → [B,64,48]: 4x4 non-overlapping patches
        p = x.unfold(2, 4, 4).unfold(3, 4, 4)   # [B, 3, 8, 8, 4, 4]
        p = p.contiguous().view(B, 3, 64, 16)    # [B, 3, 64, 16]
        seq = p.permute(0, 2, 1, 3).reshape(B, 64, 48)  # [B, 64, 48]
        emb = self.act(self.embed(seq))          # [B, 64, width]
        if taps is not None:
            # Store as [B, 64, width] so pool() reduces to [B, width] (d_z=width),
            # matching all other architectures.  This makes d_effective and
            # layerwise kappa_input comparable across archs.
            taps["input"] = emb.detach()
        out, h_n = self.gru(emb)               # out:[B,64,width] h_n:[layers,B,width]

        if taps is not None:
            for l in range(self._gru_layers):
                taps[f"L{l}_ffn_out"] = h_n[l].detach()

        # Mean-pool over all 64 patch hidden states.
        z = out.mean(dim=1)                     # [B, width]
        if taps is not None:
            taps["ffn_out"] = z.detach()
        return self.head(z)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# Transformer (Pre-LN and Post-LN)
# ---------------------------------------------------------------------------

class _TransBlock(nn.Module):
    """
    Single transformer layer supporting Pre-LN and Post-LN variants.

    Pre-LN:  h = x + attn(LN(x))
                  out = h + ffn(LN(h))
             Standard Pre-LN formulation (Wang et al. 2019, Xiong et al. 2020).
             Final LayerNorm in TransformerModel stabilises the output.

    Post-LN: h = LN(x + attn(x))
                 then LN(h + ffn(h))
             Standard BERT/original-transformer style.

    Both use multi-head self-attention without causal masking (bidirectional).
    """

    def __init__(
        self,
        width:             int,
        variant:           str,      # "preln" or "postln"
        activation:        str,
        seed:              int   = 0,
        attn_temperature:  float = 1.0,
    ):
        super().__init__()
        nh = _n_heads(width)
        hd = _head_dim(width)
        # verify head count x head_dim == width
        assert nh * hd == width or width < HEAD_DIM

        self.variant          = variant
        self.nh               = nh
        self.hd               = hd
        self.attn_temperature = attn_temperature
        # scale = 1/(sqrt(hd) * T): T=1 recovers standard scaled dot-product.
        # Increasing T flattens attention (→ uniform), decreasing T sharpens it.
        # This mirrors the Boltzmann sigmoid temperature, enabling direct
        # T-invariance comparison between Transformer and Boltzmann in P3.
        self.scale = 1.0 / (math.sqrt(hd) * attn_temperature)

        self.n1  = nn.LayerNorm(width)
        self.n2  = nn.LayerNorm(width)
        self.Wq  = nn.Linear(width, width, bias=False)
        self.Wk  = nn.Linear(width, width, bias=False)
        self.Wv  = nn.Linear(width, width, bias=False)
        self.Wo  = nn.Linear(width, width, bias=False)
        self.ff1 = nn.Linear(width, width * 4, bias=False)
        self.ff2 = nn.Linear(width * 4, width, bias=False)
        self.act = _make_act(activation)

        for i, w in enumerate([self.Wq, self.Wk, self.Wv,
                                self.Wo, self.ff1, self.ff2]):
            _ortho(w, seed + i)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        nh, hd  = self.nh, self.hd
        q = self.Wq(x).view(B, L, nh, hd).transpose(1, 2)
        k = self.Wk(x).view(B, L, nh, hd).transpose(1, 2)
        v = self.Wv(x).view(B, L, nh, hd).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        a = F.softmax(s, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, L, D)
        return self.Wo(o)

    def forward(
        self,
        x:                 torch.Tensor,
        taps:              Optional[Dict],
        layer_idx:         int,
        attn_temperature:  Optional[float] = None,
    ) -> torch.Tensor:
        # Override per-block temperature if given (used in T sweep).
        if attn_temperature is not None:
            saved = self.scale
            self.scale = 1.0 / (math.sqrt(self.hd) * attn_temperature)
        else:
            saved = None
        try:
            if self.variant == "preln":
                a   = self._attn(self.n1(x))
                h   = x + a
                f   = self.ff2(self.act(self.ff1(self.n2(h))))
                out = h + f
            else:    # postln
                a   = self._attn(x)
                h   = self.n1(x + a)
                f   = self.ff2(self.act(self.ff1(h)))
                out = self.n2(h + f)
        finally:
            if saved is not None:
                self.scale = saved   # always restore, even on exception

        if taps is not None:
            taps[f"L{layer_idx}_ffn_out"] = out.detach()
        return out


class TransformerModel(nn.Module):
    """
    variant "preln"  : Pre-LN transformer (more stable, used as primary)
    variant "postln" : Post-LN transformer (original BERT/GPT style)

    Input: CIFAR [B, 3, 32, 32]
    Stem:  4x4 conv with stride 4 -> [B, 64 tokens, width]
    Positional embedding: learnable [1, 64, width] added after stem.
      Without PE the model is permutation-invariant over patches, which
      is a deliberate ablation point.  Standard ViT includes PE.
    """

    N_TOKENS = 64    # 8x8 grid from 32x32 image with 4x4 patches

    def __init__(
        self,
        width:      int   = 256,
        depth:      int   = 4,
        variant:    str   = "preln",
        activation: str   = "relu",
        n_classes:  int   = 10,
        seed:       int   = 0,
    ):
        super().__init__()
        self._width   = width
        self._depth   = depth
        self._n_cls   = n_classes

        self.stem = nn.Conv2d(3, width, kernel_size=4, stride=4, bias=False)
        g = torch.Generator()
        g.manual_seed(seed)
        nn.init.orthogonal_(self.stem.weight.view(width, -1), generator=g)

        # Learnable positional embedding — standard ViT design.
        # Initialised from N(0, 0.02) following Dosovitskiy et al. (2021).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.N_TOKENS, width)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            _TransBlock(width, variant, activation, seed + 100 + i * 10)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(
        self,
        x:                 torch.Tensor,
        taps:              Optional[Dict] = None,
        temperature:       Optional[float] = None,
    ) -> torch.Tensor:
        """
        temperature: attention softmax temperature (T sweep for P3).
          T=1.0 (default) is standard scaled dot-product attention.
          T>1 → flatter attention (closer to uniform, high entropy).
          T<1 → sharper attention (closer to hard argmax, low entropy).
          Transformer is expected T-invariant: kappa stable across T.
        """
        h = self.stem(x)
        B, D, H, W = h.shape
        h = h.flatten(2).transpose(1, 2)          # [B, 64, width]
        h = h + self.pos_embed                    # add positional embedding

        if taps is not None:
            taps["input"] = h.detach()

        for i, layer in enumerate(self.layers):
            h = layer(h, taps=taps, layer_idx=i,
                      attn_temperature=temperature)

        h = self.norm(h)
        z = _pool_seq(h)
        if taps is not None:
            taps["ffn_out"] = z.detach()
        return self.head(z)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# Serial
# ---------------------------------------------------------------------------

class SerialModel(nn.Module):
    """
    Pure serial composition without residual connections.

    Used to validate the composition law:  kappa_total = sum(kappa_l).
    Residual connections would partially bypass layers and inflate kappa
    relative to a true serial chain, so they are omitted here.

    Each block: LN -> Linear(w, 2w) -> act -> Linear(2w, w).
    No residual add.  Information must flow through the transform.
    """

    def __init__(
        self,
        width:      int = 256,
        depth:      int = 4,
        activation: str = "relu",
        n_classes:  int = 10,
        seed:       int = 0,
    ):
        """Build serial chain of LN→2x→act→x blocks, no residual."""
        super().__init__()
        self._width = width
        self._depth = depth
        self._n_cls = n_classes

        self.stem = nn.Conv2d(3, width, kernel_size=4, stride=4, bias=False)
        g = torch.Generator()
        g.manual_seed(seed)
        nn.init.orthogonal_(self.stem.weight.view(width, -1), generator=g)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            fc1 = nn.Linear(width, width * 2, bias=False)
            fc2 = nn.Linear(width * 2, width, bias=False)
            _ortho(fc1, seed + i * 2 + 10)
            _ortho(fc2, seed + i * 2 + 11)
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(width), fc1, _make_act(activation), fc2
            ))

        self.head = nn.Linear(width, n_classes)

    def forward(
        self, x: torch.Tensor, taps: Optional[Dict] = None
    ) -> torch.Tensor:
        """Serial composition without residual; each block must learn a full transform."""
        h = self.stem(x)
        h = h.flatten(2).transpose(1, 2)          # [B, 64, W]

        if taps is not None:
            taps["input"] = h.detach()

        for i, blk in enumerate(self.blocks):
            h = blk(h)                            # no residual
            if taps is not None:
                taps[f"L{i}_ffn_out"] = h.detach()

        z = _pool_seq(h)
        if taps is not None:
            taps["ffn_out"] = z.detach()
        return self.head(z)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------

class ParallelModel(nn.Module):
    """
    'depth' independent branches processed in parallel; outputs merged
    with a learned softmax gate.

    Used to validate the parallel composition law:
      kappa_total ≈ max(kappa_i)  (not sum)
    because each branch sees the same input, so they carry redundant info.

    Gate: g = softmax(W * mean_pool(input))  [B, depth]
    z    = sum_i  g_i * branch_i(input)
    """

    def __init__(
        self,
        width:      int = 256,
        depth:      int = 4,
        activation: str = "relu",
        n_classes:  int = 10,
        seed:       int = 0,
    ):
        """Build parallel branches with learned softmax gate."""
        super().__init__()
        self._width = width
        self._depth = depth
        self._n_cls = n_classes

        self.stem = nn.Conv2d(3, width, kernel_size=4, stride=4, bias=False)
        g = torch.Generator()
        g.manual_seed(seed)
        nn.init.orthogonal_(self.stem.weight.view(width, -1), generator=g)

        self.branches = nn.ModuleList()
        for i in range(depth):
            fc1 = nn.Linear(width, width * 2, bias=False)
            fc2 = nn.Linear(width * 2, width, bias=False)
            _ortho(fc1, seed + i * 2 + 50)
            _ortho(fc2, seed + i * 2 + 51)
            self.branches.append(nn.Sequential(
                nn.LayerNorm(width), fc1, _make_act(activation), fc2
            ))

        self.gate = nn.Linear(width, depth)
        self.head = nn.Linear(width, n_classes)

    def forward(
        self, x: torch.Tensor, taps: Optional[Dict] = None
    ) -> torch.Tensor:
        """Run all branches in parallel; z = Σ gate_i · branch_i(x)."""
        h  = self.stem(x)
        h  = h.flatten(2).transpose(1, 2)         # [B, 64, W]
        h0 = _pool_seq(h)                          # [B, W]  for gating

        if taps is not None:
            taps["input"] = h.detach()

        outs = []
        for i, branch in enumerate(self.branches):
            b = _pool_seq(branch(h))               # [B, W]
            outs.append(b)
            if taps is not None:
                taps[f"B{i}_ffn_out"] = b.detach()

        stack = torch.stack(outs, dim=1)           # [B, depth, W]
        g     = F.softmax(self.gate(h0), dim=-1)  # [B, depth]
        z     = (stack * g.unsqueeze(-1)).sum(dim=1)  # [B, W]

        if taps is not None:
            taps["ffn_out"] = z.detach()
        return self.head(z)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# Boltzmann (RBM)
# ---------------------------------------------------------------------------

class BoltzmannModel(nn.Module):
    """
    Mean-field RBM: single visible→hidden layer with RBM energy formulation,
    trained end-to-end with backpropagation (NOT Contrastive Divergence).

    This is technically a temperature-controlled sigmoid network whose weight
    matrix and biases follow the RBM parameterisation.  The RBM energy
    structure allows computing the standard Helmholtz free energy F(v) for
    the physics analysis, while backprop training makes it directly comparable
    to the other architectures in terms of representational capacity.

    Temperature T controls sigmoid sharpness: h = sigmoid((v @ W + b_h) / T).
      T -> 0 : units saturate to 0 or 1  (deterministic / low entropy)
      T -> inf: units approach 0.5        (maximum entropy)

    free_energy(v) = -v @ b_v - sum(softplus(v @ W + b_h))
      This is the standard RBM marginal free energy (Hinton 2010).
      T=1 is assumed (temperature enters only in the inference pass, not F).

    depth is always 1 (stacked RBM not implemented).
    Taps: L0_ffn_out == ffn_out (single hidden layer).
    """

    def __init__(
        self,
        width:       int   = 256,
        depth:       int   = 1,
        activation:  str   = "relu",   # unused; kept for interface compatibility
        n_classes:   int   = 10,
        temperature: float = 1.0,
        seed:        int   = 0,
    ):
        """Initialise mean-field RBM; W~N(0,0.01), biases zero."""
        super().__init__()
        self._width      = width
        self._depth      = depth
        self._n_cls      = n_classes
        self.temperature = temperature

        d_v = 3 * 32 * 32    # flattened CIFAR
        self.W   = nn.Parameter(torch.empty(d_v, width))
        self.b_v = nn.Parameter(torch.zeros(d_v))
        self.b_h = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.W, 0.0, 0.01)

        self.head = nn.Linear(width, n_classes)

    def _mean_field(self, v: torch.Tensor, T: float) -> torch.Tensor:
        return torch.sigmoid((v @ self.W + self.b_h) / T)   # [B, width]

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Marginal free energy F(v) for the physics paper."""
        vb  = (v * self.b_v).sum(dim=1)
        pre = v @ self.W + self.b_h
        return -vb - F.softplus(pre).sum(dim=1)

    def forward(
        self,
        x:           torch.Tensor,
        taps:        Optional[Dict]  = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Flatten → mean-field h = σ((v@W+b_h)/T) → classify."""
        T = temperature if temperature is not None else self.temperature
        v = x.reshape(x.shape[0], -1)

        if taps is not None:
            taps["input"] = v.detach()

        h = self._mean_field(v, T)

        if taps is not None:
            taps["L0_ffn_out"] = h.detach()
            taps["ffn_out"]    = h.detach()

        return self.head(h)

    @property
    def width_proxy(self):  return self._width
    @property
    def depth_proxy(self):  return self._depth
    @property
    def n_classes(self):    return self._n_cls


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    arch:        str,
    width:       int,
    depth:       int,
    activation:  str = "relu",
    n_classes:   int = 10,
    seed:        int = 0,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a model by name.  kwargs are forwarded to the constructor
    and currently only used by Boltzmann (temperature=...).
    """
    a = arch.lower()
    common = dict(
        width=width, depth=depth, activation=activation,
        n_classes=n_classes, seed=seed,
    )
    if   a == "mlp":                 return MLPModel(**common)
    elif a == "cnn":                 return CNNModel(**common)
    elif a == "gru":                 return GRUModel(**common)
    elif a == "transformer_preln":   return TransformerModel(**common, variant="preln")
    elif a == "transformer_postln":  return TransformerModel(**common, variant="postln")
    elif a == "serial":              return SerialModel(**common)
    elif a == "parallel":            return ParallelModel(**common)
    elif a == "boltzmann":
        return BoltzmannModel(**common, temperature=kwargs.get("temperature", 1.0))
    else:
        valid = ("mlp cnn gru transformer_preln transformer_postln "
                 "serial parallel boltzmann")
        raise ValueError(f"unknown arch '{arch}'.  Valid: {valid}")
