"""
TDB-style introspection hooks for the nanochat GPT.

Design goals (kept aligned with OpenAI's transformer-debugger terminology):
  - Component  -> attention head, MLP neuron, block
  - Node       -> per-token invocation of a component (one forward pass)
  - Write vec  -> what a node writes into the residual stream
  - Direct effect / write magnitude -> derived scalars on those write vectors

Everything is opt-in: the model only records or mutates state when
``DEBUG['enabled']`` is True.  This keeps the hot training path untouched.

Usage::

    from nanochat import debug_hooks as dh
    with dh.capture(layers='all', attn_probs=True) as cap:
        logits = model(idx)
    cap.layer_states[3].delta_attn_norm   # write magnitude of attn at layer 3
    cap.attn_probs[5][0, 2]               # head 2, batch 0, layer 5

Ablations::

    with dh.ablate(attn_layers=[7], mlp_layers=[]):
        logits_ablated = model(idx)

Both context managers are nestable and thread-local-safe enough for the MCP
server's single-threaded use.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch


DEBUG: dict[str, Any] = {
    "enabled": False,
    "capture": None,
    "ablate": None,
}


@dataclass
class LayerState:
    layer_idx: int
    resid_pre_norm: float = 0.0
    resid_post_attn_norm: float = 0.0
    resid_post_mlp_norm: float = 0.0
    delta_attn_norm: float = 0.0
    delta_mlp_norm: float = 0.0
    attn_entropy_per_head: list[float] = field(default_factory=list)
    direct_effect_attn: float | None = None
    direct_effect_mlp: float | None = None


@dataclass
class Capture:
    """Container for everything captured during a forward pass."""
    layers: set[int] | str = "all"
    want_attn_probs: bool = False
    layer_states: dict[int, LayerState] = field(default_factory=dict)
    # attn_probs[layer_idx] -> tensor (B, H, T, T) on cpu
    attn_probs: dict[int, torch.Tensor] = field(default_factory=dict)
    # residual stream snapshots (B, T, C) post each block, kept on cpu/fp32
    residual_post_block: dict[int, torch.Tensor] = field(default_factory=dict)
    # final pre-lm_head normalised residual
    final_resid: torch.Tensor | None = None
    # raw logits (last position only) for cheap top-k inspection
    last_logits: torch.Tensor | None = None
    # direction-of-interest unit vector in residual space, set externally
    direction: torch.Tensor | None = None

    def wants(self, layer_idx: int) -> bool:
        return self.layers == "all" or layer_idx in self.layers


@dataclass
class Ablate:
    attn_layers: set[int] = field(default_factory=set)
    mlp_layers: set[int] = field(default_factory=set)
    # attn_head_mask[layer_idx] -> set of head indices to zero in v_out
    attn_head_mask: dict[int, set[int]] = field(default_factory=dict)


@contextlib.contextmanager
def capture(layers="all", attn_probs: bool = False, direction: torch.Tensor | None = None):
    """Enable debug capture for the duration of the with-block."""
    layer_set = layers if layers == "all" else set(layers)
    cap = Capture(layers=layer_set, want_attn_probs=attn_probs, direction=direction)
    prev_enabled = DEBUG["enabled"]
    prev_cap = DEBUG["capture"]
    DEBUG["enabled"] = True
    DEBUG["capture"] = cap
    try:
        yield cap
    finally:
        DEBUG["capture"] = prev_cap
        DEBUG["enabled"] = prev_enabled or DEBUG["ablate"] is not None


@contextlib.contextmanager
def ablate(attn_layers=(), mlp_layers=(), attn_head_mask=None):
    abl = Ablate(
        attn_layers=set(attn_layers),
        mlp_layers=set(mlp_layers),
        attn_head_mask={k: set(v) for k, v in (attn_head_mask or {}).items()},
    )
    prev_enabled = DEBUG["enabled"]
    prev_abl = DEBUG["ablate"]
    DEBUG["enabled"] = True
    DEBUG["ablate"] = abl
    try:
        yield abl
    finally:
        DEBUG["ablate"] = prev_abl
        DEBUG["enabled"] = prev_enabled or DEBUG["capture"] is not None


def is_active() -> bool:
    return DEBUG["enabled"]


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def record_block_io(layer_idx: int, x_in, attn_out, x_after_attn, mlp_out, x_after_mlp):
    """Called from Block.forward when capture is active."""
    cap: Capture | None = DEBUG.get("capture")
    if cap is None or not cap.wants(layer_idx):
        return
    st = cap.layer_states.setdefault(layer_idx, LayerState(layer_idx))
    st.resid_pre_norm = _norm(x_in)
    st.delta_attn_norm = _norm(attn_out)
    st.resid_post_attn_norm = _norm(x_after_attn)
    st.delta_mlp_norm = _norm(mlp_out)
    st.resid_post_mlp_norm = _norm(x_after_mlp)
    cap.residual_post_block[layer_idx] = x_after_mlp.detach().float().cpu()
    if cap.direction is not None:
        d = cap.direction.to(attn_out.device, attn_out.dtype)
        st.direct_effect_attn = float((attn_out[:, -1] @ d).mean().item())
        st.direct_effect_mlp = float((mlp_out[:, -1] @ d).mean().item())


def record_attn_probs(layer_idx: int, probs: torch.Tensor):
    """probs shape (B, H, T, T). Called from sdpa fallback in attention."""
    cap: Capture | None = DEBUG.get("capture")
    if cap is None or not cap.wants(layer_idx) or not cap.want_attn_probs:
        return
    p = probs.detach().float().cpu()
    cap.attn_probs[layer_idx] = p
    eps = 1e-9
    # entropy averaged over batch and queries: per-head scalar in nats
    ent = -(p.clamp_min(eps) * (p.clamp_min(eps).log())).sum(-1).mean(dim=(0, 2))
    st = cap.layer_states.setdefault(layer_idx, LayerState(layer_idx))
    st.attn_entropy_per_head = [float(v) for v in ent.tolist()]


def record_final(resid: torch.Tensor, logits: torch.Tensor):
    cap: Capture | None = DEBUG.get("capture")
    if cap is None:
        return
    cap.final_resid = resid.detach().float().cpu()
    cap.last_logits = logits[:, -1, :].detach().float().cpu()


def maybe_zero_attn(layer_idx: int, attn_out: torch.Tensor) -> torch.Tensor:
    abl: Ablate | None = DEBUG.get("ablate")
    if abl is None:
        return attn_out
    if layer_idx in abl.attn_layers:
        return torch.zeros_like(attn_out)
    return attn_out


def maybe_zero_mlp(layer_idx: int, mlp_out: torch.Tensor) -> torch.Tensor:
    abl: Ablate | None = DEBUG.get("ablate")
    if abl is None:
        return mlp_out
    if layer_idx in abl.mlp_layers:
        return torch.zeros_like(mlp_out)
    return mlp_out


def need_sdpa_fallback() -> bool:
    """Force the SDPA path so we can extract attention probabilities."""
    cap: Capture | None = DEBUG.get("capture")
    return cap is not None and cap.want_attn_probs


def sdpa_with_probs(q, k, v, causal: bool, window_size):
    """Manual scaled-dot-product attention that also returns probs.

    q, k, v are in (B, T, H, D) layout to match the FA3 call sites.
    """
    # GQA expansion
    h_q, h_kv = q.size(2), k.size(2)
    if h_q != h_kv:
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    q_ = q.transpose(1, 2).float()  # (B,H,T,D)
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()
    scale = 1.0 / math.sqrt(q_.size(-1))
    scores = q_ @ k_.transpose(-1, -2) * scale
    T = scores.size(-1)
    if causal:
        mask = torch.ones(T, T, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    if window_size is not None:
        left, _ = window_size
        if left is not None and left >= 0:
            i = torch.arange(T, device=scores.device).view(-1, 1)
            j = torch.arange(T, device=scores.device).view(1, -1)
            window_mask = (i - j) > left
            scores = scores.masked_fill(window_mask, float("-inf"))
    probs = scores.softmax(dim=-1)
    out = probs @ v_
    return out.transpose(1, 2).to(q.dtype), probs


__all__ = [
    "DEBUG",
    "Capture",
    "Ablate",
    "LayerState",
    "capture",
    "ablate",
    "is_active",
    "record_block_io",
    "record_attn_probs",
    "record_final",
    "maybe_zero_attn",
    "maybe_zero_mlp",
    "need_sdpa_fallback",
    "sdpa_with_probs",
]
