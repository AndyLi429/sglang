"""A3 DSV4.1 mHC coefficients and residual mixing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v4_mhc import HcSubLayer, HcTriplet


def mix_stats(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet:
    cfg = hc.cfg
    if x.shape[0] == 0 or cfg.mult != 4 or cfg.sinkhorn_iters != 20:
        return mix_stats_torch(hc, x)

    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

    # DSV4.1 consumes the preceding sublayer's pre-mix for this input.
    # Only this sublayer's coefficients are needed from the CANN operator.
    _, post, comb, pre = mhc_pre_sinkhorn(
        x,
        hc.fn,
        hc.scale,
        hc.base,
        cfg.mult,
        cfg.sinkhorn_iters,
        cfg.eps,
        cfg.rms_eps,
        return_pre=True,
    )
    return pre, post, comb.reshape(-1, cfg.mult, cfg.mult)


def mix_stats_torch(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet:
    """FP32 counterpart of hc_mix_stats + hc_split_sinkhorn.

    The first row normalization adds eps *after* softmax; subsequent row and
    column normalizations add it to the denominator, matching the fused kernel.
    The loop bound is model-static and introduces no host reads during capture.
    """
    cfg = hc.cfg
    flat = x.flatten(1).float()
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_eps)
    mixes = F.linear(flat, hc.fn.float()) * inv_rms
    h = cfg.mult
    pre = torch.sigmoid(mixes[:, :h] * hc.scale[0] + hc.base[:h]) + cfg.eps
    post_mix = 2.0 * torch.sigmoid(
        mixes[:, h : 2 * h] * hc.scale[1] + hc.base[h : 2 * h]
    )
    comb = (mixes[:, 2 * h :] * hc.scale[2] + hc.base[2 * h :]).reshape(-1, h, h)
    comb = torch.softmax(comb, dim=-1) + cfg.eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.eps)
    for _ in range(cfg.sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + cfg.eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.eps)
    return pre, post_mix, comb


def post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Accumulate residual mixing and the sublayer output in FP32."""
    return (
        torch.bmm(comb.float().transpose(1, 2), residual.float())
        + post_mix.float().unsqueeze(-1) * x.float().unsqueeze(1)
    ).to(x.dtype)
