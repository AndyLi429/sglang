"""A3 DSV4.1 mHC coefficients and residual mixing."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v4_mhc import HcSubLayer, HcTriplet

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _has_a3_mhc_coeff() -> bool:
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version

    return (
        torch.npu.get_device_name() == "Ascend910_9362"
        and detect_cann_version() == "9.2.0"
    )


@lru_cache(maxsize=None)
def _log_mhc_coeff_route(rows: int) -> None:
    logger.info("MhcPreSinkhornCoeff route M=%d", rows)


def mix_stats(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet:
    cfg = hc.cfg
    if (
        envs.SGLANG_NPU_DSV41_A3_MHC_COEFF.get()
        and x.shape[0] in (6, 12, 24, 48, 72, 4096, 8192)
        and x.shape[1:] == (4, 5120)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and hc.fn.dtype == hc.scale.dtype == hc.base.dtype == torch.float32
        and hc.fn.shape == (24, 20480)
        and hc.scale.shape == (3,)
        and hc.base.shape == (24,)
        and hc.fn.is_contiguous()
        and hc.scale.is_contiguous()
        and hc.base.is_contiguous()
        and cfg.mult == 4
        and cfg.sinkhorn_iters == 20
        and cfg.eps == 1e-6
        and cfg.rms_eps == 1e-20
        and _has_a3_mhc_coeff()
    ):
        from sglang.srt.hardware_backend.npu.dsv4.mhc_coeff import coefficients

        pre, post, comb = coefficients(
            x,
            hc.fn,
            hc.scale,
            hc.base,
            cfg.mult,
            cfg.sinkhorn_iters,
            cfg.eps,
            cfg.rms_eps,
        )
        _log_mhc_coeff_route(x.shape[0])
        return pre, post, comb.reshape(-1, cfg.mult, cfg.mult)

    return mix_stats_torch(hc, x)


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
