"""Atlas A3 MegaMoE adapter for ``ascend_fuseep`` mode 3.

Routing and routed-output scaling remain with the model. MegaMoE owns the
dispatch, expert FFN and combine; shared experts stay outside this operator.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_parallel, get_resources

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)


def validate_mega_moe_server_args(cfg) -> None:
    if cfg.device != "npu" or cfg.moe_a2a_backend != "ascend_fuseep":
        raise ValueError(
            "--fuseep-mode 3 requires --device npu --moe-a2a-backend ascend_fuseep"
        )
    graphs = cfg.cuda_graph_config
    if (
        cfg.enable_torch_compile
        or graphs is None
        or any(phase.backend != "disabled" for phase in (graphs.decode, graphs.prefill))
    ):
        raise ValueError(
            "A3 MegaMoE requires eager execution: --disable-prefill-cuda-graph "
            "--disable-decode-cuda-graph, without torch.compile"
        )
    # The public mode-3 API derives its active mask from x.shape[0]. It cannot
    # represent SGLang's per-rank attention-DP padding yet.
    if (
        cfg.attn_dp_size != 1
        or cfg.attn_cp_size != 1
        or cfg.dcp_size != 1
        or cfg.pp_size != 1
    ):
        raise ValueError("A3 MegaMoE currently requires attention TP with DP=CP=PP=1")
    if cfg.elastic_ep_backend is not None or cfg.ep_join_mode is not None:
        raise ValueError("A3 MegaMoE requires a static expert-parallel group")
    unsupported = (
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_eplb",
        "enable_waterfill",
        "enable_lora",
        "enable_mixed_chunk",
        "enable_dynamic_chunking",
        "enable_memory_saver",
    )
    for name in unsupported:
        if getattr(cfg, name):
            raise ValueError(f"A3 MegaMoE does not support --{name.replace('_', '-')}")
    if cfg.speculative_algorithm is not None or cfg.ep_num_redundant_experts:
        raise ValueError(
            "A3 MegaMoE does not support speculative decoding or redundant experts"
        )
    if cfg.cpu_offload_gb:
        raise ValueError("A3 MegaMoE does not support CPU weight offloading")
    if cfg.moe_runner_backend not in ("auto", "ascend"):
        raise ValueError("A3 MegaMoE requires --moe-runner-backend auto or ascend")
    capacity = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
    if not 1 <= capacity <= 4096:
        raise ValueError(
            "A3 MegaMoE requires SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK in [1, 4096]"
        )
    # Bound the scheduler, rather than selecting a different collective locally
    # when a rank happens to receive an oversized batch.
    if (
        cfg.chunked_prefill_size is None
        or not 0 < cfg.chunked_prefill_size <= capacity
        or not 0 < cfg.max_prefill_tokens <= capacity
    ):
        raise ValueError(
            f"A3 MegaMoE requires positive --chunked-prefill-size and --max-prefill-tokens <= {capacity}"
        )
    if cfg.max_running_requests is None or not 0 < cfg.max_running_requests <= capacity:
        raise ValueError(
            f"A3 MegaMoE requires explicit --max-running-requests in [1, {capacity}]"
        )
    for budget in (
        cfg.chunked_prefill_size,
        cfg.max_prefill_tokens,
        cfg.max_running_requests,
    ):
        if (budget + cfg.tp_size - 1) // cfg.tp_size * cfg.tp_size > capacity:
            raise ValueError(
                "A3 MegaMoE token capacity must cover TP-aligned scheduler budgets"
            )


def validate_mega_moe_layer(layer: FusedMoE) -> None:
    config = layer.moe_runner_config
    if layer.moe_tp_size != 1 or layer.num_experts % layer.moe_ep_size:
        raise ValueError(
            "A3 MegaMoE requires pure expert parallelism with equally sized expert partitions"
        )
    if not 1 <= config.top_k <= min(16, layer.num_experts) or layer.num_experts > 1024:
        raise ValueError(
            "A3 MegaMoE supports TopK <= 16 and at most 1024 global experts"
        )
    if (
        not 0 < layer.hidden_size <= 8192
        or not 0 < layer.intermediate_size_per_partition <= 3584
    ):
        raise ValueError(
            "A3 MegaMoE requires hidden <= 8192 and expert intermediate <= 3584"
        )
    if layer.hidden_size % 32 or layer.intermediate_size_per_partition % 32:
        raise ValueError(
            "A3 MegaMoE requires hidden and intermediate dimensions aligned to 32"
        )
    if config.params_dtype != torch.bfloat16:
        raise ValueError("A3 MegaMoE currently supports BF16 model activations only")
    if (
        layer.with_bias
        or config.num_fused_shared_experts
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
    ):
        raise ValueError(
            "A3 MegaMoE requires gated routed experts, output routing weights, and separate shared experts"
        )
    if config.activation not in ("silu", "situ") or config.swiglu_limit is not None:
        raise ValueError(
            "A3 MegaMoE supports plain SwiGLU or SiTU, without SwiGLU clamping"
        )
    if config.activation == "silu" and (
        config.gemm1_alpha is not None
        or config.gemm1_beta is not None
        or config.gemm1_clamp_limit is not None
    ):
        raise ValueError(
            "A3 MegaMoE does not support modified SwiGLU activation parameters"
        )
    if config.activation == "situ":
        for value in (config.gemm1_alpha, config.gemm1_clamp_limit):
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "A3 MegaMoE SiTU requires explicit positive beta and linear_beta"
                )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUUnquantMoEMethod,
        NPUW4A8Int8MoEMethod,
        NPUW8A8Int8MoEMethod,
    )

    for prefix in ("w13", "w2"):
        kernel = getattr(layer, f"{prefix}_kernel", None)
        if not isinstance(
            kernel, (NPUUnquantMoEMethod, NPUW8A8Int8MoEMethod, NPUW4A8Int8MoEMethod)
        ):
            raise ValueError(
                "A3 MegaMoE supports BF16, W8A8 INT8 or per-channel W4A8 INT4 experts; FP4/MXFP4 are not INT4"
            )
        if isinstance(kernel, NPUW4A8Int8MoEMethod) and (
            not kernel.is_per_channel_weight or kernel.activation_use_clip
        ):
            raise ValueError(
                "A3 MegaMoE W4A8 requires per-channel weights without activation clipping"
            )
        if isinstance(kernel, NPUW4A8Int8MoEMethod) and (
            layer.hidden_size % 64 or layer.intermediate_size_per_partition % 64
        ):
            raise ValueError(
                "A3 MegaMoE W4A8 requires hidden and intermediate dimensions aligned to 64"
            )
    if type(layer.w13_kernel) is not type(layer.w2_kernel):
        raise ValueError(
            "A3 MegaMoE requires matching quantization for both expert projections"
        )


def _pack_scale(scale: torch.Tensor, *, int4: bool) -> torch.Tensor:
    """Match the A3 per-channel dequant-scale bit encoding, not a numeric cast."""
    bits = scale.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if int4:
        bits = (bits & 0xFFFFE000) | (1 << 46)
    return bits.view(torch.uint64)


def _int4_compensation(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A8W4 compensation: 8 * sum_K(W * scale), before converting to NZ."""
    # Each checkpoint byte stores two signed INT4 output channels. Do this
    # once on CPU at load time, matching the PR's reference preparation.
    nibbles = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1)
    values = nibbles.reshape(packed.shape[0], -1)
    values = torch.where(values >= 8, values - 16, values).float()
    scale_fp32 = scale.view(torch.int64).to(torch.int32).view(torch.float32)
    return (values * scale_fp32).sum(dim=0) * 8


def maybe_process_mega_moe_weights(
    layer, weight_prefix: str, quantization: str
) -> bool:
    moe = get_exec().moe
    if moe.moe_a2a_backend != "ascend_fuseep" or moe.fuseep_mode != 3:
        return False
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

    validate_mega_moe_layer(layer)
    device = layer.w13_weight.device
    device_name = torch.npu.get_device_name(device)
    if "910_93" not in device_name and "910C" not in device_name:
        raise ValueError(f"MegaMoE mode 3 requires Atlas A3, got {device_name}")
    payload = getattr(layer, "_ascend_mega_moe_weights", None)
    if payload is None:
        payload = layer._ascend_mega_moe_weights = {}
    weight = getattr(layer, f"{weight_prefix}_weight")
    scale = getattr(layer, f"{weight_prefix}_weight_scale", None)
    out_features = (
        2 * layer.intermediate_size_per_partition
        if weight_prefix == "w13"
        else layer.hidden_size
    )
    in_features = (
        layer.hidden_size
        if weight_prefix == "w13"
        else layer.intermediate_size_per_partition
    )
    expected_dtype = torch.bfloat16 if quantization == "bf16" else torch.int8
    packed_out = out_features // 2 if quantization == "w4a8" else out_features
    if weight.dtype != expected_dtype or tuple(weight.shape) != (
        layer.num_local_experts,
        packed_out,
        in_features,
    ):
        raise ValueError(
            f"Unsupported {quantization} {weight_prefix} checkpoint layout: {weight.shape}, {weight.dtype}"
        )
    if quantization != "bf16" and (
        scale is None
        or scale.dtype != torch.float32
        or tuple(scale.shape) != (layer.num_local_experts, out_features, 1)
    ):
        raise ValueError(
            "MegaMoE quantized experts require per-channel float32 checkpoint scales"
        )
    if getattr(layer, f"{weight_prefix}_weight_bias", None) is not None:
        raise ValueError("MegaMoE does not support expert projection bias")
    weights, scales, biases = [], [], []
    for expert in range(layer.num_local_experts):
        matrix = weight[expert].detach().transpose(0, 1).contiguous()
        if quantization == "bf16":
            weights.append(npu_format_cast(matrix))
            continue
        packed_scale = _pack_scale(
            scale[expert].detach().cpu().flatten(), int4=quantization == "w4a8"
        )
        scales.append(packed_scale.to(device))
        if quantization == "w4a8":
            biases.append(_int4_compensation(matrix.cpu(), packed_scale).to(device))
            # Cast each expert independently; casting the stacked tensor and
            # then unbinding does not preserve the per-expert NZ storage.
            weights.append(npu_format_cast(matrix).view(torch.int32))
        else:
            weights.append(npu_format_cast(matrix))
    payload[weight_prefix] = (weights, scales or None, biases or None)
    layer._ascend_mega_moe_quant_mode = 0 if quantization == "bf16" else 1
    if "w13" in payload and "w2" in payload:
        # Initialize/JIT/warm up on every rank before serving. Empty ranks
        # still enter the collective; the strategy pads them with an inactive mask.
        forward_mega_moe(
            layer,
            weight.new_empty((0, layer.hidden_size), dtype=torch.bfloat16),
            _empty_topk(layer, device),
        )
    return True


def _empty_topk(layer, device):
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    return StandardTopKOutput(
        torch.empty((0, layer.top_k), dtype=torch.float32, device=device),
        torch.empty((0, layer.top_k), dtype=torch.int32, device=device),
        None,
    )


def _get_mega_moe_buffer():
    from deep_ep import Buffer
    from deep_ep.buffer import FuseMode

    if not hasattr(FuseMode, "MEGA_MOE"):
        raise RuntimeError(
            "MegaMoE mode 3 requires a DeepEP-Ascend build containing sgl-kernel-npu PR #816"
        )
    group = get_parallel().moe_ep_group.device_group
    buffers = get_resources().buffers.setdefault("ascend_mega_moe", {})
    if group not in buffers:
        # MegaMoE's strategy owns its symmetric buffer. It does not use the
        # legacy runtime's NVL/RDMA allocation or its low-latency capacity hints.
        buffers[group] = Buffer(group, low_latency_mode=False)
        logger.info("Atlas A3 MegaMoE enabled (ascend_fuseep mode 3, eager execution)")
    return buffers[group]


def forward_mega_moe(
    layer: FusedMoE, hidden_states: torch.Tensor, topk_output: TopKOutput
) -> torch.Tensor:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(topk_output):
        raise ValueError("MegaMoE requires standard TopK ids and weights")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("A3 MegaMoE requires BF16 inputs")
    config = layer.moe_runner_config
    capacity = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
    payload = layer._ascend_mega_moe_weights
    w13, s13, b13 = payload["w13"]
    w2, s2, b2 = payload["w2"]
    # Standard routing may mark TP padding with -1. MegaMoE's public API
    # supplies its own active mask, so turn these slots into zero-weight routes.
    ids = topk_output.topk_ids.to(torch.int32)
    valid = ids >= 0
    weights = torch.where(valid, topk_output.topk_weights.float(), 0.0)
    ids = torch.where(valid, ids, 0)
    hidden_states = torch.where(valid.any(dim=1, keepdim=True), hidden_states, 0.0)
    activation_kwargs = {"activation": "swiglu"}
    if config.activation == "situ":
        activation_kwargs = {
            "activation": "situ",
            "beta": config.gemm1_alpha,
            "linear_beta": config.gemm1_clamp_limit,
        }
    output, _ = _get_mega_moe_buffer().fused_deep_moe(
        x=hidden_states,
        topk_idx=ids,
        topk_weights=weights,
        gmm1_permuted_weight=w13,
        gmm1_permuted_weight_scale=s13,
        gmm2_weight=w2,
        gmm2_weight_scale=s2,
        l1_bias=b13,
        l2_bias=b2,
        num_max_dispatch_tokens_per_rank=capacity,
        num_experts=layer.num_experts,
        quant_mode=layer._ascend_mega_moe_quant_mode,
        fuse_mode=3,
        **activation_kwargs,
    )
    return output
