"""Ascend CANN MegaMOE capability checks and fused forward wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import (
    cutedsl_moe_max_num_tokens,
    get_lora,
    get_resources,
)
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


logger = logging.getLogger(__name__)

_BUFFER_CACHE_KEY = "ascend_megamoe_symm_buffers"
_DISPATCH_QUANT_MODE = 2
_DISPATCH_QUANT_DTYPE = torch.int8
_fallback_warning_emitted = False


@dataclass(frozen=True)
class MegaMoeW8A8Payload:
    """Per-expert W8A8 tensors kept in the layout consumed by MegaMOE."""

    w13: tuple[torch.Tensor, ...]
    w2: tuple[torch.Tensor, ...]
    w13_scale: tuple[torch.Tensor, ...]
    w2_scale: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _Availability:
    available: bool
    reason: str
    ops: tuple[Callable, Callable] | None = None
    capacity: int = 0
    ep_world_size: int = 0


def _load_ops() -> tuple[Callable, Callable] | None:
    """Load the optional CANN extension only after the backend is selected."""
    try:
        from cann_ops_transformer.ops import (  # type: ignore[import-not-found]
            get_symm_buffer_for_mega_moe,
            mega_moe,
        )
    except ImportError:
        return None
    return get_symm_buffer_for_mega_moe, mega_moe


def is_ascend_megamoe_backend() -> bool:
    """Return whether the distinct Ascend MegaMOE backend is selected."""
    return get_moe_a2a_backend().is_ascend_megamoe()


def _lora_enabled(layer: FusedMoE) -> bool:
    runner = getattr(layer, "runner", None)
    if bool(getattr(runner, "lora_enabled", False)):
        return True
    try:
        return bool(get_lora().enable_lora)
    except ValueError:
        # Unit-level callers may not have published ServerArgs yet.
        return False


def _max_tokens_per_rank(layer: FusedMoE) -> int:
    """Resolve a rank-invariant per-forward token ceiling.

    Prefer an explicit layer/dispatcher ceiling when one exists. Otherwise use
    the scheduler/graph-derived ceiling shared with the other static MoE
    buffers. Never size a collective buffer from the current forward.
    """
    for owner in (layer, getattr(layer, "dispatcher", None)):
        if owner is None:
            continue
        for name in (
            "num_max_dispatch_tokens_per_rank",
            "max_num_tokens_per_rank",
        ):
            value = getattr(owner, name, None)
            if value is not None and int(value) > 0:
                return int(value)

    try:
        return int(cutedsl_moe_max_num_tokens())
    except ValueError:
        # The runtime configuration has not been published.
        return 0


def _ep_world_size(layer: FusedMoE) -> int:
    value = getattr(layer, "moe_ep_size", None)
    if value is not None:
        return int(value)

    coordinator = get_moe_ep_group()
    value = getattr(coordinator, "world_size", None)
    if value is not None:
        return int(value)
    return int(coordinator.device_group.size())


def _top_k(layer: FusedMoE) -> int:
    value = getattr(layer, "top_k", None)
    if value is None:
        value = getattr(layer.moe_runner_config, "top_k", None)
    return int(value or 0)


def _activation_is_supported(layer: FusedMoE) -> bool:
    config = layer.moe_runner_config
    return bool(getattr(config, "is_gated", False)) and getattr(
        config, "activation", None
    ) in ("silu", "swiglu")


def _payload_is_valid(layer: FusedMoE, experts_per_rank: int) -> bool:
    payload = getattr(layer, "_megamoe_w8a8_payload", None)
    if payload is None:
        return False

    hidden_size = int(layer.hidden_size)
    intermediate_size = int(layer.intermediate_size_per_partition)
    expected = {
        "w13": (torch.int8, (2 * intermediate_size, hidden_size)),
        "w2": (torch.int8, (hidden_size, intermediate_size)),
        "w13_scale": (torch.float32, (2 * intermediate_size,)),
        "w2_scale": (torch.float32, (hidden_size,)),
    }
    for name, (dtype, shape) in expected.items():
        tensors = getattr(payload, name, None)
        if not isinstance(tensors, (list, tuple)) or len(tensors) != experts_per_rank:
            return False
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            for tensor in tensors
        ):
            return False
    return True


def _rank_invariant_admission_tokens(num_tokens: int) -> int:
    """Return the maximum live token count known identically by all ranks."""
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        return max(int(tokens) for tokens in global_num_tokens)

    # All EP ranks must make the same admission decision before buffer creation.
    # This live count is used only for admission, never for allocation sizing.
    token_count = torch.tensor([num_tokens], dtype=torch.int64, device="cpu")
    torch.distributed.all_reduce(
        token_count,
        op=torch.distributed.ReduceOp.MAX,
        group=get_moe_ep_group().cpu_group,
    )
    return int(token_count.item())


def _check_availability(layer: FusedMoE, num_tokens: int) -> _Availability:
    # Keep all cheap, non-collective gates ahead of the optional import and
    # symmetric-buffer allocation. In particular, generic GPU ``megamoe`` must
    # never attempt to load the Ascend extension.
    if not is_ascend_megamoe_backend():
        return _Availability(False, "ascend_megamoe backend is not selected")
    if not envs.SGLANG_NPU_ENABLE_MEGAMOE.get():
        return _Availability(False, "SGLANG_NPU_ENABLE_MEGAMOE is disabled")
    if not is_npu():
        return _Availability(False, "Ascend MegaMOE requires an NPU device")

    ep_world_size = _ep_world_size(layer)
    if ep_world_size <= 1:
        return _Availability(False, "Ascend MegaMOE requires EP size greater than 1")

    num_experts = int(layer.num_experts)
    if num_experts <= 0 or num_experts % ep_world_size != 0:
        return _Availability(
            False,
            "the number of experts must be divisible by EP size",
        )

    experts_per_rank = num_experts // ep_world_size
    if not _payload_is_valid(layer, experts_per_rank):
        return _Availability(False, "a compatible W8A8 payload is unavailable")
    if _lora_enabled(layer):
        return _Availability(False, "Ascend MegaMOE does not support LoRA")
    if not _activation_is_supported(layer):
        return _Availability(False, "Ascend MegaMOE requires SwiGLU activation")

    top_k = _top_k(layer)
    if top_k <= 0:
        return _Availability(False, "MegaMOE top_k must be positive")

    capacity = _max_tokens_per_rank(layer)
    if capacity <= 0:
        return _Availability(
            False,
            "a rank-invariant MegaMOE token capacity is unavailable",
        )

    admission_tokens = _rank_invariant_admission_tokens(num_tokens)
    if admission_tokens < 0 or admission_tokens > capacity:
        return _Availability(
            False,
            f"MegaMOE token capacity exceeded: {admission_tokens} > {capacity}",
        )

    configured_recv = envs.SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS.get()
    if configured_recv < 0:
        return _Availability(
            False,
            "SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS must be non-negative",
        )

    experts_per_rank = num_experts // ep_world_size
    routing_fanout = ep_world_size * min(top_k, experts_per_rank)
    send_capacity = _max_recv_tokens(layer, capacity, ep_world_size) // routing_fanout
    if admission_tokens > send_capacity:
        return _Availability(
            False,
            "MegaMOE receive capacity cannot safely admit the scheduled token count: "
            f"{admission_tokens} > {send_capacity}",
        )

    ops = _load_ops()
    if ops is None:
        return _Availability(False, "cann_ops_transformer is unavailable")
    return _Availability(True, "", ops, capacity, ep_world_size)


def is_megamoe_available(layer: FusedMoE, num_tokens: int) -> tuple[bool, str]:
    """Report whether this forward can use the Ascend MegaMOE operator."""
    result = _check_availability(layer, num_tokens)
    return result.available, result.reason


def _max_recv_tokens(layer: FusedMoE, capacity: int, ep_world_size: int) -> int:
    experts_per_rank = int(layer.num_experts) // ep_world_size
    safe_bound = capacity * ep_world_size * min(_top_k(layer), experts_per_rank)
    configured = envs.SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS.get()
    return safe_bound if configured == 0 else min(configured, safe_bound)


def _get_symm_buffer(
    layer: FusedMoE,
    get_buffer: Callable,
    capacity: int,
    ep_world_size: int,
):
    coordinator = get_moe_ep_group()
    group = coordinator.device_group
    top_k = _top_k(layer)
    intermediate_hidden = 2 * int(layer.intermediate_size_per_partition)
    max_recv_tokens = _max_recv_tokens(layer, capacity, ep_world_size)
    key = (
        id(group),
        ep_world_size,
        int(layer.num_experts),
        capacity,
        top_k,
        int(layer.hidden_size),
        intermediate_hidden,
        max_recv_tokens,
        _DISPATCH_QUANT_MODE,
        _DISPATCH_QUANT_DTYPE,
    )

    buffers = get_resources().buffers
    cache = buffers.setdefault(_BUFFER_CACHE_KEY, {})
    buffer = cache.get(key)
    if buffer is None:
        buffer = get_buffer(
            group,
            int(layer.num_experts),
            capacity,
            top_k,
            hidden=int(layer.hidden_size),
            intermediate_hidden=intermediate_hidden,
            max_recv_token_num=max_recv_tokens,
            dispatch_quant_mode=_DISPATCH_QUANT_MODE,
            dispatch_quant_out_dtype=_DISPATCH_QUANT_DTYPE,
        )
        cache[key] = buffer
    return buffer


def _call_mega_moe(
    *,
    mega_moe: Callable,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    payload: MegaMoeW8A8Payload,
    buffer,
    activation: str,
    activation_clamp: float | None,
) -> torch.Tensor:
    result = mega_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        list(payload.w13),
        list(payload.w2),
        buffer,
        l1_weights_sf=list(payload.w13_scale),
        l2_weights_sf=list(payload.w2_scale),
        activation=activation,
        activation_clamp=activation_clamp,
        weight1_type=torch.int8,
        weight2_type=torch.int8,
    )
    return result[0] if isinstance(result, tuple) else result


def _forward_available(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    availability: _Availability,
) -> torch.Tensor:
    assert availability.ops is not None
    get_buffer, mega_moe = availability.ops
    buffer = _get_symm_buffer(
        layer,
        get_buffer,
        availability.capacity,
        availability.ep_world_size,
    )
    clamp = getattr(layer.moe_runner_config, "swiglu_limit", None)
    if clamp is not None and clamp <= 0:
        clamp = None
    return _call_mega_moe(
        mega_moe=mega_moe,
        hidden_states=hidden_states,
        topk_ids=topk_output.topk_ids.to(torch.int32).contiguous(),
        topk_weights=topk_output.topk_weights.to(torch.float32).contiguous(),
        payload=layer._megamoe_w8a8_payload,
        buffer=buffer,
        activation="swiglu",
        activation_clamp=clamp,
    )


def _warn_fallback_once(reason: str) -> None:
    global _fallback_warning_emitted
    if _fallback_warning_emitted:
        return
    _fallback_warning_emitted = True
    logger.warning("Ascend MegaMOE is unavailable; falling back: %s", reason)


def forward_megamoe_or_none(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> torch.Tensor | None:
    """Run MegaMOE, or return ``None`` so the existing MoE path can run."""
    availability = _check_availability(layer, int(hidden_states.shape[0]))
    if not availability.available:
        message = f"Ascend MegaMOE is unavailable: {availability.reason}"
        if envs.SGLANG_NPU_MEGAMOE_STRICT.get():
            raise RuntimeError(message)
        _warn_fallback_once(availability.reason)
        return None
    return _forward_available(layer, hidden_states, topk_output, availability)


def forward_megamoe(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> torch.Tensor:
    """Run MegaMOE and raise if the direct-call contract cannot be met."""
    availability = _check_availability(layer, int(hidden_states.shape[0]))
    if not availability.available:
        raise RuntimeError(f"Ascend MegaMOE is unavailable: {availability.reason}")
    return _forward_available(layer, hidden_states, topk_output, availability)


def _format_weight_for_megamoe(weight: torch.Tensor) -> torch.Tensor:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

    return npu_format_cast(weight)


def cache_megamoe_w8a8_payload(layer: torch.nn.Module) -> None:
    """Preserve an additive per-expert W8A8 payload for a later MegaMOE call."""
    if getattr(layer, "_megamoe_w8a8_payload", None) is not None:
        return

    tensors = {
        name: getattr(layer, name)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        )
    }
    num_experts = int(tensors["w13_weight"].shape[0])
    if any(int(tensor.shape[0]) != num_experts for tensor in tensors.values()):
        raise ValueError("MegaMOE W8A8 tensors must have the same expert count")

    def clone_weights(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            _format_weight_for_megamoe(tensor.detach()[expert].clone().contiguous())
            for expert in range(num_experts)
        )

    def clone_scales(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.detach()[expert].clone().squeeze(-1).contiguous()
            for expert in range(num_experts)
        )

    layer._megamoe_w8a8_payload = MegaMoeW8A8Payload(
        w13=clone_weights(tensors["w13_weight"]),
        w2=clone_weights(tensors["w2_weight"]),
        w13_scale=clone_scales(tensors["w13_weight_scale"]),
        w2_scale=clone_scales(tensors["w2_weight_scale"]),
    )
