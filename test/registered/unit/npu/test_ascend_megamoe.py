from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe import megamoe
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


def _hidden_states(num_tokens):
    return torch.zeros((num_tokens, 8), dtype=torch.bfloat16)


def _topk_output(num_tokens=1):
    return SimpleNamespace(
        topk_ids=torch.zeros((num_tokens, 2), dtype=torch.int64),
        topk_weights=torch.ones((num_tokens, 2), dtype=torch.bfloat16),
    )


def _fake_ops(calls):
    buffer = object()

    def get_buffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        **kwargs,
    ):
        calls.setdefault("buffer", []).append(
            {
                "group": group,
                "num_experts": num_experts,
                "num_max_tokens_per_rank": num_max_tokens_per_rank,
                "num_topk": num_topk,
                **kwargs,
            }
        )
        return buffer

    def mega_moe(*args, **kwargs):
        calls.setdefault("mega_moe", []).append((args, kwargs))
        return args[0] + 1, torch.zeros(1, dtype=torch.int32)

    return get_buffer, mega_moe


@pytest.fixture
def fake_group():
    return SimpleNamespace(device_group=object(), cpu_group=object(), world_size=4)


@pytest.fixture
def fake_layer():
    w13 = tuple(torch.ones((32, 8), dtype=torch.int8) for _ in range(4))
    w2 = tuple(torch.ones((8, 16), dtype=torch.int8) for _ in range(4))
    w13_scale = tuple(torch.ones(32, dtype=torch.float32) for _ in range(4))
    w2_scale = tuple(torch.ones(8, dtype=torch.float32) for _ in range(4))
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size_per_partition=16,
        moe_ep_size=4,
        num_experts=16,
        top_k=2,
        moe_runner_config=SimpleNamespace(
            activation="silu", is_gated=True, swiglu_limit=10.0
        ),
        _megamoe_w8a8_payload=SimpleNamespace(
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        ),
    )


@pytest.fixture
def fake_resources():
    return SimpleNamespace(buffers={})


@pytest.fixture(autouse=True)
def supported_megamoe(monkeypatch, fake_group, fake_resources):
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1")
    monkeypatch.delenv("SGLANG_NPU_MEGAMOE_STRICT", raising=False)
    monkeypatch.delenv("SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS", raising=False)
    monkeypatch.setattr(
        megamoe,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.ASCEND_MEGAMOE,
    )
    monkeypatch.setattr(megamoe, "get_moe_ep_group", lambda: fake_group)
    monkeypatch.setattr(megamoe, "get_resources", lambda: fake_resources)
    monkeypatch.setattr(megamoe, "is_npu", lambda: True)
    monkeypatch.setattr(megamoe, "_lora_enabled", lambda _layer: False)
    monkeypatch.setattr(megamoe, "_max_tokens_per_rank", lambda _layer: 128)
    monkeypatch.setattr(
        megamoe, "get_dp_global_num_tokens", lambda: [128, 128, 128, 128]
    )


def test_ascend_megamoe_is_distinct_backend(monkeypatch):
    assert MoeA2ABackend("ascend_megamoe").is_ascend_megamoe()
    assert not MoeA2ABackend("megamoe").is_ascend_megamoe()
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1")
    assert envs.SGLANG_NPU_ENABLE_MEGAMOE.get() is True


def test_ascend_megamoe_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_NPU_ENABLE_MEGAMOE", raising=False)
    assert envs.SGLANG_NPU_ENABLE_MEGAMOE.get() is False


def test_buffer_uses_rank_invariant_capacity(monkeypatch, fake_layer, fake_group):
    calls = {}
    monkeypatch.setattr(megamoe, "_load_ops", lambda: _fake_ops(calls))

    output = megamoe.forward_megamoe(fake_layer, _hidden_states(7), _topk_output(7))

    assert torch.equal(output, _hidden_states(7) + 1)
    assert calls["buffer"][0]["group"] is fake_group.device_group
    assert calls["buffer"][0]["num_max_tokens_per_rank"] == 128
    assert calls["buffer"][0]["max_recv_token_num"] == 1024
    assert calls["buffer"][0]["dispatch_quant_mode"] == 2
    assert calls["buffer"][0]["dispatch_quant_out_dtype"] == torch.int8

    mega_args, mega_kwargs = calls["mega_moe"][0]
    assert mega_args[1].dtype == torch.int32
    assert mega_args[2].dtype == torch.float32
    assert mega_kwargs["activation"] == "swiglu"
    assert mega_kwargs["activation_clamp"] == 10.0


def test_buffer_is_cached_across_runtime_token_counts(monkeypatch, fake_layer):
    calls = {}
    monkeypatch.setattr(megamoe, "_load_ops", lambda: _fake_ops(calls))

    megamoe.forward_megamoe(fake_layer, _hidden_states(7), _topk_output(7))
    megamoe.forward_megamoe(fake_layer, _hidden_states(3), _topk_output(3))

    assert len(calls["buffer"]) == 1
    assert len(calls["mega_moe"]) == 2


def test_unavailable_ops_falls_back_or_raises_in_strict_mode(monkeypatch, fake_layer):
    monkeypatch.setattr(megamoe, "_load_ops", lambda: None)

    assert megamoe.is_megamoe_available(fake_layer, 1) == (
        False,
        "cann_ops_transformer is unavailable",
    )
    monkeypatch.setenv("SGLANG_NPU_MEGAMOE_STRICT", "1")
    with pytest.raises(RuntimeError, match="cann_ops_transformer"):
        megamoe.forward_megamoe(fake_layer, _hidden_states(1), _topk_output(1))


def test_unequal_rank_token_counts_make_same_capacity_decision(monkeypatch, fake_layer):
    monkeypatch.setattr(megamoe, "_max_tokens_per_rank", lambda _layer: 4)
    monkeypatch.setattr(megamoe, "get_dp_global_num_tokens", lambda: [5, 4])
    get_buffer = Mock()
    op = Mock()
    load_ops = Mock(return_value=(get_buffer, op))
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)

    decisions = [megamoe.is_megamoe_available(fake_layer, tokens) for tokens in (5, 4)]
    outputs = [
        megamoe.forward_megamoe_or_none(
            fake_layer, _hidden_states(tokens), _topk_output(tokens)
        )
        for tokens in (5, 4)
    ]

    assert decisions[0] == decisions[1]
    assert decisions[0][0] is False
    assert "5 > 4" in decisions[0][1]
    assert outputs == [None, None]
    load_ops.assert_not_called()
    get_buffer.assert_not_called()
    op.assert_not_called()


def test_ep_max_rejects_129_and_128_tokens_before_loading_ops(
    monkeypatch, fake_layer, fake_group
):
    monkeypatch.setattr(megamoe, "_max_tokens_per_rank", lambda _layer: 128)
    monkeypatch.setattr(megamoe, "get_dp_global_num_tokens", lambda: None)
    local_counts = []

    def set_ep_max(token_count, *, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is fake_group.cpu_group
        local_counts.append(token_count.item())
        token_count.fill_(129)

    ep_all_reduce = Mock(side_effect=set_ep_max)
    monkeypatch.setattr(torch.distributed, "all_reduce", ep_all_reduce)
    get_buffer = Mock()
    op = Mock()
    load_ops = Mock(return_value=(get_buffer, op))
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)

    outputs = [
        megamoe.forward_megamoe_or_none(
            fake_layer, _hidden_states(tokens), _topk_output(tokens)
        )
        for tokens in (129, 128)
    ]

    assert outputs == [None, None]
    assert ep_all_reduce.call_count == 2
    assert local_counts == [129, 128]
    load_ops.assert_not_called()
    get_buffer.assert_not_called()
    op.assert_not_called()


def test_scheduler_token_metadata_skips_ep_max_all_reduce(monkeypatch, fake_layer):
    calls = {}
    monkeypatch.setattr(megamoe, "get_dp_global_num_tokens", lambda: [7, 3, 4, 1])
    ep_all_reduce = Mock()
    monkeypatch.setattr(torch.distributed, "all_reduce", ep_all_reduce)
    monkeypatch.setattr(megamoe, "_load_ops", lambda: _fake_ops(calls))

    output = megamoe.forward_megamoe(fake_layer, _hidden_states(7), _topk_output(7))

    assert torch.equal(output, _hidden_states(7) + 1)
    ep_all_reduce.assert_not_called()


def test_generic_megamoe_backend_never_loads_ascend_ops(monkeypatch, fake_layer):
    load_ops = Mock()
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)
    monkeypatch.setattr(
        megamoe,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.MEGAMOE,
    )

    assert not megamoe.is_ascend_megamoe_backend()
    assert megamoe.is_megamoe_available(fake_layer, 1)[0] is False
    load_ops.assert_not_called()


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda monkeypatch, layer: monkeypatch.setenv(
                "SGLANG_NPU_ENABLE_MEGAMOE", "0"
            ),
            "disabled",
        ),
        (
            lambda monkeypatch, layer: monkeypatch.setattr(
                megamoe, "is_npu", lambda: False
            ),
            "NPU",
        ),
        (
            lambda monkeypatch, layer: setattr(layer, "moe_ep_size", 1),
            "EP size",
        ),
        (
            lambda monkeypatch, layer: setattr(layer, "num_experts", 15),
            "divisible",
        ),
        (
            lambda monkeypatch, layer: delattr(layer, "_megamoe_w8a8_payload"),
            "W8A8",
        ),
        (
            lambda monkeypatch, layer: setattr(
                layer.moe_runner_config, "is_gated", False
            ),
            "SwiGLU",
        ),
        (
            lambda monkeypatch, layer: monkeypatch.setattr(
                megamoe, "_lora_enabled", lambda _layer: True
            ),
            "LoRA",
        ),
    ],
)
def test_capability_failures_do_not_initialize_collective(
    monkeypatch, fake_layer, mutate, reason
):
    get_buffer = Mock()
    load_ops = Mock(return_value=(get_buffer, Mock()))
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)
    mutate(monkeypatch, fake_layer)

    available, actual_reason = megamoe.is_megamoe_available(fake_layer, 1)

    assert available is False
    assert reason in actual_reason
    load_ops.assert_not_called()
    get_buffer.assert_not_called()


def test_configured_receive_capacity_is_clamped_to_safe_bound(monkeypatch, fake_layer):
    calls = {}
    monkeypatch.setenv("SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS", "4096")
    monkeypatch.setattr(megamoe, "_load_ops", lambda: _fake_ops(calls))

    megamoe.forward_megamoe(fake_layer, _hidden_states(1), _topk_output(1))

    assert calls["buffer"][0]["max_recv_token_num"] == 1024


def test_receive_capacity_below_safe_bound_constrains_send_admission(
    monkeypatch, fake_layer
):
    monkeypatch.setenv("SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS", "1")
    monkeypatch.setattr(megamoe, "get_dp_global_num_tokens", lambda: [1, 1, 1, 1])
    get_buffer = Mock()
    op = Mock()
    load_ops = Mock(return_value=(get_buffer, op))
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)

    available, reason = megamoe.is_megamoe_available(fake_layer, 1)

    assert available is False
    assert "receive capacity" in reason
    load_ops.assert_not_called()
    get_buffer.assert_not_called()
    op.assert_not_called()


@pytest.mark.parametrize(
    "invalid_w13",
    [
        torch.ones((32, 8), dtype=torch.float32),
        torch.ones(32, dtype=torch.int8),
    ],
)
def test_invalid_w8a8_payload_is_rejected_before_collective(
    monkeypatch, fake_layer, invalid_w13
):
    payload = fake_layer._megamoe_w8a8_payload
    fake_layer._megamoe_w8a8_payload = SimpleNamespace(
        w13=(invalid_w13, *payload.w13[1:]),
        w2=payload.w2,
        w13_scale=payload.w13_scale,
        w2_scale=payload.w2_scale,
    )
    load_ops = Mock()
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)

    available, reason = megamoe.is_megamoe_available(fake_layer, 1)

    assert available is False
    assert "W8A8" in reason
    load_ops.assert_not_called()


def test_cache_megamoe_w8a8_payload_clones_per_expert_weights(monkeypatch):
    layer = SimpleNamespace(
        w13_weight=torch.arange(24, dtype=torch.int8).reshape(2, 4, 3),
        w2_weight=torch.arange(24, dtype=torch.int8).reshape(2, 3, 4),
        w13_weight_scale=torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        w2_weight_scale=torch.arange(6, dtype=torch.float32).reshape(2, 3, 1),
    )
    original_w13 = layer.w13_weight.clone()
    monkeypatch.setattr(megamoe, "_format_weight_for_megamoe", lambda weight: weight)

    megamoe.cache_megamoe_w8a8_payload(layer)

    payload = layer._megamoe_w8a8_payload
    assert len(payload.w13) == 2
    assert torch.equal(payload.w13[0], original_w13[0])
    assert payload.w13[0].data_ptr() != layer.w13_weight[0].data_ptr()
    assert payload.w13_scale[0].shape == (4,)
    layer.w13_weight.zero_()
    assert torch.equal(payload.w13[0], original_w13[0])


@pytest.mark.parametrize("enabled", [True, False])
def test_w8a8_post_load_preserves_payload_before_normal_conversion(
    monkeypatch, enabled
):
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt import runtime_context
    from sglang.srt.hardware_backend.npu.quantization import moe_methods
    from sglang.srt.layers import moe

    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1" if enabled else "0")
    monkeypatch.setattr(
        moe, "get_moe_a2a_backend", lambda: MoeA2ABackend.ASCEND_MEGAMOE
    )
    monkeypatch.setattr(megamoe, "_format_weight_for_megamoe", lambda weight: weight)
    monkeypatch.setattr(moe_methods, "npu_format_cast", lambda weight: weight)
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
    )
    layer = torch.nn.Module()
    for prefix, shape in (("w13", (2, 32, 8)), ("w2", (2, 8, 16))):
        layer.register_parameter(
            f"{prefix}_weight",
            torch.nn.Parameter(
                torch.ones(shape, dtype=torch.int8), requires_grad=False
            ),
        )
        layer.register_parameter(
            f"{prefix}_weight_scale",
            torch.nn.Parameter(torch.ones((*shape[:2], 1)), requires_grad=False),
        )
    original_w13 = layer.w13_weight
    method = moe_methods.NPUW8A8Int8MoEMethod.__new__(moe_methods.NPUW8A8Int8MoEMethod)
    method.process_weights_after_loading(layer, "w13")
    payload = getattr(layer, "_megamoe_w8a8_payload", None)
    method.process_weights_after_loading(layer, "w2")

    assert layer.w13_weight is original_w13
    assert layer.w13_weight.shape == (2, 8, 32)
    assert layer.w2_weight.shape == (2, 16, 8)
    assert layer.w13_weight_scale.dtype == torch.bfloat16
    if not enabled:
        assert payload is None
        assert not hasattr(layer, "_megamoe_w8a8_payload")
        return
    assert layer._megamoe_w8a8_payload is payload
    assert payload.w13[0].shape == (32, 8)
    assert payload.w2[0].shape == (8, 16)
    assert payload.w13_scale[0].dtype == torch.float32
    assert payload.w2_scale[0].dtype == torch.float32


def test_w8a8_post_load_rejects_eplb_before_caching(monkeypatch):
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt import runtime_context
    from sglang.srt.hardware_backend.npu.quantization import moe_methods
    from sglang.srt.layers import moe

    monkeypatch.setattr(
        moe, "get_moe_a2a_backend", lambda: MoeA2ABackend.ASCEND_MEGAMOE
    )
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=True)),
    )
    cache = Mock()
    monkeypatch.setattr(megamoe, "cache_megamoe_w8a8_payload", cache)
    method = moe_methods.NPUW8A8Int8MoEMethod.__new__(moe_methods.NPUW8A8Int8MoEMethod)
    with pytest.raises(RuntimeError, match="--enable-eplb"):
        method.process_weights_after_loading(torch.nn.Module(), "w13")
    cache.assert_not_called()


@pytest.fixture
def fused_layer(fake_layer, monkeypatch):
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.layers.moe.fused_moe_triton import layer as layer_module

    monkeypatch.setattr(layer_module, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(
        layer_module,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
    )
    fake_layer._use_ascend_fuseep = False
    fake_layer._use_ascend_megamoe = True
    fake_layer.forward_impl = Mock()
    fake_layer.forward = layer_module.FusedMoE.forward.__get__(fake_layer)
    return fake_layer


def test_fused_moe_forward_passes_clipped_swiglu(monkeypatch, fused_layer):
    calls = {}
    monkeypatch.setattr(megamoe, "_load_ops", lambda: _fake_ops(calls))
    output = fused_layer.forward(_hidden_states(2), _topk_output(2))
    assert torch.equal(output, _hidden_states(2) + 1)
    assert calls["mega_moe"][0][1]["activation"] == "swiglu"
    assert calls["mega_moe"][0][1]["activation_clamp"] == 10.0
    fused_layer.forward_impl.assert_not_called()


def test_fused_moe_rejects_unsafe_ep_fallback(monkeypatch, fused_layer):
    monkeypatch.setattr(megamoe, "forward_megamoe_or_none", lambda *_: None)
    hidden_states, topk = _hidden_states(1), _topk_output()
    pre_quant_input = (object(), object())
    with pytest.raises(RuntimeError, match="--moe-a2a-backend deepep"):
        fused_layer.forward(hidden_states, topk, pre_quant_input=pre_quant_input)
    fused_layer.forward_impl.assert_not_called()


def test_fused_moe_rejects_eplb_before_operator_launch(monkeypatch, fused_layer):
    from sglang.srt.layers.moe.fused_moe_triton import layer as layer_module

    monkeypatch.setattr(
        layer_module,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=True)),
    )
    call = Mock()
    monkeypatch.setattr(megamoe, "forward_megamoe_or_none", call)
    with pytest.raises(RuntimeError, match="--enable-eplb"):
        fused_layer.forward(_hidden_states(1), _topk_output())
    call.assert_not_called()
    fused_layer.forward_impl.assert_not_called()


def test_fused_moe_other_backend_keeps_existing_forward(monkeypatch, fused_layer):
    fused_layer._use_ascend_megamoe = False
    call = Mock(side_effect=AssertionError("MegaMOE must not be called"))
    monkeypatch.setattr(megamoe, "forward_megamoe_or_none", call)
    hidden_states, topk = _hidden_states(1), _topk_output()
    pre_quant_input = (object(), object())
    assert (
        fused_layer.forward(hidden_states, topk, pre_quant_input)
        is fused_layer.forward_impl.return_value
    )
    fused_layer.forward_impl.assert_called_once_with(
        hidden_states, topk, pre_quant_input=pre_quant_input
    )
    call.assert_not_called()


def test_fused_moe_propagates_strict_failure(monkeypatch, fused_layer):
    monkeypatch.setenv("SGLANG_NPU_MEGAMOE_STRICT", "1")
    monkeypatch.setattr(megamoe, "_load_ops", lambda: None)
    with pytest.raises(RuntimeError, match="cann_ops_transformer"):
        fused_layer.forward(_hidden_states(1), _topk_output())
    fused_layer.forward_impl.assert_not_called()


def test_opt_in_w8a8_post_load_to_operator(monkeypatch, fused_layer):
    from sglang.srt import runtime_context
    from sglang.srt.hardware_backend.npu.quantization import moe_methods
    from sglang.srt.layers import moe

    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1")
    monkeypatch.setattr(
        moe, "get_moe_a2a_backend", lambda: MoeA2ABackend.ASCEND_MEGAMOE
    )
    monkeypatch.setattr(megamoe, "_format_weight_for_megamoe", lambda weight: weight)
    monkeypatch.setattr(moe_methods, "npu_format_cast", lambda weight: weight)
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
    )
    layer = torch.nn.Module()
    for name, value in vars(fused_layer).items():
        if name not in ("_megamoe_w8a8_payload", "forward"):
            setattr(layer, name, value)
    for prefix, shape in (("w13", (4, 32, 8)), ("w2", (4, 8, 16))):
        layer.register_parameter(
            f"{prefix}_weight",
            torch.nn.Parameter(
                torch.ones(shape, dtype=torch.int8), requires_grad=False
            ),
        )
        layer.register_parameter(
            f"{prefix}_weight_scale",
            torch.nn.Parameter(torch.ones((*shape[:2], 1)), requires_grad=False),
        )
    method = moe_methods.NPUW8A8Int8MoEMethod.__new__(moe_methods.NPUW8A8Int8MoEMethod)
    method.process_weights_after_loading(layer, "w13")
    method.process_weights_after_loading(layer, "w2")
    calls = {}
    monkeypatch.setattr(megamoe, "_load_ops", lambda: _fake_ops(calls))

    output = fused_layer.forward.__func__(layer, _hidden_states(2), _topk_output(2))

    assert output.shape == (2, layer.hidden_size)
    assert torch.equal(output, _hidden_states(2) + 1)
    assert len(calls["buffer"]) == 1
    assert len(calls["mega_moe"]) == 1
    args, kwargs = calls["mega_moe"][0]
    assert args[3][0] is layer._megamoe_w8a8_payload.w13[0]
    assert args[4][0] is layer._megamoe_w8a8_payload.w2[0]
    assert kwargs["activation_clamp"] == 10.0
    layer.forward_impl.assert_not_called()


@pytest.mark.parametrize("selected", [False, True])
def test_disabled_forward_never_loads_operator(monkeypatch, fused_layer, selected):
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "0")
    fused_layer._use_ascend_megamoe = selected
    load_ops = Mock(side_effect=AssertionError("disabled MegaMOE must not load ops"))
    monkeypatch.setattr(megamoe, "_load_ops", load_ops)
    hidden_states, topk = _hidden_states(2), _topk_output(2)

    if selected:
        with pytest.raises(RuntimeError, match="--moe-a2a-backend deepep"):
            fused_layer.forward(hidden_states, topk)
        fused_layer.forward_impl.assert_not_called()
    else:
        assert (
            fused_layer.forward(hidden_states, topk)
            is fused_layer.forward_impl.return_value
        )
        fused_layer.forward_impl.assert_called_once_with(
            hidden_states, topk, pre_quant_input=None
        )
    load_ops.assert_not_called()
