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
    return SimpleNamespace(device_group=object(), world_size=4)


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
        megamoe, "get_dp_global_num_tokens", lambda: None, raising=False
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
