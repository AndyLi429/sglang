from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.deepseek_v4_dspark import DSparkAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeLinear(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        *args,
        params_dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype),
            requires_grad=False,
        )


def test_dspark_explicitly_disables_arch35_mxfp8_wo_a():
    config = SimpleNamespace(
        hidden_size=16,
        qk_rope_head_dim=4,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        o_groups=2,
        q_lora_rank=8,
        o_lora_rank=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        compress_rope_theta=10_000,
        window_size=8,
    )
    parallel = SimpleNamespace(attn_tp_rank=0, attn_tp_size=1, tp_size=1)
    platform = SimpleNamespace(is_blackwell=False)
    quant_config = SimpleNamespace(get_name=lambda: "fp8")

    with (
        patch("sglang.srt.models.deepseek_v4.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.models.deepseek_v4.use_npu_arch35_mxfp8_wo_a",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.deepseek_v4.get_rope_config", return_value=(10_000, None)
        ),
        patch("sglang.srt.models.deepseek_v4.ColumnParallelLinear", _FakeLinear),
        patch("sglang.srt.models.deepseek_v4.RowParallelLinear", _FakeLinear),
        patch("sglang.srt.models.deepseek_v4.ReplicatedLinear", _FakeLinear),
        patch(
            "sglang.srt.models.deepseek_v4_dspark.get_parallel", return_value=parallel
        ),
        patch(
            "sglang.srt.models.deepseek_v4_dspark.get_platform", return_value=platform
        ),
        patch("sglang.srt.models.deepseek_v4_dspark.RadixAttention", _FakeLinear),
        patch(
            "sglang.kernels.ops.attention.deepseek_v4_rope.precompute_freqs_cis",
            return_value=torch.empty(16, 2, dtype=torch.complex64),
        ),
    ):
        attention = DSparkAttention(config, layer_id=0, quant_config=quant_config)

    assert attention.use_npu_arch35_mxfp8_wo_a is False
    assert attention.wo_a.weight.dtype == torch.bfloat16
    assert not hasattr(attention.wo_a, "weight_scale_inv")
