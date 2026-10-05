import math
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe import mega_moe
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def server_config(**overrides):
    fields = dict(
        device="npu",
        moe_a2a_backend="ascend_fuseep",
        moe_runner_backend="ascend",
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="disabled"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
        enable_torch_compile=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_eplb=False,
        enable_waterfill=False,
        enable_lora=False,
        enable_mixed_chunk=False,
        enable_dynamic_chunking=False,
        enable_memory_saver=False,
        cpu_offload_gb=0,
        speculative_algorithm=None,
        ep_num_redundant_experts=0,
        attn_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        pp_size=1,
        elastic_ep_backend=None,
        ep_join_mode=None,
        tp_size=4,
        chunked_prefill_size=256,
        max_prefill_tokens=512,
        max_running_requests=8,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestAscendMegaMoE(CustomTestCase):
    def test_scale_encoding(self):
        scales = torch.tensor([0.1, 0.25, 1.0, 3.14159], dtype=torch.float32)
        for int4 in (False, True):
            with self.subTest(int4=int4):
                expected = []
                for scale in scales.tolist():
                    bits = struct.unpack("<I", struct.pack("<f", scale))[0]
                    if int4:
                        bits = (bits >> 13 << 13) + 2**46
                    expected.append(bits)
                encoded = mega_moe._pack_scale(scales, int4=int4)
                self.assertEqual(encoded.dtype, torch.uint64)
                self.assertEqual(encoded.tolist(), expected)

    def test_int4_compensation_signed_channels(self):
        # Includes every signed nibble and a different scale per channel.
        logical = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
        packed = (logical[:, 0::2] & 15) | ((logical[:, 1::2] & 15) << 4)
        scales = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0])
        encoded = mega_moe._pack_scale(scales, int4=True)
        # Decode independently through Python's IEEE-754 representation.
        truncated = torch.tensor(
            [
                struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]
                for value in encoded.tolist()
            ]
        )
        expected = logical.float().sum(dim=0) * truncated * 8
        torch.testing.assert_close(
            mega_moe._int4_compensation(packed, encoded), expected
        )
        self.assertLess(expected[0].item(), 0)
        self.assertGreater(expected[-1].item(), 0)

    def test_scheduler_budget(self):
        with envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(512):
            mega_moe.validate_mega_moe_server_args(server_config())
            for values in (
                dict(chunked_prefill_size=None),
                dict(chunked_prefill_size=-1),
                dict(chunked_prefill_size=513),
                dict(max_prefill_tokens=513),
                dict(max_running_requests=None),
                dict(max_running_requests=0),
                dict(max_running_requests=513),
            ):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    mega_moe.validate_mega_moe_server_args(server_config(**values))
        # Raw budget fits, but TP alignment would exceed the symmetric buffer.
        with envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(511):
            with self.assertRaisesRegex(ValueError, "TP-aligned"):
                mega_moe.validate_mega_moe_server_args(
                    server_config(max_prefill_tokens=511)
                )
        for capacity in (0, 4097):
            with envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(capacity):
                with self.assertRaisesRegex(ValueError, "4096"):
                    mega_moe.validate_mega_moe_server_args(server_config())

    def test_unsupported_execution_modes(self):
        for values in (
            dict(device="cuda"),
            dict(moe_a2a_backend="megamoe"),
            dict(moe_runner_backend="triton"),
            dict(enable_torch_compile=True),
            dict(enable_two_batch_overlap=True),
            dict(enable_single_batch_overlap=True),
            dict(enable_eplb=True),
            dict(enable_waterfill=True),
            dict(enable_lora=True),
            dict(enable_mixed_chunk=True),
            dict(enable_dynamic_chunking=True),
            dict(enable_memory_saver=True),
            dict(cpu_offload_gb=1),
            dict(speculative_algorithm="EAGLE"),
            dict(ep_num_redundant_experts=1),
            dict(attn_dp_size=2),
            dict(attn_cp_size=2),
            dict(dcp_size=2),
            dict(pp_size=2),
            dict(elastic_ep_backend="mooncake"),
            dict(ep_join_mode="scale"),
        ):
            with (
                self.subTest(values=values),
                envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(512),
            ):
                with self.assertRaises(ValueError):
                    mega_moe.validate_mega_moe_server_args(server_config(**values))
        for phase in ("decode", "prefill"):
            cfg = server_config()
            getattr(cfg.cuda_graph_config, phase).backend = "full"
            with self.assertRaisesRegex(ValueError, "eager"):
                mega_moe.validate_mega_moe_server_args(cfg)

    def test_layer_rejects_unsupported_contracts_before_weight_conversion(self):
        fields = dict(
            moe_tp_size=1,
            moe_ep_size=4,
            num_experts=128,
            hidden_size=2048,
            intermediate_size_per_partition=768,
            with_bias=False,
        )
        config = dict(
            top_k=8,
            params_dtype=torch.bfloat16,
            num_fused_shared_experts=0,
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            activation="silu",
            swiglu_limit=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
        )
        for changes in (
            dict(moe_tp_size=2),
            dict(num_experts=127),
            dict(hidden_size=8193),
            dict(hidden_size=2049),
            dict(intermediate_size_per_partition=3585),
            dict(with_bias=True),
        ):
            layer = SimpleNamespace(
                **(fields | changes), moe_runner_config=SimpleNamespace(**config)
            )
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                mega_moe.validate_mega_moe_layer(layer)
        for changes in (
            dict(top_k=17),
            dict(params_dtype=torch.float16),
            dict(num_fused_shared_experts=1),
            dict(is_gated=False),
            dict(apply_router_weight_on_input=True),
            dict(no_combine=True),
            dict(activation="swigluoai"),
            dict(swiglu_limit=7.0),
            dict(gemm1_beta=1.0),
            dict(activation="situ", gemm1_alpha=math.nan),
        ):
            layer = SimpleNamespace(
                **fields, moe_runner_config=SimpleNamespace(**(config | changes))
            )
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                mega_moe.validate_mega_moe_layer(layer)

    def test_routed_ffn_adapter_and_padding(self):
        torch.manual_seed(7)
        w13 = torch.randn(2, 32, 64).bfloat16() / 32
        w2 = torch.randn(2, 32, 32).bfloat16() / 32
        x = torch.randn(3, 32).bfloat16()
        x[2] = float("nan")
        ids = torch.tensor([[0, 1], [1, -1], [-1, -1]])
        weights = torch.tensor([[0.3, 0.7], [0.5, 99.0], [99.0, 99.0]])
        layer = SimpleNamespace(
            num_experts=2,
            top_k=2,
            moe_runner_config=SimpleNamespace(
                activation="silu", routed_scaling_factor=2.0
            ),
            _ascend_mega_moe_weights={
                "w13": (list(w13), None, None),
                "w2": (list(w2), None, None),
            },
            _ascend_mega_moe_quant_mode=0,
        )

        class ReferenceBuffer:
            # Replace only the external collective/operator with a CPU FFN.
            def fused_deep_moe(self, **kwargs):
                output = torch.zeros_like(kwargs["x"])
                for token in range(output.shape[0]):
                    for slot in range(kwargs["topk_idx"].shape[1]):
                        expert = kwargs["topk_idx"][token, slot]
                        gate, up = (
                            kwargs["x"][token].float()
                            @ kwargs["gmm1_permuted_weight"][expert].float()
                        ).chunk(2)
                        result = (F.silu(gate) * up) @ kwargs["gmm2_weight"][
                            expert
                        ].float()
                        output[token] += (
                            result * kwargs["topk_weights"][token, slot]
                        ).bfloat16()
                return output, None

        expected = torch.zeros_like(x)
        for token, routes in enumerate(([(0, 0.3), (1, 0.7)], [(1, 0.5)])):
            for expert, weight in routes:
                gate, up = (x[token].float() @ w13[expert].float()).chunk(2)
                expected[token] += (
                    (F.silu(gate) * up @ w2[expert].float()) * weight
                ).bfloat16()
        with patch.object(
            mega_moe, "_get_mega_moe_buffer", return_value=ReferenceBuffer()
        ):
            actual = mega_moe.forward_mega_moe(
                layer, x, StandardTopKOutput(weights, ids, None)
            )
            torch.testing.assert_close(actual, expected)
            empty = mega_moe.forward_mega_moe(
                layer, x[:0], mega_moe._empty_topk(layer, x.device)
            )
            self.assertEqual(empty.shape, (0, 32))
        self.assertTrue(torch.isfinite(actual).all())
        # Model-owned routed scaling must not be applied again by the adapter.
        self.assertFalse(torch.equal(actual[:2], expected[:2] * 2))


if __name__ == "__main__":
    unittest.main()
