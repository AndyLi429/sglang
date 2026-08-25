import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Load the quantization package first so `base_config`, `moe_methods`, and
# `linear_method_npu` initialize in dependency order. Importing
# `linear_method_npu` directly from a cold process triggers a circular import:
# linear_method_npu -> base_config -> quantization/__init__ ->
# gguf/unquant/gptq_moe -> moe_methods -> linear_method_npu (partially
# initialized, `_get_float8_e8m0fnu_dtype` not yet defined). Initializing the
# package first mirrors how the engine loads quantization at model-config time.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8BlockFP8LinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod


class TestNPUW8A8BlockFP8Linear(unittest.TestCase):
    def test_fp8_method_selects_npu_kernel_instead_of_function_dispatch(self):
        quant_config = SimpleNamespace(
            use_mxfp8=False,
            weight_block_size=[64, 128],
            is_checkpoint_fp8_serialized=True,
        )

        with (
            patch(
                "sglang.srt.layers.quantization.fp8.cutlass_fp8_supported",
                return_value=False,
            ),
            patch("sglang.srt.layers.quantization.fp8._is_npu", True),
            patch(
                "sglang.srt.layers.quantization.fp8.has_npu_a5_support",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8.dispatch_w8a8_block_fp8_linear",
                side_effect=AssertionError("NPU must not use function dispatch"),
            ),
        ):
            method = Fp8LinearMethod(quant_config)

        self.assertIsInstance(method.npu_block_fp8_kernel, NPUW8A8BlockFP8LinearMethod)

    def test_compressed_tensors_selects_npu_kernel(self):
        weight_quant = SimpleNamespace(
            strategy=QuantizationStrategy.BLOCK,
            block_structure=[64, 128],
        )

        with (
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8._is_npu",
                True,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8.has_npu_a5_support",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8.dispatch_w8a8_block_fp8_linear",
                side_effect=AssertionError("NPU must not use function dispatch"),
            ),
        ):
            scheme = CompressedTensorsW8A8Fp8(weight_quant, False)

        self.assertIsInstance(scheme.npu_block_fp8_kernel, NPUW8A8BlockFP8LinearMethod)

    def test_requantizes_non_128_block_fp8_weights_for_a5_mxfp8(self):
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.ones(64, 128, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.ones(1, 1, dtype=torch.float32), requires_grad=False
            ),
        )
        method = NPUW8A8BlockFP8LinearMethod([64, 128])

        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.data.shape, (128, 64))
        self.assertEqual(layer.weight_scale_inv.data.shape, (2, 64, 2))
        self.assertTrue(layer.weight_scale_inv.format_ue8m0)

    def test_rejects_non_fp8_weight(self):
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.empty(64, 128, dtype=torch.bfloat16), requires_grad=False
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.ones(1, 1, dtype=torch.float32), requires_grad=False
            ),
        )
        method = NPUW8A8BlockFP8LinearMethod([128, 128])

        with self.assertRaisesRegex(ValueError, "expects float8_e4m3fn weights"):
            method.process_weights_after_loading(layer)

    def test_quantizes_flattened_input_and_restores_batch_shape(self):
        input_tensor = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        bias = torch.randn(64, dtype=torch.float32)
        quantized = torch.empty(6, 128, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(6, 2, 2, dtype=torch.uint8)
        matmul_output = torch.randn(6, 64, dtype=torch.bfloat16)

        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (quantized, input_scale)
        npu_ops.npu_quant_matmul.return_value = matmul_output
        layer = SimpleNamespace(
            weight=weight,
            weight_scale_inv=weight_scale,
            bias=bias,
            bias_fp32=None,
        )
        method = NPUW8A8BlockFP8LinearMethod([64, 128])
        with patch.object(torch.ops, "npu", npu_ops, create=True):
            output = method.apply(layer, input_tensor, bias)

        self.assertEqual(output.shape, (2, 3, 64))
        quant_call = npu_ops.npu_dynamic_mx_quant.call_args
        self.assertEqual(quant_call.args[0].shape, (6, 128))
        self.assertEqual(quant_call.kwargs["dst_type"], torch.float8_e4m3fn)

        matmul_call = npu_ops.npu_quant_matmul.call_args
        self.assertIs(matmul_call.args[0], quantized)
        self.assertIs(matmul_call.args[1], weight)
        self.assertIs(matmul_call.args[2], weight_scale)
        self.assertIs(matmul_call.kwargs["pertoken_scale"], input_scale)
        self.assertIs(matmul_call.kwargs["bias"], bias)
        self.assertEqual(matmul_call.kwargs["group_sizes"], [1, 1, 32])

    def test_preserves_supported_input_dtype(self):
        input_tensor = torch.randn(2, 128, dtype=torch.float16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)
        layer = SimpleNamespace(
            weight=weight,
            weight_scale_inv=weight_scale,
            bias=None,
            bias_fp32=None,
        )
        method = NPUW8A8BlockFP8LinearMethod([128, 128])

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            method.apply(layer, input_tensor)

        self.assertEqual(
            npu_ops.npu_quant_matmul.call_args.kwargs["output_dtype"],
            torch.float16,
        )

    def test_converts_bias_to_float32(self):
        input_tensor = torch.randn(2, 128, dtype=torch.bfloat16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        bias = torch.randn(64, dtype=torch.bfloat16)
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)
        layer = SimpleNamespace(
            weight=weight,
            weight_scale_inv=weight_scale,
            bias=bias,
            bias_fp32=None,
        )
        method = NPUW8A8BlockFP8LinearMethod([128, 128])

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            method.apply(layer, input_tensor, bias)

        quant_bias = npu_ops.npu_quant_matmul.call_args.kwargs["bias"]
        self.assertEqual(quant_bias.dtype, torch.float32)
        torch.testing.assert_close(quant_bias, bias.float())

    def test_supports_compressed_tensors_weight_scale_name(self):
        input_tensor = torch.randn(2, 128, dtype=torch.bfloat16)
        weight_scale = torch.nn.Parameter(
            torch.ones(1, 1, dtype=torch.float32), requires_grad=False
        )
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.ones(64, 128, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            weight_scale=weight_scale,
            bias=None,
            bias_fp32=None,
        )
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)
        method = NPUW8A8BlockFP8LinearMethod(
            [64, 128], weight_scale_name="weight_scale"
        )

        method.process_weights_after_loading(layer)

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            method.apply(layer, input_tensor)

        self.assertEqual(layer.weight.shape, (128, 64))
        self.assertEqual(weight_scale.shape, (2, 64, 2))
        self.assertTrue(weight_scale.format_ue8m0)
        self.assertIs(npu_ops.npu_quant_matmul.call_args.args[2], weight_scale)


if __name__ == "__main__":
    unittest.main()
