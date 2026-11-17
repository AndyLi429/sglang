"""Regress packed W4A8 NZ migration failing TransData during EPLB."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.eplb import _ExpertWeightStaging
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import torch_npu
except ImportError:
    torch_npu = None

register_npu_ci(est_time=15, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(torch_npu is not None and torch.npu.is_available(), "NPU required")
class TestNpuExpertWeightStaging(CustomTestCase):
    def setUp(self):
        self.previous_internal_format = (
            torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        )
        torch.npu.config.allow_internal_format = True

    def tearDown(self):
        torch.npu.synchronize()
        torch.npu.config.allow_internal_format = self.previous_internal_format

    @staticmethod
    def _physical_copy(weight):
        shape = (
            weight.shape[0],
            weight.untyped_storage().nbytes() // weight.shape[0],
        )
        out = torch.empty(shape, dtype=torch.int8, device=weight.device)
        torch.ops.npu.copy_memory_(out, weight.view(torch.int8), True)
        return out

    def test_packed_nz_migration_preserves_storage_and_graph(self):
        # Match ModelSlim: cast packed bytes to NZ before viewing them as int32.
        for rows, cols, shared in [(128, 64, 0), (5120, 320, 0), (129, 68, 1)]:
            with self.subTest(rows=rows, cols=cols, shared=shared):
                data = (
                    torch.arange((3 + shared) * rows * cols, dtype=torch.int32)
                    .remainder(251)
                    .to(torch.int8)
                    .reshape(3 + shared, rows, cols)
                    .npu()
                )
                full_weight = torch.ops.npu.npu_format_cast(data, 29).view(torch.int32)
                weight = full_weight[:3]
                pointer = weight.data_ptr()
                original = self._physical_copy(full_weight).cpu()
                staging = _ExpertWeightStaging({0: [weight]}, [0], rank=0)
                temp = staging.create_temp_buffers([weight])[0]
                observed = torch.empty_like(self._physical_copy(full_weight))
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    torch.ops.npu.copy_memory_(
                        observed, full_weight.view(torch.int8), True
                    )

                for shift in [1, 2]:
                    buffers = staging.load([weight])
                    torch.npu.synchronize()
                    self.assertEqual(tuple(temp.shape), tuple(buffers[0].shape))
                    buffers[0].copy_(
                        torch.roll(original[:3], shifts=shift, dims=0).to(weight.device)
                    )
                    staging.store([weight], buffers)
                    graph.replay()
                    torch.npu.synchronize()
                    expected = original.clone()
                    expected[:3] = original[:3].roll(shift, dims=0)
                    self.assertTrue(torch.equal(observed.cpu(), expected))
                    self.assertEqual(weight.data_ptr(), pointer)
                    self.assertEqual(torch.ops.npu.get_npu_format(weight), 29)
                    unpacked = torch.ops.npu.npu_format_cast(
                        full_weight.view(torch.int8), 2
                    )
                    expected = data.cpu().clone()
                    expected[:3] = expected[:3].roll(shift, dims=0)
                    self.assertTrue(torch.equal(unpacked.cpu(), expected))

    def test_nd_and_strided_weights_keep_logical_expert_values(self):
        base = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 4, 8).npu()
        for weight in [base.clone(), base.clone().transpose(1, 2)]:
            with self.subTest(stride=weight.stride()):
                original = weight.cpu().clone()
                pointer = weight.data_ptr()
                staging = _ExpertWeightStaging({0: [weight]}, [0], rank=0)
                buffers = staging.load([weight])
                buffers[0].copy_(buffers[0].flip(0))
                staging.store([weight], buffers)
                self.assertTrue(torch.equal(weight.cpu(), original.flip(0)))
                self.assertEqual(weight.data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()
