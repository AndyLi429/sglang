"""Host tests for the DSPark/NPU sampling boundary, without device execution."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _module(name, **values):
    return types.SimpleNamespace(**values)


class TestDsparkSamplingAdapter(CustomTestCase):
    def _load(self):
        def buffers(*, bs, draft_token_num, device):
            index = torch.arange(bs * draft_token_num).view(bs, draft_token_num)
            return (
                index,
                index.clone(),
                index.clone(),
                torch.empty(bs * draft_token_num, dtype=torch.int32),
                torch.empty(bs, draft_token_num, dtype=torch.int32),
                torch.empty(bs, dtype=torch.int32),
            )

        def chain(**kw):
            # Model the external sampler's output contract; enforce the NPU
            # random-buffer stride that differs from the GPU sampler.
            if kw["uniform_samples"].shape != kw["candidates"].shape:
                raise ValueError("incorrect NPU random-buffer stride")
            kw["accept_token_num"].copy_(torch.tensor([0, 2], dtype=torch.int32))
            kw["accept_index"].copy_(kw["retrive_index"].int())
            kw["predicts"].copy_(torch.arange(kw["predicts"].numel()) + 1000)

        def cap(*, correct_len, verify_lens):
            capped = torch.minimum(correct_len, verify_lens - 1)
            return capped, correct_len - capped

        def softmax(*, logits, temperatures, rows_per_request):
            return torch.softmax(
                logits
                / temperatures.reshape(-1).repeat_interleave(rows_per_request)[:, None],
                -1,
            )

        def gather(*, accept_index, predicts, correct_len):
            return predicts[
                accept_index[
                    torch.arange(correct_len.numel()), correct_len.long()
                ].long()
            ].long()

        def target_probs(**kw):
            if kw["max_top_k"] != 2 or kw["uniform_top_k_value"] != 2:
                raise ValueError("top-k metadata was not forwarded")
            logits = kw["next_token_logits"]
            vals, ids = logits.topk(2)
            return (
                torch.zeros_like(logits)
                .scatter_(1, ids, vals.softmax(-1))
                .view(kw["bs"], kw["draft_token_num"], -1)
            )

        modules = {
            "sgl_kernel_npu.sample.chain_speculative_sampling": _module(
                "chain", chain_speculative_sampling_triton=chain
            ),
            "sgl_kernel_npu.sample.dspark_accept": _module(
                "accept",
                CapCorrectLen=_module("cap", execute=cap),
                SoftmaxTemp=_module("softmax", execute=softmax),
                gather_two_level_bonus_triton=gather,
            ),
            "sglang.srt.speculative.dflash_info_v2": _module(
                "info", DFlashDraftInputV2=object
            ),
            "sglang.srt.speculative.dflash_utils": _module(
                "utils",
                _get_or_create_chain_verify_buffers=buffers,
                build_speculative_verify_target_probs=target_probs,
            ),
        }
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/hardware_backend/npu/dspark_sampling.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_dspark_sampling_adapter_under_test", path
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        return module

    def _kwargs(self, top_k=False):
        return dict(
            candidates=torch.zeros(2, 4, dtype=torch.int64),
            target_logits=torch.randn(8, 7),
            draft_probs=torch.full((2, 3, 7), 1 / 7),
            sampling_info=types.SimpleNamespace(
                need_top_k_sampling=top_k,
                need_top_p_sampling=False,
                temperatures=torch.ones(2, 1),
            ),
            draft_input=types.SimpleNamespace(max_top_k=2, uniform_top_k_value=2),
            gamma=3,
            verify_num_draft_tokens=4,
        )

    def test_sampling_returns_npu_lengths_and_bonus(self):
        module = self._load()
        lengths, bonus, trim = module.AcceptSampling.execute(**self._kwargs())
        torch.testing.assert_close(lengths, torch.tensor([0, 2], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([1000, 1006]))
        torch.testing.assert_close(trim, torch.zeros(2, dtype=torch.int32))

    def test_cutoff_changes_bonus_and_reports_trim(self):
        module = self._load()
        lengths, bonus, trim = module.AcceptSampling.execute(
            **self._kwargs(top_k=True),
            cutoff_verify_lens=torch.tensor([1, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(lengths, torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([1000, 1005]))
        torch.testing.assert_close(trim, torch.tensor([0, 1], dtype=torch.int32))

    def test_block_verification_is_not_silently_downgraded(self):
        module = self._load()
        with self.assertRaisesRegex(NotImplementedError, "block verification"):
            module.AcceptSampling.execute(**self._kwargs(), block_verification=True)


if __name__ == "__main__":
    unittest.main()
