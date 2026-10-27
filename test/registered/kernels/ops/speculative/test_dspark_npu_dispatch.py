"""Numerical coverage for DSpark's SGLang-to-NPU dispatch boundary."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.dflash import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.kernels.ops.speculative.dspark.dispatch import (
    inputs_on_cuda,
    inputs_on_npu,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    SoftmaxTemp,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import MarkovGreedyStep
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.kernels.ops.speculative.row_argmax import row_argmax
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftNoise,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="base-b-test-1-npu-a3")


class TestDSparkNpuDispatch(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_build_out_tokens_matches_full_torch_buffer(self):
        from sgl_kernel_npu import sample

        for bs, gamma in ((0, 7), (1, 0), (33, 7)):
            for len_dtype in (torch.int32, torch.int64):
                for step in (1, 2):
                    with self.subTest(bs=bs, gamma=gamma, dtype=len_dtype, step=step):
                        draft = torch.arange(bs * gamma * step).reshape(
                            bs, gamma * step
                        )
                        draft = (draft + 2**40).npu()[:, ::step]
                        lengths = (
                            (torch.arange(bs * step) // step % (gamma + 1))
                            .to(len_dtype)
                            .npu()[::step]
                        )
                        bonus = (torch.arange(bs * step) - 2**40).npu()[::step]
                        inputs = {
                            "draft_tokens": draft,
                            "correct_len": lengths,
                            "bonus": bonus,
                            "verify_num_draft_tokens": gamma + 1,
                            "gamma": gamma,
                        }
                        reference = dict(
                            inputs,
                            draft_tokens=draft.cpu(),
                            correct_len=lengths.cpu(),
                            bonus=bonus.cpu(),
                        )
                        expected = BuildOutTokens.torch(**reference)
                        with mock.patch.object(
                            sample, "build_out_tokens", wraps=sample.build_out_tokens
                        ) as kernel:
                            actual = BuildOutTokens.execute(**inputs)
                        kernel.assert_called_once_with(**inputs)
                        self.assertEqual(actual.dtype, torch.int64)
                        self.assertEqual(actual.device, draft.device)
                        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_placement_uses_first_tensor(self):
        cpu = torch.zeros(1)
        npu = cpu.npu()
        self.assertFalse(inputs_on_npu(None, cpu, npu))
        self.assertTrue(inputs_on_npu(None, npu, cpu))
        self.assertFalse(inputs_on_cuda(None, cpu, npu))
        with self.assertRaises(AssertionError):
            inputs_on_npu(1, flag=True)

    def test_markov_greedy_matches_native_rounding(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.randn(3, 4097, device="npu", dtype=dtype)
            embeds = torch.randn(3, 33, device="npu", dtype=dtype)
            weight = torch.randn(4097, 33, device="npu", dtype=dtype)
            expected = (base + torch.nn.functional.linear(embeds, weight)).argmax(-1)
            actual = MarkovGreedyStep.execute(
                base_logits=base, prev_embeds=embeds, w2_weight=weight
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_argmax_and_accept_match_torch(self):
        rows, block, vocab = 9, 7, 4097
        logits = torch.randn(rows * block, vocab, device="npu")
        logits[0, 5] = float("nan")
        expected_top1 = logits.cpu().argmax(-1)
        self.assertTrue(torch.equal(row_argmax(logits).cpu(), expected_top1))
        predict = expected_top1.reshape(rows, block)
        candidates = predict.clone()
        candidates[:, 1:] = predict[:, :-1]
        for row in range(rows):
            if row % block < block - 1:
                col = row % block
                candidates[row, col + 1] = (predict[row, col] + 1) % vocab
        lengths = (candidates[:, 1:] == predict[:, :-1]).int().cumprod(1).sum(1)
        bonus = predict[torch.arange(rows), lengths]
        npu_candidates = candidates.npu()
        direct = compute_dflash_correct_drafts_and_bonus(
            candidates=npu_candidates, target_predict=predict.npu()
        )
        integrated = AcceptGreedy.execute(
            candidates=npu_candidates,
            target_logits=logits,
            verify_num_draft_tokens=block,
        )
        for result in (direct, integrated):
            self.assertEqual(result[0].dtype, torch.int32)
            self.assertEqual(result[1].dtype, torch.int64)
            self.assertTrue(torch.equal(result[0].cpu(), lengths.int()))
            self.assertTrue(torch.equal(result[1].cpu(), bonus))

    def test_softmax_temperature_mapping(self):
        logits = torch.randn(9, 4097, device="npu")
        temperatures = torch.tensor([0.3, 0.7, 1.2], device="npu")
        actual = SoftmaxTemp.execute(
            logits=logits, temperatures=temperatures, rows_per_request=3
        )
        expected = torch.softmax(
            logits.cpu() / temperatures.cpu().repeat_interleave(3)[:, None], -1
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=1e-7)

    def test_sampled_tokens_match_same_noise_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.randn(8, 4097, device="npu", dtype=dtype)
            bias = torch.randn_like(base)
            temperatures = torch.linspace(0.2, 1.3, 8, device="npu")
            temperatures[::2] = 0
            kept = torch.empty_like(base)
            offset = torch.tensor([7], device="npu")
            torch.manual_seed(123)
            noise = torch.empty(base.shape, device="npu").exponential_().cpu()
            value = (base.cpu().float() + bias.cpu().float()).to(dtype).float()
            safe = temperatures.cpu().clamp_min(1e-5)
            score = value / safe[:, None] - noise.log()
            score[temperatures.cpu() == 0] = value[temperatures.cpu() == 0]
            expected = score.argmax(-1)
            torch.manual_seed(123)
            actual = MarkovSampleStep.execute(
                base_logits=base,
                bias=bias,
                temperatures=temperatures,
                key=12345,
                offset=offset,
                step=2,
                step_dtype=dtype,
                logits_out=kept,
            )
            self.assertTrue(torch.equal(actual.cpu(), expected))
            self.assertTrue(torch.equal(kept.cpu(), value.to(dtype)))

    def test_cpu_reference_stays_available(self):
        logits = torch.randn(6, 257)
        temperatures = torch.tensor([0.7, 1.3])
        expected = torch.softmax(
            logits / temperatures.repeat_interleave(3)[:, None], -1
        )
        actual = SoftmaxTemp.execute(
            logits=logits, temperatures=temperatures, rows_per_request=3
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_sampling_gathers_logits_before_drawing_full_vocabulary_noise(self):
        base = torch.randn(3, 260, device="npu", dtype=torch.bfloat16)
        bias = torch.randn_like(base)
        remote = torch.randn_like(base)
        temperatures = torch.tensor([0.0, 0.7, 1.3], device="npu")
        kept = torch.empty(3, 513, device="npu", dtype=base.dtype)
        local = (base.float() + bias.float()).to(base.dtype)
        value = torch.cat((local.cpu(), remote.cpu()), dim=1)[:, :513].float()
        torch.manual_seed(123)
        noise = torch.empty(kept.shape, device="npu").exponential_().cpu()
        score = value / temperatures.cpu().clamp_min(1e-5)[:, None] - noise.log()
        score[0] = value[0]
        torch.manual_seed(123)
        actual = MarkovSampleStep.execute(
            base_logits=base,
            bias=bias,
            temperatures=temperatures,
            key=12345,
            offset=torch.tensor([7], device="npu"),
            step=2,
            step_dtype=base.dtype,
            logits_out=kept,
            gather=lambda partial: partial,
            gather_logits=lambda shard: torch.cat((shard, remote), dim=1),
        )
        self.assertTrue(torch.equal(actual.cpu(), score.argmax(-1)))
        self.assertTrue(torch.equal(kept.cpu(), value.to(base.dtype)))

    def test_sampling_graph_replay_reads_refreshed_native_noise(self):
        base = torch.zeros(32, 257, device="npu")
        noise = torch.empty_like(base).exponential_()
        kwargs = dict(
            base_logits=base,
            bias=base,
            temperatures=torch.ones(32, device="npu"),
            key=12345,
            offset=torch.tensor([7], device="npu"),
            step=2,
            step_dtype=base.dtype,
            exp_noise=noise,
        )
        MarkovSampleStep.execute(**kwargs)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = MarkovSampleStep.execute(**kwargs)
        torch.npu.current_stream().wait_stream(stream)
        previous = None
        for _ in range(3):
            noise.exponential_()
            graph.replay()
            current = output.cpu()
            self.assertTrue(torch.equal(current, (-noise.cpu().log()).argmax(-1)))
            if previous is not None:
                self.assertFalse(torch.equal(current, previous))
            previous = current


class TestDSparkNpuDraftGraph(CustomTestCase):
    def test_replay_refreshes_noise_even_when_sampling_params_are_cached(self):
        torch.manual_seed(42)
        bs, gamma, vocab, hidden = 4, 3, 257, 8
        head = VanillaMarkov(vocab_size=vocab, markov_rank=8).to(
            device="npu", dtype=torch.bfloat16
        )
        weight = torch.randn(vocab, hidden, device="npu", dtype=torch.bfloat16)
        model = SimpleNamespace(
            markov_head=head,
            confidence_head=None,
            sample_from_anchor=True,
            lm_head=SimpleNamespace(org_vocab_size=vocab),
            compute_base_logits=lambda h: (h @ weight.T, h),
        )
        sampler = DsparkDraftSampler(
            model=model,
            gamma=gamma,
            max_bs=2 * bs,
            device="npu",
            noise=DraftNoise(
                key=5, offset=torch.zeros(1, dtype=torch.int64, device="npu")
            ),
            tp_sync=SimpleNamespace(sync=lambda site, value: value),
            greedy_draft=False,
        )
        # Allocate before capture, as the real runner's warmup does.
        sampler.step_logits = torch.empty(
            2 * bs, gamma, vocab, device="npu", dtype=torch.bfloat16
        )
        features = torch.randn(bs * gamma, hidden, device="npu", dtype=torch.bfloat16)
        input_ids = torch.randint(vocab, (bs * gamma,), device="npu")
        info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.full((bs, 1), 0.7, device="npu"),
            top_ks=torch.tensor([1, 50, 1, 50], device="npu"),
        )
        proposer = SimpleNamespace(
            _staged_params_key=None,
            num_staged_params_calls=0,
            num_staged_params_skips=0,
        )
        DraftBlockProposer._stage_sampling_params(
            proposer, sampler, bs=bs, sampling_info=info
        )
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                sampler(features, input_ids)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                sampler(features, input_ids)
        torch.npu.current_stream().wait_stream(stream)
        base = model.compute_base_logits(features)[0].view(bs, gamma, vocab)
        for _ in range(3):
            before = torch.npu.get_rng_state()
            DraftBlockProposer._stage_sampling_params(
                proposer, sampler, bs=bs, sampling_info=info
            )
            self.assertFalse(torch.equal(before, torch.npu.get_rng_state()))
            graph.replay()
            tokens = sampler.out[: bs * gamma].view(bs, gamma).cpu()
            kept = sampler.step_logits[:bs].cpu()
            noise = sampler.exp_noise[:, :bs].cpu()
            prev = input_ids.view(bs, gamma)[:, 0]
            temps = sampler.params.step_temperatures[:bs].cpu()
            for step in range(gamma):
                bias, _ = head.step_bias(prev, hidden_states=None, state=None)
                value = (base[:, step].float() + bias.float()).to(torch.bfloat16).cpu()
                score = (
                    value.float() / temps.clamp_min(1e-5)[:, None] - noise[step].log()
                )
                score[temps == 0] = value[temps == 0].float()
                expected = score.argmax(-1)
                self.assertTrue(torch.equal(tokens[:, step], expected))
                self.assertTrue(torch.equal(kept[:, step], value))
                prev = expected.npu()
        self.assertEqual(proposer.num_staged_params_skips, 3)
        greedy = SimpleNamespace(
            is_all_greedy=True,
            temperatures=info.temperatures,
            top_ks=torch.ones(bs, device="npu", dtype=torch.int64),
        )
        before = torch.npu.get_rng_state()
        DraftBlockProposer._stage_sampling_params(
            proposer, sampler, bs=bs, sampling_info=greedy
        )
        self.assertTrue(torch.equal(before, torch.npu.get_rng_state()))
        graph.replay()
        self.assertTrue(
            torch.equal(
                sampler.out[: bs * gamma].view(bs, gamma),
                sampler.step_logits[:bs].float().argmax(-1),
            )
        )


if __name__ == "__main__":
    unittest.main()
