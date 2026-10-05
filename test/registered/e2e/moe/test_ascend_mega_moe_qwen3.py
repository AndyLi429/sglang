"""Qwen3-30B-A3B W8A8: DeepEP reference versus A3 MegaMoE, on four NPUs."""

import json
import math
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=2400, suite="full-4-npu-a3", nightly=True)


class TestAscendMegaMoEQwen3(CustomTestCase):
    model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None

    def _launch(self, mega_moe):
        args = [
            "--device",
            "npu",
            "--dtype",
            "bfloat16",
            "--tp-size",
            "4",
            "--attention-backend",
            "ascend",
            "--moe-runner-backend",
            "ascend",
            "--disable-prefill-cuda-graph",
            "--disable-decode-cuda-graph",
            "--disable-shared-experts-fusion",
            "--chunked-prefill-size",
            "256",
            "--max-prefill-tokens",
            "512",
            "--max-running-requests",
            "8",
            "--mem-fraction-static",
            "0.75",
            "--trust-remote-code",
        ]
        if mega_moe:
            args += ["--moe-a2a-backend", "ascend_fuseep", "--fuseep-mode", "3"]
        else:
            args += ["--moe-a2a-backend", "deepep", "--deepep-mode", "normal"]
        type(self).process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            device="npu",
            env={
                **os.environ,
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "HCCL_BUFFSIZE": "200",
                "HCCL_EXEC_TIMEOUT": "200",
            },
        )
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        info = response.json()
        self.assertEqual(
            info["moe_a2a_backend"], "ascend_fuseep" if mega_moe else "deepep"
        )
        if mega_moe:
            self.assertEqual(info["fuseep_mode"], 3)

    def _generate(self, prompts):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def _exercise_server(self):
        # Single-token decode can leave EP input shards empty. The long prompt
        # exceeds both the prefill chunk and the operator capacity, so the
        # scheduler must split it rather than forwarding an oversized tensor.
        prompts = [
            "The capital of France is",
            "Explain why the sum of two even integers is always even.",
            "A library has red and blue books. " * 128 + "Describe this library.",
        ]
        single = self._generate(prompts[0])
        mixed = self._generate(prompts)
        self.assertEqual(len(mixed), len(prompts))
        self.assertGreater(mixed[-1]["meta_info"]["prompt_tokens"], 512)
        for result in [single, *mixed]:
            self.assertEqual(result["meta_info"]["completion_tokens"], 16)
            self.assertTrue(result["text"])
            self.assertTrue(
                all(
                    math.isfinite(item[0])
                    for item in result["meta_info"]["output_token_logprobs"]
                )
            )
        # More requests than the running-request limit exercises queuing,
        # repeated collectives, mixed lengths and decode/prefill transitions.
        with ThreadPoolExecutor(max_workers=12) as executor:
            outputs = list(
                executor.map(
                    self._generate,
                    [f"Request {i}: {prompts[i % 3]}" for i in range(12)],
                )
            )
        for result in outputs:
            self.assertEqual(result["meta_info"]["completion_tokens"], 16)
            self.assertTrue(result["text"])
        metrics = run_eval(
            SimpleNamespace(
                max_tokens=512,
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                num_examples=200,
                num_threads=8,
                num_shots=5,
            )
        )
        self.assertGreaterEqual(metrics["score"], 0.90)
        return [single, *mixed], metrics

    def test_logprobs_scheduling_and_gsm8k(self):
        results = []
        for mega_moe in (False, True):
            try:
                self._launch(mega_moe)
                results.append(self._exercise_server())
            finally:
                type(self).tearDownClass()
        (reference, baseline_metrics), (actual, mega_metrics) = results
        errors = []
        for before, after in zip(reference, actual):
            before_probs = before["meta_info"]["input_token_logprobs"]
            after_probs = after["meta_info"]["input_token_logprobs"]
            self.assertEqual(
                [item[1] for item in before_probs], [item[1] for item in after_probs]
            )
            self.assertGreater(len(before_probs), 1)
            for lhs, rhs in zip(before_probs, after_probs):
                self.assertEqual(lhs[0] is None, rhs[0] is None)
                if lhs[0] is not None:
                    self.assertTrue(math.isfinite(lhs[0]) and math.isfinite(rhs[0]))
                    errors.append(abs(lhs[0] - rhs[0]))
        rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
        max_error = max(errors)
        print(
            json.dumps(
                {
                    "input_logprob_rmse": rmse,
                    "input_logprob_max_error": max_error,
                    "baseline": baseline_metrics,
                    "mega_moe": mega_metrics,
                },
                indent=2,
            )
        )
        self.assertLessEqual(rmse, 0.1)
        self.assertLessEqual(max_error, 0.5)
        self.assertGreaterEqual(mega_metrics["score"], baseline_metrics["score"] - 0.02)


if __name__ == "__main__":
    unittest.main()
