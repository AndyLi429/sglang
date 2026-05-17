import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Suite/est_time chosen to match other 4-NPU-a3 V4 tests. CP=4 + DP=2 uses
# all 8 NPUs of one Atlas 800T A3 box.
register_npu_ci(est_time=700, suite="stage-b-test-4-npu-a3", nightly=False)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DSV4_NPU_REAL_COMPRESSOR": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}


class TestDeepseekV4CPRoundRobinAscend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--attn-cp-size",
            "4",
            "--enable-nsa-prefill-context-parallel",
            "--nsa-prefill-cp-mode",
            "round-robin-split",
            "--moe-a2a-backend",
            "deepep",
            "--mem-fraction-static",
            "0.7",
            "--cuda-graph-max-bs",
            "32",
            "--max-running-requests",
            "32",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=32,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_a_gsm8k (deepseek-v4-cp-round-robin-ascend)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.935)


if __name__ == "__main__":
    unittest.main()
