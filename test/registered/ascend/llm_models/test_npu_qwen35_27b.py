import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN35_27B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from types import SimpleNamespace

register_npu_ci(est_time=600, suite="nightly-2-npu-a3", nightly=True)


class TestQwen3527B(CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen3.5-27B model
    on the GSM8K dataset is no less than 0.85.

    [Test Category] Model
    [Test Target] Qwen3.5-27B
    """

    model = QWEN35_27B_WEIGHTS_PATH
    accuracy = 0.85
    gsm8k_num_shots = 5
    other_args = [
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--tp-size", "2",
        "--chunked-prefill-size", "-1",
        "--max-prefill-tokens", "8192",
        "--max-running-requests", "128",
        "--mem-fraction-static", "0.8",
        "--cuda-graph-bs", "2", "4", "6", "8", "10", "16", "20", "24",
        "--enable-multimodal",
        "--mm-attention-backend", "ascend_attn",
        "--max-total-tokens", "100000",
        "--dtype", "bfloat16",
        "--mamba-ssm-dtype", "bfloat16",
        "--disable-radix-cache",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["STREAMS_PER_DEVICE"] = "32"
        os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "32"
        os.environ["HCCL_BUFFSIZE"] = "3000"
        os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
        os.environ["HCCL_SOCKET_IFNAME"] = "lo"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        os.environ["SGLANG_NPU_PROFILING"] = "0"
        os.environ["SGLANG_NPU_PROFILING_STAGE"] = "prefill"
        os.environ["DEEPEP_NORMAL_LONG_SEQ_ROUND"] = "32"
        os.environ["DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"] = "3584"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24669"
        os.environ["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "3600"
        os.environ["HCCL_EXEC_TIMEOUT"] = "200"
        env = os.environ.copy()

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=self.gsm8k_num_shots,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f"Accuracy of {self.model} is {metrics['accuracy']}, lower than threshold {self.accuracy}",
        )


if __name__ == "__main__":
    unittest.main()
