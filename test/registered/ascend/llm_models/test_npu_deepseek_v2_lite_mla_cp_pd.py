"""GSM8K accuracy test for DeepSeek-V2-Lite MLA Context Parallel on Ascend NPU.

This test validates the **MLA prefill context-parallel (CP)** code path on NPU
under a **PD-disaggregated** deployment, which is the only configuration that
produces correct results for MLA CP today:

  - Prefill node: TP=2, ATTN_CP=2, ``--enable-prefill-context-parallel``.
    MLA CP forces ``enable_dp_attention``, ``ep_size == tp_size`` and
    ``max_running_requests == 1`` (batch_size==1 is a hard CP restriction).
    Each CP rank stores only its slice of the context KV and rebuilds the full
    KV (via ``cp_all_gather_rerange_output``) before handing it to decode.
  - Decode node: TP=2, no CP (CP is a prefill-only optimization).
  - A mini load balancer (sglang_router) fronts both sides.

KV is transferred over the Ascend backend (``--disaggregation-transfer-backend
ascend`` + ``ASCEND_MF_STORE_URL``), so all four NPUs live on a single node:
prefill on NPU 0-1, decode on NPU 2-3.

Run locally::

    python3 -m unittest \
        test.registered.ascend.llm_models.test_npu_deepseek_v2_lite_mla_cp_pd

Tune the constants below (model path, NPU placement, accuracy threshold) for
your environment. The DeepSeek-Coder-V2-Lite weights share the DeepSeek-V2 MLA
architecture; point ``MODEL_PATH`` at any V2-Lite checkpoint you want to test.
"""

import os
import shlex
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree

# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_with_error_check,
    start_subprocess_fail_fast_watcher,
)
from sglang.utils import wait_for_http_ready

# 4 NPUs (prefill TP=2 + decode TP=2); MLA CP is experimental → nightly only.
register_npu_ci(est_time=900, suite="nightly-4-npu-a3", nightly=True)

MODEL_PATH = "/home/weights/DeepSeek-V2-Lite/"

# Parallelism / placement
PREFILL_TP = 2
PREFILL_ATTN_CP = 2
DECODE_TP = 2
PREFILL_BASE_GPU_ID = 0  # NPUs 0..(PREFILL_TP-1)
DECODE_BASE_GPU_ID = PREFILL_TP  # decode starts after the prefill NPUs

# GSM8K accuracy gate (DeepSeek-Coder-V2-Lite-Instruct, 5-shot). This is a
# correctness sanity check for the CP path, not a tight perf bar.
GSM8K_MIN_ACCURACY = 0.85
GSM8K_NUM_QUESTIONS = 200
GSM8K_NUM_SHOTS = 5

# Canonical Ascend env for MLA CP + Ascend KV transfer (see gsm8k_ascend_mixin
# and the Qwen3-235B PCP deployment reference).
_NPU_ENV_VARS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "200",
    "HCCL_EXEC_TIMEOUT": "200",
}


class TestDeepSeekV2LiteMLACPPD(CustomTestCase):
    """PD-disaggregated MLA context-parallel GSM8K accuracy test on NPU."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.npu_env = {**os.environ, **_NPU_ENV_VARS}

        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        base_port = int(parsed.port)
        cls.lb_port = str(base_port)
        cls.prefill_port = str(base_port + 100)
        cls.decode_port = str(base_port + 200)
        cls.bootstrap_port = str(base_port + 500)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url
        print(
            f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} "
            f"{cls.decode_port=} {cls.bootstrap_port=}"
        )

        cls.process_prefill = None
        cls.process_decode = None
        cls.process_lb = None
        cls._fail_fast_stop = None

        try:
            cls._launch_all()
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def _launch_all(cls):
        cls._start_prefill()
        cls._start_decode()
        wait_for_http_ready(
            cls.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_prefill,
        )
        wait_for_http_ready(
            cls.decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_decode,
        )
        cls._launch_lb()
        cls._fail_fast_stop = start_subprocess_fail_fast_watcher(
            [
                ("prefill", cls.process_prefill),
                ("decode", cls.process_decode),
                ("lb", cls.process_lb),
            ]
        )

    @classmethod
    def _start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--base-gpu-id",
            str(PREFILL_BASE_GPU_ID),
            "--tp-size",
            str(PREFILL_TP),
            "--enable-prefill-context-parallel",
            "--attn-cp-size",
            str(PREFILL_ATTN_CP),
            # MLA CP hard limit: prefill batch_size must be 1.
            "--max-running-requests",
            "1",  # prefill bs==1
            "--mem-fraction-static",
            "0.8",
            "--skip-server-warmup",
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=cls.npu_env,
        )

    @classmethod
    def _start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--base-gpu-id",
            str(DECODE_BASE_GPU_ID),
            "--tp-size",
            str(DECODE_TP),
            "--max-running-requests",
            "32",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--skip-server-warmup",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=cls.npu_env,
        )

    @classmethod
    def _launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        wait_for_http_ready(
            cls.lb_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_lb,
        )

    @classmethod
    def tearDownClass(cls):
        # Stop the watcher first: kill_process_tree makes children exit with a
        # negative signal rc, which would otherwise trip the watcher mid-teardown.
        if getattr(cls, "_fail_fast_stop", None) is not None:
            cls._fail_fast_stop.set()
        for name in ("process_lb", "process_decode", "process_prefill"):
            process = getattr(cls, name, None)
            if process is not None:
                try:
                    kill_process_tree(process.pid, wait_timeout=60)
                except Exception as e:
                    print(f"Error killing {name} ({process.pid}): {e}")
        time.sleep(5)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            num_shots=GSM8K_NUM_SHOTS,
            data_path="/home/lws/aisbench_auto_tools_prefix/GSM8K.jsonl",
            num_questions=GSM8K_NUM_QUESTIONS,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(
            "GSM8K accuracy "
            f"(PD-disagg MLA CP: prefill TP={PREFILL_TP} ATTN_CP={PREFILL_ATTN_CP}, "
            f"decode TP={DECODE_TP}, {GSM8K_NUM_QUESTIONS} samples): "
            f"{metrics['accuracy']:.3f}"
        )
        self.assertGreaterEqual(metrics["accuracy"], GSM8K_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
