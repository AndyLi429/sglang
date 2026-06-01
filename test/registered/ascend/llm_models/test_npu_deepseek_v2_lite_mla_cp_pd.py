"""GSM8K accuracy test for DeepSeek-V2-Lite MLA Context Parallel on Ascend NPU.

This test validates the **MLA prefill context-parallel (CP)** code path on NPU
under a **PD-disaggregated** deployment, which is the only configuration that
produces correct results for MLA CP today:

  - Prefill node: TP=4, MOE_DP=2, ATTN_CP=2, ``--enable-prefill-context-parallel``.
    On Ascend, CP is driven by ``--attn-cp-size`` + ``--moe-dp-size`` and the
    total NPU count == tp_size (no replication). ``--moe-dp-size > 1`` turns on
    the MoE all2all backend so the MLP-sync / DP-gather buffer
    (``set_dp_buffer_len`` → ``_dp_max_padding``) is initialized for the CP
    communicator. ``max_running_requests == 1`` because batch_size==1 is a hard
    CP restriction. Each CP rank stores only its slice of the context KV and
    rebuilds the full KV (via ``cp_all_gather_rerange_output``) before handing
    it to decode.
    NOTE: do NOT pass ``--dp-size`` here — on Ascend that triggers classic
    data-parallel replication (dp_size * tp_size NPUs) and does not enable CP.
  - Decode node: TP=2, no CP (CP is a prefill-only optimization).
  - A mini load balancer (sglang_router) fronts both sides.

KV is transferred over the Ascend backend (``--disaggregation-transfer-backend
ascend`` + ``ASCEND_MF_STORE_URL``), so all six NPUs live on a single node:
prefill on NPU 0-3, decode on NPU 4-5.

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

# 6 NPUs: prefill TP=4 (DP=2, ATTN_CP=2) + decode TP=2. MLA CP is experimental.
register_npu_ci(est_time=900, suite="nightly-8-npu-a3", nightly=True)

MODEL_PATH = "/home/weights/DeepSeek-V2-Lite/"

# Parallelism / placement (Ascend CP recipe; mirrors test_npu_qwen3_30b_attn_cp).
# On Ascend, prefill CP is driven by --attn-cp-size + --moe-dp-size, and total
# GPUs == tp_size (NO replication). Do NOT use --dp-size: that triggers classic
# data-parallel replication (dp_size * tp_size GPUs) and does NOT enable CP.
# --moe-dp-size>1 turns on the MoE all2all backend, which makes
# require_attn_tp_gather True so the MLP sync runs and the DP-gather buffer
# (set_dp_buffer_len → `_dp_max_padding`) is initialized for the CP communicator.
PREFILL_TP = 4
PREFILL_ATTN_CP = 2
PREFILL_MOE_DP = 2
DECODE_TP = 2
PREFILL_BASE_GPU_ID = 0  # NPUs 0..(PREFILL_TP-1)  → 0..3
DECODE_BASE_GPU_ID = PREFILL_TP  # NPUs 4..(4+DECODE_TP-1) → 4..5

# GSM8K accuracy gate (DeepSeek-Coder-V2-Lite-Instruct, 5-shot). This is a
# correctness sanity check for the CP path, not a tight perf bar.
GSM8K_MIN_ACCURACY = 0.85
GSM8K_NUM_QUESTIONS = 200
GSM8K_NUM_SHOTS = 5

# Everything here is local (127.0.0.1): prefill/decode bootstrap registration
# and the GSM8K client must NOT go through any inherited proxy. A SOCKS proxy in
# the shell env without PySocks installed makes `requests` raise
# "Missing dependencies for SOCKS support", so the prefill instance never
# registers to the bootstrap server and the whole run hangs.
_NO_PROXY_ENV = {
    "no_proxy": "*",
    "NO_PROXY": "*",
    "all_proxy": "",
    "ALL_PROXY": "",
    "http_proxy": "",
    "HTTP_PROXY": "",
    "https_proxy": "",
    "HTTPS_PROXY": "",
}

# Canonical Ascend env for MLA CP + Ascend KV transfer (see gsm8k_ascend_mixin
# and the Qwen3-235B PCP deployment reference).
_NPU_ENV_VARS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "200",
    "HCCL_EXEC_TIMEOUT": "200",
    **_NO_PROXY_ENV,
}


class TestDeepSeekV2LiteMLACPPD(CustomTestCase):
    """PD-disaggregated MLA context-parallel GSM8K accuracy test on NPU."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        # Bypass any inherited (SOCKS) proxy for the in-process GSM8K client too;
        # it talks to the LB on 127.0.0.1.
        os.environ.update(_NO_PROXY_ENV)
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
            "--moe-dp-size",
            str(PREFILL_MOE_DP),
            "--attn-cp-size",
            str(PREFILL_ATTN_CP),
            "--enable-prefill-context-parallel",
            # CP hard limit: prefill batch_size must be 1.
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
            f"(PD-disagg MLA CP: prefill TP={PREFILL_TP} MOE_DP={PREFILL_MOE_DP} "
            f"ATTN_CP={PREFILL_ATTN_CP}, decode TP={DECODE_TP}, "
            f"{GSM8K_NUM_QUESTIONS} samples): {metrics['accuracy']:.3f}"
        )
        self.assertGreaterEqual(metrics["accuracy"], GSM8K_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
