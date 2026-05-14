"""DeepSeek-V4-Flash NPU attn_cp (NSA prefill context-parallel) fully-automated test.

This test launches the server itself, runs a sanity check and GSM8K evaluation,
then shuts the server down. No manual server setup is required.

Hardware requirement: 8 × Ascend 910C (A3) NPUs visible via ASCEND_RT_VISIBLE_DEVICES.

Usage:
    python3 -m unittest test_dsv4_npu_attn_cp -v
    python3 -m unittest test_dsv4_npu_attn_cp.TestDSV4NPUAttnCP.test_a_sanity

Override via env vars:
    SGLANG_TEST_MODEL_PATH   -- default /home/weights/DeepSeek-V4-Flash-W8A8/
    SGLANG_TEST_SERVER_URL   -- default http://127.0.0.1:30201
    SGLANG_TEST_GSM8K_N      -- number of GSM8K questions (default 1319)
    SGLANG_TEST_ACCURACY_TH  -- GSM8K accuracy threshold  (default 0.80)
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# ---------------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ---------------------------------------------------------------------------

_MODEL_PATH = os.environ.get(
    "SGLANG_TEST_MODEL_PATH", "/home/weights/DeepSeek-V4-Flash-W8A8/"
)
_SERVER_URL = os.environ.get("SGLANG_TEST_SERVER_URL", "http://127.0.0.1:30201")
_GSM8K_N = int(os.environ.get("SGLANG_TEST_GSM8K_N", "1319"))
_ACCURACY_TH = float(os.environ.get("SGLANG_TEST_ACCURACY_TH", "0.80"))

# Server startup can take >30 min for this large model.
_SERVER_TIMEOUT = int(os.environ.get("SGLANG_TEST_SERVER_TIMEOUT", str(3600)))

_SANITY_PROMPT = "Where is the capital of France?"
_SANITY_KEYWORD = "paris"

# ---------------------------------------------------------------------------
# Environment variables required by the server process
# Mirrors the companion shell script; source scripts (set_env.sh, etc.) are
# expected to have been run in the parent shell so PATH/LD_LIBRARY_PATH are
# already populated. The vars below are additive on top of os.environ.
# ---------------------------------------------------------------------------

_SERVER_ENV = {
    # NPU device selection — match ASCEND_RT_VISIBLE_DEVICES in shell script
    "ASCEND_RT_VISIBLE_DEVICES": os.environ.get(
        "ASCEND_RT_VISIBLE_DEVICES", "2,3,4,5,6,7,8,9"
    ),
    # Memory / allocator
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    # Network
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "2000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_DETERMINISTIC": "true",
    # DeepEP / dispatch
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    # DSV4 feature flags
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "USE_FUSED_HC_POST_ASCENDC": "1",
    "ASCEND_USE_FIA": "1",
    "USE_PA_DECODE": "1",
    "USE_PA_PREFILL": "1",
    "USE_FUSED_TRANSPOSE_BATCHMATMUL": "0",
    "USE_FUSED_COMPRESSOR": "1",
    "LI_KV_DTYPE_INT8": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    # DSV4 NPU attention flags (sparse + compressor paths)
    "SGLANG_DSV4_NPU_SPARSE_C4_NO_TOPK": "1",
    "SGLANG_DSV4_NPU_SPARSE_ATTN": "1",
    "SGLANG_DSV4_NPU_REAL_COMPRESSOR": "1",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    # Performance / optimisation
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "1",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "USE_ROPE_PARTIAL_IN_PLACE_ASCENDC": "True",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "False",
    # Custom op library (supplement LD_LIBRARY_PATH if not already set)
    "LD_LIBRARY_PATH": ":".join(
        filter(
            None,
            [
                "/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib/",
                os.environ.get("LD_LIBRARY_PATH", ""),
            ],
        )
    ),
}

# ---------------------------------------------------------------------------
# Server launch args (mirror the shell script exactly)
# ---------------------------------------------------------------------------

_SERVER_ARGS = [
    "--page-size", "128",
    "--tp-size", "8",
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode",
    "--round-robin-split",
    "--attn-cp-size", "2",
    "--trust-remote-code",
    "--attention-backend", "ascend",
    "--device", "npu",
    "--watchdog-timeout", "9000",
    "--host", "0.0.0.0",
    "--port", "30201",
    "--dist-init-addr", "127.0.0.1:31000",
    "--nccl-port", "41000",
    "--mem-fraction-static", "0.8",
    "--disable-radix-cache",
    "--chunked-prefill-size", "-1",
    "--max-prefill-tokens", "65535",
    "--context-length", "65535",
    "--max-running-requests", "32",
    "--disable-overlap-schedule",
    "--dp-size", "8",
    "--enable-dp-attention",
    "--moe-a2a-backend", "deepep",
    "--deepep-mode", "auto",
    "--quantization", "compressed-tensors",
    "--enable-dp-lm-head",
    "--kv-cache-dtype", "auto",
    "--disable-cuda-graph",
    "--skip-server-warmup",
    "--random-seed", "42",
]  # fmt: skip


class TestDSV4NPUAttnCP(CustomTestCase, GSM8KMixin):
    """NSA prefill context-parallel accuracy gate for DeepSeek-V4-Flash on Ascend NPU.

    Launches the server in setUpClass, runs sanity + GSM8K, kills in tearDownClass.
    """

    # -- GSM8KMixin required attrs ------------------------------------------
    base_url: str = _SERVER_URL
    model: str = _MODEL_PATH
    gsm8k_accuracy_thres: float = _ACCURACY_TH
    gsm8k_num_questions: int = _GSM8K_N
    gsm8k_num_threads: int = 128

    _sanity_passed: bool = False

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            model=_MODEL_PATH,
            base_url=_SERVER_URL,
            timeout=_SERVER_TIMEOUT,
            other_args=_SERVER_ARGS,
            env=_SERVER_ENV,
            device="npu",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # -----------------------------------------------------------------------
    # Stage A — sanity: does the model know the capital of France?
    # -----------------------------------------------------------------------

    def test_a_sanity(self):
        """Single request: response must contain 'Paris' (case-insensitive)."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": _SANITY_PROMPT,
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=120,
        )
        response.raise_for_status()
        text = response.json().get("text", "")
        print(f"\n[sanity] prompt={_SANITY_PROMPT!r}")
        print(f"[sanity] response={text!r}")

        self.assertIn(
            _SANITY_KEYWORD,
            text.lower(),
            f"Expected 'Paris' in response but got: {text!r}",
        )
        TestDSV4NPUAttnCP._sanity_passed = True

    # -----------------------------------------------------------------------
    # Stage B — GSM8K (only runs if sanity passed)
    # -----------------------------------------------------------------------

    def test_b_gsm8k(self):
        """GSM8K {_GSM8K_N}-question eval — skipped if sanity failed."""
        if not TestDSV4NPUAttnCP._sanity_passed:
            self.skipTest(
                "Skipping GSM8K: sanity check did not find 'Paris' in the response. "
                "Fix attn_cp correctness before running the accuracy gate."
            )
        super().test_gsm8k()


if __name__ == "__main__":
    unittest.main()
