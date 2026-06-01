"""GSM8K accuracy test for DeepSeek-V2-Lite MLA Context Parallel on Ascend NPU.

Goal: verify that turning on MLA **prefill context parallel (CP)** does not
change model quality on NPU — i.e. GSM8K accuracy with CP enabled stays at the
CUDA / non-CP baseline. This is a co-located, single-server test (the same shape
as ``test/registered/cp/test_deepseek_v3_cp_single_node.py``); CP correctness is
validated without PD disaggregation, which only adds KV-transfer/bootstrap
failure surface and does not affect the accuracy check.

CP geometry (minimal): TP=4, DP=2 → ATTN_CP = tp_size // dp_size = 2, on 4 NPUs.

IMPORTANT — V2 arch caveat:
SGLang's MLA-CP auto-config (server_args.py:1936) only runs for
DeepseekV3ForCausalLM / V3.2 / Kimi-K2.5 / GLM-MoE-DSA. DeepSeek-V2-Lite is
DeepseekV2ForCausalLM and is NOT in that list, so this test replicates the
auto-config knobs by hand (--enable-dp-attention, --attn-cp-size,
--moe-a2a-backend deepep, --ep-size == tp_size, --moe-dense-tp-size 1,
--disable-piecewise-cuda-graph). The modeling code in deepseek_v2.py is shared
by V2/V3, so this runs, but MLA CP on the V2 arch is not an officially supported
combination — treat the numbers accordingly.

Run locally::

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    python3 -m unittest \
        test.registered.ascend.llm_models.test_npu_deepseek_v2_lite_mla_cp_pd

Requires DeepEP on Ascend (``--moe-a2a-backend deepep``).
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=900, suite="nightly-4-npu-a3", nightly=True)

MODEL_PATH = "/home/weights/DeepSeek-V2-Lite/"

# Minimal CP geometry: ATTN_CP = TP // DP. TP=4, DP=2 → ATTN_CP=2 on 4 NPUs.
TP_SIZE = 4
DP_SIZE = 2

# GSM8K accuracy gate. Set this to the CUDA / non-CP baseline so the test fails
# only if CP changes model quality.
GSM8K_MIN_ACCURACY = 0.60
GSM8K_NUM_QUESTIONS = 200
GSM8K_NUM_SHOTS = 5

# All client traffic is local (127.0.0.1); never route through an inherited
# (SOCKS) proxy, which without PySocks makes requests raise
# "Missing dependencies for SOCKS support".
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

_NPU_ENV_VARS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "200",
    **_NO_PROXY_ENV,
}


class TestDeepSeekV2LiteMLACP(CustomTestCase):
    """Co-located MLA context-parallel GSM8K accuracy test on NPU."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ.update(_NO_PROXY_ENV)
        cls.npu_env = {**os.environ, **_NPU_ENV_VARS}
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                # --- MLA CP geometry (TP=4, DP=2 → ATTN_CP=2) ---
                "--tp-size",
                str(TP_SIZE),
                "--dp-size",
                str(DP_SIZE),
                "--enable-dp-attention",
                "--attn-cp-size",
                str(TP_SIZE // DP_SIZE),
                "--enable-prefill-context-parallel",
                # MLA-CP auto-config knobs, applied by hand because the V2 arch
                # is not covered by server_args.py:1936 (see module docstring).
                "--moe-a2a-backend",
                "deepep",
                "--ep-size",
                str(TP_SIZE),
                "--moe-dense-tp-size",
                "1",
                "--disable-piecewise-cuda-graph",
                # max_running_requests is split across dp workers, so keep it
                # comfortably >= dp_size (1 would floor to 0 per worker).
                "--max-running-requests",
                "32",
                "--mem-fraction-static",
                "0.8",
            ],
            env=cls.npu_env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            num_shots=GSM8K_NUM_SHOTS,
            data_path="/home/lws/aisbench_auto_tools_prefix/GSM8K.jsonl",
            num_questions=GSM8K_NUM_QUESTIONS,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(
            "GSM8K accuracy "
            f"(MLA CP: TP={TP_SIZE} DP={DP_SIZE} ATTN_CP={TP_SIZE // DP_SIZE}, "
            f"{GSM8K_NUM_QUESTIONS} samples): {metrics['accuracy']:.3f}"
        )
        self.assertGreaterEqual(metrics["accuracy"], GSM8K_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
