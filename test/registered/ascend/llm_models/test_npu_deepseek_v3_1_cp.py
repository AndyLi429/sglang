import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_1_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="nightly-16-npu-a3", nightly=True)


class TestDeepSeekV31PrefillCP(GSM8KAscendMixin, CustomTestCase):
    """GSM8K smoke test for DeepSeek-V3.1 with MLA prefill context parallelism.

    [Test Category] Model / parallelism
    [Test Target] DeepSeek-V3.1 (MLA) on Ascend with context parallel prefill

    Mirrors test_npu_qwen3_30b_attn_cp.py (the mixed/co-located attention-CP
    deployment), scaled up for DeepSeek-V3.1 (W8A8) on 16 NPUs.

    Exercises the MLA-CP path: each rank zigzag-splits the query, all-gathers
    the latent KV across the attention CP group, and runs npu_ring_mla
    (mask + nomask). A broken CP split or metadata mapping collapses accuracy,
    so the GSM8K score is the end-to-end correctness gate.

    Deployment (16 NPUs, co-located / mixed):
      - TP = 16
      - MOE_DP = 2
      - ATTN_CP = 2
      - prefill context parallel enabled
    """

    model = DEEPSEEK_V3_1_W8A8_WEIGHTS_PATH

    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--max-running-requests",
        "32",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "16",
        "--moe-dp-size",
        "2",
        "--attn-cp-size",
        "2",
        "--cuda-graph-max-bs",
        "32",
        "--enable-prefill-context-parallel",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
    ]

    # Reuse the mixin's DeepEP/HCCL tuning for the 16-NPU MoE run and add FIA
    # (the Qwen CP reference replaces the whole env; we merge so the large-scale
    # DeepEP/HCCL vars are kept).
    env = {**GSM8KAscendMixin.env, "ASCEND_USE_FIA": "1"}

    # GSM8K configs. DeepSeek-V3.1 is stronger than the Qwen3-30B reference
    # (whose bar is 0.92); a healthy MLA-CP path stays well above this, while a
    # CP correctness regression collapses the score.
    accuracy = 0.92
    num_questions = 200
    gsm8k_num_shots = 5
    gsm8k_parallel = 32


if __name__ == "__main__":
    unittest.main()
