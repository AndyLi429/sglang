import json
import os
import tempfile
import textwrap
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_V4_W8A8_MODEL = os.environ.get(
    "SGLANG_TEST_DEEPSEEK_V4_MODEL_PATH",
    "/home/weights/deepseek-ai/DeepSeek-V4-Flash-W8A8",
)
TP_SIZE = int(os.environ.get("SGLANG_TEST_DEEPSEEK_V4_CP_TP_SIZE", "16"))
ATTN_CP_SIZE = int(os.environ.get("SGLANG_TEST_DEEPSEEK_V4_CP_ATTN_CP_SIZE", "2"))
DP_SIZE = int(
    os.environ.get(
        "SGLANG_TEST_DEEPSEEK_V4_CP_DP_SIZE",
        str(max(1, TP_SIZE // ATTN_CP_SIZE)),
    )
)
SERVER_TIMEOUT = int(
    os.environ.get(
        "SGLANG_TEST_DEEPSEEK_V4_CP_SERVER_TIMEOUT",
        str(max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800)),
    )
)


def write_deepseek_v4_npu_compat_sitecustomize():
    compat_dir = tempfile.mkdtemp(prefix="sglang_deepseek_v4_npu_")
    sitecustomize_path = os.path.join(compat_dir, "sitecustomize.py")
    with open(sitecustomize_path, "w") as f:
        f.write(textwrap.dedent("""
                import contextlib

                with contextlib.suppress(Exception):
                    import torch
                    import torch_npu
                    import torchair.ge._ge_graph as _ge_graph

                    if not hasattr(_ge_graph, "torch_dtype_value_to_ge_type"):
                        _ge_graph.torch_dtype_value_to_ge_type = _ge_graph.torch_type_to_ge_type
                    if not hasattr(_ge_graph, "torch_dtype_value_to_ge_proto_type"):
                        _ge_graph.torch_dtype_value_to_ge_proto_type = _ge_graph.torch_type_to_ge_proto_type

                with contextlib.suppress(Exception):
                    from transformers import AutoConfig, PretrainedConfig

                    class DeepseekV4HFConfig(PretrainedConfig):
                        model_type = "deepseek_v4"

                        def __init__(self, **kwargs):
                            super().__init__(**kwargs)
                            if not hasattr(self, "sliding_window") and hasattr(self, "sliding_window_size"):
                                self.sliding_window = self.sliding_window_size
                            if not hasattr(self, "window_size") and hasattr(self, "sliding_window"):
                                self.window_size = self.sliding_window

                    with contextlib.suppress(ValueError):
                        AutoConfig.register("deepseek_v4", DeepseekV4HFConfig)
                """))
    return compat_dir


class TestAscendDeepSeekV4CP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DEEPSEEK_V4_W8A8_MODEL):
            raise unittest.SkipTest(
                f"DeepSeek V4 model path does not exist: {DEEPSEEK_V4_W8A8_MODEL}"
            )
        if ATTN_CP_SIZE <= 1:
            raise unittest.SkipTest("This smoke test requires ATTN_CP_SIZE > 1")
        if DP_SIZE <= 0:
            raise unittest.SkipTest(f"Invalid DP_SIZE: {DP_SIZE}")
        if TP_SIZE % (DP_SIZE * ATTN_CP_SIZE) != 0:
            raise unittest.SkipTest(
                f"Invalid CP topology: tp={TP_SIZE}, dp={DP_SIZE}, "
                f"attn_cp={ATTN_CP_SIZE}"
            )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.compat_dir = write_deepseek_v4_npu_compat_sitecustomize()
        cls.extra_envs = {
            "SGLANG_SET_CPU_AFFINITY": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "STREAMS_PER_DEVICE": "64",
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "TASK_QUEUE_ENABLE": "0",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
            "HCCL_BUFFSIZE": "1600",
            "USE_FUSED_HC_POST_ASCENDC": "1",
            "USE_FUSED_HC_PRE_ASCENDC": "1",
            "USE_PA_DECODE": "1",
            "USE_PA_PREFILL": "1",
            "USE_FUSED_COMPRESSOR": "1",
            "USE_NPU_MOE_GATING_TOP_K": "1",
            "ASCEND_USE_FIA": "1",
            "LI_KV_DTYPE_INT8": "1",
            "USE_ROPE_PARTIAL_IN_PLACE_ASCENDC": "1",
            "INF_NAN_MODE_FORCE_DISABLE": "1",
            "IS_DEEPSEEK_V4": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_APPLY_CONFIG_BACKUP": "none",
        }
        env = os.environ.copy()
        env.update(cls.extra_envs)
        for key in [
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ASCEND_LAUNCH_BLOCKING",
        ]:
            env.pop(key, None)
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in [cls.compat_dir, env.get("PYTHONPATH", "")] if path
        )

        cls.process = popen_launch_server(
            DEEPSEEK_V4_W8A8_MODEL,
            cls.base_url,
            timeout=SERVER_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--enable-nsa-prefill-context-parallel",
                "--nsa-prefill-cp-mode",
                "round-robin-split",
                "--page-size",
                "128",
                "--tp-size",
                str(TP_SIZE),
                "--attention-context-parallel-size",
                str(ATTN_CP_SIZE),
                "--dp-size",
                str(DP_SIZE),
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--max-prefill-tokens",
                "8192",
                "--max-running-requests",
                "64",
                "--dtype",
                "bfloat16",
                "--quantization",
                "compressed-tensors",
                "--disable-shared-experts-fusion",
                "--skip-server-warmup",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--disable-overlap-schedule",
            ],
            env=env,
            device="npu",
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion(self):
        payload = {
            "model": DEEPSEEK_V4_W8A8_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me Where is capital of France",
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            self.assertEqual(response.status, 200)
            output = json.loads(response.read().decode("utf-8"))
        print("Chat completion response:")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        content = output["choices"][0]["message"]["content"]
        print(f"Assistant content: {content}")
        self.assertGreater(len(content), 0)


if __name__ == "__main__":
    unittest.main()
