"""Nightly accuracy and performance tests for Qwen3.5-27B-W8A8 on Ascend NPU.

Starts the server in setUpClass with the full NPU launch configuration,
runs GSM8K accuracy, C-Eval accuracy, and throughput benchmark tests,
then kills the server in tearDownClass.
"""

import json
import os
import subprocess
import unittest
from types import SimpleNamespace

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server


def _source_cann_env() -> dict:
    """Source CANN toolkit env scripts and return the resulting environment dict.

    The bash startup script runs:
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/nnal/atb/set_env.sh
    These set LD_LIBRARY_PATH, ASCEND_HOME_PATH, driver paths, etc. that the NPU
    runtime requires.  Without them the NPU kernel raises MTE address-range faults.
    """
    cmd = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        "source /usr/local/Ascend/nnal/atb/set_env.sh && "
        "env"
    )
    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to source CANN env scripts:\n{result.stderr}")
    env = {}
    for line in result.stdout.splitlines():
        key, sep, val = line.partition("=")
        if sep:
            env[key] = val
    return env


register_npu_ci(est_time=14400, suite="nightly-4-npu-a3", nightly=True)

MODEL_PATH = "/home/weights/Qwen3.5-27B-w8a8-mtp"
API_HOST = "127.0.0.1"
API_PORT = 31125
API_URL = f"http://{API_HOST}:{API_PORT}"
SERVER_LAUNCH_TIMEOUT = 3600

GSM8K_DATASET_PATH = "/home/q30063557/workspace/acc-eval/gsm8k/dataset_eval"
CEVAL_DATASET_PATH = "/home/q30063557/workspace/acc-eval/ceval/dataset_eval"
SHAREGPT_PATH = (
    "/home/q30063557/workspace/acc-eval/gsm8k/ShareGPT_V3_unfiltered_cleaned_split.json"
)

GSM8K_SCORE_THRESHOLD = 0.85
CEVAL_SCORE_THRESHOLD = 0.85
PERF_OUTPUT_THROUGHPUT_THRESHOLD = 500.0  # tok/s, adjust after baseline measurement

# Server launch args (host/port are appended by popen_launch_server from base_url)
_SERVER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    "4",
    "--base-gpu-id",
    "12",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--chunked-prefill-size",
    "-1",
    "--max-prefill-tokens",
    "100000",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-total-tokens",
    "800000",
    "--max-running-requests",
    "32",
    "--mem-fraction-static",
    "0.75",
    # Each batch size must be a separate list element; passing a single
    # space-joined string would be treated as one literal value.
    "--cuda-graph-bs",
    "2",
    "4",
    "6",
    "8",
    "10",
    "16",
    "20",
    "24",
    "28",
    "32",
    "48",
    "56",
    "64",
    "96",
    "112",
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]

# Environment overrides applied on top of the CANN env captured by _source_cann_env().
_SERVER_ENV_OVERRIDES = {
    # Clear proxy vars so loopback traffic isn't routed through an external proxy.
    "https_proxy": "",
    "http_proxy": "",
    "HTTPS_PROXY": "",
    "HTTP_PROXY": "",
    # NPU / HCCL tuning
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "PYTHONPATH": "/home/l00567497/sglang/python:" + os.environ.get("PYTHONPATH", ""),
}


def _run_evalscope(cmd: list, timeout: int = 7200) -> int:
    """Run evalscope, streaming its output directly to the terminal.

    Returns the process exit code.  Accuracy is parsed from the JSON files
    evalscope writes under outputs/ rather than from captured stdout.
    """
    print(f"\n[evalscope] running: {' '.join(cmd)}", flush=True)
    # Don't capture — let evalscope print progress in real time.
    result = subprocess.run(cmd, timeout=timeout)
    return result.returncode


def _parse_evalscope_accuracy(dataset: str) -> float:
    """Parse accuracy from the JSON report evalscope writes under outputs/."""
    # evalscope writes: outputs/<timestamp>/<model>/<dataset>/report.json  (or similar)
    for root, _dirs, files in os.walk("outputs"):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, fname)) as f:
                    data = json.load(f)
                # Top-level dict keyed by dataset name
                for key, val in data.items():
                    if dataset.lower() in key.lower() and isinstance(val, dict):
                        for metric in ("accuracy", "acc", "score"):
                            if metric in val:
                                return float(val[metric])
                # Flat structure: {"accuracy": 0.91, ...}
                if dataset.lower() in root.lower():
                    for metric in ("accuracy", "acc", "score"):
                        if metric in data:
                            return float(data[metric])
            except Exception:
                continue

    raise AssertionError(
        f"Could not find {dataset} accuracy in any JSON file under outputs/. "
        "Check the evalscope output above for the actual results path."
    )


class TestQwen35_27B_W8A8(CustomTestCase):
    """GSM8K accuracy, C-Eval accuracy, and throughput tests for Qwen3.5-27B-W8A8."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = API_URL
        print(f"\n[setup] sourcing CANN env scripts ...", flush=True)
        env = _source_cann_env()
        env.update(_SERVER_ENV_OVERRIDES)
        print(
            f"[setup] launching server at {API_URL} (timeout={SERVER_LAUNCH_TIMEOUT}s) ...",
            flush=True,
        )
        cls.process = popen_launch_server(
            model=MODEL_PATH,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=_SERVER_ARGS,
            env=env,
            device="npu",  # prevents auto-detection from appending a second --device flag
        )
        print("[setup] server is healthy, starting tests ...", flush=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        print("\n[test] starting GSM8K accuracy eval ...", flush=True)
        generation_config = json.dumps(
            {
                "do_sample": True,
                "max_tokens": 1024,
                "seed": 3407,
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0,
                "n": 1,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "timeout": 3600,
                "stream": True,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        )
        cmd = [
            "evalscope",
            "eval",
            "--model",
            MODEL_PATH,
            "--api-url",
            f"{API_URL}/v1",
            "--api-key",
            "EMPTY",
            "--eval-type",
            "openai_api",
            "--generation-config",
            generation_config,
            "--datasets",
            "gsm8k",
            "--dataset-hub",
            "Local",
            "--dataset-args",
            json.dumps({"gsm8k": {"local_path": GSM8K_DATASET_PATH}}),
            "--eval-batch-size",
            "32",
            "--ignore-errors",
            "--limit",
            "100",
        ]
        rc = _run_evalscope(cmd, timeout=7200)
        self.assertEqual(rc, 0, f"evalscope gsm8k exited with code {rc}")

        score = _parse_evalscope_accuracy("gsm8k")
        print(
            f"\nGSM8K score: {score:.4f} (threshold: {GSM8K_SCORE_THRESHOLD})",
            flush=True,
        )
        self.assertGreaterEqual(
            score,
            GSM8K_SCORE_THRESHOLD,
            f"GSM8K accuracy {score:.4f} is below threshold {GSM8K_SCORE_THRESHOLD}",
        )

    def test_ceval_accuracy(self):
        print("\n[test] starting C-Eval accuracy eval ...", flush=True)
        generation_config = json.dumps(
            {
                "max_tokens": 20000,
                "seed": 3407,
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0,
                "n": 1,
                "timeout": 60,
                "stream": True,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        )
        cmd = [
            "evalscope",
            "eval",
            "--model",
            MODEL_PATH,
            "--api-url",
            f"{API_URL}/v1",
            "--api-key",
            "EMPTY",
            "--eval-type",
            "openai_api",
            "--generation-config",
            generation_config,
            "--datasets",
            "ceval",
            "--dataset-hub",
            "Local",
            "--dataset-args",
            json.dumps({"ceval": {"local_path": CEVAL_DATASET_PATH}}),
            "--eval-batch-size",
            "256",
            "--ignore-errors",
        ]
        rc = _run_evalscope(cmd, timeout=7200)
        self.assertEqual(rc, 0, f"evalscope ceval exited with code {rc}")

        score = _parse_evalscope_accuracy("ceval")
        print(
            f"\nC-Eval score: {score:.4f} (threshold: {CEVAL_SCORE_THRESHOLD})",
            flush=True,
        )
        self.assertGreaterEqual(
            score,
            CEVAL_SCORE_THRESHOLD,
            f"C-Eval accuracy {score:.4f} is below threshold {CEVAL_SCORE_THRESHOLD}",
        )

    def test_throughput_3500in_1500out(self):
        print(
            "\n[test] starting throughput benchmark (3500in/1500out, 352 concurrent) ...",
            flush=True,
        )
        bench_args = SimpleNamespace(
            backend="sglang",
            base_url=API_URL,
            host=None,
            port=None,
            dataset_name="random",
            dataset_path=SHAREGPT_PATH,
            model=None,
            tokenizer=None,
            num_prompts=352,
            sharegpt_output_len=None,
            sharegpt_context_len=None,
            random_input_len=3500,
            random_output_len=1500,
            random_range_ratio=1.0,
            request_rate=float("inf"),
            multi=None,
            output_file=None,
            disable_tqdm=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            seed=0,
            disable_ignore_eos=False,
            extra_request_body=None,
            apply_chat_template=False,
            profile=None,
            lora_name=None,
            lora_request_distribution="uniform",
            lora_zipf_alpha=1.5,
            prompt_suffix="",
            device="auto",
            pd_separated=False,
            gsp_num_groups=4,
            gsp_prompts_per_group=4,
            gsp_system_prompt_len=128,
            gsp_question_len=32,
            gsp_output_len=32,
            gsp_num_turns=1,
            header=None,
            max_concurrency=352,
            ready_check_timeout_sec=60,
        )
        result = run_benchmark(bench_args)
        output_throughput = result.get("output_throughput", 0.0)
        print(
            f"Output throughput: {output_throughput:.2f} tok/s "
            f"(threshold: {PERF_OUTPUT_THROUGHPUT_THRESHOLD})"
        )
        self.assertGreater(
            output_throughput,
            PERF_OUTPUT_THROUGHPUT_THRESHOLD,
            f"Output throughput {output_throughput:.2f} tok/s is below "
            f"threshold {PERF_OUTPUT_THROUGHPUT_THRESHOLD}",
        )


if __name__ == "__main__":
    # buffer=False: show print() output in real time instead of buffering until
    # each test completes (default unittest behavior hides long-running progress).
    unittest.main(verbosity=2, buffer=False)
