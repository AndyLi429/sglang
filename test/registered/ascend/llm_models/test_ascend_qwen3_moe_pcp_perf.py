"""Performance comparison: PCP=1 (baseline) vs PCP=2 with 32K random input.

PD-disaggregated deployment on 8 NPUs:
  Prefill server: TP=4 — NPUs 0–3
    baseline: --attn-cp-size=1 (or omitted)
    PCP=2:    --attn-cp-size=2
  Decode  server: TP=4 — NPUs 4–7 (--base-gpu-id=4, unchanged across both groups)

For each configuration:
  1. Start prefill + decode servers + load balancer.
  2. Send NUM_PERF_REQUESTS requests with a ~32K-token random input.
  3. Measure TTFT (Time to First Token) via SSE streaming.
  4. Print avg / p50 / p90 / p99 TTFT.

Finally a side-by-side comparison table is printed.
"""

import json
import os
import random
import statistics
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_with_error_check,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=1800, suite="nightly-8-npu-a3", nightly=True)

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

MODELPATH = "LOCAL_PATH"  # modify to actual model path when running the test

PREFILL_TP = 4
DECODE_TP = 4
DECODE_BASE_GPU_ID = PREFILL_TP  # NPUs 4–7

# Number of requests per configuration during the performance benchmark.
NUM_PERF_REQUESTS = 5

# Approximate target input length in tokens.  We generate random English words;
# Qwen3 tokenizer averages roughly 1.2–1.5 tokens/word for English, so
# 32 768 tokens ≈ 22 000–27 000 words.  We use 26 000 words as a safe target.
TARGET_TOKENS = 32_768
APPROX_WORDS_PER_TOKEN = 0.8  # conservative: words / token  →  more words = more tokens
TARGET_WORD_COUNT = int(TARGET_TOKENS * APPROX_WORDS_PER_TOKEN)

# Max new tokens generated per request (only a few needed for TTFT measurement).
MAX_NEW_TOKENS = 16

# Warm-up requests before measurement (not counted in results).
NUM_WARMUP_REQUESTS = 1

# Ascend-specific environment variables.
ASCEND_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "200",
    "HCCL_EXEC_TIMEOUT": "200",
    "STREAMS_PER_DEVICE": "32",
    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
    "SGLANG_ENBLE_TORCH_COMILE": "1",
    "AUTO_USE_UC_MEMORY": "0",
    "P2P_HCCL_BUFFSIZE": "20",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
}

# Common server args shared by both prefill configurations and the decode server.
ASCEND_COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    "0.8",
    "--disaggregation-transfer-backend",
    "ascend",
]

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

# Simple vocabulary for random prompt generation (common English words).
_VOCAB = (
    "the quick brown fox jumps over a lazy dog and cat sat on mat "
    "hello world model inference performance benchmark token stream "
    "large language neural network deep learning transformer attention "
    "context window prefill decode pipeline parallel tensor shard rank "
    "memory cache prefix sequence batch size latency throughput request "
    "generate output result input prompt response compute gradient weight "
).split()


def make_random_prompt(target_word_count: int, seed: int = 42) -> str:
    """Return a random English-word string of approximately *target_word_count* words."""
    rng = random.Random(seed)
    words = [rng.choice(_VOCAB) for _ in range(target_word_count)]
    return " ".join(words)


def measure_ttft(base_url: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> float:
    """Send a streaming generation request and return Time-To-First-Token (seconds).

    Uses the /generate endpoint with ``stream=True``.  The timer starts just
    before the HTTP request is issued and stops when the first non-empty SSE
    data line is received.
    """
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        "stream": True,
    }
    t0 = time.perf_counter()
    with requests.post(
        f"{base_url}/generate",
        json=payload,
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            # SSE lines look like  "data: {...}"  or  "data: [DONE]"
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                # Got a real token — record first-token time.
                return time.perf_counter() - t0
    # Fallback: return total elapsed if no SSE token line was found.
    return time.perf_counter() - t0


def summarise(values: list[float], label: str) -> dict:
    """Compute and print summary statistics; return a dict with the metrics."""
    values_ms = [v * 1000 for v in values]
    avg = statistics.mean(values_ms)
    p50 = statistics.median(values_ms)
    p90 = sorted(values_ms)[int(len(values_ms) * 0.9)] if len(values_ms) >= 10 else max(values_ms)
    p99 = sorted(values_ms)[int(len(values_ms) * 0.99)] if len(values_ms) >= 100 else max(values_ms)
    print(
        f"[{label}]  n={len(values_ms)}  "
        f"avg={avg:.0f}ms  p50={p50:.0f}ms  p90={p90:.0f}ms  p99={p99:.0f}ms"
    )
    return {"avg_ms": avg, "p50_ms": p50, "p90_ms": p90, "p99_ms": p99, "n": len(values_ms)}


# --------------------------------------------------------------------------- #
# Test class                                                                   #
# --------------------------------------------------------------------------- #


class TestAscendQwen3MoePCPPerf(CustomTestCase):
    """Compare TTFT of PCP=1 (baseline) vs PCP=2 for ~32K-token random input.

    Both groups use PD-disaggregated deployment on 8 NPUs.  The decode server
    is identical in both cases; only the prefill server's ``--attn-cp-size``
    flag differs.
    """

    # Populated by the individual run helpers.
    _results: dict = {}

    @classmethod
    def _apply_ascend_envs(cls):
        os.environ.update(ASCEND_ENVS)

    @classmethod
    def setUpClass(cls):
        cls._apply_ascend_envs()
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        cls._base_port = parsed.port
        cls._results = {}
        cls._prompt = make_random_prompt(TARGET_WORD_COUNT)
        print(
            f"[setup] random prompt generated: "
            f"~{len(cls._prompt.split())} words, {len(cls._prompt)} chars"
        )

    @classmethod
    def tearDownClass(cls):
        # Servers are torn down inside each run; nothing to do here.
        cls._print_comparison()

    # ------------------------------------------------------------------ #
    # Server lifecycle helpers                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def _port_urls(cls, port_offset: int):
        """Return (prefill_url, decode_url, lb_url) using *port_offset* to avoid conflicts."""
        prefill_port = cls._base_port + port_offset
        decode_port = cls._base_port + port_offset + 100
        lb_port = cls._base_port + port_offset + 200
        host = cls.base_host
        return (
            f"http://{host}:{prefill_port}",
            f"http://{host}:{decode_port}",
            f"http://{host}:{lb_port}",
            lb_port,
        )

    @classmethod
    def _start_servers(cls, prefill_cp: int, port_offset: int, env: dict):
        """Start prefill + decode + LB with the given CP size; return (proc_p, proc_d, proc_lb, lb_url)."""
        prefill_url, decode_url, lb_url, lb_port = cls._port_urls(port_offset)

        # ---- Prefill -------------------------------------------------- #
        prefill_args = ASCEND_COMMON_ARGS + [
            "--disaggregation-mode", "prefill",
            "--tp", str(PREFILL_TP),
            "--disable-radix-cache",
            "--chunked-prefill-size", "-1",
        ]
        if prefill_cp > 1:
            prefill_args += ["--attn-cp-size", str(prefill_cp)]

        proc_prefill = popen_launch_pd_server(
            MODELPATH,
            prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

        # ---- Decode --------------------------------------------------- #
        decode_args = ASCEND_COMMON_ARGS + [
            "--disaggregation-mode", "decode",
            "--tp", str(DECODE_TP),
            "--base-gpu-id", str(DECODE_BASE_GPU_ID),
        ]
        proc_decode = popen_launch_pd_server(
            MODELPATH,
            decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

        # Wait for both to be healthy.
        wait_for_http_ready(
            url=prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=proc_prefill,
        )
        wait_for_http_ready(
            url=decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=proc_decode,
        )

        # ---- Load balancer -------------------------------------------- #
        lb_cmd = [
            "python3", "-m", "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill", prefill_url,
            "--decode", decode_url,
            "--host", cls.base_host,
            "--port", str(lb_port),
            # 32K prefill can be slow; give the router enough headroom.
            "--request-timeout-secs", "1200",
        ]
        proc_lb = popen_with_error_check(lb_cmd)
        wait_for_http_ready(
            url=lb_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=proc_lb,
        )

        return proc_prefill, proc_decode, proc_lb, lb_url

    @classmethod
    def _stop_servers(cls, proc_prefill, proc_decode, proc_lb):
        for proc in [proc_lb, proc_decode, proc_prefill]:
            if proc:
                try:
                    kill_process_tree(proc.pid)
                except Exception as exc:
                    print(f"[teardown] error killing pid {proc.pid}: {exc}")
        time.sleep(5)

    # ------------------------------------------------------------------ #
    # Benchmark runner                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def _run_benchmark(cls, prefill_cp: int, port_offset: int):
        """Start servers, measure TTFT, stop servers; store results in cls._results."""
        label = f"PCP={prefill_cp}"
        env = os.environ.copy()

        print(f"\n{'='*60}")
        print(f"Starting benchmark group: {label}")
        print(f"{'='*60}")

        proc_prefill, proc_decode, proc_lb, lb_url = cls._start_servers(
            prefill_cp=prefill_cp, port_offset=port_offset, env=env
        )

        try:
            # Warm-up
            print(f"[{label}] warming up ({NUM_WARMUP_REQUESTS} request(s)) ...")
            for i in range(NUM_WARMUP_REQUESTS):
                measure_ttft(lb_url, cls._prompt)
                print(f"[{label}] warm-up {i+1}/{NUM_WARMUP_REQUESTS} done")

            # Measurement
            print(f"[{label}] measuring TTFT over {NUM_PERF_REQUESTS} requests ...")
            ttft_values = []
            for i in range(NUM_PERF_REQUESTS):
                # Use a different seed per request so prompts differ slightly.
                prompt = make_random_prompt(TARGET_WORD_COUNT, seed=100 + i)
                t = measure_ttft(lb_url, prompt)
                ttft_values.append(t)
                print(f"[{label}] request {i+1}/{NUM_PERF_REQUESTS}: TTFT={t*1000:.0f}ms")

            stats = summarise(ttft_values, label)
            cls._results[label] = stats

        finally:
            cls._stop_servers(proc_prefill, proc_decode, proc_lb)

    # ------------------------------------------------------------------ #
    # Comparison printer                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _print_comparison(cls):
        if not cls._results:
            return
        print("\n" + "=" * 60)
        print("  PERFORMANCE COMPARISON (TTFT, ~32K-token input, PD-disagg)")
        print("=" * 60)
        header = f"{'Config':<12}  {'Avg (ms)':>10}  {'P50 (ms)':>10}  {'P90 (ms)':>10}  {'P99 (ms)':>10}  {'N':>5}"
        print(header)
        print("-" * len(header))
        for label, s in cls._results.items():
            print(
                f"{label:<12}  {s['avg_ms']:>10.0f}  {s['p50_ms']:>10.0f}  "
                f"{s['p90_ms']:>10.0f}  {s['p99_ms']:>10.0f}  {s['n']:>5}"
            )
        print("-" * len(header))

        # If both groups are present, compute speedup.
        baseline_key = "PCP=1"
        pcp2_key = "PCP=2"
        if baseline_key in cls._results and pcp2_key in cls._results:
            baseline_avg = cls._results[baseline_key]["avg_ms"]
            pcp2_avg = cls._results[pcp2_key]["avg_ms"]
            if pcp2_avg > 0:
                speedup = baseline_avg / pcp2_avg
                print(
                    f"\n  PCP=2 avg TTFT is "
                    f"{'faster' if speedup > 1 else 'slower'} than baseline by "
                    f"{abs(speedup - 1) * 100:.1f}%  (speedup={speedup:.2f}x)"
                )
        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Test methods                                                         #
    # ------------------------------------------------------------------ #

    def test_01_baseline_pcp1(self):
        """Group 1: PCP=1 (no context parallel) — 32K random input, measure TTFT."""
        # Use port offset 300 for this group.
        self.__class__._run_benchmark(prefill_cp=1, port_offset=300)
        self.assertIn("PCP=1", self.__class__._results, "PCP=1 benchmark did not produce results")

    def test_02_pcp2(self):
        """Group 2: PCP=2 — 32K random input, measure TTFT."""
        # Use port offset 500 to avoid any leftover sockets from group 1.
        self.__class__._run_benchmark(prefill_cp=2, port_offset=500)
        self.assertIn("PCP=2", self.__class__._results, "PCP=2 benchmark did not produce results")


if __name__ == "__main__":
    unittest.main(verbosity=2)
