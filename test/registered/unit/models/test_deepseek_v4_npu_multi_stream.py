"""NPU multi-stream scheduling for the DeepSeek-V4 MQA preparation path."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeTensor:
    dtype = torch.bfloat16

    def view(self, *args):
        return self

    def unsqueeze(self, dim):
        return self

    def record_stream(self, stream):
        stream.calls.append((stream.name, "record_q"))


class _FakeStream:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.waited_streams = []

    def wait_stream(self, stream):
        self.waited_streams.append(stream)

    def wait_event(self, event):
        self.calls.append((self.name, "wait_event"))

    def record_event(self):
        self.calls.append((self.name, "record_event"))
        return object()


class _FakeNpu:
    def __init__(self, current):
        self.current = current
        self.active = current

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous


class _Harness(deepseek_v4.MQALayer):
    def __init__(self, npu, calls):
        torch.nn.Module.__init__(self)
        self.alt_streams = [
            _FakeStream("kv", calls),
            _FakeStream("q", calls),
            _FakeStream("compressor", calls),
        ]
        self.fuse_wqa_wkv = False
        self.q_lora_rank = 1
        self.n_local_heads = 1
        self.head_dim = 2
        self.eps = 1e-6
        self.q_rms_norm_ones = torch.ones(2)
        self.qk_nope_head_dim = 1
        self.layer_id = 0
        self.compressor = object()
        self.indexer = lambda **kwargs: calls.append((npu.active.name, "indexer"))
        self.wq_a = lambda x: (_FakeTensor(), None)
        self.q_norm = lambda q: q
        self.wkv = lambda x: (_FakeTensor(), None)
        self.kv_norm = lambda kv: kv
        self.wq_b = lambda q: (_FakeTensor(), None)

    def _get_npu_rope_position_cache(self, *args, **kwargs):
        return _FakeTensor(), _FakeTensor()


def _run_prepare():
    calls = []
    current = _FakeStream("current", calls)
    npu = _FakeNpu(current)
    layer = _Harness(npu, calls)
    backend = SimpleNamespace(
        store_cache=lambda **kwargs: calls.append((npu.active.name, "store_cache")),
        forward_core_compressor=lambda *args: calls.append(
            (npu.active.name, "compressor")
        ),
    )

    with (
        patch.object(deepseek_v4.torch, "npu", npu, create=True),
        patch.object(
            deepseek_v4,
            "torch_npu",
            SimpleNamespace(npu_rms_norm=lambda q, *args: (q,)),
            create=True,
        ),
        patch.object(deepseek_v4.Dsv4NpuRoPE, "apply_rotary_mul_inplace"),
    ):
        layer._forward_prepare_multi_stream_npu(
            _FakeTensor(),
            _FakeTensor(),
            SimpleNamespace(),
            backend,
        )

    return calls, current, layer


def test_npu_multi_stream_runs_compressor_on_its_own_stream():
    """Keeping compressor on current serializes a branch that only reads x."""
    calls, _, _ = _run_prepare()

    assert ("compressor", "compressor") in calls


def test_npu_multi_stream_joins_compressor_before_returning_q():
    """Attention must not consume compressor output before its stream completes."""
    _, current, layer = _run_prepare()

    assert layer.alt_streams[2] in current.waited_streams
