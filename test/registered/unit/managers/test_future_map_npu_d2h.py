from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import overlap_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self):
        self.record_count = 0
        self.wait_count = 0
        self.synchronize_count = 0

    def record(self):
        self.record_count += 1

    def wait(self):
        self.wait_count += 1

    def synchronize(self):
        self.synchronize_count += 1


class _FakeStream:
    def __init__(self):
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class _FakeDeviceModule:
    def __init__(self):
        self.entered_streams = []

    def Event(self):
        return _FakeEvent()

    def Stream(self):
        return _FakeStream()

    @contextmanager
    def stream(self, stream):
        self.entered_streams.append(stream)
        yield


def test_npu_eagle_seq_lens_d2h_starts_at_publish():
    """The NPU copy is issued at publish and consumed without a second copy."""
    device_module = _FakeDeviceModule()
    future_map = object.__new__(overlap_utils.FutureMap)
    future_map.device = torch.device("cpu")
    future_map.spec_algo = SimpleNamespace(is_some=lambda: True)
    future_map.needs_cpu_seq_lens = True
    future_map.needs_confidence_relay = False
    future_map.new_seq_lens_buf = torch.full((4,), -1, dtype=torch.int64)
    future_map.new_seq_lens_cpu_pinned = torch.full((4,), -1, dtype=torch.int64)
    future_map.npu_seq_lens_d2h_stream = _FakeStream()
    future_map.npu_seq_lens_d2h_done = _FakeEvent()
    future_map._npu_seq_lens_publish_generation = 0
    future_map._npu_seq_lens_consumed_generation = 0
    future_map.publish_ready = None
    future_map._publish_fresh = False

    batch = SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=torch.tensor([3, 1])),
        req_pool_indices_cpu=torch.tensor([3, 1]),
        seq_lens=None,
        seq_lens_cpu=None,
        seq_lens_sum=None,
    )
    with (
        patch.object(overlap_utils, "_is_npu", True),
        patch.object(overlap_utils, "_DEBUG_ASSERT", False),
        patch.object(
            overlap_utils.torch,
            "get_device_module",
            return_value=device_module,
        ),
    ):
        future_map.publish(torch.tensor([1, 3]), torch.tensor([11, 13]))
        future_map.resolve_seq_lens_cpu(batch)

    assert batch.seq_lens.tolist() == [13, 11]
    assert batch.seq_lens_cpu.tolist() == [13, 11]
    assert batch.seq_lens_sum == 24
    assert future_map.npu_seq_lens_d2h_done.record_count == 1
    assert future_map.npu_seq_lens_d2h_done.synchronize_count == 1
    assert future_map.npu_seq_lens_d2h_stream.waited_events == [
        future_map.publish_ready
    ]


def test_npu_d2h_is_limited_to_non_frozen_eagle():
    device_module = _FakeDeviceModule()
    req_pool = SimpleNamespace(req_to_token=torch.empty((4, 1)))
    torch_empty = torch.empty

    def empty_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_empty(*args, **kwargs)

    for is_eagle, is_frozen_kv_mtp, expected in (
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ):
        spec_algo = SimpleNamespace(
            is_eagle=lambda is_eagle=is_eagle: is_eagle,
            is_frozen_kv_mtp=lambda is_frozen_kv_mtp=is_frozen_kv_mtp: is_frozen_kv_mtp,
        )
        with (
            patch.object(overlap_utils, "_is_cuda", False),
            patch.object(overlap_utils, "_is_npu", True),
            patch.object(
                overlap_utils.torch, "empty", side_effect=empty_without_pinning
            ),
            patch.object(
                overlap_utils.torch,
                "get_device_module",
                return_value=device_module,
            ),
        ):
            future_map = overlap_utils.FutureMap(
                device=torch.device("cpu"),
                spec_algo=spec_algo,
                req_to_token_pool=req_pool,
            )

        assert (future_map.npu_seq_lens_d2h_stream is not None) is expected
