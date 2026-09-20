from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


def test_ascend_megamoe_is_distinct_backend(monkeypatch):
    assert MoeA2ABackend("ascend_megamoe").is_ascend_megamoe()
    assert not MoeA2ABackend("megamoe").is_ascend_megamoe()
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1")
    assert envs.SGLANG_NPU_ENABLE_MEGAMOE.get() is True


def test_ascend_megamoe_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_NPU_ENABLE_MEGAMOE", raising=False)
    assert envs.SGLANG_NPU_ENABLE_MEGAMOE.get() is False
