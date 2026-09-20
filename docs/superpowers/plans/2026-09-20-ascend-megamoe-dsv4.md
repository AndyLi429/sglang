# DeepSeek V4 Ascend MegaMOE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Ascend MegaMOE backend that executes supported DeepSeek V4 routed experts through `cann_ops_transformer.ops.mega_moe` and safely falls back when unavailable.

**Architecture:** A new `ascend_megamoe` A2A backend bypasses the normal dispatcher just as `ascend_fuseep` does, but owns a separate operator wrapper and symmetric EP buffer. W8A8 model loading keeps an additional per-expert MegaMOE payload without mutating the ordinary runner's weights; the wrapper uses this payload only after all runtime capability checks pass.

**Tech Stack:** Python, PyTorch NPU, SGLang MoE runtime, HCCL EP process groups, `cann_ops_transformer.ops`.

**Spec:** `docs/superpowers/specs/2026-09-19-ascend-megamoe-dsv4-design.md`

## Global Constraints

- Start from branch `feat/ascend-megamoe-dsv4`, based on `origin/main` commit `83e29d6c5ae`.
- Do not modify `sgl-kernel-npu` or DeepEP public APIs.
- New public backend name is exactly `ascend_megamoe`; it must not reuse generic GPU `megamoe`.
- MegaMOE is opt-in only when `SGLANG_NPU_ENABLE_MEGAMOE=1`; default is disabled.
- First enabled quantization is W8A8 only. W4A8/MXFP4 must remain unsupported until their real NPU layout is separately validated.
- Use a rank-invariant symmetric-buffer capacity; never allocate a collective buffer from per-rank current token count.
- Non-strict unsupported cases must retain ordinary MoE behavior; `SGLANG_NPU_MEGAMOE_STRICT=1` must raise before launching a mismatched collective.
- Preserve DeepSeek V4 clipped SwiGLU by passing `layer.moe_runner_config.swiglu_limit` as `activation_clamp`.

## Review Focus

- Missing `cann_ops_transformer`: an opt-in request must fall back once, without an import-time failure; covered in Task 2.
- EP ranks with unequal current batches: buffer construction must derive only rank-invariant values; covered in Task 2.
- A token count exceeding the allocated capacity: no MegaMOE launch occurs; fallback or strict failure occurs before collective entry; covered in Task 2.
- Generic CUDA `megamoe` selection: it must not select Ascend code; covered in Task 1.
- DeepSeek V4 `swiglu_limit=10.0`: the wrapper must pass it as `activation_clamp`; covered in Task 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `python/sglang/srt/environ.py` | Define MegaMOE enable, strict, and capacity environment fields. |
| `python/sglang/srt/layers/moe/utils.py` | Define and query the distinct `ASCEND_MEGAMOE` backend. |
| `python/sglang/srt/arg_groups/fields/exec_.py` and validation call sites | Accept the backend as a server argument without changing generic GPU behavior. |
| `python/sglang/srt/hardware_backend/npu/moe/megamoe.py` | Lazily load the ops-transformer API, perform capability checks, cache buffers, and invoke MegaMOE. |
| `python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py` | Preserve W8A8 per-expert MegaMOE payload alongside normal weights. |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | Use no-op dispatcher and direct MegaMOE forward only for the new backend. |
| `python/sglang/srt/models/deepseek_v2.py` | Treat `ascend_megamoe` as an EP backend in DeepSeek V2/V4 inherited logic. |
| `test/registered/unit/npu/test_ascend_megamoe.py` | Mocked host-safe tests for controls, capability checks, buffer arguments, payload and fallback. |

### Task 1: Register the opt-in Ascend backend

**Files:**

- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/layers/moe/utils.py`
- Modify: `python/sglang/srt/arg_groups/fields/exec_.py`
- Modify: existing A2A backend validation sites found by `rg -n "ascend_fuseep" python/sglang/srt/arg_groups python/sglang/srt/models/deepseek_v2.py`
- Test: `test/registered/unit/npu/test_ascend_megamoe.py`

**Interfaces:**

- Produces `MoeA2ABackend.ASCEND_MEGAMOE = "ascend_megamoe"` and `MoeA2ABackend.is_ascend_megamoe() -> bool`.
- Produces `envs.SGLANG_NPU_ENABLE_MEGAMOE: EnvBool`, `envs.SGLANG_NPU_MEGAMOE_STRICT: EnvBool`, and `envs.SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS: EnvInt`.
- Consumed by Tasks 2 and 3.

- [ ] **Step 1: Write failing backend and environment tests**

```python
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeA2ABackend

def test_ascend_megamoe_is_distinct_backend(monkeypatch):
    assert MoeA2ABackend("ascend_megamoe").is_ascend_megamoe()
    assert not MoeA2ABackend("megamoe").is_ascend_megamoe()
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1")
    assert envs.SGLANG_NPU_ENABLE_MEGAMOE.get() is True

def test_ascend_megamoe_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_NPU_ENABLE_MEGAMOE", raising=False)
    assert envs.SGLANG_NPU_ENABLE_MEGAMOE.get() is False
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py -k backend`

Expected: FAIL because `ascend_megamoe` and its environment fields do not exist.

- [ ] **Step 3: Add the minimal backend registration**

```python
class MoeA2ABackend(Enum):
    ASCEND_MEGAMOE = "ascend_megamoe"

    def is_ascend_megamoe(self) -> bool:
        return self == MoeA2ABackend.ASCEND_MEGAMOE
```

Add the three env fields adjacent to the existing MegaMOE/FuseEP fields, add `"ascend_megamoe"` to the `moe_a2a_backend` literal/validation lists, and add the same EP-condition branches as `ascend_fuseep` in `deepseek_v2.py`. Do not add it to CUDA-only backend capability sets.

- [ ] **Step 4: Run focused registration tests**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py -k backend`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/utils.py python/sglang/srt/arg_groups test/registered/unit/npu/test_ascend_megamoe.py python/sglang/srt/models/deepseek_v2.py
git commit -s -m "feat(npu): register Ascend MegaMOE backend"
```

### Task 2: Implement lazy MegaMOE loading, capability checks, and symmetric buffer management

**Files:**

- Create: `python/sglang/srt/hardware_backend/npu/moe/megamoe.py`
- Test: `test/registered/unit/npu/test_ascend_megamoe.py`

**Interfaces:**

- Consumes Task 1 backend and environment fields, a `FusedMoE`-like layer, `get_moe_ep_group().device_group`, and `TopKOutput`.
- Produces `is_megamoe_available(layer: FusedMoE, num_tokens: int) -> tuple[bool, str]`.
- Produces `forward_megamoe(layer: FusedMoE, hidden_states: torch.Tensor, topk_output: TopKOutput) -> torch.Tensor`.
- Produces `cache_megamoe_w8a8_payload(layer: torch.nn.Module) -> None` for Task 3.

- [ ] **Step 1: Write failing wrapper tests with a fake ops module**

```python
def test_buffer_uses_rank_invariant_capacity(monkeypatch, fake_layer, fake_group):
    calls = []
    monkeypatch.setattr(megamoe, "_load_ops", lambda: fake_ops(calls))
    monkeypatch.setattr(megamoe, "_max_tokens_per_rank", lambda _layer: 128)
    megamoe.forward_megamoe(fake_layer, hidden_states(7), topk_output())
    assert calls["buffer"]["num_max_tokens_per_rank"] == 128
    assert calls["buffer"]["max_recv_token_num"] >= 128

def test_unavailable_ops_falls_back_or_raises_in_strict_mode(monkeypatch, fake_layer):
    monkeypatch.setattr(megamoe, "_load_ops", lambda: None)
    assert megamoe.is_megamoe_available(fake_layer, 1) == (False, "cann_ops_transformer is unavailable")
    monkeypatch.setenv("SGLANG_NPU_MEGAMOE_STRICT", "1")
    with pytest.raises(RuntimeError, match="cann_ops_transformer"):
        megamoe.forward_megamoe(fake_layer, hidden_states(1), topk_output())
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py -k "buffer or unavailable"`

Expected: FAIL because `hardware_backend.npu.moe.megamoe` does not exist.

- [ ] **Step 3: Implement the wrapper with no import-time operator dependency**

```python
def _load_ops() -> tuple[Callable, Callable] | None:
    try:
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    except ImportError:
        return None
    return get_symm_buffer_for_mega_moe, mega_moe

def is_megamoe_available(layer, num_tokens: int) -> tuple[bool, str]:
    # Check enable flag, NPU, operator availability, EP > 1, divisibility,
    # W8A8 payload, no LoRA, and capacity before any collective allocation.
    ...
```

Store buffers in `get_resources().buffers` using a key made only from EP identity and rank-invariant shape/configuration values. Derive capacity from the configured maximum or graph/scheduler maximum; reject `num_tokens > capacity` before calling the operator. Clamp receive capacity to the mathematically safe bound `capacity * ep_world_size * min(top_k, experts_per_rank)`.

- [ ] **Step 4: Add capability and buffer tests**

```python
def test_capacity_overflow_does_not_call_operator(monkeypatch, fake_layer):
    monkeypatch.setattr(megamoe, "_max_tokens_per_rank", lambda _layer: 4)
    op = Mock()
    monkeypatch.setattr(megamoe, "_load_ops", lambda: fake_ops({"mega_moe": op}))
    assert megamoe.forward_megamoe_or_none(fake_layer, hidden_states(5), topk_output()) is None
    op.assert_not_called()

def test_generic_megamoe_backend_never_loads_ascend_ops(monkeypatch, fake_layer):
    monkeypatch.setattr(megamoe, "_load_ops", Mock())
    assert not megamoe.is_ascend_megamoe_backend()
```

- [ ] **Step 5: Run wrapper tests**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py -k "buffer or unavailable or capacity or generic"`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add python/sglang/srt/hardware_backend/npu/moe/megamoe.py test/registered/unit/npu/test_ascend_megamoe.py
git commit -s -m "feat(npu): add MegaMOE operator wrapper"
```

### Task 3: Preserve W8A8 payload and invoke MegaMOE from FusedMoE

**Files:**

- Modify: `python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py`
- Modify: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- Test: `test/registered/unit/npu/test_ascend_megamoe.py`

**Interfaces:**

- Consumes `cache_megamoe_w8a8_payload(layer)` and `forward_megamoe(...)` from Task 2.
- Produces `layer._megamoe_w8a8_payload` containing per-expert `w13`, `w2`, `w13_scale`, and `w2_scale` without deleting normal runner weights.
- Produces direct `FusedMoE.forward()` dispatch for `MoeA2ABackend.ASCEND_MEGAMOE`.

- [ ] **Step 1: Write failing payload and forward tests**

```python
def test_w8a8_payload_keeps_normal_weights(fake_w8a8_layer):
    original_w13 = fake_w8a8_layer.w13_weight
    megamoe.cache_megamoe_w8a8_payload(fake_w8a8_layer)
    payload = fake_w8a8_layer._megamoe_w8a8_payload
    assert len(payload.w13) == fake_w8a8_layer.w13_weight.shape[0]
    assert fake_w8a8_layer.w13_weight is original_w13

def test_forward_passes_clipped_swiglu(monkeypatch, fake_layer):
    fake_layer.moe_runner_config.swiglu_limit = 10.0
    captured = {}
    monkeypatch.setattr(megamoe, "_call_mega_moe", lambda **kw: captured.update(kw) or output())
    megamoe.forward_megamoe(fake_layer, hidden_states(2), topk_output())
    assert captured["activation"] == "swiglu"
    assert captured["activation_clamp"] == 10.0
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py -k "payload or clipped"`

Expected: FAIL because the W8A8 payload is not retained and the operator call is not wired.

- [ ] **Step 3: Add the W8A8-only post-load branch**

```python
if get_moe_a2a_backend().is_ascend_megamoe():
    cache_megamoe_w8a8_payload(layer)
    # Continue ordinary post-load processing; the cache is additive.
```

Build the payload before the existing transpose/format conversion destroys MegaMOE's native per-expert layout. Make cloning/format-casting explicit and preserve the normal tensors for fallback. Do not add branches to W4A8, FP4, unquantized, or MXFP methods in this task.

In `FusedMoE`, set `_use_ascend_megamoe`, select `StandardDispatcher` as a never-used placeholder, and call `forward_megamoe_or_none`. If it returns `None`, execute the existing `forward_impl` path; if strict mode raises, do not catch it.

- [ ] **Step 4: Add direct-forward fallback tests**

```python
def test_fused_moe_falls_back_when_megamoe_returns_none(monkeypatch, fused_layer, topk):
    monkeypatch.setattr(megamoe, "forward_megamoe_or_none", lambda *_: None)
    monkeypatch.setattr(fused_layer, "forward_impl", Mock(return_value=sentinel))
    assert fused_layer.forward(hidden_states(1), topk) is sentinel
    fused_layer.forward_impl.assert_called_once()
```

- [ ] **Step 5: Run payload and forward tests**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py -k "payload or clipped or falls_back"`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py python/sglang/srt/layers/moe/fused_moe_triton/layer.py test/registered/unit/npu/test_ascend_megamoe.py
git commit -s -m "feat(npu): run W8A8 MoE through MegaMOE"
```

### Task 4: Validate integration boundaries and publish the launch contract

**Files:**

- Modify: `test/registered/unit/npu/test_ascend_megamoe.py`
- Modify: `docs/docs/hardware-platforms/ascend-npus/model-deployment/` model deployment document selected by the existing DeepSeek V4 documentation layout

**Interfaces:**

- Consumes the public backend/env contract from Task 1 and complete wrapper from Tasks 2–3.
- Produces a documented opt-in launch configuration and test evidence of no impact when disabled.

- [ ] **Step 1: Write failing end-to-end mocked integration tests**

```python
def test_opt_in_w8a8_invokes_operator_once(monkeypatch, fused_layer, topk):
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "1")
    op = Mock(return_value=(output(), expert_counts()))
    monkeypatch.setattr(megamoe, "_load_ops", lambda: fake_ops({"mega_moe": op}))
    assert fused_layer.forward(hidden_states(2), topk).shape == (2, fused_layer.hidden_size)
    op.assert_called_once()

def test_disabled_backend_never_imports_operator(monkeypatch, fused_layer, topk):
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MEGAMOE", "0")
    monkeypatch.setattr(megamoe, "_load_ops", Mock(side_effect=AssertionError))
    fused_layer.forward(hidden_states(2), topk)
```

- [ ] **Step 2: Run the full host-safe MegaMOE test module**

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py`

Expected: PASS.

- [ ] **Step 3: Document only the supported W8A8 launch path**

```bash
export SGLANG_NPU_ENABLE_MEGAMOE=1
export SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS=1024
python -m sglang.launch_server \
  --model-path <DeepSeek-V4-W8A8-checkpoint> \
  --device npu --tp <ep-size> \
  --moe-a2a-backend ascend_megamoe
```

Document required EP constraints, the strict-mode behavior, and that W4A8/MXFP4 remain unsupported in this first release. Do not describe this as DeepEP acceleration.

- [ ] **Step 4: Run formatting and focused regression tests**

Run: `pre-commit run --files python/sglang/srt/environ.py python/sglang/srt/layers/moe/utils.py python/sglang/srt/hardware_backend/npu/moe/megamoe.py python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py python/sglang/srt/layers/moe/fused_moe_triton/layer.py test/registered/unit/npu/test_ascend_megamoe.py`

Run: `pytest -q test/registered/unit/npu/test_ascend_megamoe.py test/registered/unit/npu/quantization/test_fp4_moe_methods.py`

Expected: PASS.

- [ ] **Step 5: Record required real-NPU verification without claiming it ran**

```bash
export PYTHONPATH=$PWD/python:$PYTHONPATH
export SGLANG_NPU_ENABLE_MEGAMOE=1
python -c "from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe; print('MegaMOE API import passed')"
# Run the documented DeepSeek V4 W8A8 EP launch, compare output against the
# disabled path, then benchmark equal topology, batch and CANN versions.
```

Expected: This command is run only on the target Ascend/CANN host. Capture exact package versions, EP topology, accuracy threshold, and throughput/latency in the PR evidence.

- [ ] **Step 6: Commit Task 4**

```bash
git add test/registered/unit/npu/test_ascend_megamoe.py docs/docs/hardware-platforms/ascend-npus/model-deployment
git commit -s -m "docs(npu): document DeepSeek V4 MegaMOE launch"
```

## Plan Self-Review

- Spec coverage: Tasks 1–3 implement the backend, controls, payload, symmetric buffer, clamp and fallback; Task 4 covers public launch behavior and verification. DeepEP remains untouched throughout.
- No-placeholder check: all tasks name files, interfaces, test commands, expected outcomes and concrete snippets.
- Type consistency: Task 2 defines `forward_megamoe_or_none`; Task 3 is its only FusedMoE consumer. W8A8 payload is produced by Task 3 using Task 2's `cache_megamoe_w8a8_payload`.
- Review Focus mapping: missing dependency, unequal rank batches and capacity overflow are tested in Task 2; generic GPU backend separation in Task 1/2; DeepSeek V4 clamp in Task 3.
