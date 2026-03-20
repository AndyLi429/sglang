# Chunked Prefill Priority Preemption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SJF-based preemption so short requests can interrupt an in-progress chunked prefill, reducing TTFT in mixed-context PD-disaggregated deployments.

**Architecture:** A new `_should_preempt_chunked_req()` method on `Scheduler` is called after `init_next_round_input()` but before `add_chunked_req()`. It checks whether the highest-priority waiting request is short enough to justify yielding. Calling `init_next_round_input()` first ensures `extend_input_len` reflects the accurate remaining token count (not the prior chunk's truncated value). A `preemption_skip_count` counter on `Req` prevents starvation. Feature is opt-in via `enable_chunked_prefill_preemption=False` by default.

**Tech Stack:** Python, SGLang scheduler (`managers/scheduler.py`), request dataclass (`managers/schedule_batch.py`), server args (`server_args.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-03-19-chunked-prefill-priority-preemption-design.md`

**PP mode:** Out of scope for this task. The `event_loop_overlap_schedule_batch()` path (Pipeline Parallel) would need the same logic, but is explicitly deferred. A `# TODO: apply preemption check here for PP mode` comment will be added at the PP call site.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `python/sglang/srt/managers/schedule_batch.py` | Modify | Add `preemption_skip_count` field to `Req` |
| `python/sglang/srt/server_args.py` | Modify | Add 3 dataclass fields + 3 CLI args |
| `python/sglang/srt/managers/scheduler.py` | Modify | Add `_should_preempt_chunked_req()` method + modify 3-line call site |
| `test/srt/cpu/test_chunked_prefill_preemption.py` | Create | Unit tests covering all preemption scenarios |

---

## Task 1: Add `preemption_skip_count` field to `Req`

**Files:**
- Modify: `python/sglang/srt/managers/schedule_batch.py` (after line 673)

The `Req` class has a block of chunking-related state around line 668 (`self.is_chunked = 0`).
The new counter goes after the retraction block that ends at line 673 (`self.retracted_stain = False`).

- [ ] **Step 1: Write failing test**

Create `test/srt/cpu/test_chunked_prefill_preemption.py`:

```python
"""Unit tests for chunked prefill priority preemption (test/srt/cpu/).

Run with:  python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py -v
No GPU required.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers shared across tasks
# ---------------------------------------------------------------------------

def _make_chunked_req(extend_input_len: int, skip_count: int = 0):
    """Minimal duck-typed Req for preemption tests."""
    req = SimpleNamespace(
        extend_input_len=extend_input_len,
        preemption_skip_count=skip_count,
    )
    return req


def _make_waiting_req(input_len: int):
    req = SimpleNamespace(origin_input_ids=list(range(input_len)))
    return req


def _make_sched(enable=True, ratio=0.3, max_yield=3, chunked_req=None):
    """Duck-typed Scheduler for calling the unbound method under test."""
    return SimpleNamespace(
        server_args=SimpleNamespace(
            enable_chunked_prefill_preemption=enable,
            chunked_prefill_preemption_ratio=ratio,
            chunked_prefill_preemption_max_yield=max_yield,
        ),
        chunked_req=chunked_req,
    )


# ---------------------------------------------------------------------------
# Task 1: Req field
# ---------------------------------------------------------------------------

def test_req_has_preemption_skip_count():
    """Req must initialise preemption_skip_count to 0 in __init__."""
    # We instantiate a real Req via __new__ + __init__ shortcut to avoid
    # needing the full constructor args; we then check the attribute exists
    # on an instance produced by the real constructor path (import-time check).
    from sglang.srt.managers.schedule_batch import Req
    # A real Req.__init__ sets the attribute; if the attribute is missing this
    # test will raise AttributeError, which pytest reports as a failure.
    req = MagicMock(spec=Req)
    # spec= means only attributes on Req are accessible; accessing a missing
    # attribute raises AttributeError → test fails until we add the field.
    _ = req.preemption_skip_count
```

Run:
```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py::test_req_has_preemption_skip_count -v 2>&1 | head -30
```

Expected: FAIL — `AttributeError: Mock object has no attribute 'preemption_skip_count'`
(because `Req` does not yet have this field, so `spec=Req` disallows it).

- [ ] **Step 2: Add field to `Req.__init__`**

In `python/sglang/srt/managers/schedule_batch.py`, after line 673 (`self.retracted_stain = False`):

```python
        # Chunked prefill priority preemption
        # Counts how many consecutive scheduler iterations this chunked request
        # has been skipped to let shorter requests run first (anti-starvation).
        self.preemption_skip_count: int = 0
```

- [ ] **Step 3: Run test to verify it passes**

```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py::test_req_has_preemption_skip_count -v 2>&1 | head -30
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd d:/Github/sglang && git add python/sglang/srt/managers/schedule_batch.py test/srt/cpu/test_chunked_prefill_preemption.py && git commit -m "feat: add preemption_skip_count field to Req for chunked prefill preemption"
```

---

## Task 2: Add server args

**Files:**
- Modify: `python/sglang/srt/server_args.py` (dataclass fields after line 361; CLI args after line 3724)

- [ ] **Step 1: Write failing test**

Add to `test/srt/cpu/test_chunked_prefill_preemption.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: ServerArgs defaults
# ---------------------------------------------------------------------------

def test_server_args_defaults():
    """New server args must exist with correct defaults."""
    from sglang.srt.server_args import ServerArgs
    args = ServerArgs(model_path="dummy")
    assert args.enable_chunked_prefill_preemption is False
    assert args.chunked_prefill_preemption_ratio == 0.3
    assert args.chunked_prefill_preemption_max_yield == 3
```

Run:
```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py::test_server_args_defaults -v 2>&1 | head -30
```

Expected: FAIL — `AttributeError` (fields not yet added).

- [ ] **Step 2: Add dataclass fields**

In `python/sglang/srt/server_args.py`, after line 361
(`prefill_delayer_wait_seconds_buckets: Optional[List[float]] = None`), insert:

```python
    enable_chunked_prefill_preemption: bool = False
    chunked_prefill_preemption_ratio: float = 0.3
    chunked_prefill_preemption_max_yield: int = 3
```

- [ ] **Step 3: Add CLI args**

In `python/sglang/srt/server_args.py`, inside `add_cli_args()`, after line 3724
(the closing `)` of the `--chunked-prefill-size` argument block, immediately before the
`--prefill-max-requests` argument at line 3725), insert:

```python
        parser.add_argument(
            "--enable-chunked-prefill-preemption",
            action="store_true",
            help="Enable priority preemption for chunked prefill: short requests can "
            "interrupt an in-progress chunked request to reduce TTFT.",
        )
        parser.add_argument(
            "--chunked-prefill-preemption-ratio",
            type=float,
            default=ServerArgs.chunked_prefill_preemption_ratio,
            help="A waiting request triggers preemption only if its input length is "
            "less than this ratio times the chunked request's remaining tokens. "
            "Default: 0.3.",
        )
        parser.add_argument(
            "--chunked-prefill-preemption-max-yield",
            type=int,
            default=ServerArgs.chunked_prefill_preemption_max_yield,
            help="Maximum consecutive scheduler iterations a chunked request can be "
            "skipped before anti-starvation forces it through. Default: 3.",
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py::test_server_args_defaults -v 2>&1 | head -30
```

Expected: PASS.

- [ ] **Step 5: Verify CLI args parse correctly**

```bash
cd d:/Github/sglang && python -c "
from sglang.srt.server_args import ServerArgs
import argparse
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
args = p.parse_args(['--model-path', 'dummy', '--enable-chunked-prefill-preemption', '--chunked-prefill-preemption-ratio', '0.5'])
sa = ServerArgs.from_cli_args(args)
assert sa.enable_chunked_prefill_preemption is True
assert sa.chunked_prefill_preemption_ratio == 0.5
assert sa.chunked_prefill_preemption_max_yield == 3
print('CLI args OK')
"
```

Expected: `CLI args OK`.

- [ ] **Step 6: Commit**

```bash
cd d:/Github/sglang && git add python/sglang/srt/server_args.py && git commit -m "feat: add chunked prefill preemption server args and CLI flags"
```

---

## Task 3: Implement `_should_preempt_chunked_req` and modify call site

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`

**Key design note:** `init_next_round_input()` is called **before** the preemption check.
This ensures `chunked_req.extend_input_len` is set to the accurate remaining token count
(not the prior chunk's truncated value). Calling `init_next_round_input()` multiple times
is safe — it just recalculates `fill_ids` and `extend_input_len`.

- [ ] **Step 1: Write failing tests**

Add to `test/srt/cpu/test_chunked_prefill_preemption.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: Core preemption logic — tested against the real Scheduler method
# ---------------------------------------------------------------------------
#
# We call Scheduler._should_preempt_chunked_req as an unbound method on a
# duck-typed SimpleNamespace.  This gives a true red phase: before the method
# is added, the call raises AttributeError.

def _call_method(sched, waiting_queue):
    from sglang.srt.managers.scheduler import Scheduler
    return Scheduler._should_preempt_chunked_req(sched, waiting_queue)


def test_preemption_triggers_for_short_request():
    """Short request (< ratio * chunked_remaining) must trigger preemption."""
    chunked = _make_chunked_req(extend_input_len=10000)
    sched = _make_sched(ratio=0.3, max_yield=3, chunked_req=chunked)
    short_req = _make_waiting_req(input_len=2000)  # 2000 < 10000*0.3=3000
    assert _call_method(sched, [short_req]) is True
    assert chunked.preemption_skip_count == 1


def test_no_preemption_for_long_request():
    """Request >= ratio * chunked_remaining must NOT trigger preemption."""
    chunked = _make_chunked_req(extend_input_len=10000)
    sched = _make_sched(ratio=0.3, max_yield=3, chunked_req=chunked)
    long_req = _make_waiting_req(input_len=4000)  # 4000 >= 10000*0.3=3000
    assert _call_method(sched, [long_req]) is False
    assert chunked.preemption_skip_count == 0


def test_antistarvation_forces_resume():
    """After max_yield skips, next call must force resume (False) and reset counter."""
    # With max_yield=3, force-resume fires when skip_count >= 3, i.e. on the 4th call.
    chunked = _make_chunked_req(extend_input_len=10000, skip_count=3)
    sched = _make_sched(max_yield=3, chunked_req=chunked)
    short_req = _make_waiting_req(input_len=100)
    assert _call_method(sched, [short_req]) is False  # forced resume
    assert chunked.preemption_skip_count == 0           # counter reset


def test_disabled_by_default_no_preemption():
    """Feature must be a no-op when enable_chunked_prefill_preemption=False."""
    chunked = _make_chunked_req(extend_input_len=10000)
    sched = _make_sched(enable=False, chunked_req=chunked)
    short_req = _make_waiting_req(input_len=100)
    assert _call_method(sched, [short_req]) is False
    assert chunked.preemption_skip_count == 0  # untouched


def test_empty_waiting_queue_no_preemption():
    """Empty waiting_queue must never trigger preemption."""
    chunked = _make_chunked_req(extend_input_len=10000)
    sched = _make_sched(chunked_req=chunked)
    assert _call_method(sched, []) is False


def test_skip_count_increments_then_resets():
    """skip_count increments on each preemption; force-resume fires when it hits max_yield."""
    # With max_yield=2:
    #   call 1: skip_count=0 < 2  → increment to 1, return True
    #   call 2: skip_count=1 < 2  → increment to 2, return True
    #   call 3: skip_count=2 >= 2 → reset to 0, return False (forced resume)
    #   call 4: skip_count=0 < 2  → increment to 1, return True  (cycle restarts)
    chunked = _make_chunked_req(extend_input_len=10000, skip_count=0)
    sched = _make_sched(ratio=0.3, max_yield=2, chunked_req=chunked)
    short = _make_waiting_req(input_len=500)

    assert _call_method(sched, [short]) is True   # call 1
    assert chunked.preemption_skip_count == 1

    assert _call_method(sched, [short]) is True   # call 2
    assert chunked.preemption_skip_count == 2

    assert _call_method(sched, [short]) is False  # call 3: forced resume
    assert chunked.preemption_skip_count == 0

    assert _call_method(sched, [short]) is True   # call 4: cycle restarts
    assert chunked.preemption_skip_count == 1
```

Run:
```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py -k "preemption or antistarvation or disabled or empty_waiting or skip_count" -v 2>&1 | head -40
```

Expected: FAIL — `AttributeError: type object 'Scheduler' has no attribute '_should_preempt_chunked_req'`

- [ ] **Step 2: Add `_should_preempt_chunked_req` method to `Scheduler`**

In `python/sglang/srt/managers/scheduler.py`, find `_get_new_batch_prefill_raw`
(search: `def _get_new_batch_prefill_raw`) and insert the following method **immediately before** it:

```python
    def _should_preempt_chunked_req(self, waiting_queue) -> bool:
        """Return True if chunked_req should yield this iteration to a shorter request.

        Must be called AFTER init_next_round_input() so that extend_input_len
        reflects the accurate remaining token count for this round.
        Uses len(top_req.origin_input_ids) for waiting requests since
        init_next_round_input() has not yet been called on them; this slightly
        over-estimates their cost when prefix cache hits exist (conservative bias).
        """
        if not self.server_args.enable_chunked_prefill_preemption:
            return False
        if not waiting_queue:
            return False

        chunked_req = self.chunked_req
        ratio = self.server_args.chunked_prefill_preemption_ratio
        max_yield = self.server_args.chunked_prefill_preemption_max_yield

        # Anti-starvation: force resume after max_yield consecutive skips
        if chunked_req.preemption_skip_count >= max_yield:
            chunked_req.preemption_skip_count = 0
            return False

        # extend_input_len is set by init_next_round_input() called just before this
        chunked_remaining = chunked_req.extend_input_len

        top_req = waiting_queue[0]
        top_remaining = len(top_req.origin_input_ids)

        if top_remaining < chunked_remaining * ratio:
            chunked_req.preemption_skip_count += 1
            return True

        return False
```

- [ ] **Step 3: Modify the call site in `_get_new_batch_prefill_raw`**

In `python/sglang/srt/managers/scheduler.py`, replace the 3-line block at lines 2257-2259:

**Old code (3 lines):**
```python
        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

**New code:**
```python
        if self.chunked_req is not None:
            # init_next_round_input() must be called first so extend_input_len
            # is accurate before the preemption check.
            self.chunked_req.init_next_round_input()
            if self._should_preempt_chunked_req(self.waiting_queue):
                pass  # yield this round; init_next_round_input is idempotent,
                      # so it will be called again next iteration safely.
            else:
                self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

Also find the PP overlap scheduling path. Open `python/sglang/srt/managers/scheduler_pp_mixin.py` and search for any `chunked_req` resume block (similar `init_next_round_input` / `add_chunked_req` calls). Add a comment there (do NOT implement the logic — that is out of scope):

```python
                # TODO: apply _should_preempt_chunked_req here for PP mode (out of scope)
```

- [ ] **Step 4: Run all preemption tests**

```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py -v 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 5: Run existing CPU tests to check for regressions**

```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/ --ignore=test/srt/cpu/test_chunked_prefill_preemption.py -v 2>&1 | tail -30
```

Expected: All pre-existing tests PASS.

- [ ] **Step 6: Run lint**

```bash
cd d:/Github/sglang && pre-commit run --files \
  python/sglang/srt/managers/schedule_batch.py \
  python/sglang/srt/server_args.py \
  python/sglang/srt/managers/scheduler.py \
  test/srt/cpu/test_chunked_prefill_preemption.py 2>&1 | tail -20
```

Expected: All hooks pass. If `black` or `isort` reformats files, stage the changes and re-run lint until clean.

- [ ] **Step 7: Commit**

```bash
cd d:/Github/sglang && git add \
  python/sglang/srt/managers/scheduler.py \
  test/srt/cpu/test_chunked_prefill_preemption.py && \
git commit -m "feat: implement chunked prefill priority preemption to reduce TTFT for short requests"
```

---

## Task 4: Add integration-style scenario tests

These tests simulate multi-iteration scheduling sequences to validate the full preemption
cycle (not just individual method calls).

- [ ] **Step 1: Add scenario tests**

Add to `test/srt/cpu/test_chunked_prefill_preemption.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: Scenario tests (multi-iteration sequences)
# ---------------------------------------------------------------------------

def test_short_request_skips_ahead_then_chunked_resumes():
    """
    Scenario:
    Iter 1 — chunked_req (10K) + short_req (500) in queue → preempt (short first)
    Iter 2 — chunked_req still active, queue now empty → resume chunked_req
    """
    chunked = _make_chunked_req(extend_input_len=10000, skip_count=0)
    sched = _make_sched(ratio=0.3, max_yield=3, chunked_req=chunked)

    # Iter 1: short present → preempt
    assert _call_method(sched, [_make_waiting_req(500)]) is True
    assert chunked.preemption_skip_count == 1

    # Iter 2: queue empty → resume
    assert _call_method(sched, []) is False
    assert chunked.preemption_skip_count == 1  # unchanged when no preemption


def test_full_antistarvation_cycle_with_max_yield_2():
    """
    With max_yield=2: two skips then force resume → counter resets → cycle restarts.
    Sequence from skip_count=0:
      call 1: skip_count=0 < 2  → increment to 1, return True
      call 2: skip_count=1 < 2  → increment to 2, return True
      call 3: skip_count=2 >= 2 → reset to 0, return False (forced resume)
      call 4: skip_count=0 < 2  → increment to 1, return True (cycle restarts)
    """
    chunked = _make_chunked_req(extend_input_len=10000, skip_count=0)
    sched = _make_sched(ratio=0.3, max_yield=2, chunked_req=chunked)
    short = _make_waiting_req(input_len=100)

    assert _call_method(sched, [short]) is True   # call 1: skip 1
    assert chunked.preemption_skip_count == 1

    assert _call_method(sched, [short]) is True   # call 2: skip 2
    assert chunked.preemption_skip_count == 2

    assert _call_method(sched, [short]) is False  # call 3: forced resume
    assert chunked.preemption_skip_count == 0

    # Cycle restarts
    assert _call_method(sched, [short]) is True   # call 4: skip 1 again
    assert chunked.preemption_skip_count == 1


def test_boundary_ratio_exactly_at_threshold():
    """Request at exactly ratio*chunked_remaining must NOT preempt (strict less-than)."""
    chunked = _make_chunked_req(extend_input_len=1000)
    sched = _make_sched(ratio=0.3, chunked_req=chunked)
    # Exactly at threshold: 300 < 1000*0.3=300.0 is False
    exact_req = _make_waiting_req(input_len=300)
    assert _call_method(sched, [exact_req]) is False
```

- [ ] **Step 2: Run all tests**

```bash
cd d:/Github/sglang && python -m pytest test/srt/cpu/test_chunked_prefill_preemption.py -v 2>&1 | tail -25
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd d:/Github/sglang && git add test/srt/cpu/test_chunked_prefill_preemption.py && \
git commit -m "test: add scenario tests for chunked prefill preemption anti-starvation cycle"
```

---

## Summary

| Task | Files Changed | New Tests |
|------|--------------|-----------|
| 1: Req field | `schedule_batch.py` | `test_req_has_preemption_skip_count` |
| 2: Server args | `server_args.py` | `test_server_args_defaults` |
| 3: Core logic | `scheduler.py` + test | 6 unit tests |
| 4: Scenario tests | test file only | 3 scenario tests |

Total new production code: ~35 lines across 3 files.
Total new test code: ~130 lines in 1 file.
Feature is disabled by default (`enable_chunked_prefill_preemption=False`).
PP mode deferred with a `# TODO` comment.
