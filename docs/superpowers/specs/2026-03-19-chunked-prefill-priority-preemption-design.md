# Chunked Prefill Priority Preemption Design

**Date**: 2026-03-19
**Status**: Approved
**Goal**: Reduce TTFT for short requests in a PD-disaggregated, mixed-context-length deployment

---

## Background

SGLang supports chunked prefill: long requests are split into multiple chunks processed across
successive scheduler iterations. The current scheduler unconditionally resumes the in-progress
`chunked_req` before examining the waiting queue. A 32K-token request chunked into 16 × 2K
pieces holds `chunked_req` for 16 consecutive iterations, blocking all shorter requests and
inflating their TTFT by the full remaining prefill time of the long request.

This design targets **PD-disaggregated deployments** with **mixed context lengths**, where
TTFT is the primary optimization metric.

---

## Problem Statement

In `scheduler.py`, `_get_new_batch_prefill_raw()` unconditionally resumes the chunked request:

```python
# scheduler.py ~line 2257
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

A short request (e.g., 512 tokens) arriving while a 32K-token request is mid-chunk must wait
for all remaining chunks to complete before it can start prefill. This is a classic
Head-of-Line (HOL) blocking problem.

---

## Design

### Core Algorithm: SJF with Anti-Starvation

Before resuming `chunked_req`, check whether the highest-priority waiting request is
significantly shorter than the chunked request's remaining prefill work. If so, yield this
iteration to the waiting queue. A starvation counter ensures the long request is eventually
forced through.

```
Each scheduler iteration:
1. Sort waiting_queue (existing logic, priority + FCFS)
2. If chunked_req exists:
   a. top_req = waiting_queue[0]
   b. If top_req.remaining_prefill < chunked_req.remaining_prefill * ratio:
       → If chunked_req.skip_count < max_yield:
           skip_count++
           skip chunked_req, process waiting_queue this round
       → Else (anti-starvation limit reached):
           reset skip_count, force resume chunked_req
   c. Else: resume chunked_req normally
3. Continue with normal waiting_queue iteration
```

**Remaining prefill tokens** — correct field references:
- **Chunked request**: `chunked_req.extend_input_len` — `init_next_round_input()` is called
  **before** the preemption check so this field holds the accurate remaining token count for
  this round. Calling `init_next_round_input()` multiple times is safe (idempotent).
- **Waiting request**: `len(top_req.origin_input_ids)` — `init_next_round_input()` has not
  been called yet on waiting requests, so `extend_input_len` and `prefix_indices` are stale.
  `origin_input_ids` is the raw input and is always valid. Prefix cache hits are not subtracted
  here; this slightly over-estimates short request cost, which is a conservative bias
  (makes preemption less aggressive, not more).

---

## Changes

### `python/sglang/srt/managers/schedule_batch.py`

Add one field to `Req.__init__()`:

```python
# Chunked prefill priority preemption
self.preemption_skip_count: int = 0
```

### `python/sglang/srt/server_args.py`

Add three new fields to the `ServerArgs` dataclass **and** corresponding entries in
`ServerArgs.add_cli_args()`:

```python
# Dataclass fields:
enable_chunked_prefill_preemption: bool = False
chunked_prefill_preemption_ratio: float = 0.3
chunked_prefill_preemption_max_yield: int = 3

# add_cli_args() entries:
parser.add_argument("--enable-chunked-prefill-preemption", action="store_true")
parser.add_argument("--chunked-prefill-preemption-ratio", type=float, default=0.3)
parser.add_argument("--chunked-prefill-preemption-max-yield", type=int, default=3)
```

### `python/sglang/srt/managers/scheduler.py`

Add helper method and modify `_get_new_batch_prefill_raw()`:

```python
def _should_preempt_chunked_req(self, waiting_queue: List[Req]) -> bool:
    """
    Returns True if the chunked_req should yield this iteration to a shorter
    waiting request.  Uses the prior-round extend_input_len as an approximation
    of remaining chunked work (accurate to within one chunk size).
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

    # extend_input_len from the prior round is the remaining token count
    chunked_remaining = chunked_req.extend_input_len

    # origin_input_ids length is the only reliable estimate before
    # init_next_round_input() is called on waiting requests
    top_req = waiting_queue[0]
    top_remaining = len(top_req.origin_input_ids)

    if top_remaining < chunked_remaining * ratio:
        chunked_req.preemption_skip_count += 1
        return True

    return False


# In _get_new_batch_prefill_raw(), replace lines ~2257-2259:
if self.chunked_req is not None:
    # init_next_round_input() called first so extend_input_len is accurate
    self.chunked_req.init_next_round_input()
    if self._should_preempt_chunked_req(self.waiting_queue):
        pass  # yield this round; init_next_round_input is idempotent
    else:
        self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_chunked_prefill_preemption` | `False` | Enable the feature (opt-in) |
| `chunked_prefill_preemption_ratio` | `0.3` | Short request must have < ratio × chunked_remaining tokens to trigger preemption |
| `chunked_prefill_preemption_max_yield` | `3` | Max consecutive skips before anti-starvation forces chunked_req through |

**Intuition for `ratio=0.3`**: If `chunked_req` has 10K tokens remaining, only requests with
< 3K tokens in `origin_input_ids` are short enough to justify interruption. A 4K request
does not provide enough TTFT benefit to warrant disrupting the long request's progress.

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| `waiting_queue` empty, `chunked_req` active | `_should_preempt` returns False immediately |
| Multiple short requests waiting | `waiting_queue` already sorted; all served in one round |
| Short request also relatively long | `top_remaining ≥ chunked_remaining * ratio` → no preemption |
| `chunked_req` skipped `max_yield` times | Reset counter, force resume (anti-starvation) |
| `chunked_req` on last chunk | `chunked_remaining` is small (≤ one chunk); threshold `chunked_remaining * ratio` also shrinks, making preemption harder to trigger — correct behavior since the long request is nearly done |
| PP mode | Same check added to `event_loop_overlap_schedule_batch()` path |
| Feature disabled (default) | Zero overhead; behavior identical to current |
| Large prefix cache hit | `origin_input_ids` slightly over-estimates true compute cost; conservatively reduces preemption aggressiveness |

---

## Parallelism Compatibility

| Parallelism | Impact | Notes |
|-------------|--------|-------|
| **PCP** | **Incompatible** | `server_args.py` asserts `chunked_prefill_size is None or chunked_prefill_size == -1` when PCP is enabled (i.e., the user must not have explicitly set a positive chunk size). Chunked prefill and PCP cannot be active simultaneously; this feature only activates when `chunked_req is not None` (i.e., chunked prefill is enabled). No conflict at runtime. |
| **TP** | None | TP is below the scheduler; transparent |
| **DP** | None | DP operates within a forward pass |
| **EP** | None | EP operates within a forward pass |
| **PP** | Minor | Same logic needed in PP's overlap scheduling path (`event_loop_overlap_schedule_batch`) |

---

## Expected Impact

- **Short request TTFT**: Reduced by up to `max_yield × chunked_prefill_size` tokens of
  blocked compute. With defaults (max_yield=3, chunked_prefill_size=4K): worst-case blocking
  reduced from full remaining chunks to at most ~12K tokens. Actual benefit depends on when
  the short request arrives relative to `skip_count`.
- **Long request TTFT**: Increases by at most `max_yield` scheduler iterations (bounded by
  anti-starvation).
- **Throughput**: Neutral — total token computation unchanged; only ordering changes.

---

## Files Modified

| File | Change |
|------|--------|
| `python/sglang/srt/managers/schedule_batch.py` | +1 field in `Req` |
| `python/sglang/srt/server_args.py` | +3 dataclass fields + 3 CLI args in `add_cli_args()` |
| `python/sglang/srt/managers/scheduler.py` | +~35 lines (helper method + modified block) |
| `test/srt/cpu/test_chunked_prefill_preemption.py` | New unit test: preemption triggers, anti-starvation reset, disabled-by-default path |
