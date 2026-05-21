"""Chunked-prefill layout planner for DeepSeek-V4 compressor.

The DSV4 ``Compressor.forward_npu`` (and its CUDA counterpart) needs to know,
for a single request inside a chunked-prefill batch:

1. Which compressed-token writes the current chunk produces (= compressed-seq
   positions newly completed by appending ``chunk_len`` tokens after
   ``prefix_len`` already-processed tokens).
2. For each such compressed-token output, which raw input tokens feed it. With
   chunked prefill those raw tokens can live in two places:
   - ``state``: the compressor state ring (tokens stashed by a previous chunk).
   - ``chunk``: the current Q tensor (current chunk-local indices).
3. Which current-chunk tokens must be stashed to the state ring so the next
   chunk (or decode step) can complete its compressed chunks.

This module exposes a *pure* (no torch / no NPU) planner that returns plain
Python lists of indices and global positions; the caller is responsible for
turning ``("state", global_pos)`` into a slot in the state ring via the
``DeepSeekV4TokenToKVPool`` helpers, and for materialising the tensor inputs
from current-chunk activations.

The intent is that any compressor implementation (CUDA triton, NPU custom
ops, or a future fallback) can drive its tensor assembly off this layout
without re-deriving cutoff / remainder / overlap edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# Source kinds. Use plain strings so the layout is trivially serialisable
# and has no torch dependency. Callers should match on these constants.
SRC_STATE = "state"  # read from state ring at the given global position
SRC_CHUNK = "chunk"  # read from current Q tensor at the given chunk-local index
SRC_ZERO = "zero"  # overlap-right-half for k=0: fill with neutral value


# A single source for one of the ``ratio`` (non-overlap) or ``2 * ratio``
# (overlap) raw-token slots that feed a compressed output.
#
# - kind == SRC_STATE: ``idx`` is the GLOBAL token position; caller translates
#   via the V4 KV pool helper to a state-ring slot.
# - kind == SRC_CHUNK: ``idx`` is the index into the current chunk's local Q
#   activation tensor (0 <= idx < chunk_len).
# - kind == SRC_ZERO: ``idx`` is unused; signals overlap-right-half for k=0
#   (no previous chunk exists), which the existing _overlap_transform fills
#   with 0 for kv and -inf for score.
Source = Tuple[str, int]


@dataclass(frozen=True)
class CompressOutput:
    """One compressed-token write for the current chunk."""

    # k: global compressed-seq position (= absolute index in the request's
    # compressed kv pool).
    compressed_seq_pos: int
    # Global token position to feed into rope for this compressed output.
    # Matches CUDA convention `positions[k * ratio]` (the first token of the
    # raw range that produced this compressed token).
    rope_position: int
    # Length is exactly ``ratio`` for non-overlap or ``2 * ratio`` for overlap.
    # For overlap, order is [current_half_0..ratio-1, prev_half_0..ratio-1],
    # matching ``Compressor._overlap_transform`` (current chunk's left half
    # rows first, previous chunk's right half rows second).
    sources: Tuple[Source, ...]


@dataclass(frozen=True)
class StateStash:
    """A range of current-chunk tokens to push into the state ring."""

    # Indices into current Q (0 <= idx < chunk_len).
    chunk_local_indices: Tuple[int, ...]
    # Corresponding global token positions; caller derives state-ring slots
    # from these.
    global_positions: Tuple[int, ...]
    # "remainder" (post-cutoff tail of a chunk; spans 0..ratio-1 tokens) or
    # "overlap" (last `ratio` tokens of the last newly compressed chunk, kept
    # for next batch's overlap right-half). The kinds match the two stash
    # branches in the existing ``Compressor.forward_npu`` prefill loop.
    kind: str


@dataclass(frozen=True)
class ChunkCompressLayout:
    prefix_len: int
    chunk_len: int
    ratio: int
    overlap: bool

    compress_outputs: Tuple[CompressOutput, ...] = field(default_factory=tuple)
    state_stashes: Tuple[StateStash, ...] = field(default_factory=tuple)


def compute_chunked_compress_layout(
    *,
    prefix_len: int,
    chunk_len: int,
    ratio: int,
    overlap: bool,
) -> ChunkCompressLayout:
    """Compute the per-request compress layout for a chunked-prefill batch.

    ``prefix_len`` = tokens of this request already processed by previous
    forward passes (so already represented in the kv / state pools).
    ``chunk_len`` = tokens of this request in the current forward batch.
    ``ratio`` ∈ {4, 128}; ``overlap`` should be ``ratio == 4``.

    Returned layout describes:

      - which compressed-token outputs (``k`` indices) this batch produces,
      - for each output, the ratio (or 2 * ratio) input sources,
      - which current-chunk tokens to stash to the state ring afterwards.

    The function is pure: no torch, no NPU, deterministic on (prefix_len,
    chunk_len, ratio, overlap). It validates basic invariants but never
    touches the kv pool — the caller resolves slots.
    """
    if prefix_len < 0 or chunk_len < 0:
        raise ValueError(
            f"prefix_len and chunk_len must be non-negative; got "
            f"prefix_len={prefix_len}, chunk_len={chunk_len}"
        )
    if ratio <= 0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    if overlap and ratio != 4:
        # The existing code only treats ratio == 4 as overlap; guard against
        # accidental misuse on the 128 path.
        raise ValueError(
            f"overlap=True is only valid for ratio=4 in DSV4; got ratio={ratio}"
        )

    total_len = prefix_len + chunk_len

    # k indices newly completed by this chunk.
    # Before: prefix_len // ratio chunks done.
    # After:  total_len // ratio  chunks done.
    k_first = prefix_len // ratio
    k_last_exclusive = total_len // ratio

    compress_outputs: List[CompressOutput] = []
    for k in range(k_first, k_last_exclusive):
        token_global_start = k * ratio
        # Build the `ratio` "current-half" sources for this k.
        current_sources: List[Source] = []
        for off in range(ratio):
            p = token_global_start + off
            if p < prefix_len:
                # Token already lives in the state ring (stashed by a
                # previous chunk's remainder / overlap branch).
                current_sources.append((SRC_STATE, p))
            else:
                # Token is in current Q at local index p - prefix_len.
                current_sources.append((SRC_CHUNK, p - prefix_len))

        if overlap:
            # Previous-chunk right-half: tokens [(k-1)*ratio, k*ratio).
            prev_sources: List[Source] = []
            if k == 0:
                # First compressed chunk has no predecessor — neutral fill.
                # Caller materialises 0 for kv and -inf for score, matching
                # ``Compressor._overlap_transform``.
                prev_sources = [(SRC_ZERO, 0)] * ratio
            else:
                prev_start = (k - 1) * ratio
                for off in range(ratio):
                    p = prev_start + off
                    if p < prefix_len:
                        prev_sources.append((SRC_STATE, p))
                    else:
                        prev_sources.append((SRC_CHUNK, p - prefix_len))
            sources = tuple(current_sources) + tuple(prev_sources)
        else:
            sources = tuple(current_sources)

        compress_outputs.append(
            CompressOutput(
                compressed_seq_pos=k,
                rope_position=token_global_start,
                sources=sources,
            )
        )

    state_stashes: List[StateStash] = []

    # 1) Remainder stash: tokens at global positions [L_cmp, total_len) that
    #    are within the current chunk. L_cmp is the largest multiple of ratio
    #    not exceeding total_len.
    l_cmp = (total_len // ratio) * ratio
    rem_global_start = max(l_cmp, prefix_len)
    rem_global_end = total_len
    if rem_global_end > rem_global_start:
        local_indices = tuple(
            p - prefix_len for p in range(rem_global_start, rem_global_end)
        )
        global_positions = tuple(range(rem_global_start, rem_global_end))
        state_stashes.append(
            StateStash(
                chunk_local_indices=local_indices,
                global_positions=global_positions,
                kind="remainder",
            )
        )

    # 2) Overlap stash: last `ratio` tokens of the last newly compressed chunk
    #    (k = k_last_exclusive - 1) — only the portion that lives in the
    #    current chunk needs to be stashed (the state-ring portion is already
    #    there from a previous batch).
    if overlap and k_last_exclusive > k_first:
        k_last = k_last_exclusive - 1
        ov_global_start = k_last * ratio
        ov_global_end = ov_global_start + ratio
        ov_chunk_start = max(ov_global_start, prefix_len)
        if ov_global_end > ov_chunk_start:
            local_indices = tuple(
                p - prefix_len for p in range(ov_chunk_start, ov_global_end)
            )
            global_positions = tuple(range(ov_chunk_start, ov_global_end))
            state_stashes.append(
                StateStash(
                    chunk_local_indices=local_indices,
                    global_positions=global_positions,
                    kind="overlap",
                )
            )

    return ChunkCompressLayout(
        prefix_len=prefix_len,
        chunk_len=chunk_len,
        ratio=ratio,
        overlap=overlap,
        compress_outputs=tuple(compress_outputs),
        state_stashes=tuple(state_stashes),
    )
