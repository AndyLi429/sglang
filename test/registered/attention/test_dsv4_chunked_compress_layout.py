"""CPU unit tests for the DSV4 chunked-compress layout planner.

Validates ``compute_chunked_compress_layout`` against three correctness
properties:

1. Whole-sequence baseline: ``layout(prefix=0, chunk=N)`` produces exactly
   ``N // ratio`` compressed outputs, all sourced from current chunk, with
   stash kinds matching the existing ``Compressor.forward_npu`` prefill
   branch (overlap stash of the trailing ratio tokens + remainder stash).
2. Chunked split invariance: splitting a request into multiple chunks
   produces the same set of compressed outputs (by k) as processing the
   whole request at once. The union of state-ring stashes plus current-chunk
   sources for any compressed k is equal across splits.
3. Edge cases: empty chunk, chunk shorter than ratio, prefix straddling a
   ratio boundary, overlap k=0 ``SRC_ZERO`` fill, ratio=128 non-overlap.
"""

import sys
import unittest

from sglang.srt.layers.attention.dsv4.chunked_compress_layout import (
    SRC_CHUNK,
    SRC_STATE,
    SRC_ZERO,
    compute_chunked_compress_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _all_global_positions_for(layout):
    """Return the set of global token positions that the layout describes as
    raw inputs to compressed outputs (excluding SRC_ZERO), for cross-split
    equivalence checks.
    """
    out = {}
    for co in layout.compress_outputs:
        positions = []
        for kind, idx in co.sources:
            if kind == SRC_ZERO:
                positions.append(None)
            elif kind == SRC_CHUNK:
                positions.append(layout.prefix_len + idx)
            elif kind == SRC_STATE:
                positions.append(idx)
            else:
                raise AssertionError(f"unexpected source kind {kind}")
        out[co.compressed_seq_pos] = tuple(positions)
    return out


class TestChunkedCompressLayoutWholeSequence(CustomTestCase):
    """Property 1 — whole-sequence baseline."""

    def test_whole_sequence_no_overlap_aligned(self):
        # prefix=0, chunk=512, ratio=128, overlap=False (the c128 path):
        # 4 compressed outputs, each from 128 consecutive chunk-local tokens,
        # no stash (no remainder, overlap=False so no overlap stash).
        layout = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=512, ratio=128, overlap=False
        )
        self.assertEqual(len(layout.compress_outputs), 4)
        for k, co in enumerate(layout.compress_outputs):
            self.assertEqual(co.compressed_seq_pos, k)
            self.assertEqual(co.rope_position, k * 128)
            self.assertEqual(len(co.sources), 128)
            for off, (kind, idx) in enumerate(co.sources):
                self.assertEqual(kind, SRC_CHUNK)
                self.assertEqual(idx, k * 128 + off)
        self.assertEqual(len(layout.state_stashes), 0)

    def test_whole_sequence_no_overlap_with_remainder(self):
        # prefix=0, chunk=515, ratio=128: 4 compressed outputs + 3 remainder
        # tokens (no overlap stash since overlap=False).
        layout = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=515, ratio=128, overlap=False
        )
        self.assertEqual(len(layout.compress_outputs), 4)
        self.assertEqual(len(layout.state_stashes), 1)
        rem = layout.state_stashes[0]
        self.assertEqual(rem.kind, "remainder")
        self.assertEqual(rem.chunk_local_indices, (512, 513, 514))
        self.assertEqual(rem.global_positions, (512, 513, 514))

    def test_whole_sequence_overlap_aligned(self):
        # prefix=0, chunk=16, ratio=4, overlap=True: 4 compressed outputs,
        # plus an overlap stash of the trailing 4 tokens (k=3's raw tokens).
        # No remainder.
        layout = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=16, ratio=4, overlap=True
        )
        self.assertEqual(len(layout.compress_outputs), 4)
        # Each output: 4 current-half sources + 4 previous-half sources.
        for k, co in enumerate(layout.compress_outputs):
            self.assertEqual(len(co.sources), 8)
            # First 4 = current half.
            for off in range(4):
                kind, idx = co.sources[off]
                self.assertEqual(kind, SRC_CHUNK)
                self.assertEqual(idx, k * 4 + off)
            # Next 4 = previous half. For k=0, SRC_ZERO; else from chunk.
            for off in range(4):
                kind, idx = co.sources[4 + off]
                if k == 0:
                    self.assertEqual(kind, SRC_ZERO)
                else:
                    self.assertEqual(kind, SRC_CHUNK)
                    self.assertEqual(idx, (k - 1) * 4 + off)
        # Stash kinds: overlap only (no remainder since 16 % 4 == 0).
        self.assertEqual(len(layout.state_stashes), 1)
        overlap_stash = layout.state_stashes[0]
        self.assertEqual(overlap_stash.kind, "overlap")
        self.assertEqual(overlap_stash.chunk_local_indices, (12, 13, 14, 15))

    def test_whole_sequence_overlap_with_remainder(self):
        # prefix=0, chunk=18, ratio=4, overlap=True: 4 compressed outputs,
        # overlap stash of the trailing 4 of compressed region, plus
        # remainder stash of the trailing 2 tokens.
        layout = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=18, ratio=4, overlap=True
        )
        self.assertEqual(len(layout.compress_outputs), 4)
        kinds = [s.kind for s in layout.state_stashes]
        self.assertIn("overlap", kinds)
        self.assertIn("remainder", kinds)
        # Remainder = chunk-local indices 16, 17.
        rem = next(s for s in layout.state_stashes if s.kind == "remainder")
        self.assertEqual(rem.chunk_local_indices, (16, 17))
        # Overlap = chunk-local indices 12, 13, 14, 15 (last ratio of cutoff).
        ov = next(s for s in layout.state_stashes if s.kind == "overlap")
        self.assertEqual(ov.chunk_local_indices, (12, 13, 14, 15))


class TestChunkedCompressLayoutSplitInvariance(CustomTestCase):
    """Property 2 — chunked split produces the same compressed outputs as
    whole-sequence."""

    def test_split_invariance_overlap_aligned(self):
        ratio = 4
        # Whole: prefix=0, chunk=32.
        whole = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=32, ratio=ratio, overlap=True
        )
        # Split: (0, 16) then (16, 16).
        split_a = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=16, ratio=ratio, overlap=True
        )
        split_b = compute_chunked_compress_layout(
            prefix_len=16, chunk_len=16, ratio=ratio, overlap=True
        )
        # Union of compressed_seq_pos covers exactly the same k set.
        whole_ks = {co.compressed_seq_pos for co in whole.compress_outputs}
        split_ks = {co.compressed_seq_pos for co in split_a.compress_outputs} | {
            co.compressed_seq_pos for co in split_b.compress_outputs
        }
        self.assertEqual(whole_ks, split_ks)
        # For each k, the materialised global positions must agree.
        whole_map = _all_global_positions_for(whole)
        split_map_a = _all_global_positions_for(split_a)
        split_map_b = _all_global_positions_for(split_b)
        merged = {**split_map_a, **split_map_b}
        self.assertEqual(whole_map, merged)

    def test_split_invariance_unaligned_prefix(self):
        # Split chunks at a NON-ratio-aligned boundary. The second chunk's
        # first compressed output mixes state-ring (positions 4..5 from the
        # first chunk's remainder stash) and current-chunk (positions 6..7)
        # raw inputs.
        ratio = 4
        whole = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=12, ratio=ratio, overlap=True
        )
        # Split: (0, 6) then (6, 6).
        split_a = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=6, ratio=ratio, overlap=True
        )
        split_b = compute_chunked_compress_layout(
            prefix_len=6, chunk_len=6, ratio=ratio, overlap=True
        )
        # Whole: k ∈ {0, 1, 2}. split_a: k ∈ {0}. split_b: k ∈ {1, 2}.
        self.assertEqual(
            {co.compressed_seq_pos for co in whole.compress_outputs}, {0, 1, 2}
        )
        self.assertEqual(
            {co.compressed_seq_pos for co in split_a.compress_outputs}, {0}
        )
        self.assertEqual(
            {co.compressed_seq_pos for co in split_b.compress_outputs}, {1, 2}
        )
        # k=1 in split_b: current half = positions 4..7; positions 4,5 are
        # in state ring (prefix=6 means positions 0..5 are prefix), so
        # sources for k=1 mix SRC_STATE and SRC_CHUNK.
        k1 = next(co for co in split_b.compress_outputs if co.compressed_seq_pos == 1)
        # Source layout: first 4 = current half (positions 4,5,6,7),
        # next 4 = previous half (positions 0,1,2,3 → all SRC_STATE).
        current_half = k1.sources[:4]
        prev_half = k1.sources[4:]
        self.assertEqual(current_half[0], (SRC_STATE, 4))
        self.assertEqual(current_half[1], (SRC_STATE, 5))
        self.assertEqual(current_half[2], (SRC_CHUNK, 0))
        self.assertEqual(current_half[3], (SRC_CHUNK, 1))
        for off in range(4):
            self.assertEqual(prev_half[off], (SRC_STATE, off))
        # Materialised position maps must equal whole-sequence's.
        whole_map = _all_global_positions_for(whole)
        merged_map = {
            **_all_global_positions_for(split_a),
            **_all_global_positions_for(split_b),
        }
        self.assertEqual(whole_map, merged_map)


class TestChunkedCompressLayoutEdgeCases(CustomTestCase):
    """Property 3 — edge cases."""

    def test_empty_chunk(self):
        layout = compute_chunked_compress_layout(
            prefix_len=128, chunk_len=0, ratio=4, overlap=True
        )
        self.assertEqual(layout.compress_outputs, ())
        self.assertEqual(layout.state_stashes, ())

    def test_chunk_smaller_than_ratio_no_completion(self):
        # prefix=0, chunk=2, ratio=4: no compressed output, remainder stash
        # of the 2 tokens.
        layout = compute_chunked_compress_layout(
            prefix_len=0, chunk_len=2, ratio=4, overlap=True
        )
        self.assertEqual(layout.compress_outputs, ())
        self.assertEqual(len(layout.state_stashes), 1)
        rem = layout.state_stashes[0]
        self.assertEqual(rem.kind, "remainder")
        self.assertEqual(rem.chunk_local_indices, (0, 1))

    def test_chunk_completes_within_prefix_remainder(self):
        # prefix=2 (so positions 0,1 are stashed in state ring as remainder),
        # chunk=6 (positions 2..7). Total=8 → 2 compressed chunks (k=0, k=1).
        # k=0 current half = positions 0..3: first two from state, last two
        # from chunk; previous half = SRC_ZERO.
        # k=1 current half = positions 4..7: all from chunk; previous half =
        # positions 0..3 (positions 0,1 SRC_STATE; 2,3 SRC_CHUNK).
        layout = compute_chunked_compress_layout(
            prefix_len=2, chunk_len=6, ratio=4, overlap=True
        )
        ks = [co.compressed_seq_pos for co in layout.compress_outputs]
        self.assertEqual(ks, [0, 1])

        k0 = layout.compress_outputs[0]
        self.assertEqual(
            k0.sources[:4],
            ((SRC_STATE, 0), (SRC_STATE, 1), (SRC_CHUNK, 0), (SRC_CHUNK, 1)),
        )
        # Previous half for k=0 is SRC_ZERO.
        self.assertTrue(all(s[0] == SRC_ZERO for s in k0.sources[4:]))

        k1 = layout.compress_outputs[1]
        self.assertEqual(
            k1.sources[:4],
            ((SRC_CHUNK, 2), (SRC_CHUNK, 3), (SRC_CHUNK, 4), (SRC_CHUNK, 5)),
        )
        self.assertEqual(
            k1.sources[4:],
            ((SRC_STATE, 0), (SRC_STATE, 1), (SRC_CHUNK, 0), (SRC_CHUNK, 1)),
        )

    def test_ratio_128_non_overlap(self):
        layout = compute_chunked_compress_layout(
            prefix_len=128, chunk_len=128, ratio=128, overlap=False
        )
        self.assertEqual(len(layout.compress_outputs), 1)
        co = layout.compress_outputs[0]
        self.assertEqual(co.compressed_seq_pos, 1)
        self.assertEqual(co.rope_position, 128)
        self.assertEqual(len(co.sources), 128)
        # All from current chunk (prefix_len = 128 = lower edge).
        for off, src in enumerate(co.sources):
            self.assertEqual(src, (SRC_CHUNK, off))
        # No stash (chunk fully consumed by the compressed chunk).
        self.assertEqual(layout.state_stashes, ())

    def test_invalid_overlap_for_ratio_128_raises(self):
        with self.assertRaises(ValueError):
            compute_chunked_compress_layout(
                prefix_len=0, chunk_len=128, ratio=128, overlap=True
            )

    def test_negative_inputs_raise(self):
        with self.assertRaises(ValueError):
            compute_chunked_compress_layout(
                prefix_len=-1, chunk_len=4, ratio=4, overlap=True
            )
        with self.assertRaises(ValueError):
            compute_chunked_compress_layout(
                prefix_len=0, chunk_len=-1, ratio=4, overlap=True
            )


if __name__ == "__main__":
    sys.exit(unittest.main())
