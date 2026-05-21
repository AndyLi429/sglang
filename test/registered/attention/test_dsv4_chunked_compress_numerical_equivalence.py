"""Numerical equivalence test for DSV4 chunked-compress layout.

Stronger than the structural UT (``test_dsv4_chunked_compress_layout``):
this test reconstructs the actual ape + softmax-weighted sum that
``Compressor.forward_npu`` performs, drives it through the layout
planner for arbitrary chunked schedules, and asserts the per-token
compressed output is numerically identical (to fp64 tolerance) to a
single-shot baseline mirroring the in-tree CUDA / NPU prefill math.

The test is the precondition for wiring the layout into
``Compressor.forward_npu`` — if any chunked schedule diverged from the
whole-sequence path, the wiring would silently mis-compress tokens
under chunked prefill. Catching that here means the wiring step can
proceed as a mechanical translation rather than a fresh correctness
proof.

Validated invariants (the ones that drive Phase 1.5 wiring):

1. State-ring tokens carry ``score + ape[gp % ratio]`` at stash time
   (gp = global position). The enumeration-index convention used by
   the existing single-shot code is a coincidence of cutoff alignment;
   chunked schedules require the explicit ``gp % ratio`` form.
2. ``SRC_ZERO`` sources contribute ``kv=0`` and ``score=-inf`` — the
   latter zeroes the softmax weight, so the row contributes nothing
   to the sum (matches the ``_overlap_transform`` neutral-fill).
3. Overlap slicing convention: current-half sources use the ``[d:]``
   slice of each token's ``coff*d`` activation; prev-half sources use
   the ``[:d]`` slice. Order of the 2*ratio sources is irrelevant
   for the sum.
"""

import sys
import unittest
from typing import Dict, List, Tuple

import torch

from sglang.srt.layers.attention.dsv4.chunked_compress_layout import (
    SRC_CHUNK,
    SRC_STATE,
    SRC_ZERO,
    compute_chunked_compress_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _overlap_transform(tensor: torch.Tensor, value: float, d: int) -> torch.Tensor:
    n_chunks, r, _ = tensor.shape
    out = tensor.new_full((n_chunks, 2 * r, d), value)
    out[:, r:] = tensor[..., d:]
    out[1:, :r] = tensor[:-1, :, :d]
    return out


def reference_compress(
    kv_full: torch.Tensor,
    score_full: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    overlap: bool,
    d: int,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Single-shot baseline mirroring ``Compressor.forward_npu`` prefill."""
    T = kv_full.shape[0]
    cutoff = T - T % ratio
    remainder = T % ratio
    compressed = kv_full.new_zeros((0, d))
    final_state_kv: Dict[int, torch.Tensor] = {}
    final_state_score: Dict[int, torch.Tensor] = {}

    if cutoff >= ratio:
        kv_chunks = kv_full[:cutoff].unflatten(0, (-1, ratio))
        score_chunks = score_full[:cutoff].unflatten(0, (-1, ratio)) + ape
        if overlap:
            kv2 = _overlap_transform(kv_chunks, 0.0, d)
            sc2 = _overlap_transform(score_chunks, float("-inf"), d)
            compressed = (kv2 * sc2.softmax(dim=1)).sum(dim=1)
        else:
            compressed = (kv_chunks * score_chunks.softmax(dim=1)).sum(dim=1)

    if overlap and cutoff >= ratio:
        for j in range(ratio):
            gp = cutoff - ratio + j
            final_state_kv[gp] = kv_full[gp].clone()
            final_state_score[gp] = score_full[gp] + ape[j]
    if remainder > 0:
        for j in range(remainder):
            gp = cutoff + j
            final_state_kv[gp] = kv_full[gp].clone()
            final_state_score[gp] = score_full[gp] + ape[j]

    return compressed, final_state_kv, final_state_score


def _materialize_source(
    source,
    chunk_kv: torch.Tensor,
    chunk_score: torch.Tensor,
    state_kv: Dict[int, torch.Tensor],
    state_score: Dict[int, torch.Tensor],
    ape: torch.Tensor,
    j_in_chunk: int,
    coff_d: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kind, idx = source
    if kind == SRC_ZERO:
        return chunk_kv.new_zeros((coff_d,)), chunk_kv.new_full(
            (coff_d,), float("-inf")
        )
    if kind == SRC_CHUNK:
        return chunk_kv[idx], chunk_score[idx] + ape[j_in_chunk]
    if kind == SRC_STATE:
        return state_kv[idx], state_score[idx]
    raise AssertionError(kind)


def _apply_one_compress_output(
    co,
    chunk_kv: torch.Tensor,
    chunk_score: torch.Tensor,
    state_kv: Dict[int, torch.Tensor],
    state_score: Dict[int, torch.Tensor],
    ape: torch.Tensor,
    ratio: int,
    overlap: bool,
    d: int,
) -> torch.Tensor:
    coff_d = chunk_kv.shape[1]
    if not overlap:
        rows_kv, rows_sc = [], []
        for j, src in enumerate(co.sources):
            kv_row, sc_row = _materialize_source(
                src, chunk_kv, chunk_score, state_kv, state_score, ape, j, coff_d
            )
            rows_kv.append(kv_row)
            rows_sc.append(sc_row)
        kv_stack = torch.stack(rows_kv, dim=0)
        sc_stack = torch.stack(rows_sc, dim=0)
        return (kv_stack * sc_stack.softmax(dim=0)).sum(dim=0)

    current_srcs = co.sources[:ratio]
    prev_srcs = co.sources[ratio:]
    rows_kv, rows_sc = [], []
    for j, src in enumerate(current_srcs):
        kv_row, sc_row = _materialize_source(
            src, chunk_kv, chunk_score, state_kv, state_score, ape, j, coff_d
        )
        rows_kv.append(kv_row[d:])
        rows_sc.append(sc_row[d:])
    for j, src in enumerate(prev_srcs):
        kv_row, sc_row = _materialize_source(
            src, chunk_kv, chunk_score, state_kv, state_score, ape, j, coff_d
        )
        rows_kv.append(kv_row[:d])
        rows_sc.append(sc_row[:d])
    kv_stack = torch.stack(rows_kv, dim=0)
    sc_stack = torch.stack(rows_sc, dim=0)
    return (kv_stack * sc_stack.softmax(dim=0)).sum(dim=0)


def chunked_compress(
    kv_full: torch.Tensor,
    score_full: torch.Tensor,
    ape: torch.Tensor,
    chunk_boundaries: List[int],
    ratio: int,
    overlap: bool,
    d: int,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    T = kv_full.shape[0]
    assert chunk_boundaries[-1] == T
    state_kv: Dict[int, torch.Tensor] = {}
    state_score: Dict[int, torch.Tensor] = {}
    all_outputs: List[Tuple[int, torch.Tensor]] = []

    cursor = 0
    for end in chunk_boundaries:
        prefix_len = cursor
        chunk_len = end - cursor
        chunk_kv = kv_full[cursor:end]
        chunk_score = score_full[cursor:end]
        layout = compute_chunked_compress_layout(
            prefix_len=prefix_len,
            chunk_len=chunk_len,
            ratio=ratio,
            overlap=overlap,
        )
        for co in layout.compress_outputs:
            row = _apply_one_compress_output(
                co,
                chunk_kv,
                chunk_score,
                state_kv,
                state_score,
                ape,
                ratio,
                overlap,
                d,
            )
            all_outputs.append((co.compressed_seq_pos, row))
        for stash in layout.state_stashes:
            # Invariant: ape index = gp % ratio for all stash kinds.
            for local_idx, gp in zip(stash.chunk_local_indices, stash.global_positions):
                j = gp % ratio
                state_kv[gp] = chunk_kv[local_idx].clone()
                state_score[gp] = chunk_score[local_idx] + ape[j]
        cursor = end

    all_outputs.sort(key=lambda kv: kv[0])
    compressed = (
        torch.stack([row for _, row in all_outputs], dim=0)
        if all_outputs
        else kv_full.new_zeros((0, d))
    )
    return compressed, state_kv, state_score


class TestChunkedCompressNumericalEquivalence(CustomTestCase):
    def setUp(self):
        torch.manual_seed(20260520)

    def _gen_inputs(self, T: int, ratio: int, overlap: bool, d: int = 8):
        coff = 2 if overlap else 1
        kv_full = torch.randn(T, coff * d, dtype=torch.float64)
        score_full = torch.randn(T, coff * d, dtype=torch.float64) * 0.3
        ape = torch.randn(ratio, coff * d, dtype=torch.float64) * 0.5
        return kv_full, score_full, ape

    def _assert_outputs_equal(self, ref, chk, atol=1e-10):
        ref_c, ref_kv, ref_sc = ref
        chk_c, chk_kv, chk_sc = chk
        self.assertEqual(ref_c.shape, chk_c.shape)
        if ref_c.numel() > 0:
            self.assertLess((ref_c - chk_c).abs().max().item(), atol)
        # Reference's state ring describes only the final batch's stash;
        # chunked path accumulates historical entries. Require subset only.
        self.assertTrue(set(ref_kv.keys()).issubset(set(chk_kv.keys())))
        for gp in ref_kv:
            self.assertLess(
                (ref_kv[gp] - chk_kv[gp]).abs().max().item(),
                atol,
                f"state-kv at gp={gp} diverges",
            )
            self.assertLess(
                (ref_sc[gp] - chk_sc[gp]).abs().max().item(),
                atol,
                f"state-score at gp={gp} diverges",
            )

    # ---- Overlap (ratio=4) ----

    def test_overlap_whole_sequence(self):
        T, ratio, d = 16, 4, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=True, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, True, d),
            chunked_compress(kv, sc, ape, [T], ratio, True, d),
        )

    def test_overlap_aligned_split(self):
        T, ratio, d = 16, 4, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=True, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, True, d),
            chunked_compress(kv, sc, ape, [8, 16], ratio, True, d),
        )

    def test_overlap_unaligned_split(self):
        T, ratio, d = 16, 4, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=True, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, True, d),
            chunked_compress(kv, sc, ape, [5, 10, 16], ratio, True, d),
        )

    def test_overlap_many_one_token_chunks(self):
        # Most aggressive split: T tokens processed one at a time. Forces
        # the layout to handle every cross-batch state-ring read pattern.
        T, ratio, d = 20, 4, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=True, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, True, d),
            chunked_compress(kv, sc, ape, list(range(1, T + 1)), ratio, True, d),
        )

    # ---- Non-overlap (ratio=128) ----

    def test_non_overlap_whole_sequence(self):
        T, ratio, d = 256, 128, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=False, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, False, d),
            chunked_compress(kv, sc, ape, [T], ratio, False, d),
        )

    def test_non_overlap_split_at_boundary(self):
        T, ratio, d = 384, 128, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=False, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, False, d),
            chunked_compress(kv, sc, ape, [128, 384], ratio, False, d),
        )

    def test_non_overlap_split_mid_chunk(self):
        T, ratio, d = 300, 128, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=False, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, False, d),
            chunked_compress(kv, sc, ape, [70, 200, 300], ratio, False, d),
        )

    def test_non_overlap_remainder_then_complete(self):
        # First chunk only stashes remainder; second chunk completes k=0
        # using mixed state-ring + current-chunk sources.
        T, ratio, d = 130, 128, 8
        kv, sc, ape = self._gen_inputs(T, ratio, overlap=False, d=d)
        self._assert_outputs_equal(
            reference_compress(kv, sc, ape, ratio, False, d),
            chunked_compress(kv, sc, ape, [50, 130], ratio, False, d),
        )


if __name__ == "__main__":
    sys.exit(unittest.main())
