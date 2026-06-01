"""Unit tests for the NPU MLA-CP fields in
``cp_utils.prepare_context_parallel_metadata`` (consumed by the Ascend
``forward_mla_pcp`` path). All tests run on CPU — no GPU/NPU required.

The metadata indexes into the all-gathered KV (which ``rebuild_cp_kv_cache``
reranges to original per-sequence order). For each rank's head (prev) and tail
(next) Q block we validate the diagonal "mask" block and the preceding "nomask"
prefix block, and that the union over all ranks reproduces full causal
attention.
"""

import unittest

import sglang.srt.layers.attention.dsa.utils as dsa_utils
import sglang.srt.layers.utils.cp_utils as cp_utils
from sglang.srt.layers.utils.cp_utils import prepare_context_parallel_metadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _block_sizes(L, cp_size):
    seg = cp_size * 2
    base, rem = L // seg, L % seg
    return [base + 1 if i < rem else base for i in range(seg)]


class TestCpUtilsNpuMetadata(CustomTestCase):
    def setUp(self):
        # Force the NPU field-population branch and neutralize DSA round-robin
        # / prefill-cp helpers so the function runs standalone on CPU.
        self._orig_is_npu = cp_utils._is_npu
        cp_utils._is_npu = True
        self._orig_rr = dsa_utils.is_dsa_prefill_cp_round_robin_split
        self._orig_enable = dsa_utils.is_dsa_enable_prefill_cp
        dsa_utils.is_dsa_prefill_cp_round_robin_split = lambda: False
        dsa_utils.is_dsa_enable_prefill_cp = lambda: False

    def tearDown(self):
        cp_utils._is_npu = self._orig_is_npu
        dsa_utils.is_dsa_prefill_cp_round_robin_split = self._orig_rr
        dsa_utils.is_dsa_enable_prefill_cp = self._orig_enable

    def _meta(self, extend_seqs_len, cp_rank, cp_size):
        kv_len = sum(extend_seqs_len)
        return prepare_context_parallel_metadata(
            kv_len=kv_len,
            cp_rank=cp_rank,
            cp_size=cp_size,
            seqs_len=list(extend_seqs_len),  # no prefix → prefix_offsets == 0
            extend_seqs_len=list(extend_seqs_len),
            device="cpu",
        )

    def test_bs1_small_example(self):
        # Colleague's example: L=8, cp_size=2 → blocks [2,2,2,2].
        L, cp_size = 8, 2
        seg = cp_size * 2
        for cp_rank in range(cp_size):
            m = self._meta([L], cp_rank, cp_size)
            tail_block = seg - cp_rank - 1
            head_start, head_end = 2 * cp_rank, 2 * (cp_rank + 1)
            tail_start, tail_end = 2 * tail_block, 2 * (tail_block + 1)
            self.assertEqual(
                m.npu_head_mask_idx.tolist(), list(range(head_start, head_end))
            )
            self.assertEqual(m.npu_head_nomask_idx.tolist(), list(range(0, head_start)))
            self.assertEqual(
                m.npu_tail_mask_idx.tolist(), list(range(tail_start, tail_end))
            )
            self.assertEqual(m.npu_tail_nomask_idx.tolist(), list(range(0, tail_start)))
            # mask seqlens = diagonal block length (q == kv).
            self.assertEqual(m.npu_head_mask_seqlens.tolist(), [2])
            self.assertEqual(m.npu_tail_mask_seqlens.tolist(), [2])
            # nomask seqlens = [q_lens, kv_prefix_lens].
            self.assertEqual(m.npu_head_nomask_seqlens.tolist(), [[2], [head_start]])
            self.assertEqual(m.npu_tail_nomask_seqlens.tolist(), [[2], [tail_start]])

    def _assert_causal_and_coverage(self, extend_seqs_len, cp_size):
        seg = cp_size * 2
        seq_base = [sum(extend_seqs_len[:s]) for s in range(len(extend_seqs_len))]
        # (q_block, kv_block) pairs covered across all ranks, per sequence.
        covered = {s: set() for s in range(len(extend_seqs_len))}
        for cp_rank in range(cp_size):
            m = self._meta(extend_seqs_len, cp_rank, cp_size)
            tail_block = seg - cp_rank - 1
            # per-chunk mask∪nomask must be the contiguous causal prefix, no overlap.
            for name_mask, name_nomask in (
                ("npu_head_mask_idx", "npu_head_nomask_idx"),
                ("npu_tail_mask_idx", "npu_tail_nomask_idx"),
            ):
                rows = getattr(m, name_mask).tolist() + getattr(m, name_nomask).tolist()
                self.assertEqual(len(rows), len(set(rows)), "mask/nomask overlap")
            off_h = off_t = 0
            for s, L in enumerate(extend_seqs_len):
                blk = _block_sizes(L, cp_size)
                base = seq_base[s]
                h_diag, t_diag = blk[cp_rank], blk[tail_block]
                h_end = base + sum(blk[: cp_rank + 1])
                t_end = base + sum(blk[: tail_block + 1])
                self.assertEqual(
                    m.npu_head_mask_idx.tolist()[off_h : off_h + h_diag],
                    list(range(h_end - h_diag, h_end)),
                )
                self.assertEqual(
                    m.npu_tail_mask_idx.tolist()[off_t : off_t + t_diag],
                    list(range(t_end - t_diag, t_end)),
                )
                off_h += h_diag
                off_t += t_diag
                covered[s].add((cp_rank, cp_rank))
                covered[s].update((cp_rank, kvb) for kvb in range(cp_rank))
                covered[s].add((tail_block, tail_block))
                covered[s].update((tail_block, kvb) for kvb in range(tail_block))
        # Union over all ranks == full lower-triangular block structure.
        expected = {(qb, kvb) for qb in range(seg) for kvb in range(qb + 1)}
        for s in range(len(extend_seqs_len)):
            self.assertEqual(covered[s], expected, f"seq {s} causal coverage")

    def test_non_divisible_length(self):
        self._assert_causal_and_coverage([7], 2)
        self._assert_causal_and_coverage([17], 4)

    def test_batch_gt_1(self):
        self._assert_causal_and_coverage([16, 24], 2)
        self._assert_causal_and_coverage([100, 64, 80], 4)

    def test_fields_none_on_non_npu(self):
        cp_utils._is_npu = False
        try:
            m = self._meta([16], 2)
        finally:
            cp_utils._is_npu = True
        self.assertIsNone(m.npu_head_mask_idx)
        self.assertIsNone(m.npu_tail_nomask_seqlens)


if __name__ == "__main__":
    unittest.main()
