"""Unit test for CPU precomputation of GDN chunked-prefill indices.

Validates that `build_gdn_chunked_prefill_meta` produces the same
`chunk_indices` / `chunk_offsets` tensors as the in-tree lazy helpers
`prepare_chunk_indices` / `prepare_chunk_offsets`, which are the ground
truth used by all existing fla kernels.
"""

import unittest

import torch

from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear.gdn_chunk_meta import (
    build_gdn_chunked_prefill_meta,
)


def _cu_seqlens_from(seq_lens):
    return torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.long
    )


class TestGDNChunkMeta(unittest.TestCase):

    def _check(self, seq_lens, chunk_size):
        device = torch.device("cpu")
        cu_seqlens = _cu_seqlens_from(seq_lens)

        expected_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        expected_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)

        meta = build_gdn_chunked_prefill_meta(
            extend_seq_lens_cpu=list(seq_lens),
            chunk_size=chunk_size,
            device=device,
            use_pinned_memory=False,
        )
        self.assertIsNotNone(meta)
        torch.testing.assert_close(meta.chunk_indices.cpu(), expected_indices.cpu())
        torch.testing.assert_close(meta.chunk_offsets.cpu(), expected_offsets.cpu())

    def test_single_seq_exact_multiple(self):
        self._check([128], chunk_size=64)

    def test_single_seq_ragged(self):
        self._check([100], chunk_size=64)

    def test_mixed_batch(self):
        self._check([64, 65, 130, 1], chunk_size=64)

    def test_small_chunk(self):
        self._check([3, 7, 1], chunk_size=2)

    def test_empty_batch_returns_none(self):
        meta = build_gdn_chunked_prefill_meta(
            extend_seq_lens_cpu=[],
            chunk_size=64,
            device=torch.device("cpu"),
            use_pinned_memory=False,
        )
        self.assertIsNone(meta)


if __name__ == "__main__":
    unittest.main()
