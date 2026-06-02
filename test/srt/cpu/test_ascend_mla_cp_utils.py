import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HELPER_PATH = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "attention"
    / "cp_mla_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("cp_mla_utils_under_test", _HELPER_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_mla_cp_ring_segments = _MODULE.build_mla_cp_ring_segments


def _cp_meta_from_split(cp_rank, cp_size, extend_lens):
    bs = len(extend_lens)
    cp_segment_num = 2 * cp_size
    per_seq_block_sizes = []
    split_list = []
    for seq_len in extend_lens:
        base = seq_len // cp_segment_num
        rem = seq_len % cp_segment_num
        blocks = [base + 1 if i < rem else base for i in range(cp_segment_num)]
        per_seq_block_sizes.append(blocks)
        split_list.extend(blocks)

    prev_q_lens = [blocks[cp_rank] for blocks in per_seq_block_sizes]
    next_q_lens = [
        blocks[cp_segment_num - cp_rank - 1] for blocks in per_seq_block_sizes
    ]
    prev_kv_lens = [sum(blocks[: cp_rank + 1]) for blocks in per_seq_block_sizes]
    next_kv_lens = [
        sum(blocks[: cp_segment_num - cp_rank]) for blocks in per_seq_block_sizes
    ]

    return SimpleNamespace(
        total_q_prev_tokens=sum(prev_q_lens),
        total_q_next_tokens=sum(next_q_lens),
        actual_seq_q_prev_list=prev_q_lens,
        actual_seq_q_next_list=next_q_lens,
        kv_len_prev_list=prev_kv_lens,
        kv_len_next_list=next_kv_lens,
        split_list=split_list,
        bs=bs,
    )


def _meta(
    prev_q_len,
    next_q_len,
    prev_kv_len,
    next_kv_len,
    *,
    bs=1,
    split_list=None,
):
    return SimpleNamespace(
        total_q_prev_tokens=prev_q_len,
        total_q_next_tokens=next_q_len,
        actual_seq_q_prev_list=[prev_q_len],
        actual_seq_q_next_list=[next_q_len],
        kv_len_prev_list=[prev_kv_len],
        kv_len_next_list=[next_kv_len],
        bs=bs,
        split_list=split_list,
    )


def test_build_mla_cp_ring_segments_without_prefix():
    segments = build_mla_cp_ring_segments(_meta(64, 64, 64, 256))

    assert segments[0].query_start == 0
    assert segments[0].query_end == 64
    assert segments[0].q_lens == (64,)
    assert segments[0].mask_kv_ranges == ((0, 64),)
    assert segments[0].nomask_kv_len == 0

    assert segments[1].query_start == 64
    assert segments[1].query_end == 128
    assert segments[1].q_lens == (64,)
    assert segments[1].mask_kv_ranges == ((192, 256),)
    assert segments[1].nomask_kv_ranges == ((0, 192),)
    assert segments[1].nomask_kv_len == 192


def test_build_mla_cp_ring_segments_allows_prior_extend_blocks():
    segments = build_mla_cp_ring_segments(_meta(32, 32, 64, 224))

    assert segments[0].mask_kv_ranges == ((32, 64),)
    assert segments[0].nomask_kv_ranges == ((0, 32),)


def test_build_mla_cp_ring_segments_supports_multi_batch():
    cp_meta = SimpleNamespace(
        total_q_prev_tokens=26,
        total_q_next_tokens=37,
        actual_seq_q_prev_list=[20, 6],
        actual_seq_q_next_list=[30, 7],
        kv_len_prev_list=[30, 11],
        kv_len_next_list=[60, 18],
        split_list=[10, 20, 30, 40, 5, 6, 7, 8],
        bs=2,
    )

    segments = build_mla_cp_ring_segments(cp_meta)

    assert segments[0].query_start == 0
    assert segments[0].query_end == 26
    assert segments[0].q_lens == (20, 6)
    assert segments[0].mask_kv_ranges == ((10, 30), (105, 111))
    assert segments[0].nomask_kv_ranges == ((0, 10), (100, 105))

    assert segments[1].query_start == 26
    assert segments[1].query_end == 63
    assert segments[1].q_lens == (30, 7)
    assert segments[1].mask_kv_ranges == ((30, 60), (111, 118))
    assert segments[1].nomask_kv_ranges == ((0, 30), (100, 111))


@pytest.mark.parametrize(
    "cp_rank,cp_size,extend_lens",
    [(1, 2, [101, 54]), (2, 4, [65, 73, 80])],
)
def test_build_mla_cp_ring_segments_matches_cp_metadata_formula(
    cp_rank, cp_size, extend_lens
):
    cp_meta = _cp_meta_from_split(cp_rank, cp_size, extend_lens)
    segments = build_mla_cp_ring_segments(cp_meta)
    seq_offsets = [0]
    for seq_len in extend_lens:
        seq_offsets.append(seq_offsets[-1] + seq_len)

    for segment, q_lens, kv_lens in (
        (segments[0], cp_meta.actual_seq_q_prev_list, cp_meta.kv_len_prev_list),
        (segments[1], cp_meta.actual_seq_q_next_list, cp_meta.kv_len_next_list),
    ):
        assert segment.q_lens == tuple(q_lens)
        for batch_id, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            seq_start = seq_offsets[batch_id]
            assert segment.mask_kv_ranges[batch_id] == (
                seq_start + kv_len - q_len,
                seq_start + kv_len,
            )
            assert segment.nomask_kv_ranges[batch_id] == (
                seq_start,
                seq_start + kv_len - q_len,
            )


def test_build_mla_cp_ring_segments_rejects_prefix_for_v1():
    with pytest.raises(NotImplementedError, match="prefix cache"):
        build_mla_cp_ring_segments(
            _meta(64, 64, 64, 256),
            prefix_lens=[16],
        )


def test_build_mla_cp_ring_segments_requires_split_list_for_multi_batch():
    cp_meta = SimpleNamespace(
        total_q_prev_tokens=128,
        total_q_next_tokens=128,
        actual_seq_q_prev_list=[64, 64],
        actual_seq_q_next_list=[64, 64],
        kv_len_prev_list=[64, 64],
        kv_len_next_list=[256, 256],
        bs=2,
    )

    with pytest.raises(ValueError, match="split_list"):
        build_mla_cp_ring_segments(cp_meta)
