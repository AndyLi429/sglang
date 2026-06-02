from dataclasses import dataclass
from itertools import accumulate
from typing import Sequence, Tuple


@dataclass(frozen=True)
class MLACPRingSegment:
    query_start: int
    query_end: int
    q_lens: Tuple[int, ...]
    mask_kv_ranges: Tuple[Tuple[int, int], ...]
    nomask_kv_ranges: Tuple[Tuple[int, int], ...]

    @property
    def q_len(self) -> int:
        return self.query_end - self.query_start

    @property
    def max_q_len(self) -> int:
        return max(self.q_lens)

    @property
    def mask_kv_len(self) -> int:
        return sum(end - start for start, end in self.mask_kv_ranges)

    @property
    def nomask_kv_len(self) -> int:
        return sum(self.nomask_kv_lens)

    @property
    def nomask_kv_lens(self) -> Tuple[int, ...]:
        return tuple(end - start for start, end in self.nomask_kv_ranges)


def _int_tuple(values: Sequence[int], name: str, bs: int) -> Tuple[int, ...]:
    if values is None or len(values) != bs:
        raise ValueError(f"Expected {name} to have length {bs}, got {values}")
    return tuple(int(x) for x in values)


def _seq_offsets(cp_meta, bs: int) -> Tuple[int, ...]:
    split_list = getattr(cp_meta, "split_list", None)
    if split_list is None:
        if bs != 1:
            raise ValueError("split_list is required for multi-batch MLA CP metadata")
        seq_len = max(cp_meta.kv_len_prev_list[0], cp_meta.kv_len_next_list[0])
        return (0, seq_len)

    if len(split_list) % bs != 0:
        raise ValueError(
            f"split_list length {len(split_list)} is not divisible by bs={bs}"
        )

    cp_segment_num = len(split_list) // bs
    seq_lens = [
        sum(int(x) for x in split_list[s * cp_segment_num : (s + 1) * cp_segment_num])
        for s in range(bs)
    ]
    return tuple(accumulate([0] + seq_lens))


def _build_segment(
    query_start: int,
    q_lens: Tuple[int, ...],
    kv_lens: Tuple[int, ...],
    seq_offsets: Tuple[int, ...],
) -> MLACPRingSegment:
    mask_kv_ranges = []
    nomask_kv_ranges = []
    for batch_id, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if q_len <= 0:
            raise ValueError(f"CP MLA query segment must be non-empty, got {q_len}")
        if kv_len < q_len:
            raise ValueError(f"CP MLA kv_len={kv_len} must be >= q_len={q_len}")
        seq_start = seq_offsets[batch_id]
        seq_end = seq_offsets[batch_id + 1]
        if kv_len > seq_end - seq_start:
            raise ValueError(
                f"CP MLA kv_len={kv_len} exceeds sequence length "
                f"{seq_end - seq_start} for batch {batch_id}"
            )

        mask_kv_start = seq_start + kv_len - q_len
        mask_kv_end = seq_start + kv_len
        mask_kv_ranges.append((mask_kv_start, mask_kv_end))
        nomask_kv_ranges.append((seq_start, mask_kv_start))

    q_len_total = sum(q_lens)
    return MLACPRingSegment(
        query_start=query_start,
        query_end=query_start + q_len_total,
        q_lens=q_lens,
        mask_kv_ranges=tuple(mask_kv_ranges),
        nomask_kv_ranges=tuple(nomask_kv_ranges),
    )


def build_mla_cp_ring_segments(cp_meta, prefix_lens=None):
    """Build the two rank-local ring MLA segments for Ascend MLA CP.

    The CP query layout is [prev_half, next_half]. Each half attends to
    [0, kv_len) with a no-mask prefix and a triangular current block.
    """
    bs = int(getattr(cp_meta, "bs", 1))

    if prefix_lens is not None and any(int(x) > 0 for x in prefix_lens):
        raise NotImplementedError(
            "Ascend MLA prefill CP with prefix cache is not supported yet"
        )

    prev_q_lens = _int_tuple(cp_meta.actual_seq_q_prev_list, "prev_q_lens", bs)
    next_q_lens = _int_tuple(cp_meta.actual_seq_q_next_list, "next_q_lens", bs)
    prev_kv_lens = _int_tuple(cp_meta.kv_len_prev_list, "prev_kv_lens", bs)
    next_kv_lens = _int_tuple(cp_meta.kv_len_next_list, "next_kv_lens", bs)
    seq_offsets = _seq_offsets(cp_meta, bs)

    prev_start = 0
    next_start = int(cp_meta.total_q_prev_tokens)
    if next_start != sum(prev_q_lens):
        raise ValueError(
            f"Unexpected CP MLA split: total_q_prev_tokens={next_start}, "
            f"prev_q_lens={prev_q_lens}"
        )
    if int(cp_meta.total_q_next_tokens) != sum(next_q_lens):
        raise ValueError(
            f"Unexpected CP MLA next tokens: total_q_next_tokens="
            f"{cp_meta.total_q_next_tokens}, next_q_lens={next_q_lens}"
        )

    return (
        _build_segment(prev_start, prev_q_lens, prev_kv_lens, seq_offsets),
        _build_segment(next_start, next_q_lens, next_kv_lens, seq_offsets),
    )
