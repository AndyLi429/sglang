from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_GDN_CHUNK_SIZE = 64
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18


@dataclass
class GDNChunkedPrefillMetadata:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def compute_gdn_cumsum_block_size(
    num_heads: int, chunk_size: int = _GDN_CHUNK_SIZE
) -> int:
    chunks = max(1, _GDN_CUMSUM_WORKING_SET // (num_heads * chunk_size))
    return _next_power_of_2(chunks)


def _chunk_counts_cpu(cu_seqlens_cpu: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")


def _build_chunk_indices(counts: torch.Tensor) -> torch.Tensor:
    total = int(counts.sum().item())
    if total == 0:
        return torch.empty((0, 2), dtype=torch.int32)

    seq_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.int32), counts
    )
    starts = torch.empty_like(counts)
    starts[:1] = 0
    starts[1:] = counts.cumsum(0)[:-1]
    chunk_ids = torch.arange(total, dtype=torch.int32) - torch.repeat_interleave(
        starts, counts
    )
    return torch.stack([seq_ids, chunk_ids], dim=1)


def _offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.int32)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets


def build_gdn_chunked_prefill_meta(
    *,
    cu_seqlens_cpu: torch.Tensor,
    num_heads: int,
    device: torch.device,
    chunk_size: int = _GDN_CHUNK_SIZE,
    large_block_size: int = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
) -> GDNChunkedPrefillMetadata:
    device = torch.device(device) if isinstance(device, str) else device
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.int32, device="cpu")
    cumsum_block_size = compute_gdn_cumsum_block_size(num_heads, chunk_size)

    counts64 = _chunk_counts_cpu(cu_seqlens_cpu, chunk_size)
    counts_large = _chunk_counts_cpu(cu_seqlens_cpu, large_block_size)
    counts_cumsum = _chunk_counts_cpu(cu_seqlens_cpu, cumsum_block_size)

    ci64 = _build_chunk_indices(counts64)
    co64 = _offsets_from_counts(counts64)
    uco64 = _offsets_from_counts(counts64 + 1)
    fci64 = uco64[1:] - 1
    ci_large = _build_chunk_indices(counts_large)
    ci_cumsum = _build_chunk_indices(counts_cumsum)

    if device.type == "cpu":
        return GDNChunkedPrefillMetadata(
            chunk_indices_chunk64=ci64,
            chunk_offsets_chunk64=co64,
            update_chunk_offsets_chunk64=uco64,
            final_chunk_indices_chunk64=fci64,
            chunk_indices_large_block=ci_large,
            block_indices_cumsum=ci_cumsum,
            chunk_indices=ci64,
            chunk_offsets=co64,
        )

    sizes = [
        ci64.numel(),
        co64.numel(),
        uco64.numel(),
        fci64.numel(),
        ci_large.numel(),
        ci_cumsum.numel(),
    ]
    packed_cpu = torch.cat(
        [ci64.view(-1), co64, uco64, fci64, ci_large.view(-1), ci_cumsum.view(-1)]
    )
    packed_dev = packed_cpu.to(device=device, non_blocking=False)

    cursor = 0

    def take(numel: int, shape: torch.Size) -> torch.Tensor:
        nonlocal cursor
        tensor = packed_dev[cursor : cursor + numel].view(shape)
        cursor += numel
        return tensor

    ci64_d = take(sizes[0], ci64.shape)
    co64_d = take(sizes[1], co64.shape)
    uco64_d = take(sizes[2], uco64.shape)
    fci64_d = take(sizes[3], fci64.shape)
    ci_large_d = take(sizes[4], ci_large.shape)
    ci_cumsum_d = take(sizes[5], ci_cumsum.shape)

    return GDNChunkedPrefillMetadata(
        chunk_indices_chunk64=ci64_d,
        chunk_offsets_chunk64=co64_d,
        update_chunk_offsets_chunk64=uco64_d,
        final_chunk_indices_chunk64=fci64_d,
        chunk_indices_large_block=ci_large_d,
        block_indices_cumsum=ci_cumsum_d,
        chunk_indices=ci64_d,
        chunk_offsets=co64_d,
    )


class GDNChunkedPrefillCache:
    def __init__(
        self,
        num_heads: int,
        chunk_size: int = _GDN_CHUNK_SIZE,
        large_block_size: int = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
    ) -> None:
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.large_block_size = large_block_size
        self._cached_key: Optional[bytes] = None
        self._cached_meta: Optional[GDNChunkedPrefillMetadata] = None

    def _key(self, cu_seqlens_cpu: torch.Tensor) -> bytes:
        return cu_seqlens_cpu.contiguous().numpy().tobytes()

    def get_or_build(
        self, cu_seqlens_cpu: torch.Tensor, device: torch.device
    ) -> GDNChunkedPrefillMetadata:
        cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.int32, device="cpu")
        key = self._key(cu_seqlens_cpu)
        if key == self._cached_key and self._cached_meta is not None:
            return self._cached_meta

        meta = build_gdn_chunked_prefill_meta(
            cu_seqlens_cpu=cu_seqlens_cpu,
            num_heads=self.num_heads,
            device=device,
            chunk_size=self.chunk_size,
            large_block_size=self.large_block_size,
        )
        self._cached_key = key
        self._cached_meta = meta
        return meta
