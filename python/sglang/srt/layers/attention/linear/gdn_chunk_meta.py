from __future__ import annotations

from dataclasses import dataclass

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


def _fill_chunk_indices(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    cursor = 0
    for seq_idx, num_chunks in enumerate(chunk_counts.tolist()):
        if num_chunks <= 0:
            continue
        out[cursor : cursor + num_chunks, 0].fill_(seq_idx)
        out[cursor : cursor + num_chunks, 1] = torch.arange(num_chunks, dtype=out.dtype)
        cursor += num_chunks
    return cursor


def _fill_chunk_offsets(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts, dim=0, out=out[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_update_chunk_offsets(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_final_chunk_indices(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out[: chunk_counts.numel()])
        out[: chunk_counts.numel()].sub_(1)
    return chunk_counts.numel()


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return tensor
    return tensor.to(device=device, non_blocking=True)


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
    counts_chunk64 = _chunk_counts_cpu(cu_seqlens_cpu, chunk_size)
    counts_large = _chunk_counts_cpu(cu_seqlens_cpu, large_block_size)
    counts_cumsum = _chunk_counts_cpu(
        cu_seqlens_cpu, compute_gdn_cumsum_block_size(num_heads, chunk_size)
    )
    num_seqs = counts_chunk64.numel()

    chunk_indices_chunk64 = torch.empty(
        (int(counts_chunk64.sum().item()), 2), dtype=torch.int32
    )
    chunk_offsets_chunk64 = torch.empty((num_seqs + 1,), dtype=torch.int32)
    update_chunk_offsets_chunk64 = torch.empty((num_seqs + 1,), dtype=torch.int32)
    final_chunk_indices_chunk64 = torch.empty((num_seqs,), dtype=torch.int32)
    chunk_indices_large_block = torch.empty(
        (int(counts_large.sum().item()), 2), dtype=torch.int32
    )
    block_indices_cumsum = torch.empty(
        (int(counts_cumsum.sum().item()), 2), dtype=torch.int32
    )

    _fill_chunk_indices(chunk_indices_chunk64, counts_chunk64)
    _fill_chunk_offsets(chunk_offsets_chunk64, counts_chunk64)
    _fill_update_chunk_offsets(update_chunk_offsets_chunk64, counts_chunk64)
    _fill_final_chunk_indices(final_chunk_indices_chunk64, counts_chunk64)
    _fill_chunk_indices(chunk_indices_large_block, counts_large)
    _fill_chunk_indices(block_indices_cumsum, counts_cumsum)

    chunk_indices_chunk64 = _to_device(chunk_indices_chunk64, device)
    chunk_offsets_chunk64 = _to_device(chunk_offsets_chunk64, device)
    return GDNChunkedPrefillMetadata(
        chunk_indices_chunk64=chunk_indices_chunk64,
        chunk_offsets_chunk64=chunk_offsets_chunk64,
        update_chunk_offsets_chunk64=_to_device(update_chunk_offsets_chunk64, device),
        final_chunk_indices_chunk64=_to_device(final_chunk_indices_chunk64, device),
        chunk_indices_large_block=_to_device(chunk_indices_large_block, device),
        block_indices_cumsum=_to_device(block_indices_cumsum, device),
        chunk_indices=chunk_indices_chunk64,
        chunk_offsets=chunk_offsets_chunk64,
    )


@dataclass
class _GDNChunkedPrefillBufferSlot:
    chunk_indices_chunk64_cpu: torch.Tensor
    chunk_offsets_chunk64_cpu: torch.Tensor
    update_chunk_offsets_chunk64_cpu: torch.Tensor
    final_chunk_indices_chunk64_cpu: torch.Tensor
    chunk_indices_large_block_cpu: torch.Tensor
    block_indices_cumsum_cpu: torch.Tensor
    chunk_indices_chunk64_dev: torch.Tensor
    chunk_offsets_chunk64_dev: torch.Tensor
    update_chunk_offsets_chunk64_dev: torch.Tensor
    final_chunk_indices_chunk64_dev: torch.Tensor
    chunk_indices_large_block_dev: torch.Tensor
    block_indices_cumsum_dev: torch.Tensor
    h2d_event: torch.npu.Event | None = None
    h2d_event_pending: bool = False


def _alloc_cpu_dev(shape, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pin_memory = device.type != "cpu"
    cpu = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
    dev = (
        cpu
        if device.type == "cpu"
        else torch.empty(shape, dtype=torch.int32, device=device)
    )
    return cpu, dev


def _make_slot(
    device: torch.device, max_num_batched_tokens: int, max_num_seqs: int
) -> _GDNChunkedPrefillBufferSlot:
    ci_cpu, ci_dev = _alloc_cpu_dev((max_num_batched_tokens, 2), device)
    co_cpu, co_dev = _alloc_cpu_dev((max_num_seqs + 1,), device)
    uco_cpu, uco_dev = _alloc_cpu_dev((max_num_seqs + 1,), device)
    fci_cpu, fci_dev = _alloc_cpu_dev((max_num_seqs,), device)
    large_cpu, large_dev = _alloc_cpu_dev((max_num_batched_tokens, 2), device)
    cumsum_cpu, cumsum_dev = _alloc_cpu_dev((max_num_batched_tokens, 2), device)
    return _GDNChunkedPrefillBufferSlot(
        chunk_indices_chunk64_cpu=ci_cpu,
        chunk_offsets_chunk64_cpu=co_cpu,
        update_chunk_offsets_chunk64_cpu=uco_cpu,
        final_chunk_indices_chunk64_cpu=fci_cpu,
        chunk_indices_large_block_cpu=large_cpu,
        block_indices_cumsum_cpu=cumsum_cpu,
        chunk_indices_chunk64_dev=ci_dev,
        chunk_offsets_chunk64_dev=co_dev,
        update_chunk_offsets_chunk64_dev=uco_dev,
        final_chunk_indices_chunk64_dev=fci_dev,
        chunk_indices_large_block_dev=large_dev,
        block_indices_cumsum_dev=cumsum_dev,
        h2d_event=torch.npu.Event() if device.type == "npu" else None,
    )


class GDNChunkedPrefillBufferPool:
    def __init__(
        self,
        *,
        device: torch.device,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        num_heads: int,
        chunk_size: int = _GDN_CHUNK_SIZE,
        large_block_size: int = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.large_block_size = large_block_size
        self.cumsum_block_size = compute_gdn_cumsum_block_size(num_heads, chunk_size)
        self._idx = -1
        self._slots = [
            _make_slot(self.device, max_num_batched_tokens, max_num_seqs),
            _make_slot(self.device, max_num_batched_tokens, max_num_seqs),
        ]

    def build(self, cu_seqlens_cpu: torch.Tensor) -> GDNChunkedPrefillMetadata:
        if self.device.type == "cpu":
            return build_gdn_chunked_prefill_meta(
                cu_seqlens_cpu=cu_seqlens_cpu,
                num_heads=self.num_heads,
                device=self.device,
                chunk_size=self.chunk_size,
                large_block_size=self.large_block_size,
            )

        self._idx = (self._idx + 1) % len(self._slots)
        slot = self._slots[self._idx]

        # Wait for previous H2D to complete before overwriting pinned CPU buffers
        if slot.h2d_event_pending:
            slot.h2d_event.wait()
            slot.h2d_event_pending = False

        cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.int32, device="cpu")
        counts_chunk64 = _chunk_counts_cpu(cu_seqlens_cpu, self.chunk_size)
        counts_large = _chunk_counts_cpu(cu_seqlens_cpu, self.large_block_size)
        counts_cumsum = _chunk_counts_cpu(cu_seqlens_cpu, self.cumsum_block_size)

        n_ci = _fill_chunk_indices(slot.chunk_indices_chunk64_cpu, counts_chunk64)
        n_co = _fill_chunk_offsets(slot.chunk_offsets_chunk64_cpu, counts_chunk64)
        n_uco = _fill_update_chunk_offsets(
            slot.update_chunk_offsets_chunk64_cpu, counts_chunk64
        )
        n_fci = _fill_final_chunk_indices(
            slot.final_chunk_indices_chunk64_cpu, counts_chunk64
        )
        n_large = _fill_chunk_indices(slot.chunk_indices_large_block_cpu, counts_large)
        n_cumsum = _fill_chunk_indices(slot.block_indices_cumsum_cpu, counts_cumsum)

        def copy(cpu: torch.Tensor, dev: torch.Tensor, n: int) -> torch.Tensor:
            view = dev[:n]
            view.copy_(cpu[:n], non_blocking=True)
            return view

        chunk_indices_chunk64 = copy(
            slot.chunk_indices_chunk64_cpu, slot.chunk_indices_chunk64_dev, n_ci
        )
        chunk_offsets_chunk64 = copy(
            slot.chunk_offsets_chunk64_cpu, slot.chunk_offsets_chunk64_dev, n_co
        )

        # Record H2D event so next reuse of this slot can wait
        if slot.h2d_event is not None:
            slot.h2d_event.record()
            slot.h2d_event_pending = True

        return GDNChunkedPrefillMetadata(
            chunk_indices_chunk64=chunk_indices_chunk64,
            chunk_offsets_chunk64=chunk_offsets_chunk64,
            update_chunk_offsets_chunk64=copy(
                slot.update_chunk_offsets_chunk64_cpu,
                slot.update_chunk_offsets_chunk64_dev,
                n_uco,
            ),
            final_chunk_indices_chunk64=copy(
                slot.final_chunk_indices_chunk64_cpu,
                slot.final_chunk_indices_chunk64_dev,
                n_fci,
            ),
            chunk_indices_large_block=copy(
                slot.chunk_indices_large_block_cpu,
                slot.chunk_indices_large_block_dev,
                n_large,
            ),
            block_indices_cumsum=copy(
                slot.block_indices_cumsum_cpu,
                slot.block_indices_cumsum_dev,
                n_cumsum,
            ),
            chunk_indices=chunk_indices_chunk64,
            chunk_offsets=chunk_offsets_chunk64,
        )
