import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
META_PATH = (
    ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "linear"
    / "gdn_chunk_meta.py"
)

spec = importlib.util.spec_from_file_location("gdn_chunk_meta", META_PATH)
gdn_chunk_meta = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gdn_chunk_meta)

GDNChunkedPrefillCache = gdn_chunk_meta.GDNChunkedPrefillCache
GDNChunkedPrefillMetadata = gdn_chunk_meta.GDNChunkedPrefillMetadata
build_gdn_chunked_prefill_meta = gdn_chunk_meta.build_gdn_chunked_prefill_meta
compute_gdn_cumsum_block_size = gdn_chunk_meta.compute_gdn_cumsum_block_size


def _cu(seq_lens: list[int]) -> torch.Tensor:
    out = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    if seq_lens:
        out[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    return out


def _legacy_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    pairs: list[list[int]] = []
    for seq_idx, seq_len in enumerate((cu_seqlens[1:] - cu_seqlens[:-1]).tolist()):
        for chunk_idx in range((seq_len + chunk_size - 1) // chunk_size):
            pairs.append([seq_idx, chunk_idx])
    if not pairs:
        return torch.empty((0, 2), dtype=torch.int32)
    return torch.tensor(pairs, dtype=torch.int32)


def _legacy_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")
    out = torch.zeros(counts.numel() + 1, dtype=torch.int32)
    torch.cumsum(counts, dim=0, out=out[1:])
    return out


def _legacy_update_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor") + 1
    out = torch.zeros(counts.numel() + 1, dtype=torch.int32)
    torch.cumsum(counts, dim=0, out=out[1:])
    return out


def test_build_gdn_chunked_prefill_meta_matches_legacy_indices():
    cu = _cu([0, 1, 64, 65, 128, 129])
    meta = build_gdn_chunked_prefill_meta(
        cu_seqlens_cpu=cu,
        num_heads=8,
        device=torch.device("cpu"),
    )
    assert isinstance(meta, GDNChunkedPrefillMetadata)
    assert torch.equal(meta.chunk_indices_chunk64, _legacy_chunk_indices(cu, 64))
    assert torch.equal(meta.chunk_offsets_chunk64, _legacy_chunk_offsets(cu, 64))
    assert torch.equal(
        meta.update_chunk_offsets_chunk64, _legacy_update_offsets(cu, 64)
    )
    assert torch.equal(
        meta.final_chunk_indices_chunk64,
        _legacy_update_offsets(cu, 64)[1:] - 1,
    )
    assert torch.equal(meta.chunk_indices, meta.chunk_indices_chunk64)
    assert torch.equal(meta.chunk_offsets, meta.chunk_offsets_chunk64)


def test_build_gdn_chunked_prefill_meta_uses_cumsum_working_set_formula():
    assert compute_gdn_cumsum_block_size(num_heads=8) == 512
    assert compute_gdn_cumsum_block_size(num_heads=32) == 128
    assert compute_gdn_cumsum_block_size(num_heads=4096) == 1
    cu = _cu([256, 1024])
    meta = build_gdn_chunked_prefill_meta(
        cu_seqlens_cpu=cu,
        num_heads=32,
        device=torch.device("cpu"),
    )
    assert torch.equal(meta.block_indices_cumsum, _legacy_chunk_indices(cu, 128))


def test_cache_returns_same_object_on_identical_cu_seqlens():
    cu = _cu([4, 8, 96])
    cache = GDNChunkedPrefillCache(num_heads=16)
    first = cache.get_or_build(cu, torch.device("cpu"))
    second = cache.get_or_build(cu, torch.device("cpu"))
    assert first is second


def test_cache_rebuilds_on_different_cu_seqlens():
    cache = GDNChunkedPrefillCache(num_heads=16)
    first = cache.get_or_build(_cu([4, 8]), torch.device("cpu"))
    second = cache.get_or_build(_cu([4, 16]), torch.device("cpu"))
    assert first is not second
