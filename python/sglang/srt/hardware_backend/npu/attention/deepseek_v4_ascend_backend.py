"""DeepSeek V4 attention backend on Ascend NPU.

This bridges sgl-project/sglang's V4 model code (which expects a backend
that mixes ``CompressorBackendMixin`` + ``C4IndexerBackendMixin`` on top of
``AttentionBackend``) with ``AscendAttnBackend`` (the NPU implementation
that knows nothing about V4's c4/c128 compress paths). The CUDA reference
is ``DeepseekV4AttnBackend``; this class is its NPU counterpart.

Strategy:

* Inherit from ``AscendAttnBackend`` plus the two V4 mixins. The mixins
  give us ``forward_compress`` / ``forward_core_compressor`` / ``forward_c4_indexer``
  signatures the model calls. Their default implementations call CUDA JIT
  kernels (``compress_forward``, ``compress_fused_norm_rope_inplace``,
  ``act_quant``, ``rotate_activation``, etc.); on NPU each of these has to
  be replaced with an ATB / torch_npu / pure-torch equivalent. We override
  one method at a time as we hit them at runtime.

* ``init_forward_metadata`` has to compute both the regular ascend metadata
  and the V4 ``DSV4Metadata`` with ``DSV4AttnMetadata`` + indexer metadata
  + c4/c128 compress metadata. We delegate the ascend half and add a thin
  V4 layer on top.

* ``forward()`` accepts V4-specific kwargs (``compress_ratio``, ``attn_sink``,
  ``save_kv_cache``). For ``compress_ratio==0`` (regular MQA layers) we
  delegate to ``AscendAttnBackend.forward``; for 4 / 128 we have to route
  to the c4 / c128 sparse path.

This file deliberately leaves the harder methods unimplemented behind
``NotImplementedError`` with explicit messages — the goal is to surface
exact method names + arguments at first NPU forward, then fill them in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

# custom_ops registers torch.ops.custom.npu_* via side-effect on import.
# It's installed in the cann image at /usr/local/python*/site-packages/custom_ops.
# Without this import the npu_sparse_attn_sharedkv_metadata op isn't visible
# the first time we touch torch.ops.custom (lazy namespace population).
try:
    import custom_ops  # noqa: F401
except ImportError:
    logging.getLogger(__name__).warning(
        "custom_ops package not importable — V4 ascend attention will fall back."
    )

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.attention.dsv4.compressor import CompressorBackendMixin
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_prefill_cp_round_robin_split,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_round_robin_split,
    nsa_cp_round_robin_split_q_seqs,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_kv_cache

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _stub(method_name: str):
    raise NotImplementedError(
        f"DeepseekV4AscendAttnBackend.{method_name} is not implemented yet on NPU. "
        "The CUDA reference is in deepseek_v4_backend.py / dsv4/{compressor,indexer}.py; "
        "the NPU port has to either (a) call into torch_npu / ATB / sgl_kernel_npu "
        "for the corresponding fused op, or (b) provide a pure-torch fallback."
    )


def _build_hadamard_matrix(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Sylvester-construction Walsh-Hadamard matrix of size n × n.

    n must be a power of 2 (asserted by callers). Caches per (n, dtype, device)
    on the function so repeated calls within a forward batch don't rebuild.
    """
    cache = _build_hadamard_matrix._cache  # type: ignore[attr-defined]
    key = (n, dtype, str(device))
    if key in cache:
        return cache[key]
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.size(0) < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H.to(dtype=dtype, device=device).contiguous()
    cache[key] = H
    return H


_build_hadamard_matrix._cache = {}  # type: ignore[attr-defined]


def _compute_c4_q_npu(
    c4_indexer,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """NPU equivalent of ``C4Indexer.compute_q``.

    ``compute_q`` does:
        q, _ = wq_b(q_lora)
        q = q.view(-1, n_local_heads, head_dim)
        fused_rope(q[..., -rope_head_dim:], None, freqs_cis, positions=...)
        q = rotate_activation(q)            # triton hadamard_transform

    On NPU, ``fused_rope`` is a tvm_ffi CUDA kernel and ``rotate_activation``
    is a triton hadamard. Replace with ``_v4_rope_inplace_npu`` and a torch
    Walsh-Hadamard matmul. Note: Sylvester ordering may not match the triton
    kernel's ordering — final consumer (``npu_quant_lightning_indexer``) is
    insensitive to the basis since both q and k are rotated by the same H.
    """
    from sglang.srt.models.deepseek_v4 import _v4_rope_inplace_npu

    q, _ = c4_indexer.wq_b(q_lora)
    q = q.view(-1, c4_indexer.n_local_heads, c4_indexer.head_dim)
    _v4_rope_inplace_npu(
        q[..., -c4_indexer.rope_head_dim :],
        None,
        c4_indexer.freqs_cis,
        positions,
    )
    H = _build_hadamard_matrix(c4_indexer.head_dim, torch.float32, q.device)
    scale = c4_indexer.head_dim**-0.5
    q_f32 = q.to(torch.float32)
    q_rotated = torch.matmul(q_f32, H) * scale
    return q_rotated.to(torch.bfloat16)


class DeepseekV4AscendAttnBackend(
    AscendAttnBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    """V4 attention dispatcher for Ascend NPU.

    Method resolution order is intentional: AscendAttnBackend ships the
    NPU-side ``init_forward_metadata`` / ``forward_extend`` / ``forward_decode``
    surface; the V4 mixins only add the c4/c128 compress + c4 indexer
    helpers. When both define a method (e.g. ``forward``), MRO picks
    Ascend's, which is what we want for the regular MQA path.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        speculative_step_id: int = 0,
    ):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        # Pull the V4-specific config that compute_kernel_metadata needs.
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        cfg = model_runner.model_config
        self._dsv4_config = cfg
        tp_size = get_attention_tp_size()
        self._dsv4_q_head_num = cfg.num_attention_heads // tp_size
        self._dsv4_kv_head_num = 1  # V4 MQA / latent
        # V4-Flash config.json sets head_dim=512 directly (qk_nope_head_dim is
        # null in HF config); mirror iforgetmyname/dsv4_release which uses
        # self.config.head_dim verbatim for the metadata kernel arg.
        self._dsv4_head_dim = cfg.head_dim
        hf = getattr(cfg, "hf_config", cfg)
        self._dsv4_index_topk = getattr(hf, "index_topk", 512)
        self._dsv4_index_n_heads = getattr(hf, "index_n_heads", 64)
        self._dsv4_index_head_dim = getattr(hf, "index_head_dim", 128)
        self._dsv4_compress_ratios = getattr(hf, "compress_ratios", None)
        self._dsv4_has_c4 = (
            self._dsv4_compress_ratios is not None and 4 in self._dsv4_compress_ratios
        )
        self._dsv4_has_c128 = (
            self._dsv4_compress_ratios is not None and 128 in self._dsv4_compress_ratios
        )
        self._dsv4_sliding_window_size = (
            cfg.sliding_window_size if cfg.sliding_window_size is not None else 128
        )

    # ------------------------------------------------------------------
    # V4-specific metadata + dispatch — all stubbed pending real impls.
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        super().init_forward_metadata(forward_batch)
        fm = self.forward_metadata

        # Build TND cu_seqlens_q (= cumulative QUERY seq lens, int32 device tensor).
        # The kernel uses cu_seqlens_q to slice the q tensor by request, so
        # the per-request length here must equal the per-request token count
        # in q — NOT the KV/context length.
        #
        #   extend / prefill: q has extend_seq_lens_cpu tokens per request →
        #                     cumsum(extend_seq_lens_cpu).
        #   decode:           q has exactly 1 new token per request → [1, 1, ..., 1].
        #   target_verify /
        #   draft_extend:     q has speculative_num_draft_tokens per request.
        #
        # Earlier this branch fell back to `forward_batch.seq_lens_cpu` (the
        # full KV length) on the non-extend path, which made the kernel slice
        # q at offset = full_seq_len while q.shape[0] = batch_size for decode.
        # That is the V4-NPU root cause of token-1+ divergence — kernel
        # metadata says q has e.g. 257 tokens but q tensor only has 1.
        device = forward_batch.seq_lens.device
        # round-robin CP rewrites each rank's per-request Q length to
        # `ceil(extend_seq_len / cp_size)` (with the remainder distributed by
        # cp_rank). After the rewrite, len(q) on this rank equals
        # sum(round_robin_q_lens), so the kernel's cu_seqlens_q MUST be derived
        # from the rewritten lengths — using the original extend_seq_lens_cpu
        # here would point the kernel past the end of q. Mirrors what the
        # generic NSA backend does at nsa_backend.py:550-566.
        rr_active = (
            forward_batch.forward_mode.is_extend()
            and can_nsa_prefill_cp_round_robin_split(forward_batch)
        )
        if forward_batch.forward_mode.is_extend():
            if rr_active:
                extend_cpu_raw = forward_batch.extend_seq_lens_cpu
                if isinstance(extend_cpu_raw, list):
                    extend_cpu_t = torch.tensor(extend_cpu_raw, dtype=torch.int32)
                else:
                    extend_cpu_t = extend_cpu_raw.int()
                extend_dev = (
                    forward_batch.extend_seq_lens.to(torch.int32)
                    if forward_batch.extend_seq_lens is not None
                    else extend_cpu_t.to(device)
                )
                rr_q_lens_cpu, rr_q_lens, rr_bs_idx_cpu, rr_bs_idx = (
                    nsa_cp_round_robin_split_q_seqs(extend_cpu_t.tolist(), extend_dev)
                )
                # Stash for downstream paths (forward_compressed / indexer) that
                # need to know which batch indices survived this rank and what
                # the post-split lengths are.
                fm.rr_q_lens_cpu = rr_q_lens_cpu
                fm.rr_q_lens = rr_q_lens
                fm.rr_bs_idx_cpu = rr_bs_idx_cpu
                fm.rr_bs_idx = rr_bs_idx
                if len(rr_q_lens_cpu) == 0:
                    # This rank owns zero queries (e.g. very short seq with
                    # cp_size > seq_len). Build empty metadata; the model's
                    # downstream all-gather is responsible for reconstructing.
                    fm.actual_seq_lengths_q = torch.zeros(
                        0, dtype=torch.int32, device=device
                    )
                    fm.actual_seq_lengths_q_pa = torch.zeros(
                        1, dtype=torch.int32, device=device
                    )
                else:
                    actual_q = (
                        torch.tensor(rr_q_lens_cpu, dtype=torch.int32)
                        .cumsum(dim=0)
                        .int()
                        .to(device)
                    )
                    fm.actual_seq_lengths_q = actual_q
                    fm.actual_seq_lengths_q_pa = torch.cat(
                        [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                        dim=0,
                    )
            else:
                seq_lens_cpu = forward_batch.extend_seq_lens_cpu
                if isinstance(seq_lens_cpu, list):
                    seq_lens_cpu = torch.tensor(seq_lens_cpu, dtype=torch.int32)
                else:
                    seq_lens_cpu = seq_lens_cpu.int()
                actual_q = torch.cumsum(seq_lens_cpu, dim=0).int().to(device)
                fm.actual_seq_lengths_q = actual_q
                fm.actual_seq_lengths_q_pa = torch.cat(
                    [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                    dim=0,
                )
        elif forward_batch.forward_mode.is_decode():
            B = forward_batch.batch_size
            fm.actual_seq_lengths_q = torch.arange(
                1, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q_pa = torch.arange(
                0, B + 1, dtype=torch.int32, device=device
            )
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            B = forward_batch.batch_size
            from sglang.srt.utils.common import get_global_server_args

            n_draft = get_global_server_args().speculative_num_draft_tokens or 1
            actual_q = torch.arange(
                n_draft, B * n_draft + 1, n_draft, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )
        else:
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None

        # SWA page table -- populated by AscendAttnBackend when the model is
        # hybrid-SWA, else None. Aliased under the name forward_sparse uses.
        # Use explicit `is not None` check (not `or`) because
        # `bool(multi-element tensor)` raises.
        block_tables_swa = getattr(fm, "block_tables_swa", None)
        fm.swa_page_table = (
            block_tables_swa if block_tables_swa is not None else fm.block_tables
        )

        # actual_seq_lengths_kv defaults to None on main; the V4 metadata
        # kernel needs an int32 device tensor of per-request KV lengths.
        if fm.actual_seq_lengths_kv is None:
            if fm.seq_lens_cpu_int is not None:
                fm.actual_seq_lengths_kv = fm.seq_lens_cpu_int.to(
                    device=forward_batch.seq_lens.device, dtype=torch.int32
                )
            else:
                fm.actual_seq_lengths_kv = forward_batch.seq_lens.to(torch.int32)

        # Build kernel_metadata dict. For V4-Flash we mainly need c1a (no
        # compress KV) right now; c4a/c128a follow when we add those paths.
        fm.kernel_metadata = self._compute_kernel_metadata(forward_batch)

        # Step-3 NPU compress metadata: only built when forward_npu paths are
        # active (env-gated). Each field is a per-request tensor consumed by
        # dsv4/{compressor,indexer}.py forward_npu. See iforgetmyname/dsv4_
        # release ascend_backend.init_forward_metadata @ ~L735-790 for the
        # reference impl on top of pre-allocated req_to_token_c{N} tables;
        # main has no req_to_token_c{N}, so we compute equivalents on the
        # fly from req_to_token + the V4 KV pool's swa translation.
        from sglang.srt.environ import envs as _envs

        if _envs.SGLANG_DSV4_NPU_REAL_COMPRESSOR.get() and self._dsv4_compress_ratios:
            self._build_npu_compress_metadata(forward_batch)

    def _compute_kernel_metadata(self, forward_batch: "ForwardBatch") -> dict:
        fm = self.forward_metadata
        common = {
            "cu_seqlens_q": fm.actual_seq_lengths_q_pa,
            "seqused_kv": fm.actual_seq_lengths_kv,
            "cmp_ratio": 1,
            "ori_mask_mode": 4,  # sliding window
            "cmp_mask_mode": 3,  # causal
            "ori_win_left": self._dsv4_sliding_window_size - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }
        base_kwargs = {
            "batch_size": forward_batch.batch_size,
            "num_heads_q": self._dsv4_q_head_num,
            "num_heads_kv": self._dsv4_kv_head_num,
            "head_dim": self._dsv4_head_dim,
            "has_ori_kv": True,
            "has_cmp_kv": False,
        }
        c1a_kwargs = base_kwargs | common
        kernel_metadata = {
            "c1a_metadata": torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                **c1a_kwargs
            )
        }

        if self._dsv4_has_c4:
            c4a_overrides = {
                "cmp_ratio": 4,
                "has_cmp_kv": True,
                "cmp_topk": self._dsv4_index_topk,
            }
            c4a_kwargs = c1a_kwargs | c4a_overrides
            kernel_metadata["c4a_metadata"] = (
                torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c4a_kwargs)
            )

            # The lightning indexer is only attached to c4 layers.
            # Pass actual_seq_lengths_q (no leading 0, B-element cumsum)
            # exactly as iforgetmyname/dsv4_release builds it — a fresh
            # contiguous int32 device tensor, not a slice.
            actual_q = fm.actual_seq_lengths_q
            if actual_q is None:
                actual_q = fm.actual_seq_lengths_kv
            kernel_metadata["li_quant_metadata"] = (
                torch.ops.custom.npu_quant_lightning_indexer_metadata(
                    device=str(actual_q.device),
                    actual_seq_lengths_query=actual_q,
                    actual_seq_lengths_key=fm.actual_seq_lengths_kv,
                    layout_key="PA_BSND",
                    sparse_count=self._dsv4_index_topk,
                    sparse_mode=3,
                    layout_query="TND",
                    cmp_ratio=4,
                    key_quant_mode=0,
                    query_quant_mode=0,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                )
            )

        if self._dsv4_has_c128:
            c128a_overrides = {"cmp_ratio": 128, "has_cmp_kv": True}
            c128a_kwargs = c1a_kwargs | c128a_overrides
            kernel_metadata["c128a_metadata"] = (
                torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c128a_kwargs)
            )

        return kernel_metadata

    def _build_npu_compress_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Populate c{4,128}_{page_table,state_page_table,state_loc,loc} on
        forward_metadata for the NPU compressor / indexer forward_npu paths.

        Reference: iforgetmyname/dsv4_release ascend_backend.init_forward_metadata
        @ ~L735-790. iforgetmyname pre-allocates per-request mapping tables
        (req_to_token_c4 / req_to_token_c4_state) when the request enters the
        scheduler; main has no such tables, so we compute equivalents on the
        fly from req_to_token + the V4 KV pool's swa translation. This is
        slower but avoids cross-cutting allocator surgery on the request pool.
        """
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool = forward_batch.req_pool_indices
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        is_decode = forward_batch.forward_mode.is_decode()

        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
        n_pages = max(1, (seq_lens_max + self.page_size - 1) // self.page_size)

        # State page tables — for each request, for each page, the state-buffer
        # page index. Use the FIRST token of each page as the representative
        # (tokens within the same SWA page produce contiguous state-buffer slots).
        page_starts = torch.arange(
            0, n_pages * self.page_size, self.page_size, device=device
        )  # [n_pages]
        # [bs, n_pages] flattened token positions; positions past seq_len are
        # clamped to 0 (will be masked out by _get_kv_indices' kv_len).
        page_starts_2d = page_starts.unsqueeze(0).expand(bs, n_pages)
        # Index req_to_token: [bs, n_pages] of full-kv-pool slot ids.
        raw_loc = req_to_token[
            req_pool.unsqueeze(1).expand(-1, n_pages), page_starts_2d
        ]

        for ratio in self._dsv4_compress_ratios:
            if ratio not in (4, 128):
                continue
            # State page table — translate each (bs, n_pages) raw kv slot to a
            # state-buffer page id. translate_kv_loc_to_compress_state_loc gives
            # the flat state slot; divide by page_size for the page id.
            state_loc_2d = pool.translate_kv_loc_to_compress_state_loc(raw_loc, ratio)
            state_page_2d = (state_loc_2d // self.page_size).to(torch.int32)

            # State loc — single state-buffer slot for the new decode token.
            # In decode, out_cache_loc has shape [bs] (one new token per req).
            if is_decode:
                state_loc_decode = pool.translate_kv_loc_to_compress_state_loc(
                    forward_batch.out_cache_loc, ratio
                )
                # Compressor write loc — step 5c slab allocator. For each
                # request that just completed a ratio-aligned chunk, the new
                # compressed token writes to slot
                #   k_seq = seqlen_after // ratio - 1     (compressed seq pos)
                #   slot  = req_to_token_c{N}_pages[req_pool_idx, k_seq // page_size]
                #           * page_size + k_seq % page_size
                # Replaces the old `raw_out_loc // ratio` formula which only
                # worked when the request happened to land on a page-aligned
                # raw kv slot (= almost never).
                pages_table = pool.get_req_to_token_c_pages(ratio)
                should_compress = (seq_lens % ratio) == 0
                k_seq = (seq_lens.to(torch.int64) // ratio - 1).clamp(min=0)
                page_seq = (k_seq // self.page_size).to(torch.int64)
                offset = (k_seq % self.page_size).to(torch.int64)
                kernel_page = pages_table[req_pool.to(torch.int64), page_seq].to(
                    torch.int64
                )
                compress_out_loc = (kernel_page * self.page_size + offset).to(
                    torch.int32
                )
                compress_out_loc = torch.where(
                    should_compress,
                    compress_out_loc,
                    torch.zeros_like(compress_out_loc),
                )
            else:
                state_loc_decode = None
                compress_out_loc = None

            attr_state_pt = f"c{ratio}_state_page_table"
            attr_state_loc = f"c{ratio}_state_loc"
            attr_loc = f"c{ratio}_loc"
            setattr(fm, attr_state_pt, state_page_2d)
            setattr(fm, attr_state_loc, state_loc_decode)
            setattr(fm, attr_loc, compress_out_loc)

            # c{ratio}_page_table — kernel-view page table for c{N}_kv_pool.
            # Step 5c: read directly from the slab — gives each request its
            # own dedicated kernel pages so cmp_kv reads at compressed seq
            # pos 0..N-1 land in the right physical slots regardless of how
            # the raw_kv allocator scattered the request's full pages.
            pages_table = pool.get_req_to_token_c_pages(ratio)
            n_pages_c = (n_pages + ratio - 1) // ratio
            n_pages_c = max(1, min(n_pages_c, pages_table.shape[1]))
            c_page_table = pages_table[req_pool.to(torch.int64), :n_pages_c].to(
                torch.int32
            )
            setattr(fm, f"c{ratio}_page_table", c_page_table)

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # li_quant_metadata is computed inside _compute_kernel_metadata; nothing
        # extra to do here. Return None to satisfy the mixin contract.
        return None

    def _seed_c4_topk_indices(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Allocate a [T, index_topk] int32 tensor on the compute device,
        filled with -1 (= "no valid sparse index" sentinel that npu_sparse_
        attn_sharedkv accepts). Real ``forward_c4_indexer`` will overwrite the
        contents via ``npu_quant_lightning_indexer``; until then this lets the
        c4 path of ``_forward_compressed`` consume a well-shaped tensor."""
        if forward_batch.input_ids is not None:
            T = forward_batch.input_ids.shape[0]
        else:
            T = int(forward_batch.seq_lens.sum().item())
        return torch.full(
            (T, self._dsv4_index_topk),
            -1,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        *,
        compress_ratio: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if compress_ratio not in (0, 1, 4, 128):
            raise ValueError(
                f"V4 attention expects compress_ratio in (0, 1, 4, 128); got {compress_ratio}"
            )
        # Honor save_kv_cache=True contract. With SGLANG_OPT_USE_OVERLAP_STORE_CACHE
        # default TRUE, MQALayer._forward_prepare already writes K via store_cache
        # and passes save_kv_cache=False here (no dup-write). With overlap=False,
        # the previous code silently dropped the write — decode then read an
        # unwritten swa_kv_pool and produced garbage. Always respect the flag.
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if compress_ratio in (0, 1):
            return self._forward_dense(q, layer, forward_batch, attn_sink)
        # ratio 4 / 128 routing — TWO independent gates:
        #   SGLANG_DSV4_NPU_REAL_COMPRESSOR=1 turns on the in-module
        #     forward_npu (compressor writes real KV; output unchanged
        #     because attention still falls back to dense here).
        #   SGLANG_DSV4_NPU_SPARSE_ATTN=1 additionally routes attention
        #     through _forward_compressed (has_cmp_kv=True kernel path).
        # The second gate stays OFF by default until the kernel call's
        # size / sparse-indices mismatch is resolved; with it OFF, output
        # is bit-for-bit identical to the flag-OFF baseline.
        from sglang.srt.environ import envs as _envs

        sparse_on = _envs.SGLANG_DSV4_NPU_SPARSE_ATTN.get()
        c128_only = _envs.SGLANG_DSV4_NPU_SPARSE_ATTN_C128_ONLY.get()
        # Bisect mode: only c128 layers route to _forward_compressed.
        if c128_only and compress_ratio != 128:
            return self._forward_dense(q, layer, forward_batch, attn_sink)
        if sparse_on or c128_only:
            return self._forward_compressed(
                q, layer, forward_batch, attn_sink, compress_ratio
            )
        return self._forward_dense(q, layer, forward_batch, attn_sink)

    def _forward_dense(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """ratio=1 / ratio=0 dense layers — sliding-window attention via
        npu_sparse_attn_sharedkv with has_cmp_kv=False."""
        if (
            is_nsa_enable_prefill_cp()
            and nsa_use_prefill_cp(forward_batch)
            and forward_batch.attn_cp_metadata is not None
            and forward_batch.attn_cp_metadata.actual_seq_q_prev >= 0
        ):
            return self._forward_dense_cp(q, layer, forward_batch, attn_sink)
        # Round-robin CP dense path: rebuild a contiguous full-sequence KV
        # buffer (prefix + cp-all-gathered extend) and call the kernel with a
        # trivial temp page_table that just walks 0..num_pages-1. Each local
        # round-robin query becomes its own 1-query "batch" so the kernel's
        # per-batch seqused_kv encodes the correct global causal cutoff for
        # that query — without this, the per-batch (single-value) kv_len
        # would force every local query to see the same KV horizon.
        if (
            is_nsa_enable_prefill_cp()
            and nsa_use_prefill_cp(forward_batch)
            and is_nsa_prefill_cp_round_robin_split()
        ):
            return self._forward_dense_round_robin(q, layer, forward_batch, attn_sink)
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        ori_kv = pool.get_swa_buffer(layer.layer_id)  # (num_pages, page_size, 1, dim)

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=fm.swa_page_table,
            sinks=attn_sink,
            metadata=fm.kernel_metadata["c1a_metadata"],
            softmax_scale=layer.scaling,
        )
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def _forward_dense_cp(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CP-aware dense forward for zigzag-split prefill on Ascend NPU.

        Mirrors do_cp_balance_attn() from AscendAttnBackend but uses
        npu_sparse_attn_sharedkv (SWA paged kernel) instead of
        npu_fused_infer_attention_score (standard MLA kernel).

        Key difference from do_cp_balance_attn: the SWA buffer includes
        prefix tokens, so seqused_kv for each half must add prefix_len to
        kv_len_{prev,next} from attn_cp_metadata (which carries only the
        extend-token portion in NSA CP mode).
        """
        fm = self.forward_metadata
        cp_meta = forward_batch.attn_cp_metadata
        pool = forward_batch.token_to_kv_pool
        ori_kv = pool.get_swa_buffer(layer.layer_id)
        device = q.device

        # prefix_len = total KV in pool - extend tokens written this step.
        # actual_seq_lengths_kv[0] = prefix + extend (full seq len for batch=1).
        extend_cpu = forward_batch.extend_seq_lens_cpu
        extend_total = (
            sum(extend_cpu)
            if isinstance(extend_cpu, list)
            else int(extend_cpu.sum().item())
        )
        prefix_len = int(fm.actual_seq_lengths_kv[0].item()) - extend_total

        # In NSA CP mode, kv_len_{prev,next} counts only extend tokens visible
        # to each zigzag half (no prefix). Add prefix_len for SWA seqused_kv.
        kv_prev = prefix_len + cp_meta.kv_len_prev
        kv_next = prefix_len + cp_meta.kv_len_next

        # Split q into zigzag prev / next halves (same as do_cp_balance_attn).
        # torch.chunk gives ceil / floor halves matching actual_seq_q_prev/next.
        q_prev, q_next = torch.chunk(q, 2, dim=0)

        # Build or reuse cached per-half kernel metadata.
        # npu_sparse_attn_sharedkv requires a precomputed metadata struct;
        # we cache keyed by (T_q, kv_len) so it is only built once per fwd pass.
        if not hasattr(fm, "_cp_c1a_cache"):
            fm._cp_c1a_cache = {}

        def _get_meta(T_q: int, kv_len: int):
            key = (T_q, kv_len)
            if key not in fm._cp_c1a_cache:
                cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=device)
                kv_t = torch.tensor([kv_len], dtype=torch.int32, device=device)
                fm._cp_c1a_cache[key] = (
                    torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                        cu_seqlens_q=cu_q,
                        seqused_kv=kv_t,
                        cmp_ratio=1,
                        ori_mask_mode=4,
                        cmp_mask_mode=3,
                        ori_win_left=self._dsv4_sliding_window_size - 1,
                        ori_win_right=0,
                        layout_q="TND",
                        layout_kv="PA_ND",
                        batch_size=1,
                        num_heads_q=self._dsv4_q_head_num,
                        num_heads_kv=self._dsv4_kv_head_num,
                        head_dim=self._dsv4_head_dim,
                        has_ori_kv=True,
                        has_cmp_kv=False,
                    )
                )
            return fm._cp_c1a_cache[key]

        def _attn_half(q_half: torch.Tensor, T_q: int, kv_len: int) -> torch.Tensor:
            cu_q_pa = torch.tensor([0, T_q], dtype=torch.int32, device=device)
            kv_t = torch.tensor([kv_len], dtype=torch.int32, device=device)
            out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
                cu_seqlens_q=cu_q_pa,
                seqused_kv=kv_t,
                ori_mask_mode=4,
                ori_win_left=self._dsv4_sliding_window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                q=q_half,
                ori_kv=ori_kv,
                ori_block_table=fm.swa_page_table,
                sinks=attn_sink,
                metadata=_get_meta(T_q, kv_len),
                softmax_scale=layer.scaling,
            )
            return out

        out_prev = _attn_half(q_prev, cp_meta.actual_seq_q_prev, kv_prev)
        out_next = _attn_half(q_next, cp_meta.actual_seq_q_next, kv_next)
        return torch.cat([out_prev, out_next], dim=0)

    # ------------------------------------------------------------------
    # Round-robin CP helpers — Plan A: in-place KV gather inside the
    # backend. Each rank fetches its own extend KV from the local pool,
    # all-gathers + reinterleaves into global token order, prepends the
    # cached prefix (same on all ranks), and hands the kernel a fresh
    # contiguous (num_pages, page_size, 1, dim) buffer plus a trivial
    # page_table = [[0, 1, ..., num_pages-1]]. No KV pool surgery needed.
    # ------------------------------------------------------------------

    def _dsv4_npu_gather_full_kv_round_robin(
        self,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> tuple:
        """Build a (num_pages, page_size, 1, dim) global-order KV buffer and
        its trivial page_table for a round-robin-CP prefill step.

        Returns (full_kv_paged, page_table, full_seq_len, num_pages).

        Constraints (matches the rest of NSA CP):
          - batch_size == 1
          - prefix slots on every rank hold identical KV content (radix cache
            hit, or zero prefix)
          - this rank wrote its round-robin-local extend KV at
            forward_batch.out_cache_loc in pool.get_swa_buffer(layer_id)
        """
        from sglang.srt.layers.dp_attention import get_attention_cp_size
        from sglang.srt.utils.common import get_current_device_stream_fast

        pool = forward_batch.token_to_kv_pool
        cp_size = get_attention_cp_size()

        # Rank-local extend KV — flat-indexed by out_cache_loc.
        local_extend_kv = pool.get_swa_buffer(
            layer.layer_id, forward_batch.out_cache_loc
        )  # (T_local, 1, dim)

        # All-gather + reinterleave to global token order.
        gathered_extend_kv = cp_all_gather_rerange_kv_cache(
            local_extend_kv,
            cp_size,
            forward_batch,
            get_current_device_stream_fast(),
        )  # (extend_total, 1, dim) — global order

        # Prefix is identical on every rank (radix cache hit); fetch locally.
        extend_total_cpu = forward_batch.extend_seq_lens_cpu
        extend_total = (
            sum(extend_total_cpu)
            if isinstance(extend_total_cpu, list)
            else int(extend_total_cpu.sum().item())
        )
        full_seq_len = int(forward_batch.seq_lens_cpu[0].item())
        prefix_len = full_seq_len - extend_total

        if prefix_len > 0:
            req_to_token = forward_batch.req_to_token_pool.req_to_token
            req_idx = forward_batch.req_pool_indices[0]
            prefix_slots = req_to_token[req_idx, :prefix_len]
            prefix_kv = pool.get_swa_buffer(layer.layer_id, prefix_slots)
            full_kv = torch.cat([prefix_kv, gathered_extend_kv], dim=0)
        else:
            full_kv = gathered_extend_kv

        # Pad to page-aligned and reshape to PA_ND.
        page_size = pool.page_size
        num_pages = (full_seq_len + page_size - 1) // page_size
        padded = num_pages * page_size
        pad_n = padded - full_kv.shape[0]
        if pad_n > 0:
            pad_tail = full_kv.new_zeros(pad_n, *full_kv.shape[1:])
            full_kv = torch.cat([full_kv, pad_tail], dim=0)
        full_kv_paged = full_kv.view(num_pages, page_size, *full_kv.shape[1:])

        # Trivial page_table.
        page_table = torch.arange(
            num_pages, dtype=torch.int32, device=full_kv.device
        ).unsqueeze(0)
        return full_kv_paged, page_table, full_seq_len, num_pages

    def _forward_dense_round_robin(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dense (ratio=1) forward under round-robin CP.

        Treats each rank-local query as its own 1-query "batch" so the
        per-batch seqused_kv encodes the correct global causal cutoff:
        local query at rank-local index i corresponds to global position
        prefix_len + cp_rank + i*cp_size, and its valid KV horizon is
        prefix_len + cp_rank + i*cp_size + 1 tokens.
        """
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_rank,
            get_attention_cp_size,
        )

        fm = self.forward_metadata
        device = q.device
        cp_size = get_attention_cp_size()
        cp_rank = get_attention_cp_rank()

        full_kv_paged, page_table, full_seq_len, _ = (
            self._dsv4_npu_gather_full_kv_round_robin(layer, forward_batch)
        )

        # If this rank owns zero queries (very short seqs), short-circuit.
        T_local = q.shape[0]
        if T_local == 0:
            n_heads = q.shape[1] if q.ndim >= 2 else 1
            head_dim_v = getattr(layer, "v_head_dim", q.shape[-1])
            return q.new_zeros((0, n_heads, head_dim_v))

        extend_total_cpu = forward_batch.extend_seq_lens_cpu
        extend_total = (
            sum(extend_total_cpu)
            if isinstance(extend_total_cpu, list)
            else int(extend_total_cpu.sum().item())
        )
        prefix_len = full_seq_len - extend_total

        # Per-query KV horizons (causal): each local idx i → global pos
        #   gp(i) = prefix_len + cp_rank + i * cp_size
        # KV horizon = gp(i) + 1 tokens.
        local_idx = torch.arange(T_local, dtype=torch.int64, device=device)
        global_pos = prefix_len + cp_rank + local_idx * cp_size
        per_q_kv_len = (global_pos + 1).to(torch.int32)

        # cu_seqlens_q for T_local 1-query batches: [0, 1, 2, ..., T_local].
        cu_q = torch.arange(0, T_local + 1, dtype=torch.int32, device=device)

        # Per-call kernel metadata — bs = T_local, num_heads/head_dim
        # mirror the regular c1a metadata. Cache by T_local so we don't
        # rebuild across layers within the same forward step.
        if not hasattr(fm, "_cp_rr_c1a_cache"):
            fm._cp_rr_c1a_cache = {}
        meta_key = (T_local, full_seq_len)
        if meta_key not in fm._cp_rr_c1a_cache:
            fm._cp_rr_c1a_cache[meta_key] = (
                torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                    cu_seqlens_q=cu_q,
                    seqused_kv=per_q_kv_len,
                    cmp_ratio=1,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self._dsv4_sliding_window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    batch_size=T_local,
                    num_heads_q=self._dsv4_q_head_num,
                    num_heads_kv=self._dsv4_kv_head_num,
                    head_dim=self._dsv4_head_dim,
                    has_ori_kv=True,
                    has_cmp_kv=False,
                )
            )
        kernel_meta = fm._cp_rr_c1a_cache[meta_key]

        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
            cu_seqlens_q=cu_q,
            seqused_kv=per_q_kv_len,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=full_kv_paged,
            ori_block_table=page_table.expand(T_local, -1).contiguous(),
            sinks=attn_sink,
            metadata=kernel_meta,
            softmax_scale=layer.scaling,
        )
        return out

    def _forward_compressed_round_robin(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """Compressed (ratio=4 / ratio=128) forward under round-robin CP.

        Scope note: cmp_kv is currently produced by a stubbed compressor
        write path (see _forward_compressed docstring), so its contribution
        to the output is zero in the current implementation. This
        round-robin variant therefore wires the ori_kv (SWA) full-gather
        identically to _forward_dense_round_robin and passes a zero cmp_kv
        with cmp_sparse_indices=None, matching the non-CP compressed
        kernel call shape contract. Indexer-side all-gather for proper
        sparse top-k under CP is a follow-up — for now this returns the
        same value as the rank-replicated compressed path because cmp_kv
        is zero on every rank.
        """
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_rank,
            get_attention_cp_size,
        )

        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        device = q.device
        cp_size = get_attention_cp_size()
        cp_rank = get_attention_cp_rank()

        # Compressed buffer for this layer — used only for its dtype / dim
        # info to construct a matching zero buffer for the kernel. The
        # real values are zero anyway under the current compressor stub.
        cmp_kv_real = pool.get_compress_buffer(layer.layer_id, False)
        if cmp_kv_real is None:
            # Dense layer dispatched here by mistake; mirror the regular
            # _forward_compressed fallback.
            T = q.shape[0]
            n_heads = q.shape[1] if q.ndim >= 2 else 1
            head_dim_v = getattr(layer, "v_head_dim", q.shape[-1])
            return q.new_zeros((T, n_heads, head_dim_v))

        full_kv_paged, page_table, full_seq_len, _ = (
            self._dsv4_npu_gather_full_kv_round_robin(layer, forward_batch)
        )

        T_local = q.shape[0]
        if T_local == 0:
            n_heads = q.shape[1] if q.ndim >= 2 else 1
            head_dim_v = getattr(layer, "v_head_dim", q.shape[-1])
            return q.new_zeros((0, n_heads, head_dim_v))

        extend_total_cpu = forward_batch.extend_seq_lens_cpu
        extend_total = (
            sum(extend_total_cpu)
            if isinstance(extend_total_cpu, list)
            else int(extend_total_cpu.sum().item())
        )
        prefix_len = full_seq_len - extend_total

        local_idx = torch.arange(T_local, dtype=torch.int64, device=device)
        global_pos = prefix_len + cp_rank + local_idx * cp_size
        per_q_kv_len = (global_pos + 1).to(torch.int32)
        cu_q = torch.arange(0, T_local + 1, dtype=torch.int32, device=device)

        # cmp_kv: build a zero buffer that shares page_size with ori_kv
        # (the kernel requires this). Shape (num_pages, page_size, 1, dim).
        ori_page_size = full_kv_paged.shape[1]
        num_pages = full_kv_paged.shape[0]
        cmp_dim = cmp_kv_real.shape[-1] if cmp_kv_real.ndim >= 1 else 1
        cmp_kv_zero = q.new_zeros(
            num_pages, ori_page_size, 1, cmp_dim, dtype=cmp_kv_real.dtype
        )

        # Per-call kernel metadata (cached by T_local + full_seq_len).
        if not hasattr(fm, "_cp_rr_compressed_meta_cache"):
            fm._cp_rr_compressed_meta_cache = {}
        meta_key = (compress_ratio, T_local, full_seq_len)
        if meta_key not in fm._cp_rr_compressed_meta_cache:
            fm._cp_rr_compressed_meta_cache[meta_key] = (
                torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                    cu_seqlens_q=cu_q,
                    seqused_kv=per_q_kv_len,
                    cmp_ratio=compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self._dsv4_sliding_window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    batch_size=T_local,
                    num_heads_q=self._dsv4_q_head_num,
                    num_heads_kv=self._dsv4_kv_head_num,
                    head_dim=self._dsv4_head_dim,
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    cmp_topk=(self._dsv4_index_topk if compress_ratio == 4 else None),
                )
            )
        kernel_meta = fm._cp_rr_compressed_meta_cache[meta_key]

        attn_kwargs = dict(
            cu_seqlens_q=cu_q,
            seqused_kv=per_q_kv_len,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=full_kv_paged,
            ori_block_table=page_table.expand(T_local, -1).contiguous(),
            sinks=attn_sink,
            metadata=kernel_meta,
            softmax_scale=layer.scaling,
            cmp_ratio=compress_ratio,
            cmp_mask_mode=3,
            cmp_kv=cmp_kv_zero,
            cmp_block_table=page_table.expand(T_local, -1).contiguous(),
            cmp_sparse_indices=None,
        )
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def _forward_compressed(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """ratio=4 / ratio=128 layers — sliding-window + compressed-KV
        sparse attention via npu_sparse_attn_sharedkv with has_cmp_kv=True.

        cmp_kv (compressed KV) is read from the c4 / c128 pool buffer,
        which is currently zeros (compressor write path is still stubbed),
        so the compressed contribution to the output is zero. cmp_sparse_
        indices for c4 comes from forward_metadata.c4_topk_indices, which
        forward_c4_indexer currently seeds with -1 (= no valid sparse
        index) for the same reason. The point of this commit is to validate
        the kernel-call shape/dtype contract end-to-end before we land the
        compressor + indexer compute paths.
        """
        if (
            is_nsa_enable_prefill_cp()
            and nsa_use_prefill_cp(forward_batch)
            and is_nsa_prefill_cp_round_robin_split()
        ):
            return self._forward_compressed_round_robin(
                q, layer, forward_batch, attn_sink, compress_ratio
            )

        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        metadata = fm.kernel_metadata.get(f"c{compress_ratio}a_metadata")
        cmp_kv = pool.get_compress_buffer(layer.layer_id, False)

        if metadata is None or cmp_kv is None:
            # No metadata or no compress pool for this layer — likely means
            # the layer is dense (ratio 0/1) but the model dispatched here
            # by mistake, or the model has fewer compressed layers than the
            # config implies. Return zeros to avoid crashing the forward.
            T = q.shape[0]
            n_heads = q.shape[1] if q.ndim >= 2 else 1
            head_dim_v = getattr(layer, "v_head_dim", q.shape[-1])
            return q.new_zeros((T, n_heads, head_dim_v))

        ori_kv = pool.get_swa_buffer(layer.layer_id)

        # Reshape cmp_kv to share page_size with ori_kv before the kernel call.
        # main's V4 pool layout: c{N}_kv_pool buffer is (num_pages, page_size//
        # ratio, 1, dim) so each native page holds page_size//ratio compressed
        # tokens. The aclnn kernel expects cmp_kv to share its page_size with
        # ori_kv (=global page_size). We slice the buffer to a ratio-aligned
        # native-page count and view it as (N_kernel, global_page_size, 1, dim).
        #
        # cmp_block_table values: step 5c slab (`req_to_token_c{N}_pages`)
        # already gives kernel-view page indices in [0, N_kernel), so no
        # further `// page_ratio` divide is needed — the divide was a leftover
        # from step 5b when block_table came from raw kv pool page indices.
        ori_page_size = ori_kv.shape[1]
        cmp_native_page_size = cmp_kv.shape[1]
        cmp_block_table = getattr(
            fm, f"c{compress_ratio}_page_table", fm.swa_page_table
        )
        if cmp_native_page_size != ori_page_size:
            page_ratio = ori_page_size // cmp_native_page_size
            assert page_ratio == compress_ratio, (
                f"page_ratio={page_ratio} != compress_ratio={compress_ratio}; "
                "main's V4 pool keeps c{N}_native_page_size = global_page_size//ratio"
            )
            n_native = cmp_kv.shape[0]
            n_kernel = n_native // page_ratio
            cmp_kv = cmp_kv[: n_kernel * page_ratio].reshape(
                n_kernel, ori_page_size, *cmp_kv.shape[2:]
            )
            # Slab already in kernel-view page space — no divide.
            cmp_block_table = cmp_block_table.to(torch.int32)

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=fm.swa_page_table,
            sinks=attn_sink,
            metadata=metadata,
            softmax_scale=layer.scaling,
            cmp_ratio=compress_ratio,
            cmp_mask_mode=3,
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
        )
        # Step-5c diagnosis: route c4 with cmp_sparse_indices=None (= same
        # treatment as c128) when SGLANG_DSV4_NPU_SPARSE_C4_NO_TOPK is set.
        # This bypasses the -1 sentinel topk path that was used to "mask"
        # all c4 history, and instead lets the kernel use the entire
        # populated c4 history (up to seqused_kv // ratio compressed
        # tokens). If output stabilizes after this, the divergence we see
        # in step-5c is due to the kernel mis-handling -1 in the c4 sparse
        # indices tensor, not due to slab/cmp_kv layout. If output still
        # diverges from dense baseline, the issue is in compressor write
        # values (ape/wkv split) or in lingering pool state.
        from sglang.srt.environ import envs as _envs

        if compress_ratio == 4 and not _envs.SGLANG_DSV4_NPU_SPARSE_C4_NO_TOPK.get():
            topk = fm.c4_topk_indices
            if topk is None:
                topk = self._seed_c4_topk_indices(forward_batch)
                fm.c4_topk_indices = topk
            attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
        else:
            attn_kwargs["cmp_sparse_indices"] = None
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        """Write the SWA layer's K cache into the bf16 PA_ND buffer.

        ``swa_k`` arrives shaped (T, num_kv_heads=1, dim) where dim packs
        K_nope + K_rope in bf16 (same layout as get_swa_buffer returns).
        ``forward_batch.out_cache_loc`` is in FULL-pool index space (size
        = sum of all KV pools); the swa_kv_pool buffer is its own smaller
        space. We must translate full→swa first — otherwise the
        index_put hits the wrong slot (or wraps OOB), and decode reads
        garbage K back. This mirrors what the CUDA radix path does at
        set_swa_key_buffer_radix.
        """
        pool = forward_batch.token_to_kv_pool
        swa_loc = pool.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
        pool.set_swa_buffer(
            layer_id=layer_id,
            loc=swa_loc,
            cache=swa_k,
        )

    # PHASE-0 STUBS: all c4/c128 compressor / indexer paths are no-ops
    # while we surface the full forward chain. attention forward already
    # returns zeros for compress_ratio in (4, 128) (see forward()), so
    # whatever these compute would only feed a zero attention anyway.
    # The real impl of these (porting iforgetmyname's compressor/indexer
    # NPU kernels onto main's KV pool layout) is the bulk of the V4-NPU
    # attention port and lives behind these stubs.

    def forward_compress(self, *args, **kwargs):  # type: ignore[override]
        return None

    def forward_core_compressor(  # type: ignore[override]
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        layer_id: int,
        compressor,
    ) -> None:
        """Run the OUTER attention compressor on NPU.

        On CUDA, ``CompressorBackendMixin.forward_core_compressor`` calls
        ``compressor(x, forward_batch)`` (which produces compressed kv) and
        then writes the result via ``token_to_kv_pool.set_extra_key_buffer*``.
        On NPU, ``Compressor.forward_npu`` does the write inline (calls
        ``set_compress_buffer`` and ``set_compress_state_buffer`` itself), so
        we just trigger the compressor call and return — no separate set-
        buffer step. Gated by SGLANG_DSV4_NPU_REAL_COMPRESSOR; flag off keeps
        the previous stub (compressor never invoked, c4/c128 layers fall
        back to dense SWA in forward()).
        """
        if forward_batch.forward_mode.is_idle():
            return
        from sglang.srt.environ import envs as _envs

        if not _envs.SGLANG_DSV4_NPU_REAL_COMPRESSOR.get():
            return
        compressor(x, forward_batch)

    def forward_c4_indexer(  # type: ignore[override]
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: "ForwardBatch",
        c4_indexer=None,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
    ) -> None:
        """Wire up ``forward_metadata.c4_topk_indices`` for c4 sparse attention.

        Stage 1 (this commit): seed ``c4_topk_indices`` with -1 sentinel so
        downstream ``_forward_compressed`` (when implemented for ratio=4) can
        read a well-shaped tensor. The real NPU compute path needs:
          1. q from ``c4_indexer.wq_b(q_lora)`` + rope + hadamard rotation
             (``compute_q`` in the model uses the tvm_ffi ``fused_rope``; on
             NPU we need to inline ``_v4_rope_inplace_npu`` + a torch hadamard)
          2. weights from ``c4_indexer.weights_proj(x)``
          3. indexer-K cache (currently absent — comes from the c4 indexer
             compressor write path which is also stubbed)
          4. ``torch_npu.npu_dynamic_quant`` for q quantization
          5. ``torch.ops.custom.npu_quant_lightning_indexer`` to produce the
             real top-k indices
        Each piece needs its own commit + 217 relaunch verification.
        """
        if forward_batch.forward_mode.is_idle():
            return
        # Stage 1 baseline: just seed c4_topk_indices=-1 sentinel for
        # _forward_compressed to read. Real path requires forking main's
        # Compressor / C4Indexer modules with NPU-style self-contained
        # impl (see ascend ref iforgetmyname/dsv4_release nsa_indexer.py
        # Compressor.forward_ori @ L241 — wkv + wgate + ape weighted sum
        # + norm + rope + write KV pool, all in-module, no backend
        # delegation). Stage 2A-I exploration (wq_b call + various store
        # patterns) showed the issue isn't the wq_b op itself but the
        # architectural mismatch — main's forward_compress is a triton
        # mixin path with no NPU equivalent; we must fork the model
        # modules, not the backend mixin.
        self.forward_metadata.c4_topk_indices = self._seed_c4_topk_indices(
            forward_batch
        )
