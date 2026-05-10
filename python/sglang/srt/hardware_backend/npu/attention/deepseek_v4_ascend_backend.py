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

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.attention.dsv4.compressor import CompressorBackendMixin
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin

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
        self._dsv4_head_dim = (
            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        )  # 448 + 64 = 512
        self._dsv4_sliding_window_size = (
            cfg.sliding_window_size
            if cfg.sliding_window_size is not None
            else 128
        )

    # ------------------------------------------------------------------
    # V4-specific metadata + dispatch — all stubbed pending real impls.
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        super().init_forward_metadata(forward_batch)
        fm = self.forward_metadata

        # Build TND cu_seqlens_q (= cumulative seq lens, int32 device tensor).
        # AscendAttnBackend already populates seq_lens_list_cumsum for the
        # extend / prefill path; reuse where available.
        if forward_batch.forward_mode.is_extend():
            seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        else:
            seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None:
            cumsum = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    torch.cumsum(seq_lens_cpu.int(), dim=0).int(),
                ]
            ).to(forward_batch.seq_lens.device)
            fm.actual_seq_lengths_q_pa = cumsum
        else:
            fm.actual_seq_lengths_q_pa = None

        # SWA page table — already populated by AscendAttnBackend when the
        # model is hybrid-SWA. Alias it under the name forward_sparse uses.
        fm.swa_page_table = getattr(fm, "block_tables_swa", None) or fm.block_tables

        # Build kernel_metadata dict. For V4-Flash we mainly need c1a (no
        # compress KV) right now; c4a/c128a follow when we add those paths.
        fm.kernel_metadata = self._compute_kernel_metadata(forward_batch)

    def _compute_kernel_metadata(self, forward_batch: "ForwardBatch") -> dict:
        fm = self.forward_metadata
        common = {
            "cu_seqlens_q": fm.actual_seq_lengths_q_pa,
            "seqused_kv": fm.actual_seq_lengths_kv,
            "cmp_ratio": 1,  # placeholder; per-meta call overrides
            "ori_mask_mode": 4,  # sliding window
            "cmp_mask_mode": 3,  # causal
            "ori_win_left": self._dsv4_sliding_window_size - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }
        c1a_kwargs = {
            "batch_size": forward_batch.batch_size,
            "num_heads_q": self._dsv4_q_head_num,
            "num_heads_kv": self._dsv4_kv_head_num,
            "head_dim": self._dsv4_head_dim,
            "has_ori_kv": True,
            "has_cmp_kv": False,
        }
        c1a_kwargs.update(common)
        c1a_metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c1a_kwargs)
        return {"c1a_metadata": c1a_metadata}

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # PHASE-0: no metadata for the indexer (we no-op forward_c4_indexer).
        return None

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
        if compress_ratio in (0, 1):
            return self._forward_dense(q, layer, forward_batch, attn_sink)
        # 4 / 128 still stubbed — return zeros until compressor + indexer +
        # sparse-attn-with-cmp-kv are wired up.
        T = q.shape[0]
        n_heads = q.shape[1] if q.ndim >= 2 else 1
        head_dim_v = getattr(layer, "v_head_dim", q.shape[-1])
        return q.new_zeros((T, n_heads, head_dim_v))

    def _forward_dense(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """ratio=1 / ratio=0 dense layers — sliding-window attention via
        npu_sparse_attn_sharedkv with has_cmp_kv=False."""
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

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        """Write the SWA layer's K cache into the bf16 PA_ND buffer.

        ``swa_k`` arrives shaped (T, num_kv_heads=1, dim) where dim packs
        K_nope + K_rope in bf16 (same layout as get_swa_buffer returns).
        We use forward_batch.out_cache_loc as the per-token write
        positions — those map to flat (page * page_size + slot) indices
        on the swa_kv_pool buffer.
        """
        loc = forward_batch.out_cache_loc
        forward_batch.token_to_kv_pool.set_swa_buffer(
            layer_id=layer_id,
            loc=loc,
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

    def forward_core_compressor(self, *args, **kwargs):  # type: ignore[override]
        return None

    def forward_c4_indexer(self, *args, **kwargs):  # type: ignore[override]
        return None
