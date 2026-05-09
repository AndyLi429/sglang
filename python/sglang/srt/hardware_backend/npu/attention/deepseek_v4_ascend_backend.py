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
        # The CUDA backend hardcodes ``page_size == 256``; on NPU we run with
        # page_size = 128 (matching the rest of the ascend backend), so we
        # explicitly skip that assert.

    # ------------------------------------------------------------------
    # V4-specific metadata + dispatch — all stubbed pending real impls.
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        # AscendAttnBackend computes core metadata. V4 needs the additional
        # DSV4Metadata wrapper (DSV4AttnMetadata + indexer + compress
        # metadata). For now delegate to the base; the moment V4 model code
        # reads ``self.forward_metadata.core_metadata`` we'll have to layer
        # the wrapper on top.
        super().init_forward_metadata(forward_batch)

    def init_forward_metadata_indexer(self, core_attn_metadata):
        _stub("init_forward_metadata_indexer")

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
        if compress_ratio not in (0, 4, 128):
            raise ValueError(
                f"V4 attention expects compress_ratio in (0, 4, 128); got {compress_ratio}"
            )
        if compress_ratio == 0:
            # Regular MQA layer — delegate to ascend's forward. ``attn_sink``
            # is V4-only and not exposed on the ascend signature; if the V4
            # model passes it on a compress_ratio==0 layer we drop it for
            # now (TODO: verify this is safe; CUDA path uses attn_sink
            # whenever it isn't None).
            return AscendAttnBackend.forward(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
        _stub(f"forward(compress_ratio={compress_ratio})")

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        _stub("store_cache")

    # ``forward_compress`` and ``forward_core_compressor`` come from
    # CompressorBackendMixin and call CUDA JIT kernels (compress_forward,
    # compress_fused_norm_rope_inplace, linear_bf16_fp32). Until we wire NPU
    # equivalents we want a clear error rather than a confusing one from
    # inside JIT compile.

    def forward_compress(self, *args, **kwargs):  # type: ignore[override]
        _stub("forward_compress")

    def forward_core_compressor(self, *args, **kwargs):  # type: ignore[override]
        _stub("forward_core_compressor")

    def forward_c4_indexer(self, *args, **kwargs):  # type: ignore[override]
        _stub("forward_c4_indexer")
