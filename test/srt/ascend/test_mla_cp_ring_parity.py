"""Ascend npu_ring_mla numerical parity for MLA prefill context parallelism.

Validates the rank-local zigzag-split ring-MLA path
(``AscendAttnBackend.do_cp_attn_mla`` + ``_ring_mla_mask_nomask`` +
``_build_mla_cp_index_metadata``) against a single non-CP ``npu_ring_mla`` over
the full sequence. Because both sides use the same kernel, this isolates the
correctness of the zigzag split and the shared CP metadata -> ring-MLA index /
seqlen mapping from kernel numerics.

Mirrors ``test/registered/kernels/test_mla_cp_fa3_parity.py`` (the FA3 / CUDA
equivalent), but for the Ascend backend. Single process, single layer; loops
over every CP rank and reassembles the full output.

Run:
    python3 test/srt/ascend/test_mla_cp_ring_parity.py
"""

import unittest
from types import SimpleNamespace

import torch

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    _HAS_NPU = False

if _HAS_NPU:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
        AscendAttnMaskBuilder,
    )
    from sglang.srt.layers.utils.cp_utils import ContextParallelMetadata

DEVICE = "npu"
DTYPE = torch.bfloat16

# DeepSeek V3/R1 MHA-prefill MLA dims (small head count for a fast test).
NUM_HEADS = 4
V_HEAD_DIM = 128  # == qk_nope_head_dim for DeepSeek
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = V_HEAD_DIM + QK_ROPE_HEAD_DIM
SCALING = QK_HEAD_DIM**-0.5


def _make_layer():
    return SimpleNamespace(
        tp_q_head_num=NUM_HEADS,
        tp_k_head_num=NUM_HEADS,
        tp_v_head_num=NUM_HEADS,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        scaling=SCALING,
    )


def _make_backend_stub():
    """A minimal object exposing exactly what the do_cp_attn_mla methods read."""
    stub = SimpleNamespace()
    stub.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    # 512x512 compressed triangular mask, same as AscendAttnMaskBuilder.
    stub.ringmla_mask = AscendAttnMaskBuilder.generate_attn_mask(
        512, "norm", torch.bfloat16
    ).to(DEVICE)
    # Bind the unbound methods to the stub.
    stub._build_mla_cp_index_metadata = AscendAttnBackend._build_mla_cp_index_metadata
    stub._ring_mla_mask_nomask = AscendAttnBackend._ring_mla_mask_nomask.__get__(stub)
    stub.do_cp_attn_mla = AscendAttnBackend.do_cp_attn_mla.__get__(stub)
    return stub


def _full_ring_mla(q, k, v, layer):
    """Non-CP causal MLA prefill over the full sequence: a single first-ring
    npu_ring_mla with a triangular mask. This is the reference."""
    seq_len = q.shape[0]
    q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    q_nope, q_pe = q.split([layer.v_head_dim, QK_ROPE_HEAD_DIM], dim=-1)
    k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    k_nope, k_pe = k.split([layer.v_head_dim, QK_ROPE_HEAD_DIM], dim=-1)
    v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    out = torch.zeros(
        seq_len, layer.tp_q_head_num, layer.v_head_dim, dtype=q.dtype, device=DEVICE
    )
    lse = torch.zeros(layer.tp_q_head_num, seq_len, dtype=torch.float32, device=DEVICE)
    mask = AscendAttnMaskBuilder.generate_attn_mask(512, "norm", torch.bfloat16).to(
        DEVICE
    )
    torch_npu.atb.npu_ring_mla(
        q_nope=q_nope.contiguous(),
        q_rope=q_pe.contiguous(),
        k_nope=k_nope,
        k_rope=k_pe,
        value=v,
        mask=mask,
        seqlen=torch.tensor([seq_len], dtype=torch.int32),
        head_num=layer.tp_q_head_num,
        kv_head_num=layer.tp_k_head_num,
        pre_out=None,
        prev_lse=None,
        qk_scale=layer.scaling,
        kernel_type="kernel_type_high_precision",
        mask_type="mask_type_triu",
        calc_type="calc_type_first_ring",
        output=out,
        softmax_lse=lse,
    )
    return out.reshape(seq_len, layer.tp_q_head_num * layer.v_head_dim)


def _block_layout(seq_len, cp_size):
    """Original-position [start, len] of each of the 2*cp_size zigzag blocks."""
    seg = cp_size * 2
    base, rem = seq_len // seg, seq_len % seg
    sizes = [base + 1 if i < rem else base for i in range(seg)]
    starts, acc = [], 0
    for s in sizes:
        starts.append(acc)
        acc += s
    return sizes, starts


def _cp_metadata(sizes, starts, cp_rank, cp_size):
    """Hand-build the bs=1 ContextParallelMetadata fields that the ring-MLA CP
    path consumes, mirroring prepare_context_parallel_metadata (prefix-free) but
    without the global-server-args dependency."""
    seg = cp_size * 2
    prev_b, next_b = cp_rank, seg - 1 - cp_rank
    kv_len_prev = starts[prev_b] + sizes[prev_b]
    kv_len_next = starts[next_b] + sizes[next_b]
    return ContextParallelMetadata(
        split_list=list(sizes),
        bs=1,
        kv_len_prev_list=[kv_len_prev],
        kv_len_next_list=[kv_len_next],
        actual_seq_q_prev_list=[sizes[prev_b]],
        actual_seq_q_next_list=[sizes[next_b]],
        total_q_prev_tokens=sizes[prev_b],
    )


class TestMlaCpRingParity(unittest.TestCase):
    @unittest.skipUnless(_HAS_NPU, "Ascend NPU required")
    def test_zigzag_split_matches_full_sequence(self):
        torch.manual_seed(0)
        layer = _make_layer()
        backend = _make_backend_stub()

        for cp_size in (2, 4):
            for seq_len in (64, 130):  # divisible and non-divisible by 2*cp_size
                with self.subTest(cp_size=cp_size, seq_len=seq_len):
                    q = torch.randn(
                        seq_len, NUM_HEADS, QK_HEAD_DIM, dtype=DTYPE, device=DEVICE
                    )
                    k = torch.randn(
                        seq_len, NUM_HEADS, QK_HEAD_DIM, dtype=DTYPE, device=DEVICE
                    )
                    v = torch.randn(
                        seq_len, NUM_HEADS, V_HEAD_DIM, dtype=DTYPE, device=DEVICE
                    )

                    ref = _full_ring_mla(q, k, v, layer).float()

                    sizes, starts = _block_layout(seq_len, cp_size)
                    seg = cp_size * 2
                    recon = torch.zeros_like(ref)

                    for cp_rank in range(cp_size):
                        meta = _cp_metadata(sizes, starts, cp_rank, cp_size)
                        # This rank's local q = [prev block, next block].
                        prev_b, next_b = cp_rank, seg - 1 - cp_rank
                        q_local = torch.cat(
                            [
                                q[starts[prev_b] : starts[prev_b] + sizes[prev_b]],
                                q[starts[next_b] : starts[next_b] + sizes[next_b]],
                            ],
                            dim=0,
                        )
                        fb = SimpleNamespace(attn_cp_metadata=meta)
                        out_local = backend.do_cp_attn_mla(
                            q_local, k, v, layer, fb
                        ).float()

                        split = meta.total_q_prev_tokens
                        recon[starts[prev_b] : starts[prev_b] + sizes[prev_b]] = (
                            out_local[:split]
                        )
                        recon[starts[next_b] : starts[next_b] + sizes[next_b]] = (
                            out_local[split:]
                        )

                    torch.testing.assert_close(recon, ref, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
