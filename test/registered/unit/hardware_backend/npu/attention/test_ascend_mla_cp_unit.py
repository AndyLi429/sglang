"""Unit tests for AscendAttnBackend MLA context-parallel (attn_cp) path.

These tests run on CPU with mocked NPU ops. They verify:
  1. `_forward_mla_pcp` calls `npu_ring_mla` four times: 2 per Q half
     (first_ring with triu mask, then default with prev_lse merge).
  2. Slice boundaries and seqlen tensors fed to ring_mla are derived
     correctly from `forward_batch.attn_cp_metadata` community fields
     (no idx_select copy, no `(seq+1)//2` shortcut).
  3. The all-gather helper concats k+v into one collective and splits
     back to the original trailing-dim layout.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_cp_utils_stub():
    """Stub cp_utils + dsa.utils so we don't pull their (potentially
    older-codebase-incompatible) transitive deps just to test the backend."""
    if "sglang.srt.layers.utils.cp_utils" not in sys.modules:
        mod = _stub_module("sglang.srt.layers.utils.cp_utils")
        mod.cp_all_gather_rerange_kv_cache = MagicMock()
        mod.mla_use_prefill_cp = MagicMock(return_value=False)

        class ContextParallelMetadata:  # noqa: D401
            pass

        mod.ContextParallelMetadata = ContextParallelMetadata
        sys.modules["sglang.srt.layers.utils.cp_utils"] = mod
    if "sglang.srt.layers.attention.dsa" not in sys.modules:
        sys.modules["sglang.srt.layers.attention.dsa"] = _stub_module(
            "sglang.srt.layers.attention.dsa"
        )
    if "sglang.srt.layers.attention.dsa.utils" not in sys.modules:
        dsa_utils = _stub_module("sglang.srt.layers.attention.dsa.utils")
        dsa_utils.is_dsa_enable_prefill_cp = MagicMock(return_value=False)
        sys.modules["sglang.srt.layers.attention.dsa.utils"] = dsa_utils


def _install_npu_stubs():
    _install_cp_utils_stub()
    # Force-replace torch_npu (even if already imported on a real NPU host)
    # so npu_ring_mla is a MagicMock we can inspect.
    tnpu = _stub_module("torch_npu")
    tnpu._npu_flash_attention_qlens = MagicMock()
    tnpu._npu_paged_attention = MagicMock()
    tnpu._npu_paged_attention_mla = MagicMock()
    tnpu._npu_fused_infer_attention_score_get_max_workspace = MagicMock(
        return_value=torch.empty(0)
    )
    tnpu.npu_fused_infer_attention_score = MagicMock()
    tnpu.npu_sparse_flash_attention = MagicMock(return_value=(MagicMock(), None, None))
    atb = _stub_module("torch_npu.atb")
    atb.npu_ring_mla = MagicMock()
    tnpu.atb = atb
    sys.modules["torch_npu"] = tnpu
    sys.modules["torch_npu.atb"] = atb

    if not hasattr(torch, "ops"):
        torch.ops = MagicMock()
    npu_ops = MagicMock()
    npu_ops.npu_fused_infer_attention_score = MagicMock(
        return_value=(MagicMock(), MagicMock())
    )
    torch.ops.npu = npu_ops

    for pkg in [
        "sgl_kernel_npu",
        "sgl_kernel_npu.attention",
        "sgl_kernel_npu.attention.sinks_attention",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = _stub_module(
                pkg,
                attention_sinks_prefill_triton=MagicMock(),
                attention_sinks_triton=MagicMock(),
            )


_install_npu_stubs()

from sglang.srt.hardware_backend.npu.attention import ascend_backend  # noqa: E402
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (  # noqa: E402
    AscendAttnBackend,
    _cp_allgather_mla_kv_npu,
)


def _make_mla_backend(attn_cp_size: int = 2) -> AscendAttnBackend:
    from sglang.srt.configs.model_config import AttentionArch

    mr = MagicMock()
    mr.device = "cpu"
    mr.dtype = torch.bfloat16
    mr.page_size = 16
    mr.model_config.dtype = torch.bfloat16
    mr.model_config.context_len = 4096
    mr.model_config.attention_arch = AttentionArch.MLA
    mr.model_config.kv_lora_rank = 512
    mr.model_config.qk_rope_head_dim = 64
    mr.model_config.qk_nope_head_dim = 128
    mr.model_config.num_attention_heads = 16
    mr.model_config.hf_config.architectures = ["DeepseekV3ForCausalLM"]
    mr.is_hybrid_swa = False
    mr.server_args.enable_torch_compile = False
    mr.server_args.speculative_num_draft_tokens = None
    pool_size = 32
    mr.req_to_token_pool.req_to_token = torch.zeros(pool_size, 4096, dtype=torch.int32)
    mr.attn_cp_size = attn_cp_size

    with patch.object(
        ascend_backend, "get_bool_env_var", return_value=False
    ), patch.object(
        ascend_backend.DllmConfig, "from_server_args", return_value=None
    ), patch.object(
        ascend_backend, "get_attention_tp_size", return_value=1
    ):
        backend = AscendAttnBackend(mr)
    return backend


def _make_attn_cp_metadata(
    q_prev_len: int,
    q_next_len: int,
    kv_len_prev: int,
    kv_len_next: int,
):
    """Build a minimal ContextParallelMetadata-like object (bs=1)."""
    meta = MagicMock()
    meta.bs = 1
    meta.total_q_prev_tokens = q_prev_len
    meta.total_q_next_tokens = q_next_len
    meta.actual_seq_q_prev_list = [q_prev_len]
    meta.actual_seq_q_next_list = [q_next_len]
    meta.kv_len_prev_list = [kv_len_prev]
    meta.kv_len_next_list = [kv_len_next]
    return meta


def _make_layer(tp_q_head_num: int = 16, v_head_dim: int = 128):
    layer = MagicMock()
    layer.tp_q_head_num = tp_q_head_num
    layer.tp_k_head_num = tp_q_head_num
    layer.v_head_dim = v_head_dim
    layer.scaling = 1.0 / (v_head_dim**0.5)
    layer.layer_id = 0
    layer.is_cross_attention = False
    return layer


class TestCpAllgatherMlaKvNpu(CustomTestCase):
    """The free helper merges k+v into one collective and reshapes back."""

    def test_shapes_round_trip(self):
        s_local, h, q_dim, v_dim = 8, 4, 192, 128
        cp_size = 2
        k = torch.randn(s_local, h, q_dim)
        v = torch.randn(s_local, h, v_dim)

        # Mock the all-gather to be a deterministic 2x repeat for cp_size=2.
        def fake_gather(t, cp, fb, stream):
            return torch.cat([t, t], dim=0)

        fb = MagicMock()
        with patch.object(
            ascend_backend, "cp_all_gather_rerange_kv_cache", side_effect=fake_gather
        ), patch.object(
            ascend_backend, "get_current_device_stream_fast", return_value=None
        ):
            k_full, v_full = _cp_allgather_mla_kv_npu(fb, k, v, cp_size)

        self.assertEqual(k_full.shape, (s_local * cp_size, h, q_dim))
        self.assertEqual(v_full.shape, (s_local * cp_size, h, v_dim))
        # First-half slice must equal the input (fake gather repeats).
        self.assertTrue(torch.allclose(k_full[:s_local], k))
        self.assertTrue(torch.allclose(v_full[:s_local], v))


class TestForwardMlaPcp(CustomTestCase):
    """`_forward_mla_pcp` issues 4 ring_mla calls with correct args."""

    def _run(self, q_prev=4, q_next=4, kv_prev=8, kv_next=16):
        backend = _make_mla_backend(attn_cp_size=2)
        layer = _make_layer()

        q_total = q_prev + q_next
        # Each rank holds full local Q = q_prev + q_next tokens.
        q = torch.randn(
            q_total, layer.tp_q_head_num * (layer.v_head_dim + backend.qk_rope_head_dim)
        )
        # Local KV (will be 2x'd by the fake all-gather).
        s_local = max(kv_prev, kv_next) // 2 + 1
        k = torch.randn(
            s_local, layer.tp_k_head_num, layer.v_head_dim + backend.qk_rope_head_dim
        )
        v = torch.randn(s_local, layer.tp_k_head_num, layer.v_head_dim)

        fb = MagicMock()
        fb.attn_cp_metadata = _make_attn_cp_metadata(q_prev, q_next, kv_prev, kv_next)

        # Capture ring_mla calls.
        ring_mock = sys.modules["torch_npu.atb"].npu_ring_mla
        ring_mock.reset_mock()

        def fake_gather(t, cp, fb_, stream):
            return torch.cat([t] * cp, dim=0)

        with patch.object(
            ascend_backend, "cp_all_gather_rerange_kv_cache", side_effect=fake_gather
        ), patch.object(
            ascend_backend, "get_current_device_stream_fast", return_value=None
        ):
            out = backend._forward_mla_pcp(q, k, v, layer, fb)

        return backend, layer, ring_mock, out

    def test_four_ring_calls(self):
        _, _, ring_mock, _ = self._run()
        self.assertEqual(
            ring_mock.call_count,
            4,
            "Expected 4 ring_mla calls (2 per Q half: first_ring + default)",
        )

    def test_first_call_is_first_ring_with_triu_mask(self):
        _, _, ring_mock, _ = self._run()
        first_call = ring_mock.call_args_list[0]
        self.assertEqual(first_call.kwargs["calc_type"], "calc_type_first_ring")
        self.assertEqual(first_call.kwargs["mask_type"], "mask_type_triu")
        self.assertIsNone(first_call.kwargs["pre_out"])
        self.assertIsNone(first_call.kwargs["prev_lse"])

    def test_second_call_is_lse_merge(self):
        _, _, ring_mock, _ = self._run()
        second_call = ring_mock.call_args_list[1]
        self.assertEqual(second_call.kwargs["calc_type"], "calc_type_default")
        self.assertEqual(second_call.kwargs["mask_type"], "no_mask")
        self.assertIsNotNone(second_call.kwargs["pre_out"])
        self.assertIsNotNone(second_call.kwargs["prev_lse"])

    def test_seqlens_match_metadata(self):
        """Verify ring_mla seqlen args match attn_cp_metadata exactly,
        proving we do NOT use the legacy (seq_len+1)//2 path.

        seqlen format follows the proven npu_ring_mla convention:
          - first_ring (diagonal, square q x q): 1D [bs] q-lengths.
          - default (q x kv): 2D [2, bs] = [q_lengths; kv_lengths].
        """
        q_prev, q_next, kv_prev, kv_next = 5, 7, 11, 19
        _, _, ring_mock, _ = self._run(q_prev, q_next, kv_prev, kv_next)
        # Call 0: head mask   (first_ring) → [q_prev]
        # Call 1: head nomask (default)    → [[q_prev], [kv_prev - q_prev]]
        # Call 2: tail mask   (first_ring) → [q_next]
        # Call 3: tail nomask (default)    → [[q_next], [kv_next - q_next]]
        seqlens = [c.kwargs["seqlen"].tolist() for c in ring_mock.call_args_list]
        self.assertEqual(seqlens[0], [q_prev])
        self.assertEqual(seqlens[1], [[q_prev], [kv_prev - q_prev]])
        self.assertEqual(seqlens[2], [q_next])
        self.assertEqual(seqlens[3], [[q_next], [kv_next - q_next]])

    def test_q_split_uses_total_q_prev_tokens(self):
        """The Q split must use total_q_prev_tokens, not (S+1)//2.
        With q_prev=3, q_next=5 (asymmetric), the head Q must have 3
        tokens — a naive (8+1)//2 = 4 split would fail this test."""
        q_prev, q_next, kv_prev, kv_next = 3, 5, 6, 11
        _, layer, ring_mock, _ = self._run(q_prev, q_next, kv_prev, kv_next)
        # First call's q_nope shape[0] should equal q_prev (not 4)
        first_q_nope = ring_mock.call_args_list[0].kwargs["q_nope"]
        self.assertEqual(first_q_nope.shape[0], q_prev)
        # Third call's q_nope shape[0] should equal q_next
        third_q_nope = ring_mock.call_args_list[2].kwargs["q_nope"]
        self.assertEqual(third_q_nope.shape[0], q_next)


if __name__ == "__main__":
    unittest.main()
