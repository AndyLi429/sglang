"""Characterization unit tests for srt/hardware_backend/npu/attention/ascend_backend.py.

These tests run on CPU with torch_npu / sgl_kernel_npu stubbed. They pin down
the observable behavior of AscendAttnBackend so the file can be refactored
safely:

  * pure helpers (mask builders, FIA-NZ reshape, alibi math)
  * backend construction (MLA / non-MLA, q-head padding, SWA flags)
  * init_forward_metadata (block tables, spec-decode offsets, MLA prefix
    tables, SWA translation)
  * graph capture / replay metadata (init_cuda_graph_state,
    init_forward_metadata_out_graph)
  * forward_extend / forward_decode dispatch (dllm / sparse / mtp / graph)
  * kernel-call argument construction for the NPU ops (captured via mocks)
  * the CP all-gather KV save helper
  * AscendAttnMultiStepDraftBackend looping
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

if sys.platform == "win32":
    # Local dev convenience only — CI runs on Linux where these exist.
    for _m in ("resource", "fcntl", "pwd", "grp", "termios"):
        sys.modules.setdefault(_m, types.ModuleType(_m))
    for _m in ("triton", "triton.language"):
        sys.modules.setdefault(_m, MagicMock())

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_npu_stubs():
    """Force-replace torch_npu / sgl_kernel_npu so NPU kernels become
    inspectable MagicMocks and the module imports on CPU."""
    tnpu = _stub_module("torch_npu")
    tnpu._npu_flash_attention_qlens = MagicMock()
    tnpu._npu_paged_attention = MagicMock()
    tnpu._npu_paged_attention_mla = MagicMock()
    tnpu._npu_fused_infer_attention_score_get_max_workspace = MagicMock(
        return_value=torch.empty(0)
    )
    tnpu._npu_fused_infer_attention_score_v2_get_max_workspace = MagicMock(
        return_value=torch.empty(0)
    )
    tnpu.npu_fused_infer_attention_score = MagicMock()
    tnpu.npu_fused_infer_attention_score.out = MagicMock()
    tnpu.npu_fused_infer_attention_score_v2 = MagicMock()
    tnpu.npu_fused_infer_attention_score_v2.out = MagicMock()
    tnpu.npu_sparse_flash_attention = MagicMock(
        return_value=(torch.zeros(1), None, None)
    )
    atb = _stub_module("torch_npu.atb")
    atb.npu_ring_mla = MagicMock()
    tnpu.atb = atb
    sys.modules["torch_npu"] = tnpu
    sys.modules["torch_npu.atb"] = atb

    npu_ops = MagicMock()
    torch.ops.npu = npu_ops

    sinks_mod = _stub_module(
        "sgl_kernel_npu.attention.sinks_attention",
        attention_sinks_prefill_triton=MagicMock(),
        attention_sinks_triton=MagicMock(),
    )
    sys.modules.setdefault("sgl_kernel_npu", _stub_module("sgl_kernel_npu"))
    sys.modules.setdefault(
        "sgl_kernel_npu.attention", _stub_module("sgl_kernel_npu.attention")
    )
    sys.modules["sgl_kernel_npu.attention.sinks_attention"] = sinks_mod


_install_npu_stubs()

from sglang.srt.hardware_backend.npu.attention import ascend_backend  # noqa: E402
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (  # noqa: E402
    AscendAttnBackend,
    AscendAttnMaskBuilder,
    AscendAttnMultiStepDraftBackend,
    ForwardMetadata,
    _cp_allgather_and_save_kv_npu,
    _reshape_kv_for_fia_nz,
)

_torch_npu_stub = sys.modules["torch_npu"]

_REAL_TORCH_TENSOR = torch.tensor


def _cpu_tensor(*args, **kwargs):
    """torch.tensor wrapper mapping device='npu' to CPU for constructor tests."""
    if kwargs.get("device") == "npu":
        kwargs["device"] = "cpu"
    return _REAL_TORCH_TENSOR(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAGE_SIZE = 16
CONTEXT_LEN = 256
POOL_SIZE = 8


def _make_model_runner(
    use_mla: bool = False,
    num_attention_heads: int = 16,
    tp_size: int = 1,
    is_hybrid_swa: bool = False,
    speculative_num_draft_tokens=None,
    architectures=None,
):
    from sglang.srt.configs.model_config import AttentionArch

    mr = MagicMock()
    mr.device = "cpu"
    mr.dtype = torch.float16
    mr.page_size = PAGE_SIZE
    mr.model_config.dtype = torch.float16
    mr.model_config.context_len = CONTEXT_LEN
    mr.model_config.num_attention_heads = num_attention_heads
    mr.model_config.use_alibi = False
    if use_mla:
        mr.model_config.attention_arch = AttentionArch.MLA
        mr.model_config.kv_lora_rank = 512
        mr.model_config.qk_rope_head_dim = 64
        mr.model_config.qk_nope_head_dim = 128
        mr.model_config.hf_config.qk_nope_head_dim = 96  # MiniCPM3 override value
    else:
        mr.model_config.attention_arch = AttentionArch.MHA
    mr.model_config.hf_config.architectures = architectures or ["LlamaForCausalLM"]
    mr.is_hybrid_swa = is_hybrid_swa
    mr.sliding_window_size = 128
    mr.server_args.enable_torch_compile = False
    mr.server_args.speculative_num_draft_tokens = speculative_num_draft_tokens
    mr.attn_cp_size = 1
    # Deterministic req_to_token: req i, position j -> i * 1000 + j (page-aligned
    # values so // page_size is meaningful).
    req_to_token = torch.arange(POOL_SIZE * CONTEXT_LEN, dtype=torch.int32).reshape(
        POOL_SIZE, CONTEXT_LEN
    )
    mr.req_to_token_pool.req_to_token = req_to_token
    if is_hybrid_swa:
        # full->swa mapping: identity + 1 so it is distinguishable
        mr.token_to_kv_pool.full_to_swa_index_mapping = (
            torch.arange(POOL_SIZE * CONTEXT_LEN, dtype=torch.int64) + PAGE_SIZE
        )
    return mr


def _make_backend(model_runner=None, speculative_step_id: int = 0, tp_size: int = 1):
    mr = model_runner if model_runner is not None else _make_model_runner()
    with patch.object(
        ascend_backend, "get_bool_env_var", return_value=False
    ), patch.object(
        ascend_backend.DllmConfig, "from_server_args", return_value=None
    ), patch.object(
        ascend_backend, "get_attention_tp_size", return_value=tp_size
    ), patch.object(
        torch, "tensor", _cpu_tensor
    ):
        return AscendAttnBackend(mr, speculative_step_id=speculative_step_id)


def _make_forward_mode(active=()):
    """ForwardMode-like mock where only the modes in `active` return True."""
    mode = MagicMock()
    for name in (
        "is_extend",
        "is_decode_or_idle",
        "is_target_verify",
        "is_draft_extend",
        "is_draft_extend_v2",
        "is_dllm_extend",
        "is_context_parallel_extend",
    ):
        getattr(mode, name).return_value = name in active
    return mode


def _make_forward_batch(
    backend,
    mode_active=("is_decode_or_idle",),
    batch_size=2,
    seq_lens=(32, 48),
    extend_seq_lens=None,
    extend_prefix_lens=None,
    spec_info=None,
):
    fb = MagicMock()
    fb.forward_mode = _make_forward_mode(mode_active)
    fb.batch_size = batch_size
    fb.req_pool_indices = torch.arange(batch_size, dtype=torch.int64)
    fb.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    fb.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    fb.spec_info = spec_info
    fb.out_cache_loc = torch.arange(batch_size, dtype=torch.int64)
    fb.encoder_lens = None
    fb.attn_cp_metadata = None
    if extend_seq_lens is not None:
        fb.extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int64)
        fb.extend_seq_lens_cpu = list(extend_seq_lens)
        fb.num_token_non_padded_cpu = int(sum(extend_seq_lens))
    else:
        fb.extend_seq_lens = None
        fb.extend_seq_lens_cpu = [1] * batch_size
        fb.num_token_non_padded_cpu = batch_size
    if extend_prefix_lens is not None:
        fb.extend_prefix_lens = torch.tensor(extend_prefix_lens, dtype=torch.int64)
        fb.extend_prefix_lens_cpu = list(extend_prefix_lens)
    else:
        fb.extend_prefix_lens = torch.zeros(batch_size, dtype=torch.int64)
        fb.extend_prefix_lens_cpu = [0] * batch_size
    return fb


def _make_layer(
    tp_q_head_num=8,
    tp_k_head_num=8,
    qk_head_dim=128,
    v_head_dim=128,
    sliding_window_size=-1,
    layer_id=0,
):
    layer = MagicMock()
    layer.tp_q_head_num = tp_q_head_num
    layer.tp_k_head_num = tp_k_head_num
    layer.tp_v_head_num = tp_k_head_num
    layer.qk_head_dim = qk_head_dim
    layer.v_head_dim = v_head_dim
    layer.head_dim = qk_head_dim
    layer.scaling = qk_head_dim**-0.5
    layer.layer_id = layer_id
    layer.is_cross_attention = False
    layer.sliding_window_size = sliding_window_size
    layer.logit_cap = 0
    return layer


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestReshapeKvForFiaNz(CustomTestCase):
    def test_shape(self):
        t = torch.zeros(4 * PAGE_SIZE, 1, 128)
        out = _reshape_kv_for_fia_nz(t, num_heads=1, head_dim=128, page_size=PAGE_SIZE)
        self.assertEqual(out.shape, (4, 1, 128 // 16, PAGE_SIZE, 16))


class TestAscendAttnMaskBuilder(CustomTestCase):
    def _builder(self, use_mla=False):
        mr = MagicMock()
        mr.dtype = torch.float16
        return AscendAttnMaskBuilder(mr, "cpu", use_fia=False, use_mla=use_mla)

    def test_generate_mask_flag_is_strict_upper_triangle(self):
        flag = AscendAttnMaskBuilder.generate_mask_flag(4)
        expected = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(flag, expected))

    def test_generate_attn_mask_norm_fp16(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertEqual(mask[0, 0].item(), 0)
        # upper triangle filled with float32 min (saturates to -inf in fp16)
        self.assertTrue(mask[0, 1].item() < -1e30 or mask[0, 1].item() == float("-inf"))
        self.assertEqual(mask[3, 0].item(), 0)

    def test_generate_attn_mask_norm_bf16_uses_one(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask[0, 1].item(), 1)

    def test_generate_attn_mask_mix_bf16_uses_neg_inf(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.bfloat16)
        self.assertEqual(mask[0, 1].item(), float("-inf"))

    def test_get_attention_mask_id(self):
        seq_lens = torch.tensor([5, 7])
        extend_lens = torch.tensor([2, 3])
        out = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        self.assertEqual(out.tolist(), [3, 4, 4, 5, 6])

    def test_update_attn_cache_grows_and_converts(self):
        b = self._builder()
        cache, cached_len = b.update_attn_cache(8, b.mask, 128, torch.bfloat16, "norm")
        # 8 <= 128: no regen, but dtype converted
        self.assertEqual(cached_len, 128)
        self.assertEqual(cache.dtype, torch.bfloat16)
        cache, cached_len = b.update_attn_cache(256, b.mask, 128, torch.float16, "norm")
        self.assertEqual(cached_len, 256)
        self.assertEqual(cache.shape, (256, 256))

    def test_get_splitfuse_attn_mask(self):
        b = self._builder()
        mask = b.get_splitfuse_attn_mask(4)
        self.assertEqual(mask.dtype, torch.int8)
        expected = torch.triu(torch.ones(4, 4), diagonal=1).to(torch.int8)
        self.assertTrue(torch.equal(mask, expected))

    def test_get_swa_mask_window_semantics(self):
        b = self._builder()
        seq_lens = torch.tensor([6])
        mask = b.get_swa_mask(seq_lens, s2=8, left_context=4)
        self.assertEqual(mask.shape, (1, 1, 8))
        # window is [seq_len - left_context, seq_len) = [2, 6)
        self.assertEqual(
            mask[0, 0].tolist(),
            [True, True, False, False, False, False, True, True],
        )

    def test_mla_builder_has_ringmla_mask(self):
        b = self._builder(use_mla=True)
        self.assertEqual(b.ringmla_mask.shape, (512, 512))
        self.assertEqual(b.ringmla_mask.dtype, torch.bfloat16)
        b2 = self._builder(use_mla=False)
        self.assertFalse(hasattr(b2, "ringmla_mask"))


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


class TestBackendInit(CustomTestCase):
    def test_non_mla_init(self):
        backend = _make_backend()
        self.assertFalse(backend.use_mla)
        self.assertFalse(backend.use_alibi)
        self.assertFalse(backend.graph_mode)
        self.assertIsNone(backend.q_head_num_padding)
        self.assertEqual(backend.page_size, PAGE_SIZE)
        self.assertEqual(backend.max_context_len, CONTEXT_LEN)
        # mask shortcuts wired from the builder
        self.assertIs(backend.mask, backend.ascend_attn_mask_builder.mask)
        self.assertIs(backend.fia_mask, backend.ascend_attn_mask_builder.fia_mask)
        self.assertFalse(hasattr(backend, "ringmla_mask"))

    def test_native_sdpa_flag_for_gemma2_classifier(self):
        mr = _make_model_runner(architectures=["Gemma2ForSequenceClassification"])
        backend = _make_backend(mr)
        self.assertTrue(getattr(backend, "use_native_sdpa", False))

    def test_mla_init_head_dims(self):
        backend = _make_backend(_make_model_runner(use_mla=True))
        self.assertTrue(backend.use_mla)
        self.assertEqual(backend.kv_lora_rank, 512)
        self.assertEqual(backend.qk_rope_head_dim, 64)
        self.assertEqual(backend.qk_nope_head_dim, 128)
        self.assertEqual(backend.q_head_dim, 192)
        self.assertEqual(backend.ringmla_mask.shape, (512, 512))

    def test_mla_minicpm3_uses_hf_config_nope_dim(self):
        mr = _make_model_runner(use_mla=True, architectures=["MiniCPM3ForCausalLM"])
        backend = _make_backend(mr)
        self.assertEqual(backend.qk_nope_head_dim, 96)
        self.assertEqual(backend.q_head_dim, 96 + 64)

    def test_q_head_num_padding_power_of_two(self):
        # 16 heads / tp1 -> already power of two -> padding == 16
        backend = _make_backend(_make_model_runner(use_mla=True))
        self.assertEqual(backend.tp_q_head_num, 16)
        self.assertEqual(backend.q_head_num_padding, 16)
        # 22 heads -> padded to 32
        backend = _make_backend(
            _make_model_runner(use_mla=True, num_attention_heads=22)
        )
        self.assertEqual(backend.q_head_num_padding, 32)

    def test_speculative_step_offset(self):
        backend = _make_backend(speculative_step_id=2)
        self.assertEqual(backend.speculative_step_id, 2)
        self.assertEqual(backend.speculative_step_offset_npu.item(), 3)

    def test_is_swa_layer(self):
        backend = _make_backend(_make_model_runner(is_hybrid_swa=True))
        self.assertTrue(backend.is_hybrid_swa)
        self.assertTrue(backend._is_swa_layer(_make_layer(sliding_window_size=128)))
        self.assertFalse(backend._is_swa_layer(_make_layer(sliding_window_size=-1)))
        layer = _make_layer()
        layer.sliding_window_size = None
        self.assertFalse(backend._is_swa_layer(layer))
        backend2 = _make_backend()
        self.assertFalse(backend2._is_swa_layer(_make_layer(sliding_window_size=128)))

    def test_can_use_tnd(self):
        cases = {
            (128, 128): True,
            (192, 192): True,
            (192, 128): True,
            (64, 64): False,
            (256, 256): False,
            (128, 64): False,
        }
        for (d, v), expected in cases.items():
            layer = _make_layer(qk_head_dim=d, v_head_dim=v)
            self.assertEqual(
                AscendAttnBackend._can_use_tnd(layer), expected, msg=f"d={d} v={v}"
            )

    def test_verify_buffers_protocol(self):
        backend = _make_backend()
        self.assertEqual(backend.get_verify_buffers_to_fill_after_draft(), [None, None])
        backend.update_verify_buffers_to_fill_after_draft(MagicMock(), None)  # no-op
        self.assertEqual(backend.get_cuda_graph_seq_len_fill_value(), 0)


# ---------------------------------------------------------------------------
# init_forward_metadata
# ---------------------------------------------------------------------------


class TestInitForwardMetadata(CustomTestCase):
    def test_decode_block_tables_and_seq_lens(self):
        backend = _make_backend()
        fb = _make_forward_batch(backend, seq_lens=(32, 48))
        backend.init_forward_metadata(fb)
        md = backend.forward_metadata

        req_to_token = backend.req_to_token_pool.req_to_token
        expected = req_to_token[fb.req_pool_indices, :48][:, ::PAGE_SIZE] // PAGE_SIZE
        self.assertTrue(torch.equal(md.block_tables, expected))
        self.assertEqual(md.seq_lens.dtype, torch.int32)
        self.assertEqual(md.seq_lens_cpu_int.tolist(), [32, 48])
        self.assertEqual(list(md.seq_lens_list_cumsum), [1, 2])
        self.assertFalse(backend.graph_mode)

    def test_target_verify_extends_seq_lens(self):
        backend = _make_backend(_make_model_runner(speculative_num_draft_tokens=4))
        fb = _make_forward_batch(backend, mode_active=("is_target_verify",))
        backend.init_forward_metadata(fb)
        md = backend.forward_metadata
        self.assertEqual(md.seq_lens_cpu_int.tolist(), [36, 52])
        # block tables sized by seq_lens_max + draft tokens
        self.assertEqual(
            md.block_tables.shape[1], (48 + 4 + PAGE_SIZE - 1) // PAGE_SIZE
        )
        self.assertIsNone(md.seq_lens_list_cumsum)

    def test_decode_with_spec_info_adds_step_offset(self):
        backend = _make_backend(speculative_step_id=1)
        fb = _make_forward_batch(backend, spec_info=MagicMock())
        backend.init_forward_metadata(fb)
        self.assertEqual(backend.forward_metadata.seq_lens_cpu_int.tolist(), [34, 50])

    def test_extend_records_extend_seq_lens(self):
        backend = _make_backend()
        fb = _make_forward_batch(
            backend, mode_active=("is_extend",), extend_seq_lens=(8, 16)
        )
        backend.init_forward_metadata(fb)
        md = backend.forward_metadata
        self.assertEqual(md.extend_seq_lens_cpu_int.tolist(), [8, 16])
        self.assertEqual(list(md.seq_lens_list_cumsum), [8, 24])

    def test_mla_extend_prefix_block_tables(self):
        backend = _make_backend(_make_model_runner(use_mla=True))
        fb = _make_forward_batch(
            backend,
            mode_active=("is_extend",),
            seq_lens=(40, 48),
            extend_seq_lens=(8, 16),
            extend_prefix_lens=(32, 32),
        )
        backend.init_forward_metadata(fb)
        md = backend.forward_metadata
        self.assertEqual(md.prefix_lens.tolist(), [32, 32])
        req_to_token = backend.req_to_token_pool.req_to_token
        expected = torch.cat(
            [
                req_to_token[0][:32][::PAGE_SIZE] // PAGE_SIZE,
                req_to_token[1][:32][::PAGE_SIZE] // PAGE_SIZE,
            ]
        )
        self.assertTrue(torch.equal(md.flatten_prefix_block_tables.cpu(), expected))

    def test_swa_block_tables(self):
        backend = _make_backend(_make_model_runner(is_hybrid_swa=True))
        fb = _make_forward_batch(backend, seq_lens=(32, 48))
        backend.init_forward_metadata(fb)
        md = backend.forward_metadata
        mapping = backend.full_to_swa_index_mapping
        req_to_token = backend.req_to_token_pool.req_to_token
        expected = (
            mapping[req_to_token[fb.req_pool_indices, :48]][:, ::PAGE_SIZE] // PAGE_SIZE
        ).to(torch.int32)
        self.assertTrue(torch.equal(md.block_tables_swa, expected))

    def test_swa_pool_translates_out_cache_loc(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        class _FakeSWAPool(SWAKVPool):
            def __init__(self):
                pass

        pool = _FakeSWAPool()
        pool.swa_layer_nums = 2
        translated = torch.tensor([100, 101])
        pool.translate_loc_from_full_to_swa = MagicMock(return_value=translated)
        mr = _make_model_runner()
        mr.token_to_kv_pool = pool
        backend = _make_backend(mr)
        self.assertTrue(backend.use_sliding_window_kv_pool)
        fb = _make_forward_batch(backend)
        backend.init_forward_metadata(fb)
        self.assertIs(backend.forward_metadata.swa_out_cache_loc, translated)
        pool.translate_loc_from_full_to_swa.assert_called_once_with(fb.out_cache_loc)


# ---------------------------------------------------------------------------
# Graph (capture/replay) metadata
# ---------------------------------------------------------------------------


class TestGraphMetadata(CustomTestCase):
    def test_init_cuda_graph_state_shapes(self):
        backend = _make_backend()
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=64)
        total = CONTEXT_LEN + PAGE_SIZE - 1
        self.assertEqual(
            backend.graph_metadata["block_tables"].shape,
            (4, total // PAGE_SIZE),
        )

    def test_init_cuda_graph_state_with_draft_tokens(self):
        backend = _make_backend(_make_model_runner(speculative_num_draft_tokens=4))
        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=32)
        total = CONTEXT_LEN + PAGE_SIZE - 1 + 4
        self.assertEqual(
            backend.graph_metadata["block_tables"].shape[1], total // PAGE_SIZE
        )

    def test_capture_then_replay_decode(self):
        backend = _make_backend()
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=64)
        bs = 2
        fb = _make_forward_batch(backend, batch_size=bs, seq_lens=(32, 48))
        backend.init_forward_metadata_out_graph(fb, in_capture=True)

        self.assertTrue(backend.graph_mode)
        md = backend.forward_metadata
        self.assertIs(md, backend.graph_metadata[bs])
        # actual_seq_lengths_q for plain decode = [1, 2, ..., bs]
        self.assertEqual(md.actual_seq_lengths_q.tolist(), [1, 2])
        self.assertEqual(md.seq_lens_cpu_list, [32, 48])
        # block tables refilled from req_to_token
        req_to_token = backend.req_to_token_pool.req_to_token
        max_pages = (48 + PAGE_SIZE - 1) // PAGE_SIZE
        expected = req_to_token[fb.req_pool_indices, :48][:, ::PAGE_SIZE] // PAGE_SIZE
        self.assertTrue(
            torch.equal(md.block_tables[:bs, :max_pages], expected.to(torch.int32))
        )
        # padding regions zeroed
        self.assertTrue((md.block_tables[:bs, max_pages:] == 0).all())
        self.assertTrue((md.block_tables[bs:, :] == 0).all())
        self.assertEqual(md.seq_lens[:bs].tolist(), [32, 48])

    def test_replay_target_verify_offsets_seq_lens(self):
        backend = _make_backend(_make_model_runner(speculative_num_draft_tokens=4))
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=64)
        fb = _make_forward_batch(
            backend, mode_active=("is_target_verify",), seq_lens=(32, 48)
        )
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        md = backend.forward_metadata
        # verify mode: q lens are arange of draft tokens
        self.assertEqual(md.actual_seq_lengths_q.tolist(), [4, 8])
        self.assertEqual(md.seq_lens[:2].tolist(), [36, 52])

    def test_replay_decode_with_spec_info_offsets_seq_lens(self):
        backend = _make_backend(speculative_step_id=1)
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=64)
        fb = _make_forward_batch(backend, seq_lens=(32, 48), spec_info=MagicMock())
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        self.assertEqual(backend.forward_metadata.seq_lens[:2].tolist(), [34, 50])

    def test_swa_replay_mask_and_block_tables(self):
        backend = _make_backend(_make_model_runner(is_hybrid_swa=True))
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=64)
        bs = 1
        seq_len = 200  # > sliding_window_size = 128
        fb = _make_forward_batch(backend, batch_size=bs, seq_lens=(seq_len,))
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        md = backend.forward_metadata
        mask_row = md.swa_mask[0, 0]
        start = seq_len - backend.sliding_window_size
        self.assertTrue(mask_row[:start].all().item())  # before window: masked
        self.assertFalse(mask_row[start:seq_len].any().item())  # window: attend
        self.assertTrue(mask_row[seq_len:].all().item())  # after seq: masked
        # rows beyond bs fully masked
        self.assertTrue(md.swa_mask[bs:].all().item())
        # swa block tables filled from mapping
        mapping = backend.full_to_swa_index_mapping
        req_to_token = backend.req_to_token_pool.req_to_token
        max_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        expected = (
            mapping[req_to_token[fb.req_pool_indices, :seq_len]][:, ::PAGE_SIZE]
            // PAGE_SIZE
        ).to(torch.int32)
        self.assertTrue(torch.equal(md.block_tables_swa[:bs, :max_pages], expected))

    def test_mla_padding_buffers_created_when_heads_not_power_of_two(self):
        backend = _make_backend(
            _make_model_runner(use_mla=True, num_attention_heads=22)
        )
        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=32)
        fb = _make_forward_batch(backend, batch_size=2, seq_lens=(32, 48))
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        md = backend.forward_metadata
        self.assertEqual(md.nope_padding.shape, (2, 1, 32 - 22, 512))
        self.assertEqual(md.rope_padding.shape, (2, 1, 32 - 22, 64))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch(CustomTestCase):
    def test_extend_dispatches_to_dllm(self):
        backend = _make_backend()
        backend.is_dllm_model = True
        backend.forward_dllm = MagicMock(return_value="dllm")
        fb = _make_forward_batch(backend, mode_active=("is_extend",))
        out = backend.forward_extend(None, None, None, _make_layer(), fb)
        self.assertEqual(out, "dllm")
        backend.forward_dllm.assert_called_once()

    def test_extend_dispatches_to_sparse_on_topk(self):
        backend = _make_backend()
        backend.forward_sparse = MagicMock(return_value="sparse")
        fb = _make_forward_batch(backend, mode_active=("is_extend",))
        out = backend.forward_extend(
            None, None, None, _make_layer(), fb, topk_indices=torch.zeros(1)
        )
        self.assertEqual(out, "sparse")

    def test_extend_dispatches_to_mtp_on_target_verify(self):
        backend = _make_backend()
        backend.forward_mtp = MagicMock(return_value="mtp")
        fb = _make_forward_batch(backend, mode_active=("is_target_verify",))
        out = backend.forward_extend(None, None, None, _make_layer(), fb)
        self.assertEqual(out, "mtp")

    def test_decode_dispatches_to_sparse_on_topk(self):
        backend = _make_backend()
        backend.forward_sparse = MagicMock(return_value="sparse")
        fb = _make_forward_batch(backend)
        out = backend.forward_decode(
            None, None, None, _make_layer(), fb, topk_indices=torch.zeros(1)
        )
        self.assertEqual(out, "sparse")

    def test_decode_dispatches_to_graph_in_graph_mode(self):
        backend = _make_backend()
        backend.graph_mode = True
        backend.forward_decode_graph = MagicMock(return_value="graph")
        fb = _make_forward_batch(backend)
        out = backend.forward_decode(None, None, None, _make_layer(), fb)
        self.assertEqual(out, "graph")

    def test_decode_skips_graph_under_torch_compile(self):
        backend = _make_backend()
        backend.graph_mode = True
        backend.enable_torch_compile = True
        backend.forward_decode_graph = MagicMock()
        backend.forward_metadata = ForwardMetadata(
            seq_lens_cpu_int=torch.tensor([32, 48], dtype=torch.int32),
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
        )
        fb = _make_forward_batch(backend)
        q = torch.randn(2, 8 * 128)
        backend.token_to_kv_pool.get_key_buffer.return_value = torch.zeros(
            4, PAGE_SIZE, 8, 128
        )
        backend.token_to_kv_pool.get_value_buffer.return_value = torch.zeros(
            4, PAGE_SIZE, 8, 128
        )
        backend.forward_decode(q, q, q, _make_layer(), fb, save_kv_cache=False)
        backend.forward_decode_graph.assert_not_called()

    def test_mla_preprocess_disables_save_kv_cache(self):
        backend = _make_backend(_make_model_runner(use_mla=True))
        backend.forward_sparse = MagicMock(return_value="x")
        fb = _make_forward_batch(backend)
        with patch.object(
            ascend_backend, "is_mla_preprocess_enabled", return_value=True
        ):
            backend.forward_decode(
                None, None, None, _make_layer(), fb, topk_indices=torch.zeros(1)
            )
        # save_kv_cache positional arg forced to False
        self.assertFalse(backend.forward_sparse.call_args.args[5])


# ---------------------------------------------------------------------------
# forward_sparse seq-length selection
# ---------------------------------------------------------------------------


class TestForwardSparse(CustomTestCase):
    def _run(self, mode_active, T=4, metadata_q=None, extend_seq_lens=None):
        backend = _make_backend(_make_model_runner(use_mla=True))
        backend.speculative_num_draft_tokens = 2
        H = 16
        q = torch.randn(T, H, 512)
        q_rope = torch.randn(T, H, 64)
        k = torch.randn(T, 1, 512)
        k_rope = torch.randn(T, 1, 64)
        backend.token_to_kv_pool.get_kv_buffer.return_value = (
            torch.randn(8, 512),
            torch.randn(8, 64),
        )
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([10] * 2, dtype=torch.int32),
            actual_seq_lengths_q=metadata_q,
        )
        fb = _make_forward_batch(
            backend, mode_active=mode_active, extend_seq_lens=extend_seq_lens
        )
        sparse_mock = _torch_npu_stub.npu_sparse_flash_attention
        sparse_mock.reset_mock()
        sparse_mock.return_value = (torch.zeros(T, H, 512), None, None)
        layer = _make_layer(tp_q_head_num=H, tp_k_head_num=1)
        backend.forward_sparse(
            q,
            k,
            None,
            layer,
            fb,
            save_kv_cache=True,
            q_rope=q_rope,
            k_rope=k_rope,
            topk_indices=torch.zeros(T, 1, dtype=torch.int32),
        )
        return backend, sparse_mock

    def test_prefill_uses_cumsum_extend_lens(self):
        _, mock = self._run(("is_extend",), T=12, extend_seq_lens=(4, 8))
        qlen = mock.call_args.kwargs["actual_seq_lengths_query"]
        self.assertEqual(qlen.tolist(), [4, 12])

    def test_decode_uses_arange(self):
        _, mock = self._run(("is_decode_or_idle",), T=4)
        qlen = mock.call_args.kwargs["actual_seq_lengths_query"]
        self.assertEqual(qlen.tolist(), [1, 2, 3, 4])

    def test_target_verify_uses_draft_token_steps(self):
        _, mock = self._run(("is_target_verify",), T=4)
        qlen = mock.call_args.kwargs["actual_seq_lengths_query"]
        self.assertEqual(qlen.tolist(), [2, 4])

    def test_metadata_q_lens_take_priority(self):
        meta_q = torch.tensor([3, 4], dtype=torch.int32)
        _, mock = self._run(("is_decode_or_idle",), T=4, metadata_q=meta_q)
        qlen = mock.call_args.kwargs["actual_seq_lengths_query"]
        self.assertEqual(qlen.tolist(), [3, 4])

    def test_kv_lens_from_cpu_int(self):
        _, mock = self._run(("is_decode_or_idle",), T=4)
        kvlen = mock.call_args.kwargs["actual_seq_lengths_kv"]
        self.assertEqual(kvlen.tolist(), [10, 10])

    def test_save_kv_cache_writes_pool(self):
        backend, _ = self._run(("is_decode_or_idle",), T=4)
        backend.token_to_kv_pool.set_kv_buffer.assert_called_once()


# ---------------------------------------------------------------------------
# forward_decode kernel-arg construction
# ---------------------------------------------------------------------------


class TestForwardDecodePaths(CustomTestCase):
    def test_non_mla_paged_attention(self):
        backend = _make_backend()
        H, D, bs = 8, 128, 2
        layer = _make_layer(tp_q_head_num=H, tp_k_head_num=H, qk_head_dim=D)
        block_tables = torch.zeros(bs, 4, dtype=torch.int32)
        seq_lens_cpu_int = torch.tensor([32, 48], dtype=torch.int32)
        backend.forward_metadata = ForwardMetadata(
            block_tables=block_tables, seq_lens_cpu_int=seq_lens_cpu_int
        )
        q = torch.randn(bs, H * D)
        k = torch.randn(bs, H, D)
        k_cache = torch.zeros(4, PAGE_SIZE, H, D)
        backend.token_to_kv_pool.get_key_buffer.return_value = k_cache
        backend.token_to_kv_pool.get_value_buffer.return_value = k_cache
        fb = _make_forward_batch(backend)
        paged = _torch_npu_stub._npu_paged_attention
        paged.reset_mock()
        out = backend.forward_decode(q, k, k, layer, fb)
        kwargs = paged.call_args.kwargs
        self.assertIs(kwargs["block_table"], block_tables)
        self.assertIs(kwargs["context_lens"], seq_lens_cpu_int)
        self.assertEqual(kwargs["num_heads"], H)
        self.assertEqual(out.shape, (bs, H * D))
        backend.token_to_kv_pool.set_kv_buffer.assert_called_once()

    def test_non_mla_cross_attention_uses_encoder_cache_loc(self):
        backend = _make_backend()
        H, D, bs = 8, 128, 2
        layer = _make_layer(tp_q_head_num=H, qk_head_dim=D)
        layer.is_cross_attention = True
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(bs, 4, dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([32, 48], dtype=torch.int32),
        )
        q = torch.randn(bs, H * D)
        k = torch.randn(bs, H, D)
        k_cache = torch.zeros(4, PAGE_SIZE, H, D)
        backend.token_to_kv_pool.get_key_buffer.return_value = k_cache
        backend.token_to_kv_pool.get_value_buffer.return_value = k_cache
        fb = _make_forward_batch(backend)
        fb.encoder_out_cache_loc = torch.tensor([7, 8])
        fb.encoder_lens = None
        backend.forward_decode(q, k, k, layer, fb)
        write_loc = backend.token_to_kv_pool.set_kv_buffer.call_args.args[1]
        self.assertTrue(torch.equal(write_loc.loc, fb.encoder_out_cache_loc))

    def test_mla_paged_attention_mla(self):
        backend = _make_backend(_make_model_runner(use_mla=True))
        H = 16
        layer = _make_layer(tp_q_head_num=H, tp_k_head_num=1, qk_head_dim=576)
        layer.head_dim = 576
        bs = 2
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(bs, 4, dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([16, 16], dtype=torch.int32),
        )
        q = torch.randn(bs, H, 512)
        q_rope = torch.randn(bs, H, 64)
        k = torch.randn(bs, 1, 512)
        k_rope = torch.randn(bs, 1, 64)
        kv_c = torch.randn(2 * PAGE_SIZE, 1, 512)
        k_pe = torch.randn(2 * PAGE_SIZE, 1, 64)
        backend.token_to_kv_pool.get_key_buffer.return_value = kv_c
        backend.token_to_kv_pool.get_value_buffer.return_value = k_pe
        fb = _make_forward_batch(backend)
        mla_mock = _torch_npu_stub._npu_paged_attention_mla
        mla_mock.reset_mock()
        out = backend.forward_decode(
            q, k, None, layer, fb, q_rope=q_rope, k_rope=k_rope
        )
        kwargs = mla_mock.call_args.kwargs
        self.assertEqual(kwargs["mla_vheadsize"], 512)
        self.assertEqual(kwargs["num_heads"], H)
        # q and q_rope concatenated -> last dim 576
        self.assertEqual(kwargs["query"].shape, (bs, H, 576))
        self.assertEqual(kwargs["key_cache"].shape, (2, PAGE_SIZE, 1, 576))
        self.assertEqual(out.shape, (bs, H * 512))

    def test_decode_save_kv_cache_skipped_when_k_none(self):
        backend = _make_backend()
        H, D, bs = 8, 128, 2
        layer = _make_layer(tp_q_head_num=H, qk_head_dim=D)
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(bs, 4, dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([32, 48], dtype=torch.int32),
        )
        q = torch.randn(bs, H * D)
        k_cache = torch.zeros(4, PAGE_SIZE, H, D)
        backend.token_to_kv_pool.get_key_buffer.return_value = k_cache
        backend.token_to_kv_pool.get_value_buffer.return_value = k_cache
        fb = _make_forward_batch(backend)
        backend.forward_decode(q, None, None, layer, fb)
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()


# ---------------------------------------------------------------------------
# forward_extend kernel-arg construction (non-MLA, non-FIA qlens path)
# ---------------------------------------------------------------------------


class TestForwardExtendQlens(CustomTestCase):
    def test_qlens_kernel_args(self):
        backend = _make_backend()
        H, D = 8, 128
        layer = _make_layer(tp_q_head_num=H, tp_k_head_num=H, qk_head_dim=D)
        extend_lens = torch.tensor([4, 4], dtype=torch.int32)
        seq_lens_cpu_int = torch.tensor([4, 4], dtype=torch.int32)
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens_cpu_int=seq_lens_cpu_int,
            extend_seq_lens_cpu_int=extend_lens,
        )
        T = 8
        q = torch.randn(T, H * D)
        k = torch.randn(T, H, D)
        k_cache = torch.zeros(4, PAGE_SIZE, H, D)
        backend.token_to_kv_pool.get_key_buffer.return_value = k_cache
        backend.token_to_kv_pool.get_value_buffer.return_value = k_cache
        fb = _make_forward_batch(
            backend, mode_active=("is_extend",), extend_seq_lens=(4, 4)
        )
        qlens_mock = _torch_npu_stub._npu_flash_attention_qlens
        qlens_mock.reset_mock()
        out = backend.forward_extend(q, k, k, layer, fb)
        kwargs = qlens_mock.call_args.kwargs
        self.assertIs(kwargs["seq_len"], extend_lens)
        self.assertIs(kwargs["context_lens"], seq_lens_cpu_int)
        self.assertIs(kwargs["mask"], backend.mask)
        self.assertEqual(out.shape, (T, H * D))
        backend.token_to_kv_pool.set_kv_buffer.assert_called_once()

    def test_encoder_only_falls_back_to_native_sdpa(self):
        backend = _make_backend()
        from sglang.srt.layers.radix_attention import AttentionType

        H, D = 8, 128
        layer = _make_layer(tp_q_head_num=H, tp_k_head_num=H, qk_head_dim=D)
        layer.attn_type = AttentionType.ENCODER_ONLY
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([4, 4], dtype=torch.int32),
        )
        backend.native_attn = MagicMock()
        backend.native_attn.run_sdpa_forward_extend.return_value = torch.zeros(8, H, D)
        T = 8
        q = torch.randn(T, H * D)
        k = torch.randn(T, H, D)
        k_cache = torch.zeros(4 * PAGE_SIZE, H, D)
        backend.token_to_kv_pool.get_key_buffer.return_value = k_cache
        backend.token_to_kv_pool.get_value_buffer.return_value = k_cache
        fb = _make_forward_batch(
            backend, mode_active=("is_extend",), extend_seq_lens=(4, 4)
        )
        backend.forward_extend(q, k, k, layer, fb)
        kwargs = backend.native_attn.run_sdpa_forward_extend.call_args.kwargs
        self.assertFalse(kwargs["causal"])


# ---------------------------------------------------------------------------
# Alibi math
# ---------------------------------------------------------------------------


class TestAlibi(CustomTestCase):
    def test_generate_alibi_bias_values(self):
        backend = _make_backend()
        slopes = torch.tensor([0.5, 0.25])
        bias = backend.generate_alibi_bias(
            q_seq_len=3,
            kv_seq_len=4,
            slopes=slopes,
            num_heads=2,
            device="cpu",
            is_extend=False,
            dtype=torch.float32,
        )
        self.assertEqual(bias.shape, (2, 1, 4))
        self.assertTrue(
            torch.allclose(bias[0, 0], 0.5 * torch.arange(4, dtype=torch.float32))
        )
        # extend variant applies a causal -inf upper triangle
        bias_ext = backend.generate_alibi_bias(
            q_seq_len=3,
            kv_seq_len=3,
            slopes=slopes,
            num_heads=2,
            device="cpu",
            is_extend=True,
            dtype=torch.float32,
        )
        self.assertEqual(bias_ext.shape, (2, 3, 3))
        self.assertEqual(bias_ext[0, 0, 1].item(), float("-inf"))
        self.assertNotEqual(bias_ext[0, 1, 1].item(), float("-inf"))

    def test_attn_alibi_decode_matches_reference(self):
        backend = _make_backend()
        torch.manual_seed(0)
        H, D, seq_len = 2, 8, 6
        num_blocks, block_size = 2, 4
        k_cache = torch.randn(num_blocks, block_size, H, D)
        v_cache = torch.randn(num_blocks, block_size, H, D)
        q = torch.randn(1, H, D)
        block_tables = torch.tensor([[0, 1]])
        seq_lens = torch.tensor([seq_len])
        scale = D**-0.5
        out = backend.attn_alibi(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_lens=torch.ones(1, dtype=torch.int32),
            scale_value=scale,
            num_heads=H,
            slopes=None,
            is_extend=False,
        )
        # reference: flat attention over the first seq_len cached tokens
        k = k_cache.view(-1, H, D)[:seq_len]
        v = v_cache.view(-1, H, D)[:seq_len]
        score = torch.einsum("hd,shd->hs", q[0] * scale, k.float())
        p = torch.softmax(score.float(), dim=-1)
        ref = torch.einsum("hs,shd->hd", p, v.float()).reshape(1, H * D)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


# ---------------------------------------------------------------------------
# CP all-gather KV save helper
# ---------------------------------------------------------------------------


class TestCpAllgatherAndSaveKv(CustomTestCase):
    def test_merged_gather_round_trip(self):
        s_local, hk, hv, dk, dv = 4, 2, 2, 16, 8
        cp_size = 2
        k = torch.randn(s_local, hk, dk)
        v = torch.randn(s_local, hv, dv)
        fb = MagicMock()
        fb.out_cache_loc = torch.arange(s_local * cp_size)
        layer = _make_layer(tp_q_head_num=hk, tp_k_head_num=hk)
        layer.is_cross_attention = False
        pool = MagicMock()

        def fake_gather(t, cp, fb_, stream):
            return torch.cat([t] * cp, dim=0)

        with patch.object(
            ascend_backend, "cp_all_gather_rerange_kv_cache", side_effect=fake_gather
        ), patch.object(
            ascend_backend, "get_current_device_stream_fast", return_value=None
        ):
            _cp_allgather_and_save_kv_npu(fb, layer, k, v, cp_size, pool, swa_loc=None)

        args = pool.set_kv_buffer.call_args.args
        k_full, v_full = args[2], args[3]
        self.assertEqual(k_full.shape, (s_local * cp_size, hk, dk))
        self.assertEqual(v_full.shape, (s_local * cp_size, hv, dv))
        self.assertTrue(torch.allclose(k_full[:s_local], k))
        self.assertTrue(torch.allclose(v_full[s_local:], v))
        self.assertTrue(torch.equal(args[1].loc, fb.out_cache_loc))


# ---------------------------------------------------------------------------
# forward_mixed guard rails
# ---------------------------------------------------------------------------


class TestForwardMixed(CustomTestCase):
    def test_rejects_mla(self):
        backend = _make_backend(_make_model_runner(use_mla=True))
        with self.assertRaises(NotImplementedError):
            backend.forward_mixed(
                None, None, None, _make_layer(), MagicMock(), topk_indices=None
            )

    def test_rejects_large_head_dim_without_fia(self):
        backend = _make_backend()
        self.assertFalse(backend.use_fia)
        with self.assertRaises(NotImplementedError):
            backend.forward_mixed(
                None, None, None, _make_layer(qk_head_dim=192), MagicMock()
            )

    def test_mixed_kernel_args(self):
        backend = _make_backend()
        H, D = 8, 128
        layer = _make_layer(tp_q_head_num=H, tp_k_head_num=H, qk_head_dim=D)
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([8, 8], dtype=torch.int32),
            seq_lens_list_cumsum=[4, 8],
        )
        k_cache = torch.zeros(4, PAGE_SIZE, H, D)
        backend.token_to_kv_pool.get_key_buffer.return_value = k_cache
        backend.token_to_kv_pool.get_value_buffer.return_value = k_cache
        fia = torch.ops.npu.npu_fused_infer_attention_score
        fia.reset_mock()
        fia.return_value = (torch.zeros(8, H, D), None)
        q = torch.randn(8, H * D)
        fb = _make_forward_batch(backend)
        out = backend.forward_mixed(q, q.view(8, H, D), q.view(8, H, D), layer, fb)
        kwargs = fia.call_args.kwargs
        self.assertIs(kwargs["atten_mask"], backend.mix_mask)
        self.assertEqual(kwargs["actual_seq_lengths"], [4, 8])
        self.assertEqual(out.shape, (8, H * D))


# ---------------------------------------------------------------------------
# Multi-step draft backend
# ---------------------------------------------------------------------------


class TestMultiStepDraftBackend(CustomTestCase):
    def _make(self, num_steps=3):
        mr = _make_model_runner(speculative_num_draft_tokens=2)
        with patch.object(
            ascend_backend, "get_bool_env_var", return_value=False
        ), patch.object(
            ascend_backend.DllmConfig, "from_server_args", return_value=None
        ), patch.object(
            ascend_backend, "get_attention_tp_size", return_value=1
        ), patch.object(
            torch, "tensor", _cpu_tensor
        ):
            return AscendAttnMultiStepDraftBackend(
                mr, topk=1, speculative_num_steps=num_steps
            )

    def test_creates_one_backend_per_step(self):
        multi = self._make(3)
        self.assertEqual(len(multi.attn_backends), 3)
        for i, b in enumerate(multi.attn_backends):
            self.assertEqual(b.speculative_step_id, i)

    def test_init_forward_metadata_loops_n_minus_one(self):
        multi = self._make(3)
        for b in multi.attn_backends:
            b.init_forward_metadata = MagicMock()
        fb = MagicMock()
        fb.spec_info = MagicMock()
        multi.init_forward_metadata(fb)
        multi.attn_backends[0].init_forward_metadata.assert_called_once()
        multi.attn_backends[1].init_forward_metadata.assert_called_once()
        multi.attn_backends[2].init_forward_metadata.assert_not_called()

    def test_init_cuda_graph_state_loops_all(self):
        multi = self._make(2)
        for b in multi.attn_backends:
            b.init_cuda_graph_state = MagicMock()
        multi.init_cuda_graph_state(4, 64)
        for b in multi.attn_backends:
            b.init_cuda_graph_state.assert_called_once_with(4, 64)


if __name__ == "__main__":
    unittest.main()
