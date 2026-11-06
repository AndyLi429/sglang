"""CPU regression coverage for NPU NGRAM's in-place KV contract.

Execute host methods from source without importing accelerator extensions.
"""

import ast
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.srt.speculative.ngram_chain import (
    bound_chain_acceptance,
    chain_lengths,
    prepare_chain_metadata,
    validate_npu_ngram_request,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


def _method(path, name, *, strip_imports=False, **scope):
    source = ROOT / "python/sglang/srt" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    if strip_imports:
        node.body = [
            n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, node],
        type_ignores=[],
    )
    exec(compile(module, str(source), "exec"), scope)  # noqa: S102
    return scope[name]


def test_short_misses_and_zero_tokens_stop_before_padding():
    tokens = np.array([[9, 2, 3, 4], [9, 2, 0, 0], [9, 0, 0, 0], [0, 2, 0, 4]])
    mask = np.tile(np.tri(4, dtype=bool), (4, 1, 1))
    # Corpus padding is a root child, not a continuation of the previous node.
    mask[1, 3] = [1, 0, 0, 1]
    mask[2, 2] = [1, 0, 1, 0]
    np.testing.assert_array_equal(chain_lengths(tokens, mask, 4), [4, 2, 1, 2])


def test_branching_mask_is_truncated_to_a_causal_prefix():
    mask = np.tri(4, dtype=bool)
    mask[2] = [1, 0, 1, 0]
    assert chain_lengths(np.array([9, 2, 3, 4]), mask, 4).tolist() == [2]


@pytest.mark.parametrize(
    "field",
    [
        None,
        "top_k",
        "return_logprob",
        "regex",
        "json_schema",
        "ebnf",
        "structural_tag",
        "watermark",
    ],
)
def test_requests_are_validated_before_scheduler_dispatch(field):
    params = SimpleNamespace(
        top_k=1,
        regex=None,
        json_schema=None,
        ebnf=None,
        structural_tag=None,
        watermark=None,
    )
    if field not in (None, "return_logprob"):
        setattr(params, field, 4 if field == "top_k" else "enabled")
    if field is None:
        validate_npu_ngram_request(params, False)
    else:
        with pytest.raises(ValueError, match="NPU NGRAM"):
            validate_npu_ngram_request(params, field == "return_logprob")


def test_tokenizer_rejects_sampling_before_building_scheduler_request():
    class GenerateRequest:
        sampling_params = {}
        watermark = None
        max_thinking_tokens = None
        return_logprob = False

    params = SimpleNamespace(
        top_k=8,
        regex=None,
        json_schema=None,
        ebnf=None,
        structural_tag=None,
        watermark=None,
        normalize=Mock(),
        verify=Mock(),
    )
    create = _method(
        "managers/tokenizer_manager.py",
        "_create_tokenized_object",
        array=array,
        GenerateReqInput=GenerateRequest,
        get_device=lambda: SimpleNamespace(device="npu"),
        get_spec=lambda: SimpleNamespace(speculative_algorithm="NGRAM"),
        _validate_watermark_request=Mock(),
    )
    manager = SimpleNamespace(
        preferred_sampling_params=None,
        sampling_params_class=lambda **kwargs: params,
        tokenizer=None,
        model_config=SimpleNamespace(vocab_size=128),
    )
    with pytest.raises(ValueError, match="greedy"):
        create(manager, GenerateRequest(), "prompt", [1, 2])
    params.normalize.assert_called_once()
    params.verify.assert_called_once_with(128)


def test_worker_prepares_npu_chain_without_cuda_reconstruction():
    worker = SimpleNamespace(max_batch_size=2, draft_token_num=4, device="cpu")
    init = _method(
        "speculative/ngram_worker.py", "_init_preallocated_tensors", torch=torch
    )
    init(worker)
    tokens = np.array([9, 2, 0, 0, 8, 3, 4, 5])
    mask = np.tile(np.tri(4), (2, 1, 1))
    worker._prepare_draft_tokens = lambda batch: (tokens, mask.flatten())
    slots = torch.tensor([7, 12, 13, 14, 23, 28, 29, 30], dtype=torch.int32)
    assign = Mock(return_value=slots)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        reqs=[None, None],
        seq_lens=torch.tensor([3, 7]),
        has_grammar=False,
        req_pool_indices=torch.tensor([0, 1]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(2, 16)),
    )
    prepare = _method(
        "speculative/ngram_worker.py",
        "_prepare_for_speculative_decoding",
        torch=torch,
        _is_npu=True,
        _is_cpu=False,
        USE_FULL_MASK=True,
        chain_lengths=chain_lengths,
        prepare_chain_metadata=prepare_chain_metadata,
        reconstruct_indices_from_tree_mask_func=Mock(
            side_effect=AssertionError("CUDA op")
        ),
        assign_extend_cache_locs_func=assign,
        prepare_mamba_track_for_verify=Mock(),
        NgramVerifyInput=SimpleNamespace,
        ForwardMode=SimpleNamespace(TARGET_VERIFY="verify"),
    )
    prepare(worker, batch)
    assert worker.chain_lengths.tolist() == [2, 4]
    assert batch.spec_info.positions.tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
    assert batch.spec_info.retrieve_index.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert batch.out_cache_loc is slots
    assert batch.forward_mode == "verify"
    assert assign.call_args.kwargs["start_offset"].tolist() == [3, 7]
    assert assign.call_args.kwargs["end_offset"].tolist() == [7, 11]


def test_accept_indices_remain_in_place_for_reject_partial_and_full_accept():
    width = 4
    lengths = torch.tensor([1, 2, 4], dtype=torch.int32)
    positions, indices, next_token, siblings = prepare_chain_metadata(
        torch.tensor([3, 7, 11]), lengths, width
    )
    assert positions.tolist() == list(range(3, 7)) + list(range(7, 11)) + list(
        range(11, 15)
    )
    assert next_token.tolist() == [[-1] * 4, [1, -1, -1, -1], [1, 2, 3, -1]]
    assert (siblings == -1).all()
    # The fixed-width MTP kernel may accept every zero padding proposal.
    for raw_lens in ([1, 1, 1], [4, 3, 2], [4, 4, 4]):
        lens, accepted = bound_chain_acceptance(
            torch.tensor(raw_lens), indices, lengths
        )
        for b, count in enumerate(lens.tolist()):
            assert accepted[b, :count].tolist() == list(
                range(b * width, b * width + count)
            )
            assert (accepted[b, count:] == -1).all()
            # Nonconsecutive physical pages still map source and target identically.
            slots = torch.tensor([20, 21, 40, 41, 60, 61, 80, 81, 100, 101, 120, 121])
            assert torch.equal(
                slots[accepted[b, :count]], slots[b * width : b * width + count]
            )


@pytest.mark.parametrize("accepted", [1, 2, 4])
@pytest.mark.parametrize("finished_end", [None, 4])
def test_kv_release_keeps_live_pages_then_reclaims_all_reserved_tail(
    accepted, finished_end
):
    # Logical row crosses discontiguous pages. A second request owns page 7.
    page_size = 4
    row = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23])
    live_pages = {1, 3, 5, 7}
    freed_pages = []

    def free_row(kv, spans):
        for start, end in spans:
            for page in torch.unique(row[start:end] // page_size).tolist():
                assert page in live_pages, "double free or cross-request corruption"
                live_pages.remove(page)
                freed_pages.append(page)

    kv = SimpleNamespace(
        holds_kv=True,
        is_kv_released=False,
        kv_committed_len=3,
        kv_allocated_len=12,
        cache_protected_len=0,
    )
    commit = _method("managers/schedule_batch.py", "commit_decode")
    commit(kv, accepted)
    assert live_pages == {1, 3, 5, 7}  # rejected slots remain reserved
    owned = (
        min(kv.kv_committed_len, finished_end) if finished_end else kv.kv_committed_len
    )
    req = SimpleNamespace(kv=kv, lock=None, owned_kv_len=lambda: owned)

    def mark_released():
        kv.holds_kv = False
        kv.is_kv_released = True

    kv.mark_kv_released = mark_released
    tree = SimpleNamespace(
        claim_kv_row=lambda req: False,
        free_kv_row=free_row,
        unlock=Mock(),
        on_release=Mock(),
        logical_page_size=page_size,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    )
    tail = _method(
        "mem_cache/common.py",
        "_release_overallocated_kv_indices",
        get_spec=lambda: SimpleNamespace(speculative_algorithm="NGRAM"),
        ceil_align=lambda value, alignment: (
            (value + alignment - 1) // alignment * alignment
        ),
    )
    storage = SimpleNamespace(release_row=Mock())
    release = _method(
        "mem_cache/common.py",
        "release_kv_cache",
        _release_overallocated_kv_indices=tail,
        kv_storage_of=lambda *args, **kwargs: storage,
    )
    release(req, tree, checkpoint=False)
    assert live_pages == {7}
    assert sorted(freed_pages) == [1, 3, 5]
    assert kv.is_kv_released
    storage.release_row.assert_called_once_with(req)


def test_worker_skips_mover_and_bounds_padding_before_sequence_commit():
    width = 4
    indices = torch.arange(8, dtype=torch.int32).view(2, width)
    # predict contains the target's next token at each input position, even if
    # the fixed-width kernel accepted beyond the real corpus chain.
    predict = torch.tensor([2, 3, 0, 0, 6, 7, 8, 9], dtype=torch.int32)
    mover = Mock(side_effect=AssertionError("a chain must not move MLA KV"))
    worker = SimpleNamespace(
        device="cpu",
        draft_token_num=width,
        speculative_num_draft_tokens=width,
        chain_lengths=torch.tensor([2, 4], dtype=torch.int32),
        model_runner=SimpleNamespace(watermark_state=None),
        target_worker=SimpleNamespace(
            forward_batch_generation=Mock(
                return_value=SimpleNamespace(
                    logits_output=SimpleNamespace(next_token_logits=torch.zeros(8, 16)),
                    can_run_cuda_graph=False,
                )
            )
        ),
        _prepare_for_speculative_decoding=Mock(),
        _update_ngram_corpus=Mock(),
        _prev_decode_rids=set(),
    )
    batch = SimpleNamespace(
        sampling_info=SimpleNamespace(is_all_greedy=True),
        has_grammar=False,
        return_logprob=False,
        spec_info=None,
        seq_lens=torch.tensor([3, 7]),
        reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")],
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
    )
    forward = _method(
        "speculative/ngram_worker.py",
        "forward_batch_generation",
        torch=torch,
        _is_npu=True,
        record_stream_for_v2_verify=Mock(),
        set_time_batch=Mock(),
        maybe_detect_nan=Mock(),
        maybe_detect_inf=Mock(),
        eagle_sample=lambda *args, **kwargs: (predict, torch.tensor([4, 4]), indices),
        bound_chain_acceptance=bound_chain_acceptance,
        commit_mamba_states_after_verify=Mock(),
        move_accept_tokens_to_target_kvcache=mover,
        NgramVerifyInput=SimpleNamespace,
        GenerationBatchResult=SimpleNamespace,
        ForwardMode=SimpleNamespace(DECODE="decode"),
    )
    result = forward(worker, batch)
    assert result.accept_lens.tolist() == [2, 4]
    assert result.new_seq_lens.tolist() == [5, 11]
    assert result.next_token_ids[:2].tolist() == [2, 3]
    assert result.next_token_ids[4:].tolist() == [6, 7, 8, 9]
    mover.assert_not_called()


@pytest.mark.parametrize(
    "greedy,grammar,logprob",
    [(False, False, False), (True, True, False), (True, False, True)],
)
def test_worker_rejects_unsupported_requests_before_forward(greedy, grammar, logprob):
    forward = _method(
        "speculative/ngram_worker.py", "forward_batch_generation", _is_npu=True
    )
    batch = SimpleNamespace(
        sampling_info=SimpleNamespace(is_all_greedy=greedy),
        has_grammar=grammar,
        return_logprob=logprob,
    )
    with pytest.raises(ValueError, match="NPU NGRAM"):
        forward(SimpleNamespace(), batch)


def _npu_config(**changes):
    defaults = {
        "speculative_ngram_min_bfs_breadth": 1,
        "speculative_ngram_max_bfs_breadth": 1,
        "speculative_ngram_external_corpus_path": None,
        "attn_cp_size": 1,
        "attn_dcp_size": 1,
        "pp_size": 1,
        "disaggregation_mode": "null",
        "enable_torch_compile": False,
        "speculative_adaptive": False,
        "enable_two_batch_overlap": False,
        "speculative_use_rejection_sampling": False,
        "enable_unified_memory": False,
        "cuda_graph_config": {"decode": "full", "prefill": "full"},
    }
    defaults.update(changes)
    return SimpleNamespace(**defaults)


def _config_handler(
    cfg, *, architecture="DeepseekV3ForCausalLM", dsa=False, backend="ascend", dp=False
):
    declared = {}
    handler = _method(
        "arg_groups/speculative_hook.py",
        "_handle_npu_ngram",
        strip_imports=True,
        AttentionArch=SimpleNamespace(MLA="mla"),
        Backend=SimpleNamespace(DISABLED="disabled"),
        Phase=SimpleNamespace(ALL=("decode", "prefill")),
        with_phase=lambda config, phase, backend: {**config, phase: backend},
        resolving_view=lambda args: cfg,
        model_config_of=lambda args: SimpleNamespace(
            attention_arch="mla",
            hf_config=SimpleNamespace(architectures=[architecture]),
        ),
        is_deepseek_dsa=lambda hf: dsa,
        attention_backends_of=lambda cfg: (backend, backend),
        attn_dp_enabled_of=lambda cfg: dp,
        envs=SimpleNamespace(
            SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=SimpleNamespace(get=lambda: False),
            SGLANG_SIMULATE_ACC_LEN=SimpleNamespace(get=lambda: -1),
        ),
        declare_resolution=lambda args, owner, **kwargs: declared.update(kwargs),
        logger=SimpleNamespace(warning=Mock()),
    )
    handler(cfg)
    return declared


def test_npu_config_disables_both_graph_phases_and_overlap():
    changes = _config_handler(_npu_config())
    assert changes["disable_overlap_schedule"] is True
    assert changes["cuda_graph_config"] == {"decode": "disabled", "prefill": "disabled"}


@pytest.mark.parametrize(
    "change",
    [
        {"speculative_ngram_max_bfs_breadth": 2},
        {"speculative_ngram_min_bfs_breadth": 2},
        {"speculative_ngram_external_corpus_path": "corpus.txt"},
        {"attn_cp_size": 2},
        {"attn_dcp_size": 2},
        {"pp_size": 2},
        {"disaggregation_mode": "decode"},
        {"enable_torch_compile": True},
        {"speculative_adaptive": True},
        {"enable_two_batch_overlap": True},
        {"speculative_use_rejection_sampling": True},
        {"enable_unified_memory": True},
    ],
)
def test_npu_config_rejects_unsupported_modes(change):
    with pytest.raises(ValueError, match="NPU NGRAM"):
        _config_handler(_npu_config(**change))


@pytest.mark.parametrize(
    "change",
    [
        {"architecture": "DeepseekV4ForCausalLM"},
        {"dsa": True},
        {"backend": "flashinfer"},
        {"dp": True},
    ],
)
def test_npu_config_rejects_other_models_and_backends(change):
    with pytest.raises(ValueError, match="NPU NGRAM"):
        _config_handler(_npu_config(), **change)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
