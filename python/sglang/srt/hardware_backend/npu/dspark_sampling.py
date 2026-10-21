"""DSPark NPU sampling orchestration; compute kernels live in sgl-kernel-npu."""

from __future__ import annotations

from typing import Optional

import torch
from sgl_kernel_npu.sample.chain_speculative_sampling import (
    chain_speculative_sampling_triton,
)
from sgl_kernel_npu.sample.dspark_accept import (
    CapCorrectLen,
    SoftmaxTemp,
    gather_two_level_bonus_triton,
)

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_speculative_verify_target_probs,
)


def _accept_sampling_core(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
    block_verification: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        target_probs = SoftmaxTemp.execute(
            logits=target_logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=verify_num_draft_tokens,
        ).view(bs, verify_num_draft_tokens, -1)
    else:
        target_probs = build_speculative_verify_target_probs(
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_token_num=verify_num_draft_tokens,
            bs=bs,
            max_top_k=draft_input.max_top_k,
            uniform_top_k_value=draft_input.uniform_top_k_value,
        )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    # NPU sampling uses the full candidate width as its random-buffer stride.
    uniform_width = candidates.shape[1]
    uniform_samples = torch.rand(
        (bs, uniform_width), dtype=torch.float32, device=device
    )
    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
    sampling_fn = chain_speculative_sampling_triton
    if block_verification:
        raise NotImplementedError(
            "NPU chain sampling does not support block verification"
        )
    sampling_fn(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    correct_len = accept_token_num
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
    else:
        cap_trim_lens = torch.zeros_like(correct_len)
    return correct_len, cap_trim_lens, accept_index, predicts


def accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
    block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
        block_verification=block_verification,
    )
    bonus = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    return correct_len, bonus, cap_trim_lens


class AcceptSampling:
    execute = staticmethod(accept_sampling)
    triton = staticmethod(accept_sampling)
