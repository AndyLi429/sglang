"""Host metadata and acceptance bounds for eager NPU NGRAM chains."""

import numpy as np
import torch


def validate_npu_ngram_request(sampling_params, return_logprob: bool):
    if sampling_params.top_k != 1:
        raise ValueError("NPU NGRAM only supports greedy sampling (temperature=0).")
    if return_logprob or any(
        getattr(sampling_params, name)
        for name in ("json_schema", "regex", "ebnf", "structural_tag", "watermark")
    ):
        raise ValueError(
            "NPU NGRAM does not yet support grammar, logprobs, or watermarking."
        )


def chain_lengths(draft_tokens: np.ndarray, mask: np.ndarray, width: int):
    """Keep only the contiguous causal prefix of each corpus result.

    The corpus pads with token 0 and root-child masks. A padded first child is
    indistinguishable from a real zero token, so conservatively stop before any
    zero proposal (the root is always retained).
    """
    tokens = draft_tokens.reshape(-1, width)
    tree = mask.reshape(-1, width, width)
    causal = np.tri(width, dtype=bool)
    valid = (tree == causal).all(axis=-1)
    valid[:, 1:] &= tokens[:, 1:] != 0
    valid[:, 0] = True
    return np.logical_and.accumulate(valid, axis=1).sum(axis=1).astype(np.int32)


def prepare_chain_metadata(seq_lens: torch.Tensor, lengths: torch.Tensor, width: int):
    offsets = torch.arange(width, dtype=torch.int64, device=seq_lens.device)
    positions = (seq_lens[:, None] + offsets).flatten()
    retrieve_index = torch.arange(
        seq_lens.numel() * width, dtype=torch.int64, device=seq_lens.device
    ).view(-1, width)
    next_token = torch.where(offsets + 1 < lengths[:, None], offsets + 1, -1)
    next_sibling = torch.full_like(retrieve_index, -1)
    return positions, retrieve_index, next_token, next_sibling


def bound_chain_acceptance(accept_lens, accept_index, lengths):
    """The NPU MTP verifier scans the full window, including corpus padding."""
    accept_lens = torch.minimum(accept_lens, lengths)
    offsets = torch.arange(accept_index.shape[1], device=accept_index.device)
    accept_index = accept_index.masked_fill(offsets >= accept_lens[:, None], -1)
    return accept_lens, accept_index
