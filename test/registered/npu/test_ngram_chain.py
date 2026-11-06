"""Real NPU MTP verifier contract used by eager single-chain NGRAM."""

import sys

import numpy as np
import pytest
import torch

from sglang.srt.speculative.ngram_chain import (
    bound_chain_acceptance,
    chain_lengths,
    prepare_chain_metadata,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("mismatch", [None, 1, 2])
def test_npu_greedy_acceptance_is_an_in_place_prefix(width, mismatch):
    pytest.importorskip("torch_npu")
    from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

    tokens = np.tile(np.arange(10, 10 + width), (3, 1))
    if width > 1:
        tokens[0, 1:] = 0  # no match, including accidentally matching padding
        tokens[1, 2:] = 0  # short match
    lengths = torch.from_numpy(
        chain_lengths(tokens, np.tile(np.tri(width), (3, 1, 1)), width)
    ).to("npu")
    _, indices, next_token, siblings = prepare_chain_metadata(
        torch.tensor([3, 7, 11], device="npu"), lengths, width
    )
    candidates = torch.from_numpy(tokens).to("npu")
    target = torch.zeros_like(candidates)
    if width > 1:
        target[:, :-1] = candidates[:, 1:]
    target[:, -1] = 99
    if mismatch is not None and mismatch < width:
        target[:, mismatch - 1] = 127
    predicts = torch.zeros(3 * width, dtype=torch.int32, device="npu")
    accepted = torch.full((3, width), -1, dtype=torch.int32, device="npu")
    counts = torch.empty(3, dtype=torch.int32, device="npu")
    verify_tree_greedy(
        predicts, accepted, counts, candidates, indices, next_token, siblings, target
    )
    lens, accepted = bound_chain_acceptance(counts + 1, accepted, lengths)
    for b, count in enumerate(lens.cpu().tolist()):
        expected = lengths[b].item()
        if mismatch is not None and mismatch < width:
            expected = min(expected, mismatch)
        assert count == expected
        assert accepted[b, :count].cpu().tolist() == list(
            range(b * width, b * width + count)
        )
        assert (accepted[b, count:] == -1).all().item()
        assert (
            predicts[b * width : b * width + count].cpu().tolist()
            == target[b, :count].cpu().tolist()
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
