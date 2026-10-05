# Atlas A3 MegaMoE integration and Qwen3 acceptance test

The SGLang entry point is `--moe-a2a-backend ascend_fuseep --fuseep-mode 3`.
The existing `megamoe` backend selects a different implementation and must not
be used for this Ascend adapter. Mode 3 calls DeepEP-Ascend's
`Buffer.fused_deep_moe`: dispatch, both expert projections, activation and
combine execute inside MegaMoE. Models retain their gate, TopK, shared experts
and routed-output scaling.

## Prerequisites

- Atlas A3 and compatible CANN, torch-npu and HCCL installations.
- A DeepEP-Ascend build containing [sgl-kernel-npu PR #816](https://github.com/sgl-project/sgl-kernel-npu/pull/816).
  This adapter targets its API at commit `3102ae0296644ed25c4a3564942ffdec655f09ff`.
  Older builds fail during weight preparation with a dependency error.
- `cann_ops_transformer.ops.mega_moe` and
  `get_symm_buffer_for_mega_moe`, plus `ninja` for their JIT build. Follow the
  kernel PR's installation instructions for the matching CANN operator package.
- Four free A3 NPUs and the ModelSlim Qwen3-30B-A3B W8A8 checkpoint at
  `MODEL_WEIGHTS_DIR/Qwen/Qwen3-30B-A3B-w8a8`, as defined by the existing
  `QWEN3_30B_A3B_W8A8_WEIGHTS_PATH` test constant.
- The normal SGLang test dependencies and GSM8K evaluation data.

## Supported contract

Support is checked against each MoE layer rather than a model-name whitelist:

| Property | Initial contract |
| --- | --- |
| Execution | Eager; static, evenly partitioned EP; attention TP with DP=CP=PP=1 |
| Activation dtype | BF16 |
| Expert weights | BF16, per-channel W8A8 INT8, per-channel W4A8 signed INT4 |
| Routed expert FFN | Gated SwiGLU, or SiTU with explicit positive activation parameters; no projection bias |
| Dimensions | H <= 8192, I <= 3584; aligned to 32, or 64 for W4A8 |
| Routing | Standard global expert ids, K <= 16, E <= 1024; router weights applied to outputs |
| Shared experts | Separate; shared-expert fusion is automatically disabled |
| Token capacity | Explicit scheduler bounds covered by an operator capacity in [1, 4096], including TP alignment |

FP4/MXFP4, groupwise W4A8, activation clipping, custom/clamped SwiGLU,
expert LoRA, EPLB, redundant experts, speculative decoding, TBO/SBO,
mixed/dynamic chunking, CPU weight offloading and memory-saver are rejected.
Attention DP/CP needs an extension of the kernel API to accept SGLang's actual
active-token mask. The current API derives that mask from input row count.
TP padding routes marked `-1` are converted to zero-weight routes; completely
inactive rows are zeroed before the call. Empty ranks still enter the collective.

Prepared per-expert weights are stored separately from checkpoint parameters.
This preserves the checkpoint layout but increases model memory use. Lower
`--mem-fraction-static` if necessary. Online weight updates and release/resume
are outside this version's acceptance scope.

W8A8 scales encode float32 bits into uint64. W4A8 scales clear the low 13
mantissa bits and set marker bit 46; compensation is
`8 * sum_K(signed_int4_weight * decoded_scale)`. Each expert is transposed and
converted to FRACTAL_NZ independently; W4A8 is then reinterpreted as int32.
The operator's symmetric communication buffer belongs to the runtime EP group,
and is initialized by an empty collective after both projections are prepared.

## Launch

```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 \
sglang serve /path/to/Qwen3-30B-A3B-w8a8 \
  --device npu --dtype bfloat16 --tp-size 4 \
  --attention-backend ascend --moe-runner-backend ascend \
  --moe-a2a-backend ascend_fuseep --fuseep-mode 3 \
  --disable-prefill-cuda-graph --disable-decode-cuda-graph \
  --chunked-prefill-size 256 --max-prefill-tokens 512 \
  --max-running-requests 8 --mem-fraction-static 0.75
```

There is no rank-local fallback: switching one rank to a different collective
could hang the remaining ranks. Startup validation bounds the scheduler;
oversized unbounded prefill and unsupported execution modes fail explicitly.

## Tests

Run CPU contracts in the usual Linux SGLang test environment:

```bash
python test/registered/unit/moe/test_ascend_mega_moe.py
```

Run the full model test on A3 with the dependencies above:

```bash
python test/registered/e2e/moe/test_ascend_mega_moe_qwen3.py
```

It is registered with the `full-4-npu-a3` nightly suite. The test launches the
DeepEP normal reference and mode 3 sequentially using the same checkpoint,
BF16 dtype, TP4/EP4 topology and scheduler limits. It terminates each server
before loading the next one.

| Case | Acceptance condition |
| --- | --- |
| Mode selection | `/server_info` reports `ascend_fuseep` and mode 3 |
| Single request, mixed-length batch | 16 generated tokens per request, nonempty output, finite output logprobs |
| Long prompt > 512 tokens | Completes through 256-token chunked prefill; input logprobs returned |
| 12 concurrent requests, running limit 8 | All finish within the HTTP timeout; repeated prefill/decode collectives do not hang |
| Fixed-input comparison | Identical input token ids and logprob positions; finite values; logprob RMSE <= 0.1 and max error <= 0.5 |
| GSM8K, 200 questions, 5 shots | Both backends >= 0.90 accuracy; mode 3 loses at most 0.02 absolute accuracy |

Logprob tolerances are initial acceptance targets and need calibration on A3;
they are not measured results. Generated continuations are checked for liveness,
while numerical comparison uses teacher-forced input tokens to avoid comparing
logprobs after different generated prefixes. Evaluation metrics for both
backends and the numerical error summary are printed by the test. Performance
metrics are recorded but have no unmeasured speedup threshold.

The CPU suite independently checks bit encodings and signed INT4 compensation,
invalid scheduler/feature/layer contracts, routing weights, TP padding and
empty-tensor execution against a CPU expert-FFN reference. It replaces the
external collective only; it does not validate CANN kernel numerics or HCCL.
BF16, W4A8 and SiTU model runs require separate A3 acceptance before claiming
production support beyond the Qwen3 W8A8 test target.
