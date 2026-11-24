#!/usr/bin/env bash
set -eo pipefail
source /tmp/dsv41-a3-moe-runtime-20261119/env.sh
export PYTHONPATH="/tmp/dsv41-a3-mhc-coeff-20261120/integration/sglang/python:/tmp/dsv41-a3-mhc-coeff-20261120/round1/coeff-only-a3:$PYTHONPATH"
export SGLANG_NPU_DSV41_A3_DISPATCH_FFN_W4A8=1
export SGLANG_NPU_DSV41_A3_ENGRAM_E8M0_LUT=1
export SGLANG_NPU_DSV41_A3_MHC_POST=1
export HCCL_OP_EXPANSION_MODE="${DISPATCH_TEST_EXPANSION_MODE:-HOST}"
export ASCEND_CUSTOM_OPP_PATH="/tmp/dsv41-a3-dispatch-ffn-20261119/opp/vendors/custom_transformer:$ASCEND_CUSTOM_OPP_PATH"
export SGLANG_NPU_DSV41_A3_MHC_COEFF=1
export ASCEND_CUSTOM_OPP_PATH="/tmp/dsv41-a3-mhc-coeff-20261120/round1/coeff-only-a3/opp/vendors/sgl_mhc_coeff_transformer:$ASCEND_CUSTOM_OPP_PATH"
export LD_LIBRARY_PATH="/tmp/dsv41-a3-mhc-coeff-20261120/round1/coeff-only-a3/opp/vendors/sgl_mhc_coeff_transformer/op_api/lib:$LD_LIBRARY_PATH"
export SGLANG_NPU_MHC_COEFF_BUILD_DIR="/tmp/dsv41-a3-mhc-coeff-20261120/round1/coeff-only-a3/torch-extensions/sglang_mhc_coeff_a3"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
exec python -m sglang.launch_server --json-model-override-args '{"quantization_config": null}' --model-path /mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8 --language-model-only --served-model-name dsv41-a3-int8-indexer --trust-remote-code --device npu --quantization modelslim --dtype bfloat16 --kv-cache-dtype bfloat16 --attention-backend dsv4 --tp-size 16 --attn-dp-size 4 --mem-fraction-static 0.87 --context-length 98304 --max-total-tokens 131072 --max-running-requests 32 --page-size 128 --chunked-prefill-size 16384 --max-prefill-tokens 4096 --cuda-graph-bs-decode 1 2 4 8 --disable-overlap-schedule --random-seed 12345 --watchdog-timeout 1800 --host 127.0.0.1 --port 25013 --ep-size 16 --enable-eplb --ep-num-redundant-experts 16 --eplb-algorithm deepseek --eplb-rebalance-num-iterations 256 --eplb-rebalance-layers-per-chunk 1 --speculative-algorithm DSPARK --speculative-draft-model-path /mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8 --speculative-draft-attention-backend dsv4 --speculative-num-draft-tokens 6 --speculative-dspark-block-size 5 --moe-a2a-backend deepep --deepep-mode auto --speculative-draft-model-quantization unquant --enable-dp-lm-head
