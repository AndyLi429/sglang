#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source /tmp/dsv41-a3-moe-runtime-20261119/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
exec python -u validate_prefill.py
