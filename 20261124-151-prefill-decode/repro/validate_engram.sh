#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source /tmp/dsv41-a3-moe-runtime-20261119/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
exec python validate_engram.py "$@"
