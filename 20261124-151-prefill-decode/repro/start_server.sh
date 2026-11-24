#!/usr/bin/env bash
set -euo pipefail
cd "$1"
exec setsid --wait bash -c 'echo $$ > server.pid; exec bash ./launch.sh >server.log 2>&1'
