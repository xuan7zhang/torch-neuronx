#!/bin/bash
# Usage: run.sh <script> <tag> case1 case2 ...   (each case in its own torchrun; DEVN=cpu env for gloo control)
cd ~/torch-neuronx/agent_space/overnight/dist5
source ~/torch-neuronx/.venv/bin/activate
script=$1; tag=$2; shift 2
port=$((30000 + RANDOM % 20000))
for c in "$@"; do
  port=$((port + 17))
  timeout ${CASE_TIMEOUT:-900} torchrun --nproc-per-node 4 --master-port $port $script $c > ${tag}_$c.log 2>&1
  ec=$?
  echo "=== $c (exit $ec)"
  grep -E "^R[0-9] " ${tag}_$c.log | sort -k1,1 -s | cut -c1-1200
  [ $ec -ne 0 ] && grep -E "Error|error" ${tag}_$c.log | grep -v -E "NRT:nrt_init|Warning" | tail -3 | cut -c1-300
  pkill -f "$script $c" 2>/dev/null
done
