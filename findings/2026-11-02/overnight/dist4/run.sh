#!/bin/bash
# Run each case in its own torchrun. Usage: run.sh case1 case2 ...
cd ~/torch-neuronx/agent_space/overnight/dist4
source ~/torch-neuronx/.venv/bin/activate
port=29610
for c in "$@"; do
  port=$((port + 13))
  timeout 600 torchrun --nproc-per-node 4 --master-port $port probe_dist4.py $c > $c.log 2>&1
  ec=$?
  echo "=== $c (exit $ec)"
  grep -E "^R[0-9] " $c.log | sort | cut -c1-700
  [ $ec -ne 0 ] && grep -E "Error|error" $c.log | grep -v -E "NRT:nrt_init|Warning" | tail -3 | cut -c1-300
done
