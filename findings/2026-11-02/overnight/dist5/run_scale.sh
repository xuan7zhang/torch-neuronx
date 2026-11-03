#!/bin/bash
cd ~/torch-neuronx/agent_space/overnight/dist5
source ~/torch-neuronx/.venv/bin/activate
(TORCH_DEVICE_BACKEND_AUTOLOAD=0 DEVN=cpu OMP_NUM_THREADS=4 REF_DP=1 timeout 7200 python llama_scale.py ref 300 > ref_dp1_300.log 2>&1) &
for m in fsdp fsdp_bf16 fsdp_compile ddp tp_fsdp tp; do
  t0=$(date +%s)
  timeout 3600 torchrun --nproc-per-node 4 --master-port $((32000 + RANDOM % 3000)) llama_scale.py $m 300 > scale_$m.log 2>&1
  echo "=== $m exit $? $(( $(date +%s) - t0 ))s"; grep -E "step (0|100|200|275) |done" scale_$m.log | head -6
  pkill -f "llama_scale.py $m" 2>/dev/null
done
wait
