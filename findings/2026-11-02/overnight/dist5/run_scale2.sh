#!/bin/bash
cd ~/torch-neuronx/agent_space/overnight/dist5; source ~/torch-neuronx/.venv/bin/activate
(SEQ=120 TORCH_DEVICE_BACKEND_AUTOLOAD=0 DEVN=cpu OMP_NUM_THREADS=6 REF_DP=4 timeout 3600 python llama_scale.py ref 300 > ref_dp4_300_s120.log 2>&1) &
for m in tp_fsdp_compile fsdp_bf16_compile ddp_compile fsdp; do
  SEQ=120 timeout 3000 torchrun --nproc-per-node 4 --master-port $((50000 + RANDOM % 3000)) llama_scale.py $m 300 > scale_${m}_s120.log 2>&1
  echo "=== $m s120 exit $?"; grep -E "step (0|150|275) |done" scale_${m}_s120.log | head -4
done
wait
