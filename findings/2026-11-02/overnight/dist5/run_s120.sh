#!/bin/bash
cd ~/torch-neuronx/agent_space/overnight/dist5; source ~/torch-neuronx/.venv/bin/activate
(SEQ=120 TORCH_DEVICE_BACKEND_AUTOLOAD=0 DEVN=cpu OMP_NUM_THREADS=6 REF_DP=2 timeout 3600 python llama_scale.py ref 300 > ref_dp2_300_s120.log 2>&1) &
SEQ=120 timeout 3600 torchrun --nproc-per-node 4 --master-port $((34000 + RANDOM % 2000)) llama_scale.py tp_fsdp 300 > scale_tp_fsdp_s120.log 2>&1; echo "tp_fsdp s120 exit $?"
wait
