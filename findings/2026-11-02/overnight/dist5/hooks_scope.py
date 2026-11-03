import os, sys, torch, torch.nn as nn, torch.distributed as dist
DEVN = os.environ.get("DEVN", "neuron")
if DEVN == "neuron":
    import torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
dev = torch.device("neuron", r) if DEVN == "neuron" else torch.device("cpu")
if DEVN == "neuron": torch.neuron.set_device(r)
dist.init_process_group("neuron" if DEVN == "neuron" else "gloo")
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook, quantization_hooks
name = sys.argv[1]
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev); d = DDP(m)
if name == "fp16_compress": d.register_comm_hook(None, default_hooks.fp16_compress_hook)
elif name == "quant_per_tensor": d.register_comm_hook(None, quantization_hooks.quantization_pertensor_hook)
elif name == "quant_per_channel": d.register_comm_hook(None, quantization_hooks.quantization_perchannel_hook)
elif name == "batched_powersgd": d.register_comm_hook(powerSGD_hook.PowerSGDState(process_group=None, matrix_approximation_rank=1, start_powerSGD_iter=2), powerSGD_hook.batched_powerSGD_hook)
elif name == "fsdp1_fp16_hook":
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.algorithms._comm_hooks import default_hooks as fdh
    m2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev)
    d = FSDP(m2, device_id=dev if DEVN == "neuron" else None)
    st = fdh.LowPrecisionState(process_group=dist.group.WORLD, parameter_type=torch.float32)
    d.register_comm_hook(st, fdh.fp16_compress_hook)
o = torch.optim.SGD(d.parameters(), lr=0.05)
try:
    for s in range(4):
        o.zero_grad(); d(torch.randn(8, 16, generator=torch.Generator().manual_seed(s * 10 + r)).to(dev)).pow(2).mean().backward(); o.step()
        if DEVN == "neuron": torch.neuron.synchronize()
    msg = "4 steps ok"
except Exception as e:
    msg = "ERR " + str(e).replace("\n", " ")[:200]
if r == 0: print(f"RES {DEVN} {name}: {msg}")
dist.destroy_process_group()
