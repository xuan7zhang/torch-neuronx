import os, sys, traceback, torch, torch.nn as nn, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("cpu:gloo,neuron:neuron")
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
from torch.nn.parallel import DistributedDataParallel as DDP
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev); d = DDP(m)
st = powerSGD_hook.PowerSGDState(process_group=None, matrix_approximation_rank=int(sys.argv[1]), start_powerSGD_iter=2,
                                 use_error_feedback=sys.argv[2] == "1", warm_start=sys.argv[3] == "1")
d.register_comm_hook(st, powerSGD_hook.powerSGD_hook)
o = torch.optim.SGD(d.parameters(), lr=0.05)
try:
    for s in range(4):
        o.zero_grad(); d(torch.randn(8, 16, device=dev)).pow(2).mean().backward(); o.step(); torch.neuron.synchronize()
        if r == 0: print(f"RES step {s} ok", flush=True)
except Exception as e:
    if r == 0:
        print("RES FAIL", str(e)[:900].replace("\n", " "), flush=True)
        tb = traceback.format_exc().splitlines(); print("RES TB", " | ".join(l.strip() for l in tb if "File" in l)[-900:], flush=True)
dist.destroy_process_group()
