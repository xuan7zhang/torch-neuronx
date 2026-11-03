import os, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
dist.init_process_group("gloo"); mesh = init_device_mesh("cpu", (int(os.environ["WORLD_SIZE"]),))
m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.LayerNorm(8))
parallelize_module(m, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()})
opt = torch.optim.AdamW(m.parameters(), lr=1e-2, foreach=True)
try:
    m(torch.randn(4, 8)).sum().backward(); opt.step(); r = "ok"
except Exception as e: r = "ERR " + str(e)[:140]
if dist.get_rank() == 0: print("RES cpu/gloo TP + plain LayerNorm + AdamW foreach=True:", r)
dist.destroy_process_group()
