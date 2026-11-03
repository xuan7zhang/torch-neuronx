import os, time, torch, torch.nn as nn, torch.distributed as dist
dist.init_process_group("gloo"); r = dist.get_rank(); print(f"R{r} init", flush=True)
from torch.distributed.fsdp import fully_shard
torch.manual_seed(0); m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
fully_shard(m[0]); fully_shard(m[2]); fully_shard(m); print(f"R{r} sharded", flush=True)
o = torch.optim.AdamW(m.parameters(), lr=1e-2)
m(torch.randn(8, 16)).pow(2).mean().backward(); print(f"R{r} bwd", flush=True); o.step(); print(f"R{r} step", flush=True)
dist.destroy_process_group()
