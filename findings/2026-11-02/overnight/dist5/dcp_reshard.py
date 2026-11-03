"""DCP save with one parallel layout, load with another. torchrun --nproc-per-node 4 dcp_reshard.py <save_layout> <load_layout>
layouts: tp4, fsdp4, tp2fsdp2, full (full = load into a plain model on every rank)."""
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

import torch_neuronx  # noqa: F401

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
dist.init_process_group("cpu:gloo,neuron:neuron")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

save_l, load_l = sys.argv[1], sys.argv[2]
path = f"/tmp/claude-1000/dcp_reshard_{save_l}_{load_l}"


def mk(seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16), nn.Linear(16, 8))
    m.register_buffer("counter", torch.tensor([20_000_001, 3_000_000_007], dtype=torch.int64))
    return m


def place(m, layout):
    m = m.to(dev)
    if layout == "tp4":
        parallelize_module(m, init_device_mesh("neuron", (W,)), {"0": ColwiseParallel(), "2": RowwiseParallel()})
    elif layout == "fsdp4":
        fully_shard(m[0]); fully_shard(m[2]); fully_shard(m)
    elif layout == "tp2fsdp2":
        mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp"))
        parallelize_module(m, mesh["tp"], {"0": ColwiseParallel(), "2": RowwiseParallel()})
        fully_shard(m[0], mesh=mesh["dp"]); fully_shard(m[2], mesh=mesh["dp"]); fully_shard(m, mesh=mesh["dp"])
    return m


def full_sd(m):
    out = {}
    for k, v in m.state_dict().items():
        v = v.full_tensor() if hasattr(v, "full_tensor") else v
        out[k] = v.detach().cpu()
    return out


src = place(mk(0), save_l)
with torch.no_grad():
    for p in src.parameters():
        p.mul_(1.5)  # make values differ from a fresh init
src_full = full_sd(src)
dcp.save({"model": src.state_dict()}, checkpoint_id=path)
dist.barrier()
dst = place(mk(1), load_l) if load_l != "full" else mk(1).to(dev)
sd = {"model": dst.state_dict()}
dcp.load(sd, checkpoint_id=path)
dst.load_state_dict(sd["model"])
dst_full = full_sd(dst)
torch.neuron.synchronize()
errs = {k: ((src_full[k].double() - dst_full[k].double()).abs().max().item()) for k in src_full}
x = torch.randn(4, 16, generator=torch.Generator().manual_seed(5)).to(dev)
with torch.no_grad():
    ys, yd = src(x), dst(x)
    ys = ys.full_tensor() if hasattr(ys, "full_tensor") else ys
    yd = yd.full_tensor() if hasattr(yd, "full_tensor") else yd
if rank == 0:
    print(f"RES save {save_l} -> load {load_l}: max param abs diff {max(v for k, v in errs.items() if k != 'counter'):.3g}, "
          f"counter src {src_full['counter'].tolist()} dst {dst_full['counter'].tolist()}, out diff {(ys - yd).abs().max().item():.3g}")
dist.destroy_process_group()
