"""Distributed round 5, batch J: GradScaler across ranks, FSDP2 reduce options. torchrun --nproc-per-node 4 probe_dist5j.py <case>"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn

import torch_neuronx  # noqa: F401

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
dist.init_process_group("cpu:gloo,neuron:neuron")
CASES = {}


def sync():
    torch.neuron.synchronize()


def case(f):
    CASES[f.__name__] = f
    return f


def g(s):
    return torch.Generator().manual_seed(s)


def mk():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


@case
def fsdp2_sharded_grad_scaler_inf_one_rank():
    from torch.distributed.fsdp import fully_shard
    from torch.amp.grad_scaler import GradScaler
    try:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    except Exception:
        ShardedGradScaler = None
    out = {}
    for name, S in (("ShardedGradScaler", ShardedGradScaler), ("amp.GradScaler", lambda: GradScaler("neuron", init_scale=1024.0))):
        if S is None:
            out[name] = "not available"; continue
        try:
            m = mk().to(dev); fully_shard(m[0]); fully_shard(m[2]); fully_shard(m)
            o = torch.optim.SGD(m.parameters(), lr=0.1)
            sc = S() if name == "amp.GradScaler" else S(init_scale=1024.0)
            before = [full(p).detach().cpu().clone() for p in m.parameters()]
            x = torch.randn(8, 16, generator=g(rank)).to(dev)
            if rank == 2:
                x[0, 0] = float("inf")
            o.zero_grad(); sc.scale(m(x).pow(2).mean()).backward(); sc.step(o); sc.update(); sync()
            after = [full(p).detach().cpu() for p in m.parameters()]
            changed = any(not torch.equal(a, b) for a, b in zip(before, after))
            fin = all(torch.isfinite(a).all().item() for a in after)
            out[name] = f"params_changed={changed} params_finite={fin} new_scale={sc.get_scale()}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    out["expect"] = "step skipped on all ranks (params_changed False, finite), scale 512"
    return out


@case
def ddp_grad_scaler_inf_one_rank():
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.amp.grad_scaler import GradScaler
    m = mk().to(dev); d = DDP(m)
    o = torch.optim.SGD(d.parameters(), lr=0.1)
    sc = GradScaler("neuron", init_scale=1024.0)
    before = [p.detach().cpu().clone() for p in m.parameters()]
    x = torch.randn(8, 16, generator=g(rank)).to(dev)
    if rank == 1:
        x[0, 0] = float("inf")
    with torch.autocast("neuron", dtype=torch.float16):
        l = d(x).float().pow(2).mean()
    o.zero_grad(); sc.scale(l).backward(); sc.step(o); sc.update(); sync()
    after = [p.detach().cpu() for p in m.parameters()]
    p0 = after[0].clone().to(dev); dist.broadcast(p0, src=0); sync()
    return {"params_changed": any(not torch.equal(a, b) for a, b in zip(before, after)), "finite": all(torch.isfinite(a).all().item() for a in after),
            "scale": sc.get_scale(), "same_as_rank0": torch.equal(p0.cpu(), after[0])}


@case
def fsdp2_reduce_options():
    from torch.distributed.fsdp import fully_shard
    out = {}
    batch = lambda r: torch.randn(8, 16, generator=g(r))
    ref = mk(); (sum(ref(batch(r)).pow(2).mean() for r in range(W))).backward()  # SUM over ranks (no /W)
    ref_mean = [p.grad / W for p in ref.parameters()]
    ref_sum = [p.grad for p in ref.parameters()]
    for name in ("default", "divide_factor_1", "force_sum"):
        try:
            m = mk().to(dev); fully_shard(m[0]); fully_shard(m[2]); fully_shard(m)
            if name == "divide_factor_1":
                m.set_gradient_divide_factor(1.0)
            elif name == "force_sum":
                if hasattr(m, "set_force_sum_reduction_for_comms"):
                    m.set_force_sum_reduction_for_comms(True)
                else:
                    out[name] = "api not available"; continue
            m(batch(rank).to(dev)).pow(2).mean().backward(); sync()
            gs = [full(p.grad).cpu() for p in m.parameters()]
            em = max(((a - b).norm() / a.norm()).item() for a, b in zip(ref_mean, gs))
            es = max(((a - b).norm() / a.norm()).item() for a, b in zip(ref_sum, gs))
            out[name] = f"vs_mean {em:.2g} vs_sum {es:.2g}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    out["expect"] = "default: mean; divide_factor_1: sum; force_sum: mean"
    return out


if __name__ == "__main__":
    c = sys.argv[1]
    try:
        print(f"R{rank} {c}: {CASES[c]()}", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"R{rank} {c}: ERROR {type(e).__name__}: {str(e)[:300]}".replace("\n", " "), flush=True)
    try:
        dist.destroy_process_group()
    except Exception:
        pass
