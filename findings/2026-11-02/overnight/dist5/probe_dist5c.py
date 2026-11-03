"""Distributed round 5, batch C. torchrun --nproc-per-node 4 probe_dist5c.py <case>  (DEVN=cpu for gloo control)."""
import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

DEVN = os.environ.get("DEVN", "neuron")
if DEVN == "neuron":
    import torch_neuronx  # noqa: F401
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
if DEVN == "neuron":
    torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
else:
    dev = torch.device("cpu")
dist.init_process_group("cpu:gloo,neuron:neuron" if DEVN == "neuron" else "gloo")
CASES = {}


def sync():
    if DEVN == "neuron":
        torch.neuron.synchronize()


def case(f):
    CASES[f.__name__] = f
    return f


def g(s):
    return torch.Generator().manual_seed(s)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


@case
def nan_inf_through_collectives():
    out = {}
    nan, inf = float("nan"), float("inf")
    for dt in (torch.float32, torch.bfloat16):
        for opn, op in (("SUM", dist.ReduceOp.SUM), ("MAX", dist.ReduceOp.MAX), ("MIN", dist.ReduceOp.MIN), ("AVG", dist.ReduceOp.AVG)):
            # col0: NaN only on rank 2; col1: +inf on rank 1; col2: -inf on rank 3; col3: +inf r1 and -inf r3
            v = torch.tensor([1.0, 1.0, 1.0, 1.0])
            if rank == 2: v[0] = nan
            if rank == 1: v[1] = inf; v[3] = inf
            if rank == 3: v[2] = -inf; v[3] = -inf
            t = v.to(dt).to(dev)
            dist.all_reduce(t, op=op); sync()
            got = t.float().cpu().tolist()
            out[f"{str(dt)[6:]}/{opn}"] = got
    out["expect"] = "col0 nan for all ops; col1 SUM/MAX/AVG inf, MIN 1; col2 SUM/MIN/AVG -inf, MAX 1; col3 SUM/AVG nan, MAX inf, MIN -inf"
    # reduce_scatter and all_gather keep NaN
    x = torch.ones(W * 2, device=dev)
    if rank == 0:
        x[3] = nan
    o = torch.empty(2, device=dev); dist.reduce_scatter_tensor(o, x); sync()
    out["reduce_scatter_nan_on_rank1_slot"] = o.cpu().tolist()
    a = torch.full((2,), nan if rank == 1 else 1.0, device=dev); ag = torch.empty(W * 2, device=dev)
    dist.all_gather_into_tensor(ag, a); sync()
    out["all_gather_nan"] = ag.cpu().tolist()
    return out


@case
def fsdp_clip_nan_one_rank():
    from torch.distributed.fsdp import fully_shard
    out = {}
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev)
    fully_shard(m[0]); fully_shard(m[2]); fully_shard(m)
    x = torch.randn(8, 16, generator=g(rank)).to(dev)
    if rank == 2:
        x[0, 0] = float("nan")
    m(x).pow(2).mean().backward()
    n = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    out["total_norm"] = full(n).item() if hasattr(n, "full_tensor") else n.item()
    out["grads_finite_all"] = all(torch.isfinite(full(p.grad)).all().item() for p in m.parameters())
    try:
        m.zero_grad(); m(x).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        out["error_if_nonfinite"] = "NO ERROR"
    except RuntimeError as e:
        out["error_if_nonfinite"] = "raised"
    # inf-norm clip uses MAX reduce
    m.zero_grad(); m(x).pow(2).mean().backward()
    ninf = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, norm_type=float("inf"))
    out["inf_norm"] = full(ninf).item() if hasattr(ninf, "full_tensor") else ninf.item()
    return out


@case
def int64_dtensor():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    mesh = init_device_mesh(DEVN, (W,))
    base = torch.arange(8, dtype=torch.int64) + 3_000_000_000
    d = distribute_tensor(base.to(dev), mesh, [Shard(0)])
    out = {"local_shard": d.to_local().cpu().tolist(), "full_tensor": d.full_tensor().cpu().tolist(),
           "redistribute_R": d.redistribute(mesh, [Replicate()]).to_local().cpu().tolist(), "expected": base.tolist()}
    return out


@case
def fsdp2_mixed_precision_nobuf():
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); torch.manual_seed(0)
            s.inp = nn.Linear(16, 32)
            s.blocks = nn.ModuleList([nn.Sequential(nn.RMSNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)) for _ in range(3)])
            s.out = nn.Linear(32, 4)
        def forward(s, x):
            h = s.inp(x)
            for b in s.blocks:
                h = h + b(h)
            return s.out(h)
    batch = lambda st, r: torch.randn(8, 16, generator=g(st * 10 + r))
    ref = Net(); ro = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    for st in range(3):
        ro.zero_grad(); (sum(ref(batch(st, r)).pow(2).mean() for r in range(W)) / W).backward(); ro.step()
    out = {}
    for name, mp in {
        "bf16_param_fp32_reduce": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        "bf16_param_bf16_reduce": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
        "bf16_out_fp32": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32),
        "fp32_param_bf16_reduce": MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.bfloat16),
    }.items():
        try:
            m = Net().to(dev)
            for b in m.blocks:
                fully_shard(b, mp_policy=mp)
            fully_shard(m, mp_policy=mp)
            o = torch.optim.AdamW(m.parameters(), lr=1e-2)
            for st in range(3):
                o.zero_grad(); y = m(batch(st, rank).to(dev)); y.float().pow(2).mean().backward(); o.step()
            sync()
            out[name] = (f"params {max(rel(a, full(b)) for a, b in zip(ref.parameters(), m.parameters())):.3g} out_dtype {y.dtype} "
                         f"param_dtype {sorted({str(p.dtype)[6:] for p in m.parameters()})} grad_dtype {sorted({str(p.grad.dtype)[6:] for p in m.parameters()})}")
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    return out


@case
def uneven_batch_per_rank():
    from torch.distributed.fsdp import fully_shard
    from torch.nn.parallel import DistributedDataParallel as DDP
    out = {}
    bs = [8, 5, 3, 8]  # last batch of an epoch differs per rank
    mk = lambda: (torch.manual_seed(0), nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)))[1]
    xs = [torch.randn(bs[r], 16, generator=g(r)) for r in range(W)]
    ref = mk(); (sum(ref(x).pow(2).mean() for x in xs) / W).backward()
    m = mk().to(dev); d = DDP(m); d(xs[rank].to(dev)).pow(2).mean().backward(); sync()
    out["ddp_grads"] = max(rel(a.grad, b.grad) for a, b in zip(ref.parameters(), m.parameters()))
    f = mk().to(dev); fully_shard(f); f(xs[rank].to(dev)).pow(2).mean().backward(); sync()
    out["fsdp_grads"] = max(rel(a.grad, full(b.grad)) for a, b in zip(ref.parameters(), f.parameters()))
    cm = mk().to(dev); cd = torch.compile(DDP(cm), backend="neuron", dynamic=False)
    cd(xs[rank].to(dev)).pow(2).mean().backward(); sync()
    out["compiled_ddp_grads"] = max(rel(a.grad, b.grad) for a, b in zip(ref.parameters(), cm.parameters()))
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
