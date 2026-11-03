"""Distributed round 5, batch D. torchrun --nproc-per-node 4 probe_dist5d.py <case>"""
import os
import sys
import time
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


@case
def fp64_collectives():
    out = {}
    vals = [torch.tensor([1.0 + 1e-12 * (q + 1), 1e300, 123456789.123456789, 1e-310], dtype=torch.float64) for q in range(W)]
    t = vals[rank].to(dev)
    out["device_tensor_dtype"] = str(t.dtype)
    out["device_roundtrip_exact"] = torch.equal(t.cpu(), vals[rank])
    dist.all_reduce(t); sync()
    exp = sum(vals)
    got = t.cpu()
    out["all_reduce dtype"] = str(got.dtype)
    out["all_reduce got"] = [repr(x) for x in got.tolist()]
    out["all_reduce exp"] = [repr(x) for x in exp.tolist()]
    b = (vals[0] if rank == 0 else torch.zeros(4, dtype=torch.float64)).to(dev); dist.broadcast(b, src=0); sync()
    out["broadcast exact"] = torch.equal(b.cpu(), vals[0])
    o = torch.empty(W * 4, dtype=torch.float64, device=dev); dist.all_gather_into_tensor(o, vals[rank].to(dev)); sync()
    out["all_gather exact"] = torch.equal(o.cpu()[:4], vals[0])
    return out


@case
def int_reduce_gather_avg():
    out = {}
    base = 30_000_000
    v = torch.tensor([base + rank, 5 + rank], dtype=torch.int32)
    t = v.to(dev); dist.reduce(t, dst=1); sync()
    if rank == 1:
        out["reduce int32 to dst"] = (t.cpu().tolist(), [base * W + sum(range(W)), 5 * W + sum(range(W))])
    gl = [torch.empty(2, dtype=torch.int32, device=dev) for _ in range(W)] if rank == 0 else None
    dist.gather(v.to(dev), gl, dst=0); sync()
    if rank == 0:
        out["gather int32"] = ([x.cpu().tolist() for x in gl], "expect", [[base + q, 5 + q] for q in range(W)])
    try:
        a = torch.tensor([base + 4 * rank], dtype=torch.int32, device=dev); dist.all_reduce(a, op=dist.ReduceOp.AVG); sync()
        out["avg int32"] = a.cpu().tolist()
    except Exception as e:
        out["avg int32"] = f"ERR {str(e)[:80]}"
    m = torch.tensor([base + rank], dtype=torch.int32, device=dev); dist.all_reduce(m, op=dist.ReduceOp.MAX); sync()
    out["max int32 30M+r"] = (m.item(), base + W - 1)
    return out


@case
def compiled_funcol_int():
    import torch.distributed._functional_collectives as fc
    out = {}
    def f(x):
        return fc.all_reduce(x, "sum", dist.group.WORLD) + 0
    def b(x):
        return fc.broadcast(x, 0, dist.group.WORLD) if hasattr(fc, "broadcast") else x
    x = torch.tensor([20_000_001 + rank], dtype=torch.int32, device=dev)
    for name, fn in (("eager_funcol_sum", f), ("compiled_funcol_sum", torch.compile(f, backend="neuron", dynamic=False))):
        try:
            y = fn(x); sync(); out[name] = (y.item(), 20_000_001 * W + sum(range(W)))
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:100]}".replace("\n", " ")
    xf = torch.tensor([1.0 + rank], device=dev)
    yc = torch.compile(lambda t: fc.all_reduce(t, "sum", dist.group.WORLD) * 2, backend="neuron", dynamic=False)(xf); sync()
    out["compiled_funcol_float"] = (yc.item(), 2.0 * sum(range(1, W + 1)))
    return out


@case
def ddp_mixed_dtype_params():
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.manual_seed(0)
    def mk():
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))
        m[2].to(torch.bfloat16)
        return m
    class Wrap(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, x):
            h = torch.relu(s.m[0](x))
            h = torch.relu(s.m[2](h.bfloat16()).float())
            return s.m[4](h)
    xs = [torch.randn(8, 16, generator=g(r)) for r in range(W)]
    ref = Wrap(mk()); (sum(ref(x).pow(2).mean() for x in xs) / W).backward()
    m = Wrap(mk()).to(dev); d = DDP(m, bucket_cap_mb=0.001); d(xs[rank].to(dev)).pow(2).mean().backward(); sync()
    errs = {n: ((a.grad.float() - b.grad.float().cpu()).norm() / a.grad.float().norm()).item() for (n, a), (_, b) in zip(ref.named_parameters(), m.named_parameters())}
    return {"grad_dtypes": sorted({str(p.grad.dtype)[6:] for p in m.parameters()}), "worst": max(errs.values()), "per_param": {k: f"{v:.2g}" for k, v in errs.items()}}


@case
def allreduce_bandwidth():
    out = {}
    for nbytes in (2**10, 2**16, 2**20, 2**24, 2**26, 2**28, 2**29 + 2**20, 2**30):
        n = nbytes // 4
        t = torch.ones(n, device=dev)
        dist.all_reduce(t); sync()  # warm (compile)
        it = 5 if nbytes >= 2**26 else 20
        dist.barrier(); sync()
        t0 = time.time()
        for _ in range(it):
            dist.all_reduce(t)
        sync()
        dt = (time.time() - t0) / it
        busbw = 2 * (W - 1) / W * nbytes / dt / 1e9
        out[f"{nbytes / 2**20:.3g}MiB"] = f"{dt * 1e3:.3f} ms, busbw {busbw:.2f} GB/s"
        del t
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
