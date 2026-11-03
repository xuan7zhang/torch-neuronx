"""Distributed round 5, batch F: RNG under TP / DTensor, whole-model compile with FSDP2, AC with TP.
torchrun --nproc-per-node 4 probe_dist5f.py <case>   (DEVN=cpu for gloo control)"""
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


def all_equal(t):
    """True if tensor t (local, same shape) is identical on every rank."""
    t = t.detach().float().contiguous()
    o = torch.empty(W * t.numel(), device=t.device)
    dist.all_gather_into_tensor(o, t.flatten())
    sync()
    o = o.cpu().view((W,) + tuple(t.shape))
    return all(torch.equal(o[0], o[i]) for i in range(1, W))


@case
def rank_rng_streams():
    out = {}
    torch.manual_seed(1234)
    a = torch.rand(1000, device=dev)
    out["same_seed_rand_equal_across_ranks"] = all_equal(a)
    d = F.dropout(torch.ones(1000, device=dev), 0.5, training=True)
    out["same_seed_dropout_equal_across_ranks"] = all_equal(d)
    torch.manual_seed(1234 + rank)
    b = torch.rand(1000, device=dev)
    out["diff_seed_rand_differs"] = not all_equal(b)
    return out


@case
def tp_dropout_replicated():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    mesh = init_device_mesh(DEVN, (W,))
    out = {}

    class M(nn.Module):
        def __init__(s):
            super().__init__(); torch.manual_seed(0)
            s.a = nn.Linear(32, 64); s.b = nn.Linear(64, 32); s.drop = nn.Dropout(0.5)
        def forward(s, x):
            h = s.b(F.relu(s.a(x)))        # RowwiseParallel output -> Replicate (local tensor)
            return s.drop(h)               # dropout on a replicated activation
    m = M().to(dev)
    parallelize_module(m, mesh, {"a": ColwiseParallel(), "b": RowwiseParallel()})
    torch.manual_seed(7 + rank * 0)  # same seed on all ranks
    x = torch.randn(8, 32, generator=g(0)).to(dev)
    y = m(x)
    y_local = y.to_local() if hasattr(y, "to_local") else y
    out["replicated_dropout_output_equal_across_tp_ranks"] = all_equal(y_local)
    # DTensor dropout directly
    from torch.distributed.tensor import DTensor
    xr = distribute_tensor(torch.ones(8, 64).to(dev), mesh, [Replicate()])
    dr = F.dropout(xr, 0.5, training=True)
    out["dtensor_replicate_dropout_equal"] = all_equal(dr.to_local())
    xs = distribute_tensor(torch.ones(8, 64).to(dev), mesh, [Shard(1)])
    ds = F.dropout(xs, 0.5, training=True).full_tensor()
    out["dtensor_shard_dropout_keep_rate"] = round((ds != 0).float().mean().item(), 3)
    out["dtensor_shard_dropout_shards_distinct"] = not all(torch.equal(ds[:, :16].cpu(), ds[:, 16 * i:16 * (i + 1)].cpu()) for i in range(1, W))
    try:
        from torch.distributed.tensor._random import _rng_tracker
        out["dtensor_rng_tracker"] = type(_rng_tracker).__name__ if _rng_tracker is not None else None
    except Exception as e:
        out["dtensor_rng_tracker"] = f"ERR {str(e)[:80]}"
    return out


@case
def fsdp2_whole_model_compile():
    from torch.distributed.fsdp import fully_shard
    out = {}
    def mk():
        torch.manual_seed(0); return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 4))
    batch = lambda s, r: torch.randn(8, 16, generator=g(s * 10 + r))
    ref = mk(); ro = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    for s in range(3):
        ro.zero_grad(); (sum(ref(batch(s, r)).pow(2).mean() for r in range(W)) / W).backward(); ro.step()
    try:
        m = mk().to(dev)
        for i in (0, 2, 4):
            fully_shard(m[i])
        fully_shard(m)
        cm = torch.compile(m, backend="neuron" if DEVN == "neuron" else "inductor", dynamic=False)
        o = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for s in range(3):
            o.zero_grad(); cm(batch(s, rank).to(dev)).pow(2).mean().backward(); o.step()
        sync()
        out["whole_model_compile"] = max(((a - b.full_tensor().detach().cpu()).norm() / a.norm()).item() for a, b in zip(ref.parameters(), m.parameters()))
    except Exception as e:
        out["whole_model_compile"] = f"ERR {type(e).__name__}: {str(e)[:160]}".replace("\n", " ")
    return out


@case
def tp_activation_checkpoint():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    from torch.utils.checkpoint import checkpoint
    mesh = init_device_mesh(DEVN, (W,))
    def mk():
        torch.manual_seed(0); return nn.ModuleDict({"a": nn.Linear(32, 64), "b": nn.Linear(64, 32), "c": nn.Linear(32, 64), "d": nn.Linear(64, 32)})
    def fwd(m, x):
        h = x + m["b"](F.gelu(m["a"](x)))
        return h + m["d"](F.gelu(m["c"](h)))
    x = torch.randn(8, 32, generator=g(0))
    ref = mk(); (fwd(ref, x)).pow(2).mean().backward()
    m = mk().to(dev); parallelize_module(m, mesh, {"a": ColwiseParallel(), "b": RowwiseParallel(), "c": ColwiseParallel(), "d": RowwiseParallel()})
    y = checkpoint(lambda t: fwd(m, t), x.to(dev), use_reentrant=False)
    y.pow(2).mean().backward(); sync()
    full = lambda t: t.full_tensor() if hasattr(t, "full_tensor") else t
    return {"grads": max(((a.grad - full(b.grad).cpu()).norm() / a.grad.norm()).item() for a, b in zip(ref.parameters(), m.parameters()))}


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
