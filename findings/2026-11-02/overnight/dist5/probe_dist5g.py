"""Distributed round 5, batch G. torchrun --nproc-per-node 4 probe_dist5g.py <case>"""
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
CASE = sys.argv[1]
dist.init_process_group(os.environ.get("PGB", "cpu:gloo,neuron:neuron"))
CASES = {}


def sync():
    torch.neuron.synchronize()


def case(f):
    CASES[f.__name__] = f
    return f


def g(s):
    return torch.Generator().manual_seed(s)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


@case
def elastic_restart():
    attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    t = torch.full((4,), float(rank), device=dev); dist.all_reduce(t); sync()
    if attempt == 0 and rank == 1:
        raise RuntimeError("simulated failure on attempt 0")
    if attempt == 0:
        time.sleep(30)  # wait to be killed by the agent
    x = torch.randn(256, 256, device=dev); y = (x @ x).sum(); sync()
    t = torch.full((4,), float(rank), device=dev); dist.all_reduce(t); sync()
    return {"attempt": attempt, "all_reduce_ok": torch.allclose(t.cpu(), torch.full((4,), float(sum(range(W)))))}


@case
def destroy_with_pending():
    ts = [torch.full((2**20,), float(rank), device=dev) for _ in range(4)]
    ws = [dist.all_reduce(t, async_op=True) for t in ts]
    t0 = time.time()
    try:
        dist.destroy_process_group()
        msg = f"destroy returned in {time.time() - t0:.1f}s"
    except Exception as e:
        msg = f"destroy raised {type(e).__name__}: {str(e)[:100]}"
    try:
        sync(); vals = [t[0].item() for t in ts]
        msg += f"; values {vals}"
    except Exception as e:
        msg += f"; sync raised {str(e)[:100]}"
    try:
        dist.init_process_group(os.environ.get("PGB", "cpu:gloo,neuron:neuron"))
        u = torch.ones(3, device=dev); dist.all_reduce(u); sync()
        msg += f"; reinit all_reduce {u.cpu().tolist()}"
    except Exception as e:
        msg += f"; reinit raised {str(e)[:100]}"
    return msg


@case
def side_stream_collective():
    out = {}
    s = torch.neuron.Stream()
    x = torch.randn(1024, 1024, generator=g(rank)).to(dev)
    a = x @ x  # default stream
    s.wait_stream(torch.neuron.current_stream())
    with torch.neuron.stream(s):
        dist.all_reduce(a)
    b = x * 3  # overlapped compute on default stream
    torch.neuron.current_stream().wait_stream(s)
    c = a + b
    sync()
    xs = [torch.randn(1024, 1024, generator=g(r)) for r in range(W)]
    ref = sum(t @ t for t in xs) + xs[rank] * 3
    out["allreduce_on_side_stream"] = rel(ref, c)
    # async op on side stream, wait() then use on default
    with torch.neuron.stream(s):
        y = torch.full((1000,), float(rank), device=dev)
        w = dist.all_reduce(y, async_op=True)
    w.wait()
    z = y * 2; sync()
    out["async_side_stream_wait"] = torch.allclose(z.cpu(), torch.full((1000,), 2.0 * sum(range(W))))
    return out


@case
def degenerate_mesh_dims():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    out = {}
    def mk():
        torch.manual_seed(0); return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16), nn.Linear(16, 4))
    for shape in ((1, 4), (4, 1)):
        try:
            mesh = init_device_mesh("neuron", shape, mesh_dim_names=("dp", "tp"))
            dp_rank, dp = mesh["dp"].get_local_rank(), shape[0]
            m = mk().to(dev)
            parallelize_module(m, mesh["tp"], {"0": ColwiseParallel(), "2": RowwiseParallel()})
            fully_shard(m, mesh=mesh["dp"])
            o = torch.optim.SGD(m.parameters(), lr=0.1)
            batch = lambda s, r: torch.randn(8, 16, generator=g(s * 10 + r))
            for s in range(2):
                o.zero_grad(); m(batch(s, dp_rank).to(dev)).pow(2).mean().backward(); o.step()
            ref = mk(); ro = torch.optim.SGD(ref.parameters(), lr=0.1)
            for s in range(2):
                ro.zero_grad(); (sum(ref(batch(s, r)).pow(2).mean() for r in range(dp)) / dp).backward(); ro.step()
            sync()
            out[f"mesh{shape}"] = f"{max(rel(a, b.full_tensor()) for a, b in zip(ref.parameters(), m.parameters())):.3g}"
        except Exception as e:
            out[f"mesh{shape}"] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    try:
        mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("a", "b"))
        flat = mesh["a", "b"]._flatten("ab")
        t = torch.full((3,), float(rank), device=dev); dist.all_reduce(t, group=flat.get_group()); sync()
        out["flatten_2x2"] = torch.allclose(t.cpu(), torch.full((3,), float(sum(range(W)))))
    except Exception as e:
        out["flatten_2x2"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    try:
        gg = dist.new_group(backend="gloo")
        c = torch.full((2,), float(rank)); dist.all_reduce(c, group=gg)
        n = torch.full((2,), float(rank), device=dev); dist.all_reduce(n); sync()
        out["gloo_subgroup_plus_neuron"] = bool(torch.allclose(c, torch.full((2,), float(sum(range(W))))) and torch.allclose(n.cpu(), c))
    except Exception as e:
        out["gloo_subgroup_plus_neuron"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return out


if __name__ == "__main__":
    try:
        print(f"R{rank} {CASE}: {CASES[CASE]()}", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"R{rank} {CASE}: ERROR {type(e).__name__}: {str(e)[:300]}".replace("\n", " "), flush=True)
        if CASE == "elastic_restart":
            sys.exit(1)
    try:
        dist.destroy_process_group()
    except Exception:
        pass
