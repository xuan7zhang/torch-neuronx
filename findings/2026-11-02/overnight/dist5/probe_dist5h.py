"""Distributed round 5, batch H: zero-size and zero-split collectives (MoE edge cases), compiled EP, DCP dtype.
torchrun --nproc-per-node 4 probe_dist5h.py <case>"""
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


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def tryit(out, name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = f"ERR {type(e).__name__}: {str(e)[:110]}".replace("\n", " ")
    try:
        sync()
    except Exception as e:
        out[name + "_sync"] = f"ERR {str(e)[:80]}"


@case
def zero_size_collectives():
    out = {}
    def ar():
        t = torch.empty(0, device=dev); dist.all_reduce(t); sync(); return tuple(t.shape)
    tryit(out, "all_reduce_empty", ar)
    def ag():
        o = torch.empty(0, device=dev); dist.all_gather_into_tensor(o, torch.empty(0, device=dev)); sync(); return tuple(o.shape)
    tryit(out, "all_gather_into_empty", ag)
    def agl():
        outs = [torch.empty(0, 4, device=dev) for _ in range(W)]; dist.all_gather(outs, torch.empty(0, 4, device=dev)); sync(); return [tuple(o.shape) for o in outs]
    tryit(out, "all_gather_list_empty", agl)
    def bc():
        t = torch.empty(0, device=dev); dist.broadcast(t, src=0); sync(); return "ok"
    tryit(out, "broadcast_empty", bc)
    def rs():
        o = torch.empty(0, device=dev); dist.reduce_scatter_tensor(o, torch.empty(0, device=dev)); sync(); return "ok"
    tryit(out, "reduce_scatter_empty", rs)
    def after():
        t = torch.ones(3, device=dev); dist.all_reduce(t); sync(); return t.cpu().tolist()
    tryit(out, "all_reduce_after", after)
    return out


@case
def all_to_all_zero_splits():
    out = {}
    D = 4
    # rank r sends split[r][j] rows to rank j; some entries are zero; rank 3 sends nothing at all
    split = [[2, 0, 1, 0], [0, 0, 3, 1], [1, 2, 0, 0], [0, 0, 0, 0]]
    send = split[rank]; recv = [split[j][rank] for j in range(W)]
    rows = [torch.arange(sum(split[r]) * D, dtype=torch.float32).view(-1, D) + 100 * r for r in range(W)]
    def run():
        x = rows[rank].to(dev)
        o = torch.empty(sum(recv), D, device=dev)
        dist.all_to_all_single(o, x, recv, send); sync()
        exp = torch.cat([rows[j].split(split[j])[rank] for j in range(W)]) if sum(recv) else torch.empty(0, D)
        return torch.equal(o.cpu(), exp)
    tryit(out, "a2a_single_zero_splits", run)
    def run_list():
        ins = [r_.to(dev) for r_ in rows[rank].split(send)]
        outs = [torch.empty(recv[j], D, device=dev) for j in range(W)]
        dist.all_to_all(outs, ins); sync()
        return all(torch.equal(outs[j].cpu(), rows[j].split(split[j])[rank]) for j in range(W))
    tryit(out, "a2a_list_zero_splits", run_list)
    import torch.distributed._functional_collectives as fc
    def run_autograd():
        x = rows[rank].to(dev).requires_grad_()
        y = fc.all_to_all_single_autograd(x, recv, send, dist.group.WORLD)
        (y * 2).sum().backward(); sync()
        return torch.allclose(x.grad.cpu(), torch.full_like(rows[rank], 2.0)) if x.numel() else (tuple(x.grad.shape) if x.grad is not None else "grad None")
    tryit(out, "a2a_autograd_zero_splits", run_autograd)
    def after():
        t = torch.ones(3, device=dev); dist.all_reduce(t); sync(); return t.cpu().tolist()
    tryit(out, "all_reduce_after", after)
    return out


@case
def compiled_ep_uneven():
    import torch.distributed._functional_collectives as fc
    out = {}
    D = 8
    torch.manual_seed(0); expert = nn.Linear(D, D).to(dev)
    def step(x, send, recv):
        y = fc.all_to_all_single_autograd(x, recv, send, dist.group.WORLD)
        z = expert(y)
        return fc.all_to_all_single_autograd(z, send, recv, dist.group.WORLD)
    cs = torch.compile(step, backend="neuron", dynamic=False)
    res = []
    for it in range(3):
        split = [[(r + j + it) % 3 + 1 for j in range(W)] for r in range(W)]
        send = split[rank]; recv = [split[j][rank] for j in range(W)]
        x = torch.randn(sum(send), D, generator=g(rank * 10 + it)).to(dev).requires_grad_()
        try:
            back = cs(x, send, recv); back.pow(2).sum().backward(); sync()
            ye = step(x.detach().clone().requires_grad_(), send, recv)
            res.append(f"it{it} fwd {rel(ye, back):.2g}")
        except Exception as e:
            res.append(f"it{it} ERR {type(e).__name__}: {str(e)[:100]}".replace("\n", " ")); break
    out["compiled_uneven_a2a"] = res
    return out


@case
def dcp_dtype_and_resume():
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import fully_shard
    out = {}
    path = f"/tmp/claude-1000/dcp_h_{os.environ.get('MASTER_PORT', 'x')}"
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4)).to(dev).to(torch.bfloat16)
    fully_shard(m[0]); fully_shard(m[1]); fully_shard(m)
    dcp.save({"model": m.state_dict()}, checkpoint_id=path)
    torch.manual_seed(1); m2 = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4)).to(dev)  # fp32 target
    fully_shard(m2[0]); fully_shard(m2[1]); fully_shard(m2)
    sd = {"model": m2.state_dict()}
    try:
        dcp.load(sd, checkpoint_id=path)
        m2.load_state_dict(sd["model"])
        out["bf16_saved_fp32_loaded"] = max(rel(a.full_tensor().float(), b.full_tensor()) for a, b in zip(m.parameters(), m2.parameters()))
        out["loaded_dtype"] = str(next(m2.parameters()).dtype)
    except Exception as e:
        out["bf16_saved_fp32_loaded"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    # int64 and fp64 extra state through DCP
    extra = {"step": torch.tensor(3_000_000_007, dtype=torch.int64), "tokens": torch.tensor([20_000_001 + rank], dtype=torch.int64, device=dev),
             "acc": torch.tensor([1e300], dtype=torch.float64, device=dev)}
    p2 = path + "_extra_r"
    try:
        dcp.save({"extra": extra}, checkpoint_id=p2)
        ld = {"extra": {"step": torch.tensor(0, dtype=torch.int64), "tokens": torch.zeros(1, dtype=torch.int64, device=dev), "acc": torch.zeros(1, dtype=torch.float64, device=dev)}}
        dcp.load(ld, checkpoint_id=p2)
        out["extra_state"] = (ld["extra"]["step"].item(), ld["extra"]["tokens"].cpu().tolist(), ld["extra"]["acc"].cpu().tolist())
        out["extra_expected"] = (3_000_000_007, [20_000_001 + rank], [1e300])
    except Exception as e:
        out["extra_state"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
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
