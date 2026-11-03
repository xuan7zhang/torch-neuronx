"""Distributed round 5 (batch A). Run: torchrun --nproc-per-node 4 probe_dist5.py <case>.
Each rank prints 'R<rank> <case>: ...'. CPU reference = single process on the global batch."""
import os
import sys
import time
import traceback
from datetime import timedelta

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
CASE = sys.argv[1]
init_kw = {}
if CASE.startswith("timeout"):
    init_kw["timeout"] = timedelta(seconds=30)
dist.init_process_group("cpu:gloo,neuron:neuron" if DEVN == "neuron" else "gloo", **init_kw)
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


class Block(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.n = nn.RMSNorm(d)
        self.l1 = nn.Linear(d, 64)
        self.l2 = nn.Linear(64, d)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, d))

    def forward(self, x):
        return x + self.l2(F.gelu(self.l1(self.n(x)))) * self.scale


class Net(nn.Module):
    def __init__(self, n=3, d=32):
        super().__init__()
        torch.manual_seed(0)
        self.inp = nn.Linear(16, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.out = nn.Linear(d, 4)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def batch(s, r):
    return torch.randn(8, 16, generator=g(s * 10 + r))


def ref_train(steps, opt=lambda p: torch.optim.AdamW(p, lr=1e-2), clip=None):
    m = Net()
    o = opt(m.parameters())
    for s in range(steps):
        o.zero_grad()
        sum(m(batch(s, r)).pow(2).mean() for r in range(W)).div(W).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        o.step()
    return m


def fsdp_train(shard, steps=3, opt=lambda p: torch.optim.AdamW(p, lr=1e-2), clip=None, compile_=False):
    m = Net().to(dev)
    shard(m)
    f = torch.compile(m, backend="neuron", dynamic=False) if compile_ else m
    o = opt(m.parameters())
    for s in range(steps):
        o.zero_grad()
        f(batch(s, rank).to(dev)).float().pow(2).mean().backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        o.step()
    sync()
    return m


# ---------------- FSDP2 ----------------
@case
def fsdp2_mixed_precision():
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    ref = ref_train(3)
    out = {}
    for name, mp in {
        "bf16_param_fp32_reduce": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        "bf16_param_bf16_reduce": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
        "bf16_out_fp32": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32),
        "fp32_param_bf16_reduce": MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.bfloat16),
        "no_cast_inputs": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False),
    }.items():
        try:
            def shard(m, mp=mp):
                for b in m.blocks:
                    fully_shard(b, mp_policy=mp)
                fully_shard(m, mp_policy=mp)
            m = fsdp_train(shard)
            pd = {str(p.dtype) for p in m.parameters()}
            gd = {str(p.grad.dtype) for p in m.parameters() if p.grad is not None}
            out[name] = f"params {max(rel(a, full(b)) for a, b in zip(ref.parameters(), m.parameters())):.3g} param_dtype {pd} grad_dtype {gd}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return out


@case
def fsdp2_prefetch_reshard():
    from torch.distributed.fsdp import fully_shard
    ref = ref_train(3)
    out = {}
    def s1(m):
        for b in m.blocks:
            fully_shard(b, reshard_after_forward=False)
        fully_shard(m, reshard_after_forward=False)
    out["reshard_after_forward_False"] = s1
    def s2(m):
        for b in m.blocks:
            fully_shard(b)
        fully_shard(m)
        bl = list(m.blocks)
        for i in range(len(bl) - 1):
            bl[i].set_modules_to_forward_prefetch([bl[i + 1]])
            bl[i + 1].set_modules_to_backward_prefetch([bl[i]])
    out["explicit_prefetch"] = s2
    def s3(m):
        from torch.distributed.tensor import Shard
        for b in m.blocks:
            fully_shard(b, shard_placement_fn=lambda p: Shard(1) if p.dim() == 2 else Shard(0))
        fully_shard(m)
    out["shard_dim1"] = s3
    def s4(m):
        for b in m.blocks:
            fully_shard(b)
        fully_shard(m)
    out["clip_foreach"] = (s4, 0.05)
    res = {}
    for k, v in out.items():
        try:
            fn, clip = (v if isinstance(v, tuple) else (v, None))
            m = fsdp_train(fn, clip=clip)
            rr = ref_train(3, clip=clip) if clip else ref
            res[k] = f"{max(rel(a, full(b)) for a, b in zip(rr.parameters(), m.parameters())):.3g}"
        except Exception as e:
            res[k] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return res


@case
def fsdp2_buffers_and_shared():
    from torch.distributed.fsdp import fully_shard
    out = {}
    m = Net().to(dev)
    for b in m.blocks:
        fully_shard(b)
    fully_shard(m)
    out["buffer_type"] = type(m.blocks[0].scale).__name__
    out["buffer_value"] = rel(torch.linspace(0.5, 1.5, 32), m.blocks[0].scale)
    # shared Linear used by two blocks, wrapped at root only
    class Shared(nn.Module):
        def __init__(s):
            super().__init__(); torch.manual_seed(0)
            s.lin = nn.Linear(16, 16); s.a = nn.Linear(16, 16); s.b = nn.Linear(16, 16)
        def forward(s, x):
            return s.b(s.lin(F.relu(s.a(s.lin(x)))))
    def tr(model, d, n, x_fn):
        o = torch.optim.SGD(model.parameters(), lr=0.1)
        for s in range(n):
            o.zero_grad(); x_fn(s, d).backward(); o.step()
        return model
    sm = Shared().to(dev); fully_shard(sm.a); fully_shard(sm.b); fully_shard(sm)
    tr(sm, dev, 3, lambda s, d: sm(torch.randn(8, 16, generator=g(s * 10 + rank)).to(d)).pow(2).mean())
    ref = Shared()
    tr(ref, "cpu", 3, lambda s, d: sum(ref(torch.randn(8, 16, generator=g(s * 10 + r))).pow(2).mean() for r in range(W)) / W)
    out["shared_param_root"] = max(rel(a, full(b)) for a, b in zip(ref.parameters(), sm.parameters()))
    return out


@case
def fsdp2_ac_compile_long():
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    steps = 40
    m = Net(n=4).to(dev)
    for i in range(len(m.blocks)):
        m.blocks[i] = checkpoint_wrapper(m.blocks[i])
        fully_shard(m.blocks[i])
    fully_shard(m)
    for i in range(len(m.blocks)):
        m.blocks[i].compile(backend="neuron", dynamic=False)
    o = torch.optim.AdamW(m.parameters(), lr=3e-3)
    mem = []
    losses = []
    for s in range(steps):
        o.zero_grad()
        l = m(batch(s, rank).to(dev)).pow(2).mean(); l.backward(); o.step()
        if s % 10 == 0 or s == steps - 1:
            sync(); mem.append(round(torch.neuron.memory_allocated() / 2**20, 2) if DEVN == "neuron" else 0)
            lt = l.detach().clone(); dist.all_reduce(lt); losses.append(round(lt.item() / W, 5))
    ref = Net(n=4); ro = torch.optim.AdamW(ref.parameters(), lr=3e-3); rl = []
    for s in range(steps):
        ro.zero_grad(); L = sum(ref(batch(s, r)).pow(2).mean() for r in range(W)) / W; L.backward(); ro.step()
        if s % 10 == 0 or s == steps - 1:
            rl.append(round(L.item(), 5))
    return {"loss_neuron": losses, "loss_cpu": rl, "mem_MiB": mem,
            "params": max(rel(a, full(b)) for a, b in zip(ref.parameters(), m.parameters()))}


# ---------------- TP ----------------
@case
def tp_extras():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                   SequenceParallel, parallelize_module)
    mesh = init_device_mesh(DEVN, (W,))
    out = {}

    class M(nn.Module):
        def __init__(s):
            super().__init__(); torch.manual_seed(0)
            s.emb = nn.Embedding(64, 32); s.norm = nn.RMSNorm(32); s.attn_in = nn.Linear(32, 64); s.attn_out = nn.Linear(64, 32)
            s.norm2 = nn.RMSNorm(32); s.head = nn.Linear(32, 64, bias=False)
        def forward(s, ids):
            h = s.emb(ids)
            h = h + s.attn_out(F.silu(s.attn_in(s.norm(h))))
            return s.head(s.norm2(h))
    ids = [torch.randint(0, 64, (2, 16), generator=g(r)) for r in range(1)]
    ref = M(); lr = F.cross_entropy(ref(ids[0]).flatten(0, 1), ids[0].flatten()); lr.backward()
    plans = {
        "emb_colwise_shard1_out": {"emb": ColwiseParallel(output_layouts=Replicate()), "attn_in": ColwiseParallel(), "attn_out": RowwiseParallel()},
        "seqpar_rmsnorm": {"emb": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                           "norm": SequenceParallel(),
                           "attn_in": ColwiseParallel(input_layouts=Shard(1)), "attn_out": RowwiseParallel(output_layouts=Shard(1)),
                           "norm2": SequenceParallel(),
                           "head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate())},
        "prepare_module_input": {"emb": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                                 "norm": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
                                 "attn_in": ColwiseParallel(), "attn_out": RowwiseParallel()},
    }
    for name, plan in plans.items():
        try:
            m = M().to(dev)
            parallelize_module(m, mesh, plan)
            lo = m(ids[0].to(dev))
            lo = lo.full_tensor() if hasattr(lo, "full_tensor") else lo
            ln = F.cross_entropy(lo.flatten(0, 1), ids[0].to(dev).flatten()); ln.backward(); sync()
            out[name] = f"loss {rel(lr, ln):.3g} grads {max(rel(a.grad, full(b.grad)) for a, b in zip(ref.parameters(), m.parameters()) if b.grad is not None):.3g}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    return out


@case
def tp_loss_parallel_options():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    from torch.distributed.tensor.parallel import loss_parallel
    mesh = init_device_mesh(DEVN, (W,))
    x = torch.randn(32, 64, generator=g(0)); y = torch.randint(0, 64, (32,), generator=g(1)); y[::5] = -100
    out = {}
    for name, kw in {"ignore_index": {}, "label_smooth_0.1": {"label_smoothing": 0.1}, "sum": {"reduction": "sum"},
                     "weight": {"weight": torch.rand(64, generator=g(3)) + 0.5}}.items():
        xc = x.clone().requires_grad_(); lc = F.cross_entropy(xc, y, **kw); lc.backward()
        try:
            xd = distribute_tensor(x.to(dev), mesh, [Shard(1)]).requires_grad_()
            kwn = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kw.items()}
            with loss_parallel():
                ln = F.cross_entropy(xd, y.to(dev), **kwn); ln.backward()
            sync()
            out[name] = f"loss {rel(lc, full(ln)):.3g} grad {rel(xc.grad, xd.grad.full_tensor()):.3g}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return out


@case
def dtensor_redistribute():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    mesh = init_device_mesh(DEVN, (W,))
    out = {}
    base = torch.arange(10 * 6, dtype=torch.float32).view(10, 6)  # 10 rows: uneven over 4
    local = base * (rank + 1)
    pd = DTensor.from_local(local.to(dev), mesh, [Partial()], run_check=False)
    exp = base * sum(range(1, W + 1))
    for tgt in ("R", "S0", "S1"):
        pl = {"R": [Replicate()], "S0": [Shard(0)], "S1": [Shard(1)]}[tgt]
        try:
            r = pd.redistribute(mesh, pl); sync()
            out[f"partial->{tgt}"] = torch.equal(r.full_tensor().cpu(), exp)
        except Exception as e:
            out[f"partial->{tgt}"] = f"ERR {str(e)[:80]}"
    dt = distribute_tensor(base.to(dev), mesh, [Shard(0)])
    for a, b in (("S0", "S1"), ("S0", "R")):
        r = dt.redistribute(mesh, [Shard(1)] if b == "S1" else [Replicate()]); sync()
        out[f"uneven {a}->{b}"] = torch.equal(r.full_tensor().cpu(), base)
    pa = DTensor.from_local(local.to(dev), mesh, [Partial("avg")], run_check=False)
    out["partial_avg->R"] = torch.allclose(pa.redistribute(mesh, [Replicate()]).full_tensor().cpu(), base * sum(range(1, W + 1)) / W)
    pm = DTensor.from_local(local.to(dev), mesh, [Partial("max")], run_check=False)
    out["partial_max->S0"] = torch.equal(pm.redistribute(mesh, [Shard(0)]).full_tensor().cpu(), base * W)
    return out


# ---------------- collectives edge cases ----------------
@case
def big_collectives():
    out = {}
    for dt, n in ((torch.bfloat16, 3 * 2**29), (torch.float32, 2**29 + 7)):  # 3 GiB bf16, 2 GiB fp32
        try:
            t = torch.full((n,), float(rank + 1), dtype=dt, device=dev)
            t[-1] = rank * 2
            dist.all_reduce(t); sync()
            out[f"all_reduce {str(dt)[6:]} {n * t.element_size() / 2**30:.1f}GiB"] = (t[0].item(), t[-1].item(), t[n // 2].item())
            del t
        except Exception as e:
            out[f"all_reduce {str(dt)[6:]}"] = f"ERR {str(e)[:120]}".replace("\n", " ")
        torch.neuron.empty_cache() if DEVN == "neuron" else None
    try:
        n = 2**28  # 1 GiB fp32 per rank
        x = torch.full((n,), float(rank), device=dev)
        o = torch.empty(W * n, device=dev)
        dist.all_gather_into_tensor(o, x); sync()
        out["all_gather 1GiB/rank"] = [o[i * n].item() for i in range(W)] + [o[-1].item()]
    except Exception as e:
        out["all_gather 1GiB/rank"] = f"ERR {str(e)[:120]}".replace("\n", " ")
    out["expect"] = "all_reduce (10, 12, 10); all_gather [0,1,2,3,3]"
    return out


@case
def many_small_and_coalesced():
    out = {}
    t0 = time.time()
    ts = [torch.full((7,), float(rank + i), device=dev) for i in range(500)]
    for t in ts:
        dist.all_reduce(t)
    sync()
    out["500_small_allreduce_ok"] = all(torch.allclose(t.cpu(), torch.full((7,), float(sum(range(W)) + i * W))) for i, t in enumerate(ts))
    out["500_small_secs"] = round(time.time() - t0, 2)
    try:
        from torch.distributed.distributed_c10d import _coalescing_manager
        xs = [torch.full((5,), float(rank), device=dev) for _ in range(20)]
        with _coalescing_manager(async_ops=True) as cm:
            for x in xs:
                dist.all_reduce(x)
        cm.wait(); sync()
        out["coalescing_manager"] = all(torch.allclose(x.cpu(), torch.full((5,), float(sum(range(W))))) for x in xs)
    except Exception as e:
        out["coalescing_manager"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    try:
        xs = [torch.full((3,), float(rank), device=dev), torch.full((2, 2), 1.0, device=dev)]
        dist.all_reduce_coalesced(xs); sync()
        out["all_reduce_coalesced"] = torch.allclose(xs[0].cpu(), torch.full((3,), float(sum(range(W))))) and torch.allclose(xs[1].cpu(), torch.full((2, 2), float(W)))
    except Exception as e:
        out["all_reduce_coalesced"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return out


@case
def int64_and_precision():
    out = {}
    big = 3_000_000_000 + rank  # > 2^31
    t = torch.tensor([big, -big, 2**40 + rank, 5], dtype=torch.int64, device=dev)
    dist.all_reduce(t); sync()
    exp = [sum(3_000_000_000 + r for r in range(W)), -sum(3_000_000_000 + r for r in range(W)), sum(2**40 + r for r in range(W)), 5 * W]
    out["int64_sum_big"] = (t.cpu().tolist(), exp)
    t = torch.tensor([big], dtype=torch.int64, device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); sync()
    out["int64_max_big"] = (t.item(), 3_000_000_000 + W - 1)
    o = torch.empty(W, dtype=torch.int64, device=dev)
    dist.all_gather_into_tensor(o, torch.tensor([big], dtype=torch.int64, device=dev)); sync()
    out["int64_allgather_big"] = (o.cpu().tolist(), [3_000_000_000 + r for r in range(W)])
    b = torch.tensor([3_000_000_000 + 7], dtype=torch.int64, device=dev); dist.broadcast(b, src=1); sync()
    out["int64_broadcast_big"] = (b.item(), 3_000_000_007)
    # bf16 SUM precision: each rank adds a value; compare with fp32 sum rounded once
    vals = [torch.randn(4096, generator=g(r)) for r in range(W)]
    x = vals[rank].to(torch.bfloat16).to(dev); dist.all_reduce(x); sync()
    exact = sum(v.to(torch.bfloat16).float() for v in vals)
    out["bf16_sum_rel_err_vs_exact"] = rel(exact, x)
    out["bf16_sum_rel_err_single_round"] = rel(exact, exact.to(torch.bfloat16))
    return out


@case
def mixed_cpu_and_objects():
    out = {}
    c = torch.full((4,), float(rank))
    dist.all_reduce(c)
    out["cpu_tensor_allreduce_mixed_backend"] = torch.allclose(c, torch.full((4,), float(sum(range(W)))))
    objs = [None] * W
    try:
        dist.all_gather_object(objs, {"r": rank, "t": torch.full((2,), float(rank), device=dev)})
        out["all_gather_object_neuron_tensor"] = [(o["r"], o["t"].device.type, o["t"].cpu().tolist()) for o in objs]
    except Exception as e:
        out["all_gather_object_neuron_tensor"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    lst = [torch.arange(3, device=dev) * 7 if rank == 2 else None]
    try:
        dist.broadcast_object_list(lst, src=2)
        out["broadcast_object_list_neuron_tensor"] = (str(lst[0].device), lst[0].cpu().tolist())
    except Exception as e:
        out["broadcast_object_list_neuron_tensor"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    try:
        dist.barrier(device_ids=[rank]); out["barrier_device_ids"] = "ok"
    except Exception as e:
        out["barrier_device_ids"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    try:
        dist.monitored_barrier(timeout=timedelta(seconds=20)); out["monitored_barrier"] = "ok"
    except Exception as e:
        out["monitored_barrier"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    import torch.distributed.nn.functional as dnf
    x = torch.full((3,), float(rank + 1), device=dev, requires_grad=True)
    def f(t):
        return dnf.all_reduce(t * 2).sum()
    try:
        cf = torch.compile(f, backend="neuron", dynamic=False)
        l = cf(x); l.backward(); sync()
        out["dist_nn_allreduce_compiled_grad"] = (l.item(), x.grad.cpu().tolist(), "expect", 2 * 3 * sum(range(1, W + 1)), [2.0 * W] * 3)
    except Exception as e:
        out["dist_nn_allreduce_compiled_grad"] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    return out


@case
def timeout_one_rank_stalls():
    t = torch.ones(4, device=dev)
    t0 = time.time()
    if rank == 3:
        time.sleep(90)
    try:
        dist.all_reduce(t); sync()
        return {"result": t.cpu().tolist(), "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " "), "secs": round(time.time() - t0, 1)}


@case
def pipelining_gpipe():
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
    torch.manual_seed(0)
    layers = [nn.Linear(16, 16) for _ in range(W)]
    stage_mod = layers[rank].to(dev)
    stage = PipelineStage(stage_mod, rank, W, dev)
    sched = ScheduleGPipe(stage, n_microbatches=2, loss_fn=lambda o, t: o.pow(2).mean())
    x = torch.randn(8, 16, generator=g(0)).to(dev)
    t0 = time.time()
    try:
        if rank == 0:
            sched.step(x)
        elif rank == W - 1:
            sched.step(target=torch.zeros(8, 16, device=dev))
        else:
            sched.step()
        sync()
        return {"ok": True, "secs": round(time.time() - t0, 1), "w_grad_norm": stage_mod.weight.grad.norm().item()}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " "), "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    try:
        r = CASES[CASE]()
        print(f"R{rank} {CASE}: {r}", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"R{rank} {CASE}: ERROR {type(e).__name__}: {str(e)[:300]}".replace("\n", " "), flush=True)
    try:
        dist.destroy_process_group()
    except Exception:
        pass
