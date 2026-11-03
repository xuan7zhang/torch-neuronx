"""Distributed round 4. Run: torchrun --nproc-per-node 4 probe_dist4.py <case>. Each rank prints 'R<rank> <case>: ...'."""
import copy
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
torch.neuron.set_device(local)
dev = torch.device("neuron", local)
CASE = sys.argv[1]
dist.init_process_group(backend="cpu:gloo,neuron:neuron")
CASES = {}


def case(f):
    CASES[f.__name__] = f
    return f


def g(s):
    return torch.Generator().manual_seed(s)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))


def ref_ddp_train(make, steps, batch_fn, opt_fn=lambda p: torch.optim.SGD(p, lr=0.1), loss_fn=None):
    """Single-process CPU reference for DDP: global batch = concat of all ranks' batches, mean loss."""
    m = make()
    o = opt_fn(m.parameters())
    for s in range(steps):
        xs = [batch_fn(s, r) for r in range(W)]
        o.zero_grad()
        if loss_fn is None:
            sum(m(x).pow(2).mean() for x in xs).div(W).backward()
        else:
            sum(loss_fn(m, x, s) for x in xs).div(W).backward()
        o.step()
    return m


# ---------------- collectives: ops x dtypes ----------------
@case
def allreduce_ops_dtypes():
    out = {}
    for dt in (torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool):
        for op in (dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.AVG, dist.ReduceOp.PRODUCT,
                   dist.ReduceOp.BAND, dist.ReduceOp.BOR, dist.ReduceOp.BXOR):
            vals = [torch.tensor([r + 1, (r * 3) % 5, 1 - (r % 2), 2], dtype=torch.float64) for r in range(W)]
            if dt == torch.bool:
                vals = [(v % 2).bool() for v in vals]
            else:
                vals = [v.to(dt) for v in vals]
            if op == dist.ReduceOp.SUM:
                exp = sum(v.long() for v in vals) if dt != torch.bool else torch.stack(vals).any(0)
            elif op == dist.ReduceOp.MAX:
                exp = torch.stack(vals).amax(0)
            elif op == dist.ReduceOp.MIN:
                exp = torch.stack(vals).amin(0)
            elif op == dist.ReduceOp.AVG:
                exp = torch.stack([v.double() for v in vals]).mean(0)
            elif op == dist.ReduceOp.PRODUCT:
                exp = torch.stack([v.long() for v in vals]).prod(0) if dt != torch.bool else torch.stack(vals).all(0)
            else:
                if dt.is_floating_point:
                    continue
                f = {dist.ReduceOp.BAND: torch.bitwise_and, dist.ReduceOp.BOR: torch.bitwise_or, dist.ReduceOp.BXOR: torch.bitwise_xor}[op]
                exp = vals[0]
                for v in vals[1:]:
                    exp = f(exp, v)
            t = vals[rank].clone().to(dev)
            key = f"{str(dt)[6:]}/{str(op).split('.')[-1]}"
            try:
                dist.all_reduce(t, op=op)
                torch.neuron.synchronize()
                got = t.cpu()
                ok = got.dtype == dt and torch.allclose(got.double(), exp.to(got.dtype).double() if dt != torch.bool else exp.double(), atol=1e-2)
                out[key] = "ok" if ok else f"BAD got {got.tolist()} exp {exp.tolist()}"
            except Exception as e:
                out[key] = f"ERR {type(e).__name__}: {str(e)[:70]}".replace("\n", " ")
            dist.barrier()
    bad = {k: v for k, v in out.items() if v != "ok"}
    return f"{len(out) - len(bad)}/{len(out)} ok; not ok: {bad}"


@case
def reduce_scatter_ops():
    out = {}
    for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG, dist.ReduceOp.MAX, dist.ReduceOp.MIN):
        for dt in (torch.float32, torch.bfloat16):
            inp = (torch.arange(W * 3, dtype=torch.float32) * (rank + 1)).to(dt).to(dev)
            o = torch.empty(3, dtype=dt, device=dev)
            allv = torch.stack([torch.arange(W * 3, dtype=torch.float32) * (r + 1) for r in range(W)])
            red = {dist.ReduceOp.SUM: allv.sum(0), dist.ReduceOp.AVG: allv.mean(0), dist.ReduceOp.MAX: allv.amax(0), dist.ReduceOp.MIN: allv.amin(0)}[op]
            exp = red[rank * 3:(rank + 1) * 3]
            key = f"{str(dt)[6:]}/{str(op).split('.')[-1]}"
            try:
                dist.reduce_scatter_tensor(o, inp, op=op); torch.neuron.synchronize()
                out[key] = "ok" if torch.allclose(o.float().cpu(), exp, rtol=1e-2) else f"BAD {o.float().cpu().tolist()} exp {exp.tolist()}"
            except Exception as e:
                out[key] = f"ERR {str(e)[:70]}".replace("\n", " ")
    # premul sum (used by FSDP gradient divide on NCCL)
    try:
        from torch.distributed.distributed_c10d import _make_nccl_premul_sum
        t = torch.ones(4, device=dev); dist.all_reduce(t, op=_make_nccl_premul_sum(0.5)); torch.neuron.synchronize()
        out["premul_sum"] = "ok" if torch.allclose(t.cpu(), torch.full((4,), 0.5 * W)) else f"BAD {t.cpu().tolist()}"
    except Exception as e:
        out["premul_sum"] = f"ERR {type(e).__name__}: {str(e)[:70]}".replace("\n", " ")
    bad = {k: v for k, v in out.items() if v != "ok"}
    return f"{len(out) - len(bad)}/{len(out)} ok; not ok: {bad}"


@case
def noncontig_inputs():
    out = {}
    base = (torch.arange(24, dtype=torch.float32).view(4, 6) + 100 * rank)
    allb = [torch.arange(24, dtype=torch.float32).view(4, 6) + 100 * r for r in range(W)]
    # all_reduce on transposed (no storage offset)
    t = base.to(dev).t()
    dist.all_reduce(t); torch.neuron.synchronize()
    out["allreduce_transposed"] = torch.equal(t.cpu(), sum(allb).t())
    # all_reduce on stride-2 slice (offset 0)
    t2 = base.to(dev); v = t2[:, ::2]
    dist.all_reduce(v); torch.neuron.synchronize()
    exp = base.clone(); exp[:, ::2] = sum(allb)[:, ::2]
    out["allreduce_strided_slice"] = torch.equal(t2.cpu(), exp)
    # all_gather_into_tensor with non-contiguous input
    o = torch.empty(W * 6, 4, device=dev)
    try:
        dist.all_gather_into_tensor(o, base.to(dev).t()); torch.neuron.synchronize()
        out["allgather_into_noncontig_in"] = torch.equal(o.cpu(), torch.cat([b.t() for b in allb]))
    except Exception as e:
        out["allgather_into_noncontig_in"] = f"ERR {str(e)[:80]}"
    # broadcast into transposed view from src=2
    t3 = base.to(dev).t()
    dist.broadcast(t3, src=2); torch.neuron.synchronize()
    out["broadcast_src2_transposed"] = torch.equal(t3.cpu(), allb[2].t())
    # all_gather list where output tensors are views of one buffer
    buf = torch.zeros(W, 4, 6, device=dev)
    dist.all_gather(list(buf.unbind(0)), base.to(dev)); torch.neuron.synchronize()
    out["allgather_list_into_unbind_views"] = torch.equal(buf.cpu(), torch.stack(allb))
    # reduce_scatter_tensor into a row slice (offset, contiguous)
    big = torch.zeros(3, 6, device=dev)
    dist.reduce_scatter_tensor(big[1], torch.arange(W * 6, dtype=torch.float32, device=dev) + rank); torch.neuron.synchronize()
    exp = torch.zeros(3, 6); exp[1] = (torch.arange(W * 6, dtype=torch.float32) * W + sum(range(W)))[rank * 6:(rank + 1) * 6]
    out["reduce_scatter_into_row_slice"] = torch.equal(big.cpu(), exp)
    return out


@case
def async_works():
    out = {}
    ts = [torch.full((1000,), float(rank + i), device=dev) for i in range(8)]
    works = [dist.all_reduce(t, async_op=True) for t in ts]
    # compute while comms in flight, using the tensors after wait
    for w_ in reversed(works):
        w_.wait()
    out["8_async_allreduce_wait_reverse"] = all(torch.allclose(t.cpu(), torch.full((1000,), float(sum(range(W)) + i * W))) for i, t in enumerate(ts))
    # use result immediately after wait in compute
    a = torch.full((256, 256), float(rank), device=dev)
    wk = dist.all_reduce(a, async_op=True); wk.wait(); b = a @ torch.eye(256, device=dev)
    out["wait_then_compute"] = torch.allclose(b.cpu(), torch.full((256, 256), float(sum(range(W)))))
    # mutate input after async_op before wait (user bug on CUDA too) skipped; check is_completed and get_future
    c = torch.ones(10, device=dev)
    wk = dist.all_reduce(c, async_op=True)
    fut = wk.get_future(); fut.wait()
    out["get_future"] = torch.allclose(c.cpu(), torch.full((10,), float(W)))
    out["is_completed_after_wait"] = wk.is_completed()
    # mixed async: all_gather + reduce_scatter interleaved on different tensors
    x = torch.full((4,), float(rank), device=dev); o = torch.empty(W * 4, device=dev)
    y = torch.arange(W * 2, dtype=torch.float32, device=dev); oy = torch.empty(2, device=dev)
    w1 = dist.all_gather_into_tensor(o, x, async_op=True); w2 = dist.reduce_scatter_tensor(oy, y, async_op=True)
    w2.wait(); w1.wait()
    out["ag_rs_interleaved"] = torch.equal(o.cpu(), torch.arange(W).float().repeat_interleave(4)) and \
        torch.equal(oy.cpu(), (torch.arange(W * 2, dtype=torch.float32) * W)[rank * 2:rank * 2 + 2])
    return out


@case
def subgroup_partial_membership():
    out = {}
    g01 = dist.new_group([0, 1]); g23 = dist.new_group([2, 3]); g13 = dist.new_group([1, 3])
    t = torch.full((4,), float(rank), device=dev)
    if rank in (0, 1):
        dist.all_reduce(t, group=g01)
    else:
        dist.all_reduce(t, group=g23)
    torch.neuron.synchronize()
    out["disjoint_groups_concurrent"] = torch.allclose(t.cpu(), torch.full((4,), 1.0 if rank < 2 else 5.0))
    # only ranks 1,3 call; 0,2 do something else
    u = torch.full((4,), float(rank), device=dev)
    if rank in (1, 3):
        dist.all_reduce(u, group=g13)
    else:
        u = u * 10
    torch.neuron.synchronize()
    out["subset_group_others_idle"] = torch.allclose(u.cpu(), torch.full((4,), 4.0 if rank in (1, 3) else rank * 10.0))
    # broadcast in subgroup with global src rank
    v = torch.full((3,), float(rank), device=dev)
    if rank in (1, 3):
        dist.broadcast(v, src=3, group=g13)
    torch.neuron.synchronize()
    out["subgroup_broadcast_global_src"] = torch.allclose(v.cpu(), torch.full((3,), 3.0 if rank in (1, 3) else float(rank)))
    dist.barrier()
    # world collective after subgroup collectives
    w_ = torch.ones(2, device=dev); dist.all_reduce(w_); torch.neuron.synchronize()
    out["world_after_subgroups"] = torch.allclose(w_.cpu(), torch.full((2,), float(W)))
    return out


@case
def reinit_process_group():
    out = {}
    t = torch.ones(4, device=dev); dist.all_reduce(t); torch.neuron.synchronize()
    out["first"] = torch.allclose(t.cpu(), torch.full((4,), float(W)))
    dist.destroy_process_group()
    try:
        os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 7)
        dist.init_process_group(backend="cpu:gloo,neuron:neuron")
        t = torch.full((4,), float(rank), device=dev); dist.all_reduce(t); torch.neuron.synchronize()
        out["after_reinit"] = torch.allclose(t.cpu(), torch.full((4,), float(sum(range(W)))))
    except Exception as e:
        out["after_reinit"] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return out


# ---------------- DDP ----------------
def run_ddp(name, make, steps=3, ddp_kw={}, compile_=False, batch_fn=None, fwd=None, ref_loss=None, opt=None):
    from torch.nn.parallel import DistributedDataParallel as DDP
    batch_fn = batch_fn or (lambda s, r: torch.randn(8, 16, generator=g(s * 10 + r)))
    m = make().to(dev)
    d = DDP(m, device_ids=None, **ddp_kw)
    f = torch.compile(d, backend="neuron", dynamic=False) if compile_ else d
    o = (opt or (lambda p: torch.optim.SGD(p, lr=0.1)))(d.parameters())
    for s in range(steps):
        o.zero_grad()
        x = batch_fn(s, rank).to(dev)
        (fwd(f, x, s) if fwd else f(x).pow(2).mean()).backward()
        o.step()
    ref = ref_ddp_train(make, steps, batch_fn, opt_fn=(opt or (lambda p: torch.optim.SGD(p, lr=0.1))), loss_fn=ref_loss)
    return {"params": max(rel(a, b) for a, b in zip(ref.parameters(), m.parameters()))}


@case
def ddp_variants():
    out = {}
    out["basic"] = run_ddp("basic", mlp)
    out["gradient_as_bucket_view"] = run_ddp("gabv", mlp, ddp_kw=dict(gradient_as_bucket_view=True))
    out["small_buckets"] = run_ddp("bucket", mlp, ddp_kw=dict(bucket_cap_mb=0.001))
    out["static_graph"] = run_ddp("static", mlp, ddp_kw=dict(static_graph=True))
    out["compiled"] = run_ddp("compiled", mlp, compile_=True)
    out["adamw"] = run_ddp("adamw", mlp, opt=lambda p: torch.optim.AdamW(p, lr=1e-2))

    class Branchy(nn.Module):
        def __init__(s):
            super().__init__(); torch.manual_seed(0)
            s.a = nn.Linear(16, 4); s.b = nn.Linear(16, 4); s.unused = nn.Linear(16, 4)

        def forward(s, x, step=0):
            return s.a(x) if step % 2 == 0 else s.b(x)
    out["find_unused_alternating"] = run_ddp(
        "unused", Branchy, steps=4, ddp_kw=dict(find_unused_parameters=True),
        fwd=lambda f, x, s: f(x, s).pow(2).mean(), ref_loss=lambda m, x, s: m(x, s).pow(2).mean())
    return out


@case
def ddp_no_sync_accum():
    from torch.nn.parallel import DistributedDataParallel as DDP
    m = mlp().to(dev); d = DDP(m)
    o = torch.optim.SGD(d.parameters(), lr=0.1)
    micro = 3
    for s in range(2):
        o.zero_grad()
        for k in range(micro):
            x = torch.randn(8, 16, generator=g(s * 100 + k * 10 + rank)).to(dev)
            if k < micro - 1:
                with d.no_sync():
                    (d(x).pow(2).mean() / micro).backward()
            else:
                (d(x).pow(2).mean() / micro).backward()
        o.step()
    ref = mlp(); ro = torch.optim.SGD(ref.parameters(), lr=0.1)
    for s in range(2):
        ro.zero_grad()
        sum(ref(torch.randn(8, 16, generator=g(s * 100 + k * 10 + r))).pow(2).mean() / micro for k in range(micro) for r in range(W)).div(W).backward()
        ro.step()
    return {"params": max(rel(a, b) for a, b in zip(ref.parameters(), m.parameters()))}


@case
def ddp_batchnorm_buffers():
    from torch.nn.parallel import DistributedDataParallel as DDP
    out = {}
    def mk():
        torch.manual_seed(0); return nn.Sequential(nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 4))
    # broadcast_buffers=True: rank 0 buffers broadcast at forward start
    m = mk().to(dev); d = DDP(m, broadcast_buffers=True)
    for s in range(2):
        d(torch.randn(8, 16, generator=g(s * 10 + rank)).to(dev)).sum().backward()
    rm = m[1].running_mean.cpu()
    allrm = [torch.zeros_like(rm) for _ in range(W)]
    dist.all_gather_object(allrm, rm)
    out["running_mean_same_on_all_ranks"] = all(torch.allclose(allrm[0], a) for a in allrm)
    out["num_batches_tracked"] = int(m[1].num_batches_tracked)
    # SyncBatchNorm
    torch.manual_seed(0)
    sm = nn.SyncBatchNorm.convert_sync_batchnorm(mk()).to(dev); sd = DDP(sm)
    xs = [torch.randn(8, 16, generator=g(50 + r)) for r in range(W)]
    sd.train(); y = sd(xs[rank].to(dev)); y.pow(2).mean().backward()
    ref = mk(); ref.train(); yr = ref(torch.cat(xs)); (yr.pow(2).mean()).backward()
    out["syncbn_out"] = rel(yr[rank * 8:(rank + 1) * 8], y)
    out["syncbn_running_mean"] = rel(ref[1].running_mean, sm[1].running_mean)
    out["syncbn_weight_grad"] = rel(ref[0].weight.grad, sm[0].weight.grad)
    return out


# ---------------- FSDP2 ----------------
def fsdp_mod():
    from torch.distributed.fsdp import fully_shard
    return fully_shard


def run_fsdp(make, shard_fn, steps=3, opt=lambda p: torch.optim.AdamW(p, lr=1e-2), fwd=None, ref_fwd=None, bs=lambda s, r: torch.randn(8, 16, generator=g(s * 10 + r))):
    m = make().to(dev)
    shard_fn(m)
    o = opt([p for p in m.parameters() if p.requires_grad])
    for s in range(steps):
        o.zero_grad()
        x = bs(s, rank).to(dev)
        (fwd(m, x) if fwd else m(x).pow(2).mean()).backward()
        o.step()
    ref = make(); ro = opt([p for p in ref.parameters() if p.requires_grad])
    for s in range(steps):
        ro.zero_grad()
        sum((ref_fwd or (lambda mm, x: mm(x).pow(2).mean()))(ref, bs(s, r)) for r in range(W)).div(W).backward()
        ro.step()
    full = lambda t: t.full_tensor() if hasattr(t, "full_tensor") else t
    return max(rel(a, full(b)) for a, b in zip(ref.parameters(), m.parameters()))


@case
def fsdp2_frozen_mixed():
    fs = fsdp_mod()
    def mk():
        m = mlp(); m[0].requires_grad_(False); m[2].bias.requires_grad_(False); return m
    def shard(m):
        for i in (0, 2, 4):
            fs(m[i])
        fs(m)
    return {"params": run_fsdp(mk, shard)}


@case
def fsdp2_tied():
    fs = fsdp_mod()
    class LM(nn.Module):
        def __init__(s):
            super().__init__(); torch.manual_seed(0)
            s.emb = nn.Embedding(64, 16); s.l = nn.Linear(16, 16); s.head = nn.Linear(16, 64, bias=False); s.head.weight = s.emb.weight
        def forward(s, ids):
            return F.cross_entropy(s.head(torch.relu(s.l(s.emb(ids)))).flatten(0, 1), ids.flatten())
    def shard(m):
        fs(m.l); fs(m)
    return {"params": run_fsdp(LM, shard, fwd=lambda m, x: m(x), ref_fwd=lambda m, x: m(x),
                               bs=lambda s, r: torch.randint(0, 64, (4, 8), generator=g(s * 10 + r)))}


@case
def fsdp2_no_grad_sync_accum():
    fs = fsdp_mod()
    m = mlp().to(dev)
    for i in (0, 2, 4):
        fs(m[i])
    fs(m)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    micro = 3
    for s in range(2):
        o.zero_grad()
        for k in range(micro):
            m.set_requires_gradient_sync(k == micro - 1)
            (m(torch.randn(8, 16, generator=g(s * 100 + k * 10 + rank)).to(dev)).pow(2).mean() / micro).backward()
        o.step()
    ref = mlp(); ro = torch.optim.SGD(ref.parameters(), lr=0.1)
    for s in range(2):
        ro.zero_grad()
        sum(ref(torch.randn(8, 16, generator=g(s * 100 + k * 10 + r))).pow(2).mean() / micro for k in range(micro) for r in range(W)).div(W).backward()
        ro.step()
    return {"params": max(rel(a, b.full_tensor()) for a, b in zip(ref.parameters(), m.parameters()))}


@case
def fsdp2_explicit_unshard_and_eval():
    fs = fsdp_mod()
    m = mlp().to(dev)
    for i in (0, 2, 4):
        fs(m[i], reshard_after_forward=2 if i == 0 else True)
    fs(m)
    out = {}
    x = torch.randn(8, 16, generator=g(0))
    ref = mlp()
    with torch.no_grad():
        out["no_grad_eval"] = rel(ref(x), m(x.to(dev)))
    m.unshard()
    with torch.no_grad():
        out["after_unshard"] = rel(ref(x), m(x.to(dev)))
    m.reshard()
    y = m(x.to(dev)); y.pow(2).mean().backward()
    ref(x).pow(2).mean().backward()
    out["grad_after_reshard"] = max(rel(a.grad, b.grad.full_tensor()) for a, b in zip(ref.parameters(), m.parameters()))
    return out


@case
def fsdp2_optim_state_dict():
    fs = fsdp_mod()
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
    def build():
        m = mlp().to(dev)
        for i in (0, 2, 4):
            fs(m[i])
        fs(m)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-2)
    m, o = build()
    for s in range(2):
        o.zero_grad(); m(torch.randn(8, 16, generator=g(s * 10 + rank)).to(dev)).pow(2).mean().backward(); o.step()
    msd, osd = get_state_dict(m, o, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    m2, o2 = build()
    set_state_dict(m2, o2, model_state_dict=msd, optim_state_dict=osd, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False))
    for mm, oo in ((m, o), (m2, o2)):
        oo.zero_grad(); mm(torch.randn(8, 16, generator=g(99 + rank)).to(dev)).pow(2).mean().backward(); oo.step()
    out = {"params_after_resume": max(rel(a.full_tensor(), b.full_tensor()) for a, b in zip(m.parameters(), m2.parameters()))}
    k0 = next(iter(osd["state"]))
    out["full_osd_exp_avg_is_cpu_full"] = str((osd["state"][k0]["exp_avg"].device, tuple(osd["state"][k0]["exp_avg"].shape)))
    return out


@case
def context_parallel_sdpa():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    mesh = init_device_mesh("neuron", (W,))
    B, H, S, D = 1, 2, 64 * W, 32
    q = torch.randn(B, H, S, D, generator=g(0)); k = torch.randn(B, H, S, D, generator=g(1)); v = torch.randn(B, H, S, D, generator=g(2))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    qn, kn, vn = (t.to(dev).requires_grad_() for t in (q, k, v))
    try:
        with context_parallel(mesh, buffers=[qn, kn, vn], buffer_seq_dims=[2, 2, 2]):
            o = F.scaled_dot_product_attention(qn, kn, vn, is_causal=True)
        torch.neuron.synchronize()
        return {"note": "ran", "out_shape": tuple(o.shape)}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:200]}".replace("\n", " ")}


@case
def rank_raises_then_continue():
    out = {}
    t = torch.ones(4, device=dev)
    try:
        if rank == 1:
            raise ValueError("rank 1 bad batch")
        x = t * 2
    except ValueError:
        x = t * 2  # caught locally; all ranks still call the collective
    dist.all_reduce(x); torch.neuron.synchronize()
    out["after_local_exception"] = torch.allclose(x.cpu(), torch.full((4,), 2.0 * W))
    # a collective with mismatched shapes across ranks: CUDA/NCCL hangs or errors; check we get an error not a hang
    return out


@case
def mismatched_shapes_allreduce():
    t = torch.ones(4 if rank != 2 else 6, device=dev)
    try:
        dist.all_reduce(t); torch.neuron.synchronize()
        return {"result": t.cpu().tolist()}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " ")}


@case
def mismatched_dtype_allreduce():
    t = torch.ones(4, device=dev, dtype=torch.float32 if rank != 2 else torch.bfloat16)
    try:
        dist.all_reduce(t); torch.neuron.synchronize()
        return {"result": t.cpu().tolist(), "dtype": str(t.dtype)}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " ")}


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
