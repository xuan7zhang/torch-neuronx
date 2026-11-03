"""Distributed round 4b. torchrun --nproc-per-node 4 probe_dist4b.py <case>"""
import os, sys, traceback
import torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist, torch_neuronx
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
dist.init_process_group("cpu:gloo,neuron:neuron")
g = lambda s: torch.Generator().manual_seed(s)
def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu(); return ((a - b).norm() / (a.norm() + 1e-12)).item()
CASES = {}
def case(f):
    CASES[f.__name__] = f; return f


@case
def ep_all_to_all_autograd_uneven():
    import torch.distributed._functional_collectives as fc
    # each rank sends split[r][j] rows to rank j; rows are D-dim tokens; expert on each rank is a Linear
    D = 8
    splits = [[(r + j) % 3 + 1 for j in range(W)] for r in range(W)]
    send = splits[rank]; recv = [splits[j][rank] for j in range(W)]
    toks_all = [torch.randn(sum(splits[r]), D, generator=g(r)) for r in range(W)]
    torch.manual_seed(0); experts = [nn.Linear(D, D) for _ in range(W)]
    # CPU reference: route tokens, apply expert, route back, loss
    ref_in = [t.clone().requires_grad_() for t in toks_all]
    chunks = [list(t.split(splits[r])) for r, t in enumerate(ref_in)]
    out_back = [[None] * W for _ in range(W)]
    for j in range(W):
        y = experts[j](torch.cat([chunks[r][j] for r in range(W)]))
        ys = y.split([splits[r][j] for r in range(W)])
        for r in range(W):
            out_back[r][j] = ys[r]
    ref_loss = sum(torch.cat(out_back[r]).pow(2).sum() for r in range(W))
    ref_loss.backward()
    # Neuron EP
    x = toks_all[rank].to(dev).requires_grad_()
    ex = experts[rank].to(dev)
    y = fc.all_to_all_single_autograd(x, recv, send, dist.group.WORLD)
    z = ex(y)
    back = fc.all_to_all_single_autograd(z, send, recv, dist.group.WORLD)
    back.pow(2).sum().backward()
    torch.neuron.synchronize()
    gw = ex.weight.grad.clone()
    return {"x_grad": rel(ref_in[rank].grad, x.grad), "expert_w_grad": rel(experts[rank].weight.grad, gw),
            "fwd_back": rel(torch.cat(out_back[rank]), back)}


@case
def hsdp_2x2():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("rep", "shard"))
    def mk():
        torch.manual_seed(0); return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))
    m = mk().to(dev); fully_shard(m[0], mesh=mesh); fully_shard(m[2], mesh=mesh); fully_shard(m, mesh=mesh)
    o = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ref = mk(); ro = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    for s in range(3):
        o.zero_grad(); m(torch.randn(8, 16, generator=g(s * 10 + rank)).to(dev)).pow(2).mean().backward()
        nrm = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5); o.step()
        ro.zero_grad(); sum(ref(torch.randn(8, 16, generator=g(s * 10 + r))).pow(2).mean() for r in range(W)).div(W).backward()
        rn = torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.5); ro.step()
    nv = nrm.full_tensor() if hasattr(nrm, "full_tensor") else nrm
    return {"params": max(rel(a, b.full_tensor()) for a, b in zip(ref.parameters(), m.parameters())), "grad_norm": rel(rn, nv)}


@case
def dtensor_op_sweep():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, Replicate
    mesh = init_device_mesh("neuron", (W,))
    A = torch.randn(8, 12, generator=g(0)); B = torch.randn(8, 12, generator=g(1)); M = torch.randn(12, 8, generator=g(2))
    ops = {
        "add": lambda a, b, m: a + b, "mul": lambda a, b, m: a * b, "gelu": lambda a, b, m: F.gelu(a),
        "softmax0": lambda a, b, m: a.softmax(0), "softmax1": lambda a, b, m: a.softmax(1),
        "log_softmax1": lambda a, b, m: a.log_softmax(1), "layer_norm": lambda a, b, m: F.layer_norm(a, (12,)),
        "rms_norm": lambda a, b, m: F.rms_norm(a, (12,)), "sum0": lambda a, b, m: a.sum(0), "sum1": lambda a, b, m: a.sum(1),
        "mean": lambda a, b, m: a.mean(), "max1": lambda a, b, m: a.amax(1), "matmul": lambda a, b, m: a @ m,
        "transpose_mm": lambda a, b, m: a.t() @ b, "cumsum1": lambda a, b, m: a.cumsum(1), "cumsum0": lambda a, b, m: a.cumsum(0),
        "where": lambda a, b, m: torch.where(a > 0, a, b), "clamp": lambda a, b, m: a.clamp(-0.5, 0.5),
        "view_reshape": lambda a, b, m: a.reshape(4, 24) * 2, "cat0": lambda a, b, m: torch.cat([a, b], 0),
        "cat1": lambda a, b, m: torch.cat([a, b], 1), "topk1": lambda a, b, m: a.topk(3, 1).values,
        "norm": lambda a, b, m: a.norm(), "var1": lambda a, b, m: a.var(1), "silu_mul": lambda a, b, m: F.silu(a) * b,
        "dropout_eval": lambda a, b, m: F.dropout(a, 0.5, training=False), "pow": lambda a, b, m: a.pow(2),
        "mse": lambda a, b, m: F.mse_loss(a, b), "ce": lambda a, b, m: F.cross_entropy(a, b.argmax(1)),
        "linear": lambda a, b, m: F.linear(a, m.t()), "einsum": lambda a, b, m: torch.einsum("ij,ij->i", a, b),
    }
    placements = {"S0": [Shard(0)], "S1": [Shard(1)], "R": [Replicate()]}
    bad = {}; n = 0
    for name, f in ops.items():
        ac, bc, mc = (t.clone().requires_grad_() for t in (A, B, M))
        try:
            rc = f(ac, bc, mc); (rc.float() * 1.0).sum().backward() if rc.requires_grad else None
        except Exception:
            continue
        for pn, pl in placements.items():
            n += 1
            try:
                ad, bd = (distribute_tensor(t.to(dev), mesh, pl).requires_grad_() for t in (A, B))
                md = distribute_tensor(M.to(dev), mesh, [Replicate()]).requires_grad_()
                rd = f(ad, bd, md)
                rd.sum().backward() if rd.requires_grad else None
                torch.neuron.synchronize()
                full = rd.full_tensor()
                errs = [rel(rc, full)] + [rel(x.grad, y.grad.full_tensor()) for x, y in ((ac, ad), (bc, bd)) if x.grad is not None and y.grad is not None]
                if max(errs) > 1e-4:
                    bad[f"{name}/{pn}"] = f"rel {max(errs):.3g}"
            except Exception as e:
                bad[f"{name}/{pn}"] = f"ERR {type(e).__name__}: {str(e)[:80]}".replace("\n", " ")
    return f"{n - len(bad)}/{n} ok; not ok: {bad}"


@case
def dcp_async_save():
    import tempfile, time
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import fully_shard
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4)).to(dev)
    fully_shard(m[0]); fully_shard(m[1]); fully_shard(m)
    path = "/tmp/claude-1000/dcp_async_" + os.environ.get("MASTER_PORT", "x")
    sd = {"model": m.state_dict()}
    fut = dcp.async_save(sd, checkpoint_id=path)
    # mutate params right after (async_save must have staged a copy)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(100.0)
    fut.result()
    torch.manual_seed(0); m2 = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4)).to(dev)
    fully_shard(m2[0]); fully_shard(m2[1]); fully_shard(m2)
    sd2 = {"model": m2.state_dict()}; dcp.load(sd2, checkpoint_id=path)
    torch.manual_seed(0); ref = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4))
    return {"loaded_equals_pre_mutation": max(rel(a, b.full_tensor()) for a, b in zip(ref.parameters(), m2.parameters()))}


@case
def tp_fsdp_compile_clip():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp"))
    def mk():
        torch.manual_seed(0); return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16), nn.LayerNorm(16), nn.Linear(16, 4))
    m = mk().to(dev)
    parallelize_module(m, mesh["tp"], {"0": ColwiseParallel(), "2": RowwiseParallel()})
    fully_shard(m, mesh=mesh["dp"])
    fm = torch.compile(m, backend="neuron", dynamic=False)
    o = torch.optim.AdamW(m.parameters(), lr=1e-2, foreach=False)
    ref = mk(); ro = torch.optim.AdamW(ref.parameters(), lr=1e-2, foreach=False)
    dp = mesh["dp"].get_local_rank()
    for s in range(3):
        o.zero_grad(); fm(torch.randn(8, 16, generator=g(s * 10 + dp)).to(dev)).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.3); o.step()
        ro.zero_grad(); sum(ref(torch.randn(8, 16, generator=g(s * 10 + d))).pow(2).mean() for d in range(2)).div(2).backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.3); ro.step()
    return {"params": max(rel(a, b.full_tensor()) for a, b in zip(ref.parameters(), m.parameters()))}


if __name__ == "__main__":
    c = sys.argv[1]
    try:
        print(f"R{rank} {c}: {CASES[c]()}", flush=True)
    except Exception as e:
        traceback.print_exc(); print(f"R{rank} {c}: ERROR {type(e).__name__}: {str(e)[:300]}".replace("\n", " "), flush=True)
    dist.destroy_process_group()
