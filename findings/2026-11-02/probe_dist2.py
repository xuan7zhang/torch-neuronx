"""Distributed probes round 2. Run: torchrun --nproc-per-node N probe_dist2.py <case>"""

import copy
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ["LOCAL_RANK"])
torch.neuron.set_device(local)
dev = torch.device("neuron", local)
dist.init_process_group(backend="neuron")
CASE = sys.argv[1]
msgs = []


def check(name, ok, info=""):
    msgs.append(f"{'OK      ' if ok else 'MISMATCH'} {name} {info}")


def close(a, b, atol=1e-4):
    a = a.full_tensor() if hasattr(a, "full_tensor") else a
    b = b.full_tensor() if hasattr(b, "full_tensor") else b
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return a.shape == b.shape and torch.allclose(a, b, atol=atol, rtol=1e-4), (
        (a - b).abs().max().item() if a.shape == b.shape else f"shape {tuple(a.shape)} vs {tuple(b.shape)}")


def mlp(i=32, h=64, o=16):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def gdata(n=16, d=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, 16, generator=g)


def ref_train(model, x, y, steps=3, lr=1e-2, clip=None):
    m = copy.deepcopy(model)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(m(x), y).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
    return m


def compare_params(name, ref_m, m, atol=2e-4):
    worst = 0.0
    ok_all = True
    for (n1, p1), (n2, p2) in zip(ref_m.named_parameters(), m.named_parameters()):
        ok, err = close(p1, p2, atol)
        ok_all &= ok
        worst = err if isinstance(err, str) else max(worst, err) if not isinstance(worst, str) else worst
    check(name, ok_all, f"maxdiff={worst}")


def run():
    if CASE == "collectives_all":
        for op, exp in [(dist.ReduceOp.SUM, sum(range(1, world + 1))), (dist.ReduceOp.MAX, world),
                        (dist.ReduceOp.MIN, 1),
                        (dist.ReduceOp.AVG, sum(range(1, world + 1)) / world)]:
            for dt in [torch.float32, torch.bfloat16, torch.int32, torch.int64]:
                if op == dist.ReduceOp.AVG and not dt.is_floating_point:
                    continue
                try:
                    t = torch.full((5,), rank + 1, dtype=dt, device=dev)
                    dist.all_reduce(t, op=op)
                    check(f"all_reduce {op} {dt}", bool((t.cpu().float() == exp).all()), t.cpu().tolist()[:2])
                except Exception as e:
                    check(f"all_reduce {op} {dt}", False, f"ERROR {str(e)[:150]}")
        # all_gather_into_tensor dim0, list all_gather
        out = torch.empty(world * 3, 2, device=dev)
        dist.all_gather_into_tensor(out, torch.full((3, 2), float(rank), device=dev))
        check("all_gather_into_tensor", bool((out.cpu().view(world, 3, 2) == torch.arange(world).float().view(world, 1, 1)).all()))
        # reduce_scatter list form
        out = torch.empty(3, device=dev)
        dist.reduce_scatter(out, [torch.full((3,), float(rank * 10 + i), device=dev) for i in range(world)])
        check("reduce_scatter list", bool((out.cpu() == sum(r * 10 + rank for r in range(world))).all()), out.cpu().tolist())
        # all_to_all_single even
        inp = torch.arange(world * 2, dtype=torch.float32, device=dev) + rank * 100
        out = torch.empty_like(inp)
        dist.all_to_all_single(out, inp)
        exp = torch.tensor([r * 100 + rank * 2 + k for r in range(world) for k in range(2)], dtype=torch.float32)
        check("all_to_all_single even", bool(torch.equal(out.cpu(), exp)), out.cpu().tolist())
        # all_to_all_single uneven splits: rank r sends (j+1) elems to rank j
        in_splits = [j + 1 for j in range(world)]
        out_splits = [rank + 1] * world
        inp = torch.cat([torch.full((j + 1,), float(rank * 10 + j)) for j in range(world)]).to(dev)
        out = torch.empty(sum(out_splits), device=dev)
        try:
            dist.all_to_all_single(out, inp, out_splits, in_splits)
            exp = torch.cat([torch.full((rank + 1,), float(r * 10 + rank)) for r in range(world)])
            check("all_to_all_single uneven", bool(torch.equal(out.cpu(), exp)), out.cpu().tolist())
        except Exception as e:
            check("all_to_all_single uneven", False, f"ERROR {str(e)[:200]}")
        # all_to_all list
        outs = [torch.empty(2, device=dev) for _ in range(world)]
        try:
            dist.all_to_all(outs, [torch.full((2,), float(rank * 10 + j), device=dev) for j in range(world)])
            check("all_to_all list", all(bool((o.cpu() == r * 10 + rank).all()) for r, o in enumerate(outs)))
        except Exception as e:
            check("all_to_all list", False, f"ERROR {str(e)[:200]}")
        # gather / scatter / reduce
        for nm, fn in [
            ("gather", lambda: (lambda gl: (dist.gather(torch.full((2,), float(rank), device=dev), gl, dst=0), gl)[1])(
                [torch.empty(2, device=dev) for _ in range(world)] if rank == 0 else None)),
            ("scatter", lambda: (lambda o: (dist.scatter(o, [torch.full((2,), float(i), device=dev) for i in range(world)] if rank == 1 else None, src=1), o)[1])(torch.empty(2, device=dev))),
            ("reduce", lambda: (lambda t: (dist.reduce(t, dst=2 % world), t)[1])(torch.full((2,), float(rank + 1), device=dev))),
        ]:
            try:
                r = fn()
                if nm == "gather":
                    ok = rank != 0 or all(bool((g.cpu() == i).all()) for i, g in enumerate(r))
                elif nm == "scatter":
                    ok = bool((r.cpu() == rank).all())
                else:
                    ok = rank != 2 % world or bool((r.cpu() == sum(range(1, world + 1))).all())
                check(nm, ok)
            except Exception as e:
                check(nm, False, f"ERROR {str(e)[:200]}")
        # send/recv ring + isend/irecv
        try:
            t = torch.full((3,), float(rank), device=dev)
            r = torch.empty(3, device=dev)
            reqs = [dist.isend(t, (rank + 1) % world), dist.irecv(r, (rank - 1) % world)]
            for q in reqs:
                q.wait()
            check("isend/irecv ring", bool((r.cpu() == (rank - 1) % world).all()), r.cpu().tolist())
        except Exception as e:
            check("isend/irecv ring", False, f"ERROR {str(e)[:200]}")
        try:
            if rank == 0:
                dist.send(torch.arange(4.0, device=dev), 1)
            elif rank == 1:
                r = torch.empty(4, device=dev)
                dist.recv(r, 0)
                check("send/recv", bool(torch.equal(r.cpu(), torch.arange(4.0))))
        except Exception as e:
            check("send/recv", False, f"ERROR {str(e)[:200]}")
        try:
            ops = [dist.P2POp(dist.isend, torch.full((2,), float(rank), device=dev), (rank + 1) % world),
                   dist.P2POp(dist.irecv, rr := torch.empty(2, device=dev), (rank - 1) % world)]
            for q in dist.batch_isend_irecv(ops):
                q.wait()
            check("batch_isend_irecv", bool((rr.cpu() == (rank - 1) % world).all()))
        except Exception as e:
            check("batch_isend_irecv", False, f"ERROR {str(e)[:200]}")
        objs = [None] * world
        dist.all_gather_object(objs, {"r": rank})
        check("all_gather_object", objs == [{"r": i} for i in range(world)])
        ol = [rank * 7] if rank == 0 else [None]
        dist.broadcast_object_list(ol, src=0)
        check("broadcast_object_list", ol == [0])
        # async op + wait
        t = torch.ones(1000, device=dev)
        w = dist.all_reduce(t, async_op=True)
        w.wait()
        check("all_reduce async", bool((t.cpu() == world).all()))
        # large tensor
        t = torch.ones(8 * 1024 * 1024, device=dev) * (rank + 1)
        dist.all_reduce(t)
        check("all_reduce 32MB", bool((t.cpu() == sum(range(1, world + 1))).all()))
        # uneven all_gather list (different sizes per rank)
        try:
            outs = [torch.empty(i + 1, device=dev) for i in range(world)]
            dist.all_gather(outs, torch.full((rank + 1,), float(rank), device=dev))
            check("all_gather uneven list", all(bool((o.cpu() == i).all()) for i, o in enumerate(outs)))
        except Exception as e:
            check("all_gather uneven list", False, f"ERROR {str(e)[:200]}")

    elif CASE == "funcol":
        import torch.distributed._functional_collectives as fc
        g = dist.group.WORLD
        x = torch.arange(world * 2 * 3, dtype=torch.float32).reshape(world * 2, 3).to(dev) + rank
        r = fc.all_reduce(x, "sum", g)
        check("fc.all_reduce", close(r, sum(torch.arange(world * 6.0).reshape(world * 2, 3) + i for i in range(world)))[0])
        ag = fc.all_gather_tensor(torch.full((2, 3), float(rank), device=dev), 1, g)
        check("fc.all_gather_tensor dim1", close(ag, torch.cat([torch.full((2, 3), float(i)) for i in range(world)], 1))[0], tuple(ag.shape))
        rs = fc.reduce_scatter_tensor(torch.ones(2, world * 3, device=dev) * (rank + 1), "sum", 1, g)
        check("fc.reduce_scatter dim1", close(rs, torch.ones(2, 3) * sum(range(1, world + 1)))[0], tuple(rs.shape))
        a2a = fc.all_to_all_single(torch.arange(world, dtype=torch.float32, device=dev) + 10 * rank, None, None, g)
        check("fc.all_to_all_single", close(a2a, torch.tensor([10.0 * i + rank for i in range(world)]))[0])
        # autograd collectives
        from torch.distributed.nn import functional as dnf
        w = torch.full((3,), float(rank + 1), device=dev, requires_grad=True)
        y = dnf.all_reduce(w * 2)
        (y * (rank + 1)).sum().backward()
        check("dist.nn all_reduce backward", close(w.grad, torch.full((3,), 2.0 * sum(range(1, world + 1))))[0], w.grad.cpu().tolist())
        w = torch.full((2,), float(rank + 1), device=dev, requires_grad=True)
        outs = dnf.all_gather(w)
        sum((i + 1) * o.sum() for i, o in enumerate(outs)).backward()
        check("dist.nn all_gather backward", close(w.grad, torch.full((2,), float(world * (rank + 1))))[0], w.grad.cpu().tolist())
        # funcol autograd variant
        from torch.distributed._functional_collectives import all_gather_tensor_autograd, reduce_scatter_tensor_autograd
        w = torch.full((2, 2), float(rank + 1), device=dev, requires_grad=True)
        o = all_gather_tensor_autograd(w, 0, g)
        (o * torch.arange(world * 2.0, device=dev).view(-1, 1)).sum().backward()
        exp = torch.arange(world * 2.0).view(world, 2)[rank].view(2, 1).expand(2, 2) * world
        check("all_gather_tensor_autograd bwd", close(w.grad, exp)[0], w.grad.cpu().tolist())
        w = torch.ones(world * 2, device=dev, requires_grad=True)
        o = reduce_scatter_tensor_autograd(w * (rank + 1), "sum", 0, g)
        (o * (rank + 1)).sum().backward()
        exp = torch.cat([torch.full((2,), float((rank + 1) * (j + 1))) for j in range(world)])
        check("reduce_scatter_tensor_autograd bwd", close(w.grad, exp)[0], w.grad.cpu().tolist())
        # funcol under compile
        def f(t):
            return fc.all_reduce(t * 2, "sum", g) + fc.all_gather_tensor(t, 0, g).sum()
        c = torch.compile(f, backend="neuron")
        t = torch.full((4,), float(rank), device=dev)
        check("compiled funcol", close(c(t), f(t))[0])

    elif CASE == "subgroups":
        g01 = dist.new_group([0, 1])
        g23 = dist.new_group([2, 3])
        t = torch.full((2,), float(rank + 1), device=dev)
        dist.all_reduce(t, group=g01 if rank < 2 else g23)
        check("new_group all_reduce", bool((t.cpu() == (3 if rank < 2 else 7)).all()), t.cpu().tolist())
        odd = dist.new_group([1, 3])
        if rank in (1, 3):
            t = torch.full((2,), float(rank), device=dev)
            dist.all_reduce(t, group=odd)
            check("strided group all_reduce", bool((t.cpu() == 4).all()), t.cpu().tolist())
            out = torch.empty(4, device=dev)
            dist.all_gather_into_tensor(out, torch.full((2,), float(rank), device=dev), group=odd)
            check("strided group all_gather", out.cpu().tolist() == [1, 1, 3, 3], out.cpu().tolist())
            b = torch.full((2,), float(rank), device=dev)
            dist.broadcast(b, src=3, group=odd)
            check("strided group broadcast global src", bool((b.cpu() == 3).all()), b.cpu().tolist())
        sub, _ = dist.new_subgroups(2)
        t = torch.full((1,), float(rank), device=dev)
        dist.all_reduce(t, group=sub)
        check("new_subgroups", float(t.item()) == (1.0 if rank < 2 else 5.0), t.item())
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp"))
        t = torch.full((1,), float(rank), device=dev)
        dist.all_reduce(t, group=mesh["tp"].get_group())
        t2 = torch.full((1,), float(rank), device=dev)
        dist.all_reduce(t2, group=mesh["dp"].get_group())
        exp_tp = {0: 1, 1: 1, 2: 5, 3: 5}[rank]
        exp_dp = {0: 2, 2: 2, 1: 4, 3: 4}[rank]
        check("mesh tp/dp groups", (t.item(), t2.item()) == (exp_tp, exp_dp), (t.item(), t2.item()))
        # two groups interleaved collectives
        for i in range(5):
            a = torch.ones(16, device=dev)
            b = torch.ones(16, device=dev)
            dist.all_reduce(a, group=mesh["tp"].get_group())
            dist.all_reduce(b, group=mesh["dp"].get_group())
        check("interleaved groups", bool((a.cpu() == 2).all() and (b.cpu() == 2).all()))

    elif CASE in ("tp_mlp", "tp_mlp_foreach", "tp_mlp_clip"):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        mesh = init_device_mesh("neuron", (world,))
        base = mlp()
        x, y = gdata()
        clip = 1.0 if CASE == "tp_mlp_clip" else None
        ref = ref_train(base, x, y, clip=clip)
        m = copy.deepcopy(base).to(dev)
        parallelize_module(m, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel(), "4": ColwiseParallel(output_layouts=__import__("torch.distributed.tensor", fromlist=["x"]).Replicate())})
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, foreach=(CASE != "tp_mlp"))
        for _ in range(3):
            opt.zero_grad()
            F.mse_loss(m(x.to(dev)), y.to(dev)).backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
        compare_params(CASE, ref, m)

    elif CASE == "tp_sequence_parallel":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (ColwiseParallel, RowwiseParallel, SequenceParallel,
                                                       parallelize_module)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.norm = nn.LayerNorm(32)
                self.up = nn.Linear(32, 64)
                self.down = nn.Linear(64, 32)

            def forward(self, x):
                return self.down(F.gelu(self.up(self.norm(x))))

        base = Block()
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 8, 32, generator=g)
        y = torch.randn(2, 8, 32, generator=g)
        ref = ref_train(base, x, y)
        mesh = init_device_mesh("neuron", (world,))
        m = copy.deepcopy(base).to(dev)
        parallelize_module(m, mesh, {
            "norm": SequenceParallel(),
            "up": ColwiseParallel(input_layouts=Shard(1)),
            "down": RowwiseParallel(output_layouts=Shard(1)),
        })
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        xs = x.chunk(world, dim=1)[rank].to(dev)
        ys = y.chunk(world, dim=1)[rank].to(dev)
        for _ in range(3):
            opt.zero_grad()
            out = m(xs)
            out = out.to_local() if hasattr(out, "to_local") else out
            (F.mse_loss(out, ys, reduction="sum") / y.numel()).backward()
            opt.step()
        compare_params(CASE, ref, m)

    elif CASE in ("tp_fsdp2", "tp_fsdp2_compile"):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp"))
        base = mlp()
        x, y = gdata(16)
        ref = ref_train(base, x, y)
        m = copy.deepcopy(base).to(dev)
        parallelize_module(m, mesh["tp"], {"0": ColwiseParallel(), "2": RowwiseParallel(),
                                           "4": ColwiseParallel(output_layouts=Replicate())})
        fully_shard(m, mesh=mesh["dp"])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        dp = mesh["dp"].get_local_rank()
        xs, ys = x.chunk(2)[dp].to(dev), y.chunk(2)[dp].to(dev)
        fwd = torch.compile(m, backend="neuron", dynamic=False) if CASE.endswith("compile") else m
        for _ in range(3):
            opt.zero_grad()
            F.mse_loss(fwd(xs), ys).backward()
            opt.step()
        compare_params(CASE, ref, m)

    elif CASE in ("fsdp2_ac", "fsdp2_ac_dropout", "fsdp2_ac_reentrant"):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl,
                                                                                 apply_activation_checkpointing)
        from torch.distributed.fsdp import fully_shard

        class Blk(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.l1, self.l2, self.p = nn.Linear(32, 64), nn.Linear(64, 32), p

            def forward(self, x):
                return x + self.l2(F.dropout(F.gelu(self.l1(x)), self.p, self.training))

        p = 0.3 if CASE == "fsdp2_ac_dropout" else 0.0
        torch.manual_seed(0)
        base = nn.Sequential(Blk(p), Blk(p), nn.Linear(32, 16))
        x, y = gdata(16)
        m_noac = copy.deepcopy(base).to(dev)
        m = copy.deepcopy(base).to(dev)
        import functools
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        impl = CheckpointImpl.REENTRANT if CASE.endswith("reentrant") else CheckpointImpl.NO_REENTRANT
        apply_activation_checkpointing(m, checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=impl),
                                       check_fn=lambda mod: isinstance(mod, Blk))
        for mm in (m, m_noac):
            for b in list(mm)[:2]:
                fully_shard(b)
            fully_shard(mm)
        xs, ys = x.chunk(world)[rank].to(dev), y.chunk(world)[rank].to(dev)
        grads = []
        for mm in (m, m_noac):
            torch.manual_seed(1234)
            F.mse_loss(mm(xs), ys).backward()
            grads.append([p_.grad.full_tensor().cpu() for p_ in mm.parameters()])
        worst = max((a - b).abs().max().item() for a, b in zip(*grads))
        check(CASE + " (AC vs no-AC grads, same seed)", worst < 1e-4, f"maxdiff={worst:.4g}")
        if p == 0:
            ref = copy.deepcopy(base)
            F.mse_loss(ref(x), y).backward()
            worst = max((a - p_.grad).abs().max().item() for a, p_ in zip(grads[0], ref.parameters()))
            check(CASE + " (vs CPU)", worst < 1e-4, f"maxdiff={worst:.4g}")

    elif CASE in ("dcp_save4", "dcp_load2", "dcp_load4_full"):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_state_dict, set_state_dict)
        from torch.distributed.fsdp import fully_shard
        path = "/home/ec2-user/torch-neuronx/agent_space/overnight/dcp_reshard"
        base = mlp(30, 70, 16)  # dims not divisible by 4: uneven FSDP shards
        x, y = gdata(16, 30)

        def build():
            mm = copy.deepcopy(base).to(dev)
            for i in (0, 2, 4):
                fully_shard(mm[i])
            fully_shard(mm)
            return mm, torch.optim.AdamW(mm.parameters(), lr=1e-2)

        def step(mm, opt, xs, ys):
            opt.zero_grad()
            F.mse_loss(mm(xs), ys).backward()
            opt.step()

        if CASE == "dcp_save4":
            mm, opt = build()
            xs, ys = x.chunk(world)[rank].to(dev), y.chunk(world)[rank].to(dev)
            step(mm, opt, xs, ys)
            compare_params("fsdp2 uneven dims 1 step", ref_train(base, x, y, steps=1), mm)
            msd, osd = get_state_dict(mm, opt)
            dcp.save({"model": msd, "optim": osd}, checkpoint_id=path)
            check("dcp save", True)
        elif CASE == "dcp_load2":
            mm, opt = build()
            msd, osd = get_state_dict(mm, opt)
            sd = {"model": msd, "optim": osd}
            dcp.load(sd, checkpoint_id=path)
            set_state_dict(mm, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
            xs, ys = x.chunk(world)[rank].to(dev), y.chunk(world)[rank].to(dev)
            step(mm, opt, xs, ys)
            step(mm, opt, xs, ys)
            compare_params(f"dcp reshard 4->{world} + 2 steps", ref_train(base, x, y, steps=3), mm)
        else:
            # load into a plain (unsharded) neuron model on every rank via full state dict
            mm = copy.deepcopy(base).to(dev)
            opt = torch.optim.AdamW(mm.parameters(), lr=1e-2)
            msd, osd = get_state_dict(mm, opt, options=StateDictOptions(full_state_dict=False))
            sd = {"model": msd, "optim": osd}
            dcp.load(sd, checkpoint_id=path)
            set_state_dict(mm, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
            for _ in range(2):
                opt.zero_grad()
                F.mse_loss(mm(x.to(dev)), y.to(dev)).backward()
                opt.step()
            compare_params("dcp load into unsharded model + 2 steps", ref_train(base, x, y, steps=3), mm)

    elif CASE == "ddp_uneven_join":
        from torch.distributed.algorithms.join import Join
        from torch.nn.parallel import DistributedDataParallel as DDP
        m = DDP(mlp().to(dev))
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        nb = 2 + rank  # uneven number of batches per rank
        with Join([m]):
            for i in range(nb):
                x, y = gdata(4, seed=10 * rank + i)
                opt.zero_grad()
                F.mse_loss(m(x.to(dev)), y.to(dev)).backward()
                opt.step()
        p = torch.cat([q.detach().flatten() for q in m.parameters()])
        allp = [torch.empty_like(p) for _ in range(world)]
        dist.all_gather(allp, p)
        check("ddp join uneven inputs (params equal across ranks)", all(torch.equal(allp[0].cpu(), a.cpu()) for a in allp))

    elif CASE == "ddp_fp16_hook_bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        from torch.nn.parallel import DistributedDataParallel as DDP
        base = mlp()
        x, y = gdata(16)
        for hook in ["fp16", "allreduce"]:
            m = DDP(copy.deepcopy(base).to(dev))
            if hook == "bf16":
                m.register_comm_hook(None, default_hooks.bf16_compress_hook)
            elif hook == "fp16":
                m.register_comm_hook(None, default_hooks.fp16_compress_hook)
            opt = torch.optim.SGD(m.parameters(), lr=0.1)
            for _ in range(3):
                opt.zero_grad()
                F.mse_loss(m(x.chunk(world)[rank].to(dev)), y.chunk(world)[rank].to(dev)).backward()
                opt.step()
            ref = copy.deepcopy(base)
            o2 = torch.optim.SGD(ref.parameters(), lr=0.1)
            for _ in range(3):
                o2.zero_grad()
                F.mse_loss(ref(x), y).backward()
                o2.step()
            compare_params(f"ddp comm hook {hook}", ref, m.module, atol=1e-2 if hook != "allreduce" else 2e-4)

    elif CASE == "fsdp2_misc":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        base = mlp()
        x, y = gdata(16)
        xs, ys = x.chunk(world)[rank].to(dev), y.chunk(world)[rank].to(dev)
        # reshard_after_forward=False, grad accumulation with set_requires_gradient_sync
        m = copy.deepcopy(base).to(dev)
        for i in (0, 2, 4):
            fully_shard(m[i], reshard_after_forward=False)
        fully_shard(m)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        ref = copy.deepcopy(base)
        ro = torch.optim.AdamW(ref.parameters(), lr=1e-2)
        for step in range(2):
            opt.zero_grad()
            ro.zero_grad()
            for k in range(2):
                m.set_requires_gradient_sync(k == 1)
                xk, yk = gdata(16, seed=step * 10 + k)
                F.mse_loss(m(xk.chunk(world)[rank].to(dev)), yk.chunk(world)[rank].to(dev)).backward()
                F.mse_loss(ref(xk), yk).backward()
            opt.step()
            ro.step()
        compare_params("fsdp2 grad accum no-sync + reshard_after_forward=False", ref, m)
        # fp32 reduce with bf16 params
        m = copy.deepcopy(base).to(dev)
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for i in (0, 2, 4):
            fully_shard(m[i], mp_policy=mp)
        fully_shard(m, mp_policy=mp)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            F.mse_loss(m(xs).float(), ys).backward()
            opt.step()
        compare_params("fsdp2 bf16 params fp32 reduce", ref_train(base, x, y), m, atol=3e-2)
        # full_state_dict gather on rank0 then load to cpu model
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        sd = get_model_state_dict(m, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            cpu_m = copy.deepcopy(base)
            cpu_m.load_state_dict(sd)
            check("fsdp2 full_state_dict cpu_offload", all(v.device.type == "cpu" for v in sd.values()))


try:
    run()
except Exception as e:
    traceback.print_exc()
    msgs.append(f"ERROR    {CASE}: {type(e).__name__}: {str(e)[:300]}".replace("\n", " "))
gathered = [None] * world
dist.all_gather_object(gathered, msgs)
if rank == 0:
    for r, ms in enumerate(gathered):
        for mline in ms:
            print(f"RESULT [{CASE}] rank{r} {mline}", flush=True)
dist.destroy_process_group()
