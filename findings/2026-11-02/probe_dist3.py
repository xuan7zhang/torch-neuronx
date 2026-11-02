"""Distributed probes round 3. Run: torchrun --nproc-per-node 4 probe_dist3.py <case>"""

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
CASE = sys.argv[1]
dist.init_process_group(backend="cpu:gloo,neuron:neuron" if "offload" in CASE else "neuron")
msgs = []


def check(name, ok, info=""):
    msgs.append(f"{'OK      ' if ok else 'MISMATCH'} {name} {info}")


def full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def maxdiff(ref_m, m):
    return max((p1.detach().float() - full(p2).detach().float().cpu()).abs().max().item()
               for p1, p2 in zip(ref_m.parameters(), m.parameters()))


class Attn(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.h = h
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x):
        B, S, D = x.shape
        hl = self.wq(x).shape[-1] // (D // self.h)  # local heads under TP
        q = self.wq(x).view(B, S, hl, -1).transpose(1, 2)
        k = self.wk(x).view(B, S, hl, -1).transpose(1, 2)
        v = self.wv(x).view(B, S, hl, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, S, -1))


class Block(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attn(d)
        self.w1, self.w2, self.w3 = nn.Linear(d, 128, bias=False), nn.Linear(128, d, bias=False), nn.Linear(d, 128, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class LM(nn.Module):
    def __init__(self, V=128, d=64):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d), Block(d)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def data():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 128, (4, 17), generator=g)


def ref_lm(steps=3):
    m = LM()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ids = data()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits = m(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 128), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return m, losses


def run():
    if CASE in ("tp_llama_plan", "tp_llama_loss_parallel"):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (ColwiseParallel, RowwiseParallel, loss_parallel,
                                                       parallelize_module)
        mesh = init_device_mesh("neuron", (world,))
        ref, ref_losses = ref_lm()
        m = LM().to(dev)
        lp = CASE.endswith("loss_parallel")
        plan = {"emb": RowwiseParallel(input_layouts=Replicate()),
                "head": ColwiseParallel(output_layouts=Shard(-1) if lp else Replicate(), use_local_output=not lp)}
        for i in range(2):
            for n in ("attn.wq", "attn.wk", "attn.wv", "w1", "w3"):
                plan[f"blocks.{i}.{n}"] = ColwiseParallel()
            for n in ("attn.wo", "w2"):
                plan[f"blocks.{i}.{n}"] = RowwiseParallel()
        parallelize_module(m, mesh, plan)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, foreach=False)
        ids = data().to(dev)
        losses = []
        for _ in range(3):
            opt.zero_grad()
            logits = m(ids[:, :-1])
            if lp:
                with loss_parallel():
                    loss = F.cross_entropy(logits.reshape(-1, 128), ids[:, 1:].reshape(-1))
                    loss.backward()
            else:
                logits = logits.to_local() if hasattr(logits, "to_local") else logits
                # avoid the known AsyncCollectiveTensor CE issue: use log_softmax + nll
                loss = F.nll_loss(F.log_softmax(logits.reshape(-1, 128), -1), ids[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            losses.append(full(loss).item())
        d = maxdiff(ref, m)
        check(CASE, d < 2e-4 and max(abs(a - b) for a, b in zip(losses, ref_losses)) < 1e-4,
              f"param maxdiff={d:.3g} losses={[round(v, 5) for v in losses]} ref={[round(v, 5) for v in ref_losses]}")

    elif CASE == "fsdp2_cpu_offload":
        from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
        ref, ref_losses = ref_lm()
        m = LM().to(dev)
        for b in m.blocks:
            fully_shard(b, offload_policy=CPUOffloadPolicy())
        fully_shard(m, offload_policy=CPUOffloadPolicy())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        ids = data()
        losses = []
        for _ in range(3):
            opt.zero_grad()
            logits = m(ids[:, :-1].to(dev))
            loss = F.cross_entropy(logits.reshape(-1, 128), ids[:, 1:].reshape(-1).to(dev))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        pdev = {p.to_local().device.type for p in m.parameters()}
        d = maxdiff(ref, m)
        check(CASE, d < 2e-4, f"param maxdiff={d:.3g} param devices={pdev} losses={[round(v, 5) for v in losses]} ref={[round(v, 5) for v in ref_losses]}")

    elif CASE in ("fsdp2_llama", "fsdp2_llama_compile"):
        from torch.distributed.fsdp import fully_shard
        ref, ref_losses = ref_lm()
        m = LM().to(dev)
        for b in m.blocks:
            fully_shard(b)
        fully_shard(m)
        fwd = torch.compile(m, backend="neuron", dynamic=False) if CASE.endswith("compile") else m
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        ids = data()
        losses = []
        for _ in range(3):
            opt.zero_grad()
            logits = fwd(ids[:, :-1].to(dev))
            loss = F.nll_loss(F.log_softmax(logits.reshape(-1, 128).float(), -1), ids[:, 1:].reshape(-1).to(dev))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        d = maxdiff(ref, m)
        check(CASE, d < 2e-4, f"param maxdiff={d:.3g} losses={[round(v, 5) for v in losses]} ref={[round(v, 5) for v in ref_losses]}")

    elif CASE == "dtensor_ops":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor
        mesh = init_device_mesh("neuron", (world,))
        g = torch.Generator().manual_seed(0)
        a = torch.randn(8, 12, generator=g)
        b = torch.randn(12, 6, generator=g)
        da = distribute_tensor(a.to(dev), mesh, [Shard(0)])
        db = distribute_tensor(b.to(dev), mesh, [Replicate()])
        check("dtensor matmul shard0 x rep", torch.allclose(full(da @ db).cpu(), a @ b, atol=1e-4))
        da1 = distribute_tensor(a.to(dev), mesh, [Shard(1)])
        db0 = distribute_tensor(b.to(dev), mesh, [Shard(0)])
        r = da1 @ db0
        check("dtensor matmul shard1 x shard0 (partial)", torch.allclose(full(r).cpu(), a @ b, atol=1e-4), str(r.placements))
        for src, dst in [([Shard(0)], [Shard(1)]), ([Shard(1)], [Replicate()]), ([Replicate()], [Shard(0)])]:
            x = distribute_tensor(a.to(dev), mesh, src).redistribute(mesh, dst)
            check(f"redistribute {src}->{dst}", torch.allclose(full(x).cpu(), a))
        p = distribute_tensor(a.to(dev), mesh, [Replicate()]).redistribute(mesh, [Replicate()])
        x = distribute_tensor(a.to(dev), mesh, [Shard(0)])
        check("dtensor softmax/sum/mean", torch.allclose(full(x.softmax(-1)).cpu(), a.softmax(-1), atol=1e-5)
              and abs(full(x.sum()).item() - a.sum().item()) < 1e-3 and abs(full(x.mean()).item() - a.mean().item()) < 1e-5)
        # uneven shard
        u = torch.randn(10, 3, generator=g)
        du = distribute_tensor(u.to(dev), mesh, [Shard(0)])
        check("uneven shard full_tensor", torch.allclose(full(du).cpu(), u), str([du.to_local().shape[0]]))
        check("uneven shard sum", abs(full(du.sum()).item() - u.sum().item()) < 1e-4)
        # dtensor autograd
        w = distribute_tensor(b.to(dev), mesh, [Shard(1)]).requires_grad_()
        y = (distribute_tensor(a.to(dev), mesh, [Replicate()]) @ w).sum()
        y.backward()
        bw = b.clone().requires_grad_()
        (a @ bw).sum().backward()
        check("dtensor grad shard1", torch.allclose(full(w.grad).cpu(), bw.grad, atol=1e-4))

    elif CASE == "ddp_sdpa_dropout_compile_vs_eager":
        # DDP model with attention dropout: compiled grads must be self-consistent (mask used in fwd and bwd)
        from torch.nn.parallel import DistributedDataParallel as DDP
        torch.manual_seed(0)
        S = 16
        lin = nn.Linear(S, S, bias=False).to(dev)
        m = DDP(lin)

        def f(x):
            q = m(x)
            v = torch.eye(S, device=dev).expand_as(q).contiguous().requires_grad_()
            o = F.scaled_dot_product_attention(q, q, v, dropout_p=0.3)
            return o, v

        for mode in ("eager", "compile"):
            fn = torch.compile(f, backend="neuron", dynamic=False) if mode == "compile" else f
            x = torch.randn(2, 2, S, S).to(dev)
            o, v = fn(x)
            G = torch.randn_like(o)
            gv, = torch.autograd.grad((o * G).sum(), v, retain_graph=True)
            err = (gv - o.detach().transpose(-1, -2) @ G).abs().max().item()
            check(f"ddp sdpa dropout {mode}", err < 1e-4, f"v.grad err={err:.3g}")


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
