"""Real-scale distributed train run vs CPU.
torchrun --nproc-per-node 4 llama_scale.py <mode> <steps>
mode: tp_fsdp (2x2), fsdp (4), tp (4), ddp (4); add suffix _bf16 for autocast-free bf16 MixedPrecisionPolicy (FSDP)
or _compile for torch.compile per block. CPU reference: run with DEVN=cpu WORLD=1 (single process, global batch),
written by ref mode: python llama_scale.py ref <steps> (no torchrun)."""
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVN = os.environ.get("DEVN", "neuron")
if DEVN == "neuron":
    import torch_neuronx  # noqa: F401

MODE = sys.argv[1]
STEPS = int(sys.argv[2])
V, D, L, H, KV, FF, S, B = 2048, 512, 8, 8, 4, 1408, int(os.environ.get("SEQ", "128")), 4  # B per data-parallel rank
OUT = os.path.dirname(os.path.abspath(__file__))


def rope(x, base=10000.0):
    b, h, s, d = x.shape
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    t = torch.arange(s, device=x.device, dtype=torch.float32)
    f = torch.outer(t, inv)
    cos, sin = f.cos()[None, None].to(x.dtype), f.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D, D, bias=False); self.wk = nn.Linear(D, KV * D // H, bias=False)
        self.wv = nn.Linear(D, KV * D // H, bias=False); self.wo = nn.Linear(D, D, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        hd = D // H
        q = self.wq(x).view(b, s, -1, hd).transpose(1, 2)
        k = self.wk(x).view(b, s, -1, hd).transpose(1, 2)
        v = self.wv(x).view(b, s, -1, hd).transpose(1, 2)
        q, k = rope(q), rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])
        return self.wo(o.transpose(1, 2).reshape(b, s, -1))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.RMSNorm(D); self.attn = Attn(); self.n2 = nn.RMSNorm(D)
        self.w1 = nn.Linear(D, FF, bias=False); self.w3 = nn.Linear(D, FF, bias=False); self.w2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.w2(F.silu(self.w1(self.n2(x))) * self.w3(self.n2(x)))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Block() for _ in range(L)])
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02 / (math.sqrt(2 * L) if n.endswith(("wo.weight", "w2.weight")) else 1))

    def forward(self, ids):
        h = self.emb(ids)
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h))


def data(step, dp_rank, dp_size):
    # learnable synthetic text: next token = (3 * tok + 7) % V with 20% noise
    g = torch.Generator().manual_seed(step * 1000 + dp_rank)
    ids = torch.empty(B, S + 1, dtype=torch.long)
    ids[:, 0] = torch.randint(0, V, (B,), generator=g)
    for t in range(1, S + 1):
        nxt = (3 * ids[:, t - 1] + 7) % V
        noise = torch.randint(0, V, (B,), generator=g)
        ids[:, t] = torch.where(torch.rand(B, generator=g) < 0.2, noise, nxt)
    return ids[:, :-1], ids[:, 1:]


def lr_at(s):
    warm = 20
    return 3e-3 * min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / STEPS))))


def run_ref():
    dp = int(os.environ.get("REF_DP", "4"))
    m = LM()
    o = torch.optim.AdamW(m.parameters(), lr=lr_at(0), weight_decay=0.1)
    losses = []
    t0 = time.time()
    for s in range(STEPS):
        for gp in o.param_groups:
            gp["lr"] = lr_at(s)
        o.zero_grad()
        tot = 0.0
        for r in range(dp):
            x, y = data(s, r, dp)
            l = F.cross_entropy(m(x).flatten(0, 1), y.flatten()) / dp
            l.backward(); tot += l.item()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        losses.append(tot)
        if s % 25 == 0:
            print(f"ref step {s} loss {tot:.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save({"losses": losses, "params": {k: v.detach() for k, v in m.state_dict().items()}},
               os.path.join(OUT, f"ref_dp{dp}_{STEPS}" + (f"_s{S}" if S != 128 else "") + ".pt"))


def run_dist():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
    torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
    dist.init_process_group("cpu:gloo,neuron:neuron")
    base = MODE.split("_")[0] if not MODE.startswith("tp_fsdp") else "tp_fsdp"
    bf16 = "_bf16" in MODE
    comp = "_compile" in MODE
    if base == "tp_fsdp":
        mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp")); dp_mesh, tp_mesh = mesh["dp"], mesh["tp"]
    elif base == "tp":
        tp_mesh = init_device_mesh("neuron", (W,)); dp_mesh = None
    else:
        dp_mesh = init_device_mesh("neuron", (W,)); tp_mesh = None
    dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else 0
    dp_size = dp_mesh.size() if dp_mesh is not None else 1
    m = LM().to(dev)
    if tp_mesh is not None:
        parallelize_module(m, tp_mesh, {"emb": RowwiseParallel(input_layouts=Replicate()),
                                        "head": ColwiseParallel(output_layouts=Replicate())})
        for blk in m.layers:
            parallelize_module(blk, tp_mesh, {"attn.wq": ColwiseParallel(), "attn.wk": ColwiseParallel(),
                                              "attn.wv": ColwiseParallel(), "attn.wo": RowwiseParallel(),
                                              "w1": ColwiseParallel(), "w3": ColwiseParallel(), "w2": RowwiseParallel()})
    if comp:
        for blk in m.layers:
            blk.compile(backend="neuron", dynamic=False)
    if base == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(m)
    elif dp_mesh is not None:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if bf16 else MixedPrecisionPolicy()
        for blk in m.layers:
            fully_shard(blk, mesh=dp_mesh, mp_policy=mp)
        fully_shard(m, mesh=dp_mesh, mp_policy=mp)
        model = m
    else:
        model = m
    o = torch.optim.AdamW(m.parameters(), lr=lr_at(0), weight_decay=0.1, foreach=(base != "tp_fsdp"))
    losses = []
    mem = []
    t0 = time.time()
    for s in range(STEPS):
        for gp in o.param_groups:
            gp["lr"] = lr_at(s)
        o.zero_grad()
        x, y = data(s, dp_rank, dp_size)
        logits = model(x.to(dev))
        l = F.cross_entropy(logits.float().flatten(0, 1), y.to(dev).flatten())
        l.backward()
        if base == "tp_fsdp":
            # Params live on two meshes, so clip_grad_norm_ cannot stack the norms. Use full grads.
            sq = sum(p.grad.full_tensor().float().pow(2).sum() for p in m.parameters())
            coef = torch.clamp(1.0 / (sq.sqrt() + 1e-6), max=1.0)
            for p in m.parameters():
                p.grad.mul_(coef)
        else:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        lt = l.detach().clone()
        if dp_mesh is not None:
            dist.all_reduce(lt, group=dp_mesh.get_group()); lt /= dp_size
        losses.append(lt.item())
        if s % 25 == 0:
            mem.append(round(torch.neuron.memory_allocated() / 2**20, 1))
            if rank == 0:
                print(f"{MODE} step {s} loss {losses[-1]:.4f} mem {mem[-1]}MiB {time.time() - t0:.0f}s", flush=True)
    sd = {}
    for k, v in m.state_dict().items():
        v = v.full_tensor() if hasattr(v, "full_tensor") else v
        sd[k] = v.detach().float().cpu()
    if rank == 0:
        torch.save({"losses": losses, "params": sd, "mem": mem, "secs": time.time() - t0},
                   os.path.join(OUT, f"run_{MODE}_{STEPS}" + (f"_s{S}" if S != 128 else "") + ".pt"))
        print(f"{MODE} done {time.time() - t0:.0f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    run_ref() if MODE == "ref" else run_dist()
