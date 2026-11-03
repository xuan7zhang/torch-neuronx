"""MoE expert parallel end to end vs a single-process CPU reference.
torchrun --nproc-per-node 4 moe_ep.py <steps>. One expert per rank, top-1 routing, uneven all_to_all with autograd."""
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn as nn
import torch.nn.functional as F

import torch_neuronx  # noqa: F401

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
dist.init_process_group("cpu:gloo,neuron:neuron")
STEPS = int(sys.argv[1])
D, T, E = 32, 64, W  # hidden, tokens per rank, experts


def make():
    torch.manual_seed(0)
    router = nn.Linear(D, E, bias=False)
    experts = nn.ModuleList([nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, D)) for _ in range(E)])
    head = nn.Linear(D, 1)
    return router, experts, head


def batch(s, r):
    gg = torch.Generator().manual_seed(s * 100 + r)
    x = torch.randn(T, D, generator=gg)
    y = (x[:, :4].sum(1, keepdim=True)).tanh()
    return x, y


def moe_ref(router, experts, x):
    logits = router(x)
    p = logits.softmax(-1)
    w, e = p.max(-1)
    out = torch.zeros_like(x)
    for j in range(E):
        sel = (e == j).nonzero().flatten()
        if sel.numel():
            out = out.index_add(0, sel, experts[j](x[sel]) * w[sel, None])
    aux = E * (p.mean(0) * F.one_hot(e, E).float().mean(0)).sum()
    return out, aux


# CPU reference over the global batch
r_router, r_experts, r_head = make()
rparams = list(r_router.parameters()) + list(r_experts.parameters()) + list(r_head.parameters())
ro = torch.optim.AdamW(rparams, lr=3e-3)
ref_losses = []
for s in range(STEPS):
    ro.zero_grad()
    tot = 0.0
    for q in range(W):
        x, y = batch(s, q)
        h, aux = moe_ref(r_router, r_experts, x)
        l = (F.mse_loss(r_head(x + h), y) + 0.01 * aux) / W
        l.backward(); tot += l.item()
    ro.step(); ref_losses.append(tot)

# Neuron EP: router/head replicated (grads all-reduced), expert `rank` local
router, experts, head = make()
router, head = router.to(dev), head.to(dev)
expert = experts[rank].to(dev)
rep = list(router.parameters()) + list(head.parameters())
o = torch.optim.AdamW(rep + list(expert.parameters()), lr=3e-3)
losses = []
for s in range(STEPS):
    o.zero_grad()
    x, y = batch(s, rank)
    x, y = x.to(dev), y.to(dev)
    p = router(x).softmax(-1)
    w, e = p.max(-1)
    order = e.argsort(stable=True)
    counts = torch.bincount(e, minlength=E)
    send = counts.cpu().tolist()
    allc = torch.empty(W * E, dtype=counts.dtype, device=dev)
    dist.all_gather_into_tensor(allc, counts)
    recv = allc.view(W, E)[:, rank].cpu().tolist()
    xs = x[order]
    xin = fc.all_to_all_single_autograd(xs, recv, send, dist.group.WORLD)
    yout = expert(xin)
    back = fc.all_to_all_single_autograd(yout, send, recv, dist.group.WORLD)
    h = torch.zeros_like(x).index_copy(0, order, back) * w[:, None]
    aux = E * (p.mean(0) * F.one_hot(e, E).float().mean(0)).sum()
    # global aux uses global means: average the per-rank means
    pm = p.mean(0); fm = F.one_hot(e, E).float().mean(0)
    pm_g = fc.all_reduce(pm, "avg", dist.group.WORLD) if False else pm
    l = F.mse_loss(head(x + h), y) + 0.01 * aux
    (l / W).backward()
    for prm in rep:
        dist.all_reduce(prm.grad)
    # expert grads: each expert sees tokens from all ranks already (sum over ranks via a2a); loss was /W per rank
    o.step()
    lt = l.detach().clone() / W; dist.all_reduce(lt); losses.append(lt.item())
torch.neuron.synchronize()
if rank == 0:
    d = [abs(a - b) for a, b in zip(ref_losses, losses)]
    first = next((i for i, v in enumerate(d) if v > 1e-3), -1)
    print(f"RES steps {STEPS} first|dl|>1e-3 at {first}, max |dl| {max(d):.3g}, loss[0] {losses[0]:.5f}/{ref_losses[0]:.5f}, final {losses[-1]:.5f}/{ref_losses[-1]:.5f}")
ew = torch.cat([q.detach().flatten() for q in expert.parameters()]).cpu()
rw = torch.cat([q.detach().flatten() for q in r_experts[rank].parameters()])
print(f"RES R{rank} expert param rel {((ew - rw).norm() / rw.norm()).item():.3g}")
dist.destroy_process_group()
