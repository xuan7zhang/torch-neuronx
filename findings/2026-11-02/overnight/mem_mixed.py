"""Memory growth over training steps, and mixed CPU/neuron tensor patterns common in training code."""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

dev = "neuron"
mode = sys.argv[1]


def mem():
    torch.neuron.synchronize()
    return torch.neuron.memory_allocated() / 2**20


if mode == "mem":
    from torch.utils.checkpoint import checkpoint
    for variant in ("eager", "eager_ckpt", "compile", "eager_sdpa_dropout", "eager_scaler"):
        torch._dynamo.reset()
        torch.manual_seed(0)
        m = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1 if "dropout" in variant else 0.0,
                                                             batch_first=True), 2).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sc = torch.amp.GradScaler(dev) if variant == "eager_scaler" else None
        fwd = torch.compile(m, backend="neuron", dynamic=False) if variant == "compile" else m
        x = torch.randn(8, 32, 128).to(dev)
        marks = []
        for s in range(60):
            opt.zero_grad(set_to_none=True)
            if variant == "eager_ckpt":
                y = checkpoint(fwd, x, use_reentrant=False)
            else:
                y = fwd(x)
            loss = y.pow(2).mean()
            if sc:
                sc.scale(loss).backward()
                sc.step(opt)
                sc.update()
            else:
                loss.backward()
                opt.step()
            if s in (5, 20, 40, 59):
                marks.append(round(mem(), 2))
        print(f"RES mem {variant:20s} MiB allocated at steps 5/20/40/59: {marks}  max_allocated={torch.neuron.max_memory_allocated() / 2**20:.1f}", flush=True)
        del m, opt, fwd, x, y, loss

elif mode == "mixed":
    t = torch.randn(4, 6).to(dev)
    cases = {
        "neuron * cpu 0-d tensor": lambda: t * torch.tensor(2.0),
        "cpu 0-d tensor + neuron": lambda: torch.tensor(1.0) + t,
        "neuron[cpu long index]": lambda: t[torch.tensor([0, 2])],
        "neuron[cpu bool mask]": lambda: t[torch.tensor([True, False, True, False])],
        "neuron.index_select(cpu idx)": lambda: t.index_select(0, torch.tensor([1, 3])),
        "neuron * cpu (4,6) tensor (expect error)": lambda: t * torch.ones(4, 6),
        "torch.where(cpu cond, neuron, 0)": lambda: torch.where(torch.ones(4, 6, dtype=torch.bool), t, 0.0),
        "F.cross_entropy(neuron, cpu target) (expect error)": lambda: F.cross_entropy(t, torch.tensor([0, 1, 2, 3])),
        "neuron.masked_fill(cpu 0-d bool)": lambda: t.masked_fill(torch.tensor(True), 0.0),
        "clip_grad_norm_ returns device tensor": lambda: torch.nn.utils.clip_grad_norm_([nn.Parameter(t.clone())], 1.0),
        "neuron.copy_(cpu)": lambda: t.clone().copy_(torch.ones(4, 6)),
        "cpu.copy_(neuron)": lambda: torch.zeros(4, 6).copy_(t),
        "torch.stack([neuron, cpu]) (expect error)": lambda: torch.stack([t, torch.ones(4, 6)]),
        "neuron.add_(cpu 0-d)": lambda: t.clone().add_(torch.tensor(3.0)),
        "lr tensor on cpu in AdamW step on neuron": None,
    }
    for nm, f in cases.items():
        if f is None:
            p = nn.Parameter(t.clone())
            p.grad = torch.ones_like(p)
            try:
                torch.optim.AdamW([p], lr=torch.tensor(0.01)).step()
                torch.neuron.synchronize()
                print("RES", f"{nm:48s}", "ok")
            except Exception as e:
                print("RES", f"{nm:48s}", "ERR", str(e)[:120].replace("\n", " "))
            continue
        try:
            r = f()
            torch.neuron.synchronize()
            # compare with CPU semantics
            print("RES", f"{nm:48s}", "ok", r.device, tuple(r.shape) if isinstance(r, torch.Tensor) else r)
        except Exception as e:
            print("RES", f"{nm:48s}", "ERR", str(e)[:120].replace("\n", " "))
