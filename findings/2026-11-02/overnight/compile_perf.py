"""Compile perf probes: recompiles from float LR schedulers, cold vs warm compile time."""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
from torch._dynamo.utils import counters

mode = sys.argv[1]
dev = "neuron"


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 10)).to(dev)


x = torch.randn(32, 64).to(dev)
y = torch.randint(0, 10, (32,)).to(dev)

if mode in ("float_lr_sched", "tensor_lr_sched"):
    m = mlp()
    lr = 0.01 if mode == "float_lr_sched" else torch.tensor(0.01)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 / (1 + e))

    @torch.compile(backend="neuron", dynamic=False)
    def ostep():
        opt.step()

    counters.clear()
    times = []
    for i in range(6):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        t = time.time()
        ostep()
        torch.neuron.synchronize()
        times.append(round(time.time() - t, 2))
        sch.step()
    print("RES", mode, "step times (s):", times, "unique_graphs:", counters["stats"]["unique_graphs"],
          "recompile reasons:", len(counters.get("recompiles", {})))

elif mode == "gpt2_compile_time":
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=4, n_head=8, n_embd=512, vocab_size=8000, n_positions=256,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    m = GPT2LMHeadModel(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ids = torch.randint(0, 8000, (4, 256)).to(dev)
    cm = torch.compile(m, backend="neuron", dynamic=False)
    ts = []
    for i in range(3):
        t = time.time()
        out = cm(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        torch.neuron.synchronize()
        ts.append(round(time.time() - t, 2))
    # eager reference time
    te = []
    for i in range(3):
        t = time.time()
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        torch.neuron.synchronize()
        te.append(round(time.time() - t, 2))
    print("RES gpt2 compiled step times (s):", ts, " eager step times (s):", te)
