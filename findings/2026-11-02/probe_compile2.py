"""torch.compile training probes, round 2. CPU eager fp32 is the reference."""

import copy
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

DEV = "neuron"
CASES = {}


def case(name, atol=2e-3, rtol=2e-3):
    def deco(fn):
        CASES[name] = (fn, atol, rtol)
        return fn
    return deco


def mlp(d=32):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 10))


def data(n=16, d=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, 10, (n,), generator=g)


def cc(fn, dev, **kw):
    if dev == "cpu":
        return fn
    return torch.compile(fn, backend="neuron", **kw)


def train(dev, opt_fn, steps=4, compile_step=True, model_fn=mlp, ckw=None):
    m = model_fn().to(dev)
    opt = opt_fn(m.parameters())
    x, y = data()
    x, y = x.to(dev), y.to(dev)

    def step(x, y):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        return loss.detach()

    fn = cc(step, dev, **(ckw or {})) if compile_step else step
    losses = [fn(x, y) for _ in range(steps)]
    return [torch.stack(losses)] + [p.detach() for p in m.parameters()]


# ---- compiled optimizer step for many optimizers (whole train step compiled) ----
for oname, ocls, kw in [
    ("sgd_mom", torch.optim.SGD, dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01)),
    ("adam", torch.optim.Adam, dict(lr=1e-2)),
    ("adam_capturable", torch.optim.Adam, dict(lr=1e-2, capturable=True)),
    ("adam_tensor_lr", torch.optim.Adam, dict(lr=torch.tensor(1e-2), capturable=True)),
    ("adamw_amsgrad", torch.optim.AdamW, dict(lr=1e-2, amsgrad=True)),
    ("rmsprop", torch.optim.RMSprop, dict(lr=1e-3, momentum=0.9, centered=True)),
    ("adagrad", torch.optim.Adagrad, dict(lr=1e-2)),
    ("adamax", torch.optim.Adamax, dict(lr=1e-2)),
    ("nadam", torch.optim.NAdam, dict(lr=1e-2, decoupled_weight_decay=True, weight_decay=0.1)),
    ("radam", torch.optim.RAdam, dict(lr=1e-2)),
    ("adadelta", torch.optim.Adadelta, dict(lr=1.0)),
    ("asgd", torch.optim.ASGD, dict(lr=1e-2)),
    ("rprop", torch.optim.Rprop, dict(lr=1e-2)),
    ("adafactor", torch.optim.Adafactor, dict(lr=1e-2)),
]:
    def mk(dev, ocls=ocls, kw=kw):
        kw2 = dict(kw)
        if isinstance(kw2.get("lr"), torch.Tensor):
            kw2["lr"] = kw2["lr"].clone().to(dev)
        return train(dev, lambda p: ocls(p, **kw2))
    case(f"compiled_step/{oname}")(mk)


@case("compile_opt_step_only_with_scheduler")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(0.01, device=dev) if dev != "cpu" else torch.tensor(0.01), capturable=dev != "cpu")
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    x, y = (t.to(dev) for t in data())
    step = cc(opt.step, dev)
    out = []
    for _ in range(5):
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        step()
        sch.step()
        out.append(loss.detach())
    return [torch.stack(out)] + [p.detach() for p in m.parameters()]


# ---- compile modes / options ----
for nm, ckw in [("fullgraph", dict(fullgraph=True)), ("max_autotune", dict(mode="max-autotune")),
                ("reduce_overhead", dict(mode="reduce-overhead")), ("dynamic_true", dict(dynamic=True)),
                ("max_autotune_no_cudagraphs", dict(mode="max-autotune-no-cudagraphs"))]:
    case(f"mode/{nm}")(lambda dev, ckw=ckw: train(dev, lambda p: torch.optim.AdamW(p, lr=1e-2), ckw=ckw))


# ---- dynamic shapes in training: count recompiles ----
@case("dynamic_seq_lengths_recompiles")
def _(dev):
    torch.manual_seed(0)
    emb = nn.Embedding(50, 16)
    enc = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True)
    m = nn.Sequential(emb, enc).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    counters.clear()
    dynamo.reset()

    def step(ids):
        opt.zero_grad()
        loss = m(ids).pow(2).mean()
        loss.backward()
        opt.step()
        return loss.detach()

    fn = cc(step, dev, dynamic=None)
    losses = []
    g = torch.Generator().manual_seed(0)
    for L in [8, 12, 16, 20, 24, 12, 8]:
        ids = torch.randint(0, 50, (2, L), generator=g).to(dev)
        losses.append(fn(ids))
    frames = counters["stats"]["unique_graphs"] if dev != "cpu" else 0
    print(f"  [{dev}] unique_graphs={frames} recompiles={dict(counters['recompiles']) if 'recompiles' in counters else {}}", flush=True)
    return [torch.stack(losses)] + [p.detach() for p in m.parameters()]


@case("mark_dynamic_batch")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    def step(x, y):
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        return loss.detach()

    fn = cc(step, dev)
    out = []
    for n in [16, 7, 31, 2]:
        x, y = (t.to(dev) for t in data(n))
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        out.append(fn(x, y))
    return [torch.stack(out)] + [p.detach() for p in m.parameters()]


# ---- graph breaks inside the train step ----
@case("graph_breaks_in_step")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x, y = (t.to(dev) for t in data())
    log = []

    def step(x, y):
        opt.zero_grad()
        h = m[0](x)
        log.append(float(h.abs().mean()))  # .item(): graph break
        h = m[1:](h)
        loss = F.cross_entropy(h, y)
        if loss > 0:  # data-dependent control flow
            loss = loss * 1.0
        print("", end="")  # graph break
        loss.backward()
        opt.step()
        return loss.detach()

    fn = cc(step, dev)
    out = [fn(x, y) for _ in range(3)]
    return [torch.stack(out), torch.tensor(log)] + [p.detach() for p in m.parameters()]


# ---- AC + dropout under compile: check grads match the mask that was used in forward ----
@case("compile_checkpoint_dropout_consistency")
def _(dev):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    lin = nn.Linear(16, 16).to(dev)
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(4)).to(dev)

    def f(x):
        return F.dropout(lin(x), 0.5, training=True)

    def step(x):
        o = checkpoint(f, x, use_reentrant=False)
        o.sum().backward()
        return o

    fn = cc(step, dev)
    errs = []
    for _ in range(3):
        lin.weight.grad = None
        o = fn(x)
        expect = ((o != 0).float() * 2.0).t() @ x
        errs.append((lin.weight.grad - expect).abs().max().detach().cpu())
    return [torch.stack(errs)]


@case("compile_dropout_backward_consistency")
def _(dev):
    torch.manual_seed(0)
    lin = nn.Linear(16, 16).to(dev)
    x = torch.randn(8, 16).to(dev)
    fn = cc(lambda x: F.dropout(lin(x), 0.5, training=True), dev)
    errs = []
    for _ in range(3):
        lin.weight.grad = None
        o = fn(x)
        o.sum().backward()
        expect = ((o != 0).float() * 2.0).t() @ x
        errs.append((lin.weight.grad - expect).abs().max().detach().cpu())
    return [torch.stack(errs)]


# ---- buffer mutation in forward (same bug class as BatchNorm) ----
class BufMut(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema", torch.zeros(8))
        self.register_buffer("cache", torch.zeros(6, 8))

    def forward(self, x, pos):
        self.count += 1
        self.ema.mul_(0.9).add_(x.mean(0).detach(), alpha=0.1)
        self.cache.index_copy_(0, pos, x[: pos.numel()].detach())
        self.cache[pos[0]] += 1.0
        return self.lin(x) + self.ema


@case("compile_buffer_mutation_forward")
def _(dev):
    torch.manual_seed(0)
    m = BufMut().to(dev)
    fn = cc(m, dev)
    outs = []
    for i in range(3):
        x = torch.randn(4, 8, generator=torch.Generator().manual_seed(i)).to(dev)
        pos = torch.tensor([i, i + 2]).to(dev)
        o = fn(x, pos)
        o.sum().backward()
        outs.append(o.detach())
    return outs + [m.count.float(), m.ema, m.cache, m.lin.weight.grad]


@case("eager_buffer_mutation_forward")
def _(dev):
    torch.manual_seed(0)
    m = BufMut().to(dev)
    outs = []
    for i in range(3):
        x = torch.randn(4, 8, generator=torch.Generator().manual_seed(i)).to(dev)
        pos = torch.tensor([i, i + 2]).to(dev)
        o = m(x, pos)
        o.sum().backward()
        outs.append(o.detach())
    return outs + [m.count.float(), m.ema, m.cache, m.lin.weight.grad]


@case("compile_input_mutation")
def _(dev):
    def f(a, b, c):
        a.add_(1.0)
        b[1:3].mul_(2.0)
        c.copy_(a.sum(0))
        return (a * b).sum()
    g = torch.Generator().manual_seed(3)
    a, b, c = torch.randn(4, 3, generator=g).to(dev), torch.randn(4, 3, generator=g).to(dev), torch.zeros(3).to(dev)
    r = cc(f, dev)(a, b, c)
    return [r, a, b, c]


@case("compile_mutation_of_view_input")
def _(dev):
    def f(v):
        v.mul_(3.0)
        return v.sum()
    base = torch.arange(12.0).reshape(3, 4).to(dev)
    r = cc(f, dev)(base[:, 1:3])
    r2 = cc(f, dev)(base.t()[1])
    return [r, r2, base]


# ---- autocast / scaler under compile ----
@case("compile_autocast_fp16_gradscaler", atol=3e-2, rtol=3e-2)
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(dev if dev != "cpu" else "cpu", init_scale=2.0 ** 8)
    x, y = (t.to(dev) for t in data())
    fwd = cc(lambda x, y: F.cross_entropy(m(x), y), dev)
    out = []
    for _ in range(4):
        opt.zero_grad()
        with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.float16):
            loss = fwd(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        out.append(loss.detach().float())
    return [torch.stack(out), torch.tensor([scaler.get_scale()])] + [p.detach() for p in m.parameters()]


@case("compile_autocast_inside_bf16", atol=3e-2, rtol=3e-2)
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x, y = (t.to(dev) for t in data())
    dt = dev if dev != "cpu" else "cpu"

    def step(x, y):
        opt.zero_grad()
        with torch.autocast(dt, dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        return loss.detach().float()

    fn = cc(step, dev)
    return [torch.stack([fn(x, y) for _ in range(3)])] + [p.detach() for p in m.parameters()]


# ---- misc training patterns ----
@case("compile_grad_accumulation")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    fwdbwd = cc(lambda x, y: (F.cross_entropy(m(x), y) / 4).backward(), dev)
    for k in range(4):
        x, y = (t.to(dev) for t in data(8, seed=k))
        fwdbwd(x, y)
    g = [p.grad.clone() for p in m.parameters()]
    opt.step()
    return g + [p.detach() for p in m.parameters()]


@case("compile_frozen_layers_param_groups")
def _(dev):
    m = mlp().to(dev)
    m[0].requires_grad_(False)
    opt = torch.optim.AdamW([{"params": m[2].parameters(), "lr": 1e-1}, {"params": m[3].parameters(), "lr": 1e-2}])
    x, y = (t.to(dev) for t in data())

    def step(x, y):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()

    fn = cc(step, dev)
    for _ in range(3):
        fn(x, y)
    return [p.detach() for p in m.parameters()] + [torch.tensor([float(m[0].weight.grad is None)])]


@case("compile_ema_update_no_grad")
def _(dev):
    m = mlp().to(dev)
    ema = copy.deepcopy(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x, y = (t.to(dev) for t in data())

    @torch.no_grad()
    def upd():
        torch._foreach_lerp_(list(ema.parameters()), list(m.parameters()), 0.1)

    u = cc(upd, dev)
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        u()
    return [p.detach() for p in ema.parameters()]


@case("compile_weight_tying_lm")
def _(dev):
    torch.manual_seed(0)
    emb = nn.Embedding(50, 16)
    head = nn.Linear(16, 50, bias=False)
    head.weight = emb.weight
    m = nn.ModuleDict({"emb": emb, "head": head}).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 50, (4, 9), generator=torch.Generator().manual_seed(0)).to(dev)

    def step(ids):
        opt.zero_grad()
        logits = m["head"](m["emb"](ids[:, :-1]))
        loss = F.cross_entropy(logits.reshape(-1, 50), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        return loss.detach()

    fn = cc(step, dev)
    return [torch.stack([fn(ids) for _ in range(3)]), m["emb"].weight.detach()]


@case("compile_cnn_bn_train_eval")
def _(dev):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Linear(8 * 4 * 4, 10)).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    x = torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(0)).to(dev)
    y = torch.tensor([1, 2, 3, 4]).to(dev)
    fm = cc(m, dev)
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(fm(x), y).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        ev = fm(x)
    return [ev] + [p.detach() for p in m.parameters()] + [m[1].running_mean, m[1].running_var]


@case("compile_sdpa_masks_backward")
def _(dev):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 32, generator=torch.Generator().manual_seed(i)).to(dev).requires_grad_() for i in range(3))
    mask = (torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(9)) > 0.3).to(dev)
    f = cc(lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m).pow(2).sum(), dev)
    l = f(q, k, v, mask)
    l.backward()
    f2 = cc(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.2).pow(2).sum(), dev)
    l2 = f2(q, k, v)
    return [l, l2, q.grad, k.grad, v.grad]


@case("compile_func_per_sample_grads")
def _(dev):
    from torch.func import functional_call, grad, vmap
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).to(dev)
    params = {k: v.detach() for k, v in m.named_parameters()}
    x = torch.randn(5, 4).to(dev)
    y = torch.randn(5, 1).to(dev)

    def loss(p, xi, yi):
        return ((functional_call(m, p, (xi.unsqueeze(0),)) - yi) ** 2).sum()

    f = cc(vmap(grad(loss), in_dims=(None, 0, 0)), dev)
    per = f(params, x, y)
    return [per[k] for k in sorted(per)]


@case("compile_embedding_bag_and_scatter_bwd")
def _(dev):
    torch.manual_seed(0)
    w = torch.randn(10, 4).to(dev).requires_grad_()
    src = torch.randn(6, 4).to(dev).requires_grad_()
    idx = torch.tensor([1, 1, 3, 1, 9, 0]).to(dev)

    def f(w, src, idx):
        a = F.embedding(idx, w).sum()
        b = torch.zeros(10, 4, device=w.device).index_add(0, idx, src * w[idx]).pow(2).sum()
        c = torch.zeros(10, 4, device=w.device).index_put((idx,), src, accumulate=True).pow(2).sum()
        return a + b + c

    l = cc(f, dev)(w, src, idx)
    l.backward()
    return [l, w.grad, src.grad]


@case("compile_ce_ignore_index_label_smoothing")
def _(dev):
    x = torch.randn(12, 7, generator=torch.Generator().manual_seed(4)).to(dev).requires_grad_()
    y = torch.tensor([0, 1, -100, 3, 4, 5, 6, -100, 1, 2, 3, 4]).to(dev)
    f = cc(lambda x, y: F.cross_entropy(x, y, label_smoothing=0.1) + F.cross_entropy(x, y, reduction="sum") * 0.01, dev)
    l = f(x, y)
    l.backward()
    return [l, x.grad]


@case("compile_all_ignored_ce")
def _(dev):
    x = torch.randn(4, 7, generator=torch.Generator().manual_seed(4)).to(dev).requires_grad_()
    y = torch.full((4,), -100).to(dev)
    l = cc(lambda x, y: F.cross_entropy(x, y), dev)(x, y)
    l.backward()
    return [l.nan_to_num(77.0), x.grad.nan_to_num(77.0)]


def main():
    sel = sys.argv[1:]
    names = [n for n in CASES if not sel or any(s in n for s in sel)]
    summary = []
    for name in names:
        fn, atol, rtol = CASES[name]
        try:
            ref = [t.detach().float().cpu() for t in fn("cpu")]
            dynamo.reset()
            t0 = time.time()
            out = [t.detach().float().cpu() for t in fn(DEV)]
            torch_neuronx.synchronize() if hasattr(torch_neuronx, "synchronize") else None
            dt = time.time() - t0
            status, worst = "OK", 0.0
            for i, (r, o) in enumerate(zip(ref, out)):
                if r.shape != o.shape:
                    status = f"SHAPE@{i}"
                    break
                if not torch.allclose(r, o, atol=atol, rtol=rtol, equal_nan=True):
                    d = (r - o).abs().nan_to_num(1e9).max().item()
                    status, worst = "MISMATCH", max(worst, d)
                    print(f"  [{name}] tensor {i} maxdiff {d:.4g}  ref={r.flatten()[:4].tolist()} got={o.flatten()[:4].tolist()}", flush=True)
            line = f"{status:10s} {name:45s} {worst:.4g}  ({dt:.0f}s)"
        except Exception as e:
            traceback.print_exc()
            line = f"ERROR      {name:45s} {type(e).__name__}: {str(e)[:250]}".replace("\n", " ")
        print("RESULT " + line, flush=True)
        summary.append(line)
    print("\n===== SUMMARY =====")
    print("\n".join(summary))
    print("fallback ops:", sorted(torch_neuronx.get_fallback_ops()))


if __name__ == "__main__":
    main()
