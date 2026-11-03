"""Compile round 3. Each case returns {metric: rel_err}; CPU eager fp32 is the reference."""

import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
from torch._dynamo.utils import counters

DEV = "neuron"
CASES = {}


def case(name, tol):
    def deco(fn):
        CASES[name] = (fn, tol)
        return fn
    return deco


def g(s):
    return torch.Generator().manual_seed(s)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def cc(f):
    return torch.compile(f, backend="neuron", dynamic=False)


@case("compiled_ce_V128256_N1024", 1e-4)
def _():
    x = torch.randn(1024, 128256, generator=g(0)) * 2
    y = torch.randint(0, 128256, (1024,), generator=g(1))
    y[::5] = -100
    xc = x.clone().requires_grad_()
    F.cross_entropy(xc, y).backward()
    xn = x.to(DEV).requires_grad_()
    t = time.time()
    ln = cc(lambda a, b: F.cross_entropy(a, b))(xn, y.to(DEV))
    ln.backward()
    torch.neuron.synchronize()
    return {"loss": rel(F.cross_entropy(x, y), ln), "grad": rel(xc.grad, xn.grad), "secs": time.time() - t}


@case("compiled_lm_head_ce_bf16_autocast", 2e-2)
def _():
    torch.manual_seed(0)
    head = nn.Linear(512, 32000, bias=False)
    h = torch.randn(512, 512, generator=g(0))
    y = torch.randint(0, 32000, (512,), generator=g(1))
    hc = h.clone().requires_grad_()
    lc = F.cross_entropy(head(hc), y)
    lc.backward()
    hn_mod = nn.Linear(512, 32000, bias=False).to(DEV)
    hn_mod.load_state_dict(head.state_dict())
    hn = h.to(DEV).requires_grad_()

    def f(a, b):
        with torch.autocast(DEV, dtype=torch.bfloat16):
            logits = hn_mod(a)
        return F.cross_entropy(logits.float(), b)

    ln = cc(f)(hn, y.to(DEV))
    ln.backward()
    return {"loss": rel(lc, ln), "h_grad": rel(hc.grad, hn.grad), "w_grad": rel(head.weight.grad, hn_mod.weight.grad)}


@case("train_eval_toggle_recompiles_bn", 1e-4)
def _():
    def mk():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.0), nn.Linear(32, 4))
    mc, mn = mk(), mk().to(DEV)
    fn = cc(mn)
    oc = torch.optim.SGD(mc.parameters(), lr=0.1)
    on = torch.optim.SGD(mn.parameters(), lr=0.1)
    counters.clear()
    evals_c, evals_n = [], []
    for i in range(4):
        x = torch.randn(8, 16, generator=g(i))
        mc.train(); mn.train()
        for m, f, o, xx in ((mc, mc, oc, x), (mn, fn, on, x.to(DEV))):
            o.zero_grad()
            f(xx).pow(2).mean().backward()
            o.step()
        mc.eval(); mn.eval()
        with torch.no_grad():
            evals_c.append(mc(x))
            evals_n.append(fn(x.to(DEV)))
    return {"eval_out": max(rel(a, b) for a, b in zip(evals_c, evals_n)),
            "running_mean": rel(mc[1].running_mean, mn[1].running_mean),
            "unique_graphs": float(counters["stats"]["unique_graphs"])}


@case("compiled_fused_adamw_step", 1e-4)
def _():
    res = {}
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(1e-2) if dev != "cpu" else 1e-2, fused=(dev != "cpu"))
        step = cc(opt.step) if dev != "cpu" else opt.step
        for i in range(3):
            opt.zero_grad()
            m(torch.randn(8, 16, generator=g(i)).to(dev)).pow(2).mean().backward()
            step()
        res[dev] = [p.detach() for p in m.parameters()]
    return {"params": max(rel(a, b) for a, b in zip(res["cpu"], res[DEV]))}


@case("compiled_checkpoint_sdpa_gqa_bf16", 3e-2)
def _():
    from torch.utils.checkpoint import checkpoint

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.q, self.k, self.v, self.o = nn.Linear(64, 64), nn.Linear(64, 16), nn.Linear(64, 16), nn.Linear(64, 64)

        def forward(self, x):
            B, S, _ = x.shape
            q = self.q(x).view(B, S, 8, 8).transpose(1, 2)
            k = self.k(x).view(B, S, 2, 8).transpose(1, 2)
            v = self.v(x).view(B, S, 2, 8).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return self.o(o.transpose(1, 2).reshape(B, S, 64))

    res = {}
    for dev in ("cpu", DEV):
        m = Attn().to(dev)
        x = torch.randn(2, 32, 64, generator=g(0)).to(dev).requires_grad_()

        def f(x):
            with torch.autocast(dev, dtype=torch.bfloat16):
                return checkpoint(m, x, use_reentrant=False).float()

        y = (cc(f) if dev != "cpu" else f)(x)
        y.pow(2).mean().backward()
        res[dev] = [y.detach(), x.grad] + [p.grad for p in m.parameters()]
    names = ["y", "x.grad"] + [n for n, _ in m.named_parameters()]
    out = {n: rel(a, b) for n, a, b in zip(names, res["cpu"], res[DEV])}
    out["worst"] = max(out.values())
    return out


@case("compiled_embedding_padding_scale_freq", 1e-4)
def _():
    res = {}
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        e = nn.Embedding(50, 8, padding_idx=0, scale_grad_by_freq=True).to(dev)
        opt = torch.optim.AdamW(e.parameters(), lr=0.1)
        f = (cc(lambda i: e(i).pow(2).sum()) if dev != "cpu" else (lambda i: e(i).pow(2).sum()))
        for s in range(3):
            idx = torch.randint(0, 6, (4, 10), generator=g(s)).to(dev)
            opt.zero_grad()
            f(idx).backward()
            opt.step()
        res[dev] = e.weight.detach()
    return {"weight": rel(res["cpu"], res[DEV]), "pad_row_abs": float(res[DEV][0].abs().max())}


@case("compiled_norms_h8192_bf16", 2e-2)
def _():
    x = (torch.randn(128, 8192, generator=g(0)) * 4 + 1).to(torch.bfloat16)
    out = {}
    for nm, mk in (("ln", lambda: nn.LayerNorm(8192)), ("rms", lambda: nn.RMSNorm(8192))):
        torch.manual_seed(0)
        mc = mk()
        mn = mk().to(DEV).bfloat16()
        xc = x.float().requires_grad_()
        go = torch.randn(128, 8192, generator=g(1)).to(torch.bfloat16)
        mc(xc).backward(go.float())
        xn = x.to(DEV).requires_grad_()
        cc(mn)(xn).backward(go.to(DEV))
        out[f"{nm}_x"] = rel(xc.grad, xn.grad)
        out[f"{nm}_w"] = rel(mc.weight.grad, mn.weight.grad)
    return out


def main():
    sel = sys.argv[1:]
    for name, (fn, tol) in CASES.items():
        if sel and not any(s in name for s in sel):
            continue
        torch._dynamo.reset()
        try:
            r = fn()
            torch.neuron.synchronize()
            bad = {k: v for k, v in r.items() if k not in ("secs", "unique_graphs", "pad_row_abs") and v > tol}
            if r.get("pad_row_abs", 0) != 0:
                bad["pad_row_abs"] = r["pad_row_abs"]
            st = "OK      " if not bad else "MISMATCH"
            print(f"RESULT {st} {name:42s} " + " ".join(f"{k}={v:.3g}" for k, v in r.items()) + f" (tol {tol})", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"RESULT ERROR    {name:42s} {type(e).__name__}: {str(e)[:250]}".replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
