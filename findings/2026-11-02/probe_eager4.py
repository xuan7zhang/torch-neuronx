"""Eager round 4: large-scale numerics and double backward. Reference: CPU fp32 (bf16 cases: CPU fp32 on the
bf16-rounded inputs), reported as relative error so the scale is clear."""

import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
CASES = {}


def case(name, tol):
    def deco(fn):
        CASES[name] = (fn, tol)
        return fn
    return deco


def g(seed):
    return torch.Generator().manual_seed(seed)


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


# ---- large-vocab cross entropy (chunked kernel) ----
for V, N, dt, ls, ign in [(128256, 1024, torch.float32, 0.0, False), (128256, 1024, torch.bfloat16, 0.0, False),
                          (128256, 1024, torch.bfloat16, 0.1, True), (50257, 2048, torch.float32, 0.0, True),
                          (32000, 4096, torch.bfloat16, 0.0, False)]:
    @case(f"ce_V{V}_N{N}_{str(dt)[6:]}_ls{ls}_ign{ign}", 2e-2 if dt == torch.bfloat16 else 1e-4)
    def _(V=V, N=N, dt=dt, ls=ls, ign=ign):
        x = (torch.randn(N, V, generator=g(0)) * 2).to(dt)
        y = torch.randint(0, V, (N,), generator=g(1))
        if ign:
            y[::7] = -100
        xc = x.float().clone().requires_grad_()
        lc = F.cross_entropy(xc, y, label_smoothing=ls)
        lc.backward()
        xn = x.to(DEV).requires_grad_()
        ln = F.cross_entropy(xn, y.to(DEV), label_smoothing=ls)
        ln.backward()
        return {"loss": rel(lc.detach(), ln.detach()), "grad": rel(xc.grad, xn.grad)}


# ---- large embedding backward with many duplicates ----
for dt in (torch.float32, torch.bfloat16):
    @case(f"embedding_bwd_V128k_D1024_dups_{str(dt)[6:]}", 2e-2 if dt == torch.bfloat16 else 1e-4)
    def _(dt=dt):
        w = (torch.randn(128000, 1024, generator=g(0)) * 0.02).to(dt)
        idx = torch.randint(0, 2000, (8, 2048), generator=g(1))  # heavy duplication
        go = torch.randn(8, 2048, 1024, generator=g(2)).to(dt)
        wc = w.float().clone().requires_grad_()
        F.embedding(idx, wc).backward(go.float())
        wn = w.to(DEV).requires_grad_()
        F.embedding(idx.to(DEV), wn).backward(go.to(DEV))
        rows = idx.unique()
        return {"grad_rows": rel(wc.grad[rows], wn.grad[rows.to(DEV)]), "untouched_zero": float(wn.grad[5000:6000].abs().max().item())}


# ---- long softmax / norms / matmul ----
for dt in (torch.float32, torch.bfloat16):
    @case(f"softmax_row32k_{str(dt)[6:]}", 2e-2 if dt == torch.bfloat16 else 1e-5)
    def _(dt=dt):
        x = (torch.randn(64, 32768, generator=g(0)) * 3).to(dt)
        xc = x.float().clone().requires_grad_()
        yc = xc.softmax(-1)
        go = torch.randn(64, 32768, generator=g(1))
        yc.backward(go)
        xn = x.to(DEV).requires_grad_()
        yn = xn.softmax(-1)
        yn.backward(go.to(dt).to(DEV))
        return {"fwd": rel(yc, yn), "grad": rel(xc.grad, xn.grad)}

    @case(f"layernorm_rmsnorm_h8192_{str(dt)[6:]}", 2e-2 if dt == torch.bfloat16 else 1e-4)
    def _(dt=dt):
        x = (torch.randn(256, 8192, generator=g(0)) * 4 + 1).to(dt)
        out = {}
        for nm, mk in (("ln", lambda: nn.LayerNorm(8192)), ("rms", lambda: nn.RMSNorm(8192))):
            torch.manual_seed(0)
            mc = mk()
            with torch.no_grad():
                mc.weight.normal_(1.0, 0.1)
            mn = mk().to(DEV).to(dt)
            mn.load_state_dict({k: v.to(dt) for k, v in mc.state_dict().items()})
            mc.weight.data = mc.weight.data.to(dt).float()
            if getattr(mc, "bias", None) is not None:
                mc.bias.data = mc.bias.data.to(dt).float()
            xc = x.float().clone().requires_grad_()
            go = torch.randn(256, 8192, generator=g(3))
            mc(xc).backward(go)
            xn = x.to(DEV).requires_grad_()
            mn(xn).backward(go.to(dt).to(DEV))
            out[f"{nm}_xgrad"] = rel(xc.grad, xn.grad)
            out[f"{nm}_wgrad"] = rel(mc.weight.grad, mn.weight.grad)
        return out

    @case(f"matmul_K16384_{str(dt)[6:]}", 2e-2 if dt == torch.bfloat16 else 1e-4)
    def _(dt=dt):
        a = torch.randn(256, 16384, generator=g(0)).to(dt)
        b = torch.randn(16384, 256, generator=g(1)).to(dt)
        ac, bc = a.float().clone().requires_grad_(), b.float().clone().requires_grad_()
        (ac @ bc).sum().backward()
        an, bn = a.to(DEV).requires_grad_(), b.to(DEV).requires_grad_()
        (an @ bn).float().sum().backward()
        return {"fwd": rel(a.float() @ b.float(), (an @ bn).detach()), "a_grad": rel(ac.grad, an.grad)}


@case("sum_mean_large_bf16", 2e-2)
def _():
    x = (torch.rand(4096, 4096, generator=g(0)) + 0.5).to(torch.bfloat16)
    xn = x.to(DEV)
    return {"sum": rel(x.float().sum(), xn.sum().float()), "mean": rel(x.float().mean(), xn.mean().float()),
            "sum_dim": rel(x.float().sum(1), xn.sum(1).float()), "var": rel(x.float().var(1), xn.var(1).float())}


# ---- double backward ----
def make_disc():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 8, 3, stride=2, padding=1),
                         nn.Flatten(), nn.LayerNorm(8 * 8 * 8), nn.Linear(8 * 8 * 8, 1))


@case("wgan_gp_gradient_penalty", 1e-3)
def _():
    out = {}
    res = {}
    for dev in ("cpu", DEV):
        D = make_disc().to(dev)
        real = torch.randn(4, 3, 16, 16, generator=g(0)).to(dev)
        fake = torch.randn(4, 3, 16, 16, generator=g(1)).to(dev)
        a = torch.rand(4, 1, 1, 1, generator=g(2)).to(dev)
        xi = (a * real + (1 - a) * fake).requires_grad_()
        d = D(xi)
        gr, = torch.autograd.grad(d.sum(), xi, create_graph=True)
        gp = ((gr.flatten(1).norm(dim=1) - 1) ** 2).mean()
        loss = D(fake).mean() - D(real).mean() + 10 * gp
        loss.backward()
        res[dev] = (gp.detach(), [p.grad for p in D.parameters()])
    out["gp"] = rel(res["cpu"][0], res[DEV][0])
    out["param_grads"] = max(rel(a, b) for a, b in zip(res["cpu"][1], res[DEV][1]))
    return out


@case("hessian_vector_product_mlp", 1e-3)
def _():
    res = {}
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)).to(dev)
        x = torch.randn(8, 16, generator=g(0)).to(dev)
        ps = list(m.parameters())
        loss = m(x).pow(2).mean()
        gs = torch.autograd.grad(loss, ps, create_graph=True)
        v = [torch.randn(p.shape, generator=g(5)).to(dev) for p in ps]
        hv = torch.autograd.grad(sum((gi * vi).sum() for gi, vi in zip(gs, v)), ps)
        res[dev] = hv
    return {"hvp": max(rel(a, b) for a, b in zip(res["cpu"], res[DEV]))}


@case("double_backward_softmax_layernorm_sdpa", 1e-3)
def _():
    res = {}
    for dev in ("cpu", DEV):
        x = torch.randn(2, 4, 8, 16, generator=g(0)).to(dev).requires_grad_()
        import contextlib
        ctx = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if dev == "cpu" else contextlib.nullcontext()
        with ctx:
            y = F.scaled_dot_product_attention(x, x, x, is_causal=True)
        y = F.layer_norm(y, (16,)).softmax(-1)
        gx, = torch.autograd.grad(y.pow(2).sum(), x, create_graph=True)
        gx.pow(2).sum().backward()
        res[dev] = (gx.detach(), x.grad)
    return {"g1": rel(res["cpu"][0], res[DEV][0]), "g2": rel(res["cpu"][1], res[DEV][1])}


@case("resnet_inplace_relu_bn_train", 1e-3)
def _():
    def mk():
        torch.manual_seed(0)
        blk = lambda c: nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                      nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

        class Res(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
                self.b1, self.b2 = blk(16), blk(16)
                self.head = nn.Linear(16, 10)

            def forward(self, x):
                x = self.stem(x)
                x = F.relu(x + self.b1(x), inplace=True)
                x = F.relu(x + self.b2(x), inplace=True)
                return self.head(x.mean((2, 3)))
        return Res()
    res = {}
    for dev in ("cpu", DEV):
        m = mk().to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
        for i in range(3):
            x = torch.randn(8, 3, 16, 16, generator=g(i)).to(dev)
            y = torch.randint(0, 10, (8,), generator=g(10 + i)).to(dev)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
        res[dev] = [p.detach() for p in m.parameters()]
    return {"params": max(rel(a, b) for a, b in zip(res["cpu"], res[DEV]))}


def main():
    sel = sys.argv[1:]
    for name, (fn, tol) in CASES.items():
        if sel and not any(s in name for s in sel):
            continue
        try:
            r = fn()
            torch.neuron.synchronize()
            bad = {k: v for k, v in r.items() if k != "untouched_zero" and v > tol}
            if r.get("untouched_zero", 0) != 0:
                bad["untouched_zero"] = r["untouched_zero"]
            st = "OK      " if not bad else "MISMATCH"
            print(f"RESULT {st} {name:45s} " + " ".join(f"{k}={v:.3g}" for k, v in r.items()) + f"  (tol {tol})", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"RESULT ERROR    {name:45s} {type(e).__name__}: {str(e)[:250]}".replace("\n", " "), flush=True)
    print("fallback ops:", sorted(torch_neuronx.get_fallback_ops()))


if __name__ == "__main__":
    main()
