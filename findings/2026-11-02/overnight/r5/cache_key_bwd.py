"""Same as cache_key.py but checks x.grad: backward ops with scalar args, same shapes, several values in one process."""
import torch
import torch.nn.functional as F
import torch_neuronx

g = torch.Generator().manual_seed(0)
X = torch.randn(8, 16, generator=g)
Y = torch.randn(8, 16, generator=g)
G = torch.randn(8, 16, generator=g)
C = {
    "leaky_relu": [lambda x, s=s: F.leaky_relu(x, s) for s in (0.01, 0.2, 0.5)],
    "elu": [lambda x, a=a: F.elu(x, a) for a in (1.0, 0.3, 2.0)],
    "celu": [lambda x, a=a: F.celu(x, a) for a in (1.0, 0.3, 2.0)],
    "hardtanh": [lambda x, m=m: F.hardtanh(x, -m, m) for m in (1.0, 0.3, 2.0)],
    "softplus": [lambda x, b=b: F.softplus(x, b, threshold=t) for b, t in ((1.0, 20.0), (2.0, 1.0), (5.0, 0.5))],
    "gelu": [lambda x, a=a: F.gelu(x, approximate=a) for a in ("none", "tanh")],
    "threshold": [lambda x, t=t: F.threshold(x, t, -9.0) for t in (0.0, 0.5, -0.5)],
    "clamp": [lambda x, lo=lo: x.clamp(lo, lo + 1) for lo in (-1.0, 0.0, 0.5)],
    "pow": [lambda x, p=p: x.abs().add(0.1).pow(p) for p in (2.0, 3.0, 0.5)],
    "smooth_l1": [lambda x, b=b: F.smooth_l1_loss(x, Y.to(x.device), beta=b, reduction="none") for b in (1.0, 0.1, 3.0)],
    "huber": [lambda x, b=b: F.huber_loss(x, Y.to(x.device), delta=b, reduction="none") for b in (1.0, 0.1, 3.0)],
    "softmax_dim": [lambda x, d=d: x.softmax(d) for d in (0, 1)],
    "log_softmax_dim": [lambda x, d=d: x.log_softmax(d) for d in (0, 1)],
    "layer_norm_eps": [lambda x, e=e: F.layer_norm(x * 1e-2, (16,), eps=e) for e in (1e-5, 1e-2, 1.0)],
    "rms_norm_eps": [lambda x, e=e: F.rms_norm(x * 1e-2, (16,), eps=e) for e in (1e-6, 1e-2, 1.0)],
    "group_norm_eps": [lambda x, e=e: F.group_norm(x.view(8, 16, 1) * 1e-2, 4, eps=e).view(8, 16) for e in (1e-5, 1e-2, 1.0)],
    "hardshrink": [lambda x, l=l: F.hardshrink(x, l) for l in (0.5, 1.0, 0.1)],
    "softshrink": [lambda x, l=l: F.softshrink(x, l) for l in (0.5, 1.0, 0.1)],
    "rrelu_eval": [lambda x, l=l: F.rrelu(x, l, l + 0.1, training=False) for l in (0.1, 0.3)],
    "hardsigmoid_swish": [F.hardsigmoid, F.hardswish, F.silu, F.mish],
    "logsigmoid_tanhshrink": [F.logsigmoid, F.tanhshrink, F.softsign, F.selu],
    "mse_vs_l1": [lambda x: F.mse_loss(x, Y.to(x.device), reduction="none"), lambda x: F.l1_loss(x, Y.to(x.device), reduction="none")],
    "cumsum_dim": [lambda x, d=d: x.cumsum(d) for d in (0, 1)],
    "cumprod_dim": [lambda x, d=d: (x * 0.1 + 1).cumprod(d) for d in (0, 1)],
    "norm_p": [lambda x, p=p: x.norm(p, dim=1, keepdim=True).expand(8, 16) for p in (2.0, 1.0, 3.0)],
    "glu_dim": [lambda x, d=d: F.glu(x, d).repeat(1, 2) if d == 1 else F.glu(x, d).repeat(2, 1) for d in (1, 0)],
    "dropout_eval": [lambda x, p=p: F.dropout(x, p, training=False) for p in (0.1, 0.5)],
    "div_scalar": [lambda x, s=s: x / s for s in (2.0, 3.0, -0.1)],
    "addcdiv": [lambda x, v=v: torch.addcdiv(x, x, Y.to(x.device).abs() + 1, value=v) for v in (1.0, 0.1, -3.0)],
    "lerp": [lambda x, w=w: torch.lerp(x, Y.to(x.device), w) for w in (0.1, 0.5, 0.9)],
}
bad = 0
for name, fns in C.items():
    out = []
    for i, f in enumerate(fns):
        try:
            grads = []
            for dev in ("cpu", "neuron"):
                x = X.to(dev).detach().clone().requires_grad_()
                f(x).backward(G.to(dev))
                grads.append(x.grad.cpu())
            r, n = grads
            err = ((r - n).abs().max() / (r.abs().max() + 1e-12)).item()
            ok = err < 1e-3
            out.append(f"c{i}:{'ok' if ok else 'BAD'}({err:.2g})")
            bad += not ok
        except Exception as e:
            out.append(f"c{i}:ERR[{type(e).__name__}: {str(e)[:80]}]".replace("\n", " "))
    print(("OK    " if all(":ok" in o for o in out) else "ISSUE ") + f"{name:24s} " + " ".join(out), flush=True)
print("bad:", bad)
