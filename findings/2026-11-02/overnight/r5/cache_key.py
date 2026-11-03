"""Lowering cache key probe: same shapes, different scalar args / strides / dtype on the 2nd call. Compare each call to CPU."""
import sys
import torch
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
g = torch.Generator().manual_seed(0)
X = torch.randn(8, 16, generator=g)
Y = torch.randn(8, 16, generator=g)
X4 = torch.randn(2, 4, 6, 6, generator=g)
W4 = torch.randn(4, 4, 3, 3, generator=g)

# name -> list of callables (each run on cpu then neuron, same input shapes)
C = {
    "add_alpha": [lambda x, y: torch.add(x, y, alpha=a) for a in (2.0, 3.0, -0.5)],
    "leaky_relu": [lambda x, y, s=s: F.leaky_relu(x, s) for s in (0.01, 0.2, 0.5)],
    "clamp": [lambda x, y, lo=lo: x.clamp(lo, lo + 1) for lo in (-1.0, 0.0, 0.5)],
    "pow_scalar": [lambda x, y, p=p: x.abs().pow(p) for p in (2.0, 3.0, 0.5)],
    "mul_scalar": [lambda x, y, s=s: x * s for s in (2.0, 3.0, -7.0)],
    "softmax_dim": [lambda x, y, d=d: x.softmax(d) for d in (0, 1, -1)],
    "sum_dim": [lambda x, y, d=d: x.sum(d) if d is not None else x.sum() for d in (0, 1)],
    "layer_norm_eps": [lambda x, y, e=e: F.layer_norm(x * 1e-3, (16,), eps=e) for e in (1e-5, 1e-2, 1.0)],
    "elu_alpha": [lambda x, y, a=a: F.elu(x, a) for a in (1.0, 0.3, 2.0)],
    "hardtanh": [lambda x, y, m=m: F.hardtanh(x, -m, m) for m in (1.0, 0.3, 2.0)],
    "softplus_beta": [lambda x, y, b=b: F.softplus(x, b) for b in (1.0, 2.0, 5.0)],
    "gelu_approx": [lambda x, y, a=a: F.gelu(x, approximate=a) for a in ("none", "tanh")],
    "threshold": [lambda x, y, t=t: F.threshold(x, t, -9.0) for t in (0.0, 0.5, -0.5)],
    "addcmul_value": [lambda x, y, v=v: torch.addcmul(x, x, y, value=v) for v in (1.0, 0.1, -3.0)],
    "lerp_weight": [lambda x, y, w=w: torch.lerp(x, y, w) for w in (0.1, 0.5, 0.9)],
    "div_rounding": [lambda x, y, r=r: torch.div(x, y, rounding_mode=r) for r in (None, "trunc", "floor")],
    "cumsum_dim": [lambda x, y, d=d: x.cumsum(d) for d in (0, 1)],
    "transpose_input": [lambda x, y: x + y, lambda x, y: x.t().contiguous().t() + y,
                        lambda x, y: x.t().t() * 2 + y],
    "mm_strides": [lambda x, y: x @ y.t(), lambda x, y: x @ y.t().contiguous().t().t(),
                   lambda x, y: x.t().contiguous().t() @ y.t()],
    "argmax_dim": [lambda x, y, d=d: x.argmax(d) for d in (0, 1)],
    "topk_largest": [lambda x, y, l=l: x.topk(3, dim=1, largest=l).values for l in (True, False)],
    "conv_stride_pad": [lambda x, y, s=s, p=p: F.conv2d(X4.to(x.device), W4.to(x.device), stride=s, padding=p)
                        for s, p in ((1, 1), (1, 0), (2, 1))],
    "dropout_eval_p": [lambda x, y, p=p: F.dropout(x, p, training=False) for p in (0.1, 0.5)],
    "fill_value": [lambda x, y, v=v: torch.full_like(x, v) for v in (1.0, 2.5, -3.0)],
    "masked_fill_value": [lambda x, y, v=v: x.masked_fill(y > 0, v) for v in (0.0, -1e4, 7.0)],
    "where_scalar": [lambda x, y, v=v: torch.where(y > 0, x, v) for v in (0.0, 5.0)],
    "rsub": [lambda x, y, v=v: v - x for v in (1.0, 10.0)],
    "clamp_min_tensor_vs_scalar": [lambda x, y: x.clamp_min(0.3), lambda x, y: x.clamp_min(-0.3)],
    "smooth_l1_beta": [lambda x, y, b=b: F.smooth_l1_loss(x, y, beta=b) for b in (1.0, 0.1, 3.0)],
    "huber_delta": [lambda x, y, b=b: F.huber_loss(x, y, delta=b) for b in (1.0, 0.1, 3.0)],
    "ce_label_smoothing": [lambda x, y, l=l: F.cross_entropy(x, torch.arange(8, device=x.device) % 16, label_smoothing=l)
                           for l in (0.0, 0.1, 0.4)],
    "norm_p": [lambda x, y, p=p: x.norm(p) for p in (2.0, 1.0, 3.0)],
    "logsumexp_dim": [lambda x, y, d=d: x.logsumexp(d) for d in (0, 1)],
    "roll_shift": [lambda x, y, s=s: x.roll(s, 1) for s in (1, 3, -2)],
    "narrow_start": [lambda x, y, s=s: x[:, s:s + 4] * 2 for s in (0, 3, 9)],
    "dtype_bf16_then_fp32": [lambda x, y: (x.bfloat16() * 3).float(), lambda x, y: x * 3],
}

sel = sys.argv[1:]
bad = 0
for name, fns in C.items():
    if sel and name not in sel:
        continue
    out = []
    for i, f in enumerate(fns):
        try:
            r = f(X, Y)
            n = f(X.to(DEV), Y.to(DEV)).cpu()
            if r.is_floating_point():
                err = ((r - n).abs().max() / (r.abs().max() + 1e-12)).item()
                ok = err < 1e-3
            else:
                ok, err = torch.equal(r, n), 0.0 if torch.equal(r, n) else 1.0
            out.append(f"c{i}:{'ok' if ok else 'BAD'}({err:.2g})")
            bad += not ok
        except Exception as e:
            out.append(f"c{i}:ERR[{type(e).__name__}: {str(e)[:80]}]".replace("\n", " "))
    print(("OK    " if all(":ok" in o for o in out) else "ISSUE ") + f"{name:28s} " + " ".join(out), flush=True)
print("bad:", bad)
