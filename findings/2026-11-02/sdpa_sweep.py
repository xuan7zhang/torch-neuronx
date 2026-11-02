"""SDPA sweep (no dropout): fwd + grads of q/k/v and of a learnable float mask, eager and compiled, vs CPU fp32."""

import itertools
import sys

import torch
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
comp = len(sys.argv) > 1 and sys.argv[1] == "compile"


def rel(a, b):
    if a is None or b is None:
        return 0.0 if a is None and b is None else (99.0 if b is None else 98.0)
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


configs = []
for D, (Lq, Lk), mask, causal, hq_hkv, dt in itertools.product(
        [64, 72, 128, 256], [(128, 128), (100, 100), (64, 200)], ["none", "bool", "float_grad", "bool_bcast"],
        [False, True], [(4, 4), (8, 2)], [torch.float32, torch.bfloat16]):
    if causal and mask != "none":
        continue  # SDPA forbids attn_mask with is_causal
    configs.append((D, Lq, Lk, mask, causal, hq_hkv, dt))

bad = 0
for i, (D, Lq, Lk, mask, causal, (hq, hkv), dt) in enumerate(configs):
    if dt == torch.bfloat16 and (D, Lq) not in ((128, 128), (64, 100)):
        continue
    g = torch.Generator().manual_seed(i)
    q = torch.randn(2, hq, Lq, D, generator=g)
    k = torch.randn(2, hkv, Lk, D, generator=g)
    v = torch.randn(2, hkv, Lk, D, generator=g)
    go = torch.randn(2, hq, Lq, D, generator=g)
    m = None
    if mask == "bool":
        m = torch.rand(2, hq, Lq, Lk, generator=g) > 0.3
        m[..., 0] = True
    elif mask == "bool_bcast":
        m = torch.rand(1, 1, Lq, Lk, generator=g) > 0.3
        m[..., 0] = True
    elif mask == "float_grad":
        m = torch.randn(1, hq, Lq, Lk, generator=g) * 0.5
    res = {}
    for dev in ("cpu", DEV):
        qq, kk, vv = (t.to(dt).to(dev).detach().clone().requires_grad_() for t in (q, k, v))
        mm = None
        if m is not None:
            mm = m.to(dev) if m.dtype == torch.bool else m.to(dt).to(dev).detach().clone().requires_grad_()
        f = lambda a, b, c, msk: F.scaled_dot_product_attention(a, b, c, attn_mask=msk, is_causal=causal,
                                                                 enable_gqa=hq != hkv, scale=0.3 if D == 72 else None)
        if comp and dev == DEV:
            torch._dynamo.reset()
            f = torch.compile(f, backend="neuron", dynamic=False)
        try:
            o = f(qq, kk, vv, mm)
            o.backward(go.to(dt).to(dev))
            res[dev] = [o.detach(), qq.grad, kk.grad, vv.grad] + ([mm.grad] if mask == "float_grad" else [])
        except Exception as e:
            res[dev] = str(e)[:140].replace("\n", " ")
    name = f"D={D} Lq={Lq} Lk={Lk} mask={mask} causal={causal} hq/hkv={hq}/{hkv} {str(dt)[6:]}"
    if isinstance(res[DEV], str) or isinstance(res["cpu"], str):
        print("RES ERR ", name, res[DEV] if isinstance(res[DEV], str) else "cpu:" + res["cpu"], flush=True)
        bad += 1
        continue
    errs = [rel(a, b) for a, b in zip(res["cpu"], res[DEV])]
    tol = 2e-2 if dt == torch.bfloat16 else 1e-4
    labels = ["o", "dq", "dk", "dv", "dmask"]
    if max(errs) > tol:
        bad += 1
        print("RES BAD ", name, {l: f"{e:.3g}" for l, e in zip(labels, errs)}, flush=True)
print(f"RES done {'compile' if comp else 'eager'}: {len([c for c in configs if not (c[-1] == torch.bfloat16 and (c[0], c[1]) not in ((128, 128), (64, 100)))])} configs, {bad} bad")
