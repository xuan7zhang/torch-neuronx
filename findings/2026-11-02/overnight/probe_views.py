"""View sweep: ops writing into / reading from views of different layouts, plus RNG reproducibility and int64.
Usage: probe_views.py views|rng|int64"""

import sys
import traceback

import torch
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
mode = sys.argv[1]
G = torch.Generator().manual_seed(0)

VIEWS = {
    "cols_offset": lambda b: b[:, 1:4],          # non-contig, offset 1
    "rows_offset": lambda b: b[2:5],             # contig, offset
    "transposed": lambda b: b.t()[1:4],          # non-contig, offset
    "strided": lambda b: b[1::2, ::2],           # non-contig, offset
    "diagonal": lambda b: b.diagonal(1),         # 1-d, stride 7
    "inner_block": lambda b: b[1:5, 2:5],        # non-contig, offset 8
}


def base():
    return torch.arange(36.0).reshape(6, 6) / 7


def results(nm, ok, info=""):
    print(f"RES {'OK ' if ok else 'BAD'} {nm:52s} {info}", flush=True)


if mode == "views":
    # in-place / out= ops into views; compare full base tensor with CPU
    def ops_for(v):
        shape = v.shape
        src = torch.randn(shape, generator=torch.Generator().manual_seed(1))
        return {
            "add_": lambda t: t.add_(src.to(t.device)),
            "mul_scalar_": lambda t: t.mul_(3.0),
            "copy_": lambda t: t.copy_(src.to(t.device)),
            "fill_": lambda t: t.fill_(2.5),
            "zero_": lambda t: t.zero_(),
            "clamp_": lambda t: t.clamp_(0.2, 0.6),
            "exp_": lambda t: t.exp_(),
            "addcmul_": lambda t: t.addcmul_(src.to(t.device), src.to(t.device), value=0.5),
            "lerp_": lambda t: t.lerp_(src.to(t.device), 0.3),
            "masked_fill_": lambda t: t.masked_fill_(src.to(t.device) > 0, -1.0),
            "setitem_scalar": lambda t: t.__setitem__(..., 4.0),
            "foreach_add_": lambda t: torch._foreach_add_([t], [src.to(t.device)], alpha=0.5),
            "foreach_mul_": lambda t: torch._foreach_mul_([t], 0.25),
            "foreach_lerp_": lambda t: torch._foreach_lerp_([t], [src.to(t.device)], 0.5),
            "torch.add(out=)": lambda t: torch.add(src.to(t.device), 1.0, out=t),
            "torch.mul(out=)": lambda t: torch.mul(src.to(t.device), src.to(t.device), out=t),
            "torch.sigmoid(out=)": lambda t: torch.sigmoid(src.to(t.device), out=t),
            "index_fill_(0)": lambda t: t.index_fill_(0, torch.tensor([0]).to(t.device), 9.0),
            "scatter_(-1)": lambda t: t.scatter_(-1, torch.zeros(shape, dtype=torch.long).to(t.device)[..., :1] if len(shape) > 1 else torch.tensor([0]).to(t.device), 7.0),
        }

    for vname, vf in VIEWS.items():
        for oname in ops_for(vf(base())).keys():
            try:
                bc = base()
                ops_for(vf(bc))[oname](vf(bc))
                bn = base().to(DEV)
                vn = vf(bn)
                before = (tuple(vn.shape), vn.stride(), vn.storage_offset())
                ops_for(vf(base()))[oname](vn)
                after = (tuple(vn.shape), vn.stride(), vn.storage_offset())
                got = bn.cpu()
                ok = torch.allclose(bc, got, atol=1e-5, equal_nan=True) and before == after
                info = "" if ok else (f"meta {before}->{after} " if before != after else "") + f"maxdiff={(bc - got).abs().max().item():.3g}"
                results(f"write {vname}.{oname}", ok, info)
            except Exception as e:
                results(f"write {vname}.{oname}", False, "ERR " + str(e)[:110].replace("\n", " "))
    # matmul-type outputs into views
    for vname in ("cols_offset", "inner_block", "transposed"):
        for oname, f in {"mm(out=)": lambda a, b, o: torch.mm(a, b, out=o), "addmm_": lambda a, b, o: o.addmm_(a, b),
                         "baddbmm-like bmm(out=view)": None}.items():
            if f is None:
                continue
            try:
                bc = base()
                vc = VIEWS[vname](bc)
                m, n = vc.shape
                a = torch.randn(m, 5, generator=torch.Generator().manual_seed(2))
                b = torch.randn(5, n, generator=torch.Generator().manual_seed(3))
                f(a, b, vc)
                bn = base().to(DEV)
                f(a.to(DEV), b.to(DEV), VIEWS[vname](bn))
                got = bn.cpu()
                results(f"write {vname}.{oname}", torch.allclose(bc, got, atol=1e-4), f"maxdiff={(bc - got).abs().max().item():.3g}")
            except Exception as e:
                results(f"write {vname}.{oname}", False, "ERR " + str(e)[:110].replace("\n", " "))
    # ops reading from views (inputs), with autograd through the view
    for vname, vf in VIEWS.items():
        for oname, f in {"softmax": lambda t: t.softmax(-1), "sum": lambda t: t.sum(0), "matmul": lambda t: t @ t.transpose(-1, -2) if t.dim() == 2 else t * t,
                         "layer_norm": lambda t: F.layer_norm(t, t.shape[-1:]), "cumsum": lambda t: t.cumsum(-1), "gelu": F.gelu,
                         "index_select": lambda t: t.index_select(0, torch.tensor([0, 0]).to(t.device)), "cat": lambda t: torch.cat([t, t * 2], 0)}.items():
            try:
                bc = base().requires_grad_()
                yc = f(vf(bc))
                yc.pow(2).sum().backward()
                bn = base().to(DEV).requires_grad_()
                yn = f(vf(bn))
                yn.pow(2).sum().backward()
                ok = torch.allclose(yc, yn.detach().cpu(), atol=1e-4) and torch.allclose(bc.grad, bn.grad.cpu(), atol=1e-4)
                results(f"read {vname}.{oname} (+grad)", ok, "" if ok else f"fwd={(yc - yn.detach().cpu()).abs().max().item():.3g} grad={(bc.grad - bn.grad.cpu()).abs().max().item():.3g}")
            except Exception as e:
                results(f"read {vname}.{oname} (+grad)", False, "ERR " + str(e)[:110].replace("\n", " "))

elif mode == "rng":
    ops = {
        "randn": lambda: torch.randn(4096, device=DEV),
        "rand": lambda: torch.rand(4096, device=DEV),
        "randint": lambda: torch.randint(0, 1000, (4096,), device=DEV),
        "randperm": lambda: torch.randperm(1000, device=DEV),
        "normal_": lambda: torch.empty(4096, device=DEV).normal_(2.0, 3.0),
        "uniform_": lambda: torch.empty(4096, device=DEV).uniform_(-1, 1),
        "bernoulli(p tensor)": lambda: torch.bernoulli(torch.full((4096,), 0.3, device=DEV)),
        "bernoulli_(0.3)": lambda: torch.empty(4096, device=DEV).bernoulli_(0.3),
        "multinomial": lambda: torch.multinomial(torch.ones(10, device=DEV), 4096, replacement=True),
        "dropout": lambda: F.dropout(torch.ones(4096, device=DEV), 0.3),
        "nn.init.kaiming_uniform_": lambda: torch.nn.init.kaiming_uniform_(torch.empty(64, 64, device=DEV)),
        "nn.init.trunc_normal_": lambda: torch.nn.init.trunc_normal_(torch.empty(4096, device=DEV), std=0.02),
        "nn.init.orthogonal_": lambda: torch.nn.init.orthogonal_(torch.empty(32, 32, device=DEV)),
        "exponential_": lambda: torch.empty(4096, device=DEV).exponential_(2.0),
        "randn_like bf16": lambda: torch.randn_like(torch.empty(4096, device=DEV, dtype=torch.bfloat16)),
        "rand generator=": lambda: torch.rand(4096, device=DEV, generator=torch.Generator(device=DEV).manual_seed(7)),
    }
    for nm, f in ops.items():
        try:
            torch.manual_seed(123)
            a = f().float().cpu()
            torch.manual_seed(123)
            b = f().float().cpu()
            c = f().float().cpu()
            stats = f"mean={a.mean().item():.3g} std={a.std().item():.3g} min={a.min().item():.3g} max={a.max().item():.3g}"
            uniq = a.unique().numel()
            results(f"rng {nm}: reseed repeats, next draw differs", torch.equal(a, b) and not torch.equal(b, c), stats + f" unique={uniq}")
        except Exception as e:
            results(f"rng {nm}", False, "ERR " + str(e)[:120].replace("\n", " "))
    # statistical checks
    torch.manual_seed(0)
    x = torch.randn(1_000_000, device=DEV).cpu()
    results("rng randn 1M mean~0 std~1", abs(x.mean()) < 0.01 and abs(x.std() - 1) < 0.01, f"mean={x.mean():.4f} std={x.std():.4f}")
    u = torch.rand(1_000_000, device=DEV).cpu()
    results("rng rand 1M in [0,1) mean~0.5", u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.01, f"min={u.min():.3g} max={u.max():.6f} mean={u.mean():.4f}")
    r = torch.randint(0, 10, (1_000_000,), device=DEV).cpu()
    cnt = torch.bincount(r, minlength=10).float() / 1e6
    results("rng randint uniform buckets", (cnt - 0.1).abs().max() < 0.01 and r.min() >= 0 and r.max() <= 9, f"freqs={[round(v, 3) for v in cnt.tolist()]}")
    p = torch.randperm(10000, device=DEV).cpu()
    results("rng randperm is a permutation", torch.equal(p.sort().values, torch.arange(10000)), "")
    m = torch.multinomial(torch.tensor([0.1, 0.0, 0.9], device=DEV), 100000, replacement=True).cpu()
    cm = torch.bincount(m, minlength=3).float() / 1e5
    results("rng multinomial respects probs (0 prob never drawn)", cm[1] == 0 and abs(cm[0] - 0.1) < 0.01, f"freqs={cm.tolist()}")
    tn = torch.nn.init.trunc_normal_(torch.empty(100000, device=DEV), std=1.0, a=-2.0, b=2.0).cpu()
    results("rng trunc_normal_ within [a,b]", tn.min() >= -2 and tn.max() <= 2, f"min={tn.min():.3g} max={tn.max():.3g}")

elif mode == "int64":
    big = 2**33 + 5
    cases = {
        "int64 add > 2^31": (lambda d: torch.tensor([big, -big], device=d) + 7),
        "int64 mul overflow 32b": (lambda d: torch.tensor([70000, -70000], device=d) * 70000),
        "int64 cumsum > 2^31": (lambda d: torch.full((10,), 2**30, dtype=torch.long, device=d).cumsum(0)),
        "int64 sum > 2^31": (lambda d: torch.full((10,), 2**30, dtype=torch.long, device=d).sum()),
        "floor_divide negative": (lambda d: torch.tensor([-7, 7, -8], device=d) // torch.tensor([2, -2, 3], device=d)),
        "remainder negative": (lambda d: torch.remainder(torch.tensor([-7, 7, -8], device=d), torch.tensor([2, -2, 3], device=d))),
        "fmod negative": (lambda d: torch.fmod(torch.tensor([-7, 7, -8], device=d), torch.tensor([2, -2, 3], device=d))),
        "int div true": (lambda d: torch.tensor([7, -7], device=d) / torch.tensor([2, 2], device=d)),
        "shift left": (lambda d: torch.tensor([1, 3], device=d) << 34),
        "bitwise_and big": (lambda d: torch.tensor([big], device=d) & (2**33)),
        "int64 max/argmax big": (lambda d: torch.tensor([big, big + 1, 3], device=d).argmax()),
        "int64 compare big": (lambda d: torch.tensor([big], device=d) > (2**33 + 4)),
        "int64 index with big offset": (lambda d: (torch.arange(5, device=d) + 2**32)[torch.tensor([1, 3], device=d)]),
        "int32 overflow wrap": (lambda d: torch.tensor([2**31 - 1], dtype=torch.int32, device=d) + 1),
        "int64 -> float32": (lambda d: torch.tensor([big], device=d).float()),
        "float -> int64 trunc": (lambda d: torch.tensor([-2.7, 2.7, 3e9], device=d).long()),
        "int pow": (lambda d: torch.tensor([3, -2], device=d) ** 20),
        "abs int64 big": (lambda d: torch.tensor([-big], device=d).abs()),
        "torch.arange int64 big start": (lambda d: torch.arange(2**32, 2**32 + 4, device=d)),
        "hash-embedding ids % V": (lambda d: (torch.tensor([2**40 + 17, 2**35 + 3], device=d) * 31 + 7) % 50021),
    }
    for nm, f in cases.items():
        try:
            c = f("cpu")
            n = f(DEV).cpu()
            results(f"int64 {nm}", torch.equal(c, n), f"cpu={c.tolist()} neuron={n.tolist()}")
        except Exception as e:
            results(f"int64 {nm}", False, "ERR " + str(e)[:120].replace("\n", " "))
