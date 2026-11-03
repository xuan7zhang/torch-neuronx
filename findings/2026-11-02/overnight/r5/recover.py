"""Error recovery: trigger an error, catch it, then run a reference train step in the same process.
Usage: python recover.py <case>"""
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
nm = torch.neuron


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def train_check(tag):
    """3 AdamW steps of a small MLP on CPU and Neuron. Return max param rel error, or the error text."""
    res = {}
    try:
        for dev in ("cpu", DEV):
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
            for i in range(3):
                x = torch.randn(8, 16, generator=torch.Generator().manual_seed(i)).to(dev)
                opt.zero_grad()
                m(x).pow(2).mean().backward()
                opt.step()
            res[dev] = [p.detach() for p in m.parameters()]
        nm.synchronize()
        e = max(rel(a, b) for a, b in zip(res["cpu"], res[DEV]))
        return f"{tag}: {'OK' if e < 1e-4 else 'WRONG'} rel={e:.2g}"
    except Exception as ex:
        return f"{tag}: FAIL {type(ex).__name__}: {str(ex)[:200]}".replace("\n", " ")


def trigger(fn):
    try:
        fn()
        nm.synchronize()
        return "no error"
    except Exception as ex:
        return f"{type(ex).__name__}: {str(ex)[:200]}".replace("\n", " ")


def mem():
    return f"alloc={nm.memory_allocated() / 2**20:.1f}MiB reserved={nm.memory_reserved() / 2**20:.1f}MiB"


CASES = {}


def case(f):
    CASES[f.__name__] = f
    return f


@case
def oom_single():
    print("before", mem())
    print("error:", trigger(lambda: torch.empty(2**40, device=DEV)))
    print("after", mem())
    print(train_check("train after oom"))


@case
def oom_gradual():
    print("before", mem())
    keep = []

    def grow():
        while True:
            keep.append(torch.ones(2**30, dtype=torch.float32, device=DEV))  # 4 GiB each
    print("error:", trigger(grow), f"(held {len(keep)} x 4 GiB)")
    keep.clear()
    nm.empty_cache()
    print("after clear+empty_cache", mem())
    print(train_check("train after gradual oom"))
    x = torch.ones(2**30, device=DEV)
    print("re-alloc 4 GiB:", float(x[-1]), mem())


@case
def oom_in_backward():
    print("before", mem())
    w = torch.randn(4096, 4096, device=DEV, requires_grad=True)

    def f():
        # Huge activation: (2^16, 4096) @ w -> 1 GiB each, repeated to exhaust memory during forward/backward
        h = torch.randn(2**15, 4096, device=DEV)
        outs = [h @ w for _ in range(200)]
        sum(o.sum() for o in outs).backward()
    print("error:", trigger(f))
    print("w.grad is None:", w.grad is None)
    del w
    nm.empty_cache()
    print("after", mem())
    print(train_check("train after oom in backward"))


@case
def compile_error_zero_size():
    f = torch.compile(lambda t: t.exp() + 1, backend="neuron", dynamic=False)
    print("error:", trigger(lambda: f(torch.randn(0, 3, device=DEV))))
    print(train_check("eager train after compile error"))
    g = torch.compile(nn.Linear(4, 4).to(DEV), backend="neuron", dynamic=False)
    print("new compiled fn:", trigger(lambda: g(torch.randn(2, 4, device=DEV))))


@case
def compile_error_then_same_fn_new_shape():
    f = torch.compile(lambda t: t.exp() + 1, backend="neuron", dynamic=False)
    print("error:", trigger(lambda: f(torch.randn(0, 3, device=DEV))))
    x = torch.randn(4, 3)
    try:
        y = f(x.to(DEV))
        nm.synchronize()
        print("same fn, good shape:", rel(x.exp() + 1, y))
    except Exception as ex:
        print("same fn, good shape: FAIL", type(ex).__name__, str(ex)[:200].replace("\n", " "))


@case
def eager_lowering_error():
    # lerp_ into a transposed offset view failed to lower in probe_views.py
    def f():
        a = torch.zeros(6, 6, device=DEV)
        b = a.t()[1:4]
        b.lerp_(torch.ones_like(b), 0.5)
    print("error:", trigger(f))
    print(train_check("train after eager lowering error"))


@case
def exception_in_backward_hook():
    w = torch.randn(8, 8, device=DEV, requires_grad=True)
    h = w.register_hook(lambda g: (_ for _ in ()).throw(RuntimeError("boom")))
    print("error:", trigger(lambda: (w @ w).sum().backward()))
    h.remove()
    (w @ w).sum().backward()
    ref = torch.ones(8, 8) @ w.detach().cpu().t() + w.detach().cpu().t() @ torch.ones(8, 8)
    print("retry grad:", rel(ref, w.grad))
    print(train_check("train after hook exception"))


@case
def exception_in_compiled_fn():
    calls = {"n": 0}

    def f(t):
        y = t * 2
        if calls["n"] == 0:
            calls["n"] += 1
            raise ValueError("first call fails")
        return y + 1
    cf = torch.compile(f, backend="neuron", dynamic=False)
    print("error:", trigger(lambda: cf(torch.ones(4, device=DEV))))
    print("retry:", cf(torch.ones(4, device=DEV)).cpu().tolist())
    print(train_check("train after compiled exception"))


@case
def oob_index():
    out = {}
    w = torch.randn(10, 4)
    wn = w.to(DEV)
    for name, fn in [
        ("embedding idx=10", lambda W: F.embedding(torch.tensor([1, 10], device=W.device), W)),
        ("embedding idx=-1", lambda W: F.embedding(torch.tensor([1, -1], device=W.device), W)),
        ("index_select 12", lambda W: W.index_select(0, torch.tensor([12], device=W.device))),
        ("gather 7", lambda W: W.gather(1, torch.tensor([[7]] * 10, device=W.device))),
        ("getitem [11]", lambda W: W[torch.tensor([11], device=W.device)]),
        ("index_add 15", lambda W: W.index_add(0, torch.tensor([15], device=W.device), torch.ones(1, 4, device=W.device))),
        ("scatter 9 in dim1", lambda W: W.scatter(1, torch.tensor([[9]] * 10, device=W.device), 1.0)),
        ("cross_entropy target=4 (C=4)", lambda W: F.cross_entropy(W, torch.full((10,), 4, device=W.device))),
        ("nll_loss target=-5", lambda W: F.nll_loss(W, torch.full((10,), -5, device=W.device))),
        ("one_hot 12 nc=10", lambda W: F.one_hot(torch.tensor([12], device=W.device), 10)),
    ]:
        try:
            fn(w)
            cpu = "ok"
        except Exception as ex:
            cpu = "raises"
        try:
            r = fn(wn)
            nm.synchronize()
            neu = f"no error, returns {r.flatten()[:6].cpu().tolist()}"
        except Exception as ex:
            neu = f"raises {type(ex).__name__}: {str(ex)[:120]}".replace("\n", " ")
        print(f"{name:30s} cpu={cpu:6s} neuron={neu}")
    print(train_check("train after oob"))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(" ".join(CASES))
    else:
        try:
            CASES[sys.argv[1]]()
        except Exception:
            traceback.print_exc()
