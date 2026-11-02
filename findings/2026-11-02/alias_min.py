import torch, torch_neuronx
def f1(g):
    b = torch.clone(g).detach()
    torch._foreach_add_([g], [b], alpha=0.9)
    return b
def f2(g):
    b = torch.clone(g)
    g.add_(b, alpha=0.9)
    return b
def f3(g):
    b = [torch.clone(t) for t in g]
    torch._foreach_add_(g, b, alpha=0.9)
    return b
def f4(g):  # no alpha
    b = torch.clone(g)
    torch._foreach_add_([g], [b])
    return b
def f5(g):
    b = torch.clone(g)
    torch._foreach_mul_([b], 0.5)
    torch._foreach_add_([g], [b], alpha=0.9)
    return b
for nm, f in [("foreach_add_ alpha", f1), ("add_ alpha", f2), ("foreach list", f3), ("foreach_add_ no alpha", f4), ("mul_ then add_", f5)]:
    res = []
    for dev, comp in [("cpu", False), ("neuron", True)]:
        g = torch.ones(4, device=dev) if nm != "foreach list" else [torch.ones(4, device=dev), torch.full((3,), 2.0, device=dev)]
        fn = torch.compile(f, backend="neuron") if comp else f
        b = fn(g)
        gl = g if isinstance(g, list) else [g]; bl = b if isinstance(b, list) else [b]
        res.append(([t.cpu()[0].item() for t in gl], [t.cpu()[0].item() for t in bl]))
    print(("OK  " if res[0] == res[1] else "BAD "), f"{nm:24s} cpu g,b={res[0]}  neuron-compile g,b={res[1]}")
