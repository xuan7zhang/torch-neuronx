import torch, torch_neuronx
D = "neuron"
def run(nm, f, *mk):
    res = []
    for dev, comp in [("cpu", False), (D, True)]:
        args = [m(dev) for m in mk]
        out = (torch.compile(f, backend="neuron") if comp else f)(*args)
        res.append([a.cpu().flatten()[:2].tolist() for a in args])
    print(("OK  " if res[0] == res[1] else "BAD "), f"{nm:34s} cpu={res[0]} neuron={res[1]}")
ones = lambda d: torch.ones(4, device=d)
twos = lambda d: torch.full((4,), 2.0, device=d)
run("two inputs g=1,b=2 alpha .9", lambda g, b: torch._foreach_add_([g], [b], alpha=0.9), ones, twos)
run("two inputs equal values", lambda g, b: torch._foreach_add_([g], [b], alpha=0.9), ones, ones)
run("b = g*1.0", lambda g: torch._foreach_add_([g], [g * 1.0], alpha=0.9), ones)
run("b = g+0", lambda g: torch._foreach_add_([g], [g + 0.0], alpha=0.9), ones)
run("b = clone(g) alpha 2", lambda g: torch._foreach_add_([g], [g.clone()], alpha=2.0), ones)
run("g=2 b=clone(g) alpha .9", lambda g: torch._foreach_add_([g], [g.clone()], alpha=0.9), twos)
run("out-of-place foreach_add", lambda g: g.copy_(torch._foreach_add([g], [g.clone()], alpha=0.9)[0]), ones)
run("foreach_add_ alpha=1", lambda g, b: torch._foreach_add_([g], [b], alpha=1.0), ones, twos)
run("foreach_sub_ alpha .9", lambda g, b: torch._foreach_sub_([g], [b], alpha=0.9), ones, twos)
run("foreach_add_ len 1000", lambda g, b: torch._foreach_add_([g], [b], alpha=0.9), lambda d: torch.ones(1000, device=d), lambda d: torch.full((1000,), 2.0, device=d))
