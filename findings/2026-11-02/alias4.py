import torch, torch_neuronx
cases = {
 "foreach_sub_(g, clone(g), alpha=.5)": lambda g: torch._foreach_sub_([g], [g.clone()], alpha=0.5),
 "foreach_div_(g, clone(g))": lambda g: torch._foreach_div_([g], [g.clone()]),
 "foreach_lerp_(g, clone(g), .3)": lambda g: torch._foreach_lerp_([g], [g.clone()], 0.3),
 "foreach_addcmul_(g, clone(g), clone(g))": lambda g: torch._foreach_addcmul_([g], [g.clone()], [g.clone()], value=0.5),
 "foreach_addcdiv_(g, clone(g), clone(g))": lambda g: torch._foreach_addcdiv_([g], [g.clone()], [g.clone()], value=0.5),
}
for nm, f in cases.items():
    r = []
    for dev, comp in [("cpu", False), ("neuron", True)]:
        torch._dynamo.reset(); g = torch.full((8,), 3.0, device=dev)
        (torch.compile(f, backend="neuron") if comp else f)(g); r.append(g.cpu()[0].item())
    print("RES", "OK " if abs(r[0]-r[1]) < 1e-5 else "BAD", f"{nm:42s} cpu={r[0]:.4f} neuron-compile={r[1]:.4f}")
