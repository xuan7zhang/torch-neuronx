import torch, torch_neuronx
for dev, comp in [("cpu", False), ("neuron", False), ("neuron", True)]:
    g = torch.ones(4, device=dev)
    f = lambda g: torch._foreach_add_([g], [g], alpha=0.9)
    (torch.compile(f, backend="neuron") if comp else f)(g)
    h = torch.ones(4, device=dev)
    f2 = lambda h: torch._foreach_add_([h], [h.clone()], alpha=0.9)
    (torch.compile(f2, backend="neuron") if comp else f2)(h)
    print("RES", dev, "compile" if comp else "eager", "same tensor:", g.cpu()[0].item(), " clone:", h.cpu()[0].item())
