import torch, torch_neuronx
def f1(g):
    b = torch.clone(g)
    torch._foreach_add_([g], [b], alpha=0.9)
    return b
g = torch.ones(4, device="neuron")
b = torch.compile(f1, backend="neuron")(g)
print("RESULT g", g.cpu().tolist(), "b", b.cpu().tolist())
