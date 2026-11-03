import torch, torch.nn as nn, torch_neuronx
def f(dev, comp):
    torch.manual_seed(0); m = nn.ConvTranspose2d(4, 8, 3, stride=2, groups=2).to(dev)
    x = torch.randn(2,4,5,5).to(dev).requires_grad_()
    fn = torch.compile(m, backend="neuron") if comp else m
    o = fn(x); o.sum().backward()
    return round(o.sum().item(),3), tuple(m.weight.grad.shape), round(m.weight.grad.sum().item(),3), round(x.grad.sum().item(),3)
print("cpu", f("cpu", False))
try: print("neuron compile", f("neuron", True))
except Exception as e: print("compile ERROR", str(e)[:300])
