import sys, torch, torch.nn as nn, torch_neuronx
D="neuron"
g, dil, st = map(int, sys.argv[1:4])
def f(dev):
    torch.manual_seed(0); m = nn.ConvTranspose2d(4, 8, 3, stride=st, groups=g, dilation=dil).to(dev)
    x = torch.randn(2,4,5,5).to(dev).requires_grad_(); o = m(x); s = o.sum().item()
    o.sum().backward()
    return (tuple(o.shape), round(s,3), tuple(m.weight.grad.shape), round(m.weight.grad.sum().item(),3), round(x.grad.sum().item(),3))
r = f("cpu")
try:
    o = f(D); print(f"g={g} dil={dil} stride={st}", "OK" if o == r else "DIFF", "cpu", r, "neuron", o)
except Exception as e:
    print(f"g={g} dil={dil} stride={st}", "ERROR", str(e)[:200].replace("\n"," "))
