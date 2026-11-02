import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
kw = eval(sys.argv[1]); whole = sys.argv[2] == "step"
def run(dev):
    torch.manual_seed(0); m = nn.Linear(4, 3).to(dev)
    opt = torch.optim.SGD(m.parameters(), **kw)
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(1)).to(dev)
    def step():
        opt.zero_grad(); l = m(x).pow(2).mean(); l.backward(); opt.step(); return l.detach()
    def ostep(): opt.step()
    if dev != "cpu":
        if whole: step = torch.compile(step, backend="neuron")
        else:
            co = torch.compile(ostep, backend="neuron")
            def step():
                opt.zero_grad(); l = m(x).pow(2).mean(); l.backward(); co(); return l.detach()
    ls = [step().item() for _ in range(4)]
    return ls, m.weight.detach().cpu()
lc, wc = run("cpu"); ln, wn = run("neuron")
print(f"RES {sys.argv[1]:55s} {sys.argv[2]:5s} loss cpu {[round(v,5) for v in lc]} neuron {[round(v,5) for v in ln]} wdiff {(wc-wn).abs().max().item():.3g}")
