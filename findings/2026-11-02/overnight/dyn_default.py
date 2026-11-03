import sys, torch, torch.nn as nn, torch_neuronx
kw = eval(sys.argv[1])
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16)).to("neuron")
f = torch.compile(m, backend="neuron", **kw)
out = []
for L in [8, 12, 16, 12]:
    try:
        x = torch.randn(2, L, 16).to("neuron")
        l = f(x).pow(2).mean(); l.backward(); torch.neuron.synchronize(); out.append(f"L={L}:ok")
    except Exception as e:
        out.append(f"L={L}:ERR {type(e).__name__}: {str(e)[:110]}".replace("\n", " ")); break
print("RES", sys.argv[1], out)
