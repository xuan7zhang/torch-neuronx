import sys, torch, torch.nn.functional as F, torch_neuronx
B, C, L = map(int, sys.argv[1:4]); comp = sys.argv[4] == "c"
xx = (torch.arange(B * C * L, dtype=torch.float32).reshape(B, C, L) / 100).to(torch.bfloat16)
f = lambda t: F.adaptive_avg_pool2d(t.unsqueeze(2), (1, 7)).squeeze(2)
r = f(xx).float()
try:
    g = (torch.compile(f, backend="neuron", dynamic=False) if comp else f)(xx.to("neuron")).float().cpu()
    print("RES", sys.argv[1:], "rel", f"{((r - g).norm() / r.norm()).item():.3g}")
except Exception as e:
    m = str(e); i = m.find("NCC_"); print("RES", sys.argv[1:], "ERR", (m[i:i+200] if i > 0 else m[:200]).replace("\n", " "))
