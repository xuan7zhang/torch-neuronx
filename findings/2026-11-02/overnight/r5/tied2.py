import torch, torch.nn.functional as F, torch_neuronx
DEV = "neuron"
def rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return ((a - b).norm() / (a.norm() + 1e-30)).item()
torch.manual_seed(0)
w = torch.randn(100, 32)
ids = torch.randint(0, 100, (4, 10), generator=torch.Generator().manual_seed(0))
wn, idn = w.to(DEV), ids.to(DEV)
h, hn = F.embedding(ids, w), F.embedding(idn, wn)
print("embedding", rel(h, hn))
ref = h.double() @ w.double().t()
print("h@w.t() cpu vs fp64", rel(ref, h @ w.t()), " neuron vs fp64", rel(ref, hn @ wn.t()))
print("same, contiguous w.t()", rel(ref, hn @ wn.t().contiguous()))
print("F.linear", rel(ref, F.linear(hn, wn)))
a, b = torch.randn(40, 32), torch.randn(32, 100)
print("random 40x32 @ 32x100", rel(a.double() @ b.double(), a.to(DEV) @ b.to(DEV)))
print("random scaled x5", rel((a*5).double() @ (b*5).double(), (a*5).to(DEV) @ (b*5).to(DEV)))
# magnitude of logits
print("logit absmax", ref.abs().max().item())
lg = ref.float().flatten(0, 1)
print("CE on same logits", rel(F.cross_entropy(lg, ids.flatten()), F.cross_entropy(lg.to(DEV), idn.flatten())))
for shp in [(40, 32, 100), (40, 32, 128), (64, 32, 100), (40, 64, 100), (40, 128, 100), (128, 32, 100)]:
    m, k, n = shp
    a, b = torch.randn(m, k), torch.randn(k, n)
    print("mm", shp, f"{rel(a.double() @ b.double(), a.to(DEV) @ b.to(DEV)):.2g}")
