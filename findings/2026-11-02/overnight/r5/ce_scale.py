import torch, torch.nn.functional as F, torch_neuronx
DEV = "neuron"
def rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return ((a - b).norm() / (a.norm() + 1e-30)).item()
g = torch.Generator().manual_seed(0)
for N, V in [(40, 100), (512, 32000)]:
    base = torch.randn(N, V, generator=g)
    y = torch.randint(0, V, (N,), generator=g)
    for s in (1, 5, 10, 20, 40, 80):
        x = base * s
        x64 = x.double().requires_grad_()
        l64 = F.cross_entropy(x64, y); l64.backward()
        xc = x.clone().requires_grad_(); lc = F.cross_entropy(xc, y); lc.backward()
        xn = x.to(DEV).requires_grad_(); ln = F.cross_entropy(xn, y.to(DEV)); ln.backward()
        lsm = rel(F.log_softmax(x64, -1), F.log_softmax(x.to(DEV), -1))
        sm = rel(F.softmax(x64, -1), F.softmax(x.to(DEV), -1))
        lse = rel(x64.logsumexp(-1), x.to(DEV).logsumexp(-1))
        print(f"N={N} V={V} scale={s:3d} loss={l64.item():9.4f}  loss rel: cpu {rel(l64, lc):.1e} neuron {rel(l64, ln):.1e}"
              f"  grad rel: cpu {rel(x64.grad, xc.grad):.1e} neuron {rel(x64.grad, xn.grad):.1e}"
              f"  log_softmax {lsm:.1e} softmax {sm:.1e} logsumexp {lse:.1e}", flush=True)
