import sys, torch, torch.nn.functional as F, torch_neuronx
nm = torch.neuron; D = "neuron"
op, N, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
x = torch.randn(N, C, generator=torch.Generator().manual_seed(0))
t = torch.randint(0, C, (N,), generator=torch.Generator().manual_seed(1))
t[::4] = C + 3  # 25% bad targets
xn, tn = x.to(D).requires_grad_(), t.to(D)
good = t < C
ref = F.nll_loss(x.log_softmax(1)[good], t[good])
try:
    if op == "nll":
        l = F.nll_loss(xn.log_softmax(1), tn)
    elif op == "gather":
        l = -xn.log_softmax(1).gather(1, tn[:, None]).mean()
    l.backward(); nm.synchronize()
    print(f"RES {op} N={N} C={C}: no error, loss={l.item():.4f} (loss over valid rows only {ref.item():.4f}), grad nonzero rows={int((xn.grad.abs().sum(1) > 0).sum())}/{N}")
except Exception as ex:
    print(f"RES {op} N={N} C={C}: raises ({str(ex)[50:120]})".replace("\n", " "))
