import torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
DEV = "neuron"
def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
res = {}
for dev in ("cpu", DEV):
    torch.manual_seed(0)
    emb = nn.Embedding(100, 32).to(dev)
    ids = torch.randint(0, 100, (4, 10), generator=torch.Generator().manual_seed(0)).to(dev)
    out = {}
    # tied: same tensor used twice
    w = emb.weight.detach().clone().requires_grad_()
    h = F.embedding(ids, w)
    loss = F.cross_entropy((h @ w.t()).flatten(0, 1), ids.flatten())
    loss.backward()
    out["tied_grad"] = w.grad.clone()
    # untied pieces
    w1 = emb.weight.detach().clone().requires_grad_(); w2 = emb.weight.detach().clone().requires_grad_()
    F.cross_entropy((F.embedding(ids, w1) @ w2.t()).flatten(0, 1), ids.flatten()).backward()
    out["emb_part"] = w1.grad.clone(); out["head_part"] = w2.grad.clone()
    out["sum_parts"] = w1.grad + w2.grad
    # only F.linear vs matmul
    w3 = emb.weight.detach().clone().requires_grad_()
    F.cross_entropy(F.linear(F.embedding(ids, w3), w3).flatten(0, 1), ids.flatten()).backward()
    out["tied_linear"] = w3.grad.clone()
    out["loss"] = loss.detach()
    res[dev] = out
for k in res["cpu"]:
    print(f"{k:12s} rel={rel(res['cpu'][k], res[DEV][k]):.3g}")
print("neuron tied - sum_parts:", (res[DEV]["tied_grad"] - res[DEV]["sum_parts"]).abs().max().item())
