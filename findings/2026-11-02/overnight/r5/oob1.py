import sys, torch, torch.nn.functional as F, torch_neuronx
nm = torch.neuron; D = "neuron"
w = torch.arange(40, dtype=torch.float32).view(10, 4) + 1  # rows 1..40, no zeros
wn = w.to(D)
L = {
 "embedding_10": lambda W: F.embedding(torch.tensor([1, 10], device=W.device), W),
 "embedding_1000": lambda W: F.embedding(torch.tensor([1, 1000], device=W.device), W),
 "embedding_neg1": lambda W: F.embedding(torch.tensor([1, -1], device=W.device), W),
 "gather_7": lambda W: W.gather(1, torch.full((10, 1), 7, device=W.device)),
 "index_add_15": lambda W: W.index_add(0, torch.tensor([15], device=W.device), torch.ones(1, 4, device=W.device)),
 "nll_target_-5": lambda W: F.nll_loss(W.log_softmax(1), torch.full((10,), -5, device=W.device)),
 "nll_target_7": lambda W: F.nll_loss(W.log_softmax(1), torch.full((10,), 7, device=W.device)),
 "scatter_add_12": lambda W: torch.zeros(10, 4, device=W.device).scatter_add(0, torch.full((1, 4), 12, device=W.device), W[:1]),
 "index_put_12": lambda W: W.index_put((torch.tensor([12], device=W.device),), torch.ones(4, device=W.device)),
 "take_100": lambda W: W.take(torch.tensor([100], device=W.device)),
}
name = sys.argv[1]
try:
    L[name](w); cpu = "ok"
except Exception as e:
    cpu = "raises"
try:
    r = L[name](wn); nm.synchronize(); res = f"no error -> {r.flatten()[:8].cpu().tolist()}"
except Exception as e:
    res = f"raises {str(e)[:90]}".replace("\n", " ")
try:
    a = torch.ones(3, device=D) * 2; nm.synchronize(); nxt = f"next op ok {a.cpu().tolist()}"
except Exception as e:
    nxt = f"next op FAILS {str(e)[:60]}".replace("\n", " ")
print(f"RES {name:16s} cpu={cpu:6s} neuron: {res} | {nxt}")
