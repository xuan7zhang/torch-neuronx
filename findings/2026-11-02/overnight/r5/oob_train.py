import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
nm = torch.neuron; D = "neuron"
mode = sys.argv[1]
torch.manual_seed(0)
V = 100
emb = nn.Embedding(V, 8).to(D); head = nn.Linear(8, V).to(D)
ids = torch.tensor([[3, 7, 100, 105, 2]], device=D)  # 100, 105 >= V: a tokenizer added tokens without resize
fwd = lambda i: F.cross_entropy(head(emb(i)).flatten(0, 1), i.clamp(max=V - 1).flatten())
if mode == "compile":
    fwd = torch.compile(fwd, backend="neuron", dynamic=False)
try:
    loss = fwd(ids)
    loss.backward()
    nm.synchronize()
    print(f"RES {mode}: no error, loss={loss.item():.4f}, emb.grad rows touched={int((emb.weight.grad.abs().sum(1) > 0).sum())} (valid ids: 3)")
except Exception as e:
    print(f"RES {mode}: raises {str(e)[:150]}".replace("\n", " "))
