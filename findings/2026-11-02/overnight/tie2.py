import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
V, Dm, B, T, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.manual_seed(0)
w = torch.randn(V, Dm, device="neuron", requires_grad=True)
ids = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(0)).to("neuron")
def f(ids, w):
    if mode == "emb_sum": return F.embedding(ids, w).pow(2).sum()
    if mode == "emb_ce": return F.cross_entropy(F.embedding(ids, w), ids[:, 0].repeat(T)[:B*T].view(B, T)[:, 0].repeat(1) if False else (ids[:,0]%Dm)) if False else F.cross_entropy(F.embedding(ids.flatten(), w), ids.flatten() % Dm)
    if mode == "lin_ce": return F.cross_entropy(torch.ones(B*T, Dm, device="neuron") @ w.t(), ids.flatten())
    if mode == "emb_lin_sum": return (F.embedding(ids.flatten(), w) @ w.t()).sum()
    if mode == "emb_lin_ce": return F.cross_entropy(F.embedding(ids.flatten(), w) @ w.t(), ids.flatten())
try:
    l = torch.compile(f, backend="neuron")(ids, w); l.backward(); torch.neuron.synchronize(); print("RES", sys.argv[1:], "OK")
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", sys.argv[1:], "ERROR", msg[i:i+90].replace("\n"," ") if i>0 else msg[:150].replace("\n"," "))
