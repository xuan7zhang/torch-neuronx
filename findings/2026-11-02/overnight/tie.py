import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
mode = sys.argv[1]
torch.manual_seed(0)
emb = nn.Embedding(50, 16).to("neuron")
head = nn.Linear(16, 50, bias=False).to("neuron")
if mode != "untied": head.weight = emb.weight
ids = torch.randint(0, 50, (4, 9), generator=torch.Generator().manual_seed(0)).to("neuron")
params = list({id(p): p for p in list(emb.parameters()) + list(head.parameters())}.values())
opt = torch.optim.AdamW(params, lr=1e-2)
def fwdbwd(ids):
    logits = head(emb(ids[:, :-1]))
    loss = F.cross_entropy(logits.reshape(-1, 50), ids[:, 1:].reshape(-1))
    loss.backward(); return loss.detach()
def step(ids):
    opt.zero_grad(); l = fwdbwd(ids); opt.step(); return l
def fwd_only(ids):
    return head(emb(ids[:, :-1])).sum()
def emb_only(ids):
    w = emb.weight
    return F.embedding(ids, w).sum() + (w @ w.t()).sum()
fn = {"untied": step, "tied": step, "tied_fwdbwd": fwdbwd, "tied_fwd": fwd_only, "emb_matmul_bwd": None}[mode] if mode != "emb_matmul_bwd" else None
try:
    if mode == "emb_matmul_bwd":
        w = torch.randn(50, 16, device="neuron", requires_grad=True)
        f = torch.compile(lambda ids, w: (F.embedding(ids, w).sum() + (torch.randn(3, 16, device="neuron") @ w.t()).pow(2).sum()), backend="neuron")
        f(ids, w).backward(); torch.neuron.synchronize(); print("RES", mode, "OK")
    else:
        c = torch.compile(fn, backend="neuron")
        for _ in range(2): r = c(ids)
        print("RES", mode, "OK", r.item())
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", mode, "ERROR", msg[i-20:i+150].replace("\n"," ") if i>0 else msg[:200])
