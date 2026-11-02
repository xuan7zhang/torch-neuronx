import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
mode = sys.argv[1]
torch.manual_seed(0)
w = torch.randn(50, 16, device="neuron", requires_grad=True)
hw = torch.randn(50, 16, device="neuron", requires_grad=True)
ids = torch.randint(0, 50, (4, 9), generator=torch.Generator().manual_seed(0)).to("neuron")
def f(ids):
    x = ids[:, :-1] if "inslice" in mode else ids[:, :8]
    t = ids[:, 1:] if "tslice" in mode else ids[:, :8]
    logits = F.linear(F.embedding(x, w), hw)
    if "ce" in mode: return F.cross_entropy(logits.reshape(-1, 50), t.reshape(-1))
    return logits.sum()
try:
    l = torch.compile(f, backend="neuron")(ids); l.backward(); torch.neuron.synchronize(); print("RES", mode, "OK")
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", mode, "ERROR", msg[i:i+70].replace("\n"," ") if i>0 else msg[:150].replace("\n"," "))
