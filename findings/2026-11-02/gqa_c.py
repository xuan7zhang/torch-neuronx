import sys, contextlib, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
from torch.utils.checkpoint import checkpoint
ck, gqa, amp, comp, causal = (sys.argv[1] == "1"), (sys.argv[2] == "1"), (sys.argv[3] == "1"), (sys.argv[4] == "1"), (sys.argv[5] == "1")
kvh = 2 if gqa else 8
class Attn(nn.Module):
    def __init__(self):
        super().__init__(); torch.manual_seed(0)
        self.q, self.k, self.v, self.o = nn.Linear(64, 64), nn.Linear(64, kvh * 8), nn.Linear(64, kvh * 8), nn.Linear(64, 64)
    def forward(self, x):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, 8, 8).transpose(1, 2); k = self.k(x).view(B, S, kvh, 8).transpose(1, 2); v = self.v(x).view(B, S, kvh, 8).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
        return self.o(o.transpose(1, 2).reshape(B, S, 64))
res = {}
for dev in ("cpu", "neuron"):
    m = Attn().to(dev); x = torch.randn(2, 32, 64, generator=torch.Generator().manual_seed(0)).to(dev).requires_grad_()
    def f(x):
        ctx = torch.autocast(dev, dtype=torch.bfloat16) if amp else contextlib.nullcontext()
        with ctx:
            y = checkpoint(m, x, use_reentrant=False) if ck else m(x)
        return y.float()
    torch._dynamo.reset()
    y = (torch.compile(f, backend="neuron", dynamic=False) if (comp and dev != "cpu") else f)(x)
    y.pow(2).mean().backward()
    res[dev] = {"y": y.detach().float().cpu(), "x.grad": x.grad.float().cpu(), "q.grad": m.q.weight.grad.float().cpu(), "k.grad": m.k.weight.grad.float().cpu(), "v.grad": m.v.weight.grad.float().cpu()}
r = {k: round(((res["cpu"][k] - res["neuron"][k]).norm() / res["cpu"][k].norm()).item(), 4) for k in res["cpu"]}
print("RES", f"ckpt={int(ck)} gqa={int(gqa)} amp={int(amp)} compile={int(comp)} causal={int(causal)}", r)
