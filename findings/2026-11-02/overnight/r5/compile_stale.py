import torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
DEV = "neuron"
def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
cc = lambda f: torch.compile(f, backend="neuron", dynamic=False)
x = torch.randn(8, 16)
out = {}
# 1. python constant changes between calls
c = 2.0
f = cc(lambda t: t * c + 1)
r1 = f(x.to(DEV)); c = 5.0; r2 = f(x.to(DEV))
out["const_change"] = max(rel(x * 2 + 1, r1), rel(x * 5 + 1, r2))
# 2. module attribute change (LeakyReLU slope, dropout p in eval)
m = nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU(0.1)).to(DEV); mc = cc(m)
y1 = mc(x.to(DEV)); m[1].negative_slope = 0.5; y2 = mc(x.to(DEV))
mcpu = nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU(0.1)); mcpu.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
mcpu[1].negative_slope = 0.1; e1 = rel(mcpu(x), y1); mcpu[1].negative_slope = 0.5
out["module_attr_change"] = max(e1, rel(mcpu(x), y2))
# 3. freeze a layer mid-training
res = {}
for dev in ("cpu", DEV):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev)
    fn = cc(net) if dev != "cpu" else net
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for i in range(4):
        if i == 2:
            net[0].requires_grad_(False)
        opt.zero_grad()
        fn(x.to(dev)).pow(2).mean().backward()
        opt.step()
    res[dev] = ([p.detach() for p in net.parameters()], net[0].weight.grad)
out["freeze_mid"] = max(rel(a, b) for a, b in zip(res["cpu"][0], res[DEV][0]))
out["frozen_grad_none_or_stale_equal"] = float((res[DEV][1] is None) == (res["cpu"][1] is None))
# 4. in-place param replacement via .data after compile
net = nn.Linear(16, 4).to(DEV); fn = cc(net); fn(x.to(DEV))
w = torch.randn(4, 16); net.weight.data = w.to(DEV)
out["param_data_swap"] = rel(F.linear(x, w, net.bias.detach().cpu()), fn(x.to(DEV)))
net.weight.data.copy_(w * 3)
out["param_inplace_copy"] = rel(F.linear(x, w * 3, net.bias.detach().cpu()), fn(x.to(DEV)))
# 5. buffer replaced with a new tensor
class B(nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer("scale", torch.tensor(2.0))
    def forward(s, t): return t * s.scale
b = B().to(DEV); fb = cc(b); fb(x.to(DEV)); b.scale = torch.tensor(7.0, device=DEV)
out["buffer_reassign"] = rel(x * 7, fb(x.to(DEV)))
# 6. same graph, different dtype input
f2 = cc(lambda t: (t * 3).sum(0))
f2(x.to(DEV)); out["dtype_switch_bf16"] = rel((x.bfloat16() * 3).sum(0), f2(x.bfloat16().to(DEV)))
# 7. non-contiguous input after contiguous
f3 = cc(lambda t: t * 2 + t.sum(1, keepdim=True))
sq = torch.randn(16, 16); f3(sq.to(DEV))
out["stride_switch"] = rel(sq.t() * 2 + sq.t().sum(1, keepdim=True), f3(sq.to(DEV).t()))
for k, v in out.items():
    print(f"{'OK ' if (v < 1e-4 if 'none' not in k else v == 1.0) else 'BAD'} {k:32s} {v:.3g}")
