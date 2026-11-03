import sys, types, torch, torch.nn as nn, torch.nn.functional as TF, torch_neuronx
import torch_neuronx.neuron_dynamo_backend.decompositions as D
def dilate_w(w, dil):
    nd = w.dim() - 2; dil = (dil,) * nd if isinstance(dil, int) else tuple(dil)
    for i in range(nd):
        d = dil[i]
        if d == 1: continue
        ax = 2 + i; k = w.shape[ax]
        z = torch.zeros_like(w)
        w = torch.stack([w] + [z] * (d - 1), dim=ax + 1).flatten(ax, ax + 1)
        w = w.narrow(ax, 0, (k - 1) * d + 1)
    return w
def wrap_t(fn):
    def g(inp, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        st = stride if isinstance(stride, (tuple, list)) else (stride,)
        dl = dilation if isinstance(dilation, (tuple, list)) else (dilation,)
        if any(s > 1 for s in st) and any(d > 1 for d in dl):
            return fn(inp, dilate_w(weight, dilation), bias, stride, padding, output_padding, groups, 1)
        return fn(inp, weight, bias, stride, padding, output_padding, groups, dilation)
    return g
if sys.argv[1] == "patch":
    shim = types.SimpleNamespace(**{k: getattr(TF, k) for k in dir(TF) if not k.startswith("__")})
    for n in ["conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]: setattr(shim, n, wrap_t(getattr(TF, n)))
    D.F = shim
for nd, st, dil in [(2, 2, 2), (1, 3, 2), (2, 2, 3)]:
    C = {1: nn.Conv1d, 2: nn.Conv2d}[nd]
    torch.manual_seed(0); mc = C(4, 8, 3, stride=st, dilation=dil, padding=1); mn = C(4, 8, 3, stride=st, dilation=dil, padding=1).to("neuron"); mn.load_state_dict(mc.state_dict())
    x = torch.randn(2, 4, *([int(sys.argv[2])] * nd)); xc = x.clone().requires_grad_(); xn = x.clone().to("neuron").requires_grad_()
    try:
        mc(xc).pow(2).sum().backward(); mn(xn).pow(2).sum().backward(); torch.neuron.synchronize()
        print("RES", sys.argv[1], f"conv{nd}d stride={st} dil={dil}", "x.grad diff", f"{(xc.grad - xn.grad.cpu()).abs().max().item():.2e}", "w.grad diff", f"{(mc.weight.grad - mn.weight.grad.cpu()).abs().max().item():.2e}")
    except Exception as e:
        m = str(e); i = m.find("NCC_"); print("RES", sys.argv[1], f"conv{nd}d stride={st} dil={dil}", "ERR", (m[i:i+60] if i > 0 else m[:100]).replace("\n", " "))
