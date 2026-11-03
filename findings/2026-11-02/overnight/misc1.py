import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx, traceback
D="neuron"
def t(name, fn):
    try:
        r = fn("cpu"); o = fn(D)
        print(name, "cpu:", r, "\n   neuron:", o)
    except Exception as e:
        print(name, "ERROR", type(e).__name__, str(e)[:400].replace("\n"," "))
which = sys.argv[1]
if which == "convT":
    for g, dil in [(1,1),(2,1),(1,2),(2,2)]:
        def f(dev, g=g, dil=dil):
            torch.manual_seed(0); m = nn.ConvTranspose2d(4, 8, 3, stride=2, groups=g, dilation=dil).to(dev)
            x = torch.randn(2,4,5,5).to(dev).requires_grad_(); o = m(x); o.sum().backward()
            return (tuple(o.shape), round(o.sum().item(),3), tuple(m.weight.grad.shape), round(m.weight.grad.sum().item(),3), round(x.grad.sum().item(),3))
        t(f"convT g={g} dil={dil}", f)
    def f(dev):
        torch.manual_seed(0); m = nn.ConvTranspose2d(4, 8, 3, groups=2).to(dev)
        x = torch.randn(2,4,5,5).to(dev).requires_grad_(); o = m(x); o.sum().backward()
        return (round(o.sum().item(),3), tuple(m.weight.grad.shape), round(m.weight.grad.sum().item(),3))
    t("convT g=2 stride1", f)
    def f(dev):
        torch.manual_seed(0); m = nn.Conv2d(4, 8, 3, groups=2, dilation=2).to(dev)
        x = torch.randn(2,4,9,9).to(dev).requires_grad_(); o = m(x); o.sum().backward()
        return (round(o.sum().item(),3), tuple(m.weight.grad.shape), round(m.weight.grad.sum().item(),3))
    t("conv2d g=2 dil=2", f)
elif which == "gru":
    for kw in [dict(), dict(num_layers=2), dict(bidirectional=True)]:
        def f(dev, kw=kw):
            torch.manual_seed(0); m = nn.GRU(6, 8, batch_first=True, **kw).to(dev)
            x = torch.randn(3,5,6).to(dev); o, h = m(x); return round(o.sum().item(),4)
        t(f"gru {kw}", f)
    def f(dev):
        torch.manual_seed(0); m = nn.GRUCell(6, 8).to(dev); return round(m(torch.randn(3,6).to(dev)).sum().item(),4)
    t("grucell", f)
elif which == "jac":
    from torch.func import jacrev, jacfwd, jvp, hessian
    x0 = torch.randn(4)
    g = lambda x: x.tanh() * x
    f2 = lambda x: (x.sin() * x.cumsum(0)).sum()
    for nm, fn in [("jacrev", lambda x: jacrev(g)(x)), ("jacfwd", lambda x: jacfwd(g)(x)), ("jvp", lambda x: jvp(g,(x,),(torch.ones_like(x),))[1]), ("hessian", lambda x: hessian(f2)(x)), ("jacrev_lin", lambda x: jacrev(lambda a: a*3)(x)), ("jacrev_tanh", lambda x: jacrev(torch.tanh)(x))]:
        t(nm, lambda dev, fn=fn: fn(x0.to(dev)).cpu().sum().item())
elif which == "clipnan":
    def f(dev):
        p = nn.Parameter(torch.ones(4, device=dev)); p.grad = torch.tensor([1.0, float("nan"), 0.0, 1.0], device=dev)
        n = torch.nn.utils.clip_grad_norm_([p], 1.0); return n.item(), p.grad.cpu().tolist()
    t("clip nan", f)
    for fe in [False, True]:
        def f(dev, fe=fe):
            p = nn.Parameter(torch.ones(4, device=dev)); p.grad = torch.tensor([1.0, float("nan"), 0.0, 1.0], device=dev)
            n = torch.nn.utils.clip_grad_norm_([p], 1.0, foreach=fe); return n.item(), p.grad.cpu().tolist()
        t(f"clip nan foreach={fe}", f)
    t("norm of nan", lambda dev: torch.linalg.vector_norm(torch.tensor([1.0, float("nan")], device=dev)).item())
    t("mul nan*0.3", lambda dev: (torch.tensor([1.0, float("nan")], device=dev) * torch.tensor(0.3, device=dev)).cpu().tolist())
    t("clamp nan", lambda dev: torch.tensor([float("nan")], device=dev).clamp(max=1.0).cpu().tolist())
    t("stack norms nan", lambda dev: torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(torch.tensor([1.0, float("nan")], device=dev))])).item())
    t("foreach_norm nan", lambda dev: [x.item() for x in torch._foreach_norm([torch.tensor([1.0, float("nan")], device=dev)])])
elif which == "autocast":
    a, b = torch.randn(4, 8), torch.randn(8, 4)
    for dt in [torch.bfloat16, torch.float16]:
        def f(dev, dt=dt):
            A, B = a.to(dev), b.to(dev)
            with torch.autocast(device_type=dev, dtype=dt):
                r = {"mm": A @ B, "softmax": F.softmax(A @ B, -1), "layer_norm": F.layer_norm(A @ B, (4,)), "log_softmax": F.log_softmax(A@B, -1),
                     "ce": F.cross_entropy(A @ B, torch.tensor([0,1,2,3], device=dev)), "linear": F.linear(A, B.t()),
                     "conv1d": F.conv1d(A.t().unsqueeze(0), B.t().unsqueeze(-1)), "sum": (A@B).sum(), "add": A + 1, "exp": (A@B).exp(),
                     "mse": F.mse_loss(A@B, B[:4]), "bmm": torch.bmm(A[None], B[None]), "baddbmm": torch.baddbmm(A[None,:, :4], A[None], B[None]),
                     "group_norm": F.group_norm(A[None], 2), "sigmoid": torch.sigmoid(A@B), "pow": (A@B).pow(2), "sdpa": F.scaled_dot_product_attention(A[None,None], A[None,None], A[None,None])}
            return {k: str(v.dtype).replace("torch.", "") for k, v in r.items()}
        t(f"autocast {dt}", f)
