import sys, torch, torch.nn.functional as F, torch_neuronx
D="neuron"
def drop(): return (F.dropout(torch.ones(32, device=D), 0.5) > 0).int().cpu()
mode = sys.argv[1]
if mode == "seed_first":   # manual_seed before any neuron op: reproducible across processes?
    torch.manual_seed(7); print("".join(map(str, drop().tolist())))
elif mode == "seed_after_init":
    torch.ones(1, device=D).cpu(); torch.manual_seed(7); print("".join(map(str, drop().tolist())))
elif mode == "restore_twice":
    torch.ones(1, device=D).cpu(); torch.manual_seed(7)
    st = torch.neuron.get_rng_state(); x0 = drop()
    torch.neuron.set_rng_state(st); x1 = drop()
    torch.neuron.set_rng_state(st); x2 = drop()
    print("x0", "".join(map(str,x0.tolist()))); print("x1", "".join(map(str,x1.tolist()))); print("x2", "".join(map(str,x2.tolist())))
    print("offset after draws:", torch.neuron.get_rng_state()[8:].tolist())
elif mode == "mid_stream":
    torch.ones(1, device=D).cpu(); torch.manual_seed(7); drop()
    st = torch.neuron.get_rng_state(); x0 = drop()
    torch.neuron.set_rng_state(st); x1 = drop()
    print("x0", "".join(map(str,x0.tolist()))); print("x1", "".join(map(str,x1.tolist())))
elif mode == "ckpt":
    from torch.utils.checkpoint import checkpoint
    torch.ones(1, device=D).cpu(); torch.manual_seed(0)
    lin = torch.nn.Linear(16, 16).to(D); x = torch.randn(8, 16).to(D).requires_grad_()
    f = lambda x: F.dropout(lin(x), 0.5, training=True)
    for trial in range(2):
        lin.weight.grad = None
        torch.manual_seed(123); o = checkpoint(f, x, use_reentrant=False); o.sum().backward(); g1 = lin.weight.grad.clone()
        lin.weight.grad = None
        torch.manual_seed(123); o2 = f(x); o2.sum().backward(); g2 = lin.weight.grad.clone()
        print("trial", trial, "fwd diff", (o-o2).abs().max().item(), "grad diff", (g1-g2).abs().max().item())
    # analytic check: for y = dropout(lin(x)), dL/dW = mask*scale summed; compare grad with the fwd mask actually used
    lin.weight.grad = None; torch.manual_seed(5)
    o = checkpoint(f, x, use_reentrant=False); o.sum().backward()
    mask = (o != 0).float() * 2.0
    expect = mask.t() @ x.detach()
    print("ckpt grad vs grad implied by fwd mask:", (lin.weight.grad - expect).abs().max().item())
