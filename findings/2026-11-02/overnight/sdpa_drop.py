import sys, torch, torch.nn.functional as F, torch_neuronx
dev = sys.argv[1]; comp = len(sys.argv) > 2
torch.manual_seed(0)
B, H, S = 2, 2, 16
for causal in [False, True]:
    for dt in [torch.float32, torch.bfloat16]:
        q = torch.randn(B, H, S, S).to(dev, dt).requires_grad_()
        k = torch.randn(B, H, S, S).to(dev, dt).requires_grad_()
        v = torch.eye(S).expand(B, H, S, S).contiguous().to(dev, dt).requires_grad_()
        G = torch.randn(B, H, S, S).to(dev, dt)
        f = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0.3, is_causal=causal)
        if comp: f = torch.compile(f, backend="neuron", dynamic=False)
        o = f(q, k, v)
        (o * G).sum().backward()
        expect = o.detach().float().transpose(-1, -2) @ G.float()
        err = (v.grad.float() - expect).abs().max().item(); scale = expect.abs().max().item()
        zero_frac = (o.detach() == 0).float().mean().item()
        print(f"RES {dev}{' compile' if comp else ''} causal={causal} {str(dt)[6:]:9s} v.grad err={err:.3g} (|expect|max={scale:.3g}) zero_frac={zero_frac:.2f}")
for nm, fn in [("dropout2d", lambda x: F.dropout2d(x, 0.4)), ("dropout1d", lambda x: F.dropout1d(x, 0.4)), ("alpha_dropout", lambda x: F.alpha_dropout(x, 0.4, True)), ("feature_alpha_dropout", lambda x: F.feature_alpha_dropout(x, 0.4, True)), ("dropout3d", lambda x: F.dropout3d(x.unsqueeze(-1), 0.4).squeeze(-1))]:
    x = torch.randn(4, 8, 6).to(dev).requires_grad_()
    y = fn(x); y.backward(torch.ones_like(y))
    # for these dropouts y = a*x*mask + b*(1-mask) (alpha) or x*mask*scale; grad = d y/d x = mask*scale -> y_grad_ratio consistent:
    if "alpha" in nm:
        keep = (x.grad != 0)
        a = x.grad[keep].unique()
        ok = len(a) == 1
    else:
        ok = torch.allclose(x.grad * x.detach(), y.detach(), atol=1e-5)
    print(f"RES {dev} {nm:22s} fwd/bwd mask consistent={ok}")
