import sys, torch, torch.nn.functional as F, torch_neuronx
if sys.argv[1] == "patch":
    import torch_neuronx.python_ops.torch_mlir.ops.scaled_dot_product_attention as M
    import torch_neuronx.python_ops.scaled_dot_product_fused_attention as N
    M.ScaledDotProductAttnMLIRImpl.get_lower_func = lambda self: N.ScaledDotProductFusedAttentionNKIImpl.get_lower_func(self)
    M.ScaledDotProductAttnBackwardMLIRImpl.get_lower_func = lambda self: N.ScaledDotProductAttentionBackwardNKIImpl.get_lower_func(self)
dev = "neuron"; torch.manual_seed(0); B, H = 2, 2; S = int(sys.argv[2])
for causal in [False, True]:
    q = torch.randn(B, H, S, S).to(dev).requires_grad_(); k = torch.randn(B, H, S, S).to(dev).requires_grad_()
    v = torch.eye(S).expand(B, H, S, S).contiguous().to(dev).requires_grad_(); G = torch.randn(B, H, S, S).to(dev)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3, is_causal=causal); (o * G).sum().backward()
    print("RES", sys.argv[1], "S", S, "causal", causal, "v.grad err", round((v.grad - o.detach().transpose(-1, -2) @ G).abs().max().item(), 6), "zero_frac", round((o == 0).float().mean().item(), 2))
# dropout 0 still matches CPU?
q = torch.randn(2, 2, S, 8); k = torch.randn(2, 2, S, 8); v = torch.randn(2, 2, S, 8)
qn, kn, vn = (t.to(dev).requires_grad_() for t in (q, k, v)); qc, kc, vc = (t.clone().requires_grad_() for t in (q, k, v))
F.scaled_dot_product_attention(qn, kn, vn, is_causal=True).sum().backward(); F.scaled_dot_product_attention(qc, kc, vc, is_causal=True).sum().backward()
print("RES", sys.argv[1], "no-dropout grads vs cpu", (qn.grad.cpu() - qc.grad).abs().max().item(), (vn.grad.cpu() - vc.grad).abs().max().item())
