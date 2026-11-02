import torch, torch.nn.functional as F, torch_neuronx
dev = "neuron"; torch.manual_seed(0); B, H, S = 1, 2, 16
q = torch.randn(B, H, S, S).to(dev).requires_grad_(); k = torch.randn(B, H, S, S).to(dev).requires_grad_()
v = torch.eye(S).expand(B, H, S, S).contiguous().to(dev).requires_grad_(); G = torch.randn(B, H, S, S).to(dev)
o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3); (o * G).sum().backward(); g_drop = v.grad.clone(); qg_drop = q.grad.clone()
v.grad = None; q.grad = None
o0 = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0); (o0 * G).sum().backward(); g_nodrop = v.grad.clone(); qg_nodrop = q.grad.clone()
expect = o.detach().transpose(-1, -2) @ G
print("RES v.grad(drop) vs fwd-mask expectation:", (g_drop - expect).abs().max().item())
print("RES v.grad(drop) vs no-dropout v.grad:", (g_drop - g_nodrop).abs().max().item())
print("RES v.grad(drop) vs no-dropout v.grad / 0.7:", (g_drop - g_nodrop / 0.7).abs().max().item())
print("RES q.grad(drop) vs no-dropout q.grad:", (qg_drop - qg_nodrop).abs().max().item())
r = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(q.detach(), k.detach(), v.detach(), None, 0.3)
print("RES overrideable outputs:", [(tuple(t.shape), str(t.dtype)) if isinstance(t, torch.Tensor) else t for t in r])
