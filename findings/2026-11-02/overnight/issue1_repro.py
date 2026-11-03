import torch
import torch.nn.functional as F
import torch_neuronx


def vgrad_err(device, causal):
    torch.manual_seed(0)
    B, H, S = 2, 2, 16
    q = torch.randn(B, H, S, S, device=device, requires_grad=True)
    k = torch.randn(B, H, S, S, device=device, requires_grad=True)
    # With v = I the output is the dropped attention matrix, so v.grad must be o^T @ G.
    v = torch.eye(S).expand(B, H, S, S).contiguous().to(device).requires_grad_()
    G = torch.randn(B, H, S, S, device=device)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3, is_causal=causal)
    (o * G).sum().backward()
    return (v.grad - o.detach().transpose(-1, -2) @ G).abs().max().item()


for device in ("cpu", "neuron"):
    for causal in (False, True):
        print(f"{device:6s} causal={causal!s:5s} max |v.grad - o^T @ G| = {vgrad_err(device, causal):.3g}")
