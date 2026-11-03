### 🐛 Describe the bug

In eager mode, `F.scaled_dot_product_attention` with `dropout_p > 0` applies dropout in the forward pass, but the backward pass ignores the dropout mask. The q/k/v gradients are wrong. There is no error or warning. `torch.compile` gives correct gradients.

This affects eager training of any model that uses SDPA with attention dropout. Examples: `nn.MultiheadAttention(dropout=0.1)`, `nn.TransformerEncoderLayer`, and HF GPT-2 with the default config (`attn_pdrop=0.1`).

I think the cause is in `python_ops/torch_mlir/ops/scaled_dot_product_attention.py`. With `dropout_p > 0` the NKI path declines, so eager uses the MLIR impls:

- `_mlir_sdpa_fwd_traceable` makes `dropout_mask` with `torch.bernoulli`. But it returns a new `torch.randint(...)` in the `philox_seed` slot, so the mask is lost.
- `_mlir_sdpa_bwd_traceable` always sets `dropout_mask = torch.zeros((1,))`, so the backward never applies dropout.

The other pair in `python_ops/scaled_dot_product_fused_attention.py` (`_sdpa_fwd_traceable` / `_sdpa_bwd_traceable`) returns the keep-mask in the `philox_seed` slot and uses it in the backward. That is why compile is correct. I patched the two MLIR impls in process to use that pair. Then the `v.grad` error dropped to about 5e-6, and the `dropout_p=0` gradients still matched CPU. I can open a PR for this if the approach is OK.

### Reproduction

With `v = I`, the output is the attention matrix after dropout, so the correct `v.grad` is `o^T @ G`.

```python
import torch
import torch.nn.functional as F
import torch_neuronx


def vgrad_err(device, causal):
    torch.manual_seed(0)
    B, H, S = 2, 2, 16
    q = torch.randn(B, H, S, S, device=device, requires_grad=True)
    k = torch.randn(B, H, S, S, device=device, requires_grad=True)
    v = torch.eye(S).expand(B, H, S, S).contiguous().to(device).requires_grad_()
    G = torch.randn(B, H, S, S, device=device)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3, is_causal=causal)
    (o * G).sum().backward()
    return (v.grad - o.detach().transpose(-1, -2) @ G).abs().max().item()


for device in ("cpu", "neuron"):
    for causal in (False, True):
        print(f"{device:6s} causal={causal!s:5s} max |v.grad - o^T @ G| = {vgrad_err(device, causal):.3g}")
```

Output:

```
cpu    causal=False max |v.grad - o^T @ G| = 0
cpu    causal=True  max |v.grad - o^T @ G| = 0
neuron causal=False max |v.grad - o^T @ G| = 1.74
neuron causal=True  max |v.grad - o^T @ G| = 2.93
```

On Neuron, `v.grad` with `dropout_p=0.3` is also identical to `v.grad` with `dropout_p=0`.

### Expected Behavior

The backward uses the same dropout mask as the forward, and the error is about 0, as on CPU and with `torch.compile`.

### Environment

```
torch-neuronx             : 2.14.3.0.1+b1b6c66.dev
torch                     : 2.14.0+cpu
neuronx-cc                : 2.28.7631.0+8f7b0cf5
nki                       : 0.7.0+37713544100.gfede04bb
Neuron driver             : 2.31.7.0
aws-neuronx-runtime-lib   : 2.35.30.0_d2725ccb8-1
aws-neuronx-collectives   : 2.35.43.0_354ad3064-1
aws-neuronx-tools         : 2.33.38.0-f874d5618
instance type             : trn2.3xlarge
logical NeuronCore config : 2
Python                    : 3.14.10
```
