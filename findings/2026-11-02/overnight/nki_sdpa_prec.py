import os, torch, torch.nn.functional as F, torch_neuronx
def rel(a, b): a, b = a.float().cpu(), b.float().cpu(); return ((a - b).norm() / a.norm()).item()
g = torch.Generator().manual_seed(0)
import sys
for L in [int(x) for x in sys.argv[1:]]:
    q, k, v = (torch.randn(1, 3, L, 40, generator=g) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref_bf16in = F.scaled_dot_product_attention(*(t.bfloat16().float() for t in (q, k, v)), is_causal=True)
    ref_bf16 = F.scaled_dot_product_attention(*(t.bfloat16() for t in (q, k, v)), is_causal=True)
    n = F.scaled_dot_product_attention(q.to("neuron"), k.to("neuron"), v.to("neuron"), is_causal=True).cpu()
    print(f"RES NKI_SDPA={os.environ.get('TORCH_NEURONX_ENABLE_NKI_SDPA', '1')} L={L} fp32 in: neuron vs cpu fp32 {rel(ref, n):.3g} | cpu(bf16-rounded inputs) vs cpu fp32 {rel(ref, ref_bf16in):.3g} | cpu bf16 vs cpu fp32 {rel(ref, ref_bf16):.3g} | neuron dtype {n.dtype}")
