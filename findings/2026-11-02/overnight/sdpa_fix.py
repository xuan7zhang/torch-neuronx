import sys, torch, torch.nn.functional as F, torch_neuronx
if sys.argv[1] == "patch":
    import torch_neuronx.python_ops.torch_mlir.ops.scaled_dot_product_attention as M
    import torch_neuronx.python_ops.scaled_dot_product_fused_attention as N
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel
    from torch_neuronx.python_ops.lowering_result import LoweringResult
    def get_lower_func(self):
        k = TorchMlirKernel(N._sdpa_fwd_traceable, self.op_name, static_argnums=(4, 5, 6, 7))
        def f(*a, **kw):
            print("RES patched lower called", flush=True)
            q = a[0] if a else kw.get("query")
            if q.device.type == "meta":
                return LoweringResult(op_name=self.op_name, original_output_specs=[], is_empty_tensor=True, stablehlo_bytes=b"")
            return k.lower(*a, **kw)
        return f
    M.ScaledDotProductAttnMLIRImpl.get_lower_func = get_lower_func
    for nm in ["_sdpa_fwd_traceable", "_sdpa_bwd_traceable"]:
        orig = getattr(N, nm)
        def w(*a, __o=orig, __n=nm, **kw):
            print("RES traced", __n, flush=True); return __o(*a, **kw)
        setattr(N, nm, w)
dev = "neuron"; torch.manual_seed(0); B, H, S = 2, 2, int(sys.argv[2]) if len(sys.argv) > 2 else 16
for causal in [False, True]:
    q = torch.randn(B, H, S, S).to(dev).requires_grad_(); k = torch.randn(B, H, S, S).to(dev).requires_grad_()
    v = torch.eye(S).expand(B, H, S, S).contiguous().to(dev).requires_grad_(); G = torch.randn(B, H, S, S).to(dev)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3, is_causal=causal); (o * G).sum().backward()
    print("RES", sys.argv[1], "causal", causal, "v.grad err", (v.grad - o.detach().transpose(-1, -2) @ G).abs().max().item(), "zero_frac", round((o == 0).float().mean().item(), 2))
