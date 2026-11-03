import torch, torch.nn.functional as F, torch_neuronx
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_map, tree_flatten
names = {"addbmm": None, "nn.functional.interpolate.area": None, "logcumsumexp": None, "kron": None, "nn.functional.grid_sample": None, "grid_sampler_2d": None, "grid_sampler_3d": None}
def run(op, s, dev, dt):
    cv = lambda x: (x.detach().to(dt).to(dev).requires_grad_(x.requires_grad) if x.is_floating_point() else x.to(dev)) if isinstance(x, torch.Tensor) else x
    inp, args, kw = tree_map(cv, s.input), tree_map(cv, s.args), tree_map(cv, s.kwargs)
    out = op(inp, *args, **kw); outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor) and t.requires_grad]
    leaves = [t for t in tree_flatten((inp, args, kw))[0] if isinstance(t, torch.Tensor) and t.requires_grad]
    g = torch.Generator().manual_seed(0); cots = [torch.randn(t.shape, generator=g).to(dt).to(dev) for t in outs]
    gr = torch.autograd.grad(outs, leaves, cots, allow_unused=True)
    return [t.detach().float().cpu() for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)], [x.float().cpu() if x is not None else None for x in gr]
def rel(a, b): return max(((x - y).norm() / (x.norm() + 1e-12)).item() for x, y in zip(a, b) if x is not None and y is not None)
for o in op_db:
    n = o.name + ("." + o.variant_test_name if o.variant_test_name else "")
    if n not in names: continue
    for i, s in enumerate(list(o.sample_inputs("cpu", torch.float32, requires_grad=True))[:3]):
        rf = run(o, s, "cpu", torch.float32); rb = run(o, s, "cpu", torch.bfloat16); nb = run(o, s, "neuron", torch.bfloat16)
        print(f"RES {n:32s} s{i} rel-err vs cpu fp32: fwd cpu_bf16={rel(rf[0], rb[0]):.3g} neuron_bf16={rel(rf[0], nb[0]):.3g} | grad cpu_bf16={rel(rf[1], rb[1]):.3g} neuron_bf16={rel(rf[1], nb[1]):.3g}")
