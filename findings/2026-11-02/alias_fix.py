import sys, torch, torch_neuronx
if sys.argv[1] == "patch":
    sys.path.insert(0, "/home/ec2-user/torch-neuronx/agent_space/overnight/kpatch")
    from patched_add import add_tensor_kernel_fixed
    import torch_neuronx.python_ops.nki_kernels.foreach_elementwise as fe
    from torch_neuronx.nki_hop import wrap_nki
    fe._add_tensor_kernel = wrap_nki(add_tensor_kernel_fixed)
for n in [4, 1000, 200000]:
    h = torch.ones(n, device="neuron")
    torch.compile(lambda h: torch._foreach_add_([h], [h.clone()], alpha=0.9), backend="neuron")(h)
    g, b = torch.ones(n, device="neuron"), torch.full((n,), 2.0, device="neuron")
    torch.compile(lambda g, b: torch._foreach_add_([g], [b], alpha=0.9), backend="neuron")(g, b)
    print("RES", sys.argv[1], n, "clone case (expect 1.9):", h.cpu().unique().tolist(), " distinct (expect 2.8):", g.cpu().unique().tolist())
