import sys, gc, torch, torch_neuronx
import torch_neuronx.neuron_dynamo_backend.decompositions as D
patched = 0
if sys.argv[1] == "patch":
    sys.path.insert(0, "/home/ec2-user/torch-neuronx/agent_space/overnight/kpatch")
    from patched_add import add_tensor_kernel_fixed
    from torch_neuronx.nki_hop import wrap_nki
    new = wrap_nki(add_tensor_kernel_fixed); old = D._add_tensor_kernel
    for obj in gc.get_objects():
        if isinstance(obj, type(lambda: 0)) and obj.__closure__:
            for c in obj.__closure__:
                try:
                    if c.cell_contents is old: c.cell_contents = new; patched += 1
                except ValueError: pass
torch._dynamo.config.automatic_dynamic_shapes = False
for n in [4, 1000, 200000]:
    torch._dynamo.reset()
    h = torch.ones(n, device="neuron")
    torch.compile(lambda h: torch._foreach_add_([h], [h.clone()], alpha=0.9), backend="neuron")(h)
    g, b = torch.ones(n, device="neuron"), torch.full((n,), 2.0, device="neuron")
    torch.compile(lambda g, b: torch._foreach_add_([g], [b], alpha=0.9), backend="neuron")(g, b)
    print("RES", sys.argv[1], f"cells={patched}", n, "clone case (expect 1.9):", h.cpu().unique().tolist(), " distinct (expect 2.8):", g.cpu().unique().tolist())
