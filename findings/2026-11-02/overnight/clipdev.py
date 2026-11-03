import torch, torch.nn as nn, torch_neuronx
p = nn.Parameter(torch.randn(4, 6).to("neuron")); p.grad = torch.ones(4, 6, device="neuron")
for fe in [None, True, False]:
    n = torch.nn.utils.clip_grad_norm_([p], 1.0, foreach=fe); print("RES clip foreach", fe, "->", n.device)
print("RES _foreach_norm ->", [t.device for t in torch._foreach_norm([p.grad])])
print("RES vector_norm ->", torch.linalg.vector_norm(p.grad).device)
print("RES stack of norms ->", torch.stack(torch._foreach_norm([p.grad])).device)
print("RES get_total_norm ->", torch.nn.utils.get_total_norm([p.grad]).device)
print("RES fallback", sorted(torch_neuronx.get_fallback_ops()))
