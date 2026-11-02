import torch, torch_neuronx
def rel(a, b): a, b = a.float().cpu(), b.float().cpu(); return ((a - b).norm() / a.norm()).item()
g = torch.Generator().manual_seed(0)
for dt in [torch.float32, torch.bfloat16]:
    try:
        a = torch.randn(512, 256, generator=g).to(dt); b = torch.randn(4, 256, 384, generator=g).to(dt)
        offs = torch.tensor([128, 256, 384, 512], dtype=torch.int32)
        ref = torch.cat([a[s:e].float() @ b[i].float() for i, (s, e) in enumerate(zip([0, 128, 256, 384], [128, 256, 384, 512]))])
        an, bn = a.to("neuron").requires_grad_(), b.to("neuron").requires_grad_()
        out = torch._grouped_mm(an, bn, offs=offs.to("neuron"))
        out.float().sum().backward()
        ac, bc = a.float().clone().requires_grad_(), b.float().clone().requires_grad_()
        torch.cat([ac[s:e] @ bc[i] for i, (s, e) in enumerate(zip([0, 128, 256, 384], [128, 256, 384, 512]))]).sum().backward()
        print("RES grouped_mm", dt, "fwd rel", f"{rel(ref, out):.3g}", "a.grad", f"{rel(ac.grad, an.grad):.3g}", "b.grad", f"{rel(bc.grad, bn.grad):.3g}")
    except Exception as e:
        print("RES grouped_mm", dt, "ERR", str(e)[:150])
print("RES fallback", sorted(torch_neuronx.get_fallback_ops()))
