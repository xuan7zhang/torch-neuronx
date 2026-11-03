import torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
from torch.ao.quantization import FakeQuantize, FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver
def run(dev, cls):
    torch.manual_seed(0)
    fq = cls(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8, averaging_constant=0.1).to(dev)
    lin = nn.Linear(8, 8).to(dev)
    outs = []
    for i in range(4):
        x = (torch.randn(16, 8, generator=torch.Generator().manual_seed(i)) * (i + 1)).to(dev).requires_grad_()
        y = fq(lin(x)); y.sum().backward(); outs.append(y.detach().cpu())
    obs = fq.activation_post_process
    return [torch.stack(outs), obs.min_val.cpu().reshape(-1), obs.max_val.cpu().reshape(-1), fq.scale.cpu().reshape(-1), fq.zero_point.cpu().float().reshape(-1), lin.weight.grad.cpu()]
for cls in [FakeQuantize, FusedMovingAvgObsFakeQuantize]:
    try:
        r, n = run("cpu", cls), run("neuron", cls)
        names = ["out", "min_val", "max_val", "scale", "zp", "w.grad"]
        print("RES", cls.__name__, [(nm, round((a - b).abs().max().item(), 5)) for nm, a, b in zip(names, r, n)])
        if (r[1] - n[1]).abs().max() > 1e-4: print("RES   cpu min/max", r[1].tolist(), r[2].tolist(), "neuron", n[1].tolist(), n[2].tolist())
    except Exception as e:
        print("RES", cls.__name__, "ERR", str(e)[:200].replace("\n", " "))
x = torch.randn(64, generator=torch.Generator().manual_seed(0)); 
for dev in ["cpu", "neuron"]:
    xx = x.clone().to(dev); noise = torch.empty(64, device=dev)
    torch.ops.aten.rrelu_with_noise_(xx, noise, 0.1, 0.3, True)
    ok = torch.allclose(xx, x.to(dev) * noise) and bool((noise[x.to(dev) >= 0] == 1).all())
    print("RES rrelu_with_noise_", dev, "consistent", ok)
print("RES fallback", sorted(torch_neuronx.get_fallback_ops()))
