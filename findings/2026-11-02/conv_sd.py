import sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
kind = sys.argv[1]
try:
    if kind.startswith("conv"):
        nd, st, dil = int(kind[4]), int(sys.argv[2]), int(sys.argv[3])
        C = {1: nn.Conv1d, 2: nn.Conv2d}[nd]
        m = C(4, 8, 3, stride=st, dilation=dil, padding=1).to("neuron")
        x = torch.randn(2, 4, *([16] * nd)).to("neuron").requires_grad_()
        y = m(x); torch.neuron.synchronize(); fwd = "fwd ok"
        y.sum().backward(); torch.neuron.synchronize(); print("RES", kind, f"stride={st} dil={dil}", fwd, "bwd ok")
    else:
        x = torch.randn(2, 3); t = torch.softmax(torch.randn(2, 3), -1)
        for dev in ["cpu", "neuron"]:
            xx = x.clone().to(dev).requires_grad_(); tt = t.clone().to(dev).requires_grad_()
            F.cross_entropy(xx, tt).backward()
            print("RES ce soft target", dev, "target.grad:", None if tt.grad is None else [round(v, 4) for v in tt.grad.cpu().flatten().tolist()])
except Exception as e:
    m = str(e); i = m.find("NCC_"); print("RES", kind, sys.argv[2:], "ERR", (m[i:i+120] if i > 0 else m[:120]).replace("\n", " "))
