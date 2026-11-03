"""Prototype: bounds-check the nll_loss target like cross_entropy does. No file change."""
import torch, torch.nn.functional as F, torch_neuronx
from torch_neuronx.python_ops.nki_kernels.bound_check import assert_index_in_bounds
nm = torch.neuron; D = "neuron"
N, C = 512, 32000
x = torch.randn(N, C, generator=torch.Generator().manual_seed(0))
for bad in (False, True):
    t = torch.randint(0, C, (N,), generator=torch.Generator().manual_seed(1))
    if bad:
        t[::4] = C + 3
    t[1] = -100  # ignore_index stays allowed
    xn, tn = x.to(D), t.to(D)
    try:
        safe = torch.where(tn == -100, torch.zeros_like(tn), tn)
        assert_index_in_bounds(safe, C, lo=0)
        l = F.nll_loss(xn.log_softmax(1), tn)
        nm.synchronize()
        ref = F.nll_loss(x.log_softmax(1), t) if not bad else None
        print(f"RES bad={bad}: passes, loss {l.item():.5f} cpu {ref.item() if ref is not None else '-'}")
    except Exception as e:
        print(f"RES bad={bad}: raises {str(e)[:130]}".replace("\n", " "))
