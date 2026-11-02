"""Prototype: make torch.neuron get/set_rng_state save and restore the on-device GPSIMD RNG state."""

import sys

import torch
import torch.nn.functional as F
import torch_neuronx
from torch.utils.checkpoint import checkpoint

D = "neuron"
torch.ones(1, device=D).cpu()

if sys.argv[1] == "patch":
    import torch_neuronx.neuron as N

    orig_get, orig_set = N.get_rng_state, N.set_rng_state

    def get_rng_state(device=None):
        host = orig_get(device)
        hw = torch.ops.nki_kernels.get_rng_state().view(torch.int32).cpu().view(torch.uint8).flatten()
        return torch.cat([host, hw])

    def set_rng_state(state, device=None):
        host, hw = state[:16], state[16:]
        orig_set(host, device)
        if hw.numel():
            t = hw.clone().view(torch.int32).reshape(1, -1).to(D)
            torch.ops.nki_kernels.set_rng_state(t)

    N.get_rng_state, N.set_rng_state = get_rng_state, set_rng_state
    torch.neuron.get_rng_state, torch.neuron.set_rng_state = get_rng_state, set_rng_state
    torch.neuron._initialized = True

torch.manual_seed(0)
lin = torch.nn.Linear(16, 16).to(D)
x = torch.randn(8, 16).to(D).requires_grad_()
f = lambda x: F.dropout(lin(x), 0.5, training=True)
for reentrant in (False, True):
    errs = []
    for step in range(3):
        lin.weight.grad = None
        o = checkpoint(f, x, use_reentrant=reentrant)
        o.sum().backward()
        expect = ((o != 0).float() * 2.0).t() @ x.detach()
        errs.append(round((lin.weight.grad - expect).abs().max().item(), 6))
    print("RES", sys.argv[1], "reentrant" if reentrant else "non-reentrant", "grad err per step:", errs)

# rewind check: draw after restoring a mid-stream state must repeat
for n in [32, 128, 1024, 65536]:
    drop = lambda: (F.dropout(torch.ones(n, device=D), 0.5) > 0).int().cpu()
    drop(); drop()
    st = torch.neuron.get_rng_state()
    a = drop()
    torch.neuron.set_rng_state(st)
    b = drop()
    torch.neuron.set_rng_state(st)
    c = drop()
    print("RES", sys.argv[1], "n", n, "mid-stream restore repeats:", torch.equal(a, b), "second restore repeats:", torch.equal(a, c), "match frac", round((a == b).float().mean().item(), 3))
print("RES hw state words", torch.ops.nki_kernels.get_rng_state().cpu().tolist())
