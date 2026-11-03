import torch, torch.nn.functional as F, torch_neuronx, traceback
from torch.utils.checkpoint import checkpoint
D="neuron"
nm = torch.neuron
og, os_ = nm.get_rng_state, nm.set_rng_state
def g(*a, **k):
    r = og(*a, **k); print("  get_rng_state ->", r.tolist()); return r
def s(st, *a, **k):
    print("  set_rng_state", st.tolist(), "last=", torch_neuronx.neuron._last_seed_state if hasattr(torch_neuronx,'neuron') else None); return os_(st, *a, **k)
nm.get_rng_state, nm.set_rng_state = g, s
import torch_neuronx.neuron as N
N.get_rng_state, N.set_rng_state = g, s
orig_seed = N._set_device_rng_seed
N._set_device_rng_seed = lambda sd: (print("  HW reseed", sd), orig_seed(sd))[1]
torch.ones(1, device=D).cpu(); torch.manual_seed(0)
lin = torch.nn.Linear(16, 16).to(D); x = torch.randn(8, 16).to(D).requires_grad_()
f = lambda x: F.dropout(lin(x), 0.5, training=True)
print("manual_seed(123)"); torch.manual_seed(123)
print("fwd"); o = checkpoint(f, x, use_reentrant=False)
print("bwd"); o.sum().backward()
