import sys, torch, torch.nn.functional as F, torch_neuronx
from torch.utils.checkpoint import checkpoint
D = "neuron"
if sys.argv[1] == "patch":
    torch.neuron._initialized = True
torch.ones(1, device=D).cpu(); torch.manual_seed(0)
lin = torch.nn.Linear(16, 16).to(D); x = torch.randn(8, 16).to(D).requires_grad_()
f = lambda x: F.dropout(lin(x), 0.5, training=True)
for step in range(3):  # like a training loop: several checkpointed steps, no reseed
    lin.weight.grad = None
    o = checkpoint(f, x, use_reentrant=False); o.sum().backward()
    expect = ((o != 0).float() * 2.0).t() @ x.detach()
    print(sys.argv[1], "step", step, "grad err vs fwd mask:", (lin.weight.grad - expect).abs().max().item())
