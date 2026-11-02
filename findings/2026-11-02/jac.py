import torch, torch_neuronx, traceback
from torch.func import vmap, jacrev
x = torch.randn(4, device="neuron")
for nm, fn in [
  ("vmap tanh_backward", lambda: vmap(lambda g: torch.ops.aten.tanh_backward(g, x.tanh()))(torch.eye(4, device="neuron"))),
  ("vmap mul by unbatched", lambda: vmap(lambda g: g * x)(torch.eye(4, device="neuron"))),
  ("vmap mul unbatched*batched", lambda: vmap(lambda g: x * g)(torch.eye(4, device="neuron"))),
  ("vmap sigmoid_backward", lambda: vmap(lambda g: torch.ops.aten.sigmoid_backward(g, x.sigmoid()))(torch.eye(4, device="neuron"))),
  ("jacrev sigmoid", lambda: jacrev(torch.sigmoid)(x)),
  ("jacrev exp", lambda: jacrev(torch.exp)(x)),
  ("jacrev sin", lambda: jacrev(torch.sin)(x)),
]:
    try: r = fn(); print("OK ", nm, r.shape)
    except Exception as e: print("ERR", nm, type(e).__name__, str(e)[:160].replace("\n"," "))
