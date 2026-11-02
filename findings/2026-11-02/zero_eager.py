import torch, torch.nn as nn, torch_neuronx
lin = nn.Linear(3, 4).to("neuron")
for shape in [(0, 3), (1, 0, 3)]:
    x = torch.randn(*shape).to("neuron").requires_grad_()
    try:
        y = lin(x) if shape[-1] == 3 and len(shape) == 2 else x.exp()
        y.sum().backward(); torch.neuron.synchronize(); print("RES eager OK", shape, tuple(y.shape), x.grad.shape)
    except Exception as e: print("RES eager ERR", shape, str(e)[:120])
# MoE-style: an expert receives a zero-size batch inside a compiled expert fn with static shapes
exp_fn = torch.compile(lambda t: lin(t).relu(), backend="neuron", dynamic=False)
for n in [5, 0]:
    try:
        y = exp_fn(torch.randn(n, 3).to("neuron")); torch.neuron.synchronize(); print("RES compiled expert n=%d OK" % n, tuple(y.shape))
    except Exception as e:
        m = str(e); i = m.find("NCC_"); print("RES compiled expert n=%d ERR" % n, (m[i:i+60] if i > 0 else m[:100]).replace("\n", " "))
