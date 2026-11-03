import os, sys, traceback, torch, torch.nn as nn, torch.distributed as dist, torch_neuronx
import torch.distributed._functional_collectives as fc
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
mode = sys.argv[1]
D = 8
if mode == "even":
    split = [[2] * W for _ in range(W)]
else:
    split = [[(q + j) % 3 + 1 for j in range(W)] for q in range(W)]
send = split[r]; recv = [split[j][r] for j in range(W)]
grad = sys.argv[2] == "grad"
def f(x):
    if grad:
        return fc.all_to_all_single_autograd(x, recv, send, dist.group.WORLD) * 2
    return fc.all_to_all_single(x, recv, send, dist.group.WORLD) * 2
x = torch.randn(sum(send), D, generator=torch.Generator().manual_seed(r)).to(dev).requires_grad_(grad)
try:
    y = torch.compile(f, backend="neuron", dynamic=False)(x)
    if grad: y.sum().backward()
    torch.neuron.synchronize()
    ye = f(x.detach())
    ok = torch.allclose(y.detach().cpu(), ye.detach().cpu())
    msg = f"ok match={ok}"
except Exception as e:
    msg = "ERR " + str(e).replace("\n", " ")[:700]
if r == 0: print(f"RES {mode} grad={grad}: {msg}")
dist.destroy_process_group()
