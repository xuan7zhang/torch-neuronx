import os, time, torch, torch.distributed as dist, torch_neuronx
import torch.distributed._functional_collectives as fc
r = int(os.environ["RANK"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
c = torch.ones(2**27, device=dev)
def mm_only(a):
    y = a
    for _ in range(20): y = a @ y * 0.01
    return y
def ar_only(c):
    return fc.all_reduce(c, "sum", dist.group.WORLD) + 0
def both(a, c):
    z = fc.all_reduce(c, "sum", dist.group.WORLD)
    y = a
    for _ in range(20): y = a @ y * 0.01
    return y, z + 0
cm, ca, cb = (torch.compile(f, backend="neuron", dynamic=False) for f in (mm_only, ar_only, both))
def t(f, it=3):
    f(); torch.neuron.synchronize(); dist.barrier(); torch.neuron.synchronize()
    t0 = time.time()
    for _ in range(it): f()
    torch.neuron.synchronize(); return (time.time() - t0) / it
tm, ta, tb = t(lambda: cm(a)), t(lambda: ca(c)), t(lambda: cb(a, c))
if r == 0:
    print(f"RES compiled: matmul chain {tm*1e3:.1f} ms, all_reduce 512MiB {ta*1e3:.1f} ms, both in one graph {tb*1e3:.1f} ms, serial sum {(tm+ta)*1e3:.1f} ms")
dist.destroy_process_group()
