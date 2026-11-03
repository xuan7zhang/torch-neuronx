import os, sys, torch, torch.nn as nn, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
from torch.nn.parallel import DistributedDataParallel as DDP
mode = sys.argv[1]
state = {}
def hook(st, bucket):
    buf = bucket.buffer()
    cp = buf.detach().clone() if mode != "noclone" else None
    if mode == "zeros_first":
        state.setdefault("z", torch.zeros(buf.numel(), device=buf.device))
    fut = dist.all_reduce(buf, async_op=True).get_future()
    def cb(f):
        t = f.value()[0]
        if cp is not None:
            state["err"] = cp - t
        return t.div_(W)
    return fut.then(cb)
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev); d = DDP(m)
d.register_comm_hook(None, hook)
o = torch.optim.SGD(d.parameters(), lr=0.05)
try:
    for s in range(3):
        o.zero_grad(); d(torch.randn(8, 16, device=dev)).pow(2).mean().backward(); o.step(); torch.neuron.synchronize()
        if "err" in state: _ = state["err"][0].item()
    msg = "ok"
except Exception as e:
    msg = "ERR " + str(e)[100:250]
if r == 0: print(f"RES {mode}: {msg}")
dist.destroy_process_group()
