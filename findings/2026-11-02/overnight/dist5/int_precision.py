import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("cpu:gloo,neuron:neuron")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, Replicate, distribute_tensor
mesh = init_device_mesh("neuron", (W,))
res = {}
for dt in (torch.int64, torch.int32):
    for base in (1000, 16_777_216, 100_000_000, 2_000_000_000):
        v = torch.arange(8, dtype=dt) + base
        b = (v if r == 0 else torch.zeros(8, dtype=dt)).to(dev); dist.broadcast(b, src=0); torch.neuron.synchronize()
        o = torch.empty(W * 8, dtype=dt, device=dev); dist.all_gather_into_tensor(o, v.to(dev)); torch.neuron.synchronize()
        d = distribute_tensor(v.to(dev), mesh, [Shard(0)]); f = d.full_tensor(); torch.neuron.synchronize()
        sc = torch.empty(2, dtype=dt, device=dev); dist.scatter(sc, list(v.to(dev).chunk(W)) if r == 0 else None, src=0); torch.neuron.synchronize()
        s = (v + r).to(dev); dist.all_reduce(s); torch.neuron.synchronize()
        res[f"{str(dt)[6:]} base {base}"] = dict(broadcast=torch.equal(b.cpu(), v), all_gather=torch.equal(o.cpu()[:8], v),
            distribute_full=torch.equal(f.cpu(), v), scatter=torch.equal(sc.cpu(), v.chunk(W)[r]),
            all_reduce=torch.equal(s.cpu(), v * W + sum(range(W))), sample=(f.cpu()[:3].tolist()))
if r == 0:
    for k, x in res.items(): print("RES", k, x)
dist.destroy_process_group()
