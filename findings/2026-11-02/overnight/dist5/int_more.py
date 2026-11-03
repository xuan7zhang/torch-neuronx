import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
res = {}
base = 20_000_001
for dt in (torch.int32, torch.int64):
    v = torch.arange(W * 2, dtype=dt) + base + r * 100
    o = torch.empty(2, dtype=dt, device=dev); dist.reduce_scatter_tensor(o, v.to(dev)); torch.neuron.synchronize()
    exp = sum(torch.arange(W * 2, dtype=torch.int64) + base + q * 100 for q in range(W))[r * 2:r * 2 + 2]
    res[f"reduce_scatter {str(dt)[6:]}"] = torch.equal(o.cpu().to(torch.int64), exp)
    a = torch.empty(W * 2, dtype=dt, device=dev); dist.all_to_all_single(a, v.to(dev)); torch.neuron.synchronize()
    expa = torch.cat([(torch.arange(W * 2, dtype=dt) + base + q * 100)[r * 2:r * 2 + 2] for q in range(W)])
    res[f"all_to_all_single {str(dt)[6:]}"] = torch.equal(a.cpu(), expa)
f = torch.tensor([-0.0, 1e-40, 1.0000001, 3.4e38, -1e-45], dtype=torch.float32)
b = (f if r == 1 else torch.full((5,), 7.0)).to(dev); dist.broadcast(b, src=1); torch.neuron.synchronize()
bc = b.cpu()
res["broadcast fp32 bits equal"] = torch.equal(bc.view(torch.int32), f.view(torch.int32))
res["broadcast fp32 values"] = [repr(x) for x in bc.tolist()]
res["signbit(-0.0) kept"] = bool(torch.signbit(bc[0]))
h = torch.tensor([65504.0, 6e-8, -0.0], dtype=torch.float16)
bh = (h if r == 1 else torch.zeros(3, dtype=torch.float16)).to(dev); dist.broadcast(bh, src=1); torch.neuron.synchronize()
res["broadcast fp16 bits equal"] = torch.equal(bh.cpu().view(torch.int16), h.view(torch.int16))
u = torch.tensor([200, 255, 1], dtype=torch.uint8)
bu = (u if r == 1 else torch.zeros(3, dtype=torch.uint8)).to(dev); dist.broadcast(bu, src=1); torch.neuron.synchronize()
res["broadcast uint8"] = bu.cpu().tolist()
if r == 0:
    for k, x in res.items(): print("RES", k, x)
dist.destroy_process_group()
