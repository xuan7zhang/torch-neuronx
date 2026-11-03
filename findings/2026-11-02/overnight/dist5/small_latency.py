import os, time, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
def t(f):
    f(); torch.neuron.synchronize(); dist.barrier(); torch.neuron.synchronize()
    t0 = time.time(); f(); torch.neuron.synchronize(); return time.time() - t0
same = torch.ones(7, device=dev)
res = {}
res["200x same tensor (7,)"] = t(lambda: [dist.all_reduce(same) for _ in range(200)]) / 200
ts = [torch.ones(7, device=dev) for _ in range(200)]
res["200x distinct tensors (7,) 2nd pass"] = t(lambda: [dist.all_reduce(x) for x in ts]) / 200
shapes = [torch.ones(1000 + i, device=dev) for i in range(50)]
t0 = time.time(); [dist.all_reduce(x) for x in shapes]; torch.neuron.synchronize(); res["50 new shapes, first call"] = (time.time() - t0) / 50
res["50 new shapes, second call"] = t(lambda: [dist.all_reduce(x) for x in shapes]) / 50
views = torch.ones(200, 7, device=dev)
res["200x row views of one buffer"] = t(lambda: [dist.all_reduce(views[i]) for i in range(200)]) / 200
if r == 0:
    for k, v in res.items(): print(f"RES {k:40s} {v*1e3:8.3f} ms per all_reduce")
dist.destroy_process_group()
