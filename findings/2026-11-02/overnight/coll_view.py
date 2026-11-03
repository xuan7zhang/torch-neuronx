import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["LOCAL_RANK"]); w = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
def report(nm, base, exp, v=None, vshape=None):
    try:
        ok = torch.equal(base.cpu(), exp); meta_ok = v is None or tuple(v.shape) == vshape
        msg = ("OK " if ok and meta_ok else "BAD") + ("" if meta_ok else f" view meta changed to {tuple(v.shape)}") + ("" if ok else f" base={base.cpu().flatten().tolist()} exp={exp.flatten().tolist()}")
    except Exception as e: msg = "ERR " + str(e)[:80]
    if r == 0: print("RES", f"{nm:34s}", msg[:260], flush=True)
# 1 broadcast into cols view
b = torch.zeros(3, 4, device=dev) + r; v = b[:, 1:3]
if r == 0: v.copy_(torch.full((3, 2), 7.0, device=dev))
dist.broadcast(v, src=0); e = torch.zeros(3, 4) + r; e[:, 1:3] = 7.0; report("broadcast into [:,1:3]", b, e, v, (3, 2))
# 2 all_gather_into_tensor output is a column slice
b = torch.zeros(w * 2, 5, device=dev); v = b[:, 1:4]
dist.all_gather_into_tensor(v, torch.full((2, 3), float(r), device=dev)); e = torch.zeros(w * 2, 5); e[:, 1:4] = torch.arange(w).repeat_interleave(2).float()[:, None]
report("all_gather_into_tensor out=[:,1:4]", b, e, v, (w * 2, 3))
# 3 reduce_scatter_tensor into column slice
b = torch.zeros(2, 5, device=dev); v = b[:, 1:4]
dist.reduce_scatter_tensor(v, torch.ones(w * 2, 3, device=dev)); e = torch.zeros(2, 5); e[:, 1:4] = float(w)
report("reduce_scatter_tensor out=[:,1:4]", b, e, v, (2, 3))
# 4 all_reduce on a grad column slice (realistic: sync part of a param grad)
p = torch.nn.Parameter(torch.zeros(4, 6, device=dev)); p.grad = torch.full((4, 6), float(r + 1), device=dev)
v = p.grad[:, 2:]; dist.all_reduce(v); e = torch.full((4, 6), float(r + 1)); e[:, 2:] = float(sum(range(1, w + 1)))
report("all_reduce p.grad[:, 2:]", p.grad, e, v, (4, 4))
# 5 all_reduce row slice with offset (contiguous)
b = torch.full((4, 6), float(r + 1), device=dev); v = b[1:3]; dist.all_reduce(v); e = torch.full((4, 6), float(r + 1)); e[1:3] = float(sum(range(1, w + 1)))
report("all_reduce [1:3] (contiguous)", b, e, v, (2, 6))
# 6 all_reduce transposed (offset 0)
b = torch.arange(12.0, device=dev).reshape(3, 4) * (r + 1); v = b.t(); dist.all_reduce(v); e = torch.arange(12.0).reshape(3, 4) * sum(range(1, w + 1))
report("all_reduce .t() (offset 0)", b, e, v, (4, 3))
dist.destroy_process_group()
