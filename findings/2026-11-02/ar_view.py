import os, sys, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["LOCAL_RANK"]); w = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
S = sum(range(1, w + 1))
cases = {
 "t()[1:3]": lambda b: b.t()[1:3], "t()[1]": lambda b: b.t()[1], "[:,1:3]": lambda b: b[:, 1:3], "[1:]": lambda b: b[1:],
 "[1]": lambda b: b[1], "t()": lambda b: b.t(), "[:, ::2]": lambda b: b[:, ::2],
}
for nm, f in cases.items():
    base = torch.arange(12.0, device=dev).reshape(3, 4) * (r + 1)
    v = f(base)
    try:
        dist.all_reduce(v)
        exp_base = torch.arange(12.0).reshape(3, 4) * (r + 1); f(exp_base).copy_(f(torch.arange(12.0).reshape(3, 4)) * S)
        vc = v.cpu(); bc = base.cpu()
        ok_v = torch.equal(vc, f(torch.arange(12.0).reshape(3, 4)) * S); ok_b = torch.equal(bc, exp_base)
        msg = f"view={'OK' if ok_v else 'BAD'} base={'OK' if ok_b else 'BAD ' + str(bc.tolist())}"
    except Exception as e:
        msg = "ERR " + str(e)[:100].replace("\n", " ")
    if r == 0: print("RES", f"{nm:10s}", msg, flush=True)
dist.destroy_process_group()
