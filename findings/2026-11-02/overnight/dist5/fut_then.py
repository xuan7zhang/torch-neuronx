import os, sys, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
mode = sys.argv[1]
res = []
for it in range(3):
    x = torch.full((1000,), float(r + 1), device=dev)
    cp = torch.clone(x).detach()
    fut = dist.all_reduce(x, async_op=True).get_future()
    if mode == "sub_new":
        out = fut.then(lambda f: cp - f.value()[0]).wait()
    elif mode == "inplace":
        out = fut.then(lambda f: f.value()[0].div_(W)).wait()
    elif mode == "new_tensor":
        out = fut.then(lambda f: torch.zeros(1000, device=dev) + f.value()[0]).wait()
    elif mode == "no_cb":
        fut.wait(); out = cp - x
    out = out[0] if isinstance(out, list) else out
    try:
        torch.neuron.synchronize(); res.append(round(out[0].item(), 3))
    except Exception as e:
        res.append("ERR " + str(e)[150:260]); break
if r == 0:
    exp = {"sub_new": 1 - sum(range(1, W + 1)), "inplace": sum(range(1, W + 1)) / W, "new_tensor": sum(range(1, W + 1)), "no_cb": 1 - sum(range(1, W + 1))}[mode]
    print(f"RES {mode}: {res} expect {exp}")
dist.destroy_process_group()
