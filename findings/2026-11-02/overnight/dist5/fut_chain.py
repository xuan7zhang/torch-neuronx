import os, sys, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
mode = sys.argv[1]
store = {}
res = []
for it in range(3):
    x = torch.full((1000,), float(r + 1), device=dev)
    cp = torch.clone(x).detach()
    p = torch.full((10,), 1.0, device=dev)
    def cb1(f):
        return dist.all_reduce(p, async_op=True).get_future().wait()[0]
    def cb2(f):
        if mode == "sub_store":
            store["err"] = cp - x
        elif mode == "sub_store_after_mul":
            x.mul_(2); store["err"] = cp - x
        return x
    out = dist.all_reduce(x, async_op=True).get_future().then(cb1).then(cb2).wait()
    try:
        torch.neuron.synchronize()
        res.append((round(out[0].item() if not isinstance(out, list) else out[0][0].item(), 3), round(store["err"][0].item(), 3)))
    except Exception as e:
        res.append("ERR " + str(e)[150:260]); break
if r == 0:
    print(f"RES {mode}: {res}")
dist.destroy_process_group()
