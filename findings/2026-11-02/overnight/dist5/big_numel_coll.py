import os, sys, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
mode = sys.argv[1]
GiB = 2**30
res = {}
def check(name, t, idxs, exp):
    torch.neuron.synchronize()
    got = [t[i].item() for i in idxs]
    res[name] = ("OK" if got == exp else "BAD", got, exp)
try:
    if mode == "all_gather":
        n = 2**29 + 1024  # per rank; output 2^31 + 4096 elements bf16 (4.3 GB)
        x = torch.full((n,), float(r + 1), dtype=torch.bfloat16, device=dev); x[-1] = 10.0 * (r + 1)
        o = torch.empty(W * n, dtype=torch.bfloat16, device=dev)
        dist.all_gather_into_tensor(o, x)
        idx = [0, n - 1, n, 2 * n - 1, 3 * n, 4 * n - 1, 2**31 - 1, 2**31 + 5]
        exp = [1.0, 10.0, 2.0, 20.0, 4.0, 40.0, float(2**31 - 1 >= 3 * n and 4.0 or 0), 4.0]
        exp[6] = 4.0 if 2**31 - 1 >= 3 * n else 3.0
        check("all_gather out numel 2^31+4096", o, idx, exp)
    elif mode == "all_reduce":
        n = 2**31 + 4096  # bf16 4.3 GB
        x = torch.empty(n, dtype=torch.bfloat16, device=dev); x.fill_(1.0); x[-1] = float(r)
        dist.all_reduce(x)
        check("all_reduce numel 2^31+4096", x, [0, 2**31 - 1, 2**31, n - 1], [4.0, 4.0, 4.0, float(sum(range(W)))])
    elif mode == "reduce_scatter":
        n = 2**31 + 4 * 1024
        x = torch.empty(n, dtype=torch.bfloat16, device=dev); x.fill_(1.0); x[-1] = 5.0
        o = torch.empty(n // W, dtype=torch.bfloat16, device=dev)
        dist.reduce_scatter_tensor(o, x)
        check("reduce_scatter in numel 2^31+4096", o, [0, n // W - 1], [4.0, 20.0 if r == W - 1 else 4.0])
except Exception as e:
    res[mode] = "ERR " + str(e).replace("\n", " ")[:250]
print(f"RES R{r} {res}", flush=True)
dist.destroy_process_group()
