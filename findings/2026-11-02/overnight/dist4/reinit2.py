import os, sys, torch, torch.distributed as dist
backend = sys.argv[1]
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
if backend != "gloo":
    import torch_neuronx
    torch.neuron.set_device(r); dev = torch.device("neuron", r)
else:
    dev = torch.device("cpu")
def step(tag):
    t = torch.full((4,), float(r), device=dev); dist.all_reduce(t)
    if backend != "gloo": torch.neuron.synchronize()
    print(f"R{r} {tag}: {t.cpu().tolist()[0]} (expect {sum(range(W))})", flush=True)
dist.init_process_group(backend); step("first")
dist.destroy_process_group(); print(f"R{r} destroyed", flush=True)
dist.init_process_group(backend); print(f"R{r} re-init returned", flush=True)
step("after reinit")
dist.destroy_process_group()
