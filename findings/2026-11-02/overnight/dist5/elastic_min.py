import os, sys, time, torch, torch.distributed as dist
DEVN = sys.argv[1]
if DEVN == "neuron":
    import torch_neuronx
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
if DEVN == "neuron":
    torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
else:
    dev = torch.device("cpu")
backend = {"neuron": "neuron", "mixed": "cpu:gloo,neuron:neuron", "cpu": "gloo"}[sys.argv[2]]
dist.init_process_group(backend)
t = torch.full((4,), float(rank), device=dev); dist.all_reduce(t)
if DEVN == "neuron": torch.neuron.synchronize()
print(f"R{rank} attempt {attempt} all_reduce {t.cpu().tolist()[0]}", flush=True)
if attempt == 0 and rank == 1:
    sys.exit(3)
if attempt == 0:
    time.sleep(30)
dist.destroy_process_group()
