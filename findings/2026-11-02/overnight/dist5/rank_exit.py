import os, sys, time, torch, torch.distributed as dist
from datetime import timedelta
DEVN = os.environ.get("DEVN", "neuron")
if DEVN == "neuron":
    import torch_neuronx
r = int(os.environ["RANK"])
if DEVN == "neuron": torch.neuron.set_device(r); dev = torch.device("neuron", r)
else: dev = torch.device("cpu")
dist.init_process_group("neuron" if DEVN == "neuron" else "gloo", timeout=timedelta(seconds=30))
t = torch.ones(4, device=dev); dist.all_reduce(t)
if DEVN == "neuron": torch.neuron.synchronize()
print(f"R{r} first ok", flush=True)
if r == 3:
    print("R3 exits without the next collective", flush=True); os._exit(0)
t0 = time.time()
try:
    dist.all_reduce(t)
    if DEVN == "neuron": torch.neuron.synchronize()
    print(f"R{r} second returned after {time.time()-t0:.0f}s: {t.cpu().tolist()}", flush=True)
except Exception as e:
    print(f"R{r} second raised after {time.time()-t0:.0f}s: {type(e).__name__}: {str(e)[:120]}", flush=True)
