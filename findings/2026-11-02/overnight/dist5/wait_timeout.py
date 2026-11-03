import os, sys, time, torch, torch.distributed as dist, torch_neuronx
from datetime import timedelta
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron", timeout=timedelta(seconds=300))
t = torch.ones(4, device=dev)
if r == 3:
    time.sleep(60)
t0 = time.time()
try:
    w = dist.all_reduce(t, async_op=True)
    ok = w.wait(10000) if os.environ.get("MS") else w.wait(timedelta(seconds=10))
    after_wait = time.time() - t0
    torch.neuron.synchronize()
    print(f"R{r} wait returned {ok} after {after_wait:.1f}s, sync done {time.time()-t0:.1f}s, value {t[0].item()}", flush=True)
except Exception as e:
    print(f"R{r} raised after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:120]}", flush=True)
try:
    u = torch.ones(2, device=dev); dist.all_reduce(u); torch.neuron.synchronize()
    print(f"R{r} next all_reduce {u.cpu().tolist()}", flush=True)
except Exception as e:
    print(f"R{r} next all_reduce raised: {str(e)[:100]}", flush=True)
