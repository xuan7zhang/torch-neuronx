import time
import torch, torch_neuronx
nm = torch.neuron
GiB = 2**30
keep = []
for i in range(8):  # 8 x 4 GiB = 32 GiB > 24 GiB HBM for this logical core
    t0 = time.time()
    try:
        keep.append(torch.ones(GiB, dtype=torch.float32, device="neuron"))
        print(f"alloc {i}: returned in {time.time() - t0:.2f}s, memory_allocated={nm.memory_allocated() / GiB:.0f} GiB", flush=True)
    except Exception as e:
        print(f"alloc {i}: raised {type(e).__name__}: {str(e)[:150]}", flush=True)
        break
for i, t in enumerate(keep):
    t0 = time.time()
    try:
        print(f"read {i}: t[-1]={t[-1].item()} in {time.time() - t0:.2f}s", flush=True)
    except Exception as e:
        print(f"read {i}: raised {type(e).__name__}: {str(e)[:150]}", flush=True)
