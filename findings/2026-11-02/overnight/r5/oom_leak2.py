import gc, sys
import torch, torch_neuronx
nm = torch.neuron
GiB = 2**30
do_oom = sys.argv[1] == "oom"
def try_alloc(gib):
    try:
        t = torch.ones(int(gib * GiB) // 4, device="neuron"); nm.synchronize(); v = t[-1].item(); del t; nm.synchronize(); nm.empty_cache()
        return "ok"
    except Exception as e:
        return "FAIL " + str(e)[:300].replace("\n", " ")
if do_oom:
    def f():
        w = torch.randn(4096, 4096, device="neuron")
        h = torch.randn(2**15, 4096, device="neuron")
        outs = [h @ w for _ in range(60)]
        nm.synchronize()
    try:
        f()
    except Exception as e:
        print("oom raised:", type(e).__name__)
    gc.collect(); nm.synchronize(); nm.empty_cache()
for gib in (1, 4, 8, 12, 16, 20):
    print(f"{'after oom' if do_oom else 'fresh'}: alloc {gib} GiB:", try_alloc(gib), flush=True)
print("small matmul:", end=" ")
try:
    a = torch.randn(64, 64, device="neuron"); print(((a @ a).cpu() - a.cpu() @ a.cpu()).abs().max().item())
except Exception as e:
    print("FAIL", str(e)[:200])
