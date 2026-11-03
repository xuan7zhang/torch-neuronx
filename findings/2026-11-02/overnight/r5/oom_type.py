import gc, sys
import torch, torch_neuronx
nm = torch.neuron
GiB = 2**30
mode = sys.argv[1]
def mem(): return f"allocated={nm.memory_allocated() / GiB:.2f} GiB reserved={nm.memory_reserved() / GiB:.2f} GiB"
if mode == "type":
    try:
        x = torch.empty(2**40, device="neuron"); nm.synchronize()
    except Exception as e:
        print("class:", type(e).__mro__[:3])
        print("isinstance torch.OutOfMemoryError:", isinstance(e, torch.OutOfMemoryError))
        print("'out of memory' in message:", "out of memory" in str(e).lower())
        print("message:", str(e)[:400].replace("\n", " "))
    print("after:", mem())
    try:
        from accelerate.utils.memory import should_reduce_batch_size
        print("accelerate should_reduce_batch_size:", should_reduce_batch_size(e if False else RuntimeError(str(e)) ))
    except Exception as ex:
        print("accelerate check skipped:", type(ex).__name__)
if mode == "counter":
    print("start:", mem())
    for n in (2**34, 2**36, 2**40):
        try:
            torch.empty(n, device="neuron"); nm.synchronize()
        except Exception:
            pass
        print(f"after failed alloc of {n * 4 / GiB:.0f} GiB:", mem())
    x = torch.ones(GiB // 4, device="neuron"); nm.synchronize()
    print("after good 1 GiB alloc:", mem())
if mode == "leak":
    print("start:", mem())
    def f():
        w = torch.randn(4096, 4096, device="neuron")
        h = torch.randn(2**15, 4096, device="neuron")
        outs = [h @ w for _ in range(60)]  # 60 x 0.5 GiB = 30 GiB
        nm.synchronize()
        return outs
    try:
        f()
    except Exception as e:
        print("oom:", type(e).__name__, str(e)[:100].replace("\n", " "))
    gc.collect(); nm.synchronize(); nm.empty_cache()
    print("after gc + empty_cache:", mem())
    try:
        big = torch.ones(5 * GiB, dtype=torch.float32, device="neuron"); nm.synchronize()
        print("alloc 20 GiB after:", big[-1].item(), mem())
    except Exception as e:
        print("alloc 20 GiB after: FAIL", str(e)[:120].replace("\n", " "))
