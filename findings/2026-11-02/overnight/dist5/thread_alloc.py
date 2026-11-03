"""Single process: create a Neuron tensor in one thread, use it in another (no sync between)."""
import sys, threading, torch, torch_neuronx
dev = torch.device("neuron", 0)
mode = sys.argv[1]
box = {}
errs = []
def producer(i):
    if mode == "zeros":
        box[i] = torch.zeros(4096, device=dev)
    elif mode == "sub":
        a = torch.ones(4096, device=dev); b = torch.full((4096,), 3.0, device=dev); box[i] = a - b
for i in range(20):
    t = threading.Thread(target=producer, args=(i,)); t.start(); t.join()
    try:
        y = box[i] + 1  # consumer on main thread
        torch.neuron.synchronize()
        v = y[0].item()
    except Exception as e:
        errs.append((i, str(e)[100:220])); break
print(f"RES {mode}: errors {errs[:1]} (20 iterations)")
# backward hook thread: create a tensor inside a grad hook (autograd thread), use it after backward on main thread
store = {}
w = torch.randn(64, 64, device=dev, requires_grad=True)
def hook(gr):
    store["z"] = torch.zeros_like(gr)
    store["e"] = gr.clone() - gr
    return gr
w.register_hook(hook)
try:
    for it in range(5):
        (w @ w).sum().backward()
        q = store["z"] + store["e"] + 1
        torch.neuron.synchronize(); _ = q[0, 0].item()
    print("RES grad-hook-thread alloc: ok")
except Exception as e:
    print("RES grad-hook-thread alloc: ERR", str(e)[100:250])
