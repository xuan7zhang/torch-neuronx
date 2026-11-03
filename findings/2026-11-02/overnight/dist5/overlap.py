import os, time, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
c = torch.ones(2**27, device=dev)  # 512 MiB all_reduce
def mm(n):
    y = a
    for _ in range(n): y = a @ a
    return y
def t(f, it=3):
    f(); torch.neuron.synchronize(); dist.barrier(); torch.neuron.synchronize()
    t0 = time.time()
    for _ in range(it): f()
    torch.neuron.synchronize(); return (time.time() - t0) / it
n = 20
t_mm = t(lambda: mm(n))
t_ar = t(lambda: dist.all_reduce(c))
def both_async():
    w = dist.all_reduce(c, async_op=True); mm(n); w.wait()
t_both = t(both_async)
s = torch.neuron.Stream()
def both_stream():
    s.wait_stream(torch.neuron.current_stream())
    with torch.neuron.stream(s):
        dist.all_reduce(c)
    mm(n); torch.neuron.current_stream().wait_stream(s)
t_stream = t(both_stream)
if r == 0:
    print(f"RES matmul x{n}: {t_mm*1e3:.1f} ms, all_reduce 512MiB: {t_ar*1e3:.1f} ms, async overlap: {t_both*1e3:.1f} ms, side-stream overlap: {t_stream*1e3:.1f} ms, serial sum {(t_mm+t_ar)*1e3:.1f} ms")
dist.destroy_process_group()
