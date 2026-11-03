import os, time, random, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
D = 512
times = []
for step in range(15):
    rnd = random.Random(step)
    split = [[rnd.randint(1, 64) for _ in range(W)] for _ in range(W)]
    send = split[r]; recv = [split[j][r] for j in range(W)]
    x = torch.randn(sum(send), D, device=dev); o = torch.empty(sum(recv), D, device=dev)
    torch.neuron.synchronize(); dist.barrier(); t0 = time.time()
    dist.all_to_all_single(o, x, recv, send); torch.neuron.synchronize()
    times.append((time.time() - t0) * 1e3)
even = torch.randn(W * 32, D, device=dev); oe = torch.empty_like(even)
dist.all_to_all_single(oe, even); torch.neuron.synchronize(); t0 = time.time()
for _ in range(10): dist.all_to_all_single(oe, even)
torch.neuron.synchronize(); te = (time.time() - t0) / 10 * 1e3
if r == 0:
    print("RES uneven a2a ms per step (new random splits each step):", [round(t, 1) for t in times])
    print(f"RES even a2a warm: {te:.2f} ms")
dist.destroy_process_group()
