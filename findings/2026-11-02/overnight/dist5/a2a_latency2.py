import os, time, random, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
D = 512
def run(split):
    send = split[r]; recv = [split[j][r] for j in range(W)]
    x = torch.randn(sum(send), D, device=dev); o = torch.empty(sum(recv), D, device=dev)
    torch.neuron.synchronize(); dist.barrier(); t0 = time.time()
    dist.all_to_all_single(o, x, recv, send); torch.neuron.synchronize()
    return round((time.time() - t0) * 1e3, 1)
base = [[(q + j) % 4 * 8 + 8 for j in range(W)] for q in range(W)]
res = {}
res["same splits x5"] = [run(base) for _ in range(5)]
# permute splits within each row and column keeping send totals and recv totals fixed: use a Latin-square shift of base
def shifted(k):
    return [[base[q][(j + k) % W] for j in range(W)] for q in range(W)]
res["diff splits, same send totals (recv totals change)"] = [run(shifted(k)) for k in range(1, 4)]
# doubly-stochastic style: splits where every row sum and column sum = 80
def latin(k):
    vals = [8, 16, 24, 32]
    return [[vals[(q + j + k) % W] for j in range(W)] for q in range(W)]
res["diff splits, same send AND recv totals"] = [run(latin(k)) for k in range(4)]
res["repeat latin(1) again"] = [run(latin(1)) for _ in range(2)]
if r == 0:
    for k, v in res.items(): print(f"RES {k:52s} {v} ms")
dist.destroy_process_group()
