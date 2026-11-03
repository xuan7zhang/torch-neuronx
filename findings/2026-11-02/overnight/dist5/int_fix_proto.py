"""Prototype exact integer broadcast and SUM on the Neuron backend (no file change)."""
import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")

def bcast_exact(t, src):
    # Data movement only: all_gather the int32 view, keep the src slice.
    v = t.view(torch.int32) if t.dtype == torch.int64 else t
    o = torch.empty((W,) + v.shape, dtype=v.dtype, device=v.device)
    dist.all_gather_into_tensor(o, v.contiguous())
    v.copy_(o[src])
    return t

def sum_exact(t):
    # Split into 16-bit limbs; each limb sum < W * 2^16 < 2^24, exact in fp32 reduction.
    assert t.dtype in (torch.int32, torch.int64)
    nl = 4 if t.dtype == torch.int64 else 2
    x = t.to(torch.int64) if t.dtype == torch.int32 else t
    limbs = torch.stack([((x >> (16 * i)) & 0xFFFF).to(torch.int32) for i in range(nl)])
    dist.all_reduce(limbs)
    acc = torch.zeros_like(x)
    for i in range(nl):
        acc = acc + (limbs[i].to(torch.int64) << (16 * i))
    t.copy_(acc.to(t.dtype))
    return t

ok = {}
for dt in (torch.int32, torch.int64):
    for base in (16_777_216, 2_000_000_000) + ((3_000_000_000, -(2**40)) if dt == torch.int64 else ()):
        v = torch.arange(8, dtype=dt) + base
        b = (v if r == 0 else torch.zeros(8, dtype=dt)).to(dev); bcast_exact(b, 0)
        s = (v + r).to(dev); sum_exact(s)
        torch.neuron.synchronize()
        exp_s = (v.to(torch.int64) * W + sum(range(W)))
        if dt == torch.int32:
            exp_s = exp_s.to(torch.int32)  # wraps like CPU int32
        ok[f"{str(dt)[6:]} {base}"] = (torch.equal(b.cpu(), v), torch.equal(s.cpu(), exp_s))
if r == 0:
    print("RES (broadcast exact, sum exact):", ok)
dist.destroy_process_group()
