import os, sys, time, torch, torch.nn as nn, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("cpu:gloo,neuron:neuron")
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
mode = sys.argv[1]
T0 = time.time()
def P(s): print(f"R{r} {time.time()-T0:6.1f}s {s}", flush=True)
def build():
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(dev)
    fully_shard(m[0]); fully_shard(m[2]); fully_shard(m); return m, torch.optim.AdamW(m.parameters(), lr=1e-2)
m, o = build()
o.zero_grad(); m(torch.randn(8, 16).to(dev)).pow(2).mean().backward(); o.step(); torch.neuron.synchronize(); P("trained")
if mode == "sharded":
    opts_g, opts_s = StateDictOptions(), StateDictOptions()
elif mode == "full":
    opts_g, opts_s = StateDictOptions(full_state_dict=True), StateDictOptions(full_state_dict=True)
elif mode == "full_offload_bcast":
    opts_g, opts_s = StateDictOptions(full_state_dict=True, cpu_offload=True), StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
msd, osd = get_state_dict(m, o, options=opts_g); torch.neuron.synchronize(); P(f"get_state_dict ok, osd keys {len(osd.get('state', {}))}")
m2, o2 = build(); P("built 2")
set_state_dict(m2, o2, model_state_dict=msd, optim_state_dict=osd, options=opts_s); torch.neuron.synchronize(); P("set_state_dict ok")
a = [o.state[p]["exp_avg"].full_tensor().cpu() for p in m.parameters()]
b = [o2.state[p]["exp_avg"].full_tensor().cpu() for p in m2.parameters()]
P(f"exp_avg equal: {all(torch.allclose(x, y) for x, y in zip(a, b))}")
dist.destroy_process_group()
