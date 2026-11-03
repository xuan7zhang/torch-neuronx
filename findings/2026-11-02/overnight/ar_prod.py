import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["LOCAL_RANK"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
t = torch.full((4,), float(r + 1), device=dev)
try:
    dist.all_reduce(t, op=dist.ReduceOp.PRODUCT); print("RES", r, t.cpu().tolist())
except Exception as e:
    m = str(e); i = m.find("NCC_"); print("RES", r, "ERR", m[i-30:i+250].replace("\n", " ") if i > 0 else m[:300].replace("\n", " "))
dist.destroy_process_group()
