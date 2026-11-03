import os, sys, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
kw = {"device_id": dev} if sys.argv[1] == "device_id" else {}
dist.init_process_group("neuron", **kw)
from torch.distributed.device_mesh import init_device_mesh
res = {}
for shape in ((W,), (2, 2)):
    try:
        m = init_device_mesh("neuron", shape)
        g = m.get_group(0) if len(shape) > 1 else m.get_group()
        t = torch.ones(2, device=dev); dist.all_reduce(t, group=g); torch.neuron.synchronize()
        res[str(shape)] = f"ok {t.cpu().tolist()}"
    except Exception as e:
        res[str(shape)] = f"ERR {type(e).__name__}: {str(e)[:110]}"
try:
    sg = dist.new_group([0, 1]); res["new_group"] = "ok"
except Exception as e:
    res["new_group"] = f"ERR {str(e)[:100]}"
if r == 0: print(f"RES init_process_group({sys.argv[1]}): {res}")
dist.destroy_process_group()
