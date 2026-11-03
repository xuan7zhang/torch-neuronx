"""Step-0 grads per parameter: TP (mesh tp=W or dp2 x tp2) vs CPU on the same global batch. Uses llama_scale model."""
import os, sys
mode = sys.argv[1]
sys.argv = ["llama_scale.py", "ref", "1"]
DEVN = os.environ.get("DEVN", "neuron")
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llama_scale as L
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
dist.init_process_group("cpu:gloo,neuron:neuron")
if mode == "tp":
    tp_mesh = init_device_mesh("neuron", (W,)); dp_rank, dp = 0, 1
else:
    mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp")); tp_mesh = mesh["tp"]; dp_rank, dp = mesh["dp"].get_local_rank(), 2
parts = os.environ.get("PARTS", "emb,head,attn,mlp").split(",")
m = L.LM().to(dev)
top = {}
if "emb" in parts: top["emb"] = RowwiseParallel(input_layouts=Replicate())
if "head" in parts: top["head"] = ColwiseParallel(output_layouts=Replicate())
if top: parallelize_module(m, tp_mesh, top)
for blk in m.layers:
    plan = {}
    if "attn" in parts: plan.update({"attn.wq": ColwiseParallel(), "attn.wk": ColwiseParallel(), "attn.wv": ColwiseParallel(), "attn.wo": RowwiseParallel()})
    if "mlp" in parts: plan.update({"w1": ColwiseParallel(), "w3": ColwiseParallel(), "w2": RowwiseParallel()})
    if plan: parallelize_module(blk, tp_mesh, plan)
x, y = L.data(0, dp_rank, dp)
l = F.cross_entropy(m(x.to(dev)).float().flatten(0, 1), y.to(dev).flatten())
l.backward()
grads = {}
for n, p in m.named_parameters():
    gr = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
    if dp > 1:
        gr = gr.clone(); dist.all_reduce(gr, group=mesh["dp"].get_group()); gr /= dp
    grads[n] = gr.float().cpu()
if rank == 0:
    torch.manual_seed(0)
    ref = L.LM()
    tot = 0
    for r in range(dp):
        xr, yr = L.data(0, r, dp); (F.cross_entropy(ref(xr).flatten(0, 1), yr.flatten()) / dp).backward()
    worst = sorted(((((p.grad - grads[n]).norm() / (p.grad.norm() + 1e-12)).item(), n) for n, p in ref.named_parameters()), reverse=True)
    print(f"RES {mode} parts={parts} loss {l.item():.5f} worst grads:", [(n, f"{e:.3g}") for e, n in worst[:6]], flush=True)
dist.destroy_process_group()
