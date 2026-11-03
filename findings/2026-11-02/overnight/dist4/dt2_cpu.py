import os, torch, torch.nn.functional as F, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); dev = torch.device("cpu")
dist.init_process_group("gloo")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
mesh = init_device_mesh("cpu", (W,))
g = lambda s: torch.Generator().manual_seed(s)
def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu(); return ((a - b).norm() / (a.norm() + 1e-12)).item()
A = torch.randn(8, 12, generator=g(0)); B = torch.randn(8, 12, generator=g(1))
ops = {"softmax0": lambda a, b: a.softmax(0), "softmax1": lambda a, b: a.softmax(1), "layer_norm": lambda a, b: F.layer_norm(a, (12,)),
       "mse": lambda a, b: F.mse_loss(a, b), "mse_sum": lambda a, b: F.mse_loss(a, b, reduction="sum"),
       "mse_none": lambda a, b: F.mse_loss(a, b, reduction="none"), "l1": lambda a, b: F.l1_loss(a, b),
       "smooth_l1": lambda a, b: F.smooth_l1_loss(a, b), "manual_mse": lambda a, b: ((a - b) ** 2).mean()}
out = {}
for name, f in ops.items():
    ac, bc = A.clone().requires_grad_(), B.clone().requires_grad_()
    rc = f(ac, bc); Wt = torch.randn(rc.shape, generator=g(7)); (rc * Wt).sum().backward()
    for pn, pl in {"S0": [Shard(0)], "S1": [Shard(1)], "R": [Replicate()]}.items():
        ad, bd = (distribute_tensor(t.to(dev), mesh, pl).requires_grad_() for t in (A, B))
        rd = f(ad, bd)
        wd = distribute_tensor(Wt.to(dev), mesh, rd.placements) if rd.dim() > 0 else Wt.to(dev)
        (rd * wd).sum().backward() if rd.dim() > 0 else (rd * Wt.item()).backward()

        out[f"{name}/{pn}"] = f"fwd {rel(rc, rd.full_tensor()):.2g} ga {rel(ac.grad, ad.grad.full_tensor()):.2g} gb {rel(bc.grad, bd.grad.full_tensor()) if bc.grad is not None and bd.grad is not None else (0.0 if bc.grad is None and bd.grad is None else 9.0):.2g} out_pl {rd.placements}"
if rank == 0:
    for k, v in out.items(): print("RES", k, v)
dist.destroy_process_group()
