import os, torch, torch.nn.functional as F, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, Replicate, DTensor, distribute_tensor
from torch.distributed.tensor.parallel import loss_parallel
mesh = init_device_mesh("neuron", (W,))
g = lambda s: torch.Generator().manual_seed(s)
x = torch.randn(32, 64, generator=g(0)); y = torch.randint(0, 64, (32,), generator=g(1)); y[::5] = -100
rel = lambda a, b: ((a.detach().float().cpu() - b.detach().float().cpu()).norm() / a.detach().float().cpu().norm()).item()
ls = 0.1
xc = x.clone().requires_grad_(); lc = F.cross_entropy(xc, y, label_smoothing=ls); lc.backward()
l0 = F.cross_entropy(x, y)
xd = distribute_tensor(x.to(dev), mesh, [Shard(1)]).requires_grad_()
with loss_parallel():
    ln = F.cross_entropy(xd, y.to(dev), label_smoothing=ls)
lnf = ln.full_tensor()
if r == 0:
    print(f"RES cpu ls=0.1 loss {lc.item():.5f}, cpu ls=0 loss {l0.item():.5f}, neuron loss_parallel ls=0.1 {lnf.item():.5f}")
# proposed fix: (1-ls)*nll + ls*smooth, both over valid rows
xd2 = distribute_tensor(x.to(dev), mesh, [Shard(1)]).requires_grad_()
yd = DTensor.from_local(y.to(dev), mesh, [Replicate()], run_check=False)
with loss_parallel():
    lsm = F.log_softmax(xd2, dim=1)
    nll = F.nll_loss(lsm, yd, ignore_index=-100)
    valid = (y != -100).to(dev).float()
    vd = DTensor.from_local(valid, mesh, [Replicate()], run_check=False)
    smooth = -(lsm.mean(dim=1) * vd).sum() / valid.sum()
    lf_d = (1 - ls) * nll + ls * smooth
    lf = lf_d.full_tensor()
    lf_d.backward()
gfull = xd2.grad.full_tensor(); torch.neuron.synchronize()
if r == 0:
    print(f"RES fix: loss rel {rel(lc, lf):.3g} grad rel {rel(xc.grad, gfull):.3g}")
dist.destroy_process_group()
