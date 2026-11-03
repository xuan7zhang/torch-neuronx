import os, torch, torch.nn as nn, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("cpu:gloo,neuron:neuron")
from torch.nn.parallel import DistributedDataParallel as DDP
xs = [torch.randn(8, 4, generator=torch.Generator().manual_seed(50 + q)) * 3 + 2 for q in range(W)]
ref = nn.BatchNorm1d(4); ref.train(); ref(torch.cat(xs))
bn = nn.SyncBatchNorm(4).to(dev); d = DDP(nn.Sequential(bn)); d.train()
y = d(xs[r].to(dev)); y.sum().backward(); torch.neuron.synchronize()
plain = nn.BatchNorm1d(4).to(dev); plain.train(); plain(xs[r].to(dev))
ref_local = nn.BatchNorm1d(4); ref_local(xs[r])
if r == 0:
    print("ref running_mean     ", ref.running_mean.tolist())
    print("syncbn running_mean  ", bn.running_mean.cpu().tolist())
    print("syncbn running_var   ", bn.running_var.cpu().tolist(), "ref", ref.running_var.tolist())
    print("syncbn num_batches   ", int(bn.num_batches_tracked))
    print("plain BN running_mean", plain.running_mean.cpu().tolist(), "cpu", ref_local.running_mean.tolist())
dist.destroy_process_group()
