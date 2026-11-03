"""Distributed round 5, batch E: FSDP1, ZeroRedundancyOptimizer, DDP comm hooks. torchrun --nproc-per-node 4 probe_dist5e.py <case>
DEVN=cpu runs the same case on CPU gloo."""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn

DEVN = os.environ.get("DEVN", "neuron")
if DEVN == "neuron":
    import torch_neuronx  # noqa: F401
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
if DEVN == "neuron":
    torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
else:
    dev = torch.device("cpu")
dist.init_process_group("cpu:gloo,neuron:neuron" if DEVN == "neuron" else "gloo")
CASES = {}


def sync():
    if DEVN == "neuron":
        torch.neuron.synchronize()


def case(f):
    CASES[f.__name__] = f
    return f


def g(s):
    return torch.Generator().manual_seed(s)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.l1 = nn.Linear(32, 64); s.l2 = nn.Linear(64, 32); s.n = nn.LayerNorm(32)

    def forward(s, x):
        return x + s.l2(torch.relu(s.l1(s.n(x))))


class Net(nn.Module):
    def __init__(s):
        super().__init__(); torch.manual_seed(0)
        s.inp = nn.Linear(16, 32); s.blocks = nn.Sequential(Blk(), Blk(), Blk()); s.out = nn.Linear(32, 4)

    def forward(s, x):
        return s.out(s.blocks(s.inp(x)))


def batch(st, r):
    return torch.randn(8, 16, generator=g(st * 10 + r))


def ref_params(steps=3, opt=lambda p: torch.optim.AdamW(p, lr=1e-2)):
    m = Net(); o = opt(m.parameters())
    for st in range(steps):
        o.zero_grad(); (sum(m(batch(st, r)).pow(2).mean() for r in range(W)) / W).backward(); o.step()
    return [p.detach() for p in m.parameters()]


@case
def fsdp1_variants():
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    ref = ref_params()
    out = {}
    cfgs = {
        "full_shard": dict(sharding_strategy=ShardingStrategy.FULL_SHARD),
        "shard_grad_op": dict(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP),
        "no_shard": dict(sharding_strategy=ShardingStrategy.NO_SHARD),
        "use_orig_params": dict(use_orig_params=True),
        "wrap_blocks": dict(auto_wrap_policy=ModuleWrapPolicy({Blk})),
        "mp_bf16": dict(mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)),
        "sync_module_states": dict(sync_module_states=True),
        "limit_all_gathers_wrap": dict(auto_wrap_policy=ModuleWrapPolicy({Blk}), limit_all_gathers=True, forward_prefetch=True),
    }
    for name, kw in cfgs.items():
        try:
            m = Net()
            if name == "sync_module_states" and rank != 0:
                with torch.no_grad():
                    for p in m.parameters():
                        p.mul_(0).add_(rank)  # wrong init on other ranks; sync must fix it
            f = FSDP(m.to(dev), device_id=dev, **kw)
            o = torch.optim.AdamW(f.parameters(), lr=1e-2)
            for st in range(3):
                o.zero_grad(); f(batch(st, rank).to(dev)).float().pow(2).mean().backward(); o.step()
            sync()
            with FSDP.summon_full_params(f):
                got = [p.detach().clone() for p in f.parameters()]
            out[name] = f"{max(rel(a, b) for a, b in zip(ref, got)):.3g}" if len(got) == len(ref) else f"n_params {len(got)} vs {len(ref)}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
    return out


@case
def fsdp1_state_dicts():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    out = {}
    f = FSDP(Net().to(dev), device_id=dev, auto_wrap_policy=ModuleWrapPolicy({Blk}))
    o = torch.optim.AdamW(f.parameters(), lr=1e-2)
    for st in range(2):
        o.zero_grad(); f(batch(st, rank).to(dev)).pow(2).mean().backward(); o.step()
    ref = ref_params(2)
    with FSDP.state_dict_type(f, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        sd = f.state_dict()
        osd = FSDP.optim_state_dict(f, o)
    if rank == 0:
        keys = [k for k in sd if not k.endswith("num_batches_tracked")]
        out["full_sd_matches_ref"] = f"{max(rel(a, sd[k]) for a, k in zip(ref, keys)):.3g}"
        out["full_osd_state_entries"] = len(osd["state"])
    with FSDP.state_dict_type(f, StateDictType.SHARDED_STATE_DICT):
        ssd = f.state_dict()
        out["sharded_sd_type"] = type(next(iter(ssd.values()))).__name__
    # load full sd into a fresh FSDP model and compare one step
    f2 = FSDP(Net().to(dev), device_id=dev, auto_wrap_policy=ModuleWrapPolicy({Blk}))
    full_sd = [sd] if rank == 0 else [None]
    dist.broadcast_object_list(full_sd, src=0)
    with FSDP.state_dict_type(f2, StateDictType.FULL_STATE_DICT):
        f2.load_state_dict(full_sd[0])
    x = batch(9, rank).to(dev)
    with torch.no_grad():
        d = (f(x) - f2(x)).abs().max().item()
    out["reload_full_sd_out_diff"] = d
    return out


@case
def zero_optimizer():
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP
    ref = ref_params(3)
    m = Net().to(dev); d = DDP(m)
    o = ZeroRedundancyOptimizer(d.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-2)
    for st in range(3):
        o.zero_grad(); d(batch(st, rank).to(dev)).pow(2).mean().backward(); o.step()
    sync()
    out = {"params": max(rel(a, b) for a, b in zip(ref, m.parameters()))}
    o.consolidate_state_dict(to=0)
    if rank == 0:
        out["consolidated_state_entries"] = len(o.state_dict()["state"])
    return out


@case
def ddp_comm_hooks():
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook
    from torch.nn.parallel import DistributedDataParallel as DDP
    out = {}
    ref = ref_params(3, opt=lambda p: torch.optim.SGD(p, lr=0.05))
    for name in ("allreduce", "powersgd_rank4_start_iter2", "post_localsgd"):
        try:
            m = Net().to(dev); d = DDP(m)
            if name == "allreduce":
                d.register_comm_hook(None, default_hooks.allreduce_hook)
            elif name.startswith("powersgd"):
                st_ = powerSGD_hook.PowerSGDState(process_group=None, matrix_approximation_rank=4, start_powerSGD_iter=2)
                d.register_comm_hook(st_, powerSGD_hook.powerSGD_hook)
            else:
                from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as pls
                from torch.distributed.algorithms.model_averaging.averagers import PeriodicModelAverager
                subgroup, _ = dist.new_subgroups()
                st_ = pls.PostLocalSGDState(process_group=None, subgroup=subgroup, start_localSGD_iter=2)
                d.register_comm_hook(st_, pls.post_localSGD_hook)
                avg = PeriodicModelAverager(period=1, warmup_steps=2)
            o = torch.optim.SGD(d.parameters(), lr=0.05)
            for s in range(3):
                o.zero_grad(); d(batch(s, rank).to(dev)).pow(2).mean().backward(); o.step()
                if name == "post_localsgd":
                    avg.average_parameters(d.parameters())
            sync()
            ps = [p.detach().cpu() for p in m.parameters()]
            same = [torch.zeros(1) + rel(ps[0], ps[0])]
            p0 = ps[0].clone().to(dev); dist.broadcast(p0, src=0); sync()
            out[name] = f"vs_ref {max(rel(a, b) for a, b in zip(ref, ps)):.3g} ranks_equal_rel {rel(ps[0], p0):.2g}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:140]}".replace("\n", " ")
    return out


if __name__ == "__main__":
    c = sys.argv[1]
    try:
        print(f"R{rank} {c}: {CASES[c]()}", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"R{rank} {c}: ERROR {type(e).__name__}: {str(e)[:300]}".replace("\n", " "), flush=True)
    try:
        dist.destroy_process_group()
    except Exception:
        pass
