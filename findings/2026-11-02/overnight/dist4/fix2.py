import os, sys, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("cpu:gloo,neuron:neuron")
g = lambda s: torch.Generator().manual_seed(s)
def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu(); return ((a - b).norm() / (a.norm() + 1e-12)).item()
def mlp():
    torch.manual_seed(0); return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))
c = sys.argv[1]
if c == "osd":
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
    def build():
        m = mlp().to(dev)
        for i in (0, 2, 4): fully_shard(m[i])
        fully_shard(m); return m, torch.optim.AdamW(m.parameters(), lr=1e-2)
    m, o = build()
    for s in range(2):
        o.zero_grad(); m(torch.randn(8, 16, generator=g(s * 10 + r)).to(dev)).pow(2).mean().backward(); o.step()
    res = {}
    for name, opts_get, opts_set in [
        ("sharded", StateDictOptions(), StateDictOptions()),
        ("full_bcast", StateDictOptions(full_state_dict=True, cpu_offload=True), StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)),
    ]:
        msd, osd = get_state_dict(m, o, options=opts_get)
        m2, o2 = build()
        set_state_dict(m2, o2, model_state_dict=msd, optim_state_dict=osd, options=opts_set)
        x = torch.randn(8, 16, generator=g(99 + r)).to(dev)
        import copy
        ma, oa = m, o
        # compare one more step on a copy of the original vs the loaded one
        for mm, oo in ((m2, o2),):
            oo.zero_grad(); mm(x).pow(2).mean().backward(); oo.step()
        mb, ob = build()
        msd0, osd0 = get_state_dict(m, o, options=StateDictOptions())
        set_state_dict(mb, ob, model_state_dict=msd0, optim_state_dict=osd0, options=StateDictOptions())
        ob.zero_grad(); mb(x).pow(2).mean().backward(); ob.step()
        res[name] = max(rel(a.full_tensor(), b.full_tensor()) for a, b in zip(mb.parameters(), m2.parameters()))
    # direct check: loaded exp_avg equals original exp_avg
    m3, o3 = build()
    msd, osd = get_state_dict(m, o, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    set_state_dict(m3, o3, model_state_dict=msd, optim_state_dict=osd, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    s_orig = [o.state[p]["exp_avg"].full_tensor() for p in m.parameters()]
    s_new = [o3.state[p]["exp_avg"].full_tensor() for p in m3.parameters()]
    res["exp_avg_roundtrip_full_bcast"] = max(rel(a, b) for a, b in zip(s_orig, s_new))
    res["step_roundtrip"] = str((o.state[next(iter(m.parameters()))]["step"].item(), o3.state[next(iter(m3.parameters()))]["step"].item()))
    print(f"R{r} osd: {res}", flush=True)
if c == "cp":
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    mesh = init_device_mesh("neuron", (W,))
    S, Dm, H = 64 * W, 32, 2
    torch.manual_seed(0); proj = nn.Linear(Dm, 3 * Dm)
    x = torch.randn(1, S, Dm, generator=g(0))
    def attn(m, t):
        q, k, v = m(t).view(1, -1, 3, H, Dm // H).permute(2, 0, 3, 1, 4)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(1, -1, Dm)
    ref_out = attn(proj, x); go = torch.randn(ref_out.shape, generator=g(1)); (ref_out * go).sum().backward()
    pn = nn.Linear(Dm, 3 * Dm).to(dev); pn.load_state_dict(proj.state_dict())
    xn = x.to(dev); gon = go.to(dev)
    try:
        with context_parallel(mesh, buffers=[xn, gon], buffer_seq_dims=[1, 1]):
            o = attn(pn, xn); (o * gon).sum().backward()
        torch.neuron.synchronize()
        # xn now holds this rank's shard (load-balanced order); compare grads of proj (all-reduced manually)
        gw = pn.weight.grad.clone(); dist.all_reduce(gw)
        print(f"R{r} cp: ok, shard {tuple(o.shape)}, proj.weight.grad rel={rel(proj.weight.grad, gw):.3g}", flush=True)
    except Exception as e:
        import traceback; tb = traceback.format_exc().splitlines()
        print(f"R{r} cp: ERR {type(e).__name__}: {str(e)[:200]} | at {[l for l in tb if 'File' in l][-1][:160]}".replace("\n", " "), flush=True)
dist.destroy_process_group()
