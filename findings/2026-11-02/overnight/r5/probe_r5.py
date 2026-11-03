"""Round 5 probes. Run one case per process: python probe_r5.py <case>. Prints RESULT lines."""
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def g(s):
    return torch.Generator().manual_seed(s)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


@case
def big_numel_bf16():
    n = 2**31 + 4096  # > int32 elements, 4.3 GB bf16
    x = torch.zeros(n, dtype=torch.bfloat16, device=DEV)
    x[-3:] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    x[2**31 - 1] = 5.0
    x[5] = 7.0
    tail = x[-3:].cpu().tolist()
    v = x[2**31 - 1].item()
    head = x[5].item()
    lo = x[:8].cpu().tolist()
    s = x.float().sum().item() if False else None
    nz = int((x[-4096:] != 0).sum().item())
    return {"tail": tail, "at_2^31-1": v, "x[5]": head, "x[:8]": lo, "nonzero_in_last4096": nz,
            "expect": "tail [1,2,3], 5.0, 7.0, x[:8] has 7 at 5 only, nonzero 3"}


@case
def big_embedding_rows():
    V, D = 140000, 16384  # 2.29e9 elements bf16 = 4.6 GB, > 2^31
    w = torch.zeros(V, D, dtype=torch.bfloat16, device=DEV)
    rows = torch.tensor([0, 1, 131072, 139999], device=DEV)
    w[rows] = torch.arange(1, 5, dtype=torch.bfloat16, device=DEV)[:, None].expand(4, D)
    e = F.embedding(torch.tensor([[139999, 131072, 0, 5]], device=DEV), w)
    got = e[0, :, :2].cpu().tolist()
    last = w[139999, -4:].cpu().tolist()
    return {"emb_first2cols": got, "w[last,-4:]": last, "expect": "[[4,4],[3,3],[1,1],[0,0]], [4]*4"}


@case
def big_embedding_bwd():
    V, D = 140000, 16384
    w = torch.zeros(V, D, dtype=torch.bfloat16, device=DEV, requires_grad=True)
    idx = torch.tensor([[139999, 131072, 0, 139999]], device=DEV)
    F.embedding(idx, w).sum().backward()
    gr = w.grad
    return {"g[139999,:2]": gr[139999, :2].cpu().tolist(), "g[131072,-2:]": gr[131072, -2:].cpu().tolist(),
            "g[0,0]": gr[0, 0].item(), "g[1,0]": gr[1, 0].item(), "expect": "[2,2], [1,1], 1, 0"}


@case
def optimizer_in_backward():
    res = {}
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).to(dev)
        opts = {p: torch.optim.AdamW([p], lr=1e-2, foreach=False) for p in m.parameters()}

        def hook(p):
            opts[p].step()
            opts[p].zero_grad()
        for p in m.parameters():
            p.register_post_accumulate_grad_hook(hook)
        for i in range(3):
            m(torch.randn(8, 16, generator=g(i)).to(dev)).pow(2).mean().backward()
        res[dev] = [p.detach() for p in m.parameters()]
    return {"params": max(rel(a, b) for a, b in zip(res["cpu"], res[DEV]))}


@case
def moe_routing():
    res = {}
    E, k, T, D = 8, 2, 64, 16
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        router = nn.Linear(D, E).to(dev)
        experts = nn.ModuleList([nn.Linear(D, D) for _ in range(E)]).to(dev)
        x = torch.randn(T, D, generator=g(1)).to(dev).requires_grad_()
        logits = router(x)
        probs = logits.softmax(-1)
        w, idx = probs.topk(k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        flat = idx.flatten()
        order = flat.argsort(stable=True)
        counts = torch.bincount(flat, minlength=E)
        tok = order // k
        out = torch.zeros_like(x)
        start = 0
        for e in range(E):
            c = int(counts[e])
            sel = tok[start:start + c]
            ws = w.flatten()[order[start:start + c]]
            out = out.index_add(0, sel, experts[e](x[sel]) * ws[:, None])
            start += c
        aux = (probs.mean(0) * counts.float() / (T * k)).sum() * E
        loss = out.pow(2).mean() + 0.01 * aux
        loss.backward()
        res[dev] = [loss.detach(), counts, x.grad, router.weight.grad] + [p.grad for p in experts.parameters()]
    r = {"loss": rel(res["cpu"][0], res[DEV][0]), "counts_equal": float(torch.equal(res["cpu"][1].cpu(), res[DEV][1].cpu()))}
    r["x_grad"] = rel(res["cpu"][2], res[DEV][2])
    r["router_grad"] = rel(res["cpu"][3], res[DEV][3])
    r["expert_grads"] = max(rel(a, b) if a is not None else 0.0 for a, b in zip(res["cpu"][4:], res[DEV][4:]))
    return r


@case
def routing_ops():
    x = torch.randint(0, 50, (1000,), generator=g(0))
    v = torch.randn(1000, generator=g(1))
    out = {}
    xn, vn = x.to(DEV), v.to(DEV)
    out["bincount"] = torch.equal(torch.bincount(x), torch.bincount(xn).cpu())
    out["bincount_w"] = rel(torch.bincount(x, weights=v), torch.bincount(xn, weights=vn))
    out["argsort_stable"] = torch.equal(x.argsort(stable=True), xn.argsort(stable=True).cpu())
    out["sort_stable_vals"] = torch.equal(x.sort(stable=True)[0], xn.sort(stable=True)[0].cpu())
    u, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
    un, invn, cntn = torch.unique(xn, return_inverse=True, return_counts=True)
    out["unique"] = torch.equal(u, un.cpu()) and torch.equal(inv, invn.cpu()) and torch.equal(cnt, cntn.cpu())
    b = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])
    out["bucketize"] = torch.equal(torch.bucketize(x.float(), b), torch.bucketize(xn.float(), b.to(DEV)).cpu())
    out["searchsorted_right"] = torch.equal(torch.searchsorted(b, x.float(), right=True),
                                            torch.searchsorted(b.to(DEV), xn.float(), right=True).cpu())
    out["histc"] = rel(torch.histc(v, 20, -3, 3), torch.histc(vn, 20, -3, 3))
    out["cumsum_int"] = torch.equal(torch.bincount(x).cumsum(0), torch.bincount(xn).cumsum(0).cpu())
    out["nonzero"] = torch.equal((x > 25).nonzero(), (xn > 25).nonzero().cpu())
    out["masked_select"] = torch.equal(v[x > 25], vn[xn > 25].cpu())
    out["one_hot"] = torch.equal(F.one_hot(x, 50), F.one_hot(xn, 50).cpu())
    out["scatter_add_int"] = torch.equal(torch.zeros(50, dtype=torch.long).scatter_add(0, x, torch.ones_like(x)),
                                         torch.zeros(50, dtype=torch.long, device=DEV).scatter_add(0, xn, torch.ones_like(xn)).cpu())
    out["topk_ties_idx_sorted"] = torch.equal(torch.ones(16).topk(4)[1].sort()[0].numel() * torch.ones(1),
                                              torch.ones(16, device=DEV).topk(4)[1].sort()[0].numel() * torch.ones(1))
    return out


@case
def write_after_read_hazard():
    out = {}
    x = torch.randn(512, 512, generator=g(0))
    xn = x.to(DEV)
    y = xn @ xn  # pending
    xn.copy_(torch.zeros(512, 512))  # H2D write right after
    out["mm_then_h2d_copy"] = rel(x @ x, y)
    xn = x.to(DEV)
    y = xn * 2
    xn.fill_(0)
    out["mul_then_fill"] = rel(x * 2, y)
    xn = x.to(DEV)
    y = xn.sum(0)
    xn.data = torch.zeros(512, 512, device=DEV)
    out["sum_then_data_swap"] = rel(x.sum(0), y)
    xn = x.to(DEV)
    v = xn[0]
    y = v * 3
    xn[0] = 0
    out["view_read_then_write"] = rel(x[0] * 3, y)
    c = torch.empty(512, 512)
    xn = x.to(DEV)
    zz = xn * 5
    c.copy_(zz)
    out["d2h_copy_into_cpu"] = rel(x * 5, c)
    return out


@case
def streams():
    out = {}
    nm = torch.neuron
    out["has_Stream"] = hasattr(nm, "Stream")
    if not hasattr(nm, "Stream"):
        return out
    x = torch.randn(1024, 1024, generator=g(0))
    xn = x.to(DEV)
    s = nm.Stream()
    s.wait_stream(nm.current_stream())
    with nm.stream(s):
        y = xn @ xn
        y2 = y + 1
    nm.current_stream().wait_stream(s)
    out["side_stream_mm"] = rel(x @ x + 1, y2)
    e = nm.Event()
    with nm.stream(s):
        z = xn * 3
        e.record()
    e.wait()
    out["event_wait"] = rel(x * 3, z)
    a = torch.randn(256, 256, device=DEV)
    with nm.stream(s):
        b = a * 2
    a.record_stream(s) if hasattr(a, "record_stream") else None
    nm.synchronize()
    out["record_stream"] = rel(a.cpu() * 2, b)
    return out


@case
def tied_weights():
    res = {}
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        emb = nn.Embedding(100, 32)
        head = nn.Linear(32, 100, bias=False)
        head.weight = emb.weight
        m = nn.ModuleDict({"emb": emb, "head": head}).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for i in range(3):
            ids = torch.randint(0, 100, (4, 10), generator=g(i)).to(dev)
            opt.zero_grad()
            F.cross_entropy(m["head"](m["emb"](ids)).flatten(0, 1), ids.flatten()).backward()
            opt.step()
        res[dev] = (m["emb"].weight.detach(), m["head"].weight is m["emb"].weight, len(list(m.parameters())))
    return {"w": rel(res["cpu"][0], res[DEV][0]), "still_tied": float(res[DEV][1]), "n_params": float(res[DEV][2])}


@case
def backward_hooks_and_expand_grad():
    res = {}
    for dev in ("cpu", DEV):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).to(dev)
        m[2].register_full_backward_hook(lambda mod, gi, go: (gi[0] * 0.5,))
        m[0].weight.register_hook(lambda gr: gr.clamp(-0.01, 0.01))
        x = torch.randn(8, 16, generator=g(0)).to(dev).requires_grad_()
        y = m(x)
        go = torch.randn(1, 4, generator=g(1)).to(dev).expand(8, 4)  # zero-stride grad
        y.backward(go, retain_graph=True)
        y.backward(go)  # accumulate twice
        gx, = torch.autograd.grad(m(x).sum(), x)
        res[dev] = [x.grad, gx] + [p.grad for p in m.parameters()]
    return {"worst": max(rel(a, b) for a, b in zip(res["cpu"], res[DEV]))}


def main():
    name = sys.argv[1]
    try:
        r = CASES[name]()
        torch.neuron.synchronize()
        print(f"RESULT {name}: {r}", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"RESULT {name}: ERROR {type(e).__name__}: {str(e)[:300]}".replace("\n", " "), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(" ".join(CASES))
    else:
        main()
