import os, sys, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["LOCAL_RANK"]); w = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron"); c = sys.argv[1]; res = "?"
try:
    if c == "ag_into":
        out = torch.empty(w * 3, 2, device=dev); dist.all_gather_into_tensor(out, torch.full((3, 2), float(r), device=dev))
        res = (out.cpu().view(w, 3, 2) == torch.arange(w).float().view(w, 1, 1)).all().item()
    elif c == "rs_list":
        out = torch.empty(3, device=dev); dist.reduce_scatter(out, [torch.full((3,), float(r * 10 + i), device=dev) for i in range(w)])
        res = (out.cpu() == sum(q * 10 + r for q in range(w))).all().item()
    elif c == "a2a_even":
        inp = torch.arange(w * 2, dtype=torch.float32, device=dev) + r * 100; out = torch.empty_like(inp); dist.all_to_all_single(out, inp)
        res = torch.equal(out.cpu(), torch.tensor([q * 100 + r * 2 + k for q in range(w) for k in range(2)], dtype=torch.float32))
    elif c == "a2a_uneven":
        ins = [j + 1 for j in range(w)]; outs = [r + 1] * w
        inp = torch.cat([torch.full((j + 1,), float(r * 10 + j)) for j in range(w)]).to(dev); out = torch.empty(sum(outs), device=dev)
        dist.all_to_all_single(out, inp, outs, ins); res = torch.equal(out.cpu(), torch.cat([torch.full((r + 1,), float(q * 10 + r)) for q in range(w)]))
    elif c == "a2a_list":
        outs = [torch.empty(2, device=dev) for _ in range(w)]; dist.all_to_all(outs, [torch.full((2,), float(r * 10 + j), device=dev) for j in range(w)])
        res = all((o.cpu() == q * 10 + r).all().item() for q, o in enumerate(outs))
    elif c == "gather":
        gl = [torch.empty(2, device=dev) for _ in range(w)] if r == 0 else None; dist.gather(torch.full((2,), float(r), device=dev), gl, dst=0)
        res = r != 0 or all((g.cpu() == i).all().item() for i, g in enumerate(gl))
    elif c == "scatter":
        o = torch.empty(2, device=dev); dist.scatter(o, [torch.full((2,), float(i), device=dev) for i in range(w)] if r == 1 else None, src=1); res = (o.cpu() == r).all().item()
    elif c == "reduce":
        t = torch.full((2,), float(r + 1), device=dev); dist.reduce(t, dst=2 % w); res = r != 2 % w or (t.cpu() == sum(range(1, w + 1))).all().item()
    elif c == "isend_irecv":
        t = torch.full((3,), float(r), device=dev); rr = torch.empty(3, device=dev)
        for q in [dist.isend(t, (r + 1) % w), dist.irecv(rr, (r - 1) % w)]: q.wait()
        res = (rr.cpu() == (r - 1) % w).all().item()
    elif c == "send_recv":
        if r == 0: dist.send(torch.arange(4.0, device=dev), 1); res = True
        elif r == 1: rr = torch.empty(4, device=dev); dist.recv(rr, 0); res = torch.equal(rr.cpu(), torch.arange(4.0))
        else: res = True
    elif c == "batch_p2p":
        rr = torch.empty(2, device=dev)
        for q in dist.batch_isend_irecv([dist.P2POp(dist.isend, torch.full((2,), float(r), device=dev), (r + 1) % w), dist.P2POp(dist.irecv, rr, (r - 1) % w)]): q.wait()
        res = (rr.cpu() == (r - 1) % w).all().item()
    elif c == "objects":
        objs = [None] * w; dist.all_gather_object(objs, {"r": r}); ol = [7] if r == 0 else [None]; dist.broadcast_object_list(ol, src=0)
        res = objs == [{"r": i} for i in range(w)] and ol == [7]
    elif c == "async_large":
        t = torch.ones(1000, device=dev); dist.all_reduce(t, async_op=True).wait(); t2 = torch.ones(8 * 1024 * 1024, device=dev) * (r + 1); dist.all_reduce(t2)
        res = (t.cpu() == w).all().item() and (t2.cpu() == sum(range(1, w + 1))).all().item()
    elif c == "ag_uneven_list":
        outs = [torch.empty(i + 1, device=dev) for i in range(w)]; dist.all_gather(outs, torch.full((r + 1,), float(r), device=dev))
        res = all((o.cpu() == i).all().item() for i, o in enumerate(outs))
    elif c == "ag_list_even":
        outs = [torch.empty(3, device=dev) for i in range(w)]; dist.all_gather(outs, torch.full((3,), float(r), device=dev))
        res = all((o.cpu() == i).all().item() for i, o in enumerate(outs))
    torch.neuron.synchronize()
except Exception as e:
    res = "ERR " + str(e).replace("\n", " ")[:160]
outs = [None] * w
try: dist.all_gather_object(outs, res)
except Exception as e: outs = [res, f"gather_object failed: {str(e)[:80]}"]
if r == 0: print("RES", f"{c:15s}", outs, flush=True)
dist.destroy_process_group()
