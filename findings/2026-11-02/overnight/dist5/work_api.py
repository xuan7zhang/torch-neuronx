import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
t = torch.ones(4, device=dev)
w = dist.all_reduce(t, async_op=True)
res = {"type": type(w).__module__ + "." + type(w).__name__, "isinstance dist.Work": isinstance(w, dist.Work)}
for name, call in [("wait()", lambda: w.wait()), ("is_completed()", lambda: w.is_completed()), ("is_success()", lambda: w.is_success()),
                   ("exception()", lambda: w.exception()), ("result()", lambda: [x.shape for x in w.result()]),
                   ("source_rank()", lambda: w.source_rank()), ("get_future().wait()", lambda: type(w.get_future().wait()).__name__),
                   ("get_future_result().wait()", lambda: w.get_future_result().wait()), ("boxed()", lambda: type(w.boxed()).__name__),
                   ("synchronize()", lambda: w.synchronize()), ("block_current_stream()", lambda: w.block_current_stream())]:
    try:
        res[name] = repr(call())[:60]
    except Exception as e:
        res[name] = f"ERR {type(e).__name__}: {str(e)[:70]}".replace("\n", " ")
try:
    from torch.distributed.distributed_c10d import _get_default_group
    pg = _get_default_group()
    w2 = pg.allreduce([torch.ones(2, device=dev)])
    res["pg.allreduce returns"] = type(w2).__name__ + f" isinstance Work {isinstance(w2, dist.Work)}"
except Exception as e:
    res["pg.allreduce returns"] = f"ERR {str(e)[:80]}"
torch.neuron.synchronize()
if r == 0:
    for k, v in res.items(): print("RES", k, "->", v)
dist.destroy_process_group()
