import os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp
mode = sys.argv[1]

def worker(rank, W, port, q):
    import torch_neuronx
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    try:
        torch.neuron.set_device(rank); dev = torch.device("neuron", rank)
        dist.init_process_group("neuron", rank=rank, world_size=W)
        t = torch.full((4,), float(rank), device=dev); dist.all_reduce(t); torch.neuron.synchronize()
        q.put((rank, t.cpu().tolist()[0], str(t.device)))
        dist.destroy_process_group()
    except Exception as e:
        q.put((rank, "ERR " + str(e)[:150], ""))

def child_after_init(q):
    try:
        x = torch.ones(3, device="neuron") * 2; torch.neuron.synchronize(); q.put(("child", x.cpu().tolist()))
    except Exception as e:
        q.put(("child", "ERR " + type(e).__name__ + ": " + str(e)[:150]))

if __name__ == "__main__":
    if mode == "spawn":
        ctx = mp.get_context("spawn"); q = ctx.Queue(); port = 29000 + os.getpid() % 1000
        mp.spawn(worker, args=(4, port, q), nprocs=4, join=True)
        print("RES spawn:", sorted([q.get() for _ in range(4)]))
    elif mode == "fork_after_init":
        import torch_neuronx
        a = torch.ones(2, device="neuron"); torch.neuron.synchronize()
        ctx = mp.get_context("fork"); q = ctx.Queue()
        p = ctx.Process(target=child_after_init, args=(q,)); p.start(); p.join(120)
        print("RES fork after init: child exit", p.exitcode, "msg", q.get(timeout=5) if not q.empty() else "no message")
        print("RES parent still works:", (a * 3).cpu().tolist())
    elif mode == "dataloader_workers":
        import torch_neuronx
        a = torch.ones(2, device="neuron"); torch.neuron.synchronize()
        ds = torch.utils.data.TensorDataset(torch.arange(64.0).view(32, 2))
        dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=2, pin_memory=True)
        tot = 0.0
        for (b,) in dl:
            tot += (b.to("neuron") * 2).sum().item()
        print("RES DataLoader(num_workers=2, fork) after neuron init: sum", tot, "expect", 2 * sum(range(64)))
