import torch, torch_neuronx
torch.ones(1, device="neuron").cpu()
N = 1 << 24
for pinned in [False, True]:
    for trial in range(3):
        try:
            src = torch.zeros(N)
            if pinned: src = src.pin_memory()
        except Exception as e:
            print("RES pin_memory ERR", str(e)[:120]); break
        busy = torch.randn(2048, 2048, device="neuron")
        for _ in range(20): busy = busy @ busy * 1e-3  # keep the device queue busy
        d = src.to("neuron", non_blocking=True)
        src.fill_(7.0)  # mutate the source right after the async copy
        got = d.cpu()
        print("RES H2D non_blocking pinned=%s trial %d: device copy all zeros=%s (frac of 7s %.3f)" % (pinned, trial, bool((got == 0).all()), (got == 7).float().mean().item()))
# D2H non_blocking then read
x = torch.full((N,), 3.0, device="neuron")
h = x.to("cpu", non_blocking=True)
print("RES D2H non_blocking immediate read correct:", bool((h == 3).all()))
# DataLoader with pin_memory and pin_memory_device
ds = torch.utils.data.TensorDataset(torch.arange(64.0).view(16, 4))
try:
    dl = torch.utils.data.DataLoader(ds, batch_size=4, pin_memory=True)
    b = next(iter(dl))[0]
    print("RES DataLoader pin_memory=True ok, is_pinned:", b.is_pinned())
except Exception as e:
    print("RES DataLoader pin_memory ERR", str(e)[:150])
