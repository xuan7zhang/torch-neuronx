import torch, torch_neuronx
DEV = "neuron"; nm = torch.neuron
def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
s = nm.Stream()
print("streams:", nm.current_stream(), s)
# 1) correct pattern: produce on default, wait, consume on side
a = torch.randn(256, 256, device=DEV)
s.wait_stream(nm.current_stream())
with nm.stream(s):
    b = a * 2
nm.current_stream().wait_stream(s)
print("1 wait then side mul:", rel(a.cpu() * 2, b))
# 2) same as original probe (no wait): CUDA calls this a race
a2 = torch.randn(256, 256, device=DEV)
with nm.stream(s):
    b2 = a2 * 2
nm.synchronize()
print("2 no wait (race on CUDA):", rel(a2.cpu() * 2, b2))
# 3) value in side-stream result vs a2 itself
print("   b2 absmax", b2.abs().max().item(), "a2 absmax", a2.abs().max().item())
# 4) producer on side, consumer default without wait
with nm.stream(s):
    c = torch.ones(256, 256, device=DEV) * 3
d = c + 1
nm.synchronize()
print("4 side produce, default consume no wait:", d.mean().item(), "(expect 4)")
# 5) host-sourced tensor used on side stream
x = torch.randn(256, 256)
with nm.stream(s):
    xn = x.to(DEV)
    y = xn * 2
nm.synchronize()
print("5 H2D + mul inside side stream:", rel(x * 2, y))
