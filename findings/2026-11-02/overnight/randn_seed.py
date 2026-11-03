import torch, torch_neuronx
D = "neuron"
torch.ones(1, device=D).cpu()
for n in [16, 4096]:
    torch.manual_seed(123); a = torch.randn(n, device=D).cpu()
    torch.manual_seed(123); b = torch.randn(n, device=D).cpu()
    torch.manual_seed(123); c = torch.randn(n, device=D).cpu()
    print("RES randn n", n, "a==b", torch.equal(a, b), "b==c", torch.equal(b, c), "a[:3]", a[:3].tolist(), "b[:3]", b[:3].tolist())
for nm, f in [("rand", lambda: torch.rand(8, device=D)), ("randn_like", lambda: torch.randn_like(torch.empty(8, device=D))), ("normal_", lambda: torch.empty(8, device=D).normal_()), ("randn", lambda: torch.randn(8, device=D))]:
    torch.manual_seed(5); a = f().cpu(); torch.manual_seed(5); b = f().cpu()
    print("RES", nm, "reseed repeats:", torch.equal(a, b))
# init reproducibility of a model on device
def init():
    torch.manual_seed(0); return torch.nn.Linear(64, 64, device=D).weight.cpu()
print("RES nn.Linear(device=neuron) init reproducible:", torch.equal(init(), init()))
print("RES fallback", sorted(torch_neuronx.get_fallback_ops()))
