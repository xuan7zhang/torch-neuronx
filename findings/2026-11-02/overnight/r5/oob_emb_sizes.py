import sys, torch, torch.nn.functional as F, torch_neuronx
nm = torch.neuron; D = "neuron"
V, Dm, grad = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "grad"
w = (torch.arange(V * Dm, dtype=torch.float32).view(V, Dm) % 97 + 1).to(D).requires_grad_(grad)
ids = torch.tensor([[1, V, V + 5, 2]], device=D)
try:
    e = F.embedding(ids, w)
    if grad:
        e.sum().backward()
    nm.synchronize()
    print(f"RES V={V} D={Dm} grad={grad}: no error, rows for OOB ids = {e[0, 1:3, :3].detach().cpu().tolist()}")
except Exception as ex:
    print(f"RES V={V} D={Dm} grad={grad}: raises ({str(ex)[60:110]})".replace("\n", " "))
