import torch, torch.nn.functional as F, torch_neuronx
torch.manual_seed(5)
st = torch.neuron.get_rng_state()
a = F.dropout(torch.ones(16, device="neuron"), 0.5)
torch.neuron.set_rng_state(st)
b = F.dropout(torch.ones(16, device="neuron"), 0.5)
print("state", st.dtype, st.shape, st[:16].tolist())
print("a", a.cpu().tolist()); print("b", b.cpu().tolist())
print("after set, state", torch.neuron.get_rng_state()[:16].tolist())
torch.manual_seed(5); c = F.dropout(torch.ones(16, device="neuron"), 0.5)
print("c (reseed)", c.cpu().tolist())
# fork_rng
with torch.random.fork_rng(devices=[0], device_type="neuron"):
    d = F.dropout(torch.ones(16, device="neuron"), 0.5)
e = F.dropout(torch.ones(16, device="neuron"), 0.5)
print("fork d", d.cpu().tolist()); print("after fork e", e.cpu().tolist())
