import sys, torch, torch_neuronx, torch.nn.functional as F
if sys.argv[3] == "patch":
    import torch_neuronx.neuron as N
    om, oma = N.manual_seed, N.manual_seed_all
    def ms(seed): torch_neuronx._lazy_init(); return om(seed)
    def msa(seed): torch_neuronx._lazy_init(); return oma(seed)
    N.manual_seed, N.manual_seed_all = ms, msa
    torch.neuron.manual_seed, torch.neuron.manual_seed_all = ms, msa
torch.manual_seed(int(sys.argv[1]))
op = sys.argv[2]
x = {"randn": lambda: torch.randn(6, device="neuron"), "dropout": lambda: F.dropout(torch.ones(12, device="neuron"), 0.5)}[op]()
print("RES", sys.argv[3], "seed", sys.argv[1], op, [round(v, 3) for v in x.cpu().tolist()])
