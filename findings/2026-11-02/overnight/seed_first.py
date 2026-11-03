import sys, torch, torch_neuronx, torch.nn.functional as F
torch.manual_seed(int(sys.argv[1]))
op = sys.argv[2]
x = {"randn": lambda: torch.randn(6, device="neuron"), "dropout": lambda: F.dropout(torch.ones(12, device="neuron"), 0.5),
     "linear_init": lambda: torch.nn.Linear(4, 2, device="neuron").weight.flatten()}[op]()
print("RES seed", sys.argv[1], op, [round(v, 4) for v in x.cpu().tolist()])
