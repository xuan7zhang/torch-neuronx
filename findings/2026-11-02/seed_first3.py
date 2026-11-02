import sys, torch, torch_neuronx
torch.ones(1, device="neuron").cpu()
torch.manual_seed(int(sys.argv[1]))
x = torch.randn(4, device="neuron").cpu()
print("RES init-then-seed", sys.argv[1], [round(v, 3) for v in x.tolist()])
