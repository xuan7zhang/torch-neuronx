import sys, torch, torch_neuronx
torch.manual_seed(int(sys.argv[1]))
if sys.argv[2] == "init_first": pass
xs = [torch.randn(4, device="neuron").cpu() for _ in range(3)]
print("RES seed", sys.argv[1], "3 draws:", [[round(v, 3) for v in x.tolist()] for x in xs])
