import sys, torch, torch.nn.functional as F, torch_neuronx
ids = torch.randint(0, 50, (4, 9), generator=torch.Generator().manual_seed(0)).to("neuron")
x = torch.randn(32, 50).to("neuron")
f = (lambda x, ids: F.cross_entropy(x, ids[:, 1:].reshape(-1))) if sys.argv[1] == "ce" else (lambda x, ids: F.nll_loss(x.log_softmax(1), ids[:, 1:].reshape(-1)))
try: print(torch.compile(f, backend="neuron")(x, ids).item())
except Exception as e: print("ERR")
