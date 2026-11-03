import torch, torch.nn as nn, torch_neuronx, traceback
m = nn.Linear(16, 16).to("neuron"); f = torch.compile(m, backend="neuron")
f(torch.randn(2, 8, 16).to("neuron"))
try: f(torch.randn(2, 12, 16).to("neuron"))
except Exception: traceback.print_exc()
