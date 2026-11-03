import sys, torch, torch_neuronx
import torch_neuronx.neuron as N
orig = N._set_device_rng_seed
N._set_device_rng_seed = lambda s: (print("RES _set_device_rng_seed", s, "initialized=", N.is_initialized()), orig(s))[1]
oli = torch_neuronx._lazy_init
def li():
    print("RES python _lazy_init called, _initialized=", torch_neuronx._initialized); return oli()
torch_neuronx._lazy_init = li
torch.manual_seed(int(sys.argv[1]))
print("RES after manual_seed: is_initialized", N.is_initialized(), "py flag", torch_neuronx._initialized)
x = torch.randn(4, device="neuron").cpu()
print("RES after randn: is_initialized", N.is_initialized(), "py flag", torch_neuronx._initialized, "gen initial_seed", torch.neuron.initial_seed(), x.tolist()[:2])
