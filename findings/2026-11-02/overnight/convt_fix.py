import sys, torch, torch.nn as nn, torch.nn.functional as TF, torch_neuronx
sys.argv += []
exec(open("conv_fix.py").read().split("if sys.argv[1] == \"patch\":")[0].split("import torch_neuronx.neuron_dynamo_backend.decompositions as D")[1])
orig = TF.conv_transpose2d
TF.conv_transpose2d = wrap_t(orig)
import torch_neuronx.neuron_dynamo_backend.decompositions as D
S = int(sys.argv[1])
torch.manual_seed(0); mc = nn.ConvTranspose2d(4, 8, 3, stride=2, dilation=2); mn = nn.ConvTranspose2d(4, 8, 3, stride=2, dilation=2).to("neuron"); mn.load_state_dict(mc.state_dict())
x = torch.randn(2, 4, S, S); xc = x.clone().requires_grad_(); xn = x.clone().to("neuron").requires_grad_()
try:
    yc = mc(xc); yn = mn(xn); yc.pow(2).sum().backward(); yn.pow(2).sum().backward(); torch.neuron.synchronize()
    print("RES convT2d stride2 dil2 fwd diff", f"{(yc - yn.detach().cpu()).abs().max().item():.2e}", "x.grad", f"{(xc.grad - xn.grad.cpu()).abs().max().item():.2e}", "w.grad", f"{(mc.weight.grad - mn.weight.grad.cpu()).abs().max().item():.2e}")
except Exception as e:
    m = str(e); i = m.find("NCC_"); print("RES ERR", (m[i:i+80] if i > 0 else m[:160]).replace("\n", " "))
