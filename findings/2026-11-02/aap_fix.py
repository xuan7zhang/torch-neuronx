import torch, torch.nn.functional as F, torch_neuronx
from torch_neuronx.neuron_dynamo_backend.decompositions import _adaptive_avg_pool_backward_weight
def aap2d_matmul(x, out):
    oH, oW = out; iH, iW = x.shape[-2:]
    wh = _adaptive_avg_pool_backward_weight(oH, iH, x.device, torch.float32)  # (iH, oH) averaging weights?
    ww = _adaptive_avg_pool_backward_weight(oW, iW, x.device, torch.float32)
    y = torch.einsum("hq,...hw->...qw", wh, x.float())
    y = torch.einsum("...qw,wv->...qv", y, ww)
    return y.to(x.dtype)
for B, C, L in [(2, 3, 64), (1, 3, 64), (2, 3, 22), (4, 8, 64)]:
    x = (torch.arange(B * C * L, dtype=torch.float32).reshape(B, C, 1, L) / 100).to(torch.bfloat16)
    ref = F.adaptive_avg_pool2d(x, (1, 7)).float()
    cpu_mm = aap2d_matmul(x, (1, 7)).float()
    try:
        got = aap2d_matmul(x.to("neuron"), (1, 7)).float().cpu()
        print(f"RES ({B},{C},{L}) matmul form: cpu rel {((ref - cpu_mm).norm() / ref.norm()).item():.3g} neuron rel {((ref - got).norm() / ref.norm()).item():.3g}")
    except Exception as e:
        print("RES ERR", B, C, L, str(e)[:150])
