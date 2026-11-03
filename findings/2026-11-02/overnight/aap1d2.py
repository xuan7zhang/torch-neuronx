import torch, torch.nn.functional as F, torch_neuronx
x = (torch.arange(2 * 3 * 64, dtype=torch.float32).reshape(2, 3, 64) / 100)
for dt in [torch.bfloat16]:
    xd = x.to(dt)
    ref = F.adaptive_avg_pool1d(xd, 7).float(); got = F.adaptive_avg_pool1d(xd.to("neuron"), 7).float().cpu()
    print("RES ref", ref[:, :, :].reshape(6, 7).tolist()); print("RES got", got.reshape(6, 7).tolist())
    xc = xd.contiguous(); print("RES contiguous input?", x.flatten(2).is_contiguous())
    # same op via 2d path
    got2 = F.adaptive_avg_pool2d(xd.to("neuron").unsqueeze(-2), (1, 7)).squeeze(-2).float().cpu()
    print("RES via 2d rel", ((ref - got2).norm() / ref.norm()).item())
    for B, C in [(1, 1), (1, 3), (2, 1), (2, 3), (4, 8)]:
        xx = (torch.arange(B * C * 64, dtype=torch.float32).reshape(B, C, 64) / 100).to(dt)
        r = F.adaptive_avg_pool1d(xx, 7).float(); g = F.adaptive_avg_pool1d(xx.to("neuron"), 7).float().cpu()
        print(f"RES B={B} C={C} rel {((r - g).norm() / r.norm()).item():.3g}")
