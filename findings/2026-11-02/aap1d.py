import torch, torch.nn.functional as F, torch_neuronx
for L, out in [(8, 4), (10, 4), (12, 5), (192, 7), (16, 16)]:
    for dt in [torch.float32, torch.bfloat16]:
        x = (torch.arange(2 * L, dtype=torch.float32).reshape(1, 2, L) + 1).to(dt)
        ref = F.adaptive_avg_pool1d(x, out).float(); got = F.adaptive_avg_pool1d(x.to("neuron"), out).float().cpu()
        r = ((ref - got).norm() / ref.norm()).item()
        print(f"RES L={L:3d} out={out:2d} {str(dt)[6:]:9s} rel {r:.3g}" + ("" if r < 1e-2 else f"\n    ref={ref[0].tolist()}\n    got={got[0].tolist()}"))
