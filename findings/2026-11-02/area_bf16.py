import torch, torch.nn.functional as F, torch_neuronx
x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 100
for dt in [torch.float32, torch.bfloat16, torch.float16]:
    for nm, f in [("interp area 4x4", lambda t: F.interpolate(t, size=(4, 4), mode="area")), ("interp area 3x5", lambda t: F.interpolate(t, size=(3, 5), mode="area")),
                  ("adaptive_avg_pool2d 4x4", lambda t: F.adaptive_avg_pool2d(t, (4, 4))), ("adaptive_avg_pool2d 3x5", lambda t: F.adaptive_avg_pool2d(t, (3, 5))),
                  ("adaptive_avg_pool1d", lambda t: F.adaptive_avg_pool1d(t.flatten(2), 7)), ("avg_pool2d 2", lambda t: F.avg_pool2d(t, 2)), ("mean dim", lambda t: t.mean((2, 3)))]:
        ref = f(x.to(dt)).float(); got = f(x.to(dt).to("neuron")).float().cpu()
        r = ((ref - got).norm() / ref.norm()).item()
        print(f"RES {str(dt)[6:]:9s} {nm:26s} rel err {r:.3g}" + ("" if r < 1e-2 else f"   ref[0,0,0]={ref[0,0,0].tolist()} got={got[0,0,0].tolist()}"))
