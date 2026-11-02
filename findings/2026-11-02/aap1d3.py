import torch, torch.nn.functional as F, torch_neuronx, traceback
for dt in [torch.float32, torch.bfloat16]:
    for B, C, L, O in [(2, 3, 64, 7), (1, 3, 64, 7), (2, 1, 64, 7), (4, 8, 64, 7), (2, 3, 64, 8), (2, 3, 21, 7), (2, 3, 22, 7)]:
        try:
            xx = (torch.arange(B * C * L, dtype=torch.float32).reshape(B, C, L) / 100).to(dt)
            r = F.adaptive_avg_pool1d(xx, O).float(); gn = F.adaptive_avg_pool1d(xx.to("neuron"), O); g = gn.float().cpu()
            g4 = F.adaptive_avg_pool2d(xx.to("neuron").unsqueeze(2), (1, O))
            print(f"RES {str(dt)[6:]:9s} B={B} C={C} L={L} O={O} rel {((r - g).norm() / r.norm()).item():.3g} out stride={gn.stride()} 2d-out stride={g4.stride()} 2d contiguous={g4.is_contiguous()} cl={g4.is_contiguous(memory_format=torch.channels_last)}")
        except Exception as e:
            print("RES ERR", B, C, L, O, str(e)[:150])
