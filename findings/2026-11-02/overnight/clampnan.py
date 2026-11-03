import torch, torch_neuronx
nan = float("nan")
cases = {
 "clamp(max=1)": lambda d: torch.tensor([nan, 2.0, -2.0], device=d).clamp(max=1.0),
 "clamp(min=0)": lambda d: torch.tensor([nan, 2.0, -2.0], device=d).clamp(min=0.0),
 "clamp(-1,1)": lambda d: torch.tensor([nan, 2.0, -2.0], device=d).clamp(-1.0, 1.0),
 "clamp tensor bounds": lambda d: torch.tensor([nan, 2.0], device=d).clamp(max=torch.tensor([1.0, 1.0], device=d)),
 "clamp_ inplace": lambda d: torch.tensor([nan, 2.0], device=d).clamp_(max=1.0),
 "clamp bf16": lambda d: torch.tensor([nan, 2.0], device=d, dtype=torch.bfloat16).clamp(max=1.0).float(),
 "clamp_max": lambda d: torch.clamp_max(torch.tensor([nan, 2.0], device=d), 1.0),
 "clamp_min": lambda d: torch.clamp_min(torch.tensor([nan, -2.0], device=d), 0.0),
 "minimum": lambda d: torch.minimum(torch.tensor([nan, 2.0], device=d), torch.tensor([1.0, 1.0], device=d)),
 "maximum": lambda d: torch.maximum(torch.tensor([nan, 2.0], device=d), torch.tensor([1.0, 1.0], device=d)),
 "fmin": lambda d: torch.fmin(torch.tensor([nan, 2.0], device=d), torch.tensor([1.0, 1.0], device=d)),
 "relu": lambda d: torch.relu(torch.tensor([nan, -2.0], device=d)),
 "hardtanh": lambda d: torch.nn.functional.hardtanh(torch.tensor([nan, 2.0], device=d)),
 "relu6": lambda d: torch.nn.functional.relu6(torch.tensor([nan, 7.0], device=d)),
 "max()": lambda d: torch.tensor([nan, 2.0], device=d).max().reshape(1),
 "amax": lambda d: torch.tensor([nan, 2.0], device=d).amax().reshape(1),
 "min()": lambda d: torch.tensor([nan, 2.0], device=d).min().reshape(1),
 "max(dim)": lambda d: torch.tensor([[nan, 2.0]], device=d).max(1).values,
 "argmax": lambda d: torch.tensor([1.0, nan, 2.0], device=d).argmax().reshape(1).float(),
 "sort": lambda d: torch.tensor([1.0, nan, 0.0], device=d).sort().values,
 "topk": lambda d: torch.tensor([1.0, nan, 0.0], device=d).topk(1).values,
 "where(x>0)": lambda d: (lambda x: torch.where(x > 0, x, 0.0))(torch.tensor([nan, 1.0], device=d)),
 "softmax": lambda d: torch.softmax(torch.tensor([nan, 1.0], device=d), 0),
 "isfinite.all": lambda d: torch.tensor([nan, 1.0], device=d).isfinite().all().reshape(1).float(),
 "nan_to_num": lambda d: torch.tensor([nan, float("inf")], device=d).nan_to_num(),
 "sign": lambda d: torch.tensor([nan], device=d).sign(),
 "sum": lambda d: torch.tensor([nan, 1.0], device=d).sum().reshape(1),
 "clip_coef (clip_grad_norm)": lambda d: torch.clamp(1.0 / (torch.tensor(nan, device=d) + 1e-6), max=1.0).reshape(1),
}
for k, fn in cases.items():
    try:
        c = fn("cpu").tolist(); n = fn("neuron").cpu().tolist()
        same = str(c) == str(n)
        print(("OK  " if same else "DIFF"), f"{k:28s} cpu={c} neuron={n}")
    except Exception as e:
        print("ERR ", k, str(e)[:150])
