import copy, torch, torch_neuronx
from transformers import GPT2Config, GPT2LMHeadModel
cfg = GPT2Config(n_layer=2, n_head=4, n_embd=128, vocab_size=1000, n_positions=64)  # default dropout 0.1
torch.manual_seed(0); base = GPT2LMHeadModel(cfg)
ids = torch.randint(0, 1000, (2, 32), generator=torch.Generator().manual_seed(1))
for dev in ["cpu", "neuron"]:
    if dev == "neuron": torch.ones(1, device=dev).cpu()
    res = {}
    for ck in [False, True]:
        m = copy.deepcopy(base).to(dev); m.train()
        if ck: m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(123)
        out = m(input_ids=ids.to(dev), labels=ids.to(dev)); out.loss.backward()
        res[ck] = (out.loss.item(), torch.cat([p.grad.flatten().float().cpu() for p in m.parameters()]))
    gdiff = (res[False][1] - res[True][1]).abs().max().item(); gnorm = res[False][1].abs().max().item()
    print(f"{dev}: loss no-ckpt={res[False][0]:.6f} ckpt={res[True][0]:.6f}  max grad diff={gdiff:.4g} (max |grad|={gnorm:.4g})")
