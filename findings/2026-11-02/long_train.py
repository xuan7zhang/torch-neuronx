import sys, copy, time, torch, torch.nn.functional as F, torch_neuronx
sys.path.insert(0, "/home/ec2-user/torch-neuronx/examples/llama")
from llama import LlamaModel
mode = sys.argv[1]; steps = int(sys.argv[2])
torch.manual_seed(0)
base = LlamaModel(vocab_size=512, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_length=64)
# learnable data: repeated pattern so loss can drop
g = torch.Generator().manual_seed(1)
data = [torch.randint(0, 64, (8, 33), generator=g) for _ in range(16)]
def train(dev):
    m = copy.deepcopy(base).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    losses = []; t = time.time()
    for s in range(steps):
        ids = data[s % 16].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(mode == "bf16")):
            logits = m(ids[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, 512), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if s % 25 == 0 or s == steps - 1: losses.append((s, round(loss.item(), 4)))
    return losses, time.time() - t, {n: p.detach().float().cpu() for n, p in m.named_parameters()}
lc, tc, pc = train("cpu"); ln, tn, pn = train("neuron")
print("RES", mode, "cpu   ", lc, f"{tc:.0f}s")
print("RES", mode, "neuron", ln, f"{tn:.0f}s")
rel = max(((pc[k] - pn[k]).norm() / pc[k].norm()).item() for k in pc)
print("RES", mode, "max relative param diff after", steps, "steps:", f"{rel:.3g}")
