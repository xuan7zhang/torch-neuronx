import sys, torch, torch.nn.functional as F, torch_neuronx
mode = sys.argv[1]
ids = torch.randint(0, 50, (4, 9), generator=torch.Generator().manual_seed(0)).to("neuron")
x = torch.randn(32, 50, generator=torch.Generator().manual_seed(1)).to("neuron").requires_grad_()
fs = {
 "ce_fwd": (lambda x, ids: F.cross_entropy(x, ids[:, 1:].reshape(-1)), False),
 "ce_fwdbwd": (lambda x, ids: F.cross_entropy(x, ids[:, 1:].reshape(-1)), True),
 "ce_fwdbwd_contig": (lambda x, ids: F.cross_entropy(x, ids[:, 1:].contiguous().view(-1)), True),
 "nll_fwdbwd": (lambda x, ids: F.nll_loss(x.log_softmax(-1), ids[:, 1:].reshape(-1)), True),
 "gather_fwdbwd": (lambda x, ids: x.gather(1, ids[:, 1:].reshape(-1, 1)).sum(), True),
 "ce_fwdbwd_slice_rows": (lambda x, ids: F.cross_entropy(x[:8], ids[1:2].reshape(-1)[:8]), True),
}
f, bwd = fs[mode]
ref = f(x.detach().cpu().requires_grad_(), ids.cpu())
try:
    l = torch.compile(f, backend="neuron")(x, ids)
    if bwd: l.backward()
    torch.neuron.synchronize(); print("RES", mode, "OK", round(l.item(), 5), "cpu", round(ref.item(), 5))
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", mode, "ERROR", msg[i:i+70].replace("\n"," ") if i>0 else msg[:150].replace("\n"," "))
