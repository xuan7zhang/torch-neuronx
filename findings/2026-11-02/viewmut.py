import sys, torch, torch_neuronx
mode = sys.argv[1]
base = torch.arange(12.0).reshape(3, 4)
def f(v):
    v.mul_(3.0); return v.sum()
views = {"cols": lambda b: b[:, 1:3], "trow": lambda b: b.t()[1], "row": lambda b: b[1], "rows": lambda b: b[1:], "strided": lambda b: b[:, ::2], "diag": lambda b: b.diagonal()}
ref = base.clone(); r_ref = f(views[mode](ref))
nb = base.clone().to("neuron")
try:
    r = torch.compile(f, backend="neuron")(views[mode](nb)); torch.neuron.synchronize()
    print("RES", mode, "OK" if torch.equal(nb.cpu(), ref) and r.item() == r_ref.item() else f"WRONG base={nb.cpu().tolist()}")
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", mode, "ERROR", msg[i:i+100].replace("\n", " ") if i > 0 else msg[:150].replace("\n", " "))
