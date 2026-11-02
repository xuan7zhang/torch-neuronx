import sys, torch, torch_neuronx
dt = getattr(torch, sys.argv[1]); shape = eval(sys.argv[2]); kind = sys.argv[3]
x = torch.arange(int(torch.tensor(shape).prod())).reshape(shape).to(dt).to("neuron")
f = {"slice1": lambda t: t[:, 1:].reshape(-1), "slice1_plus": lambda t: t[:, 1:].reshape(-1) + 1, "slice0": lambda t: t[:, :-1].reshape(-1)}[kind]
try:
    r = torch.compile(f, backend="neuron")(x); ok = torch.equal(r.cpu(), f(x.cpu())); print("RES", sys.argv[1:], "OK" if ok else "WRONG")
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", sys.argv[1:], "ERROR", msg[i:i+60] if i > 0 else msg[:120].replace("\n", " "))
