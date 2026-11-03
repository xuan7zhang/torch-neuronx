import sys, torch, torch_neuronx
n = int(eval(sys.argv[1])); op = sys.argv[2]; dt = getattr(torch, sys.argv[3])
try:
    if op == "full": x = torch.full((n,), 1.0, dtype=dt, device="neuron")
    elif op == "zeros_add": x = torch.zeros(n, dtype=dt, device="neuron"); x.add_(1)
    elif op == "ones": x = torch.ones(n, dtype=dt, device="neuron")
    elif op == "zeros": x = torch.zeros(n, dtype=dt, device="neuron")
    elif op == "fill": x = torch.empty(n, dtype=dt, device="neuron"); x.fill_(1.0)
    elif op == "mul": x = torch.zeros(n, dtype=dt, device="neuron") * 2
    torch.neuron.synchronize()
    r = f"ok x[0]={x[0].item()} x[-1]={x[-1].item()}"
except Exception as e:
    r = "ERR " + str(e).replace("\n", " ")[60:200]
try:
    y = torch.ones(4, device="neuron") * 2; torch.neuron.synchronize(); after = "next op ok"
except Exception as e:
    after = "next op ERR " + str(e)[60:140]
print(f"RES n={sys.argv[1]} {op} {sys.argv[3]}: {r} | {after}")
