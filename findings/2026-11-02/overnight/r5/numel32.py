import sys, torch, torch_neuronx
nm = torch.neuron
dt = getattr(torch, sys.argv[1]); n = int(eval(sys.argv[2])); op = sys.argv[3]
try:
    if op == "ones":
        t = torch.ones(n, dtype=dt, device="neuron")
    elif op == "empty_fill":
        t = torch.empty(n, dtype=dt, device="neuron"); t.fill_(1)
    elif op == "zeros_add":
        t = torch.zeros(n, dtype=dt, device="neuron"); t.add_(1)
    nm.synchronize()
    print(f"RES {sys.argv[1]} n={sys.argv[2]} {op}: ok first={t[0].item()} last={t[-1].item()}")
except Exception as e:
    print(f"RES {sys.argv[1]} n={sys.argv[2]} {op}: FAIL {str(e)[:110]}".replace("\n", " "))
