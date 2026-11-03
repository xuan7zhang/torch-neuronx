import sys, torch, torch_neuronx
nm = torch.neuron; D = "neuron"
f = torch.compile(lambda t: t.exp() + 1, backend="neuron", dynamic=False)
try:
    f(torch.randn(0, 3, device=D)); nm.synchronize(); print("RES compile: no error")
except Exception as e:
    print("RES compile error raised:", type(e).__name__)
for i in range(4):
    try:
        a = torch.full((4,), float(i), device=D) * 2
        nm.synchronize()
        print(f"RES op {i}: ok {a.cpu().tolist()}")
    except Exception as e:
        print(f"RES op {i}: FAIL {str(e)[:100]}".replace("\n", " "))
