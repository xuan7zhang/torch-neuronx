import sys, torch, torch_neuronx
cases = {
 "exp 0-d": (lambda t: t.exp(), lambda: torch.tensor(0.5)),
 "sigmoid 0-d fwd+bwd": (lambda t: t.sigmoid(), lambda: torch.tensor(0.5)),
 "loss*scale 0-d": (lambda t: t * 2.0 + 1.0, lambda: torch.tensor(0.5)),
 "exp (1,0,3)": (lambda t: t.exp(), lambda: torch.randn(1, 0, 3)),
 "linear (0,3)": (lambda t: torch.nn.functional.linear(t, torch.ones(4, 3, device=t.device)), lambda: torch.randn(0, 3)),
 "sum of empty": (lambda t: t.sum(), lambda: torch.randn(0, 3)),
}
for nm, (f, mk) in cases.items():
    torch._dynamo.reset()
    x = mk().to("neuron").requires_grad_()
    try:
        y = torch.compile(f, backend="neuron", dynamic=False)(x); y.sum().backward(); torch.neuron.synchronize()
        ref = f(mk().requires_grad_())
        print("RES OK ", nm, tuple(y.shape))
    except Exception as e:
        m = str(e); i = m.find("NCC_"); print("RES ERR", nm, (m[i:i+80] if i > 0 else m[:120]).replace("\n", " "))
