import sys, torch, torch_neuronx
fe = sys.argv[1] == "foreach"
def run(dev, comp):
    p = torch.nn.Parameter(torch.ones(4, device=dev))
    opt = torch.optim.SGD([p], lr=1.0, momentum=0.9, nesterov=True, foreach=fe)
    st = torch.compile(opt.step, backend="neuron") if comp else opt.step
    out = []
    for i in range(2):
        p.grad = torch.full((4,), 1.0, device=dev)
        before = p.detach().clone(); st()
        out.append(((before - p.detach()).cpu()[0].item(), opt.state[p]["momentum_buffer"].cpu()[0].item(), p.grad.cpu()[0].item()))
    return out
print("foreach" if fe else "single", "cpu       (delta, buf, grad after step)", run("cpu", False))
print("foreach" if fe else "single", "neuron eag", run("neuron", False))
print("foreach" if fe else "single", "neuron cmp", run("neuron", True))
