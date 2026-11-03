import sys, time, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
dev = "neuron"
def bench(name, make, loss_fn, x, y, n=10):
    torch.manual_seed(0); m = make().to(dev); opt = torch.optim.SGD(m.parameters(), lr=0.01)
    xd, yd = x.to(dev), y.to(dev)
    def step():
        opt.zero_grad(); l = loss_fn(m(xd), yd); l.backward(); opt.step(); return l
    for _ in range(3): step()
    torch.neuron.synchronize(); t = time.time()
    for _ in range(n): l = step()
    l.item(); torch.neuron.synchronize(); dt = (time.time() - t) / n * 1000
    torch.manual_seed(0); mc = make(); oc = torch.optim.SGD(mc.parameters(), lr=0.01)
    def cstep():
        oc.zero_grad(); l = loss_fn(mc(x), y); l.backward(); oc.step()
    for _ in range(2): cstep()
    t = time.time()
    for _ in range(n): cstep()
    dc = (time.time() - t) / n * 1000
    print(f"RES {name:34s} neuron {dt:8.1f} ms/step   cpu {dc:8.1f} ms/step", flush=True)
x = torch.randn(32, 3, 64, 64); y = torch.randint(0, 10, (32,))
def cnn(pool, act=nn.ReLU):
    return lambda: nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), act(), pool(), nn.Conv2d(32, 64, 3, padding=1), act(), pool(),
                                 nn.Conv2d(64, 64, 3, padding=1), act(), pool(), nn.Flatten(), nn.Linear(64 * 8 * 8, 10))
bench("cnn AvgPool2d + ReLU", cnn(lambda: nn.AvgPool2d(2)), F.cross_entropy, x, y)
bench("cnn MaxPool2d + ReLU", cnn(lambda: nn.MaxPool2d(2)), F.cross_entropy, x, y)
bench("cnn AvgPool2d + LeakyReLU", cnn(lambda: nn.AvgPool2d(2), nn.LeakyReLU), F.cross_entropy, x, y)
xm = torch.randn(256, 512); ym = (torch.rand(256, 512) > 0.5).float()
mk = lambda: nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.Sigmoid())
bench("mlp BCE (sigmoid + BCELoss)", mk, F.binary_cross_entropy, xm, ym)
mk2 = lambda: nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512))
bench("mlp BCEWithLogits", mk2, F.binary_cross_entropy_with_logits, xm, ym)
bench("mlp smooth_l1", mk2, F.smooth_l1_loss, xm, ym)
bench("mlp mse", mk2, F.mse_loss, xm, ym)
print("RES fallback:", sorted(torch_neuronx.get_fallback_ops()))
