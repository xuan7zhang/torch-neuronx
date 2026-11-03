"""Eager training probes, round 3: optimizer state, scaler, schedulers, init patterns, mutating ops."""

import copy
import io
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
CASES = {}


def case(name, atol=1e-4, rtol=1e-4):
    def deco(fn):
        CASES[name] = (fn, atol, rtol)
        return fn
    return deco


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))


def data(seed=1, n=6):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 8, generator=g), torch.randn(n, 4, generator=g)


def steps(m, opt, dev, k=3, seed=1, sched=None):
    for i in range(k):
        x, y = data(seed + i)
        opt.zero_grad()
        F.mse_loss(m(x.to(dev)), y.to(dev)).backward()
        opt.step()
        if sched is not None:
            sched.step()


# ---- optimizer state_dict across devices ----
for oname, ocls, kw in [("adamw", torch.optim.AdamW, dict(lr=1e-2)), ("sgd_mom", torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
                        ("adam_amsgrad", torch.optim.Adam, dict(lr=1e-2, amsgrad=True)), ("rmsprop", torch.optim.RMSprop, dict(lr=1e-2, momentum=0.5)),
                        ("adagrad", torch.optim.Adagrad, dict(lr=1e-2)), ("adam_capturable", torch.optim.Adam, dict(lr=1e-2, capturable=True)),
                        ("adamw_fused", torch.optim.AdamW, dict(lr=1e-2, fused=True))]:
    @case(f"opt_state_neuron_to_cpu_back/{oname}")
    def _(dev, ocls=ocls, kw=kw):
        # train 2 steps on dev, save opt+model to bytes, load into fresh model on dev (map_location=cpu then .to), train 2 more
        kw2 = dict(kw)
        if dev == "cpu":
            kw2.pop("capturable", None)
            kw2.pop("fused", None) if False else None
        m = mlp().to(dev)
        opt = ocls(m.parameters(), **kw2)
        steps(m, opt, dev, 2)
        buf = io.BytesIO()
        torch.save({"m": m.state_dict(), "o": opt.state_dict()}, buf)
        buf.seek(0)
        sd = torch.load(buf, map_location="cpu")
        m2 = mlp().to(dev)
        m2.load_state_dict(sd["m"])
        opt2 = ocls(m2.parameters(), **kw2)
        opt2.load_state_dict(sd["o"])
        devs = sorted({v.device.type for st in opt2.state.values() for v in st.values() if isinstance(v, torch.Tensor)})
        steps(m2, opt2, dev, 2, seed=10)
        return [p.detach() for p in m2.parameters()]

    @case(f"opt_state_cpu_to_neuron/{oname}")
    def _(dev, ocls=ocls, kw=kw):
        # train on CPU, move model + optimizer state to dev, continue on dev
        kwc = {k: v for k, v in kw.items() if k not in ("capturable", "fused")}
        m = mlp()
        opt = ocls(m.parameters(), **kwc)
        steps(m, opt, "cpu", 2)
        sd = copy.deepcopy(opt.state_dict())
        m = m.to(dev)
        opt2 = ocls(m.parameters(), **(kw if dev != "cpu" else kwc))
        opt2.load_state_dict(sd)
        steps(m, opt2, dev, 2, seed=10)
        return [p.detach() for p in m.parameters()]


@case("opt_state_step_device_after_load")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    steps(m, opt, dev, 1)
    sd = copy.deepcopy(opt.state_dict())
    opt2 = torch.optim.AdamW(m.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)
    st = opt2.state[next(iter(m.parameters()))]
    return [torch.tensor([float(st["step"].device.type == "cpu"), float(st["exp_avg"].device.type == dev)])]


@case("add_param_group_midway")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m[0].parameters(), lr=1e-2)
    steps(m, opt, dev, 2)
    opt.add_param_group({"params": m[2].parameters(), "lr": 5e-3})
    steps(m, opt, dev, 2, seed=5)
    return [p.detach() for p in m.parameters()]


# ---- GradScaler ----
@case("grad_scaler_inf_skip_and_backoff")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sc = torch.amp.GradScaler(dev, init_scale=2.0 ** 10, growth_interval=2, growth_factor=2.0, backoff_factor=0.5)
    scales, skipped = [], []
    for i in range(6):
        x, y = data(i)
        opt.zero_grad()
        loss = F.mse_loss(m(x.to(dev)), y.to(dev))
        sc.scale(loss).backward()
        if i == 2:
            next(m.parameters()).grad[0, 0] = float("inf")
        if i == 4:
            next(m.parameters()).grad[0, 1] = float("nan")
        before = next(m.parameters()).detach().clone()
        sc.step(opt)
        sc.update()
        skipped.append(float(torch.equal(before, next(m.parameters()).detach())))
        scales.append(sc.get_scale())
    return [torch.tensor(scales), torch.tensor(skipped)] + [p.detach() for p in m.parameters()]


@case("grad_scaler_unscale_clip")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sc = torch.amp.GradScaler(dev, init_scale=2.0 ** 8)
    norms = []
    for i in range(3):
        x, y = data(i)
        opt.zero_grad()
        sc.scale(F.mse_loss(m(x.to(dev)), y.to(dev))).backward()
        sc.unscale_(opt)
        norms.append(torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1).detach().cpu())
        sc.step(opt)
        sc.update()
    return [torch.stack(norms)] + [p.detach() for p in m.parameters()]


@case("grad_scaler_state_dict_roundtrip")
def _(dev):
    sc = torch.amp.GradScaler(dev, init_scale=128.0, growth_interval=3)
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for i in range(4):
        opt.zero_grad()
        sc.scale(F.mse_loss(m(data(i)[0].to(dev)), data(i)[1].to(dev))).backward()
        sc.step(opt)
        sc.update()
    sd = sc.state_dict()
    sc2 = torch.amp.GradScaler(dev)
    sc2.load_state_dict(sd)
    return [torch.tensor([sd["scale"], float(sd["_growth_tracker"]), sc2.get_scale()])]


@case("amp_foreach_non_finite_check_direct")
def _(dev):
    gs = [torch.tensor([1.0, 2.0], device=dev) * 4, torch.tensor([float("inf"), 1.0], device=dev)]
    found = torch.zeros(1, device=dev)
    inv = torch.full((1,), 0.25, device=dev)
    torch._amp_foreach_non_finite_check_and_unscale_(gs, found, inv)
    st = torch.full((1,), 8.0, device=dev)
    gt = torch.zeros(1, dtype=torch.int32, device=dev)
    torch._amp_update_scale_(st, gt, found, 2.0, 0.5, 2)
    return [gs[0], found, st, gt.float()]


# ---- LR schedulers ----
def sched_run(dev, mk):
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    sch = mk(opt)
    lrs = []
    for i in range(6):
        x, y = data(i)
        opt.zero_grad()
        loss = F.mse_loss(m(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(loss.detach() * 0 + 1.0)  # neuron tensor metric, constant -> plateau
        else:
            sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    return [torch.tensor(lrs)] + [p.detach() for p in m.parameters()]


for nm, mk in [
    ("step", lambda o: torch.optim.lr_scheduler.StepLR(o, 2, 0.5)),
    ("cosine_warm_restarts", lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o, 2)),
    ("onecycle", lambda o: torch.optim.lr_scheduler.OneCycleLR(o, 0.5, total_steps=6)),
    ("linear_warmup_seq", lambda o: torch.optim.lr_scheduler.SequentialLR(o, [torch.optim.lr_scheduler.LinearLR(o, 0.1, total_iters=2), torch.optim.lr_scheduler.ExponentialLR(o, 0.9)], [2])),
    ("lambda", lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda e: 1 / (1 + e))),
    ("plateau_tensor_metric", lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, patience=1, factor=0.5)),
    ("cyclic", lambda o: torch.optim.lr_scheduler.CyclicLR(o, 0.01, 0.1, step_size_up=2)),
    ("polynomial", lambda o: torch.optim.lr_scheduler.PolynomialLR(o, 6, 2.0)),
]:
    case(f"sched/{nm}")(lambda dev, mk=mk: sched_run(dev, mk))


# ---- init / device-move patterns ----
@case("meta_init_to_empty_reset")
def _(dev):
    with torch.device("meta"):
        m = mlp()
    m = m.to_empty(device=dev)
    torch.manual_seed(0)
    for mod in m.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    # reset_parameters uses the default (CPU) RNG for init? compare against cpu
    x, y = data()
    F.mse_loss(m(x.to(dev)), y.to(dev)).backward()
    return [p.detach() for p in m.parameters()] + [p.grad for p in m.parameters()]


@case("load_state_dict_assign")
def _(dev):
    src = mlp()
    sd = {k: v.to(dev) for k, v in src.state_dict().items()}
    with torch.device("meta"):
        m = mlp()
    m.load_state_dict(sd, assign=True)
    x, y = data()
    F.mse_loss(m(x.to(dev)), y.to(dev)).backward()
    return [p.grad for p in m.parameters()] + [torch.tensor([float(all(p.device.type == dev for p in m.parameters()))])]


@case("param_data_assign_dtype_device")
def _(dev):
    m = mlp().to(dev)
    for p in m.parameters():
        p.data = p.data.to(torch.bfloat16).to(torch.float32)
    m[0].weight.data = m[0].weight.data.t().contiguous().t()  # non-contiguous param data
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    steps(m, opt, dev, 2)
    return [p.detach() for p in m.parameters()]


@case("module_half_then_float_train")
def _(dev):
    m = mlp().to(dev).bfloat16()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x, y = data()
    for _ in range(2):
        opt.zero_grad()
        F.mse_loss(m(x.to(dev).bfloat16()).float(), y.to(dev)).backward()
        opt.step()
    m = m.float()
    return [p.detach() for p in m.parameters()]


@case("move_model_between_devices_midtrain")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    steps(m, opt, dev, 2)
    m.cpu()
    m.to(dev)  # .to keeps Parameter identity, so optimizer still refers to them
    steps(m, opt, dev, 2, seed=9)
    return [p.detach() for p in m.parameters()]


@case("deepcopy_model_and_optimizer")
def _(dev):
    m = mlp().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    steps(m, opt, dev, 2)
    m2 = copy.deepcopy(m)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    steps(m2, opt2, dev, 2, seed=7)
    steps(m, opt, dev, 2, seed=7)
    return [p.detach() for p in m2.parameters()] + [p.detach() for p in m.parameters()]


@case("state_dict_tensor_views_save")
def _(dev):
    # tied/shared params and view tensors through torch.save
    w = torch.randn(4, 6, generator=torch.Generator().manual_seed(0)).to(dev)
    obj = {"w": w, "wt": w.t(), "col": w[:, 2], "rows": w[1:3]}
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    o = torch.load(buf)
    o["w"].mul_(2)  # views should share storage after load
    return [o["w"], o["wt"], o["col"], o["rows"], torch.tensor([float(o["col"].device.type == dev)])]


# ---- ops that mutate a non-first argument ----
@case("rrelu_with_noise_train_mutates_noise")
def _(dev):
    torch.manual_seed(0)
    x = torch.randn(64, generator=torch.Generator().manual_seed(0)).to(dev)
    noise = torch.empty(64, device=dev)
    out = torch.ops.aten.rrelu_with_noise(x, noise, 0.1, 0.3, True)
    # noise must be filled: 1 where x>=0, in [0.1,0.3] where x<0; out = x*noise
    ok = torch.allclose(out, x * noise)
    pos_ones = bool((noise[x >= 0] == 1).all())
    neg_range = bool(((noise[x < 0] >= 0.1) & (noise[x < 0] <= 0.3)).all())
    return [torch.tensor([float(ok), float(pos_ones), float(neg_range)])]


@case("nn_rrelu_train_backward")
def _(dev):
    x = torch.randn(64, generator=torch.Generator().manual_seed(0)).to(dev).requires_grad_()
    y = F.rrelu(x, 0.1, 0.3, training=True)
    y.sum().backward()
    slope = y.detach() / x.detach()
    ok = torch.allclose(x.grad, slope, atol=1e-6)
    return [torch.tensor([float(ok)])]


@case("native_dropout_mask_matches_output")
def _(dev):
    x = torch.randn(1000, generator=torch.Generator().manual_seed(0)).to(dev)
    out, mask = torch.native_dropout(x, 0.3, True)
    ok = torch.allclose(out, x * mask / 0.7, atol=1e-6)
    return [torch.tensor([float(ok), round(float(mask.float().mean()), 1)])]


@case("batchnorm_eval_uses_running_stats_buffers_loaded")
def _(dev):
    torch.manual_seed(0)
    m = nn.BatchNorm1d(4)
    m.running_mean.copy_(torch.arange(4.0))
    m.running_var.copy_(torch.arange(1.0, 5.0))
    m.num_batches_tracked.fill_(5)
    m = m.to(dev).eval()
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(1)).to(dev)
    return [m(x), m.running_mean, m.running_var]


@case("batchnorm_momentum_none_cumulative")
def _(dev):
    m = nn.BatchNorm1d(4, momentum=None).to(dev)
    for i in range(3):
        m(torch.randn(5, 4, generator=torch.Generator().manual_seed(i)).to(dev))
    return [m.running_mean, m.running_var, m.num_batches_tracked.float()]


@case("embedding_max_norm_renorm_weight_inplace")
def _(dev):
    torch.manual_seed(0)
    e = nn.Embedding(10, 4, max_norm=1.0).to(dev)
    with torch.no_grad():
        e.weight.mul_(5.0)
    idx = torch.tensor([1, 1, 3]).to(dev)
    e(idx)
    return [e.weight]


@case("index_copy_index_fill_inplace_bwd")
def _(dev):
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(0)).to(dev).requires_grad_()
    src = torch.randn(2, 3, generator=torch.Generator().manual_seed(1)).to(dev).requires_grad_()
    y = x * 1.0
    y.index_copy_(0, torch.tensor([4, 1]).to(dev), src)
    y.index_fill_(1, torch.tensor([2]).to(dev), -1.0)
    y.masked_fill_(y > 1.0, 0.5)
    (y * torch.arange(18.0).view(6, 3).to(dev)).sum().backward()
    return [y, x.grad, src.grad]


@case("put_and_take_bwd")
def _(dev):
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(0)).to(dev).requires_grad_()
    v = torch.randn(3, generator=torch.Generator().manual_seed(1)).to(dev).requires_grad_()
    y = (x * 1.0).put(torch.tensor([0, 5, 5]).to(dev), v, accumulate=True)
    t = x.take(torch.tensor([[0, 11], [5, 5]]).to(dev))
    (y.pow(2).sum() + t.sum()).backward()
    return [y, t, x.grad, v.grad]


@case("unfold_inplace_overlap_view_write")
def _(dev):
    x = torch.zeros(10, device=dev)
    w = x.unfold(0, 4, 2)  # overlapping windows
    w[1].add_(1.0)
    w[:, 0].add_(10.0)
    return [x]


@case("expand_inplace_error_and_clone")
def _(dev):
    a = torch.ones(1, 4, device=dev).expand(3, 4)
    try:
        a.add_(1.0)
        r = 0.0
    except RuntimeError:
        r = 1.0
    b = a.clone()
    b.add_(1.0)
    return [torch.tensor([r]), b]


@case("grad_accum_into_noncontig_grad")
def _(dev):
    # user sets a non-contiguous .grad, autograd accumulates into it
    p = nn.Parameter(torch.randn(4, 6, generator=torch.Generator().manual_seed(0)).to(dev))
    p.grad = torch.zeros(6, 4, device=dev).t()
    (p * 2).sum().backward()
    (p * 3).sum().backward()
    return [p.grad, torch.tensor([float(p.grad.is_contiguous())])]


@case("zero_grad_after_noncontig_slice_ops")
def _(dev):
    m = mlp().to(dev)
    x, y = data()
    F.mse_loss(m(x.to(dev)), y.to(dev)).backward()
    with torch.no_grad():
        for p in m.parameters():
            if p.grad.dim() == 2:
                p.grad[:, ::2].mul_(0.0)
                p.grad.t()[1].add_(1.0)
    return [p.grad for p in m.parameters()]


def main():
    sel = sys.argv[1:]
    names = [n for n in CASES if not sel or any(s in n for s in sel)]
    summary = []
    for name in names:
        fn, atol, rtol = CASES[name]
        try:
            ref = [t.detach().float().cpu() for t in fn("cpu")]
            out = [t.detach().float().cpu() for t in fn(DEV)]
            torch.neuron.synchronize()
            status, worst = "OK", 0.0
            for i, (r, o) in enumerate(zip(ref, out)):
                if r.shape != o.shape:
                    status = f"SHAPE@{i}"
                    break
                if not torch.allclose(r, o, atol=atol, rtol=rtol, equal_nan=True):
                    d = (r - o).abs().nan_to_num(1e9).max().item()
                    status, worst = "MISMATCH", max(worst, d)
                    print(f"  [{name}] tensor {i} maxdiff {d:.4g} ref={r.flatten()[:6].tolist()} got={o.flatten()[:6].tolist()}", flush=True)
            line = f"{status:10s} {name:50s} {worst:.4g}"
        except Exception as e:
            traceback.print_exc()
            line = f"ERROR      {name:50s} {type(e).__name__}: {str(e)[:250]}".replace("\n", " ")
        print("RESULT " + line, flush=True)
        summary.append(line)
    print("\n===== SUMMARY =====")
    print("\n".join(summary))
    print("fallback ops:", sorted(torch_neuronx.get_fallback_ops()))


if __name__ == "__main__":
    main()
