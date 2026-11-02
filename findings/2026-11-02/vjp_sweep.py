"""OpInfo sweep: forward + VJP on CPU fp32 vs neuron fp32. Usage: vjp_sweep.py <start> <end> <logfile>"""

import sys
import traceback

import torch
import torch_neuronx
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_flatten, tree_map

start, end, logf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
DT = getattr(torch, sys.argv[4]) if len(sys.argv) > 4 else torch.float32
COMPILE = len(sys.argv) > 5 and sys.argv[5] == "compile"
DEV = "neuron"
SKIP_SUBSTR = ("fft", "linalg", "sparse", "_refs", "special.", "nn.functional.ctc", "cholesky", "svd", "eig",
               "lu", "qr", "lstsq", "pinv", "det", "inv", "solve", "matrix_exp", "histc", "bincount", "unique",
               "nonzero", "normal", "bernoulli", "multinomial", "rand", "dropout", "uniform", "empty", "new_",
               "signal.", "geqrf", "ormqr", "householder", "triangular", "cdist", "nanquantile", "quantile",
               "item", "to_sparse", "jiterator", "segment", "_native_batch_norm_legit", "batch_norm", "instance_norm")
ops = [o for o in op_db if o.supports_autograd and torch.float32 in o.supported_dtypes("cpu") and DT in o.supported_dtypes("cpu")
       and not any(s in o.name for s in SKIP_SUBSTR)]
ops = ops[start:end]


def to_dev(x):
    if isinstance(x, torch.Tensor):
        y = x.detach().to(DEV)
        if y.is_floating_point():
            y = y.to(DT)
        if x.requires_grad:
            y.requires_grad_(True)
        return y
    return x


def run(op, sample, dev):
    inp = sample.input if dev == "cpu" else tree_map(to_dev, sample.input)
    args = sample.args if dev == "cpu" else tree_map(to_dev, sample.args)
    kwargs = sample.kwargs if dev == "cpu" else tree_map(to_dev, sample.kwargs)
    if dev == "cpu" or not COMPILE:
        out = op(inp, *args, **kwargs)
    else:
        torch._dynamo.reset()
        out = torch.compile(lambda i, a, k: op(i, *a, **k), backend="neuron", dynamic=False)(inp, args, kwargs)
    flat_out = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor) and t.is_floating_point()]
    leaves = [t for t in tree_flatten((inp, args, kwargs))[0] if isinstance(t, torch.Tensor) and t.requires_grad]
    grads = []
    diff_outs = [t for t in flat_out if t.requires_grad]
    if diff_outs and leaves:
        g = torch.Generator().manual_seed(0)
        cots = [torch.randn(t.shape, generator=g).to(t.dtype).float().to(t.device).to(t.dtype) for t in diff_outs]
        grads = torch.autograd.grad(diff_outs, leaves, cots, allow_unused=True)
    return [t.detach() for t in flat_out], [None if x is None else x.detach() for x in grads]


def cmp(a, b, atol=None, rtol=None):
    atol = atol or (1e-3 if DT == torch.float32 else 5e-2)
    rtol = rtol or (1e-3 if DT == torch.float32 else 5e-2)
    worst = 0.0
    for x, y in zip(a, b):
        if x is None or y is None:
            if (x is None) != (y is None):
                return False, "none-mismatch"
            continue
        x, y = x.float().cpu(), y.float().cpu()
        if x.shape != y.shape:
            return False, f"shape {tuple(x.shape)} vs {tuple(y.shape)}"
        if not torch.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True):
            d = (x - y).abs().nan_to_num(1e9)
            worst = max(worst, d.max().item())
    return worst == 0.0, worst


with open(logf, "a") as log:
    for op in ops:
        name = op.name + ("." + op.variant_test_name if op.variant_test_name else "")
        try:
            samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=True))[:3]
            if DT != torch.float32:
                def rnd(x):
                    if isinstance(x, torch.Tensor) and x.is_floating_point():
                        y = x.detach().to(DT).float()
                        return y.requires_grad_(x.requires_grad)
                    return x
                for smp in samples:
                    smp.input = tree_map(rnd, smp.input); smp.args = tree_map(rnd, smp.args); smp.kwargs = tree_map(rnd, smp.kwargs)
        except Exception as e:
            log.write(f"SKIP   {name} sample_inputs failed {type(e).__name__}\n")
            log.flush()
            continue
        res = []
        for i, s in enumerate(samples):
            try:
                ro, rg = run(op, s, "cpu")
            except Exception as e:
                res.append(f"s{i}:cpuerr")
                continue
            try:
                no, ng = run(op, s, DEV)
                torch.neuron.synchronize()
                okf, wf = cmp(ro, no)
                okg, wg = cmp(rg, ng)
                if okf and okg:
                    res.append(f"s{i}:ok")
                else:
                    res.append(f"s{i}:{'FWD' if not okf else ''}{'BWD' if not okg else ''}(f={wf},g={wg})")
            except Exception as e:
                res.append(f"s{i}:ERR[{type(e).__name__}: {str(e)[:120]}]".replace("\n", " "))
        status = "OK    " if all(r.endswith(":ok") or r.endswith("cpuerr") for r in res) else "ISSUE "
        log.write(f"{status} {name:45s} {' '.join(res)}\n")
        log.flush()
    log.write(f"FALLBACK {sorted(torch_neuronx.get_fallback_ops())}\n")
