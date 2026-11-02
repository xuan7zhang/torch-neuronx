"""Forward-only OpInfo sweep for a dtype (int64 / bool / int32 / float32 non-autograd ops). Usage: start end log dtype"""

import sys

import torch
import torch_neuronx
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_flatten, tree_map

start, end, logf, dtn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
DT = getattr(torch, dtn)
DEV = "neuron"
SKIP = ("fft", "linalg", "sparse", "_refs", "signal.", "normal", "bernoulli", "multinomial", "rand", "dropout", "uniform",
        "empty", "new_", "jiterator", "item", "to_sparse", "nonzero_static", "geometric", "exponential", "cauchy",
        "log_normal", "cdouble", "chalf", "cfloat", "complex", "view_as_complex", "_unsafe_masked_index")
ops = [o for o in op_db if DT in o.supported_dtypes("cpu") and not any(s in o.name for s in SKIP)
       and (DT != torch.float32 or not o.supports_autograd)]
ops = ops[start:end]


def to_dev(x):
    return x.to(DEV) if isinstance(x, torch.Tensor) else x


with open(logf, "a") as log:
    for op in ops:
        name = op.name + ("." + op.variant_test_name if op.variant_test_name else "")
        try:
            samples = list(op.sample_inputs("cpu", DT, requires_grad=False))[:3]
        except Exception:
            continue
        res = []
        for i, s in enumerate(samples):
            try:
                ro = op(s.input, *s.args, **s.kwargs)
            except Exception:
                res.append(f"s{i}:cpuerr")
                continue
            try:
                no = op(tree_map(to_dev, s.input), *tree_map(to_dev, s.args), **tree_map(to_dev, s.kwargs))
                torch.neuron.synchronize()
                rf = [t for t in tree_flatten(ro)[0] if isinstance(t, torch.Tensor)]
                nf = [t for t in tree_flatten(no)[0] if isinstance(t, torch.Tensor)]
                ok = len(rf) == len(nf)
                worst = ""
                for a, b in zip(rf, nf):
                    b = b.cpu()
                    if a.shape != b.shape or a.dtype != b.dtype:
                        ok, worst = False, f"meta {tuple(a.shape)},{a.dtype} vs {tuple(b.shape)},{b.dtype}"
                        break
                    if a.is_floating_point():
                        good = torch.allclose(a, b, atol=1e-4, rtol=1e-4, equal_nan=True)
                    else:
                        good = torch.equal(a, b)
                    if not good:
                        ok = False
                        worst = f"cpu={a.flatten()[:6].tolist()} neuron={b.flatten()[:6].tolist()}"
                        break
                res.append(f"s{i}:ok" if ok else f"s{i}:BAD[{worst}]")
            except Exception as e:
                res.append(f"s{i}:ERR[{type(e).__name__}: {str(e)[:100]}]".replace("\n", " "))
        st = "OK    " if all(r.endswith(":ok") or r.endswith("cpuerr") for r in res) else "ISSUE "
        log.write(f"{st} {name:40s} {' '.join(res)}\n")
        log.flush()
