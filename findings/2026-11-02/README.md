# Overnight report: torch-neuronx training issues

Run: 2026-11-02 08:22 UTC on trn2.3xlarge (4 logical cores). Scripts and logs in `agent_space/overnight/`.
Status: done. Active work ran 08:22-12:10 UTC. At about 12:10 the memory/mixed-device probe
(`overnight/mem_mixed.py`) was killed (exit 137) and the loop did not resume, so 12:10-17:22 had no work and that
probe has no results. A second session started at 17:25 UTC to continue; its results are added below and in the table.
A third session (2026-11-02 21:00 to 2026-11-03 00:30 UTC) added findings 17-20 (error recovery round).
A fourth session (2026-11-03 07:48 to 17:48 UTC, distributed) adds findings from 21 (`overnight/dist5/`).

| # | Finding | Kind | Known? |
|---|---|---|---|
| 1 | Eager SDPA with dropout: backward ignores the mask | silent wrong grads | no (fix tested) |
| 2 | Activation checkpointing + dropout: recompute uses a new mask (also: sharded DTensor dropout repeats one mask per shard) | silent wrong grads | no (partial fix tested) |
| 3 | Collectives into offset non-contiguous views write shifted data, change the view | silent corruption | no |
| 4 | Compiled `_foreach_add_/sub_/div_` on same-value operands (compiled SGD nesterov) | silent wrong update | no (fix tested) |
| 5 | Compiled LM step with `ids[:, 1:]` labels + CE fails to compile | error | compiler part known |
| 6 | ConvTranspose groups > 1 backward wrong grad shape | error | no (1-line fix tested) |
| 7 | Conv/ConvTranspose stride > 1 and dilation > 1 fail to compile | error | no (fix tested) |
| 8 | `tanh_backward` with broadcasting fails (jacrev of tanh) | error | no |
| 9 | NaN dropped by clamp/min/max/relu, hides NaN grads in clip_grad_norm_ | silent | yes (registry) |
| 10 | Default compile crashes on the 2nd input shape | error, poor message | limit documented |
| 11 | P2P ops need EFA, fail on a single instance | error | no |
| 12 | Uneven `all_gather` list breaks the process group | error | no |
| 13 | CE with a soft target gives no target grad | silent | code comment only |
| 14 | bf16/fp16 adaptive avg pool, non-divisible size, permuted values | silent wrong | no (fix tested) |
| 15 | `manual_seed` before the first Neuron op is ignored by the device RNG | silent, reproducibility | no (fix tested) |
| 16 | fp32 SDPA with L % 128 == 0 runs at bf16 precision (NKI flash path) | silent precision loss | no |
| 17 | `nll_loss` / `gather` with an out-of-range target or index return 0, no error | silent wrong loss/grads | no (fix tested) |
| 18 | OOM is a plain `RuntimeError`, reported async; `memory_allocated` goes negative | error type, wrong counter | no |
| 19 | Large tensors: elementwise ops (`full`, `mul`, `add_`) on 2^31+ elements fail with an unrecoverable DMA error that breaks the process; fills fail at 2^32+ ("Invalid NEFF") | error, process unusable | no |
| 20 | A compile error is raised again by the next unrelated op | error reported twice | no |
| 21 | Collectives compute int32/int64/fp64 in fp32: ints wrong above 2^24 (broadcast, scatter, all_reduce SUM/MAX/AVG, reduce_scatter), int64 loses the high word, fp64 overflows to inf | silent corruption | no (fix tested in part) |
| 22 | DTensor `cross_entropy` (`loss_parallel`) drops `label_smoothing` with no error | silent wrong loss | code note only |
| 23 | Process-group `timeout` is detected by the watchdog but not enforced (a stalled peer gives a hang); `work.wait(timeout=timedelta)` raises TypeError | hang, API | no |
| 24 | Object collectives fail when the object holds a Neuron tensor (`all_gather_object`, `broadcast_object_list`) | error | no |
| 25 | DDP comm hook that makes a tensor and uses it in the future callback fails ("Tensor not bound: lazy alloc"); breaks PowerSGD with error feedback (default) | error | no |
| 26 | In one process, `t.to("neuron:1")` from `neuron:0` returns a tensor still on `neuron:0` (silent); `copy_` across cores raises | silent wrong device | no |
| 27 | No compute/communication overlap: async all_reduce, a side-stream all_reduce, and a compiled graph with independent matmuls + all_reduce all take compute + comm time | performance | not documented |
| 28 | Uneven `all_to_all_single` (MoE expert parallel) recompiles its NKI kernel for each new max row count: 1.1-3 s per call when routing changes, vs 1.5 ms when shapes repeat | performance | design documented, cost not |
| 29 | `init_process_group(..., device_id=dev)` makes every 2-D `init_device_mesh` fail ("backend does not support splitting"); breaks HF `tp_plan="auto"` | error | no |

All "fix tested" items were tested in process (monkeypatch) or with a reverted edit. No file in the repo is changed.

## Findings (sorted by impact)

### 1. Eager SDPA with `dropout_p > 0`: backward ignores the dropout mask (silently wrong grads)

In eager mode, `F.scaled_dot_product_attention(..., dropout_p=p)` applies dropout in the forward, but the
backward computes the grads as if there was no dropout. Every eager transformer that trains with attention dropout
through SDPA (for example HF models with `attn_implementation="sdpa"` and `attention_dropout > 0`, or
`nn.MultiheadAttention`/`TransformerEncoderLayer` that call SDPA with dropout) gets wrong q/k/v grads. No error,
no warning. `torch.compile` is correct.

Repro (`overnight/sdpa_drop.py`, `overnight/sdpa_drop2.py`). With `v = I`, the output is the dropped attention
matrix, so the correct `v.grad` is `o^T @ G`:
```python
import torch, torch.nn.functional as F, torch_neuronx
B, H, S = 2, 2, 16
q = torch.randn(B, H, S, S, device="neuron", requires_grad=True)
k = torch.randn(B, H, S, S, device="neuron", requires_grad=True)
v = torch.eye(S).expand(B, H, S, S).contiguous().to("neuron").requires_grad_()
G = torch.randn(B, H, S, S, device="neuron")
o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3)
(o * G).sum().backward()
print((v.grad - o.detach().transpose(-1, -2) @ G).abs().max())
```
```
cpu                     causal=False fp32  v.grad err=0        causal=True fp32 err=0
neuron eager            causal=False fp32  v.grad err=1.03     causal=True fp32 err=3.8   (bf16: 1.1 / 2.12)
neuron torch.compile    causal=False fp32  v.grad err=3.5e-06  causal=True fp32 err=4.5e-06
neuron eager: v.grad(dropout_p=0.3) - v.grad(dropout_p=0) = 0.0   (backward has no dropout at all)
_scaled_dot_product_fused_attention_overrideable outputs: philox_seed slot = ((1,), int32)
```
Affected modules, checked with a dispatch spy in train mode (`overnight/sdpa_users.py`): all of these call
`aten._scaled_dot_product_fused_attention_overrideable` with `dropout_p=0.1`:
`nn.MultiheadAttention(dropout=0.1)` with `need_weights=False`, `nn.TransformerEncoderLayer(dropout=0.1)`,
HF GPT-2 with default config (`attn_pdrop=0.1`, default attention implementation), HF Llama with `attention_dropout=0.1`.

Other dropout ops (dropout, dropout1d/2d/3d, alpha_dropout, feature_alpha_dropout, native_dropout) are consistent.

Root cause: with `dropout_p > 0` the NKI impl declines, so the eager dispatch uses
`ScaledDotProductAttnMLIRImpl` / `ScaledDotProductAttnBackwardMLIRImpl`
(`python_ops/torch_mlir/ops/scaled_dot_product_attention.py`):
- forward lowering `_mlir_sdpa_fwd_traceable` draws `dropout_mask` with `torch.bernoulli` and uses it, but returns
  a fresh `torch.randint(...)` (1,) int32 in the `philox_seed` slot, so the mask is lost;
- backward lowering `_mlir_sdpa_bwd_traceable` always sets `dropout_mask = torch.zeros((1,))`, so the backward
  never applies dropout.
The other traceable pair in `python_ops/scaled_dot_product_fused_attention.py` does it right:
`_sdpa_fwd_traceable` returns the keep-mask in the `philox_seed` slot and `_sdpa_bwd_traceable` passes it to
`compute_attention_backward_decomposition` (which is also why compile is correct).

Fix (tested in process, no file changed, `overnight/sdpa_fix2.py patch 56`): make the two MLIR impls use the same
lowering as the NKI impls (`_sdpa_fwd_traceable` / `_sdpa_bwd_traceable`). Result: `v.grad err` 5e-06 (non-causal)
and 6e-06 (causal), dropout rate 0.30, and grads with `dropout_p=0` still match CPU (4e-06). Minimal source fix:
in `_mlir_sdpa_fwd_traceable` return `dropout_mask` in the `philox_seed` slot when `0 < dropout_p < 1`, and in
`_mlir_sdpa_bwd_traceable` use `philox_seed` as `dropout_mask` when `dropout_p > 0` (this exact edit is not tested;
the delegation above is). Note: the lowering result is
cached per shape, so a test must use a new shape or clear the cache.

Known? No. The only SDPA dropout xfail (`aten_xfails_registry.py`, NaN through dropped positions) is a different issue.
Filed as https://github.com/aws-neuron/torch-neuronx/issues/1031 (2026-11-02).

### 2. Activation checkpointing gives silently wrong gradients when the block has dropout

`torch.utils.checkpoint` (reentrant and non-reentrant) does not restore the Neuron RNG before the
recompute, so the recompute draws a new dropout mask. Backward uses that new mask, so the grads
do not match the forward. No error, no warning. Any model that uses activation checkpointing and
dropout (HF `gradient_checkpointing_enable()` with dropout > 0, FSDP2 + AC) trains on wrong grads.

Repro (`overnight/rng4.py`):

```python
import torch, torch.nn.functional as F, torch_neuronx
from torch.utils.checkpoint import checkpoint
torch.ones(1, device="neuron").cpu(); torch.manual_seed(0)
lin = torch.nn.Linear(16, 16).to("neuron"); x = torch.randn(8, 16).to("neuron").requires_grad_()
f = lambda x: F.dropout(lin(x), 0.5, training=True)
for step in range(3):
    lin.weight.grad = None
    o = checkpoint(f, x, use_reentrant=False); o.sum().backward()
    expect = ((o != 0).float() * 2.0).t() @ x.detach()   # grad implied by the forward mask
    print("step", step, "grad err:", (lin.weight.grad - expect).abs().max().item())
```

```
step 0 grad err: 15.433697700500488
step 1 grad err: 15.530723571777344
step 2 grad err: 13.489973068237305
```
CPU gives 0.0 for every step.

Root cause, two layers:
- `torch.neuron` has no `_initialized` attribute. `torch/utils/checkpoint.py` checks
  `getattr(device_module, "_initialized", False)` to decide if it must save device RNG state, so it
  never calls `torch.neuron.get_rng_state()` / `set_rng_state()`. (`torch.neuron.is_initialized()`
  is True; only the attribute is missing.)
- Even with `torch.neuron._initialized = True`, only the first step after `manual_seed` is right
  (`patch step 0 grad err 0.0`, steps 1 and 2 still ~15). Dropout draws from the on-device GPSIMD
  RNG (`python_ops/nki_kernels/rng_state.py`), but `get_rng_state()` returns the host Philox
  (seed, offset), and dropout never advances the offset. So the saved state is always
  "(seed, 0)" and restoring it rewinds to the seed start, not to the forward position.
  Also `set_rng_state()` skips the device reseed when the state equals `_last_seed_state`, so
  restoring the same state twice does nothing (`overnight/rng2.py restore_twice`: x0 == x1, x2 differs).

Real model (`overnight/hf_ckpt_dropout_noattn.py`): HF GPT-2 (2 layers, resid/embd dropout 0.1, `attn_pdrop=0` so
finding 1 is not involved), same seed, with and without `gradient_checkpointing_enable()`. The two runs must give the
same grads:
```
cpu:    loss no-ckpt=6.946085 ckpt=6.946085  max grad diff=0      (max |grad|=0.2514)
neuron: loss no-ckpt=6.892866 ckpt=6.892866  max grad diff=0.1324 (max |grad|=0.251)
```

FSDP2 + `apply_activation_checkpointing` (non-reentrant), 4 ranks, dropout 0.3, same seed: AC vs no-AC grads
differ by 0.042 (`overnight/probe_dist2.py fsdp2_ac_dropout`). With dropout 0 they match exactly (diff 0, and 3e-8 vs CPU).

Also under `torch.compile`: a compiled step with `checkpoint(..., use_reentrant=False)` around dropout has
the same error (grad error 15.4, 15.5, 13.5 over 3 steps; `overnight/probe_compile2.py compile_checkpoint_dropout`).
Compiled dropout without checkpointing is consistent (error 0).

Same root cause, other symptoms (`overnight/rng2.py`):
- `st = get_rng_state(); drop(); set_rng_state(st); drop()` gives a different mask when `st` was
  taken after some draws (`mid_stream`).
- `torch.random.fork_rng(device_type="neuron")` does not isolate dropout: the draw after the block
  continues from inside the block.

Prototype (`overnight/rng_fix_proto.py patch`, in process, no file changed): set `torch.neuron._initialized = True`,
make `get_rng_state()` append the 6 GPSIMD words from `torch.ops.nki_kernels.get_rng_state()`, and make
`set_rng_state()` write them back with `torch.ops.nki_kernels.set_rng_state()`. Result: checkpoint grads are exact
for 3 consecutive steps, non-reentrant and reentrant (`[0.0, 0.0, 0.0]`; unpatched `[15.4, 15.5, 13.5]` and
`[12.9, 12.8, 12.4]`). But a plain mid-stream `st = get; a = drop(); set(st); b = drop()` still gives `a != b`
(about 50% of positions match, n = 32 to 65536), so the 6 words are not the full device RNG state (the kernel uses
128 lanes, each seeded separately). The prototype is not a complete fix.
Proper fix: make on-device dropout counter based, like CUDA Philox: derive each draw from the host generator
(seed, offset), advance the offset by the number of values on each draw, and pass (seed, offset) to the NKI kernel.
Then `get_rng_state()`/`set_rng_state()` (seed + offset) already save and restore everything. Also set
`torch.neuron._initialized` and drop the `_last_seed_state` short cut for explicit restores.

Known? Not in docs or release notes. Related but different xfails: `dynamo_xfails_registry.py`
`test_checkpoint_with_dropout_caches_correctly` (compile HOP missing),
`dtensor_xfails_registry.py` `test_deterministic_dropout_1d` (LocalTensor).


Second effect of the same root cause (found in the distributed session, `overnight/dist5/probe_dist5f.py
tp_dropout_replicated`): DTensor dropout on a sharded DTensor uses the same mask on every shard. DTensor enables
`OffsetBasedRNGTracker` on Neuron, which gives each shard its own offset through `set_rng_state`. Dropout does not read
that offset, so the shards get equal masks. `rand`, `randn` and `bernoulli` use the offset correctly.
```
F.dropout(distribute_tensor(ones(8, 64), mesh, [Shard(1)]), 0.5).full_tensor(): 4 column shards identical (keep 0.438)
offset 0 vs 128 differ / 128 vs 256 differ / offset 0 twice equal:
  rand True / True / True;   bernoulli True / True / True;   dropout True / False / False
```
Replicated DTensor dropout and TP dropout on a replicated activation are equal on all ranks, as they must be.

### 3. Collectives into a non-contiguous view with a storage offset write to the wrong place and change the view

`dist.all_reduce`, `broadcast`, `all_gather_into_tensor` and `reduce_scatter_tensor` with an output that is a
non-contiguous view with `storage_offset() > 0` (for example `t[:, 1:3]`, `p.grad[:, 2:]`, `t.t()[1]`):
- write the result shifted by `storage_offset` elements in the base tensor (silent corruption of neighbours),
- change the caller's view in place to shape `(storage numel,)`, stride `(1,)`, same offset, so a later
  `v.cpu()` fails with `Direct Neuron-to-CPU copy exceeds source storage bounds`.
Contiguous offset views (`t[1:3]`) and offset-0 non-contiguous views (`t.t()`, `t[:, ::2]`) are correct. A normal
eager op with `out=` into the same kind of view is correct, so this is in the collective path.

Repro (`overnight/ar_view2.py`, `overnight/coll_view.py`, torchrun 2 or 4 ranks):
```python
base = torch.arange(12.0, device=dev).reshape(3, 4) * (rank + 1)
v = base[:, 1:3]
dist.all_reduce(v)
```
```
before v shape=(3, 2) stride=(4, 1) off=1
after  v shape=(12,) stride=(1,) off=1
base.cpu  [[0, 0, 3, 6], [3, 4, 15, 18], [7, 8, 27, 30]]     expected [[0, 3, 6, 3], [4, 15, 18, 7], [8, 27, 30, 11]]
v.cpu()   RuntimeError: Direct Neuron-to-CPU copy exceeds source storage bounds
```
4 ranks, `p.grad = full((4, 6), rank+1); dist.all_reduce(p.grad[:, 2:])` gives rows
`[1, 1, 1, 1, 10, 10]...` instead of `[1, 1, 10, 10, 10, 10]...`. broadcast / all_gather_into_tensor /
reduce_scatter_tensor show the same 1-element (offset) shift.

Root cause: the output epilogue path in `NeuronBackend::SubmitInNeffCollective`
(`csrc/distributed/NeuronBackend.cpp`, `ProcessOutputTransformations(..., fuse_transforms=true)` ->
`TryCreateEpilogue` in `csrc/core/lazy_materialization/Transformations.cpp`). With
`TORCH_NEURONX_ENABLE_EPILOGUE=0` the result and the view metadata are correct (the contiguous temp +
`CopyBackToOriginal` path runs). `TORCH_NEURONX_ENABLE_SCATTER_IOTA=0` alone also makes all 4 collectives correct,
so the bad path is the SCATTER_IOTA slice epilogue (`TryCreateDecomposedEpilogue`, `layout.has_slice`: base =
`output.as_strided({storage_span}, {1}, 0)`, offset passed as a scatter parameter). `TORCH_NEURONX_ENABLE_PROLOGUE=0`
does not help. For normal ops, `LazyTransformationManager.cpp` adds the flat storage as an extra src input for each
SCATTER_IOTA output (around line 516) and the same view is correct with `out=`; `SubmitInNeffCollective` builds
its `XLACompilableKernelExecution` directly, so the likely gap is that this collective path does not do the same
wiring. I did not confirm the exact line.
Possible fix: in `SubmitInNeffCollective`, skip SCATTER_IOTA epilogues (or all epilogues for outputs with `storage_offset() != 0`) and use
the existing contiguous-temp + copy-back path, until the epilogue offset handling is fixed. Known? No.

### 4. Compiled `_foreach_add_/_sub_/_div_` give wrong values when both operands hold the same data (compiled SGD nesterov is wrong)

Under `torch.compile`, `torch._foreach_add_(g, b, alpha)` returns `alpha*(g + b)` instead of `g + alpha*b`
when `b` is computed in the same graph from `g` and has the same values (`g.clone()`, `g * 1.0`, `g + 0`).
Two separate graph inputs are fine. Eager is fine. `torch.optim.SGD(momentum, nesterov=True)` (foreach,
the default) creates the momentum buffer as `clone(grad)` on the first step and then runs
`_foreach_add_(grads, bufs, alpha=momentum)`, so the first update is wrong and the error stays in training.

Repro (`overnight/alias_min.py`, `overnight/sgd_n.py`):
```python
import torch, torch_neuronx
def f(g):
    b = torch.clone(g)
    torch._foreach_add_([g], [b], alpha=0.9)
g = torch.ones(4, device="neuron"); torch.compile(f, backend="neuron")(g); print(g)   # 1.8, CPU 1.9

p = torch.nn.Parameter(torch.ones(4, device="neuron"))
opt = torch.optim.SGD([p], lr=1.0, momentum=0.9, nesterov=True)
p.grad = torch.ones(4, device="neuron"); torch.compile(opt.step, backend="neuron")()
```
```
foreach cpu        (delta, buf, grad) [(1.9, 1.0, 1.9), (2.71, 1.9, 2.71)]
foreach neuron cmp (delta, buf, grad) [(1.8, 1.0, 1.8), (2.71, 1.9, 2.71)]
```
In a 4-step MLP run with a compiled train step: loss CPU `[2.4335, 1.5226, 0.7633, 0.3494]`, Neuron
`[2.4335, 1.5381, 0.7691, 0.3511]`. `foreach=False` and eager are correct.
Same hazard: `_foreach_sub_(g, g.clone(), alpha=.5)` gives 0.0 (CPU 1.5), `_foreach_div_(g, g.clone())`
gives 0.111 (CPU 1.0). lerp/addcmul/addcdiv with the same pattern are correct (`overnight/alias4.py`).

Root cause: the compile decomposition (`neuron_dynamo_backend/decompositions.py`, `_make_foreach_list_decomp`)
calls nkilib `add_tensor_kernel`. Its body (`nkilib/experimental/foreach/foreach_elementwise.py`,
`_tensor_alpha_kernel_body`) loads `data1` to tile `t1`, `data2` to tile `t2`, then scales `t2` in place
(`nisa.tensor_scalar(t2, t2, nl.multiply, alpha_sb)`) before `t1 + t2`. When the compiler sees both operands
as the same value (the clone is folded), the two loads share one tile, so `t1` is also scaled. The div kernel
has the same pattern (`nisa.reciprocal(t2, t2)`).

Fix (tested in process, no file changed): a copy of the kernel that writes `alpha*t2` to a new tile `t2s`
(`overnight/kpatch/patched_add.py`), put into the decomposition closure, gives 1.9 for n=4, 1000, 200000,
and distinct inputs stay correct (`overnight/alias_fix2.py patch`). Apply the same change to the sub and div
bodies. A torch-neuronx side guard is also possible (use the torch op when `b` is derived from `a`), but the
kernel change is simpler. Known? No.

### 5. Compiled LM train step with shifted labels fails to compile (`F.cross_entropy(logits, ids[:, 1:])`)

A compiled step that shifts the labels inside the compiled function fails with a compiler internal error. Tied
or untied weights do not matter (`overnight/tie.py`, `overnight/ce_slice.py`):
```python
import torch, torch.nn.functional as F, torch_neuronx
ids = torch.randint(0, 50, (4, 9)).to("neuron"); x = torch.randn(32, 50).to("neuron")
torch.compile(lambda x, ids: F.cross_entropy(x, ids[:, 1:].reshape(-1)), backend="neuron")(x, ids)
```
```
[INTERNAL_ERROR] [NCC_IBCG902] BIRCodeGenLoop error: tuple index out of range
```
`F.nll_loss(x.log_softmax(1), ids[:, 1:].reshape(-1))` in the same place works.

Root cause, two parts:
- `python_ops/torch_mlir/ops/loss.py` `cross_entropy_loss` raises `DataDependentOutputException` under fake
  mode on purpose, so Dynamo breaks the graph at every `F.cross_entropy` (TODO in the code: memory of the
  compiled backward). The graph before the break then holds only `slice -> clone -> _unsafe_view` on int64.
- A graph that only moves int64 data fails in neuronx-cc 2.28: `torch.compile(lambda t: t[:, 1:].reshape(-1))`
  fails for int64 and works for int32/float32/bf16, and works when any compute is added (`t[:, 1:].reshape(-1) + 1`)
  (`overnight/slice_min.py`).

Known? The compiler part is known: `neuron_compile_ops_xfails_registry.py` "int64 slice operations fail with
NCC_IBCG902" (select/unbind/vsplit int64 OpInfo tests). The training effect (any compiled LM step that shifts
labels before `F.cross_entropy`) is not in docs. Possible workaround in torch-neuronx: run a partition that has
only int64 view/copy ops in eager (or on CPU) instead of sending it to neuronx-cc. Also, every `F.cross_entropy`
in a compiled region costs a graph break.

### 6. ConvTranspose with groups > 1 cannot train (backward returns weight grad of wrong shape)

Every `nn.ConvTranspose{1,2,3}d(..., groups>1)` backward fails, eager and `torch.compile`.
Docs list ConvTranspose2d as Native.

```python
import torch, torch.nn as nn, torch_neuronx
m = nn.ConvTranspose2d(4, 8, 3, stride=2, groups=2).to("neuron")
m(torch.randn(2, 4, 5, 5, device="neuron")).sum().backward()
```
```
RuntimeError: Function ConvolutionBackward0 returned an invalid gradient at index 1 - got [4, 2, 3, 3] but expected shape compatible with [4, 4, 3, 3]
```
With `groups=4` it returns `[4, 0, 3, 3]`.

Root cause: `torch_neuronx/neuron_dynamo_backend/decompositions.py`, `torch_convolution_backward_overrideable`:
`cout = weight.shape[0] if not transposed else weight.shape[1]`. A transposed weight is
`[Cin, Cout/groups, *k]`, so `weight.shape[1]` is already per group, and `cout_per_group = cout // groups`
divides by groups a second time. The eager path uses the same decomposition.

Fix (tested locally, then reverted): `cout = weight.shape[0] if not transposed else weight.shape[1] * groups`.
After the change, eager g=2 stride 1/2, g=4 stride 3 and compiled g=2 all match CPU (`overnight/convt.py`,
`overnight/convt_compile.py`).

Known? No.

### 7. Conv/ConvTranspose with both stride > 1 and dilation > 1 fail to compile (Conv: backward, ConvTranspose: forward)

```python
nn.ConvTranspose2d(4, 8, 3, stride=2, dilation=2).to("neuron")(torch.randn(2, 4, 5, 5, device="neuron"))
```
```
[NCC_EVRF010] Operation convolution does not support both input dilation and kernel dilation simultaneously.
%convolution.6 = f32[2,8,13,13] convolution(...), window={size=3x3 pad=4_4x4_4 lhs_dilate=2x2 rhs_dilate=2x2}
```
Plain `nn.Conv1d/Conv2d(..., stride=2, dilation=2)` also cannot train: the forward works, and the backward (a
transposed conv with lhs and rhs dilation) fails with the same `NCC_EVRF010` (`overnight/conv_sd.py`; stride 2 +
dilation 1 and stride 1 + dilation 2 work). The OpInfo sweep hits it for conv1d, conv2d, conv_transpose1d/2d/3d.

Root cause: compiler limit (neuronx-cc HLO verifier). torch-neuronx lowers it to one HLO conv with both
`lhs_dilate` (stride) and `rhs_dilate` (dilation). Fix (tested in process for the Conv backward, no file changed, `overnight/conv_fix.py patch 27`): when stride > 1
and dilation > 1, expand the kernel with zeros (per spatial dim: `stack([w] + [zeros]*(d-1), dim=ax+1).flatten`
then `narrow` to `(k-1)*d+1`) and call the transposed conv with `dilation=1`, so the HLO has only `lhs_dilate`. With
this shim on the `F.conv_transpose*` calls in `torch_convolution_backward_overrideable`, Conv2d stride 2 / dilation 2,
Conv1d stride 3 / dilation 2, Conv2d stride 2 / dilation 3 train and match CPU (x.grad diff <= 1e-6, w.grad diff
<= 1.5e-5). An in-place slice assignment into a zero kernel does not lower (torch-mlir `aten.copy_` error), so use
the functional form. The same wrapper on `F.conv_transpose2d` also fixes the ConvTranspose2d stride 2 / dilation 2 forward and backward (fwd diff 4.8e-7, x.grad 9.5e-7, w.grad 3.8e-5; `overnight/convt_fix.py 13`); in the source the place is `conv_kernel` in `python_ops/torch_mlir/ops/convolution.py`. Stride 1 + dilation 2 works.
Known? No mention in docs or registries.

Side note: the compile error is raised asynchronously, so it can appear on a later unrelated op (in my probe it
surfaced from `torch.manual_seed()` in the next case). This is documented; `NEURON_LAUNCH_BLOCKING=1` makes it synchronous.

### 8. `tanh_backward` fails when its two inputs need broadcasting (breaks `torch.func.jacrev/jacfwd` of tanh)

```python
torch.ops.aten.tanh_backward(torch.ones(4, 4, device="neuron"), torch.rand(4, device="neuron"))
torch.func.jacrev(torch.tanh)(torch.randn(4, device="neuron"))   # same error
```
```
TorchMlirCompilerError: Failed Torch-MLIR Lowering Torch Backend IR -> StableHLO IR: error: 'chlo.broadcast_multiply' op inferred type(s) 'tensor<4xf32>' are incompatible with return type(s) of operation 'tensor<4x4xf32>'
```
CPU broadcasts. `sigmoid_backward` with the same shapes works. jacrev of sigmoid/exp/sin works.
Root cause: `python_ops/torch_mlir/ops/activation.py` `torch_tanh_backward` passes the raw shapes, and the
torch-mlir tanh_backward lowering takes the result type from `output`. Expanding `output` to the
broadcast shape first works (`tanh_backward expanded y` OK). Known? No. Impact: low (functorch users).

### 9. NaN is dropped by clamp/min/max/relu, so `clip_grad_norm_` hides NaN grads (known root cause)

`clamp`, `clamp_min/max`, `minimum`, `maximum`, `relu`, `relu6`, `hardtanh`, `max()/min()/amax`,
`max(dim)`, `argmax` and `sign` return a number for a NaN input; CPU returns NaN (`overnight/clampnan.py`).
Training effect: `clip_grad_norm_` computes `clamp(max_norm / (nan + eps), max=1.0)` = 1.0 on Neuron, so a
NaN grad does not spread to the other grads like on CPU/CUDA:
```
clip nan  cpu: (nan, [nan, nan, nan, nan])   neuron: (nan, [1.0, nan, 0.0, 1.0])
```
A NaN in activations is also turned into 0 by ReLU, so a diverging run can look healthy.
Known? Yes, as a compiler bug in `aten_skips_registry.py` (`MINMAX_CLAMP_NAN_PROPAGATION`) and
`aten_xfails_registry.py` (`reduction.nan_drop`). Not in `docs/torch_api_compatibility.md` (clamp is "Native").

### 10. Default `torch.compile(backend="neuron")` crashes on the second input shape (low; documented limit, poor failure)

README says to pass `dynamic=False`. Without it, PyTorch's default `automatic_dynamic_shapes` recompiles with
symbolic sizes after the first shape change, and the Neuron backend crashes instead of falling back:
```python
f = torch.compile(nn.Linear(16, 16).to("neuron"), backend="neuron")
f(torch.randn(2, 8, 16, device="neuron")); f(torch.randn(2, 12, 16, device="neuron"))
```
```
RuntimeError: .../KernelFunction_impl.h:128: SymIntArrayRef expected to contain only concrete integers
  (from neuron_dynamo_backend/executor.py:276 _allocate_output)
```
`dynamic=True` fails on the first call. `dynamic=False` works for all shapes (`overnight/dyn_default.py`).
`docs/AI-dynamo-backend.md` says `dynamic_shape_analysis` "bails out if symbolic ints remain", but the run
reaches the executor. Variable sequence length training is the common trigger. Possible fix: in the backend,
set `assume_static_by_default`/`automatic_dynamic_shapes=False` when the backend is selected (the eager
kernel path already does this in `python_ops/torch_mlir/kernel.py:833`), or raise a clear error that says
"pass dynamic=False". Known? The limit is documented; the crash is not.

### 11. Point-to-point ops fail on an instance without EFA (send/recv, isend/irecv, batch_isend_irecv)

On this trn2.3xlarge (one Neuron device, no EFA; `fi_info -p efa` -> `fi_getinfo: -61`), every P2P op fails between
ranks on the same device:
```
RdmaManager: DiscoverEFAInfo failed: fi_getinfo failed: No data available
```
(`overnight/coll_each.py send_recv | isend_irecv | batch_p2p`, torchrun 4 ranks). All collectives work on the same
setup. Pipeline parallelism (`torch.distributed.pipelining`) depends on P2P, so it cannot run here.
Root cause: `NeuronBackend::EnsureRdmaInit` -> `RdmaManager::EnsureInitialized` (`csrc/core/rdma/RdmaManager.cc`)
always uses libfabric with `prov_name="efa"` and `TORCH_CHECK`s the result. There is no intra-device path.
Possible fix: when EFA is absent (or peers are on the same device), implement send/recv with a 2-rank collective
(for example a broadcast or all_gather in a cached 2-rank subgroup), or document that P2P needs EFA.
Known? `docs/torch_api_compatibility.md` lists send/recv/isend/irecv/batch_isend_irecv as "Native" with no EFA note;
release notes 2.33 say "Point-to-point support includes send, recv, and batch_isend_irecv". Not known as a limit.

### 12. `dist.all_gather` with different tensor sizes per rank fails and breaks the process group (low)

```python
outs = [torch.empty(i + 1, device=dev) for i in range(world)]
dist.all_gather(outs, torch.full((rank + 1,), float(rank), device=dev))
```
```
rank0: copy_(): source shape [2] is not broadcastable to destination shape [1]
next collective (all_gather_object): NRT Execution error ... operation=allgather "Invalid NEFF, instruction, or input"
```
NCCL and Gloo accept uneven sizes in the list form. The Neuron backend assumes equal sizes, raises on some ranks
only, and the ranks that did not raise leave the process group in a bad state. Possible fix: check that all output
sizes match the input size and raise the same clear error on every rank, or implement the uneven case with padding.
Known? No mention in docs.

### 13. `F.cross_entropy` with a probability target that requires grad gives no target gradient (silent, low)

```python
x = torch.randn(2, 3, device="neuron", requires_grad=True)
t = torch.softmax(torch.randn(2, 3), -1).to("neuron").requires_grad_()
F.cross_entropy(x, t).backward(); print(t.grad)
```
```
cpu    target.grad: [0.4082, 0.8633, 0.4836, 0.869, 0.3382, 0.5765]
neuron target.grad: None
```
Found by the OpInfo sweep (`nn.functional.cross_entropy` sample 1), repro `overnight/conv_sd.py ce`. Matters when the
soft target comes from a trainable network (self-distillation, co-training, some label-smoothing tricks); that part of
the gradient is dropped without a warning.
Root cause: `python_ops/torch_mlir/ops/loss.py` `cross_entropy_loss` wraps the kernel in `_CrossEntropyAutograd`, and
the comment says "A differentiable target or weight is accepted and receives no gradient, matching the previous
behaviour." So it is deliberate, but it differs from PyTorch. Possible fix: when `target.requires_grad` and target is
floating point, use the CompositeImplicitAutograd decomposition (as the code already does for wrapper subclasses), or
raise. Known? Only the code comment.

### 14. bf16/fp16 adaptive average pooling with a non-divisible output size returns permuted values (silent)

`F.adaptive_avg_pool1d(x, 7)` / `F.adaptive_avg_pool2d(x, (1, 7))` / `F.interpolate(mode="area")` in bf16 or fp16, for
some shapes where the input length is not a multiple of the output size, return the right values in the wrong places.
fp32 is correct. Eager and `torch.compile` give the same wrong result. Found by the bf16 OpInfo sweep
(`interpolate.area`, rel err ~1.0); repro `overnight/aap1d2.py`, `overnight/aap1d3.py`, `overnight/aap1d4.py`:
```python
x = (torch.arange(2 * 3 * 64).reshape(2, 3, 64) / 100).to(torch.bfloat16)
ref = F.adaptive_avg_pool1d(x, 7); got = F.adaptive_avg_pool1d(x.to("neuron"), 7).cpu()
```
```
ref[0] row0 [0.0449, 0.1348, 0.2246, 0.3145, 0.4043, 0.4961, 0.5859]
got[0] row0 [0.0449, 0.6836, 1.3281, 0.1348, 0.7734, 1.4141, 0.2246]     (C and L orders mixed)
bf16 (2,3,64)->7 rel 0.262   (2,3,22)->7 rel 0.164   (1,3,64)->7 compile error NCC_INIC902 NeuronInstComb (APIndex.py:205)
bf16 (4,8,64)->7 ok   (2,3,21)->7 ok   (2,3,64)->8 ok   fp32 all ok
```
The output strides are the normal contiguous ones, so the data itself is laid out wrong. Since the compile path
(`neuron_dynamo_backend/decompositions.py`) and the eager torch-mlir path give the same wrong values, the likely
cause is in neuronx-cc for the 16-bit reduce pattern (the `(1,3,64)` shape crashes the same compiler stage).
The forward uses the upstream `aten._adaptive_avg_pool2d` decomposition (listed in `decompositions.py:203`), which
is shared by eager and compile, so the bad pattern can come from that decomposition plus the compiler.
Fix (tested on device, `overnight/aap_fix.py`): compute the forward like the existing backward decomposition, as two
separable matmuls with `_adaptive_avg_pool_backward_weight` (`einsum("hq,...hw->...qw", W_h, x)` then
`einsum("...qw,wv->...qv", ., W_w)`, accumulate in fp32, cast back). On Neuron bf16 this gives rel err 0 vs the CPU
bf16 op for (2,3,64), (2,3,22), (4,8,64), and (1,3,64) no longer crashes the compiler. Known? No mention found.
Also from the sweep, lower impact: `grid_sample`/`grid_sampler_2d` grad w.r.t. the grid in bf16 has rel err 0.5 to 1.2
vs fp32, where CPU bf16 has 0.12 to 0.44 (`overnight/bf16_check.py`).

### 15. `torch.manual_seed` called before the first Neuron op does not seed the device RNG (all seeds give the same run)

If `torch.manual_seed(seed)` runs before any Neuron tensor exists (the usual first line of a training script), on-device
random ops ignore the seed: `torch.randn(device="neuron")`, dropout masks and `nn.Linear(device="neuron")` init are the
same for every seed, and stay the same for later draws (`overnight/seed_first.py`, `overnight/seed_first2.py`):
```
seed 1  randn [0.3623, 0.104, -0.7531, -0.79, 0.9321, 0.0703]
seed 2  randn [0.3623, 0.104, -0.7531, -0.79, 0.9321, 0.0703]
seed 1/2 dropout [2,0,0,0,2,0,0,2,0,0,2,2] (same)       seed 1/2 nn.Linear(4,2,device="neuron").weight (same)
seed 1/2, 3 consecutive randn draws: identical
```
If any Neuron op runs first, then `manual_seed` works (`overnight/seed_first3.py`: seeds 1 and 2 differ). Effect: seed
sweeps, "different seed per rank" data augmentation/dropout, and run-to-run variance studies silently use one RNG stream.
Root cause: `torch_neuronx/neuron.py` `manual_seed`/`manual_seed_all` only seed the device when `is_initialized()`.
`torch_neuronx/__init__.py` `_lazy_init()` is meant to apply the pending seed ("Set any seed set via manual_seed before
runtime was ready"), but the first tensor op initializes the runtime through the C++ path, so `_lazy_init()` never runs
(`overnight/seed_trace.py`: after `randn`, `is_initialized()` is True, `torch_neuronx._initialized` is still False, the
generator has `initial_seed()=7`, and `_set_device_rng_seed` was never called).
Fix (tested in process, `overnight/seed_fix.py ... patch`): make `manual_seed`/`manual_seed_all` call
`torch_neuronx._lazy_init()` first. Seeds 1/2/1 then give different/different/same values for randn and dropout.
A cleaner fix is to apply the generator seed from the C++ runtime-init path, so seeding does not force an early init.
Seeding inside the first `_generate_raw` call does not work (NKI dispatch error). Known? No.

### 16. fp32 SDPA silently runs at bf16 precision when the sequence length is a multiple of 128 (NKI flash path)

With fp32 q/k/v, no mask, no dropout and `L % 128 == 0` (the usual 512/1024/2048 training lengths), SDPA forward and
backward have bf16-level error (rel ~3e-3); other fp32 shapes have rel ~7e-7. The output dtype is still float32, so
nothing tells the user. Eager and `torch.compile` are the same. Found by `overnight/sdpa_sweep.py` (140 configs: the 16
bad ones are exactly fp32, L=128, no mask, every head dim 64/72/128/256, causal or not, MHA or GQA); repro
`overnight/nki_sdpa_prec.py`:
```
L=768 fp32: neuron vs cpu fp32 0.00297 | cpu with bf16-rounded inputs vs cpu fp32 0.00271 | cpu bf16 0.00337
L=384 with TORCH_NEURONX_ENABLE_NKI_SDPA=0: neuron vs cpu fp32 6.8e-07
dq/dk/dv rel err on the same path: 4e-4 to 6e-3 (fp32 tolerance 1e-4)
```
Root cause: `_can_use_nki_flash_attention` (`neuron_dynamo_backend/decompositions.py:2161`, also used by the eager impl)
checks shapes, mask, dropout and scale but not dtype, so fp32 inputs go to nkilib `attention_cte`, whose docstring says
"Input dtypes can be bfloat16, float16 or float32. The kernel uses float32 for softmax and bfloat16 for other
operations." Possible fix: add `query.dtype in (torch.bfloat16, torch.float16)` to the NKI gates (fwd and bwd), or an
opt-in flag like CUDA's `allow_tf32`, and document it. Known? No (`torch_api_compatibility.md` says "Native").
Side note: the per-shape lowering cache ignores `TORCH_NEURONX_ENABLE_NKI_SDPA`. A shape first lowered with the flag at 0
keeps the non-NKI result in later processes with the flag at 1 (L=384 above), so the env var does not act for cached
shapes.

Training impact (added in the distributed session, `overnight/dist5/llama_scale.py`): an 8-layer Llama-style LM
(GQA, RoPE, SwiGLU) on TP+FSDP2 2x2, 300 AdamW steps, fp32. With S=128 the loss leaves the CPU curve at step 11 and
ends at 5.81 vs 4.63 on CPU; step-0 q/k weight grads are off by 5e-3 with or without TP (`tp_grad_check.py`). With
S=120 (not a multiple of 128) the curves are equal to step 29 and end at 2.76 vs 2.70. FSDP2-only and DDP runs at
S=128 stayed close to CPU (2.05 vs 2.04), so the size of the effect depends on the run.

### 17. `nll_loss` and `gather` with an out-of-range target or index return 0 (silent wrong loss and grads)

`F.nll_loss(log_probs, target)` with a target `>= C` or `< 0` (and not `ignore_index`) gives no error. Each bad row adds 0
to the loss but still counts in the mean, and its gradient is dropped. `x.gather(1, idx)` with an out-of-range index
returns 0. CPU raises `IndexError` and CUDA raises a device-side assert. Typical causes: a label equal to
`num_classes` (off by one), or `-1` used as the "ignore" label without `ignore_index=-1`. `F.cross_entropy` is not
affected: it has an in-graph check.

Repro (`overnight/r5/oob_nll.py`, `overnight/r5/oob1.py`): N=512, C=32000, 25% of targets set to C+3:
```
RES nll    N=512 C=32000: no error, loss=10.8600 (loss over valid rows only 10.8543), grad nonzero rows=384/512
RES gather N=512 C=32000: no error, loss=10.8600 (loss over valid rows only 10.8543), grad nonzero rows=384/512
RES nll_target_-5   cpu=raises neuron: no error -> [0.0]
RES gather_7        cpu=raises neuron: no error -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
Other index ops checked in the same way raise correctly: embedding (real table sizes), index_select, index_add,
scatter_add, index_put, take, getitem, one_hot, cross_entropy. Only a tiny embedding table (10x4) returned a wrong row
with no error (`oob_emb_sizes.py`); from 100x8 up it raises. The process recovers after each of these errors.

Root cause: `cross_entropy_loss`, `one_hot` and the index ops call
`python_ops/nki_kernels/bound_check.assert_index_in_bounds` inside the graph, and `_ce_error_handler` turns the
device fault into the CPU `IndexError`. `nll_loss_forward` (`python_ops/torch_mlir/ops/loss.py`) and `aten::gather`
(`python_ops/torch_mlir/ops/indexing.py`) have no such check, and the device gather returns 0 for an out-of-range row.

Fix (tested in process, `overnight/r5/nll_fix_proto.py`): pass the target through `assert_index_in_bounds` after
mapping `ignore_index` to 0, as `cross_entropy_loss` does. With valid targets the loss matches CPU (10.87669); with a
bad target the device raises. Add an error handler like `_ce_error_handler` to give the CPU message. Do the same for
`aten::gather` with `index`.

Known? No.

### 18. OOM is a plain `RuntimeError` and is reported later; `memory_allocated` goes negative after a failed allocation

(a) An out-of-memory error is `RuntimeError("NRT Execution error ... lazy::AllocBind: NRT_RESOURCE ...")`. It is not
`torch.OutOfMemoryError`, and the text does not contain "out of memory". Code that catches OOM to retry with a smaller
batch does not see it: `except torch.OutOfMemoryError`, accelerate `find_executable_batch_size` and HF Trainer
`auto_find_batch_size` (both check for `torch.OutOfMemoryError` or "out of memory" text), Lightning `BatchSizeFinder`.

(b) Allocation is lazy, so `torch.ones(...)` returns at once and the OOM shows at a later sync, sometimes on another
tensor. A loop that allocates until it gets an error never stops: `overnight/r5/recover.py oom_gradual` ran 15 min
with 343 failed retries and no Python exception. `oom_silent.py`, 8 x 4 GiB on a 24 GiB core, no sync in the loop:
```
alloc 0..7: returned in 0.00s, memory_allocated=4 GiB      (all 8 calls return)
read 0: raised RuntimeError: NRT Execution error ... operation=neuron::alloc::lazy   (tensor 0 is a good tensor)
read 1..4: t[-1]=1.0
read 5..7: raised RuntimeError: ... operation=neuron::copy::neuron_to_cpu
```
(c) Each failed allocation subtracts its size from `memory_allocated()`, and the value stays wrong (`oom_type.py counter`):
```
after failed alloc of 64 GiB:   allocated=-64.00 GiB
after failed alloc of 256 GiB:  allocated=-320.00 GiB
after failed alloc of 4096 GiB: allocated=-4416.00 GiB
after good 1 GiB alloc:         allocated=-4415.00 GiB reserved=1.00 GiB
```
After an OOM, training continues correctly and no device memory leaks (`oom_leak2.py`: same usable size as a fresh
process).

Root cause, (a): `csrc/core/OperationExecutionEngine.cpp` `ScheduleAllocOperation` throws
`ExecutionRuntimeException(..., NRT_RESOURCE)`, and the Python side gets a generic RuntimeError. (b) is the lazy
allocator design (`TryDeferAlloc`). (c) is not checked in the allocator; the failed-allocation path probably runs the
free-side counter update.

Possible fix: map `NRT_RESOURCE` from an allocation to `torch.OutOfMemoryError` with text that starts with
"Neuron out of memory" (CUDA uses "CUDA out of memory"). Do not change the allocated counter when the allocation fails.

Known? No. `docs/` has no text on OOM behavior.

### 19. Large tensors: elementwise ops fail at 2^31 elements (unrecoverable DMA error), fills fail at 2^32 elements

Update from the distributed session (`overnight/dist5/full_big.py`, one process): `torch.full`, `x * 2` and `x.add_(1)`
on a tensor with 2^31 or more elements (bf16 or int8) fail with NRT status 1203 "A DMA engine encountered an
unrecoverable error" (the runtime prints AXI timeout/response errors for every DMA queue). After that, every later op
in the process fails, so the job must restart. 2^31 - 4096 elements work. `torch.zeros`, `torch.ones` and `fill_` work at
2^31 + 4096 (they use another kernel), and fail later at 2^32 (below).
```
n=2**31+4096 zeros bfloat16:     ok x[0]=0.0 x[-1]=0.0 | next op ok
n=2**31+4096 ones / fill_ bf16:  ok                    | next op ok
n=2**31-4096 full bfloat16:      ok                    | next op ok
n=2**31      full bfloat16:      ERR ... NRT status: 1203, A DMA engine encountered an unrecoverable error | next op ERR
n=2**31+4096 mul / add_ bf16:    ERR ... status 1203 | next op ERR
n=2**31+4096 full int8:          ERR ... status 1203 | next op ERR
```
Collectives are not affected: `all_reduce`, `reduce_scatter_tensor` (input) and `all_gather_into_tensor` (output) of
2^31 + 4096 bf16 elements are correct when the input is made with `empty` + `fill_` (`big_numel_coll.py`); with
`torch.full` the input creation fails first. Where it can happen: a full (unsharded) parameter of 2^31+ elements, for
example a 256k x 8192 embedding (2.1e9) that FSDP unshards for the forward, or an optimizer update on such a tensor.
Probable root cause: a signed 32-bit element offset or DMA descriptor size in the elementwise kernels. Possible fix:
split the op into chunks below 2^31 elements, or raise a clear error before the launch.

Original finding (2^32):

`torch.ones`, `torch.zeros` and `fill_` fail for a tensor with 2^32 or more elements, for every dtype checked (int8,
bf16, fp32). 2^32 - 4096 elements work. The memory is available: 5 x 4 GiB tensors work, but one 16 GiB fp32 tensor
fails. (`overnight/r5/numel32.py`)
```
RES bfloat16 n=2**32-4096 ones: ok first=1.0 last=1.0
RES bfloat16 n=2**32 ones: FAIL NRT Execution error ... operation=aten::ones error message="NRT EXECUTION FAILED: Failed to load model, Invalid NEFF, instruction, or input"
RES bfloat16 n=2**32 empty_fill / zeros_add, int8 n=2**32, bfloat16 n=3*2**31: FAIL (same)
```
Impact is low on this instance (24 GiB per logical core), but a flat buffer of 4.3e9 bf16 elements (8.6 GB) is
possible, and the error does not say what is wrong. Tensors between 2^31 and 2^32 elements work (round 5).
Root cause not found; probably a 32-bit size in the fill kernel or in the NEFF it makes. Possible fix: split the fill,
or raise a clear error for numel >= 2^32.

Known? No.

### 20. A compile error is raised a second time by the next unrelated op

When a compiled graph fails to compile, the compiled call raises (as expected). The next eager op then raises the same
"Compilation failed for cache key ..." error, and its result is lost. Later ops work. `NEURON_LAUNCH_BLOCKING=1` gives
the same result. Code that catches a compile failure and falls back to eager fails on the first eager op.
(`overnight/r5/sticky.py`)
```
RES compile error raised: RuntimeError
RES op 0: FAIL Compilation failed for cache key '4ab961bb...'    <- torch.full(...) * 2, unrelated
RES op 1: ok [2.0, 2.0, 2.0, 2.0]
```
Probable root cause: the failed operation reports its error to the caller and also stashes it on the stream
(`csrc/core/streams/StreamImpl.cpp` `HandleErrorWithExceptionPtr` -> `StashException`). The first report does not
clear the stash, so the next `Synchronize()` (line 162) throws it again. Possible fix: clear the stashed exception when
the same error is already given to the caller.

Known? No.

### 21. Collectives compute integer and float64 data in fp32: ints are wrong above 2^24, int64 above 2^31 loses the high word, fp64 overflows (silent)

Two separate problems. (a) `broadcast`, `scatter` and `all_reduce` on int32 or int64 tensors are exact only up to 2^24
(16,777,216), because the values go through an fp32 sum. `all_gather` is exact. `broadcast` is a zero + all_reduce
SUM (`csrc/distributed/NeuronBackend.cpp` around line 1125, "Zero non-root ranks, then allreduce SUM"), so it has the
same limit. `distribute_tensor` and `full_tensor` of an integer DTensor use these collectives too.
(`overnight/dist5/int_precision.py`, 4 ranks, `arange(8) + base`)
```
int32 base 1000:          broadcast True,  all_gather True, distribute_full True,  scatter True,  all_reduce True
int32 base 16777216:      broadcast False, all_gather True, distribute_full False, scatter False, all_reduce False
                          distribute_tensor(...).full_tensor()[:3] = [16777216, 16777216, 16777218]
int32 base 100000000:     all False except all_gather;  full_tensor()[:3] = [100000000, 100000000, 100000000]
int64: same results as int32 for these bases
int64 DTensor arange(8)+3e9, Shard(0): every element of every local shard and of full_tensor() = -1294967296
```
`reduce_scatter_tensor` on int32/int64 has the same limit; `all_to_all_single` is exact (`int_more.py`), so MoE
routing indices are not affected. Because `broadcast` is a sum, a float broadcast is not bit exact: `-0.0` arrives as
`+0.0` (fp32 and fp16); denormals are kept.
MAX and AVG on int32 are also fp32 (`probe_dist5d.py int_reduce_gather_avg`: MAX of 30000000+r over 4 ranks gives
30000004, expected 30000003). Compiled functional collectives give the same values as eager (`compiled_funcol_int`).

(c) float64: a device tensor keeps float64 exactly, but collectives run in fp32 and cast back
(`probe_dist5d.py fp64_collectives`, all_reduce SUM over 4 ranks):
```
input per rank: [1 + 1e-12*(r+1), 1e300, 123456789.123456789, 1e-310]
got:      ['4.0', 'inf', '493827168.0', '0.0']
expected: ['4.000000000010001', '4e+300', '493827156.49382716', '4e-310']
broadcast exact: False, all_gather exact: False
```
float64 is used for metric accumulators (for example sums of losses or counts over an epoch). `DowncastTensorList`
casts float64 to float32 ("Other dtypes follow IsSupportedDtype as usual (e.g. float64 -> float32)").
For data-movement collectives the same int32-view fix applies (view float64 as int32 pairs).

(b) int64 values outside the int32 range are cast to int32 before any collective, so they saturate (sum) or lose the
high 32 bits (all_gather), described below.

Fix for (a), tested in process (`overnight/dist5/int_fix_proto.py`): implement integer `broadcast`/`scatter` as data
movement (all_gather of the int32 view, keep the src slice): exact for int32 and int64 at 2^24, 2e9, 3e9 and -2^40.
For integer SUM, reduce 16-bit limbs (each limb sum < W * 2^16 < 2^24) and recombine: exact for int32 at 2^24 and 2e9.
The int64 version of the limb sum must work on the int32 words, because device int64 shifts are limited (packed int64).

Typical uses that break: token counts summed over ranks for loss normalization (more than 16.7M tokens per step on
large batches), global sample or token counters, seeds broadcast from rank 0 (`torch.randint(0, 2**31)`), integer
DTensors (sharded token buffers, routing indices), checkpoint step counters.


Details for (b). On one device an int64 tensor keeps values outside the int32 range (`torch.tensor([3e9, 2**40])` round-trips, `+ 1`
and `clone` are right). A collective on the same tensor changes the values with no error. This includes the
data-movement collectives, which do no arithmetic. (`overnight/dist5/probe_dist5.py int64_and_precision`, 4 ranks)
```
all_reduce SUM [3e9+r, -(3e9+r), 2**40+r, 5]: got [-2147483648, 2147483647, 6, 20]
                                              expected [12000000006, -12000000006, 4398046511110, 20]
all_reduce MAX 3e9+r:            got -1294967296      expected 3000000003
all_gather_into_tensor 3e9+r:    got [-1294967296, -1294967295, -1294967294, -1294967293]
broadcast 3000000007:            got -1294967296
```
Typical uses that break: global token or sample counters (`tokens_seen` passes 2^31 early in LLM training), seeds and
RNG offsets broadcast from rank 0, int64 ids gathered for evaluation, and step counters in checkpoints.

Root cause: `csrc/core/dispatch/DispatchUtils.cpp` `DowncastTensorList` (called by `NeuronBackend::SubmitInNeffCollective`)
always casts int64 to int32 before a collective ("collectives are handled in int32"), and casts the output back.

Workaround and fix (tested, `overnight/dist5/int64_fix.py`): for data-movement collectives (all_gather, broadcast,
all_to_all, scatter, gather, send/recv), reinterpret the int64 data as int32 pairs (`t.view(torch.int32)`) instead of a
value cast. With this, all_gather and broadcast of 3e9 and -(2^40) are exact. For SUM/MAX/MIN, check the range and
raise (or fall back) when a value does not fit in int32, at least under `TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW=1`.

Known? The packed int64 design (`docs/packed_int64_design.md`) documents truncation for compute ops. It does not say
that a collective truncates values that the device tensor holds correctly.

### 22. DTensor `cross_entropy` (TP `loss_parallel`) ignores `label_smoothing` (silent wrong loss)

With vocab-sharded logits under `loss_parallel()`, `F.cross_entropy(..., label_smoothing=0.1)` returns the loss without
smoothing. Upstream PyTorch raises for this case (CPU gloo: `aten.masked_fill_.Scalar got mixed torch.Tensor and
DTensor`), so on CUDA the user gets an error and on Neuron a wrong loss. (`overnight/dist5/lp_ls.py`)
```
cpu ls=0.1 loss 4.62334, cpu ls=0 loss 4.61568, neuron loss_parallel ls=0.1 4.61568
```
Root cause: `python_ops/torch_mlir/ops/loss.py`, the DTensor branch of the cross_entropy override, computes
`log_softmax` + `nll_loss` and never reads `args[5]`. The docstring says "this branch does not forward
`label_smoothing`", but nothing tells the user at run time.

Possible fix: raise `NotImplementedError` in that branch when `label_smoothing != 0` (same result as upstream). A real
implementation cannot add a second consumer of `log_softmax`: the upstream `loss_parallel` log_softmax backward
assumes its only consumer is `nll_loss` (a prototype `(1-e)*nll + e*smooth` gives the right loss but 2.4% wrong grads).

Known? Only the docstring note.

### 23. The process-group `timeout` is detected but not enforced: a stalled rank makes the others hang, not fail

`init_process_group(..., timeout=timedelta(seconds=30))`, rank 3 sleeps 90 s before `all_reduce`. The watchdog on the
other ranks logs the timeout at 30 s and "Aborting work due to timeout", but their `all_reduce` + sync keeps waiting,
then returns the correct value at 90 s with no exception. With CPU gloo the same script raises "Timed out waiting
30000ms" at 30 s. (`overnight/dist5/probe_dist5.py timeout_one_rank_stalls`, `cpu_timeout_one_rank_stalls.log`)
```
[rank0] NeuronWork.cpp:159 Watchdog caught collective operation timeout: allreduce (seq=0) ran for 30026 milliseconds
[rank0] NeuronWork.cpp:274 Aborting work due to timeout
R0 timeout_one_rank_stalls: {'result': [4.0, 4.0, 4.0, 4.0], 'secs': 90.0}
```
So a rank that stays alive but skips a collective (a common deadlock, for example a data-dependent branch) hangs the job
forever instead of failing after `timeout`. A rank that exits is detected at once (`rank_exit.py`: NRT error in 1 s).

Root cause: `csrc/distributed/NeuronWatchdog.cpp` marks the work as timed out and `NeuronWork::abort` runs, but the
caller blocks in the device stream sync, which the abort does not cancel. `NeuronWork::wait()` only polls
`checkTimeout` when `blockingWait_` is set or a wait timeout is passed.

Related API problem (`overnight/dist5/wait_timeout.py`): `dist.all_reduce(..., async_op=True)` returns a
`torch_neuronx._C.NeuronWork`, whose binding is `wait(timeout_ms: int = -1)` (`csrc/core/NeuronBindings.cpp` near line
2402). The c10d signature is `wait(timeout: datetime.timedelta)`, so the standard call fails:
```
w.wait(timeout=timedelta(seconds=10))  -> TypeError: wait(): incompatible function arguments
w.wait(timedelta(seconds=10))          -> TypeError (same)
w.wait(10000)                          -> RuntimeError after 10.0 s: Watchdog caught collective operation timeout
```
So the only way to get a timeout error today is the non-standard `wait(<int ms>)`. Accept a `timedelta` as well.
Other `Work` methods on `NeuronWork` (`work_api.py`): `result()`, `get_future()`, `exception()`, `boxed()`,
`synchronize()` work; `get_future_result()` raises "Work::getFutureResult not implemented", `block_current_stream()`
raises "Failed to create StreamBlock", `is_success()` raises NotImplementedError (deprecated upstream too).

Possible fix: when the watchdog times out a collective, abort the communicator and fail the stream (the NCCL backend
aborts the process group and raises, `TORCH_NCCL_ASYNC_ERROR_HANDLING`).

Known? No.

### 24. Object collectives fail when the object holds a Neuron tensor (low)

`dist.all_gather_object(objs, {"r": rank, "t": torch.full((2,), float(rank), device=dev)})` raises on every rank, and
`broadcast_object_list` with a Neuron tensor raises the same way. Objects without device tensors work. On CUDA this
works: each process sees all GPUs, so the unpickled tensor goes back to `cuda:<sender>`. Typical use: gathering eval
outputs or metric dicts that hold device tensors (`accelerate.utils.gather_object`, Lightning metric sync).
(`overnight/dist5/probe_dist5.py mixed_cpu_and_objects`)
```
all_gather_object_neuron_tensor:     ERR ValueError: Device index 1 is out of range. Valid range is 0 to 0 for this process
broadcast_object_list_neuron_tensor: ERR ValueError: Device index 2 is out of range. Valid range is 0 to 0 for this process
```
Root cause: each process owns one logical core (`csrc/core/NeuronDevice.cpp`, range check in the device guard), and
c10d `_tensor_to_object` unpickles the tensor onto the sender's device index.
Possible fix: when a tensor is restored to a Neuron index that this process does not own, map it to the current device
(or to CPU), or document that object collectives must hold CPU tensors.
Known? `docs/torch_api_compatibility.md` lists these as supported (row 73-75) with no note on device tensors.

### 25. A DDP comm hook that makes a tensor and uses it in the future callback fails; PowerSGD with error feedback fails

`DDP.register_comm_hook(state, powerSGD_hook.powerSGD_hook)` with `use_error_feedback=True` (the default) fails at the
first PowerSGD step, every run. `use_error_feedback=False` works, and the same script on CPU gloo works.
With `NEURON_LAUNCH_BLOCKING=1` it also works, so it is an ordering problem in async execution.
(`overnight/dist5/powersgd.py <rank> <error_feedback> <warm_start>`)
```
rank,ef,warm = 4 1 1 / 4 1 0 / 1 1 0: RES FAIL NRT Execution error occurred on Neuron for operation=aten::sub.Tensor
  error message="NRT EXECUTION FAILED: ./torch_neuronx/csrc/core/KernelExecution.h:290: data_ptr FAILED.
  Tensor not bound: lazy alloc kAlloc may not have executed"
rank,ef,warm = 4 0 1 / 4 0 0 / 1 0 0: 4 steps ok
```
Minimal repro (`overnight/dist5/hook_min.py`, 4 ranks): a comm hook that does `cp = bucket.buffer().detach().clone()`
(or `torch.zeros(...)`), starts `dist.all_reduce(buf, async_op=True)`, and computes `cp - t` in `fut.then(...)`:
```
RES noclone: ok
RES clone: ERR ... Tensor not bound: lazy alloc kAlloc may not have executed
RES zeros_first: ERR ... Tensor not bound: lazy alloc kAlloc may not have executed
```
The same pattern on the main thread (no DDP) works (`fut_then.py`, `fut_chain.py`), and a tensor made in an autograd
grad hook and used later on the main thread works (`thread_alloc.py`).

Probable root cause: DDP runs comm hooks on the autograd engine thread. The clone there gets a lazy allocation
(`kAlloc`) that is not bound yet when the future callback, which runs on the thread that completes the `NeuronWork`
future (`csrc/distributed/NeuronWork.cpp` `markCompleted`), dispatches `aten::sub` with it. `ExtractDataPtrs`
(`csrc/core/KernelExecution.h:290`) then finds no NRT tensor. Possible fix: order the pending allocations of a tensor
before any op that reads it from another thread or stream (for example, flush or wait for pending `kAlloc` ops of the
inputs in the callback path, or bind the allocation eagerly when it is made off the main thread).

Known? No.

### 26. In one process, a Neuron-to-Neuron `.to()` across cores keeps the source core (silent wrong device)

A single process sees all 4 logical cores (`torch.neuron.device_count() == 4`), and CPU -> `neuron:1` and factories on
`neuron:1` work. But a copy from one core to another returns a tensor on the source core, with no error. A dtype change
in the same call does not help. `copy_` into a tensor on the other core raises. A CPU bounce works.
(`overnight/dist5`, inline script; output)
```
cpu -> neuron:1             neuron:1
neuron:0 .to('neuron:1')    neuron:0        <- expected neuron:1
neuron:0 .to(device(neuron,1)) / .to('neuron:1', copy=True) / .to('neuron:1', bfloat16): neuron:0
neuron:1 .to('neuron:0')    neuron:1
neuron:1 .to('neuron:2')    neuron:1
empty(4, device='neuron:1').copy_(t0): RuntimeError: Inconsistent device IDs found in kernel outputs: expected device 0 but found device 1
neuron:1 .cpu().to('neuron:2')  neuron:2
2-layer model split over neuron:0 / neuron:1: RuntimeError: Inconsistent device IDs found in kernel inputs
```
Impact: single-process model parallel across cores does not work, for example a naive pipeline split, HF
`device_map="auto"` / accelerate big-model loading over several cores, or moving a tensor to another core for a
side job. Code that checks `t.device` after `.to()` sees the wrong core and fails later.

Root cause: `csrc/core/dispatch/NeuronDispatcher.cpp` `DispatchToCopyImpl`, Neuron -> Neuron branch on the default
stream. With the same dtype, Phase 1 does `converted = self_tensor` ("No dtype change (reached only with a real layout
change)"), and the target device index is not used after that. The comment is wrong for a change of core. Only the
host-CC non-default-stream branch and the CPU bounce use `target_device`. The Python `ToCopyNeuronImpl` has a correct
Neuron -> Neuron branch (`write_neuron_to_neuron` into a tensor on `target_device`), but it is not reached.

Possible fix: when `self.device().index() != target_device.index()`, allocate the output on `target_device` and copy
(`copy_neuron_to_neuron`, or the CPU bounce if a direct core-to-core copy is not possible), before the dtype/layout
phases. Make `copy_` across cores take the same path.

Known? No. `docs/AI-distributed.md` covers multi-process training only.

### 27. Collectives do not overlap with compute (performance)

The usual way to hide communication (an async collective, or a collective on a side stream, while independent compute
runs; FSDP prefetch and DDP bucketing rely on this) gives no gain: the time is the sum of compute and communication.
In a compiled graph with an independent matmul chain and an all_reduce the gain is 0.8 ms of 8 ms.
(`overnight/dist5/overlap.py`, `overlap_compile.py`, 4 ranks, bf16 4096x4096 matmul x20, fp32 512 MiB all_reduce)
```
eager:    matmul x20 23.2 ms, all_reduce 7.8 ms, async_op overlap 31.0 ms, side-stream overlap 31.6 ms, serial sum 31.0 ms
compiled: matmul chain 23.5 ms, all_reduce 8.0 ms, both in one graph 30.7 ms, serial sum 31.5 ms
```
Impact: in data-parallel and FSDP training every gradient reduce-scatter / all-reduce and every parameter all-gather
adds its full time to the step. At the measured bus bandwidth (about 104 GB/s on this instance) that is about 15 ms per
GB of fp32 gradients per step for DDP.

Probable cause: eager collectives run as their own NEFF (`NeuronBackend::SubmitInNeffCollective`) in the same execution
queue as the compute NEFFs of the core, so a Neuron stream does not give concurrent execution. This may be a runtime or
hardware limit, but `docs/` does not say that compute and collectives serialize.

Known? Not documented.

### 28. Uneven `all_to_all_single` recompiles for each new row capacity: 1-3 s per call in MoE training (performance)

MoE expert parallel sends a different number of tokens to each expert every step. On Neuron each call with a new
combination of total rows compiles a new kernel, which takes 1.1-3.0 s. When the shapes repeat, the call takes 1.5 ms
(an even all_to_all of the same size takes 0.5 ms). (`overnight/dist5/a2a_latency.py`, `a2a_latency2.py`, 4 ranks,
hidden 512, 1-64 rows per peer)
```
new random splits each step (ms): [2992.8, 2891.9, 2651.8, 2705.1, 2710.2, 2693.0, 2703.1, 1432.6, 1115.0, 2700.0, ...]
same splits x5 (ms):               [1511.6, 2.0, 1.8, 1.6, 1.6]
different splits, same send and recv totals (ms): [1.6, 1.5, 1.6, 1.6]
even all_to_all, warm: 0.52 ms
```
With top-k routing, the number of tokens that a rank receives changes every step, so a MoE run pays this compile on
most steps until every capacity value has been seen. Each MoE layer has two all_to_all calls in forward and two in
backward. The persistent cache helps only for capacities seen in an earlier run.

Root cause: `torch_neuronx/distributed/ops/all_to_all_v.py` (documented in `docs/AI-all-to-all.md`, "NKI a2a-v buffers
and metadata") takes the MAX of send and receive rows over ranks and rounds it up to a multiple of EP (4 here). That
value sets the kernel shapes, so there is one compile for every 4 rows of capacity.

Possible fix: round the capacities up to a coarse bucket (for example the next power of two, or a configurable
multiple), so that the number of kernels is about log2(world_size * tokens). The padding cost is at most 2x of the
buffer. Users can avoid it today with a fixed expert capacity (pad/drop tokens to a constant count).

Known? The capacity rule is documented; its compile cost per new capacity is not.

### 29. `init_process_group(device_id=...)` breaks 2-D device meshes; HF `from_pretrained(..., tp_plan="auto")` fails

PyTorch recommends `init_process_group(backend, device_id=torch.device(...))` (eager communicator init); torchtitan and
HF transformers pass it. On Neuron, after that call every multi-dimensional `init_device_mesh` fails. A 1-D mesh and
`new_group` still work. (`overnight/dist5/device_id_mesh.py`, 4 ranks)
```
init_process_group(plain):     {'(4,)': 'ok [4.0, 4.0]', '(2, 2)': 'ok [2.0, 2.0]', 'new_group': 'ok'}
init_process_group(device_id): {'(4,)': 'ok [4.0, 4.0]',
                                '(2, 2)': 'ERR RuntimeError: No backend for the parent process group or its backend does not support splitting',
                                'new_group': 'ok'}
```
Real use (`overnight/dist5/hf_tp.py`, transformers 5.22.1, tiny Llama saved with `save_pretrained`, torchrun 4 ranks):
`LlamaForCausalLM.from_pretrained(path, tp_plan="auto")` raises the same error. transformers maps the `neuron` device to the
`neuron` backend and calls `init_process_group(..., device_id=...)`, then builds its mesh.

Root cause: `torch/distributed/device_mesh.py` (`_init_one_process_group` path) uses `split_group` when the default group
has a `bound_device_id` and an accelerator is present. It does not check that the backend supports splitting.
`NeuronBackend` does not implement `split` (`supportsSplitting` is false), so `split_group` raises.

Possible fix: implement `Backend::split` in `NeuronBackend` (a new backend over the subset of ranks, as `new_group` does),
or make the torch_neuronx DeviceMesh hook (`torch_neuronx/distributed/mesh_registry.py`, which already patches
`DeviceMesh._init_process_groups`) use `new_group` for Neuron when the backend cannot split. Workaround: do not pass
`device_id`.
With the process group created first without `device_id`, `from_pretrained(..., tp_plan="auto")` gets past the mesh
and then needs `accelerate`, which is not installed in this venv, so the HF TP forward was not tested further.

Known? No.

## Performance: documented CPU fallbacks that cost the most in common training (eager)

`overnight/perf_fallback.py`, ms per SGD train step after warmup, batch 32x3x64x64 CNN (3 conv blocks) and a 512-1024-512 MLP:
```
cnn AvgPool2d + ReLU          neuron  17.7   cpu 19.8
cnn MaxPool2d + ReLU          neuron  53.7   cpu 21.0    <- 3.0x slower than AvgPool, slower than CPU
cnn AvgPool2d + LeakyReLU     neuron  36.1   cpu 22.1    <- 2.0x (only the backward falls back)
mlp sigmoid + BCELoss         neuron   2.9   vs BCEWithLogits 1.2
mlp smooth_l1                 neuron   3.3   vs mse 0.8
```
Fallback ops in this run: `max_pool2d_with_indices(.out/_backward)`, `leaky_relu_backward.grad_input`,
`binary_cross_entropy(_backward)`, `smooth_l1_loss(.out/_backward)`. All are listed in
`docs/torch_api_compatibility.md` (so not new), but the cost is large for CNN and GAN training.
Cheap wins: `leaky_relu_backward` is `where(self > 0, grad, grad * slope)` and the forward is already lowered in
`python_ops/torch_mlir/ops/activation.py`; `max_pool2d_with_indices_backward` already has a compile decomposition
(`neuron_dynamo_backend/decompositions.py:5212`) that the eager path could reuse.

Compiled optimizer step with a Python-float LR and a scheduler recompiles every step (`overnight/compile_perf.py`):
```
float_lr_sched   step times (s): [4.15, 2.22, 2.21, 2.22, 2.21, 2.28]   unique_graphs: 6
tensor_lr_sched  step times (s): [3.32, 0.0, 0.0, 0.0, 0.0, 0.0]        unique_graphs: 1
```
This is upstream Dynamo behavior (the float LR is a guarded constant), but on Neuron each recompile is a 2.2 s
neuronx-cc run, and after the recompile limit Dynamo falls back to eager. `README.md` / `examples/compile` could say
"pass `lr=torch.tensor(...)` when you compile `opt.step`". GPT-2 (4 layers, 512 wide, seq 256) compiled train step:
first step 24.6 s, then 0.02 s per step (eager 0.03 s).

## Checked and correct

Eager round 3 (`overnight/probe_eager3.py`, 44 cases OK): optimizer state_dict save/load neuron->bytes->neuron and
CPU->neuron for AdamW, SGD momentum, Adam amsgrad, RMSprop, Adagrad, Adam capturable, AdamW fused; step tensor stays on
CPU after load; add_param_group midway; GradScaler inf/NaN skip + backoff + growth, unscale_ + clip, state_dict
roundtrip, `_amp_foreach_non_finite_check_and_unscale_` / `_amp_update_scale_` direct; StepLR, CosineWarmRestarts,
OneCycle, SequentialLR, LambdaLR, ReduceLROnPlateau with a neuron tensor metric, CyclicLR, PolynomialLR;
`load_state_dict(assign=True)` from meta; `.data` reassignment; bf16 module train; model .cpu()/.to() mid-training;
deepcopy model + optimizer; torch.save of views; rrelu_with_noise (noise filled); native_dropout mask; BN eval with
loaded buffers; embedding max_norm renorm in place; index_copy_/index_fill_/masked_fill_ backward; put/take backward;
writes through overlapping unfold views; expand in-place error; grad accumulation into a non-contiguous `.grad`.
Not reported: meta init + `to_empty` + `reset_parameters` gives different values than CPU (init uses the device RNG,
as on CUDA); `BatchNorm(momentum=None)` running stats (same as known BatchNorm issue).
SDPA/dropout consistency (`overnight/sdpa_drop.py`): compiled SDPA with dropout is correct; dropout1d/2d/3d,
alpha_dropout and feature_alpha_dropout forward/backward masks match in eager.


Compile round 2 (`overnight/probe_compile2.py`, CPU eager reference): whole compiled train step for SGD
(plain/momentum/dampening/weight decay), Adam, AdamW amsgrad, RMSprop centered, Adagrad, Adamax, NAdam
decoupled wd, RAdam, Adadelta, ASGD, Rprop, Adafactor; compiled `opt.step` with tensor lr + CosineAnnealingLR;
mode max-autotune / reduce-overhead / max-autotune-no-cudagraphs; graph breaks inside the step (`.item()`,
data-dependent `if`, `print`); compiled dropout fwd/bwd mask consistency; buffer mutation in forward (counter,
EMA, index_copy_ into a cache, eager and compiled); input mutation and view-input mutation; fp16 autocast +
GradScaler; bf16 autocast inside the compiled step; grad accumulation; frozen layers + param groups; EMA with
`_foreach_lerp_`; CNN + BatchNorm train/eval; SDPA with bool mask / causal+scale backward; vmap(grad) per-sample
grads; embedding/index_add/index_put backward with duplicates; CE with ignore_index and label smoothing;
all-ignored CE. Expected upstream behavior, not reported: `fullgraph=True` refuses `Optimizer.zero_grad`.


Examples: `gpt2-train-loop` (3 iters), `llama`, `simple_moe` run to the end with no error (logs in
`overnight/examples/`). `simple_moe` only runs a forward pass, and its MoE layer does not apply routing
weights (stated in its docstring), so the router gets no gradient. Model-level CPU vs Neuron, 4 AdamW steps +
clip_grad_norm (`overnight/train_models.py`): example Llama, example MoE, HF GPT-2, HF GPT-2 with gradient
checkpointing (dropout 0), HF Llama (GQA): losses match to 1e-5, step-0 grads match, final params within 1.2e-4.


Eager round 2 (`overnight/probe_eager2.py`, 130 cases, CPU fp32 reference): GroupNorm, InstanceNorm (no
running stats), RMSNorm, LayerNorm variants, LocalResponseNorm, weight_norm/spectral_norm parametrizations,
EmbeddingBag sum/mean/max/per_sample_weights (CPU fallback, as documented), Embedding padding_idx /
scale_grad_by_freq / max_norm, index_put/index_add/scatter_add/scatter_reduce (all reduces)/gather/
index_select/masked_scatter/take_along_dim backward with duplicate indices, ctc_loss (fallback), 20 more
loss functions, conv1d/2d/3d (groups, dilation, reflect/circular padding), ConvTranspose 1d/2d/3d groups=1,
max/avg/adaptive/LP pooling fwd+bwd (ties, ceil_mode, count_include_pad, divisor_override), max_unpool,
all upsample modes incl. antialias, fold/unfold, pads, grid_sample, affine_grid, SDPA with float mask / GQA /
scale / non-contiguous / q_len != kv_len causal / fully masked row, SDPA dropout statistics, MHA with key
padding mask, TransformerEncoderLayer, LSTM packed / proj_size, in-place ops on views with autograd, out=
into views, copy into non-contiguous views, multi-output out= ops (max, sort, topk, cummax, frexp),
checkpoint_sequential (no dropout), torch.func grad+vmap per-sample grads, jvp, hessian, vmap ops,
autograd.functional jacobian/hessian, custom autograd.Function, save_on_cpu, non-contiguous params,
manual grad accumulation, hooks, detect_anomaly, autocast bf16/fp16 dtype policy (matches CUDA policy,
not CPU), clip_grad_norm error_if_nonfinite / mixed dtype, parameters_to_vector, SWA/EMA AveragedModel +
SWALR, pruning, cumsum/cumprod/logcumsumexp/cummax, sort/topk/kthvalue/median backward, norms/cdist,
30 activations, matmul/einsum/baddbmm/addmm backward, linalg inv/det/slogdet/solve/cholesky/qr/svd/eigh
backward (mostly CPU fallback).
Not reported: nn.GRU / GRUCell (documented as unsupported), `nn.Embedding(sparse=True)` (no sparse
backend), InstanceNorm `track_running_stats=True` and `F.batch_norm` running stats (same root cause as
the known BatchNorm issue).

Distributed round 2 (`overnight/probe_dist2.py`, torchrun 4 ranks, CPU reference): all_reduce SUM/MAX/MIN/AVG for
fp32/bf16/int32/int64; functional collectives (all_reduce, all_gather dim 1, reduce_scatter dim 1, all_to_all_single)
and their autograd versions, `torch.distributed.nn` all_reduce/all_gather backward, compiled funcol; new_group,
strided groups, new_subgroups, 2x2 DeviceMesh dp/tp groups, interleaved groups; TP Colwise/Rowwise MLP (foreach on/off,
clip_grad_norm); SequenceParallel LayerNorm + Colwise/Rowwise with Shard(1); TP+FSDP2 2x2 eager and compiled
(`dynamic=False`); FSDP2 + AC without dropout; FSDP2 with dims not divisible by 4; DCP save on 4 ranks, load on 2 ranks
(reshard) and into an unsharded model; DDP Join with uneven inputs; DDP fp16 comm hook; FSDP2 no-sync grad accumulation +
`reshard_after_forward=False`; FSDP2 bf16 params fp32 reduce; full_state_dict cpu_offload.
Distributed round 3 (`overnight/probe_dist3.py`, 2-layer Llama-style LM with RMSNorm, SwiGLU, causal SDPA, 3 AdamW
steps vs CPU): TP with the Llama plan (Rowwise embedding, Colwise q/k/v/w1/w3, Rowwise wo/w2, Colwise head) and with
`loss_parallel` (vocab-sharded CE): losses match CPU to 5 decimals, params within 4.5e-5; FSDP2 per-block eager and
compiled; FSDP2 with `CPUOffloadPolicy` (params stay on CPU, losses match); DTensor matmul shard x replicate and
shard1 x shard0 (Partial), redistribute S0->S1, S1->R, R->S0, softmax/sum/mean, uneven shards, DTensor grad. DDP + SDPA
dropout: compiled consistent, eager wrong (finding 1 again).
Not reported: AdamW foreach with a mix of DTensor and plain Tensor params (`aten._foreach_mul_.Scalar got mixed
torch.Tensor and DTensor`) also fails on CPU/gloo with `foreach=True` (`overnight/tp_cpu_foreach.py`), so it is upstream
behavior; it shows on Neuron by default because Neuron defaults to foreach. `full_tensor()` on CPU-offloaded FSDP2
params needs a CPU backend (`cpu:gloo,neuron:neuron`), as on CUDA.
Also correct (one torchrun per case, `overnight/coll_each.py`): all_gather_into_tensor, reduce_scatter list, all_to_all_single
even/uneven splits, all_to_all list, gather, scatter, reduce, all_gather_object, broadcast_object_list, async all_reduce,
32 MB all_reduce, all_gather list.
Not reported: `all_reduce(op=PRODUCT)` fails to compile (`NCC_IBIR067 AllReduce CollectiveCompute does not support
mult`; low impact); DDP `bf16_compress_hook` refuses non-CUDA (upstream check); reentrant AC with FSDP2 gives no
param grads when the input does not require grad (upstream behavior).

Long run (`overnight/long_train.py`, example Llama 2 layers, AdamW + OneCycle + clip, 300 steps on a learnable
pattern): fp32 loss curve matches CPU to 3 decimals at every logged step (final 1.3205 vs 1.3208, max relative param
diff 0.5%). bf16 autocast curves are close (final 1.497 CPU vs 1.434 Neuron); not comparable bit for bit because the
Neuron autocast policy follows CUDA (fp32 softmax/layer_norm) and the CPU policy does not.

OpInfo sweep (`overnight/vjp_sweep.py`, 412 autograd ops with fp32 CPU support, first 3 samples each, forward and VJP
vs CPU, atol/rtol 1e-3; linalg/fft/random/sparse ops skipped): 400 ops OK. The 12 others: conv1d/conv2d/
conv_transpose1d/2d/3d (stride + dilation, finding above), cross_entropy (finding 13), scaled_dot_product_attention and
multi_head_attention_forward (samples use dropout, so random), fractional_max_pool3d (random samples), stft (fft
fallback disabled), tensor_split (tensor indices on device, same on CUDA), as_strided.partial_views (exotic views).

bf16 OpInfo sweep (same script, `bfloat16`, neuron bf16 vs CPU fp32 on the same rounded inputs, tol 5e-2): 382 OK,
19 flagged. New: finding 14 (adaptive avg pool) and the grid_sample grad note. The rest are the same as fp32
(conv stride+dilation, cross_entropy target grad, dropout samples, tensor_split), or equal to CPU bf16 error
(addbmm, kron, logcumsumexp all match CPU bf16 exactly; topk ties after bf16 rounding).

Compiled OpInfo sweep (same script with `torch.compile(backend="neuron", dynamic=False)` per sample, fp32): 300 OK,
106 flagged. 71 of the 106 already have a `test_compile_*_float32` entry in the skip/xfail registries (for example
scatter_reduce prod/amin/amax, avg_pool3d, polygamma, renorm, masked.log_softmax, tan, digamma, the `*_copy` view ops,
pad reflect/replicate), so they are known. Of the other 35, almost all fail only on the OpInfo sample with a zero-size
tensor (`(1, 0, 3)`, `(0, 3)`): `NCC_ISPP060 Unsupported use of a zero-sized tensor` (exp/log/cos/sigmoid/linear/
softsign/round/...). A later sample in the same process then also fails (async error), so those are not separate
bugs. 0-d tensors compile fine (`overnight/zero_size.py`), eager zero-size works, and a compiled MoE expert forward
on a `(0, d)` batch works (`overnight/zero_eager.py`), so impact on training is low; not listed as a finding.

Second session: memory over 60 train steps (`overnight/mem_mixed.py mem`, 2-layer TransformerEncoder + AdamW): eager,
eager + checkpoint, compiled, eager with SDPA dropout, eager + GradScaler all stay at 4.29 MiB allocated, so no leak.
Mixed CPU/Neuron patterns (`mem_mixed.py mixed`): 0-d CPU tensors with Neuron tensors, Neuron tensor indexed by a CPU
index or bool mask, `masked_fill` with a 0-d CPU bool, copy_ both ways, a CPU tensor LR in AdamW all work; non-scalar CPU
tensors in Neuron ops raise "input tensor is on cpu device, expected neuron" (same idea as CUDA).

Eager round 4 (`overnight/probe_eager4.py`, rel err vs CPU fp32): cross_entropy V=128256 N=1024 fp32 (4.7e-8) and bf16
(4.2e-4), with label smoothing 0.1 + ignore_index bf16 (3.8e-3), V=50257 N=2048 ignore_index fp32, V=32000 N=4096 bf16;
embedding 128000x1024 backward with heavy duplicate indices fp32/bf16 (untouched rows stay 0); softmax over 32768-wide
rows; LayerNorm/RMSNorm hidden 8192; matmul K=16384; large bf16 sum/mean/var; WGAN-GP gradient penalty through Conv2d +
GELU + LayerNorm (`create_graph=True`); Hessian-vector product on a Tanh/SiLU MLP; ResNet-style blocks with inplace ReLU +
BatchNorm2d over 3 SGD steps (4e-5). Not reported: SDPA double backward raises "derivative for
`_scaled_dot_product_fused_attention_overrideable_backward` is not implemented", the same limit as the CUDA flash and
efficient kernels.

Compile round 3 (`overnight/probe_compile3.py`): compiled CE V=128256 N=1024 with ignore_index (grad 5.9e-8, 1.1 s);
compiled LM head + CE under bf16 autocast; train/eval toggling a compiled BatchNorm model (2 graphs total, eval output
and running stats match); compiled fused AdamW step with tensor LR; compiled checkpoint + GQA causal SDPA under bf16
autocast (all grads at bf16 level; `k.bias` grad is 0 in exact math, both sides are noise); compiled Embedding with
`padding_idx` + `scale_grad_by_freq` (pad row stays 0); compiled LayerNorm/RMSNorm hidden 8192 bf16.

Views, RNG, int64 (`overnight/probe_views.py`): 166 of 168 view cases OK: about 20 in-place / `out=` ops (add_,
copy_, fill_, clamp_, exp_, addcmul_, masked_fill_, setitem, foreach add/mul/lerp, add/mul/sigmoid out=, index_fill_,
scatter_, mm out=, addmm_) writing into column-offset, row-offset, transposed, strided, diagonal and inner-block views,
and 8 ops reading those views with autograd. The 2 failures are lowering errors for `lerp_` and `index_fill_` into a
transposed offset view (`b.t()[1:4]`), low impact. RNG: randn/rand/randint/randperm/normal_/uniform_/bernoulli/
multinomial/dropout/kaiming/trunc_normal/orthogonal/exponential_ reseed and statistics are fine once the runtime is up
(see finding 15). int64 values outside int32 are truncated or saturated without error (`sum`/`cumsum` of 10 x 2^30 gives
2147483647, `<< 34` gives 0, indexing `arange + 2^32` loses the high word, `3**20` saturates); this is the documented
packed-int64 behavior (`docs/packed_int64_design.md`, opt-in `TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW=1`), so not
listed as a finding, but `torch_api_compatibility.md` does not mention it for sum/cumsum.

SDPA sweep (`overnight/sdpa_sweep.py`, eager and compiled, 140 configs each: head dim 64/72/128/256, L 128/100 and
Lq=64/Lk=200, no mask / bool mask / broadcast bool mask / learnable float mask (its grad checked), causal, MHA and GQA 8/2,
custom scale, fp32 and bf16): all match CPU except finding 16.

Copies and MoE kernels (`overnight/nonblock.py`, `overnight/gmm.py`): `.to("neuron", non_blocking=True)` from pageable
memory is safe if the source changes right after; from pinned memory the later change shows up on device, and a
non_blocking D2H read before a sync is wrong, both the same contract as CUDA. DataLoader `pin_memory=True` works.
`torch._grouped_mm` fp32 fwd/bwd rel ~3e-7, bf16 ~1.6e-3 (no fp32 downcast, unlike finding 16).

Forward-only OpInfo sweep (`overnight/fwd_sweep.py`, first 3 samples, exact match for ints/bools): int64 364 OK / 5
flagged, bool 299 OK / 3, fp32 non-autograd ops 93 OK / 0. Flagged items, all low impact: (a) bf16 `rsqrt` returns the
bf16 max 3.39e38 instead of inf for 0 and instead of the right value for inputs below ~1e-29 (`rsqrt(1e-30)` should be
1.0e15); fp32 and fp16 are right, normal RMSNorm eps values are not affected; (b) int64 `addmv` with fractional
`beta=0.2, alpha=0.6` uses the float scalars, CPU truncates them to 0 first; (c) `__rpow__` int64 overflow (packed int64,
documented); the rest are lowering errors on edge samples.

## Notes

Round 5 (third session, `overnight/r5/`): no new finding.
- Eager lowering cache key (`cache_key.py`, `cache_key_bwd.py`): 36 forward and 30 backward ops called several times in
  one process with the same shapes but other scalar args (alpha, slope, eps, dim, p, beta, delta, approximate,
  rounding_mode, label_smoothing), other strides or other dtype: all match CPU, so the per-shape cache keys these.
- Compiled graph reuse (`compile_stale.py`): Python constant change, module attribute change, layer frozen mid-training,
  `.data` swap and in-place param copy, buffer reassign, dtype and stride switch: all recompile or reuse correctly.
  The persistent NEFF key covers HLO bytes, NEURON_CC_FLAGS, opt level, LNC, platform and compiler version
  (`csrc/core/cache/CompilationCacheKey.cpp`).
- Tensors with more than 2^31 elements (`probe_r5.py big_*`): 2^31+4096 bf16 vector writes/reads at both ends and at
  2^31-1; Embedding 140000x16384 bf16 (2.29e9 elements) forward rows and backward with duplicate rows are right.
- Optimizer step in `register_post_accumulate_grad_hook`, MoE top-2 routing (argsort, bincount, index_add, aux loss)
  forward and backward, bincount/unique/bucketize/searchsorted/histc/nonzero/masked_select/one_hot, tied
  embedding/LM head, full backward hooks + grad hooks + zero-stride grad_output + retain_graph, write-after-read
  ordering (H2D copy, fill_, `.data` swap, view write right after a pending read), Neuron Stream/Event/wait_stream.
- Not reported: tied embedding fp32 loss near 0 has 8e-4 relative error (1e-6 absolute, CPU 1.5e-7), which is fp32
  cancellation in `logsumexp - z_y` at logit ~59 (`tied3.py`); a side-stream read with no `wait_stream` gave a wrong
  value once in 4 runs, which is a race by the CUDA stream contract.
- Error recovery (`r5/recover.py`), training is correct after: OOM from one huge allocation, OOM during a large forward,
  an exception in a gradient hook (retry gives the right grad), a Python exception inside a compiled function (retry
  works), and the out-of-bounds index errors of finding 17. `lerp_` into `b.t()[1:4]`, which failed to lower in the
  views probe, now runs.

Distributed round 4 (`overnight/dist4/`, torchrun 4 ranks, CPU reference): no new numbered finding.
- Correct: all_reduce SUM/MAX/MIN/AVG for fp32/bf16/fp16/int32/int64/int8/uint8/bool; reduce_scatter_tensor
  SUM/AVG/MAX/MIN fp32/bf16; collectives on transposed and stride-2 views with no storage offset, broadcast from src=2,
  all_gather into `unbind` views, reduce_scatter into a row slice; 8 async all_reduce waited in reverse order,
  `get_future`, interleaved async all_gather + reduce_scatter; disjoint and partial subgroups, subgroup broadcast with a
  global src; destroy + re-init of the process group; DDP basic, `gradient_as_bucket_view`, tiny buckets, `static_graph`,
  compiled DDP, AdamW, `find_unused_parameters` with alternating branches, `no_sync` grad accumulation, BN buffer
  broadcast; FSDP2 with frozen params, tied embedding/head, `set_requires_gradient_sync(False)` accumulation,
  `reshard_after_forward=2`, explicit `unshard()`/`reshard()`, no_grad eval, sharded and full optimizer state dict
  round trip; HSDP 2x2 with clip_grad_norm; DCP `async_save` (params changed right after save, load gives the old
  values); MoE expert parallel with `all_to_all_single_autograd` and uneven splits (fwd and grads 1e-7);
  `context_parallel` causal SDPA over 4 ranks (proj grad 1.9e-6); 31 DTensor ops x Shard(0)/Shard(1)/Replicate.
- Not reported, low: all_reduce with BAND/BOR/BXOR fails to lower for every int dtype (PRODUCT is in an earlier note);
  all_reduce with a different shape or dtype on one rank raises "Invalid NEFF" (no hang, but the message does not say
  what is wrong).
- Same root cause as the known BatchNorm issue: `SyncBatchNorm` running stats stay at 0/1. Its
  `aten::batch_norm_gather_stats_with_counts` lowering (`python_ops/torch_mlir/ops/normalization.py`) says in its
  docstring that running stats are not updated, so a BatchNorm fix must cover this op too.
- Upstream, not Neuron: DTensor `F.mse_loss` / `F.smooth_l1_loss` with `reduction="mean"` on Shard(0)/Shard(1)
  inputs give grads W times too large (forward right; `l1_loss` and `((a-b)**2).mean()` right); same on CPU gloo
  (`dist4/dt2_cpu.py`). FSDP2 `set_state_dict(..., broadcast_from_rank0=True)` with a full CPU state dict hangs on
  rank 0, also on CPU gloo (`dist4/osd3.py`). `clip_grad_norm_` on TP+FSDP 2D params on two meshes fails in
  `aten.stack` (upstream limit; torchtitan has its own clip).

Distributed round 5 (`overnight/dist5/`, third distributed session): checked and correct, in addition to findings 21-24.
- FSDP2: `reshard_after_forward=False`, explicit forward/backward prefetch, `shard_placement_fn` Shard(1), clip with
  foreach, buffers stay plain tensors, a Linear shared by two units at root, AC + per-block compile over 40 steps
  (loss equal to CPU at every logged step, flat memory), MixedPrecisionPolicy bf16 param + fp32/bf16 reduce and fp32 param
  + bf16 reduce (errors at bf16 level, same as CPU gloo), uneven batch size per rank (DDP, FSDP2, compiled DDP).
- TP: Embedding Colwise with Replicate output, SequenceParallel + RMSNorm + Shard(1) Colwise/Rowwise, loss_parallel with
  ignore_index and sum. DTensor Partial(sum/avg/max) -> Replicate/Shard(0)/Shard(1), uneven Shard(0) -> Shard(1)/Replicate.
- Collectives: 3 GiB bf16 and 2 GiB fp32 all_reduce, 1 GiB/rank all_gather, 500 small all_reduce (2.8 s),
  `_coalescing_manager`, `all_reduce_coalesced`, NaN/+inf/-inf through SUM/MAX/MIN/AVG/reduce_scatter/all_gather (fp32,
  bf16), FSDP2 clip_grad_norm_ with a NaN on one rank (norm NaN on all ranks, `error_if_nonfinite` raises), CPU tensors on
  the `cpu:gloo,neuron:neuron` backend, `barrier(device_ids=...)`, `monitored_barrier`, `torch.distributed.nn` all_reduce
  under compile with grad, int `all_to_all_single` and `gather` (exact), DDP with fp32 + bf16 params and tiny buckets.
  all_reduce bus bandwidth rises smoothly to about 104 GB/s at 256 MiB-1 GiB (no step at the 512 MiB bucket size); a small
  message takes about 0.14 ms.
- A peer that exits is detected at once (NRT error in 1 s on the others).
- Real-scale runs (`llama_scale.py`, 8 layers, hidden 512, GQA 8/4, RoPE, SwiGLU, vocab 2048, 300 AdamW steps with warmup
  + cosine + clip): FSDP2 fp32, FSDP2 bf16 params, FSDP2 + compiled blocks, DDP: loss equal to CPU (dp=4 global batch)
  to step 12, then the runs drift apart slowly (lr 3e-3), final 2.05 vs 2.04 on CPU; memory flat. TP+FSDP2 2x2 at S=120:
  equal to step 29, final 2.76 vs 2.70. At S=128 see finding 16. At S=120 with `torch.compile` per block: TP+FSDP2
  equal to step 29 (final 2.675 vs 2.701), DDP and FSDP2 fp32 equal to step 34 (2.027 vs 2.028), FSDP2 bf16 params
  2.036 vs 2.028; memory flat in all runs.
- Not reported: pipelining (`ScheduleGPipe`) fails at once with "RdmaManager: DiscoverEFAInfo failed" (finding 11, clear
  error, no hang). TP-only clip_grad_norm_ with plain norm params fails in `_foreach_norm` (mixed DTensor/Tensor, upstream).
  FSDP2 `output_dtype=float32` on inner blocks and a fp32 buffer under bf16 params give "mat1 and mat2 must have the same
  dtype" (same on CPU gloo). `PrepareModuleInput` without `use_local_output=True` before a plain RMSNorm fails (same on
  gloo). `loss_parallel` with `weight` and ignore_index fails (CPU gloo: "index -100 is out of bounds", upstream).
- Batch F-H (`probe_dist5f/g/h.py`): same seed gives the same rand/dropout stream on all ranks; TP dropout on a
  replicated activation is equal on all TP ranks; whole-model `torch.compile` of an FSDP2 model (3 steps, 1.3e-7);
  TP + non-reentrant activation checkpointing (3.3e-7); all_reduce on a side stream overlapped with default-stream compute
  and async wait on a side stream; meshes (1, 4) and (4, 1) for TP+FSDP2; `DeviceMesh._flatten`; a gloo subgroup next to
  the neuron default group; `destroy_process_group` with 4 async all_reduce pending (values right, re-init works with the
  `neuron` backend); `all_to_all_single` / `all_to_all` / `all_to_all_single_autograd` with zero splits and a rank that
  sends nothing; DCP save bf16 -> load into fp32, int64 3e9 and float64 1e300 in DCP state (exact: DCP does no data
  collectives); FSDP1 FULL_SHARD / SHARD_GRAD_OP / NO_SHARD / use_orig_params / wrap policy / sync_module_states /
  forward_prefetch, FULL and SHARDED state dicts, optim_state_dict, reload; ZeroRedundancyOptimizer; post-local SGD.
- Not reported (low or upstream): `all_reduce` / `all_gather` of an empty tensor raise "tensors cannot be empty" and
  `reduce_scatter_tensor` of an empty tensor fails to lower (NCCL treats these as no-ops). Compiled
  `all_to_all_single` with uneven splits fails to lower ("failed to legalize unresolved materialization"); even splits
  compile; `docs/AI-all-to-all.md` says variable a2a-v is an eager path. torchrun `--max-restarts 1`: the restarted
  attempt fails on CPU gloo too in this environment (gloo connectFullMesh), so not counted; on the `neuron` backend the
  restarted attempt hangs in the CC bootstrap ("rank 0 awaiting root parameters", retry every 240 s) instead of failing.
  With `cpu:gloo,neuron:neuron`, re-init after destroy fails in gloo (same as CPU gloo). FSDP1 `MixedPrecision` bf16
  gives 0.225 rel param error after 3 AdamW steps on Neuron and on CPU gloo alike. DCP stores one copy of a plain
  (non-DTensor) tensor that differs per rank (upstream semantics).
- Batch I-J: `torch.multiprocessing.spawn` launch (4 procs, each on its own core) works; a fork after the Neuron runtime
  starts raises a clear "Neuron runtime should not be used after fork(); use spawn or num_workers=0" in the child (parent
  keeps working); DataLoader `num_workers=2` after Neuron init works. Collectives on 2^31 + 4096 bf16 elements
  (all_reduce, reduce_scatter_tensor input, all_gather_into_tensor output) are correct. GradScaler with one rank
  producing inf: DDP + fp16 autocast and FSDP2 + `amp.GradScaler("neuron")` / `ShardedGradScaler(device="neuron")` skip
  the step on every rank and halve the scale. Not reported (upstream): `ShardedGradScaler()` with its default
  `device="cuda"` disables itself on a host without CUDA (scale 1.0, the inf step is applied and params become inf), so
  pass `device="neuron"`; FSDP2 `set_gradient_divide_factor(1.0)` and `set_force_sum_reduction_for_comms(True)` still
  give mean-reduced grads, same on CPU gloo. `Work.is_completed()` is False right after `wait()` (stream semantics, as NCCL).
- Collective first-call cost (`small_latency.py`): each new shape of an eager collective compiles a NEFF, about 400-460 ms
  per shape cold and 34 ms from the persistent cache; after that a small all_reduce takes 0.13-0.18 ms (same tensor,
  distinct tensors, row views). Expected for a NEFF-per-shape design; noted for warm-up time with many parameter shapes.
- MoE expert parallel end to end (`moe_ep.py`, 4 experts on 4 ranks, top-1 routing, uneven `all_to_all_single_autograd`
  dispatch and combine, aux loss, 30 AdamW steps): loss equal to the CPU reference at every step (max diff 1.2e-7),
  expert params within 2.2e-6.
- DDP comm hooks: `fp16_compress_hook`, `quantization_pertensor_hook`, FSDP1 `LowPrecisionState` fp16 hook run correctly.
  `quantization_perchannel_hook` and `batched_powerSGD_hook` fail on CPU gloo too (upstream).
- DCP resharding across layouts (`dcp_reshard.py`): save TP=4 -> load FSDP=4, FSDP=4 -> TP2 x FSDP2, TP2 x FSDP2 -> plain
  model, FSDP=4 -> TP=4: params bit-exact, an int64 buffer [20000001, 3000000007] exact, outputs equal (1e-7).
