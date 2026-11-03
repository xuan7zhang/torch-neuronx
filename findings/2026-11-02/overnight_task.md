# Overnight task: find more training issues in torch-neuronx

Run for about 9 hours on this trn2.3xlarge (4 logical Neuron cores). Work only locally.
Do not touch GitHub: no issues, PRs, comments, or pushes. Do not commit.
Put all scripts and logs in `agent_space/`. Use `source .venv/bin/activate` first.

## Already known (do not report again)
1. cross_entropy fails on AsyncCollectiveTensor logits (issue #1026, fix on branch
   `fix/cross-entropy-async-collective-tensor`).
2. Eager BatchNorm does not update running_mean/running_var; SyncBatchNorm too
   (draft: `agent_space/drafts/issue_batchnorm_running_stats.md`).
3. `_fused_sgd_`, `_fused_adagrad_`, `_amp_update_scale_`,
   `_amp_foreach_non_finite_check_and_unscale_` fall back to CPU (docs say Native for fused SGD/Adagrad).

Already probed and correct: 75 eager cases (`agent_space/probe_eager.py`), 11 compile cases
(`agent_space/probe_compile.py`), DDP / FSDP2 / DCP / DDP+compile / FSDP2+compile
(`agent_space/probe_dist.py`).

## Where to look next
- Run the examples: `examples/gpt2-train-loop`, `examples/llama`, `examples/simple_moe`,
  `examples/simple_asr`, `examples/simple_mlp_tpxfsdp`, `examples/compile`, `examples/torch_compile`.
  Note any example that fails or does not match its README.
- More eager training ops vs CPU fp32: norms (GroupNorm, InstanceNorm, RMSNorm), embedding_bag,
  scatter/index_put backward with duplicates, ctc_loss, conv transpose backward, pooling backward,
  interpolate backward, SDPA backward with masks and dropout, autocast bf16, gradient checkpointing,
  torch.func (grad, vmap), in-place ops on views with autograd, other in-place ops that mutate
  a non-first argument (same class as the BatchNorm bug).
- Optimizers: every torch.optim class with foreach/fused/capturable, LR schedulers, GradScaler,
  optimizer state_dict save/load across devices.
- torch.compile training: dynamic shapes, recompiles, graph breaks, compiled optimizer step,
  compiled autocast, mode="max-autotune"-style options.
- Distributed (torchrun --nproc-per-node 4): TP (ColwiseParallel/RowwiseParallel/SequenceParallel),
  TP+FSDP2, activation checkpointing with FSDP2, DCP reshard 4->2, all collectives in
  torch.distributed and _functional_collectives, uneven shapes, process subgroups.

## Method
For each case, compare Neuron with CPU fp32 (loss, grads, params, buffers). Record silent wrong
results, errors, CPU fallbacks (`torch_neuronx.get_fallback_ops()`), hangs, and big slowdowns.
Use timeouts so a hang does not stop the run. For each finding: make a minimal repro, find the
root cause in the source, and check `docs/torch_api_compatibility.md`, `release-notes/`, and the
xfail registries to see if it is already known.

## Output
Write `agent_space/overnight_report.md`, sorted by impact. For each finding: one-line summary,
minimal repro with real output, root cause, possible fix, and whether it is already known.
Update the report as you go, so partial results survive.
