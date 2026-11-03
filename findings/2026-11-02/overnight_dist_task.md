# Overnight task 2: distributed training issues (10 h)

Window: 2026-11-03 07:48 UTC to 17:48 UTC on this trn2.3xlarge (4 logical Neuron cores, LNC=2).
Use `source ~/torch-neuronx/.venv/bin/activate`. Put scripts and logs in `agent_space/overnight/dist5/`.
Report: add to `agent_space/overnight_report.md` (new findings get the next number, 21, 22, ...).

## Rules
- Work locally. Do not open issues or PRs, and do not comment on GitHub.
- Backup only: about every 3 hours, and at the end, copy the report and new scripts/logs into the worktree
  `~/tnx-findings/findings/2026-11-02/` and push branch `overnight-findings-2026-11-02` to remote `fork`
  (xuan7zhang/torch-neuronx). Never push to `origin`. Author xuan7zhang <zxq5706346@gmail.com>, no co-author,
  `--no-verify`. Redact the instance ID and host name in logs before the copy.
- Do not change files in the repo outside `agent_space/`.
- Bash foreground timeout max is 600000 ms. Run long jobs with `run_in_background` and `timeout`.
- One torchrun per case, with `timeout`. Use a new `--master-port` for each run.
- Do not report known items again: findings 1-20 in the report, the known items in `overnight_task.md`,
  and the "Not reported" notes. Upstream bugs (also on CPU gloo) go in notes, not as findings.

## Already checked (rounds 2-4), skip
Collectives (all ops x dtypes, views, async, subgroups, reinit), DDP options, FSDP2 basics/frozen/tied/no-sync/
unshard/optimizer state dict, HSDP 2x2, DCP save/load/reshard/async_save, TP Llama plan + loss_parallel,
SequenceParallel, TP+FSDP2 eager/compiled, context_parallel SDPA, EP all_to_all autograd, 31 DTensor ops.

## Where to look
1. Real-scale runs vs CPU: Llama-style model (8+ layers, hidden 512-1024) with TP+FSDP2, 200-500 steps,
   bf16 mixed precision; compare loss curves with a single-process CPU run on the global batch.
   Also FSDP2 + activation checkpointing + compile over many steps; memory over time (leaks per step).
2. FSDP2 MixedPrecisionPolicy variants (param bf16 / reduce fp32 / output_dtype, cast_forward_inputs),
   `reshard_after_forward=False`, prefetch (`set_modules_to_forward_prefetch` / backward prefetch),
   `fully_shard` on a module with buffers, shared params across FSDP units, FSDP2 + gradient clipping
   with foreach, FSDP2 + `torch.compile` per block, FSDP2 with `Shard(1)` placement (`shard_placement_fn`).
3. TP extras: `PrepareModuleInput/Output`, `SequenceParallel` + RMSNorm + RoPE, TP on Embedding with
   Shard(1) (column) output, TP + SDPA GQA with head counts not divisible by tp, `loss_parallel` with
   ignore_index and label smoothing, async TP if available, DTensor `redistribute` with Partial -> Shard.
4. Pipeline parallel (`torch.distributed.pipelining`) schedules; expected to need P2P (finding 11).
   Check whether it fails clearly or hangs.
5. Collective edge cases: very large (>2 GB) all_reduce / all_gather, many small tensors (coalesced),
   `all_reduce_coalesced`, collectives on int64 > 2^31 values, bf16 SUM precision for 4 ranks,
   `dist.barrier(device_ids=...)`, timeouts (`timeout=` on init) when one rank stalls, `monitored_barrier`.
6. Mixed: CPU tensor collectives on the `cpu:gloo,neuron:neuron` backend, object collectives with Neuron
   tensors inside, `torch.distributed.nn` functional ops with autograd under compile.
7. Elastic / robustness: one rank raises inside a collective sequence; `destroy_process_group` while
   async work is pending; torchrun `--max-restarts`.

## Method
Compare Neuron results with a CPU reference (single process on the global batch, or CPU gloo with the
same script and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`). Check every suspect on CPU gloo before calling it a
Neuron bug. For each finding: minimal repro with real output, root cause in the source, possible fix,
known or not (docs/torch_api_compatibility.md, release notes, xfail registries).
Keep a short progress log in `agent_space/overnight/dist5/state.md` so a later wakeup can continue.
