# Overnight state (for loop iterations)
start: 2026-11-02 08:22 UTC
end:   2026-11-02 17:22 UTC
## Done
- eager2 probe: findings 1-5 in report (rng/ckpt, convT groups, convT dil+stride, tanh_backward bcast, NaN known)
## Next
- eager ops probe 2 (norms, embedding_bag, scatter dup, ctc, convT, pool, interp, sdpa, autocast, ckpt, func, views)
- optimizers probe (foreach/fused/capturable, schedulers, GradScaler, state_dict)
- examples
- compile training probe 2
- distributed probe 2
- examples + model-level train compare: OK
- compile2: findings 2 (foreach alias), 3 (CE label shift int64), 8 (dynamic default); ckpt+dropout under compile added to 1
## Next (09:40)
- dist2: run each case with torchrun 4 (and dcp_load2 with 2)
- then: optimizer state_dict across devices, GradScaler details, more eager in-place-nonfirst-arg ops, LR schedulers
- dist2 + coll_each: findings 2 (coll view SCATTER_IOTA), 10 (P2P EFA), 11 (uneven all_gather); FSDP2 AC dropout added to 1
- eager3: clean (44 OK)
## Next (10:08)
- compile + distributed more: FSDP2 compile with AC; DDP compile with dropout
- eager: SDPA dropout backward consistency (mask vs grad), MHA dropout, more bf16 numerics (layernorm bf16 grads, softmax bf16)
- perf: big slowdowns (fallback ops in common training: embedding_bag, ctc, max_pool2d (fallback!), adaptive_max_pool, avg_pool3d)
- 10:25 eager SDPA dropout (finding 1), perf fallbacks section, eager3 clean
- 10:25 long train OK; compile perf (float lr recompiles) noted; rng proto: ckpt fixed, mid-stream not
- 10:55 sweep done (400/412 OK); conv stride+dil fix proto works; CE soft target grad finding 13
- 11:35 bf16 sweep: finding 14 adaptive pool bf16
- 12:05 compile sweep: 71 known, rest zero-size (low)
