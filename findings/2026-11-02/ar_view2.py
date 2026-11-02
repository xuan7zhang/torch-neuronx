import os, torch, torch.distributed as dist, torch_neuronx
r = int(os.environ["LOCAL_RANK"]); torch.neuron.set_device(r); dev = torch.device("neuron", r)
dist.init_process_group("neuron")
base = torch.arange(12.0, device=dev).reshape(3, 4) * (r + 1)
v = base[:, 1:3]
def meta(t): return f"shape={tuple(t.shape)} stride={t.stride()} off={t.storage_offset()} nbytes={t.untyped_storage().nbytes()} ptr={t.untyped_storage().data_ptr()}"
if r == 0: print("RES before v", meta(v), "| base", meta(base), flush=True)
dist.all_reduce(v)
if r == 0: print("RES after  v", meta(v), "| base", meta(base), flush=True)
try:
    bc = base.cpu(); print("RES" if r == 0 else "", "base.cpu ok", bc.tolist() if r == 0 else "")
except Exception as e:
    if r == 0: print("RES base.cpu ERR", str(e)[:80])
try:
    vc = v.contiguous().cpu()
    if r == 0: print("RES v.contiguous().cpu ok", vc.tolist())
except Exception as e:
    if r == 0: print("RES v.contiguous().cpu ERR", str(e)[:80])
try:
    vc = v.cpu()
    if r == 0: print("RES v.cpu ok", vc.tolist())
except Exception as e:
    if r == 0: print("RES v.cpu ERR", str(e)[:80])
# same view copy without collective
b2 = torch.arange(12.0, device=dev).reshape(3, 4); v2 = b2[:, 1:3]; v2.add_(1.0)
try:
    if r == 0: print("RES no-collective add_ then v2.cpu ok", v2.cpu().tolist())
except Exception as e:
    if r == 0: print("RES no-collective v2.cpu ERR", str(e)[:80])
dist.destroy_process_group()
