import torch, torch_neuronx
base = torch.arange(12.0).reshape(3, 4)
nb = base.to("neuron")
for nm, f in [("t()[1:3]", lambda b: b.t()[1:3]), ("t()[1]", lambda b: b.t()[1]), ("t()[0:2]", lambda b: b.t()[0:2]), ("t()[2:4]", lambda b: b.t()[2:4]),
              ("[:,1:3]", lambda b: b[:, 1:3]), ("[:, 3]", lambda b: b[:, 3]), ("t()", lambda b: b.t()), ("t()[3]", lambda b: b.t()[3]), ("[1:,1:].t()", lambda b: b[1:, 1:].t()),
              ("as_strided off", lambda b: b.as_strided((2, 3), (1, 4), 1)), ("as_strided off3", lambda b: b.as_strided((1, 3), (1, 4), 3))]:
    v = f(nb)
    try:
        c = v.cpu(); ok = torch.equal(c, f(base)); print("RES", "OK " if ok else "BAD", f"{nm:16s} shape={tuple(v.shape)} stride={v.stride()} off={v.storage_offset()}", "" if ok else (c.tolist(), f(base).tolist()))
    except Exception as e:
        print("RES ERR", f"{nm:16s} shape={tuple(v.shape)} stride={v.stride()} off={v.storage_offset()}", str(e)[:80])
