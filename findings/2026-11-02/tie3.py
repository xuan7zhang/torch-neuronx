import sys, torch, torch.nn.functional as F, torch_neuronx
mode = sys.argv[1]
w = torch.randn(50, 16, device="neuron", requires_grad=True)
ids = torch.randint(0, 50, (4, 9), generator=torch.Generator().manual_seed(0)).to("neuron")
fs = {
 "emb_slice_sum": lambda ids, w: F.embedding(ids[:, :-1], w).sum(),
 "emb_slice1_sum": lambda ids, w: F.embedding(ids[:, 1:], w).sum(),
 "emb_contig_slice": lambda ids, w: F.embedding(ids[:, :-1].contiguous(), w).sum(),
 "emb_row_slice": lambda ids, w: F.embedding(ids[:-1], w).sum(),
 "emb_full": lambda ids, w: F.embedding(ids, w).sum(),
 "gather_slice": lambda ids, w: w[ids[:, :-1]].sum(),
 "index_select_slice": lambda ids, w: w.index_select(0, ids[:, :-1].reshape(-1)).sum(),
}
try:
    l = torch.compile(fs[mode], backend="neuron")(ids, w); l.backward(); torch.neuron.synchronize(); print("RES", mode, "OK")
except Exception as e:
    msg = str(e); i = msg.find("NCC_"); print("RES", mode, "ERROR", msg[i:i+90].replace("\n"," ") if i>0 else msg[:150].replace("\n"," "))
