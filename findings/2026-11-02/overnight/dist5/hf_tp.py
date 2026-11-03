"""HF transformers tp_plan="auto" on Neuron vs CPU. Step 1 (no torchrun): python hf_tp.py save. Step 2: torchrun ... hf_tp.py run"""
import os
import sys

import torch
import torch.nn.functional as F

PATH = "/tmp/claude-1000/hf_tiny_llama"
if sys.argv[1] == "save":
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=4, max_position_embeddings=128, tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(PATH)
    print("saved")
    sys.exit(0)

import torch.distributed as dist
import torch_neuronx  # noqa: F401
from transformers import LlamaForCausalLM

rank = int(os.environ["RANK"])
torch.neuron.set_device(rank)
if os.environ.get("PREINIT"):
    dist.init_process_group("neuron")
ids = torch.randint(0, 512, (2, 32), generator=torch.Generator().manual_seed(1))
ref = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32)
rl = ref(input_ids=ids, labels=ids).loss
rl.backward()
try:
    m = LlamaForCausalLM.from_pretrained(PATH, tp_plan="auto", torch_dtype=torch.float32)
    devs = sorted({str(p.device) for p in m.parameters()})
    out = m(input_ids=ids.to(f"neuron:{rank}"), labels=ids.to(f"neuron:{rank}"))
    out.loss.backward()
    torch.neuron.synchronize()
    full = lambda t: t.full_tensor() if hasattr(t, "full_tensor") else t
    worst = max((((a.grad - full(b.grad).cpu()).norm() / (a.grad.norm() + 1e-12)).item(), n)
                for (n, a), (_, b) in zip(ref.named_parameters(), m.named_parameters()) if b.grad is not None)
    n_dt = sum(1 for p in m.parameters() if hasattr(p, "full_tensor"))
    msg = f"loss {out.loss.item():.6f} vs cpu {rl.item():.6f}; worst grad rel {worst[0]:.3g} ({worst[1]}); DTensor params {n_dt}; devices {devs}"
except Exception as e:
    import traceback
    tb = [l.strip() for l in traceback.format_exc().splitlines() if "File" in l][-9:]
    msg = f"ERR {type(e).__name__}: {str(e)[:250]} | {tb}".replace("\n", " ")
if rank == 0:
    print("RES hf tp_plan=auto:", msg)
if dist.is_initialized():
    dist.destroy_process_group()
