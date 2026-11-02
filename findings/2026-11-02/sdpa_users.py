import torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
from torch.utils._python_dispatch import TorchDispatchMode
class Spy(TorchDispatchMode):
    def __init__(self): super().__init__(); self.hits = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        n = str(func)
        if "scaled_dot_product" in n or "attention" in n:
            dp = args[4] if len(args) > 4 and isinstance(args[4], float) else kwargs.get("dropout_p") if kwargs else None
            self.hits.append((n, dp))
        return func(*args, **(kwargs or {}))
dev = "neuron"
def run(name, m, *inp):
    m = m.to(dev).train(); s = Spy()
    with s: out = m(*[t.to(dev) for t in inp])
    print("RES", f"{name:40s}", sorted(set(s.hits)) or "no SDPA op")
x = torch.randn(2, 8, 32)
run("nn.MultiheadAttention(dropout=0.1) need_weights=False", type("W", (nn.Module,), {"__init__": lambda self: (nn.Module.__init__(self), setattr(self, "a", nn.MultiheadAttention(32, 4, dropout=0.1, batch_first=True)))[0], "forward": lambda self, x: self.a(x, x, x, need_weights=False)[0]})(), x)
run("nn.TransformerEncoderLayer(dropout=0.1)", nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True), x)
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
ids = torch.randint(0, 100, (2, 8))
run("HF GPT2 (attn_pdrop=0.1, default attn impl)", GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=4, n_embd=32, vocab_size=100, n_positions=16)), ids)
run("HF Llama (attention_dropout=0.1)", LlamaForCausalLM(LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.1)), ids)
