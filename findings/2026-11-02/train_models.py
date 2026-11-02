"""Train example models on CPU and neuron for a few steps; compare losses, grads and params."""

import copy
import sys

import torch
import torch.nn.functional as F
import torch_neuronx

sys.path.insert(0, "/home/ec2-user/torch-neuronx/examples/llama")
sys.path.insert(0, "/home/ec2-user/torch-neuronx/examples/simple_moe")

which = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def build():
    torch.manual_seed(0)
    if which == "llama":
        from llama import LlamaModel
        m = LlamaModel(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_length=64)
        fwd = lambda m, ids: m(ids)
    elif which == "moe":
        from simple_moe import SimpleMoEModel
        m = SimpleMoEModel(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, num_experts=4, top_k=2,
                           max_seq_length=64, dropout=0.0)
        fwd = lambda m, ids: m(ids)
    elif which in ("gpt2", "gpt2_ckpt"):
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(n_layer=2, n_head=4, n_embd=128, vocab_size=1000, n_positions=64,
                         resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        m = GPT2LMHeadModel(cfg)
        if which == "gpt2_ckpt":
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        fwd = lambda m, ids: m(input_ids=ids).logits
    elif which == "llama_hf":
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        m = LlamaForCausalLM(cfg)
        fwd = lambda m, ids: m(input_ids=ids).logits
    return m, fwd


def run(dev, m0, fwd):
    m = copy.deepcopy(m0).to(dev)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(1)
    losses, grads = [], None
    for s in range(steps):
        ids = torch.randint(0, 1000, (2, 32), generator=g).to(dev)
        logits = fwd(m, ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        if s == 0:
            grads = {n: p.grad.detach().float().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    params = {n: p.detach().float().cpu() for n, p in m.named_parameters()}
    return losses, grads, params


m0, fwd = build()
lc, gc, pc = run("cpu", m0, fwd)
ln, gn, pn = run("neuron", m0, fwd)
print("loss cpu   ", [round(x, 5) for x in lc])
print("loss neuron", [round(x, 5) for x in ln])
missing = set(gc) ^ set(gn)
if missing:
    print("grad presence differs:", sorted(missing))
worst = sorted(((n, ((gc[n] - gn[n]).abs().max() / (gc[n].abs().max() + 1e-12)).item()) for n in gc if n in gn),
               key=lambda t: -t[1])[:5]
print("worst rel grad diff step0:", [(n, round(v, 5)) for n, v in worst])
pw = sorted(((n, (pc[n] - pn[n]).abs().max().item()) for n in pc), key=lambda t: -t[1])[:3]
print("worst abs param diff end:", [(n, round(v, 6)) for n, v in pw])
print("fallback:", sorted(torch_neuronx.get_fallback_ops()))
