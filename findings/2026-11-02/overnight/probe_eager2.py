"""Eager training probes, round 2. Each case returns a list of tensors; compare CPU vs neuron."""

import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

DEV = "neuron"
CASES = {}


def case(name, atol=1e-3, rtol=1e-3):
    def deco(fn):
        CASES[name] = (fn, atol, rtol)
        return fn
    return deco


def rnd(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def leaf(t, dev):
    return t.clone().to(dev).requires_grad_(True)


def mod_grads(make, x, dev, loss=lambda o: (o * o).sum(), train=True):
    torch.manual_seed(0)
    m = make().to(dev)
    m.train(train)
    xi = leaf(x, dev)
    out = m(xi)
    loss(out).backward()
    res = [out.detach(), xi.grad]
    res += [p.grad for p in m.parameters()]
    res += [b for b in m.buffers() if b.is_floating_point()]
    return res


# ---------------- norms ----------------
for nm, make, shape in [
    ("groupnorm", lambda: nn.GroupNorm(4, 16), (4, 16, 8, 8)),
    ("groupnorm_noaffine", lambda: nn.GroupNorm(2, 8, affine=False), (2, 8, 5)),
    ("instancenorm2d", lambda: nn.InstanceNorm2d(8, affine=True), (2, 8, 6, 6)),
    ("instancenorm2d_track", lambda: nn.InstanceNorm2d(8, affine=True, track_running_stats=True), (2, 8, 6, 6)),
    ("instancenorm1d", lambda: nn.InstanceNorm1d(8), (2, 8, 10)),
    ("rmsnorm", lambda: nn.RMSNorm(32), (4, 7, 32)),
    ("rmsnorm_noeps", lambda: nn.RMSNorm([7, 32], eps=None), (4, 7, 32)),
    ("layernorm_nobias", lambda: nn.LayerNorm(32, bias=False), (4, 7, 32)),
    ("layernorm_2d", lambda: nn.LayerNorm([7, 32]), (4, 7, 32)),
    ("localresponsenorm", lambda: nn.LocalResponseNorm(3), (2, 8, 5, 5)),
]:
    case(nm)(lambda dev, make=make, shape=shape: mod_grads(make, rnd(*shape), dev))


@case("instancenorm_eval_track")
def _(dev):
    torch.manual_seed(0)
    m = nn.InstanceNorm2d(4, track_running_stats=True).to(dev)
    for i in range(3):
        m(rnd(2, 4, 5, 5, seed=i).to(dev))
    m.eval()
    return [m(rnd(2, 4, 5, 5, seed=9).to(dev)), m.running_mean, m.running_var]


@case("weight_norm_param")
def _(dev):
    from torch.nn.utils.parametrizations import weight_norm
    return mod_grads(lambda: weight_norm(nn.Linear(8, 6)), rnd(4, 8), dev)


@case("spectral_norm_param")
def _(dev):
    from torch.nn.utils.parametrizations import spectral_norm
    torch.manual_seed(0)
    m = spectral_norm(nn.Linear(8, 6))
    m.parametrizations.weight[0]._u.copy_(F.normalize(rnd(6), dim=0))
    m.parametrizations.weight[0]._v.copy_(F.normalize(rnd(8, seed=1), dim=0))
    m = m.to(dev)
    x = leaf(rnd(4, 8), dev)
    o = m(x)
    o.sum().backward()
    return [o, x.grad, m.parametrizations.weight[0]._u, m.parametrizations.weight.original.grad]


# ---------------- embedding ----------------
for mode in ["sum", "mean", "max"]:
    @case(f"embedding_bag_{mode}")
    def _(dev, mode=mode):
        torch.manual_seed(0)
        m = nn.EmbeddingBag(20, 6, mode=mode).to(dev)
        idx = torch.tensor([1, 2, 2, 5, 1, 9, 9, 9, 3]).to(dev)
        off = torch.tensor([0, 3, 5]).to(dev)
        o = m(idx, off)
        (o * torch.arange(6.0, device=dev)).sum().backward()
        return [o, m.weight.grad]


@case("embedding_bag_per_sample_weights")
def _(dev):
    torch.manual_seed(0)
    m = nn.EmbeddingBag(20, 6, mode="sum").to(dev)
    idx = torch.tensor([1, 2, 2, 5, 1, 9]).to(dev)
    off = torch.tensor([0, 3]).to(dev)
    w = leaf(rnd(6), dev)
    o = m(idx, off, per_sample_weights=w)
    (o ** 2).sum().backward()
    return [o, m.weight.grad, w.grad]


@case("embedding_padding_idx_dup")
def _(dev):
    torch.manual_seed(0)
    m = nn.Embedding(10, 4, padding_idx=0).to(dev)
    idx = torch.tensor([[0, 1, 1, 3], [3, 3, 0, 9]]).to(dev)
    o = m(idx)
    (o * rnd(2, 4, 4).to(dev)).sum().backward()
    return [o, m.weight.grad]


@case("embedding_scale_grad_by_freq")
def _(dev):
    torch.manual_seed(0)
    m = nn.Embedding(10, 4, scale_grad_by_freq=True).to(dev)
    idx = torch.tensor([[1, 1, 1, 3], [3, 5, 0, 9]]).to(dev)
    m(idx).sum().backward()
    return [m.weight.grad]


@case("embedding_max_norm")
def _(dev):
    torch.manual_seed(0)
    m = nn.Embedding(10, 4, max_norm=0.5).to(dev)
    idx = torch.tensor([1, 2, 2, 7]).to(dev)
    o = m(idx)
    o.sum().backward()
    return [o, m.weight, m.weight.grad]


@case("embedding_sparse_grad")
def _(dev):
    torch.manual_seed(0)
    m = nn.Embedding(10, 4, sparse=True).to(dev)
    idx = torch.tensor([1, 2, 2, 7]).to(dev)
    m(idx).sum().backward()
    return [m.weight.grad.to_dense()]


# ---------------- scatter / index with duplicates ----------------
@case("index_put_accumulate_dup_bwd")
def _(dev):
    base = leaf(rnd(6, 3), dev)
    v = leaf(rnd(5, 3, seed=1), dev)
    idx = torch.tensor([1, 1, 4, 1, 0]).to(dev)
    o = base.index_put((idx,), v, accumulate=True)
    (o * rnd(6, 3, seed=2).to(dev)).sum().backward()
    return [o, base.grad, v.grad]


@case("index_put_noacc_dup_bwd_unique")
def _(dev):
    base = leaf(rnd(6, 3), dev)
    v = leaf(rnd(3, 3, seed=1), dev)
    idx = torch.tensor([1, 4, 0]).to(dev)
    o = base.index_put((idx,), v)
    (o * rnd(6, 3, seed=2).to(dev)).sum().backward()
    return [o, base.grad, v.grad]


@case("setitem_bool_mask_bwd")
def _(dev):
    x = leaf(rnd(5, 4), dev)
    y = x * 2
    y[y > 0] = 0.0
    y.sum().backward()
    return [y, x.grad]


@case("index_add_dup_bwd")
def _(dev):
    base = leaf(rnd(6, 3), dev)
    src = leaf(rnd(5, 3, seed=1), dev)
    idx = torch.tensor([2, 2, 2, 5, 0]).to(dev)
    o = base.index_add(0, idx, src, alpha=0.5)
    (o * rnd(6, 3, seed=2).to(dev)).sum().backward()
    return [o, base.grad, src.grad]


@case("scatter_add_dup_bwd")
def _(dev):
    base = leaf(rnd(4, 5), dev)
    src = leaf(rnd(4, 5, seed=1), dev)
    idx = torch.tensor([[0, 1, 1, 1, 4]] * 4).to(dev)
    o = base.scatter_add(1, idx, src)
    (o ** 2).sum().backward()
    return [o, base.grad, src.grad]


for red in ["sum", "prod", "mean", "amax", "amin"]:
    @case(f"scatter_reduce_{red}_bwd")
    def _(dev, red=red):
        base = leaf(rnd(4, 5).abs() + 0.5, dev)
        src = leaf(rnd(4, 6, seed=1).abs() + 0.5, dev)
        idx = torch.tensor([[0, 1, 1, 3, 4, 4]] * 4).to(dev)
        o = base.scatter_reduce(1, idx, src, reduce=red, include_self=True)
        (o * rnd(4, 5, seed=3).to(dev)).sum().backward()
        return [o, base.grad, src.grad]


@case("scatter_reduce_mean_noself_bwd")
def _(dev):
    base = leaf(rnd(4, 5), dev)
    src = leaf(rnd(4, 6, seed=1), dev)
    idx = torch.tensor([[0, 1, 1, 3, 4, 4]] * 4).to(dev)
    o = base.scatter_reduce(1, idx, src, reduce="mean", include_self=False)
    (o * rnd(4, 5, seed=3).to(dev)).sum().backward()
    return [o, base.grad, src.grad]


@case("gather_dup_bwd")
def _(dev):
    x = leaf(rnd(4, 5), dev)
    idx = torch.tensor([[0, 0, 0], [1, 4, 1], [2, 2, 3], [4, 4, 4]]).to(dev)
    o = x.gather(1, idx)
    (o * rnd(4, 3, seed=1).to(dev)).sum().backward()
    return [o, x.grad]


@case("index_select_dup_bwd")
def _(dev):
    x = leaf(rnd(6, 3), dev)
    o = x.index_select(0, torch.tensor([5, 5, 1, 5, 0]).to(dev))
    (o * rnd(5, 3, seed=1).to(dev)).sum().backward()
    return [o, x.grad]


@case("advanced_index_dup_bwd")
def _(dev):
    x = leaf(rnd(6, 4), dev)
    o = x[torch.tensor([1, 1, 3]).to(dev)][:, torch.tensor([0, 0, 2]).to(dev)]
    (o * rnd(3, 3, seed=1).to(dev)).sum().backward()
    return [o, x.grad]


@case("masked_scatter_bwd")
def _(dev):
    x = leaf(rnd(4, 4), dev)
    src = leaf(rnd(16, seed=1), dev)
    mask = (rnd(4, 4, seed=2) > 0).to(dev)
    o = x.masked_scatter(mask, src)
    (o * rnd(4, 4, seed=3).to(dev)).sum().backward()
    return [o, x.grad, src.grad]


@case("take_along_dim_bwd")
def _(dev):
    x = leaf(rnd(4, 6), dev)
    idx = torch.argsort(rnd(4, 6, seed=1), dim=1).to(dev)[:, :3]
    o = torch.take_along_dim(x, idx, dim=1)
    (o ** 2).sum().backward()
    return [o, x.grad]


# ---------------- losses ----------------
@case("ctc_loss")
def _(dev):
    T, N, C, S = 12, 3, 6, 4
    lp = leaf(rnd(T, N, C).log_softmax(-1), dev)
    tgt = torch.tensor([[1, 2, 2, 3], [4, 1, 5, 0], [3, 3, 0, 0]]).to(dev)
    il = torch.tensor([12, 10, 8]).to(dev)
    tl = torch.tensor([4, 3, 2]).to(dev)
    loss = F.ctc_loss(lp, tgt, il, tl, blank=0, reduction="mean", zero_infinity=True)
    loss.backward()
    return [loss, lp.grad]


@case("ctc_loss_sum_none")
def _(dev):
    T, N, C = 10, 2, 5
    lp = leaf(rnd(T, N, C).log_softmax(-1), dev)
    tgt = torch.tensor([1, 2, 3, 4, 4]).to(dev)
    il = torch.tensor([10, 9]).to(dev)
    tl = torch.tensor([3, 2]).to(dev)
    loss = F.ctc_loss(lp, tgt, il, tl, reduction="none")
    loss.sum().backward()
    return [loss, lp.grad]


@case("losses_more")
def _(dev):
    out = []
    x = leaf(rnd(6, 5), dev)
    y = torch.tensor([0, 4, 2, 2, 1, 3]).to(dev)
    t = rnd(6, 5, seed=1).to(dev)
    ls = [
        F.cross_entropy(x, y, label_smoothing=0.1),
        F.cross_entropy(x, y, weight=torch.arange(1.0, 6.0).to(dev), ignore_index=2),
        F.cross_entropy(x, t.softmax(-1)),
        F.nll_loss(x.log_softmax(-1), y, reduction="sum"),
        F.multi_margin_loss(x, y),
        F.multilabel_soft_margin_loss(x, (t > 0).float()),
        F.multilabel_margin_loss(x, torch.tensor([[0, 1, -1, 0, 0]] * 6).to(dev)),
        F.kl_div(x.log_softmax(-1), t.softmax(-1), reduction="batchmean"),
        F.kl_div(x.log_softmax(-1), t.log_softmax(-1), reduction="batchmean", log_target=True),
        F.poisson_nll_loss(x, t.abs()),
        F.gaussian_nll_loss(x, t, torch.ones(6, 5, device=dev) * 0.5),
        F.binary_cross_entropy_with_logits(x, (t > 0).float(), pos_weight=torch.ones(5, device=dev) * 2),
        F.binary_cross_entropy(x.sigmoid(), (t > 0).float()),
        F.soft_margin_loss(x, t.sign()),
        F.hinge_embedding_loss(x, t.sign()),
        F.cosine_embedding_loss(x, t, torch.tensor([1, -1, 1, 1, -1, 1]).to(dev)),
        F.margin_ranking_loss(x[:, 0], x[:, 1], t[:, 0].sign()),
        F.triplet_margin_loss(x[:2], x[2:4], x[4:]),
        F.l1_loss(x, t),
        F.mse_loss(x, t, reduction="sum"),
    ]
    for i, l in enumerate(ls):
        x.grad = None
        l.backward(retain_graph=True)
        out += [l.detach(), x.grad.clone()]
    return out


# ---------------- conv / pool / interp ----------------
for nm, make, shape in [
    ("conv_transpose1d", lambda: nn.ConvTranspose1d(4, 6, 3, stride=2, padding=1, output_padding=1), (2, 4, 9)),
    ("conv_transpose2d", lambda: nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1, output_padding=1), (2, 4, 7, 7)),
    ("conv_transpose2d_groups_dil", lambda: nn.ConvTranspose2d(4, 8, 3, stride=2, groups=2, dilation=2), (2, 4, 5, 5)),
    ("conv_transpose3d", lambda: nn.ConvTranspose3d(2, 3, 3, stride=2), (1, 2, 4, 4, 4)),
    ("conv2d_groups_dil_pad", lambda: nn.Conv2d(4, 8, 3, groups=4, dilation=2, padding="same", padding_mode="reflect"), (2, 4, 9, 9)),
    ("conv2d_circular", lambda: nn.Conv2d(3, 4, 3, padding=1, padding_mode="circular"), (2, 3, 6, 6)),
    ("conv1d_stride", lambda: nn.Conv1d(3, 5, 4, stride=3, padding=2), (2, 3, 20)),
    ("conv3d", lambda: nn.Conv3d(2, 4, 3, padding=1), (1, 2, 5, 5, 5)),
    ("maxpool2d_overlap", lambda: nn.MaxPool2d(3, stride=2, padding=1), (2, 3, 9, 9)),
    ("maxpool2d_ceil_dil", lambda: nn.MaxPool2d(2, stride=2, dilation=2, ceil_mode=True), (2, 3, 9, 9)),
    ("maxpool1d", lambda: nn.MaxPool1d(3, stride=2), (2, 3, 11)),
    ("maxpool3d", lambda: nn.MaxPool3d(2), (1, 2, 4, 6, 6)),
    ("avgpool2d_pad_nocount", lambda: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False), (2, 3, 9, 9)),
    ("avgpool2d_ceil_divisor", lambda: nn.AvgPool2d(3, stride=2, ceil_mode=True, divisor_override=4), (2, 3, 8, 8)),
    ("avgpool1d", lambda: nn.AvgPool1d(4, stride=2, padding=1), (2, 3, 11)),
    ("avgpool3d", lambda: nn.AvgPool3d(2), (1, 2, 4, 6, 6)),
    ("adaptive_avgpool2d", lambda: nn.AdaptiveAvgPool2d((3, 5)), (2, 3, 10, 11)),
    ("adaptive_maxpool2d", lambda: nn.AdaptiveMaxPool2d((3, 5)), (2, 3, 10, 11)),
    ("adaptive_avgpool1d", lambda: nn.AdaptiveAvgPool1d(7), (2, 3, 20)),
    ("adaptive_maxpool1d", lambda: nn.AdaptiveMaxPool1d(7), (2, 3, 20)),
    ("lppool2d", lambda: nn.LPPool2d(2, 2), (2, 3, 8, 8)),
    ("pixel_shuffle", lambda: nn.PixelShuffle(2), (2, 8, 4, 4)),
    ("upsample_nearest", lambda: nn.Upsample(scale_factor=2, mode="nearest"), (2, 3, 5, 5)),
    ("upsample_nearest_exact", lambda: nn.Upsample(size=(7, 9), mode="nearest-exact"), (2, 3, 5, 5)),
    ("upsample_bilinear", lambda: nn.Upsample(scale_factor=2, mode="bilinear"), (2, 3, 5, 5)),
    ("upsample_bilinear_ac", lambda: nn.Upsample(size=(7, 9), mode="bilinear", align_corners=True), (2, 3, 5, 5)),
    ("upsample_bicubic", lambda: nn.Upsample(scale_factor=2, mode="bicubic"), (2, 3, 5, 5)),
    ("upsample_linear1d", lambda: nn.Upsample(scale_factor=3, mode="linear"), (2, 3, 5)),
    ("upsample_trilinear", lambda: nn.Upsample(scale_factor=2, mode="trilinear"), (1, 2, 3, 3, 3)),
    ("upsample_nearest1d", lambda: nn.Upsample(size=13, mode="nearest"), (2, 3, 5)),
    ("downsample_area", lambda: (lambda x: F.interpolate(x, size=(3, 4), mode="area")), (2, 3, 9, 10)),
    ("downsample_bilinear_aa", lambda: (lambda x: F.interpolate(x, size=(3, 4), mode="bilinear", antialias=True)), (2, 3, 9, 10)),
    ("fold_unfold", lambda: nn.Sequential(nn.Unfold(3, padding=1, stride=2), nn.Fold((7, 7), 3, padding=1, stride=2)), (2, 3, 7, 7)),
    ("reflection_pad2d", lambda: nn.ReflectionPad2d((1, 2, 2, 1)), (2, 3, 5, 5)),
    ("replication_pad2d", lambda: nn.ReplicationPad2d((1, 2, 2, 1)), (2, 3, 5, 5)),
    ("circular_pad1d", lambda: nn.CircularPad1d(2), (2, 3, 5)),
]:
    def _mk(make=make):
        r = make()
        return r if isinstance(r, nn.Module) else _Wrap(r)
    case(nm)(lambda dev, mk=_mk, shape=shape: mod_grads(mk, rnd(*shape), dev))


class _Wrap(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


@case("grid_sample_bwd")
def _(dev):
    x = leaf(rnd(2, 3, 6, 6), dev)
    grid = leaf(rnd(2, 4, 5, 2, seed=1).tanh(), dev)
    o = F.grid_sample(x, grid, align_corners=False)
    (o ** 2).sum().backward()
    return [o, x.grad, grid.grad]


@case("affine_grid_bwd")
def _(dev):
    th = leaf(rnd(2, 2, 3), dev)
    g = F.affine_grid(th, (2, 3, 4, 5), align_corners=False)
    (g ** 2).sum().backward()
    return [g, th.grad]


@case("maxpool2d_ties_bwd")
def _(dev):
    x = leaf(torch.ones(1, 1, 4, 4), dev)
    o = F.max_pool2d(x, 2, stride=1)
    o.sum().backward()
    return [o, x.grad]


@case("max_unpool2d")
def _(dev):
    x = leaf(rnd(1, 2, 6, 6), dev)
    o, ind = F.max_pool2d(x, 2, return_indices=True)
    u = F.max_unpool2d(o, ind, 2)
    (u * rnd(1, 2, 6, 6, seed=1).to(dev)).sum().backward()
    return [u, x.grad, ind.float()]


# ---------------- attention ----------------
@case("sdpa_float_mask_bwd")
def _(dev):
    q, k, v = (leaf(rnd(2, 4, 8, 16, seed=i), dev) for i in range(3))
    mask = rnd(2, 1, 8, 8, seed=5).to(dev)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    (o ** 2).sum().backward()
    return [o, q.grad, k.grad, v.grad]


@case("sdpa_gqa_bwd")
def _(dev):
    q = leaf(rnd(2, 8, 8, 16), dev)
    k, v = (leaf(rnd(2, 2, 8, 16, seed=i), dev) for i in (1, 2))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    (o ** 2).sum().backward()
    return [o, q.grad, k.grad, v.grad]


@case("sdpa_scale_noncontig_bwd")
def _(dev):
    q, k, v = (leaf(rnd(2, 8, 4, 16, seed=i), dev) for i in range(3))
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.3)
    (o ** 2).sum().backward()
    return [o, q.grad, k.grad, v.grad]


@case("sdpa_q_ne_kv_len_causal")
def _(dev):
    q = leaf(rnd(1, 2, 5, 16), dev)
    k, v = (leaf(rnd(1, 2, 9, 16, seed=i), dev) for i in (1, 2))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    (o ** 2).sum().backward()
    return [o, q.grad, k.grad, v.grad]


@case("sdpa_fully_masked_row")
def _(dev):
    q, k, v = (leaf(rnd(1, 2, 4, 8, seed=i), dev) for i in range(3))
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[1] = False
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(dev))
    o.nan_to_num(0.0).sum().backward()
    return [o.nan_to_num(123.0), q.grad.nan_to_num(123.0), v.grad.nan_to_num(123.0)]


@case("sdpa_dropout_stats")
def _(dev):
    # dropout is random; compare statistics: mean of output with v=ones should be ~1 (scaled), fraction of zeroed weights ~p
    torch.manual_seed(0)
    q = torch.zeros(4, 4, 64, 64, device=dev)
    k = torch.zeros(4, 4, 64, 64, device=dev)
    v = torch.eye(64, device=dev).expand(4, 4, 64, 64).contiguous()
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.3)
    zero_frac = (o == 0).float().mean()
    return [zero_frac.round(decimals=1).cpu(), (o.mean() * 64).round(decimals=1).cpu()]


@case("mha_module_bwd")
def _(dev):
    torch.manual_seed(0)
    m = nn.MultiheadAttention(16, 4, batch_first=True).to(dev)
    x = leaf(rnd(2, 6, 16), dev)
    kpm = torch.tensor([[False] * 6, [False] * 4 + [True] * 2]).to(dev)
    o, w = m(x, x, x, key_padding_mask=kpm, need_weights=True)
    (o ** 2).sum().backward()
    return [o, w, x.grad] + [p.grad for p in m.parameters()]


@case("transformer_encoder_layer_bwd")
def _(dev):
    torch.manual_seed(0)
    m = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True, norm_first=True).to(dev)
    x = leaf(rnd(2, 6, 16), dev)
    o = m(x, src_mask=nn.Transformer.generate_square_subsequent_mask(6, device=dev), is_causal=True)
    (o ** 2).sum().backward()
    return [o, x.grad] + [p.grad for p in m.parameters()]


# ---------------- RNN ----------------
@case("gru_bidir_2layer_bwd")
def _(dev):
    torch.manual_seed(0)
    m = nn.GRU(6, 8, num_layers=2, bidirectional=True, batch_first=True).to(dev)
    x = leaf(rnd(3, 5, 6), dev)
    o, h = m(x)
    ((o ** 2).sum() + h.sum()).backward()
    return [o, h, x.grad] + [p.grad for p in m.parameters()]


@case("lstm_packed_bwd")
def _(dev):
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    torch.manual_seed(0)
    m = nn.LSTM(6, 8, batch_first=True).to(dev)
    x = leaf(rnd(3, 5, 6), dev)
    p = pack_padded_sequence(x, torch.tensor([5, 3, 2]), batch_first=True)
    o, (h, c) = m(p)
    o, _ = pad_packed_sequence(o, batch_first=True)
    ((o ** 2).sum() + h.sum()).backward()
    return [o, h, c, x.grad] + [p.grad for p in m.parameters()]


@case("rnn_tanh_proj_lstm")
def _(dev):
    torch.manual_seed(0)
    m = nn.LSTM(6, 8, proj_size=4, batch_first=True).to(dev)
    x = leaf(rnd(3, 5, 6), dev)
    o, (h, c) = m(x)
    (o ** 2).sum().backward()
    return [o, x.grad] + [p.grad for p in m.parameters()]


# ---------------- autograd mechanics ----------------
@case("inplace_view_autograd")
def _(dev):
    x = leaf(rnd(4, 6), dev)
    y = x * 1.0
    v = y[:, 1:4]
    v.mul_(2.0)
    v.add_(1.0)
    y.t()[2].mul_(-1.5)
    y.view(-1)[::5].add_(3.0)
    (y ** 2).sum().backward()
    return [y, x.grad]


@case("inplace_view_chain_unsqueeze")
def _(dev):
    x = leaf(rnd(4, 6), dev)
    y = x.clone()
    y.unsqueeze(0).squeeze(0).transpose(0, 1)[1:3].mul_(4.0)
    y.diagonal().zero_()
    y.sum().backward()
    return [y, x.grad]


@case("inplace_on_leaf_view_nograd")
def _(dev):
    torch.manual_seed(0)
    lin = nn.Linear(4, 4).to(dev)
    with torch.no_grad():
        lin.weight[:, 0].fill_(0.5)
        lin.weight.diagonal().mul_(3.0)
        lin.bias[1:3] += 2.0
    x = leaf(rnd(3, 4), dev)
    lin(x).sum().backward()
    return [lin.weight, lin.bias, lin.weight.grad, x.grad]


@case("out_variant_into_view")
def _(dev):
    a = rnd(4, 6).to(dev)
    out = torch.zeros(6, 8, device=dev)
    torch.add(a.t(), 1.0, out=out[:, 2:6])
    torch.mul(a[:2].t(), 3.0, out=out[:, :2])
    torch.matmul(a.t(), a[:, :2], out=out[:, 6:8])
    return [out]


@case("copy_into_noncontig_view")
def _(dev):
    a = torch.zeros(4, 6, device=dev)
    a.t()[1:4].copy_(rnd(3, 4).to(dev))
    a[:, ::2] = rnd(4, 3, seed=1).to(dev)
    a[1:3, 1::2].add_(5.0)
    return [a]


@case("inplace_mutate_nonfirst_args")
def _(dev):
    # ops that mutate an argument other than self
    out = []
    x = rnd(4, 6).to(dev)
    vals = torch.empty(4, device=dev)
    idxs = torch.empty(4, dtype=torch.long, device=dev)
    torch.max(x, 1, out=(vals, idxs))
    out += [vals, idxs.float()]
    s, i = torch.empty(4, 6, device=dev), torch.empty(4, 6, dtype=torch.long, device=dev)
    torch.sort(x, 1, out=(s, i))
    out += [s, i.float()]
    tv, ti = torch.empty(4, 2, device=dev), torch.empty(4, 2, dtype=torch.long, device=dev)
    torch.topk(x, 2, out=(tv, ti))
    out += [tv, ti.float()]
    m, v = torch.empty(6, device=dev), torch.empty(6, device=dev)
    torch.var_mean(x, 0, out=(v, m)) if hasattr(torch, "var_mean") and False else None
    cm, ci = torch.empty(4, 6, device=dev), torch.empty(4, 6, dtype=torch.long, device=dev)
    torch.cummax(x, 1, out=(cm, ci))
    out += [cm, ci.float()]
    fr, ex = torch.empty(4, 6, device=dev), torch.empty(4, 6, dtype=torch.int32, device=dev)
    torch.frexp(x, out=(fr, ex))
    out += [fr, ex.float()]
    return out


@case("native_layer_norm_outputs")
def _(dev):
    x = rnd(4, 8).to(dev)
    o, mean, rstd = torch.native_layer_norm(x, (8,), torch.ones(8, device=dev), torch.zeros(8, device=dev), 1e-5)
    return [o, mean, rstd]


@case("native_group_norm_outputs")
def _(dev):
    x = rnd(2, 8, 5).to(dev)
    o, mean, rstd = torch.native_group_norm(x, torch.ones(8, device=dev), torch.zeros(8, device=dev), 2, 8, 5, 2, 1e-5)
    return [o, mean, rstd]


@case("batch_norm_functional_running_stats")
def _(dev):
    rm, rv = torch.zeros(4, device=dev), torch.ones(4, device=dev)
    x = rnd(8, 4).to(dev)
    o = F.batch_norm(x, rm, rv, training=True, momentum=0.1)
    return [o, rm, rv]


@case("checkpoint_sequential")
def _(dev):
    from torch.utils.checkpoint import checkpoint_sequential
    torch.manual_seed(0)
    m = nn.Sequential(*[nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(4)]).to(dev)
    x = leaf(rnd(3, 8), dev)
    o = checkpoint_sequential(m, 2, x, use_reentrant=False)
    (o ** 2).sum().backward()
    return [o, x.grad] + [p.grad for p in m.parameters()]


@case("checkpoint_with_dropout_rng")
def _(dev):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    lin = nn.Linear(16, 16).to(dev)
    x = leaf(rnd(8, 16), dev)

    def f(x):
        return F.dropout(lin(x), 0.5, training=True)

    torch.manual_seed(123)
    o = checkpoint(f, x, use_reentrant=False, preserve_rng_state=True)
    o.sum().backward()
    g_ck = lin.weight.grad.clone()
    lin.weight.grad = None
    # recompute with same seed without checkpoint: grads must match
    torch.manual_seed(123)
    o2 = f(x)
    o2.sum().backward()
    return [(o - o2).abs().max().cpu().reshape(1), (g_ck - lin.weight.grad).abs().max().cpu().reshape(1)]


@case("checkpoint_reentrant_dropout_rng")
def _(dev):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    lin = nn.Linear(16, 16).to(dev)
    x = leaf(rnd(8, 16), dev)

    def f(x):
        return F.dropout(lin(x), 0.5, training=True)

    torch.manual_seed(123)
    o = checkpoint(f, x, use_reentrant=True, preserve_rng_state=True)
    o.sum().backward()
    g_ck = lin.weight.grad.clone()
    lin.weight.grad = None
    torch.manual_seed(123)
    o2 = f(x)
    o2.sum().backward()
    return [(o - o2).abs().max().cpu().reshape(1), (g_ck - lin.weight.grad).abs().max().cpu().reshape(1)]


@case("dropout_variants_rng_state")
def _(dev):
    torch.manual_seed(5)
    a = F.dropout(torch.ones(1000, device=dev), 0.4)
    st = torch.get_rng_state() if dev == "cpu" else torch.neuron.get_rng_state()
    b = F.dropout(torch.ones(1000, device=dev), 0.4)
    if dev == "cpu":
        torch.set_rng_state(st)
    else:
        torch.neuron.set_rng_state(st)
    c = F.dropout(torch.ones(1000, device=dev), 0.4)
    return [(b == c).all().float().cpu().reshape(1), ((a == 0).float().mean() * 10).round().cpu().reshape(1)]


@case("func_grad_vmap")
def _(dev):
    from torch.func import functional_call, grad, vmap
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).to(dev)
    params = {k: v.detach() for k, v in m.named_parameters()}
    x = rnd(5, 4).to(dev)
    y = rnd(5, 1, seed=1).to(dev)

    def loss(p, xi, yi):
        return ((functional_call(m, p, (xi.unsqueeze(0),)) - yi) ** 2).sum()

    per = vmap(grad(loss), in_dims=(None, 0, 0))(params, x, y)
    return [per[k] for k in sorted(per)]


@case("func_jacrev_jvp_hessian")
def _(dev):
    from torch.func import hessian, jacfwd, jacrev, jvp
    x = rnd(4).to(dev)
    f = lambda x: (x.sin() * x.cumsum(0)).sum()
    g = lambda x: x.tanh() * x
    return [
        jacrev(g)(x),
        jacfwd(g)(x),
        jvp(g, (x,), (torch.ones_like(x),))[1],
        hessian(f)(x),
    ]


@case("func_vmap_ops")
def _(dev):
    from torch.func import vmap
    x = rnd(3, 5, 6).to(dev)
    w = rnd(6, 4, seed=1).to(dev)
    return [
        vmap(lambda a: a @ w)(x),
        vmap(lambda a: F.softmax(a, -1))(x),
        vmap(lambda a: F.layer_norm(a, (6,)))(x),
        vmap(lambda a: a.sum(0))(x),
        vmap(vmap(lambda a: a.max()))(x),
    ]


@case("autograd_functional_jacobian")
def _(dev):
    x = rnd(3).to(dev)
    return [torch.autograd.functional.jacobian(lambda a: a.exp() * a.sum(), x),
            torch.autograd.functional.hessian(lambda a: (a ** 3).sum(), x)]


@case("custom_autograd_function")
def _(dev):
    class Sq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, k):
            ctx.save_for_backward(x)
            ctx.k = k
            return x * x * k

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * 2 * x * ctx.k, None

    x = leaf(rnd(5), dev)
    y = Sq.apply(x, 3.0)
    y.sum().backward()
    return [y, x.grad]


@case("saved_tensor_hooks_cpu_offload")
def _(dev):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).to(dev)
    x = leaf(rnd(3, 8), dev)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        o = m(x)
    (o ** 2).sum().backward()
    return [o, x.grad] + [p.grad for p in m.parameters()]


@case("grad_of_noncontig_param")
def _(dev):
    torch.manual_seed(0)
    w = nn.Parameter(rnd(6, 4).to(dev).t())  # non-contiguous param
    x = rnd(3, 6).to(dev)
    o = x @ w.t().t().t()
    o.sum().backward()
    opt = torch.optim.AdamW([w], lr=0.1)
    opt.step()
    return [o, w.grad, w.detach()]


@case("set_grad_manually_and_accumulate")
def _(dev):
    torch.manual_seed(0)
    p = nn.Parameter(rnd(4, 4).to(dev))
    p.grad = torch.ones(4, 4, device=dev)
    (p * 2).sum().backward()
    g1 = p.grad.clone()
    p.grad[::2].zero_()
    (p ** 2).sum().backward()
    return [g1, p.grad]


@case("register_hook_modify_grad")
def _(dev):
    x = leaf(rnd(4, 4), dev)
    y = x * 3
    y.register_hook(lambda g: g * 0.5 + 1)
    x.register_post_accumulate_grad_hook(lambda p: p.grad.mul_(2) and None)
    (y ** 2).sum().backward()
    return [x.grad]


@case("anomaly_detect_nan")
def _(dev):
    x = leaf(torch.tensor([0.0, 1.0, 4.0]), dev)
    with torch.autograd.detect_anomaly(check_nan=True):
        try:
            torch.sqrt(x).sum().backward()
            r = 0.0
        except RuntimeError:
            r = 1.0
    return [torch.tensor([r])]


# ---------------- autocast ----------------
@case("autocast_bf16_dtypes")
def _(dev):
    torch.manual_seed(0)
    a = rnd(4, 8).to(dev)
    b = rnd(8, 4, seed=1).to(dev)
    with torch.autocast(device_type=dev if dev == "cpu" else "neuron", dtype=torch.bfloat16):
        mm = a @ b
        sm = F.softmax(mm, -1)
        ln = F.layer_norm(mm, (4,))
        ls = F.log_softmax(mm, -1)
        ce = F.cross_entropy(mm, torch.tensor([0, 1, 2, 3], device=dev))
        lin = F.linear(a, b.t())
        conv = F.conv1d(a.t().unsqueeze(0), b.t().unsqueeze(-1))
        s = mm.sum()
        add = a + 1
    names = ["mm", "sm", "ln", "ls", "ce", "lin", "conv", "s", "add"]
    vals = [mm, sm, ln, ls, ce, lin, conv, s, add]
    code = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
    return [torch.tensor([code[v.dtype] for v in vals], dtype=torch.float32)]


@case("autocast_fp16_dtypes")
def _(dev):
    a = rnd(4, 8).to(dev)
    b = rnd(8, 4, seed=1).to(dev)
    with torch.autocast(device_type=dev if dev == "cpu" else "neuron", dtype=torch.float16):
        mm = a @ b
        sm = F.softmax(mm, -1)
        lin = F.linear(a, b.t())
    code = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
    return [torch.tensor([code[v.dtype] for v in (mm, sm, lin)], dtype=torch.float32)]


@case("autocast_nested_disable")
def _(dev):
    a = rnd(4, 8).to(dev)
    b = rnd(8, 4, seed=1).to(dev)
    dt = dev if dev == "cpu" else "neuron"
    with torch.autocast(device_type=dt, dtype=torch.bfloat16):
        with torch.autocast(device_type=dt, enabled=False):
            mm = a @ b
        mm2 = a @ b
    return [torch.tensor([mm.dtype == torch.float32, mm2.dtype == torch.bfloat16], dtype=torch.float32)]


@case("autocast_bf16_backward_param_grad_dtype")
def _(dev):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(dev)
    x = rnd(5, 8).to(dev)
    with torch.autocast(device_type=dev if dev == "cpu" else "neuron", dtype=torch.bfloat16):
        o = m(x)
        loss = (o.float() ** 2).mean()
    loss.backward()
    return [loss.float()] + [p.grad for p in m.parameters()] + [torch.tensor([float(p.grad.dtype == torch.float32) for p in m.parameters()])]


# ---------------- misc training utilities ----------------
@case("clip_grad_norm_error_if_nonfinite")
def _(dev):
    p = nn.Parameter(torch.ones(4, device=dev))
    p.grad = torch.tensor([1.0, float("inf"), 0.0, 1.0], device=dev)
    try:
        torch.nn.utils.clip_grad_norm_([p], 1.0, error_if_nonfinite=True)
        r = 0.0
    except RuntimeError:
        r = 1.0
    return [torch.tensor([r])]


@case("clip_grad_norm_nan")
def _(dev):
    p = nn.Parameter(torch.ones(4, device=dev))
    p.grad = torch.tensor([1.0, float("nan"), 0.0, 1.0], device=dev)
    n = torch.nn.utils.clip_grad_norm_([p], 1.0)
    return [n.isnan().float().cpu().reshape(1), p.grad.isnan().float()]


@case("clip_grad_norm_mixed_dtype")
def _(dev):
    p1 = nn.Parameter(torch.ones(4, device=dev))
    p2 = nn.Parameter(torch.ones(4, device=dev, dtype=torch.bfloat16))
    p1.grad = torch.arange(4.0, device=dev)
    p2.grad = torch.arange(4.0, device=dev).to(torch.bfloat16) * 3
    n = torch.nn.utils.clip_grad_norm_([p1, p2], 1.0)
    return [n, p1.grad, p2.grad.float()]


@case("parameters_to_vector_roundtrip")
def _(dev):
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).to(dev)
    v = parameters_to_vector(m.parameters())
    vector_to_parameters(v * 2, m.parameters())
    return [v] + [p.detach() for p in m.parameters()]


@case("swa_utils")
def _(dev):
    from torch.optim.swa_utils import AveragedModel, SWALR, get_ema_multi_avg_fn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2)).to(dev)
    ema = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    swa = AveragedModel(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = SWALR(opt, swa_lr=0.05, anneal_epochs=2)
    for i in range(4):
        opt.zero_grad()
        m(rnd(6, 4, seed=i).to(dev)).pow(2).sum().backward()
        opt.step()
        sch.step()
        ema.update_parameters(m)
        swa.update_parameters(m)
    return [p.detach() for p in ema.parameters()] + [p.detach() for p in swa.parameters()] + [torch.tensor([opt.param_groups[0]["lr"]])]


@case("prune_l1_unstructured")
def _(dev):
    import torch.nn.utils.prune as prune
    torch.manual_seed(0)
    m = nn.Linear(6, 4).to(dev)
    prune.l1_unstructured(m, "weight", amount=0.5)
    x = leaf(rnd(3, 6), dev)
    m(x).sum().backward()
    return [m.weight, m.weight_orig.grad, x.grad]


@case("cumsum_cumprod_logcumsumexp_bwd")
def _(dev):
    x = leaf(rnd(4, 6) * 0.5, dev)
    o = x.cumsum(1) + x.cumprod(1) + x.logcumsumexp(1) + x.cummax(1)[0]
    (o ** 2).sum().backward()
    return [o, x.grad]


@case("sort_topk_kthvalue_bwd")
def _(dev):
    x = leaf(rnd(4, 7), dev)
    s, _ = x.sort(1, descending=True)
    t, _ = x.topk(3, 1)
    k, _ = x.kthvalue(2, 1)
    m, _ = x.median(1)
    (s[:, :2].sum() + t.pow(2).sum() + k.sum() * 3 + m.sum()).backward()
    return [s, t, k, m, x.grad]


@case("norm_ops_bwd")
def _(dev):
    x = leaf(rnd(4, 6), dev)
    o = (torch.linalg.vector_norm(x, 3, dim=1).sum() + torch.linalg.matrix_norm(x).sum()
         + x.norm(p=1) + F.normalize(x, dim=1).pow(2).sum() + torch.cdist(x, x * 2).sum()
         + F.pairwise_distance(x, x.flip(0)).sum() + F.cosine_similarity(x, x.flip(0)).sum())
    o.backward()
    return [o, x.grad]


@case("activations_bwd")
def _(dev):
    x = leaf(rnd(64) * 3, dev)
    acts = [F.silu, F.mish, F.hardswish, F.hardsigmoid, F.elu, F.selu, F.celu, F.softplus,
            lambda a: F.gelu(a, approximate="tanh"), F.gelu, F.leaky_relu, F.relu6, F.hardtanh,
            F.logsigmoid, F.softsign, F.tanhshrink, F.softshrink, F.hardshrink,
            lambda a: F.glu(a.view(8, 8)), lambda a: F.prelu(a.view(8, 8), torch.full((8,), 0.2, device=dev)),
            lambda a: F.rrelu(a, training=False), F.sigmoid, torch.erf, torch.erfinv if False else torch.expm1,
            torch.log1p, lambda a: a.abs().sqrt(), lambda a: a.abs().rsqrt(), torch.special.xlogy if False else torch.atan,
            torch.sinh, torch.asinh]
    out = []
    for f in acts:
        x.grad = None
        y = f(x)
        y.sum().backward()
        out += [y, x.grad.nan_to_num(7.0, 7.0, 7.0)]
    return out


@case("matmul_variants_bwd")
def _(dev):
    a = leaf(rnd(2, 3, 4, 5), dev)
    b = leaf(rnd(5, 6, seed=1), dev)
    c = leaf(rnd(3, 6, 2, seed=2), dev)
    o = (a @ b) @ c
    e = torch.einsum("bhij,hjk->bhik", a @ b, c)
    bmm = torch.baddbmm(torch.zeros(3, 4, 2, device=dev), a[0] @ b, c, beta=0.5, alpha=2.0)
    addmm = torch.addmm(torch.ones(4, 6, device=dev), a[0, 0], b, beta=2.0)
    (o.sum() + e.pow(2).sum() + bmm.sum() + addmm.sum()).backward()
    return [o, e, bmm, addmm, a.grad, b.grad, c.grad]


@case("linalg_bwd")
def _(dev):
    a = leaf(rnd(4, 4) + torch.eye(4) * 4, dev)
    o = (torch.linalg.inv(a).sum() + torch.linalg.det(a) * 0.01 + torch.linalg.slogdet(a)[1]
         + torch.linalg.solve(a, torch.ones(4, 1, device=dev)).sum()
         + torch.linalg.cholesky(a @ a.mT).sum() + torch.linalg.qr(a)[1].abs().sum()
         + torch.linalg.svdvals(a).sum() + torch.linalg.eigvalsh(a + a.mT).sum())
    o.backward()
    return [o, a.grad]


def main():
    sel = sys.argv[1:]
    names = [n for n in CASES if not sel or any(s in n for s in sel)]
    summary = []
    for name in names:
        fn, atol, rtol = CASES[name]
        try:
            ref = [t.detach().float().cpu() if t is not None else None for t in fn("cpu")]
            out = [t.detach().float().cpu() if t is not None else None for t in fn(DEV)]
            worst, status = 0.0, "OK"
            if len(ref) != len(out):
                status = "LEN"
            for i, (r, o) in enumerate(zip(ref, out)):
                if (r is None) != (o is None):
                    status = f"NONE@{i}"
                    break
                if r is None:
                    continue
                if r.shape != o.shape:
                    status = f"SHAPE@{i} {tuple(r.shape)} vs {tuple(o.shape)}"
                    break
                if not torch.allclose(r, o, atol=atol, rtol=rtol, equal_nan=True):
                    d = (r - o).abs().nan_to_num(1e9).max().item()
                    status = "MISMATCH"
                    worst = max(worst, d)
                    print(f"  [{name}] tensor {i} maxdiff {d:.4g}", flush=True)
            line = f"{status:10s} {name:45s} {worst:.4g}"
        except Exception as e:
            traceback.print_exc()
            line = f"ERROR      {name:45s} {type(e).__name__}: {str(e)[:250]}".replace("\n", " ")
        print("RESULT " + line, flush=True)
        summary.append(line)
    print("\n===== SUMMARY =====")
    print("\n".join(summary))
    print("fallback ops:", sorted(torch_neuronx.get_fallback_ops()) if hasattr(torch_neuronx, "get_fallback_ops") else "n/a")


if __name__ == "__main__":
    main()
