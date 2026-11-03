"""Copy of nkilib add_tensor_kernel that writes alpha*data2 to a new tile (no in-place on t2)."""
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.experimental.foreach.foreach_utils import get_spmd_tiling_info

_F_SIZE_BF16 = 2048
_F_SIZE_F32 = 1024


@nki.jit
def add_tensor_kernel_fixed(data1, data2, alpha_tensor, numel):
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data1.shape, dtype=data1.dtype, buffer=nl.shared_hbm)
    alpha_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(alpha_sb, alpha_tensor[0:P_MAX, 0:1])
    f_size = _F_SIZE_F32 if data1.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t1 = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t2, data2.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        t2s = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(t2s, t2, nl.multiply, alpha_sb)
        res = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(res, t1, t2s, nl.add)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), res)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t1 = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t2, data2.ap([[1, tail], [1, 1]], tail_offset))
        t2s = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(t2s, t2, nl.multiply, alpha_sb[0:tail, 0:1])
        res = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(res, t1, t2s, nl.add)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), res)
    return out_hbm
