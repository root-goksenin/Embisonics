import numpy as np

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token_num=0):
    gf, gt = grid_size
    gh = np.arange(gf, dtype=np.float32)
    gw = np.arange(gt, dtype=np.float32)
    grid = np.meshgrid(gw, gh)               # w, h
    grid = np.stack(grid, axis=0)            # (2, gf, gt)
    grid = grid.reshape([2, 1, gf, gt])
    eh = _1d(embed_dim // 2, grid[1])        # freq-major
    ew = _1d(embed_dim // 2, grid[0])
    pe = np.concatenate([eh, ew], axis=1)    # (gf*gt, D)
    if cls_token_num > 0:
        pe = np.concatenate([np.zeros([cls_token_num, embed_dim]), pe], axis=0)
    return pe

def _1d(dim, pos):
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = pos.reshape(-1)[:, None] * omega[None, :]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)