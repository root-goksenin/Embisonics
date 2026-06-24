import math
import random
import torch
from torch import Tensor


def _sample_block(Nf, Nt, scale, aspect, rng):
    area = Nf * Nt
    tgt_area = rng.uniform(*scale) * area
    a = rng.uniform(*aspect)
    h = int(round(math.sqrt(tgt_area * a)))
    w = int(round(math.sqrt(tgt_area / a)))
    h = max(1, min(h, Nf))
    w = max(1, min(w, Nt))
    top = rng.randint(0, Nf - h)
    left = rng.randint(0, Nt - w)
    idx = []
    for f in range(top, top + h):
        for t in range(left, left + w):
            idx.append(f * Nt + t)          # freq-major, matches token order
    return set(idx)


def sample_ijepa_masks(grid, n_targets=4,
                       target_scale=(0.15, 0.2), target_aspect=(0.75, 1.5),
                       context_scale=(0.85, 1.0), seed=None):
    """One large context block + n_targets target blocks (I-JEPA, batch-shared).

    Returns:
      context_idx : LongTensor (N_ctx,)   context MINUS all target patches
      target_idx  : list of LongTensor    one per target block (disjoint from ctx)
    """
    Nf, Nt = grid
    rng = random.Random(seed)
    targets = [_sample_block(Nf, Nt, target_scale, target_aspect, rng)
               for _ in range(n_targets)]
    union_t = set().union(*targets)
    ctx = _sample_block(Nf, Nt, context_scale, (1.0, 1.0), rng)
    ctx = ctx - union_t                              # enforce disjointness
    if len(ctx) == 0:                                # degenerate guard
        ctx = set(range(Nf * Nt)) - union_t
    context_idx = torch.tensor(sorted(ctx), dtype=torch.long)
    target_idx = [torch.tensor(sorted(t), dtype=torch.long) for t in targets]
    return context_idx, target_idx


if __name__ == "__main__":
    grid = (8, 25)                                   # Nf=128/16, Nt=100/4
    P = grid[0] * grid[1]
    for s in range(3):
        ctx, tgts = sample_ijepa_masks(grid, seed=s)
        union_t = torch.cat(tgts)
        # disjointness: no context patch is in any target block
        overlap = len(set(ctx.tolist()) & set(union_t.tolist()))
        print(f"seed {s}: |ctx|={len(ctx):3d} ({len(ctx)/P:.0%})  "
              f"n_targets={len(tgts)}  |union_tgt|={len(set(union_t.tolist())):3d}  "
              f"ctx∩tgt={overlap}")