"""Visualize I-JEPA mask layout on the (Nf, Nt) token grid.

Shows, per seed: the context region (what CtxEnc sees), the 4 target blocks
(what the predictor must reconstruct; drawn as outlines, overlaps visible),
and unused cells (border punched out of context). Grid is freq-major:
token index = f*Nt + t, so f = idx // Nt, t = idx % Nt.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from masking import sample_ijepa_masks


def _grid_cat(ctx, tgts, Nf, Nt):
    cat = np.zeros((Nf, Nt), dtype=int)              # 0 = unused
    for i in ctx.tolist():
        cat[i // Nt, i % Nt] = 1                      # 1 = context
    for t in tgts:
        for i in t.tolist():
            cat[i // Nt, i % Nt] = 2                  # 2 = target (any block)
    return cat


def _bbox(idx, Nt):
    f = [i // Nt for i in idx.tolist()]
    t = [i % Nt for i in idx.tolist()]
    return min(f), max(f), min(t), max(t)


def visualize_masks(grid=(8, 25), seeds=(0, 1, 2, 3), n_targets=4,
                    out="masks.png"):
    Nf, Nt = grid
    P = Nf * Nt
    cmap = ListedColormap(["#e9e9ee", "#2a9d8f", "#f4a261"])   # unused / context / target
    blk_colors = ["#264653", "#e76f51", "#8338ec", "#1d3557", "#ff006e", "#06d6a0"]

    n = len(seeds)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 2.6 * nrow + 0.6),
                             squeeze=False)

    for k, seed in enumerate(seeds):
        ax = axes[k // ncol][k % ncol]
        ctx, tgts = sample_ijepa_masks(grid, n_targets=n_targets, seed=seed)
        cat = _grid_cat(ctx, tgts, Nf, Nt)

        ax.imshow(cat, origin="lower", cmap=cmap, vmin=0, vmax=2,
                  aspect="equal", interpolation="nearest")

        # outline each target block (overlaps show as crossing rectangles)
        for bi, t in enumerate(tgts):
            f0, f1, t0, t1 = _bbox(t, Nt)
            ax.add_patch(Rectangle((t0 - 0.5, f0 - 0.5), (t1 - t0 + 1), (f1 - f0 + 1),
                                   fill=False, edgecolor=blk_colors[bi % len(blk_colors)],
                                   lw=2.2, label=f"tgt {bi}"))

        # cell gridlines
        ax.set_xticks(np.arange(-0.5, Nt, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Nf, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.6)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(0, Nt, 4)); ax.set_yticks(range(0, Nf))
        ax.set_xlabel("time patch"); ax.set_ylabel("freq patch")

        u = torch.cat(tgts)
        nuni = len(set(u.tolist()))
        n_overlap = len(u) - nuni
        n_unused = P - len(ctx) - nuni
        ax.set_title(f"seed {seed}:  context {len(ctx)} ({len(ctx)/P:.0%})   "
                     f"targets {nuni} ({nuni/P:.0%})   unused {n_unused}   "
                     f"tgt-overlap {n_overlap}", fontsize=9.5)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # shared legend
    handles = [Rectangle((0, 0), 1, 1, color="#2a9d8f"),
               Rectangle((0, 0), 1, 1, color="#f4a261"),
               Rectangle((0, 0), 1, 1, color="#e9e9ee")]
    fig.legend(handles, ["context (encoder sees)", "target blocks (predict)",
                         "unused (punched out)"],
               loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"I-JEPA masks  |  grid {Nf}×{Nt} = {P} tokens, {n_targets} target blocks",
                 y=1.045, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    visualize_masks(grid=(8, 25), seeds=(0, 1, 2, 3), n_targets=4,
                    out="/home/claude/masks.png")