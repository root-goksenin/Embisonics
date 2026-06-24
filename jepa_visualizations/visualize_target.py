"""Visualize the Route A cross-entropy target: the categorical distribution over the
sphere grid that the DOA head regresses to, for scenes with N simultaneous sources.

Renders, per scene, the *actual* RouteATarget output for ONE patch:
  - left : grid points on the unit sphere, colored/sized by probability q_g
  - right: the same q_g in azimuth-elevation, with true DOAs marked
"""
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from route_a_target import RouteATarget


def s_xyz(az_deg, el_deg):
    ph, th = math.radians(az_deg), math.radians(el_deg)
    return torch.tensor([math.cos(ph) * math.cos(th),
                         math.sin(ph) * math.cos(th),
                         math.sin(th)], dtype=torch.float32)


def to_yzx(s):                      # canonical (x,y,z) -> extractor order (y,z,x)
    return torch.stack([s[1], s[2], s[0]])


def intensity_tile(doas, mags, fshape, tshape):
    """Build one (1,3,fshape,tshape) intensity patch (in extractor (y,z,x) order)
    with N sources placed in contiguous mel sub-bands (W-disjoint approximation)."""
    M, T = fshape, tshape
    I = torch.zeros(1, 3, M, T)
    bands = np.array_split(np.arange(M), len(doas))
    for (az, el), mag, band in zip(doas, mags, bands):
        iv = to_yzx(s_xyz(az, el)) * mag
        I[0, :, band, :] = iv.view(3, 1, 1)
    return I


def grid_azel(G):
    x, y, z = G[:, 0], G[:, 1], G[:, 2]
    az = torch.rad2deg(torch.atan2(y, x))
    el = torch.rad2deg(torch.asin(z.clamp(-1, 1)))
    return az.numpy(), el.numpy()


def visualize(scenes, fshape=16, tshape=8, n_grid=256, kappa=40.0, out="target_dists.png"):
    tgt = RouteATarget(n_grid=n_grid, kappa=kappa, fshape=fshape, tshape=tshape)
    G = tgt.G
    az_g, el_g = grid_azel(G)

    n = len(scenes)
    fig = plt.figure(figsize=(13, 4.6 * n))
    for i, (title, doas, mags) in enumerate(scenes):
        I = intensity_tile(doas, mags, fshape, tshape)
        q, Wp = tgt(I)
        qg = q[0, 0].numpy()
        qn = qg / qg.max()

        # ---- 3D sphere ----
        ax = fig.add_subplot(n, 2, 2 * i + 1, projection="3d")
        order = np.argsort(qg)                       # draw faint points first
        ax.scatter(G[order, 0], G[order, 1], G[order, 2],
                   c=qg[order], cmap="magma", s=6 + 120 * qn[order],
                   vmin=0, vmax=qg.max(), depthshade=True, linewidths=0)
        for (az, el) in doas:
            s = s_xyz(az, el) * 1.08
            ax.scatter(*s, marker="*", s=260, c="cyan", edgecolors="k",
                       linewidths=0.6, depthshade=False)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=35)
        ax.set_xlabel("x (front)"); ax.set_ylabel("y (left)"); ax.set_zlabel("z (up)")
        ax.set_title(f"{title}\nentropy={-(qg*np.log(qg+1e-12)).sum():.2f} nats, "
                     f"Wp={Wp.item():.0f}", fontsize=10)

        # ---- azimuth-elevation ----
        ax2 = fig.add_subplot(n, 2, 2 * i + 2)
        sc = ax2.scatter(az_g, el_g, c=qg, cmap="magma", s=24 + 90 * qn,
                         vmin=0, vmax=qg.max(), linewidths=0)
        for (az, el) in doas:
            ax2.scatter(az, el, marker="*", s=300, facecolors="none",
                        edgecolors="cyan", linewidths=1.8)
            ax2.annotate(f"({az:+.0f},{el:+.0f})", (az, el),
                         textcoords="offset points", xytext=(6, 6),
                         fontsize=8, color="cyan")
        ax2.set_xlim(-180, 180); ax2.set_ylim(-90, 90)
        ax2.set_xticks(range(-180, 181, 60)); ax2.set_yticks(range(-90, 91, 30))
        ax2.grid(alpha=0.25); ax2.set_xlabel("azimuth (deg)"); ax2.set_ylabel("elevation (deg)")
        ax2.set_title("CE target q over grid (cyan ★ = true DOA)", fontsize=10)
        fig.colorbar(sc, ax=ax2, label="probability", fraction=0.046)

    fig.suptitle(f"Route A target  |  patch {fshape}×{tshape}, {n_grid} grid pts, κ={kappa}",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    scenes = [
        ("1 source",                 [(30, 20)],                          [1.0]),
        ("2 sources (equal energy)", [(-70, 0), (70, 10)],                [1.0, 1.0]),
        ("3 sources (equal energy)", [(-100, -15), (10, 35), (130, 5)],   [1.0, 1.0, 1.0]),
        ("2 sources (loud + soft, 3:1)", [(-45, 0), (90, 20)],            [1.0, 0.33]),
    ]
    visualize(scenes, fshape=16, tshape=8, n_grid=256, kappa=40.0,
              out="/home/claude/target_dists.png")