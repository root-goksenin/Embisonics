import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def fibonacci_sphere(n: int) -> Tensor:
    """n approximately-uniform unit vectors on S^2, in (x,y,z) = (front,left,up)."""
    i = torch.arange(n, dtype=torch.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    theta = phi * i
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y, z], dim=1).float()        # (n, 3)


class RouteATarget(nn.Module):
    """Per-patch angular distribution target (vMF mixture on a sphere grid).

    Input intensity_mel is the RAW mel active-intensity in the extractor's
    channel order (y, z, x)  -- because YZX = spec[:,1:] for ACN [W,Y,Z,X].
    We reorder to canonical (x,y,z) so the dot products against the (x,y,z)
    Fibonacci grid (and downstream STARSS23 ACCDOA) are consistent.

    Pipeline per call (B,3,M,T) -> q (B,P,|G|), Wp (B,P):
      1. reorder (y,z,x) -> (x,y,z)
      2. smooth <I> over a small TF tile (coherence average; required so the
         per-bin direction is meaningful and diffuse energy cancels)
      3. dir_tau = <I>/||<I>||,  conf w_tau = ||<I>||  ( = <e>*(1-psi) )
      4. splat each tile onto grid G with vMF(kappa), weight w_tau, sum
         within each (fshape x tshape) patch box -> q ; Wp = sum w_tau
      5. normalise q over the sphere.
    A patch spanning sources at different mel bands gets a genuinely
    multi-modal q; a single broadband mean would have averaged them.
    """

    # extractor order (y,z,x) -> (x,y,z)
    YZX_TO_XYZ = (2, 0, 1)

    def __init__(self, n_grid: int = 256, kappa: float = 40.0,
                 fshape: int = 16, tshape: int = 4,
                 tile_f: int = 3, tile_t: int = 3, eps: float = 1e-6):
        super().__init__()
        self.kappa = kappa
        self.fshape, self.tshape = fshape, tshape
        self.tile_f, self.tile_t = tile_f, tile_t
        self.eps = eps
        self.register_buffer("G", fibonacci_sphere(n_grid))   # (|G|,3)

    @torch.no_grad()
    def forward(self, intensity_mel: Tensor):
        B, C, M, T = intensity_mel.shape
        assert C == 3
        I = intensity_mel[:, self.YZX_TO_XYZ, :, :].float()   # -> (x,y,z)

        # 1) coherence smoothing over a TF tile (stride 1, 'same')
        Is = F.avg_pool2d(I, kernel_size=(self.tile_f, self.tile_t),
                          stride=1, padding=(self.tile_f // 2, self.tile_t // 2),
                          count_include_pad=False)
        Is = Is[..., :M, :T]

        Inorm = torch.linalg.norm(Is, dim=1)                  # (B,M,T) = ||<I>|| = conf
        dir = Is / Inorm.unsqueeze(1).clamp_min(self.eps)     # (B,3,M,T) unit

        # 2) vMF splat onto grid
        cos = torch.einsum("gd,bdmt->bgmt", self.G, dir)      # (B,|G|,M,T)
        vmf = torch.exp(self.kappa * cos)                     # unnormalised vMF
        wvmf = Inorm.unsqueeze(1) * vmf                        # weighted (B,|G|,M,T)

        # 3) pool into (fshape x tshape) patch boxes  (sum within box)
        area = self.fshape * self.tshape
        q = F.avg_pool2d(wvmf, kernel_size=(self.fshape, self.tshape),
                         stride=(self.fshape, self.tshape)) * area    # (B,|G|,Nf,Nt)
        Wp = F.avg_pool2d(Inorm.unsqueeze(1),
                          kernel_size=(self.fshape, self.tshape),
                          stride=(self.fshape, self.tshape)).squeeze(1) * area  # (B,Nf,Nt)

        Nf, Nt = q.shape[-2], q.shape[-1]
        # flatten freq-major then time  -> matches SpatialFrontEnd token order
        q = q.permute(0, 2, 3, 1).reshape(B, Nf * Nt, -1)     # (B,P,|G|)
        Wp = Wp.reshape(B, Nf * Nt)                           # (B,P)

        q = q / q.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return q, Wp


if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)
    fshape, tshape, M, T = 16, 8, 128, 200
    tgt = RouteATarget(n_grid=256, kappa=40.0, fshape=fshape, tshape=tshape)
    G = tgt.G

    def s_xyz(az, el):
        ph, th = math.radians(az), math.radians(el)
        return torch.tensor([math.cos(ph)*math.cos(th),
                             math.sin(ph)*math.cos(th),
                             math.sin(th)])

    def to_yzx(s):  # (x,y,z) -> extractor (y,z,x)
        return torch.stack([s[1], s[2], s[0]])

    # ---- single source: target must peak at grid cell nearest s ----
    s = s_xyz(30, 20)
    I = to_yzx(s).view(1, 3, 1, 1).expand(1, 3, M, T).clone() * 2.0
    q, Wp = tgt(I)
    peak = q[0, 0]                                   # patch 0
    g_pred = G[peak.argmax()]
    g_true = G[(G @ s).argmax()]
    print("single-source: cos(peak_grid, s)      =", float(g_pred @ s))
    print("single-source: cos(peak_grid, nearestG)=", float(g_pred @ g_true))
    print("single-source: q entropy (nats)        =", float(-(peak*peak.clamp_min(1e-9).log()).sum()))

    # ---- two sources at different mel bands within one patch ----
    s1, s2 = s_xyz(-60, 0), s_xyz(60, 0)
    I2 = torch.zeros(1, 3, M, T)
    I2[:, :, :fshape // 2] = to_yzx(s1).view(1, 3, 1, 1) * 2.0    # lower half of patch-0 band
    I2[:, :, fshape // 2:fshape] = to_yzx(s2).view(1, 3, 1, 1) * 2.0
    q2, _ = tgt(I2)
    p2 = q2[0, 0]
    # top-2 grid modes
    top = torch.topk(p2, 2).indices
    c1 = max(float(G[top[0]] @ s1), float(G[top[0]] @ s2))
    c2 = max(float(G[top[1]] @ s1), float(G[top[1]] @ s2))
    print("two-source: top-2 modes align to {s1,s2}? cos1=%.3f cos2=%.3f" % (c1, c2))
    print("two-source: q entropy (nats)           =", float(-(p2*p2.clamp_min(1e-9).log()).sum()),
          " (should exceed single-source)")

    # ---- diffuse field: random directions per bin -> low confidence, flat q ----
    Id = torch.randn(1, 3, M, T)
    qd, Wpd = tgt(Id)
    print("diffuse: mean Wp =", float(Wpd.mean()),
          "  single-source mean Wp =", float(Wp.mean()), " (diffuse should be << )")
    print("shapes: q", tuple(q.shape), "Wp", tuple(Wp.shape), "P expected", (M//fshape)*(T//tshape))