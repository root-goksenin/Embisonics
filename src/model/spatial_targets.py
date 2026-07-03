"""Self-supervised spatial targets for the FOA MAE (SphereV5).

Contents
--------
fibonacci_sphere       : ~uniform grid of unit vectors on S^2
RouteATarget           : per-patch angular distribution q (vMF splat) + Wp
multiscale_diffuseness : psi at several coherence windows; the (short, long)
                         pair is the label-free DRR proxy discussed in design.

Conventions
-----------
* Intensity channel order coming from the extractor is (y, z, x)
  (= spec[:, 1:] for ACN [W, Y, Z, X]).  RouteATarget reorders to canonical
  (x, y, z) internally, so its grid / q are in (front, left, up) coordinates.
* RouteATarget consumes (B, 3, M, T).  SphereV5 stores features as
  (B, C, T, F) and transposes before the call.
* Patch-token order is freq-major: p = f * Nt + t.  This matches
  SpatialFrontEnd's flattening and SphereV5._patchify (verified in __main__).
"""
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

    Pipeline per call (B,3,M,T) -> q (B,P,|G|), Wp (B,P):
      1. reorder (y,z,x) -> (x,y,z)
      2. smooth <I> over a small TF tile (coherence average; required so the
         per-bin direction is meaningful and diffuse energy cancels).  The
         tile defaults to (3 mel, 5 frames) and is meant to be shared with
         the *short* window of `multiscale_diffuseness` so that per-tile
         confidence w_tau = ||<I>|| equals <E>*(1 - psi_short) by construction.
      3. dir_tau = <I>/||<I>||,  conf w_tau = ||<I>||
      4. splat each tile onto grid G with vMF(kappa), weight w_tau, sum
         within each (fshape x tshape) patch box -> q ; Wp = sum w_tau
      5. normalise q over the sphere.

    Notes
    -----
    * kappa=40 with |G|=256 is ~critically sampled: mean grid spacing
      sqrt(4*pi/256) ~ 12.7 deg vs vMF 1/e half-width sqrt(2/kappa) ~ 12.8 deg.
    * The grid loop is chunked (`grid_chunk`) to cap the transient
      (B, |G|, M, T) tensor; results are identical to the unchunked version.
    * Wp is returned RAW (it scales with signal power).  Compress it at the
      loss site (log1p) before using it as a weight, otherwise loud patches
      dominate training.
    """

    # extractor order (y,z,x) -> (x,y,z)
    YZX_TO_XYZ = (2, 0, 1)

    def __init__(self, n_grid: int = 256, kappa: float = 40.0,
                 fshape: int = 16, tshape: int = 4,
                 tile_f: int = 3, tile_t: int = 5,
                 grid_chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        assert tile_f % 2 == 1 and tile_t % 2 == 1, "coherence tile must be odd"
        self.kappa = kappa
        self.fshape, self.tshape = fshape, tshape
        self.tile_f, self.tile_t = tile_f, tile_t
        self.grid_chunk = grid_chunk
        self.eps = eps
        self.register_buffer("G", fibonacci_sphere(n_grid))   # (|G|,3)

    @torch.no_grad()
    def forward(self, intensity_mel: Tensor):
        B, C, M, T = intensity_mel.shape
        assert C == 3, f"expected 3 intensity channels, got {C}"
        assert M % self.fshape == 0 and T % self.tshape == 0, (
            f"(M={M}, T={T}) not divisible by patch "
            f"({self.fshape}, {self.tshape}); avg_pool would silently truncate"
        )
        I = intensity_mel[:, self.YZX_TO_XYZ, :, :].float()   # -> (x,y,z)

        # 1) coherence smoothing over a TF tile (stride 1, 'same')
        Is = F.avg_pool2d(I, kernel_size=(self.tile_f, self.tile_t),
                          stride=1, padding=(self.tile_f // 2, self.tile_t // 2),
                          count_include_pad=False)

        Inorm = torch.linalg.norm(Is, dim=1)                  # (B,M,T) = ||<I>||
        u = Is / Inorm.unsqueeze(1).clamp_min(self.eps)       # (B,3,M,T) unit

        # 2) vMF splat onto grid, chunked over grid cells, 3) pooled per patch
        fs, ts = self.fshape, self.tshape
        area = fs * ts
        w = Inorm.unsqueeze(1)                                # (B,1,M,T)
        q_chunks = []
        for g0 in range(0, self.G.shape[0], self.grid_chunk):
            Gc = self.G[g0:g0 + self.grid_chunk]              # (gc,3)
            cos = torch.einsum("gd,bdmt->bgmt", Gc, u)        # (B,gc,M,T)
            wvmf = w * torch.exp(self.kappa * (cos - 1.0))
            q_chunks.append(
                F.avg_pool2d(wvmf, kernel_size=(fs, ts), stride=(fs, ts)) * area
            )
        q = torch.cat(q_chunks, dim=1)                        # (B,|G|,Nf,Nt)
        Wp = F.avg_pool2d(w, kernel_size=(fs, ts),
                          stride=(fs, ts)).squeeze(1) * area  # (B,Nf,Nt)

        Nf, Nt = q.shape[-2], q.shape[-1]
        # flatten freq-major then time -> matches SpatialFrontEnd token order
        q = q.permute(0, 2, 3, 1).reshape(B, Nf * Nt, -1)     # (B,P,|G|)
        Wp = Wp.reshape(B, Nf * Nt)                           # (B,P)

        q = q / q.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return q, Wp


def multiscale_diffuseness(intensity_tf: Tensor, energy_tf: Tensor,
                           windows=((5, 3), (35, 3)),
                           c: float = 1.0, eps: float = 1e-6) -> Tensor:
    """Diffuseness psi = 1 - ||<I>|| / (c <E>) at several coherence scales.

    Args:
        intensity_tf : (B, 3, T, F) raw mel intensity, model tensor layout.
        energy_tf    : (B, 1, T, F) mel energy.
        windows      : iterable of (tau_t, tau_f) smoothing windows, odd.
                       Default: short (5 frames, 3 mel) ~ 50 ms @ 100 fps,
                       long (35 frames, 3 mel) ~ 350 ms.
    Returns:
        (B, len(windows), T, F), one psi channel per window.

    The short/long *pair* is the DRR proxy: for a close source in a
    reverberant room psi_short is low while psi_long is markedly higher
    (direct sound dominates locally, tail dominates the average); for a far
    source both are high.  Predicting both channels rewards the model for
    representing early/late structure rather than a single conflated psi.
    """
    I = intensity_tf.float()
    E = energy_tf.float()
    outs = []
    for (tt, tf_) in windows:
        assert tt % 2 == 1 and tf_ % 2 == 1, "smoothing windows must be odd"
        Iw = F.avg_pool2d(I, kernel_size=(tt, tf_), stride=1,
                          padding=(tt // 2, tf_ // 2), count_include_pad=False)
        Ew = F.avg_pool2d(E, kernel_size=(tt, tf_), stride=1,
                          padding=(tt // 2, tf_ // 2), count_include_pad=False)
        num = torch.linalg.norm(Iw, dim=1, keepdim=True)      # ||<I>||
        psi = (1.0 - num / (c * Ew + eps)).clamp(0.0, 1.0)
        outs.append(psi)
    return torch.cat(outs, dim=1)


# =============================================================================
# Smoke tests
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    fshape, tshape, M, T = 16, 8, 128, 200
    tgt = RouteATarget(n_grid=256, kappa=40.0, fshape=fshape, tshape=tshape,
                       tile_f=3, tile_t=5, grid_chunk=64)
    G = tgt.G
    Nf, Nt = M // fshape, T // tshape

    def s_xyz(az, el):
        ph, th = math.radians(az), math.radians(el)
        return torch.tensor([math.cos(ph) * math.cos(th),
                             math.sin(ph) * math.cos(th),
                             math.sin(th)])

    def to_yzx(s):  # (x,y,z) -> extractor (y,z,x)
        return torch.stack([s[1], s[2], s[0]])

    # ---- chunked == unchunked ----
    I_rand = torch.randn(2, 3, M, T)
    q_a, w_a = tgt(I_rand)
    tgt_unchunked = RouteATarget(256, 40.0, fshape, tshape, 3, 5, grid_chunk=256)
    q_b, w_b = tgt_unchunked(I_rand)
    assert torch.allclose(q_a, q_b, atol=1e-6) and torch.allclose(w_a, w_b, atol=1e-5)
    print("chunked == unchunked: OK")

    # ---- single source: target must peak at grid cell nearest s ----
    s = s_xyz(30, 20)
    I = to_yzx(s).view(1, 3, 1, 1).expand(1, 3, M, T).clone() * 2.0
    q, Wp = tgt(I)
    peak = q[0, 0]
    g_pred = G[peak.argmax()]
    print("single-source: cos(peak_grid, s) =", float(g_pred @ s))
    assert float(g_pred @ s) > 0.97

    # ---- token order: source A in first time half, source B in second ----
    s1, s2 = s_xyz(-60, 0), s_xyz(60, 0)
    I3 = torch.zeros(1, 3, M, T)
    I3[..., :T // 2] = to_yzx(s1).view(1, 3, 1, 1) * 2.0
    I3[..., T // 2:] = to_yzx(s2).view(1, 3, 1, 1) * 2.0
    q3, _ = tgt(I3)
    for f in range(Nf):
        p_early = f * Nt + 0            # freq-major flatten
        p_late = f * Nt + (Nt - 1)
        d_early = G[q3[0, p_early].argmax()]
        d_late = G[q3[0, p_late].argmax()]
        assert float(d_early @ s1) > 0.95 and float(d_late @ s2) > 0.95
    print("freq-major token order (p = f*Nt + t): OK")

    # ---- two sources at different mel bands within one patch: multimodal q ----
    I2 = torch.zeros(1, 3, M, T)
    I2[:, :, :fshape // 2] = to_yzx(s1).view(1, 3, 1, 1) * 2.0
    I2[:, :, fshape // 2:fshape] = to_yzx(s2).view(1, 3, 1, 1) * 2.0
    q2, _ = tgt(I2)
    p2 = q2[0, 0]
    top = torch.topk(p2, 2).indices
    c1 = max(float(G[top[0]] @ s1), float(G[top[0]] @ s2))
    c2 = max(float(G[top[1]] @ s1), float(G[top[1]] @ s2))
    print(f"two-source: top-2 modes align cos1={c1:.3f} cos2={c2:.3f}")
    assert c1 > 0.95 and c2 > 0.95

    # ---- diffuse: low Wp ----
    qd, Wpd = tgt(torch.randn(1, 3, M, T))
    print(f"diffuse mean Wp = {float(Wpd.mean()):.3f} vs single-source {float(Wp.mean()):.3f}")
    # residual of a K-bin coherence average of random vectors ~ 1/sqrt(K), K=15
    assert float(Wpd.mean()) < 0.3 * float(Wp.mean())

    # ---- multiscale diffuseness: direct-then-tail toy vs steady far/diffuse ----
    # (B, 3, T, F) layout.  "Near source": strongly directional bursts every
    # 40 frames, incoherent low-level tail in between.  "Far": tail only.
    Tt, Fm = 200, 32
    def make(direct_gain):
        I_tf = 0.15 * torch.randn(1, 3, Tt, Fm)               # incoherent tail
        E_tf = torch.linalg.norm(I_tf, dim=1, keepdim=True) * 3.0
        for t0 in range(0, Tt, 40):
            I_tf[:, :, t0:t0 + 4, :] += to_yzx(s).view(1, 3, 1, 1) * direct_gain
            E_tf[:, :, t0:t0 + 4, :] += direct_gain
        return I_tf, E_tf
    In_, En_ = make(3.0)
    If_, Ef_ = make(0.0)
    psi_near = multiscale_diffuseness(In_, En_)
    psi_far = multiscale_diffuseness(If_, Ef_)
    # The DRR proxy lives in the *pair*, not in the bin-mean of their gap
    # (tail bins near a source legitimately have psi_long < psi_short because
    # the long window mixes in coherent direct energy).  The discriminative
    # statistic: near source -> psi_short dips near 0 at direct frames while
    # psi_long stays high; far source -> psi_short never dips.
    short_min_near = float(psi_near[:, 0].mean(dim=-1).min())   # min over frames
    short_min_far = float(psi_far[:, 0].mean(dim=-1).min())
    long_mean_near = float(psi_near[:, 1].mean())
    print(f"min_t psi_short: near={short_min_near:.3f}, far={short_min_far:.3f}; "
          f"mean psi_long near={long_mean_near:.3f}")
    assert short_min_near < 0.4, "near source: psi_short must dip at direct frames"
    assert short_min_far > 0.7, "far/diffuse: psi_short must stay high"
    assert long_mean_near > short_min_near + 0.2, \
        "near source: psi_long (environment) must exceed min psi_short (direct)"
    print("multiscale diffuseness DRR-proxy behaviour: OK")
    print("all spatial_targets tests passed")