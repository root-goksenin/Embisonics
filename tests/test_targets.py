import math
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("/home/gyuksel2/embisonics_icassp/Embisonics")
from src.model import SphereV5                  

# ---- config: MUST match your training run -------------------------------
FSHAPE, TSHAPE   = 16, 8
NUM_MEL_BINS     = 128
TARGET_LENGTH    = 200
SR               = 32000
SEED             = 0

ANGLE_TOL_DEG    = 10.0     # mean recovery error after best proper rotation
IDENTITY_TOL     = 0.15     # max |R - I| entry for "R is identity"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def azel_to_xyz(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])


def plane_wave_foa(direction_xyz, n_samples, rng, band=None, gate_after_frac=None):
    """SN3D first-order encoding of a broadband-noise plane wave.
    W = s ; (Y,Z,X) = (d_y, d_z, d_x) * s   (ACN channel order W,Y,Z,X).
    band = (f_lo, f_hi) Hz applies an FFT brickwall; gate_after_frac zeroes
    the first fraction of the signal (time-gated onset)."""
    s = rng.standard_normal(n_samples)
    if band is not None:
        S = np.fft.rfft(s)
        f = np.fft.rfftfreq(n_samples, 1.0 / SR)
        S[(f < band[0]) | (f > band[1])] = 0.0
        s = np.fft.irfft(S, n=n_samples)
    if gate_after_frac is not None:
        s[: int(gate_after_frac * n_samples)] = 0.0
    dx, dy, dz = direction_xyz
    wav = np.stack([s, dy * s, dz * s, dx * s])
    return torch.from_numpy(wav).float()


def kabsch(D, V, allow_reflection=False):
    """Best R (v ~= R d) minimizing ||V - D R^T||_F.  D,V: (n,3) unit rows."""
    H = D.T @ V
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    corr = np.diag([1.0, 1.0, (1.0 if allow_reflection else d)])
    R = (U @ corr @ Vt).T
    ang = np.degrees(np.arccos(np.clip((V * (D @ R.T)).sum(-1), -1, 1)))
    return R, float(ang.mean())


def classify_R(R):
    Rr = np.round(R).astype(int)
    if np.abs(R - np.eye(3)).max() < IDENTITY_TOL:
        return "identity"
    if np.abs(R - Rr).max() < IDENTITY_TOL and (np.abs(Rr).sum(0) == 1).all() \
            and (np.abs(Rr).sum(1) == 1).all():
        return f"signed permutation {Rr.tolist()} (det={int(round(np.linalg.det(Rr)))})"
    return "general (axis corruption)"


def build_model():
    pf, pt = NUM_MEL_BINS // FSHAPE, TARGET_LENGTH // TSHAPE
    stub = SimpleNamespace(fshape=FSHAPE, tshape=TSHAPE,
                           fstride=FSHAPE, tstride=TSHAPE,
                           get_patch_size=lambda: (pf, pt))
    m = SphereV5(patch_strategy=stub, gramt_model_id=None,
                 rotation_mode=None, num_mel_bins=NUM_MEL_BINS,
                 target_length=TARGET_LENGTH, sr=SR)
    m.eval()
    return m, pf, pt


def targets_for(model, wav_batch):
    """Exercise the REAL path: _prepare_batch -> route_a(transpose(2,3))."""
    with torch.no_grad():
        fb7, intensity_mel, energy_mel, leveldiff, vis = \
            model._prepare_batch((wav_batch,))
        q, Wp = model.route_a(intensity_mel.transpose(2, 3).contiguous())
        Ep = model._patchify(energy_mel).float().mean(dim=(-2, -1))
    return q, Wp, Ep


# ---------------------------------------------------------------------------
# TEST A: direction recovery + axis/mirror diagnosis
# ---------------------------------------------------------------------------
def test_direction_recovery():
    model, pf, pt = build_model()
    rng = np.random.default_rng(SEED)
    doas = [(0, 0), (90, 0), (-90, 0), (180, 0), (0, 60), (45, 30), (-120, -30)]
    D = np.stack([azel_to_xyz(a, e) for a, e in doas])

    hop = SR // 100
    n = TARGET_LENGTH * hop + 1024
    wavs = torch.stack([plane_wave_foa(d, n, rng) for d in D])

    q, Wp, _ = targets_for(model, wavs)
    md = q @ model.route_a.G                                    # (N,P,3)
    v = F.normalize((Wp.unsqueeze(-1) * md).sum(1), dim=-1)     # (N,3)
    V = v.numpy().astype(np.float64)

    R_rot,  err_rot  = kabsch(D, V, allow_reflection=False)
    R_refl, err_refl = kabsch(D, V, allow_reflection=True)

    print("\n=== TEST A: direction recovery ===")
    for (a, e), vi in zip(doas, V):
        az = math.degrees(math.atan2(vi[1], vi[0]))
        el = math.degrees(math.asin(np.clip(vi[2], -1, 1)))
        print(f"  DOA ({a:4d},{e:3d}) -> recovered ({az:7.1f},{el:6.1f})")
    print(f"  best proper rotation:  mean err {err_rot:5.1f} deg, "
          f"R = {classify_R(R_rot)}")
    print(f"  best incl. reflection: mean err {err_refl:5.1f} deg")
    print("  R (proper) =\n" + np.array_str(np.round(R_rot, 2)))

    assert err_refl > err_rot - 3.0 or err_rot < ANGLE_TOL_DEG, (
        "MIRROR BUG: a reflection fits the recovered directions much better "
        "than any rotation -> q targets are antipodal/mirrored. Check the "
        "intensity sign convention and channel ordering in RouteATarget."
    )
    kind = classify_R(R_rot)
    assert err_rot < ANGLE_TOL_DEG, (
        f"axis corruption: no rotation maps true DOAs to recovered ones "
        f"(residual {err_rot:.1f} deg). Check (Iy,Iz,Ix) handling in RouteATarget."
    )
    if kind != "identity":
        print(f"  ⚠️  internally consistent but non-identity convention: {kind}.\n"
              "     Not a training bug (rotation-augmented SSL is convention-"
              "invariant), but _mean_dirs outputs and any DOA probe must use "
              "this mapping. Document it.")
    else:
        print("  ✅ grid convention = x-front / y-left / z-up (STARSS23), sign +I")


# ---------------------------------------------------------------------------
# TEST B: patch layout of Wp matches _patchify (freq-major)
# ---------------------------------------------------------------------------
def test_patch_layout():
    model, pf, pt = build_model()
    rng = np.random.default_rng(SEED + 1)
    hop = SR // 100
    n = TARGET_LENGTH * hop + 1024
    wav = plane_wave_foa(azel_to_xyz(60, 20), n, rng,
                         band=(2000, 5000), gate_after_frac=0.5).unsqueeze(0)

    q, Wp, Ep = targets_for(model, wav)
    w = torch.log1p(Wp[0]); e = Ep[0] - Ep[0].min()

    def corr(a, b):
        a, b = a.flatten(), b.flatten()
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-9))

    c_same  = corr(w.reshape(pf, pt), e.reshape(pf, pt))
    c_trans = corr(w.reshape(pt, pf).T, e.reshape(pf, pt))
    am_w = np.unravel_index(int(w.argmax()), (pf, pt))
    am_e = np.unravel_index(int(e.argmax()), (pf, pt))

    print("\n=== TEST B: patch layout (route_a vs _patchify) ===")
    print(f"  corr(Wp, Ep) freq-major layout : {c_same:.3f}")
    print(f"  corr(Wp, Ep) transposed layout : {c_trans:.3f}")
    print(f"  argmax patch: Wp {am_w}  vs  Ep {am_e}   (band-limited, "
          f"time-gated source)")

    assert c_same > 0.8, (
        f"Wp does not match _patchify's patch energy under the freq-major "
        f"layout (corr={c_same:.2f}) -> route_a patch ordering disagrees with "
        f"every other target; w_q and mask indexing are misaligned."
    )
    assert c_same > c_trans, (
        "TRANSPOSE BUG: Wp matches the patch-energy map better under the "
        "transposed layout -> route_a patchifies time-major while _patchify "
        "is freq-major. This is the transpose(2,3) issue."
    )
    print("  ✅ route_a patch indexing consistent with _patchify")


# ---------------------------------------------------------------------------
# TEST C: invariants
# ---------------------------------------------------------------------------
def test_invariants():
    model, pf, pt = build_model()
    rng = np.random.default_rng(SEED + 2)
    hop = SR // 100
    n = TARGET_LENGTH * hop + 1024
    wav = plane_wave_foa(azel_to_xyz(10, -40), n, rng,
                         gate_after_frac=0.6).unsqueeze(0)
    q, Wp, _ = targets_for(model, wav)

    print("\n=== TEST C: invariants ===")
    rowsum = q.sum(-1)
    ok_rows = ((rowsum - 1).abs() < 1e-3) | (rowsum.abs() < 1e-6)
    assert ok_rows.all(), "q rows must sum to 1 (or exactly 0 for silent patches)"
    assert (Wp >= 0).all(), "Wp must be non-negative"
    silent = Wp < 1e-6
    if silent.any():
        assert q[silent].abs().max() < 1e-6, "silent patches must carry no q mass"
    print(f"  ✅ rows sum to 1: {ok_rows.float().mean():.0%}; "
          f"silent patches: {silent.float().mean():.0%} of total, all zero-mass")


if __name__ == "__main__":
    failures = 0
    for t in (test_direction_recovery, test_patch_layout, test_invariants):
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"\n❌ {t.__name__} FAILED:\n   {e}")
    print("\n" + ("ALL TESTS PASSED — safe to launch." if failures == 0
                  else f"{failures} TEST(S) FAILED — do NOT launch until fixed."))
    sys.exit(failures)