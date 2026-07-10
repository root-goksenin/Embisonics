import math
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


sys.path.append("/home/gyuksel2/embisonics_icassp/Embisonics")
from src.model import SphereV5    

from src.model.foa_rotation import random_rotations, rotate_foa_waveform

 
FSHAPE, TSHAPE = 16, 8
NUM_MEL_BINS   = 128
TARGET_LENGTH  = 200
SR             = 32000
SEED           = 0
 
ANGLE_TOL_DEG  = 5.0      # Procrustes residual after best rotation
TRACE_TOL      = 0.10     # |trace(R_eff) - trace(R)|
EL_TOL_DEG     = 3.0      # yaw mode: elevation drift
 
torch.manual_seed(SEED)
np.random.seed(SEED)
 
 
def azel_to_xyz(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])
 
 
def plane_wave_foa(d, n, rng):
    s = rng.standard_normal(n)
    return torch.from_numpy(np.stack([s, d[1] * s, d[2] * s, d[0] * s])).float()
 
 
def kabsch(D, V, allow_reflection=False):
    H = D.T @ V
    U, S, Vt = np.linalg.svd(H)
    det = np.sign(np.linalg.det(U @ Vt))
    corr = np.diag([1.0, 1.0, (1.0 if allow_reflection else det)])
    R = (U @ corr @ Vt).T
    ang = np.degrees(np.arccos(np.clip((V * (D @ R.T)).sum(-1), -1, 1)))
    return R, float(ang.mean())
 
 
def build_model(rotation_mode=None):
    pf, pt = NUM_MEL_BINS // FSHAPE, TARGET_LENGTH // TSHAPE
    stub = SimpleNamespace(fshape=FSHAPE, tshape=TSHAPE,
                           fstride=FSHAPE, tstride=TSHAPE,
                           get_patch_size=lambda: (pf, pt))
    m = SphereV5(patch_strategy=stub, gramt_model_id=None,
                 rotation_mode=rotation_mode, num_mel_bins=NUM_MEL_BINS,
                 target_length=TARGET_LENGTH, sr=SR)
    m.eval()
    return m
 
 
def recover_dirs(model, wavs):
    """q-target mean direction per clip, via the real eval-mode path."""
    with torch.no_grad():
        _, intensity_mel, energy_mel, _, _ = model._prepare_batch((wavs,))
        q, Wp = model.route_a(intensity_mel.transpose(2, 3).contiguous())
        v = F.normalize((Wp.unsqueeze(-1) * (q @ model.route_a.G)).sum(1), dim=-1)
        E_tot = energy_mel.sum(dim=(1, 2, 3))
    return v.numpy().astype(np.float64), Wp.sum(-1), E_tot
 
 
# ---------------------------------------------------------------------------
# TEST D: rotation equivariance
# ---------------------------------------------------------------------------
def test_rotation_equivariance():
    model = build_model()                       # eval: rotation applied manually
    rng = np.random.default_rng(SEED)
    doas = [(0, 0), (90, 0), (0, 60), (45, 30), (-120, -30), (160, -50)]
    hop = SR // 100
    n = TARGET_LENGTH * hop + 1024
    wavs = torch.stack([plane_wave_foa(azel_to_xyz(a, e), n, rng)
                        for a, e in doas])
 
    # one shared random SO(3) rotation from the repo's own generator
    R = random_rotations(1, "so3", device=wavs.device)[0]      # (3,3)
    Rn = R.numpy().astype(np.float64)
    wavs_rot = rotate_foa_waveform(wavs, R.unsqueeze(0).expand(len(doas), 3, 3))
 
    print("\n=== TEST D: rotation equivariance (SO(3)) ===")
    # (1) W untouched -> GRAM-T context rotation-invariant
    w_delta = (wavs_rot[:, 0] - wavs[:, 0]).abs().max().item()
    print(f"  W-channel max |delta| after rotation: {w_delta:.2e}")
    assert w_delta < 1e-5, "rotate_foa_waveform modified the W channel!"
 
    V0, Wp0, E0 = recover_dirs(model, wavs)
    V1, Wp1, E1 = recover_dirs(model, wavs_rot)
 
    # (2) energy / Wp invariance (orthogonality)
    e_rel = ((E1 - E0).abs() / E0.clamp_min(1e-9)).max().item()
    w_rel = ((Wp1 - Wp0).abs() / Wp0.clamp_min(1e-9)).max().item()
    print(f"  invariance: max rel-delta energy {e_rel:.2%}, Wp {w_rel:.2%}")
    assert e_rel < 0.02 and w_rel < 0.05, \
        "energy/Wp changed under rotation -> R is not acting orthogonally"
 
    # (3) one rigid proper rotation explains all sources
    R_rot,  err_rot  = kabsch(V0, V1, allow_reflection=False)
    R_refl, err_refl = kabsch(V0, V1, allow_reflection=True)
    print(f"  Procrustes: proper-rotation residual {err_rot:.2f} deg "
          f"(reflection {err_refl:.2f} deg)")
    assert err_rot < ANGLE_TOL_DEG, \
        "no single rotation maps pre- to post-rotation directions (inconsistent action)"
    assert err_refl > err_rot - 2.0 or err_rot < 1.0, \
        "REFLECTION introduced by rotate_foa_waveform (mirror bug)"
 
    # (4) same rotation ANGLE as requested (trace is basis-invariant)
    tr_eff, tr_req = float(np.trace(R_rot)), float(np.trace(Rn))
    ang_eff = math.degrees(math.acos(max(-1.0, min(1.0, (tr_eff - 1) / 2))))
    ang_req = math.degrees(math.acos(max(-1.0, min(1.0, (tr_req - 1) / 2))))
    print(f"  rotation angle: applied {ang_req:.1f} deg, effective {ang_eff:.1f} deg")
    assert abs(tr_eff - tr_req) < TRACE_TOL, (
        f"effective rotation angle ({ang_eff:.1f} deg) != requested "
        f"({ang_req:.1f} deg) -> R is applied in an inconsistent basis "
        f"(channel-mapping bug in rotate_foa_waveform)")
    print("  ✅ rotation aug is a consistent proper rotation of the sound field")
 
 
def test_yaw_preserves_elevation():
    model = build_model()
    rng = np.random.default_rng(SEED + 1)
    doas = [(30, 45), (-60, -20), (120, 70), (0, -60)]
    hop = SR // 100
    n = TARGET_LENGTH * hop + 1024
    wavs = torch.stack([plane_wave_foa(azel_to_xyz(a, e), n, rng)
                        for a, e in doas])
    R = random_rotations(1, "yaw", device=wavs.device)[0]
    wavs_rot = rotate_foa_waveform(wavs, R.unsqueeze(0).expand(len(doas), 3, 3))
 
    V0, _, _ = recover_dirs(model, wavs)
    V1, _, _ = recover_dirs(model, wavs_rot)
    el0 = np.degrees(np.arcsin(np.clip(V0[:, 2], -1, 1)))
    el1 = np.degrees(np.arcsin(np.clip(V1[:, 2], -1, 1)))
    drift = np.abs(el1 - el0).max()
    print("\n=== TEST D2: yaw mode preserves elevation ===")
    print(f"  max elevation drift under yaw rotation: {drift:.2f} deg")
    assert drift < EL_TOL_DEG, \
        "yaw-mode rotation changed source elevations -> yaw axis is wrong"
    print("  ✅ yaw rotations leave elevation intact")
 
 
# ---------------------------------------------------------------------------
# TEST E: train-mode gradient smoke + buffers + checkpoint roundtrip
# ---------------------------------------------------------------------------
def test_train_smoke_and_checkpoint():
    model = build_model(rotation_mode="so3")
    model.train()                               # rotation aug ACTIVE this time
    rng = np.random.default_rng(SEED + 2)
    hop = SR // 100
    n = TARGET_LENGTH * hop + 1024
    wavs = torch.stack([plane_wave_foa(azel_to_xyz(a, e), n, rng)
                        for a, e in [(30, 10), (-45, -20)]])
 
    fb7, im, em, ld, vis = model._prepare_batch((wavs,))
    out = model.forward(fb7.float(), im, em, ld, vis)   # float32: bf16 path is GPU/autocast
    loss = out["loss"]
    print("\n=== TEST E: train-mode smoke ===")
    print(f"  loss={loss.item():.4f}  (l_q={out['l_q']:.3f} l_diff={out['l_diff']:.3f} "
          f"l_leveldiff={out['l_leveldiff']:.3f} l_level={out['l_level']:.3f})")
    assert torch.isfinite(loss), "non-finite loss on first step"
 
    loss.backward()
    # With gramt_model_id=None the decoder's cross-attention branch never runs
    # (context is None), so its params legitimately have no grad here.  They
    # are checked separately below with a synthetic context.
    xattn_idle = model.gram is None
    is_xattn = lambda n: any(k in n for k in ("norm_q", "norm_kv", "cross_attn"))
    bad, n_grads, n_idle = [], 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"frozen param {name} received a gradient"
            continue
        if xattn_idle and is_xattn(name):
            assert p.grad is None, \
                f"{name} received a gradient with context=None (should be idle)"
            n_idle += 1
            continue
        if p.grad is None:
            bad.append(f"{name}: NO grad")
        elif not torch.isfinite(p.grad).all():
            bad.append(f"{name}: non-finite grad")
        else:
            n_grads += 1
    assert not bad, "gradient problems:\n  " + "\n  ".join(bad[:20])
    print(f"  ✅ finite gradients on {n_grads} trainable tensors "
          f"({n_idle} cross-attn tensors correctly idle without GRAM), "
          f"frozen params clean")
 
    # --- exercise the cross-attention path with a synthetic context ---------
    model.zero_grad(set_to_none=True)
    B = fb7.shape[0]
    fake_ctx = torch.randn(B, model.num_patches,
                           model.decoder.decoder_embed_dim)
    z = model._encode_visible(fb7.float(), vis)[:, 1:, :]
    preds, xnorms = model.decoder(z, vis, fake_ctx)
    dummy = sum(v.float().pow(2).mean() for v in preds.values())
    dummy.backward()
    x_bad = [n for n, p in model.named_parameters()
             if p.requires_grad and is_xattn(n)
             and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not x_bad, "cross-attn params without finite grads under synthetic " \
                      "context:\n  " + "\n  ".join(x_bad[:20])
    assert all(v > 0 for v in xnorms), \
        "cross_attn_norm is 0 despite a context being provided"
    print(f"  ✅ cross-attention path: finite grads with synthetic context, "
          f"per-layer delta norms {['%.2f' % v for v in xnorms]}")
    model.zero_grad(set_to_none=True)
 
    # EMA buffers initialized by the training-mode forward
    for buf, init in [("wp_ref_inited", "wp_log_ref"),
                      ("level_stats_inited", "level_running_mean"),
                      ("leveldiff_stats_inited", "leveldiff_running_mean")]:
        assert bool(getattr(model, buf).any()), f"{buf} not set after a training forward"
        assert torch.isfinite(getattr(model, init)).all(), f"{init} non-finite"
    print(f"  ✅ EMA buffers initialized (wp_log_ref={model.wp_log_ref.item():.3f} "
          f"— synthetic scale, will differ from real-data ~12.9)")
 
    # checkpoint roundtrip with the NEW buffers, strict=True
    sd = model.state_dict()
    fresh = build_model(rotation_mode="so3")
    missing, unexpected = fresh.load_state_dict(sd, strict=True), None
    assert not missing.missing_keys and not missing.unexpected_keys
    assert torch.equal(fresh.wp_log_ref, model.wp_log_ref)
    print("  ✅ state_dict roundtrips strict=True incl. wp_log_ref/wp_ref_inited")
 
 
if __name__ == "__main__":
    failures = 0
    for t in (test_rotation_equivariance, test_yaw_preserves_elevation,
              test_train_smoke_and_checkpoint):
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"\n❌ {t.__name__} FAILED:\n   {e}")
    print("\n" + ("ALL TESTS PASSED — rotation aug and train path verified."
                  if failures == 0 else
                  f"{failures} TEST(S) FAILED — do NOT launch until fixed."))
    sys.exit(failures)