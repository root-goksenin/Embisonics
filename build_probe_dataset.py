import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--json_glob", required=True,
               help="glob for the sampling JSONs, e.g. 'sampling/*.json'")
P.add_argument("--rir_sr", type=int, required=True,
               help="sample rate of the ambisonic RIR .npy files (check your "
                    "renderer config; there is no way to infer it from .npy)")
P.add_argument("--dry_glob", required=True,
               help="glob for dry mono source wavs, e.g. 'librispeech/**/*.flac'")
P.add_argument("--out_dir", default="probe_dataset")
P.add_argument("--n_clips", type=int, default=4000)
P.add_argument("--clip_sec", type=float, default=2.0)
P.add_argument("--target_sr", type=int, default=32000)
P.add_argument("--calib_n", type=int, default=300)
P.add_argument("--test_frac", type=float, default=0.2)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--az_convention", choices=["cw", "ccw"], default="cw",
               help="azimuth handedness of the sampling JSONs. The provided "
                    "sampler uses atan2(dx,-dz) with Habitat x=right, i.e. "
                    "CLOCKWISE (90 deg = right), so the default converts to "
                    "the CCW/left-positive STARSS23 convention via az -> -az.")
P.add_argument("--min_rir_paths_exist", action="store_true",
               help="skip entries whose RIR file is missing instead of failing")
ARGS = P.parse_args()

rng = np.random.default_rng(ARGS.seed)

# ----------------------------------------------------------------------------
# geometry helpers (target frame: x front, y left, z up; az CCW; el up)
# ----------------------------------------------------------------------------
def azel_to_xyz(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])


def habitat_relative_azel(source_pos, sensor_pos):
    """Habitat frame: x right, y up, z backward; agent forward = -z (identity
    rotation). Returns (az_ccw_deg, el_deg, dist)."""
    r = np.asarray(source_pos) - np.asarray(sensor_pos)
    dist = float(np.linalg.norm(r))
    az = math.degrees(math.atan2(-r[0], -r[2]))          # CCW, left positive
    el = math.degrees(math.atan2(r[1], math.hypot(r[0], r[2])))
    return az, el, dist


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------
# 1. collect
# ----------------------------------------------------------------------------
def collect_entries():
    entries = []
    files = sorted(glob.glob(ARGS.json_glob, recursive=True))
    assert files, f"no JSONs match {ARGS.json_glob!r}"
    for jf in files:
        with open(jf) as f:
            doc = json.load(f)
        house = doc["house"]["id"]
        for reg in doc.get("sampled_regions", []):
            scene = reg["region"]["scene"]
            sensor = reg["region"]["agent_data"]["sensor_position"]
            agent_az = float(reg["region"]["agent_data"].get("azimuth", 0.0))
            items = [("source", scene["source"])] + \
                    [("noise", n) for n in scene.get("noise", [])]
            for kind, s in items:
                rp = s["rir"].get("ambisonic_rir_path", "")
                if not rp:
                    continue
                if ARGS.min_rir_paths_exist and not os.path.exists(rp):
                    continue
                rel_az = float(s["azimuth"]) - agent_az       # sampler frame
                if ARGS.az_convention == "cw":                # -> CCW target
                    rel_az = -rel_az
                entries.append(dict(
                    house=house, room=s.get("room_id", "?"), kind=kind,
                    az=wrap180(rel_az),
                    el=float(s["elevation"]),
                    dist=float(s["radius"]),
                    pos=s["position"], sensor=sensor, rir=rp,
                    visible=bool(s["rir"].get("is_source_visible", False)),
                    ray_eff=float(s["rir"].get("ambisonic_ray_efficiency", np.nan)),
                ))
    print(f"collected {len(entries)} RIR samples "
          f"({sum(e['kind'] == 'source' for e in entries)} source / "
          f"{sum(e['kind'] == 'noise' for e in entries)} noise) "
          f"from {len({e['house'] for e in entries})} houses")

    # geometry cross-checks on a sample: radius, and az/el handedness vs positions
    chk = [entries[i] for i in rng.choice(len(entries), min(300, len(entries)),
                                          replace=False)]
    r_err, az_err_p, az_err_m, el_err = [], [], [], []
    for e in chk:
        az_g, el_g, d_g = habitat_relative_azel(e["pos"], e["sensor"])
        r_err.append(abs(d_g - e["dist"]) / max(e["dist"], 1e-6))
        az_err_p.append(abs(wrap180(az_g - e["az"])))
        az_err_m.append(abs(wrap180(-az_g - e["az"])))
        el_err.append(abs(el_g - e["el"]))
    print(f"  radius vs positions: median rel err {np.median(r_err):.2%} "
          f"(should be ~0; large => positions/sensor mismatch)")
    hand = "CCW (matches target)" if np.median(az_err_p) < np.median(az_err_m) \
        else "CW (will be absorbed by RIR calibration below)"
    print(f"  JSON azimuth handedness vs Habitat geometry: {hand} "
          f"(median |err|: ccw {np.median(az_err_p):.1f} deg / "
          f"cw {np.median(az_err_m):.1f} deg; el {np.median(el_err):.1f} deg)")
    return entries


# ----------------------------------------------------------------------------
# RIR loading
# ----------------------------------------------------------------------------
def load_rir(path):
    r = np.load(path)
    if r.ndim == 1:
        raise ValueError(f"{path}: 1-channel RIR, expected ambisonic")
    if r.shape[0] > r.shape[1]:                     # (T, C) -> (C, T)
        r = r.T
    if r.shape[0] > 4:                              # HOA: keep first order
        r = r[:4]
    assert r.shape[0] == 4, f"{path}: got {r.shape[0]} channels"
    return r.astype(np.float64)


# ----------------------------------------------------------------------------
# 2. calibrate the ambisonic convention
# ----------------------------------------------------------------------------
def direct_window(rir, sr, pre_ms=0.5, post_ms=2.5):
    w = rir[0]
    peak = int(np.argmax(np.abs(w)))
    a = max(0, peak - int(pre_ms * 1e-3 * sr))
    b = min(rir.shape[1], peak + int(post_ms * 1e-3 * sr))
    return rir[:, a:b]


def kabsch(D, V, allow_reflection=True):
    H = D.T @ V
    U, S, Vt = np.linalg.svd(H)
    det = np.sign(np.linalg.det(U @ Vt))
    corr = np.diag([1.0, 1.0, (1.0 if allow_reflection else det)])
    O = (U @ corr @ Vt).T                          # v ~= O d
    ang = np.degrees(np.arccos(np.clip((V * (D @ O.T)).sum(-1), -1, 1)))
    return O, float(np.median(ang))


def calibrate(entries):
    cand = sorted([e for e in entries if e["visible"]],
                  key=lambda e: -e["ray_eff"]) or \
           sorted(entries, key=lambda e: -e["ray_eff"])
    cand = cand[: ARGS.calib_n]
    D, V, gains = [], [], []
    for e in cand:
        try:
            rir = load_rir(e["rir"])
        except Exception:
            continue
        d = direct_window(rir, ARGS.rir_sr)
        w, dirp = d[0], d[1:4]
        Ivec = (w[None] * dirp).mean(axis=1)       # time-domain direct intensity
        n = np.linalg.norm(Ivec)
        if n < 1e-12 or np.abs(w).max() < 1e-9:
            continue
        V.append(Ivec / n)
        D.append(azel_to_xyz(e["az"], e["el"]))
        # normalization gain: ||dir-part|| / |W| at the direct path
        gains.append(np.linalg.norm(dirp, axis=0).max() / np.abs(w).max())
    D, V = np.stack(D), np.stack(V)
    print(f"\ncalibration on {len(D)} direct paths")

    O, med_err = kabsch(D, V, allow_reflection=True)
    det = float(np.linalg.det(O))
    g = float(np.median(gains))
    # which convention is this closest to?
    scale = 1.0 / g if abs(g - math.sqrt(3)) < abs(g - 1.0) else 1.0
    norm_name = "N3D (rescaling to SN3D)" if scale != 1.0 else "SN3D"

    # M maps raw channels 1:4 -> (Y,Z,X) in target frame:  d ~= O^T v,
    # (y,z,x) = P d  with P = xyz->yzx permutation.
    Pm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    M = scale * (Pm @ O.T)

    print(f"  Procrustes median error: {med_err:.2f} deg  "
          f"(det O = {det:+.2f}{', reflection -> CW az or channel swap, '
          'absorbed by M' if det < 0 else ''})")
    print(f"  direct-path gain ||dir||/|W| = {g:.3f} -> {norm_name}")
    print(f"  O (raw <- xyz) =\n{np.array_str(np.round(O, 3))}")
    assert med_err < 12.0, (
        "calibration failed: no orthogonal map explains the direct-path "
        "directions. Check RIR_SR, channel-major orientation of the .npy, and "
        "whether channel 0 is really the omni channel.")

    # verify on held-out entries
    hold = [e for e in entries if e not in cand][: 100]
    errs = []
    for e in hold:
        try:
            rir = load_rir(e["rir"])
        except Exception:
            continue
        d = direct_window(rir, ARGS.rir_sr)
        v = (d[0][None] * d[1:4]).mean(axis=1)
        if np.linalg.norm(v) < 1e-12:
            continue
        yzx = M @ (v / np.linalg.norm(v))
        xyz = yzx[[2, 0, 1]]
        gt = azel_to_xyz(e["az"], e["el"])
        errs.append(math.degrees(math.acos(np.clip(xyz @ gt / np.linalg.norm(xyz), -1, 1))))
    if errs:
        print(f"  held-out verification: median DOA err {np.median(errs):.2f} deg "
              f"on {len(errs)} RIRs (visible+occluded mixed; occluded is worse "
              f"by nature)")
    return M, dict(O=O.tolist(), scale=scale, gain=g, det=det,
                   procrustes_med_deg=med_err, norm=norm_name)


def apply_calibration(rir, M):
    """raw (4,T) -> (W,Y,Z,X) SN3D in x-front/y-left/z-up frame."""
    out = np.empty_like(rir)
    out[0] = rir[0]
    out[1:4] = M @ rir[1:4]
    return out


# ----------------------------------------------------------------------------
# 3. RT60 / DRR
# ----------------------------------------------------------------------------
def schroeder_rt60(w, sr):
    """Broadband T30 from Schroeder EDC on the omni channel, T20 fallback.
    Returns (rt60_s, method) or (nan, 'fail')."""
    e = w.astype(np.float64) ** 2
    peak = int(np.argmax(e))
    e = e[peak:]
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-20)
    t = np.arange(len(edc_db)) / sr

    def fit(lo, hi):
        m = (edc_db <= lo) & (edc_db >= hi)
        if m.sum() < int(0.01 * sr):
            return None
        A = np.stack([t[m], np.ones(m.sum())], 1)
        slope, _ = np.linalg.lstsq(A, edc_db[m], rcond=None)[0]
        return -60.0 / slope if slope < -1e-9 else None

    if edc_db.min() < -38.0:
        r = fit(-5.0, -35.0)
        if r:
            return float(r), "T30"
    if edc_db.min() < -28.0:
        r = fit(-5.0, -25.0)
        if r:
            return float(r), "T20"
    return float("nan"), "fail"


def drr_db(rir, sr, win_ms=2.5):
    w = rir[0]
    peak = int(np.argmax(np.abs(w)))
    a, b = max(0, peak - int(win_ms * 1e-3 * sr)), peak + int(win_ms * 1e-3 * sr)
    e = rir ** 2
    direct = e[:, a:b].sum()
    rest = e[:, b:].sum()
    return float(10.0 * np.log10(direct / max(rest, 1e-20)))


# ----------------------------------------------------------------------------
# 4./5. synthesize + split
# ----------------------------------------------------------------------------
def house_split(house):
    h = int(hashlib.md5(house.encode()).hexdigest(), 16) % 1000
    return "test" if h < ARGS.test_frac * 1000 else "train"


def load_dry(path, need, sr_out):
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if sr != sr_out:
        g = math.gcd(sr_out, sr)
        x = resample_poly(x, sr_out // g, sr // g)
    if len(x) < need:
        x = np.tile(x, int(np.ceil(need / len(x))))
    off = rng.integers(0, len(x) - need + 1)
    return x[off: off + need]


def main():
    entries = collect_entries()
    M, calib_report = calibrate(entries)

    os.makedirs(os.path.join(ARGS.out_dir, "wavs"), exist_ok=True)
    with open(os.path.join(ARGS.out_dir, "calibration.json"), "w") as f:
        json.dump(calib_report, f, indent=2)

    # RT60/DRR per entry (cache per RIR path)
    print("\ncomputing RT60/DRR per RIR ...")
    rt_cache = {}
    rows = []
    for e in entries:
        if e["rir"] not in rt_cache:
            try:
                rir = load_rir(e["rir"])
                rt, meth = schroeder_rt60(rir[0], ARGS.rir_sr)
                rt_cache[e["rir"]] = (rt, meth, drr_db(rir, ARGS.rir_sr))
            except Exception as ex:
                rt_cache[e["rir"]] = (float("nan"), f"error:{ex}", float("nan"))
        rt, meth, dr = rt_cache[e["rir"]]
        if np.isfinite(rt):
            rows.append({**e, "rt60": rt, "rt60_method": meth, "drr": dr})
    print(f"  usable entries with finite RT60: {len(rows)}/{len(entries)}")
    df = pd.DataFrame(rows)
    print(df.rt60.describe().round(2).to_string())

    # stratified pick over rt60 x log-distance
    df["rt_bin"] = pd.qcut(df.rt60, 5, duplicates="drop")
    df["d_bin"] = pd.qcut(np.log(df.dist.clip(lower=1e-2)), 5, duplicates="drop")
    per_cell = max(1, ARGS.n_clips // (df.rt_bin.nunique() * df.d_bin.nunique()))
    pick = (df.groupby(["rt_bin", "d_bin"], observed=True, group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_cell),
                                        random_state=ARGS.seed)))
    if len(pick) > ARGS.n_clips:
        pick = pick.sample(ARGS.n_clips, random_state=ARGS.seed)
    print(f"\nsynthesizing {len(pick)} clips (stratified rt60 x distance) ...")

    dry = sorted(glob.glob(ARGS.dry_glob, recursive=True))
    assert dry, f"no dry sources match {ARGS.dry_glob!r}"

    need = int((ARGS.clip_sec + 0.5) * ARGS.rir_sr)   # margin for the RIR tail
    out_len = int(ARGS.clip_sec * ARGS.target_sr)
    meta = []
    for i, (_, e) in enumerate(pick.iterrows()):
        rir = apply_calibration(load_rir(e["rir"]), M)
        src_path = dry[rng.integers(0, len(dry))]
        s = load_dry(src_path, need, ARGS.rir_sr)
        wet = np.stack([fftconvolve(s, rir[c])[: need] for c in range(4)])
        g = math.gcd(ARGS.target_sr, ARGS.rir_sr)
        wet = resample_poly(wet, ARGS.target_sr // g, ARGS.rir_sr // g, axis=-1)
        # energy-checked random crop
        for _ in range(8):
            off = rng.integers(0, wet.shape[1] - out_len + 1)
            clip = wet[:, off: off + out_len]
            if (clip[0] ** 2).mean() > 1e-3 * (wet[0] ** 2).mean():
                break
        clip = clip / (np.abs(clip).max() + 1e-9) * 0.9
        cid = f"clip_{i:05d}"
        sf.write(os.path.join(ARGS.out_dir, "wavs", cid + ".wav"),
                 clip.T.astype(np.float32), ARGS.target_sr)
        meta.append(dict(clip_id=cid, house=e["house"], room=e["room"],
                         split=house_split(e["house"]), kind=e["kind"],
                         az=e["az"], el=e["el"], dist_m=e["dist"],
                         rt60_s=e["rt60"], rt60_method=e["rt60_method"],
                         drr_db=e["drr"], visible=e["visible"],
                         ray_eff=e["ray_eff"],
                         dry_source=os.path.basename(src_path)))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(pick)}")

    m = pd.DataFrame(meta)
    m.to_parquet(os.path.join(ARGS.out_dir, "metadata.parquet"))
    print(f"\nwrote {len(m)} clips; split sizes: "
          f"{m.split.value_counts().to_dict()}; "
          f"test houses: {sorted(m[m.split == 'test'].house.unique())}")
    print("label ranges: "
          f"dist [{m.dist_m.min():.2f}, {m.dist_m.max():.2f}] m, "
          f"rt60 [{m.rt60_s.min():.2f}, {m.rt60_s.max():.2f}] s, "
          f"drr [{m.drr_db.min():.1f}, {m.drr_db.max():.1f}] dB")


if __name__ == "__main__":
    # main()
    import pandas as pd
    p = "/projects/0/prjs1261/probe_dataset/metadata.parquet"
    m = pd.read_parquet(p)
    # pick 3 test houses stratified over RT60 (low / mid / high acoustic regime)
    h = m.groupby("house").rt60_s.median().sort_values()
    test_houses = [h.index[1], h.index[len(h)//2], h.index[-2]]
    m["split"] = m.house.map(lambda x: "test" if x in test_houses else "train")
    m.to_parquet(p)
    print("test houses:", test_houses, "| sizes:", m.split.value_counts().to_dict())