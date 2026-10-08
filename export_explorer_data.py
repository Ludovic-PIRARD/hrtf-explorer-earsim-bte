#!/usr/bin/env python3
"""
Export EarSim + BTE Windowed (with ITD) SOFAs into the JSON layout used by
https://yoyolicoris.github.io/hrtf-explorer/ (data_all.json + ird.json).

Keys keep the explorer's internal names:
  meta  → EarSim (canal)
  soni  → BTE (Quest3 + Nucleus 5)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import sofar
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

EARSIM = (
    ROOT
    / "KEMAR_Knowl_EarSim_LargeEars"
    / "HRTF"
    / "48kHz"
    / "KEMAR_Knowl_EarSim_LargeEars_Windowed_48kHz.sofa"
)
BTE = (
    ROOT
    / "KEMAR BTE"
    / "KEMAR_Knowl_LargeEars_BTE_METAQUEST3"
    / "HRTF"
    / "48kHz"
    / "BTE KEMAR Knowles_Windowed_48kHz.sofa"
)

NF = 150
F_LO, F_HI = 100.0, 20000.0
FREQS = np.geomspace(F_LO, F_HI, NF)


def wrap_az(az: np.ndarray) -> np.ndarray:
    """Map [0,360) → (-180,180]; keep SOFA CCW convention (+ = left)."""
    a = np.asarray(az, dtype=float)
    return np.where(a > 180.0, a - 360.0, a)


def common_indices(sp_a: np.ndarray, sp_b: np.ndarray):
    ka = np.round(sp_a[:, :2], 3)
    kb = np.round(sp_b[:, :2], 3)
    map_a = {tuple(row): i for i, row in enumerate(ka)}
    idx_a, idx_b = [], []
    for i, row in enumerate(kb):
        key = tuple(row)
        if key in map_a:
            idx_a.append(map_a[key])
            idx_b.append(i)
    return np.asarray(idx_a), np.asarray(idx_b)


def mag_db_on_grid(ir: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    """ir (N,) → mag dB on freqs via rfft + log-freq interpolation."""
    n = ir.shape[-1]
    spec = np.fft.rfft(ir)
    f_fft = np.fft.rfftfreq(n, 1.0 / fs)
    mag = 20.0 * np.log10(np.abs(spec) + 1e-12)
    # avoid log(0) at f=0 for interp — clamp first bin
    f_fft = np.maximum(f_fft, 1e-3)
    fn = interp1d(f_fft, mag, kind="linear", bounds_error=False, fill_value=(mag[0], mag[-1]))
    return fn(freqs).astype(float)


def xcorr_itd_us(ir_l: np.ndarray, ir_r: np.ndarray, fs: float, max_lag_ms: float = 1.0) -> float:
    max_lag = int(max_lag_ms * 1e-3 * fs)
    l = ir_l / (np.linalg.norm(ir_l) + 1e-12)
    r = ir_r / (np.linalg.norm(ir_r) + 1e-12)
    corr = np.correlate(r, l, mode="full")
    mid = len(l) - 1
    segment = corr[mid - max_lag : mid + max_lag + 1]
    lag = int(np.argmax(segment)) - max_lag
    return float(lag / fs * 1e6)


def ild_db(ir_l: np.ndarray, ir_r: np.ndarray) -> float:
    sl = float(np.mean(ir_l**2) + 1e-18)
    sr = float(np.mean(ir_r**2) + 1e-18)
    return 20.0 * np.log10(np.sqrt(sr) / np.sqrt(sl))


def sphere_triangles(az: np.ndarray, el: np.ndarray) -> list[list[int]]:
    """Delaunay on sphere via ConvexHull of unit vectors."""
    a = np.deg2rad(az)
    e = np.deg2rad(el)
    ce = np.cos(e)
    xyz = np.column_stack([ce * np.cos(a), ce * np.sin(a), np.sin(e)])
    hull = ConvexHull(xyz)
    return hull.simplices.astype(int).tolist()


def diffuse_field(mags_L: np.ndarray, mags_R: np.ndarray, az: np.ndarray, el: np.ndarray) -> dict:
    """Area-weighted power mean over directions (approx. by solid-angle ∝ cos(el))."""
    w = np.maximum(np.cos(np.deg2rad(el)), 0.05)
    w = w / w.sum()
    lin_L = 10.0 ** (mags_L / 10.0)  # power
    lin_R = 10.0 ** (mags_R / 10.0)
    mean_L = 10.0 * np.log10(np.maximum(np.sum(lin_L * w[:, None], axis=0), 1e-18))
    mean_R = 10.0 * np.log10(np.maximum(np.sum(lin_R * w[:, None], axis=0), 1e-18))
    return {"L": mean_L.tolist(), "R": mean_R.tolist()}


def pack_dataset(name: str, ir: np.ndarray, sp: np.ndarray, fs: float):
    """Build per-direction spectral entries + IR bank."""
    npos = ir.shape[0]
    nsamp = ir.shape[-1]
    az = wrap_az(sp[:, 0])
    el = sp[:, 1].astype(float)

    entries = []
    mags_L = np.zeros((npos, NF))
    mags_R = np.zeros((npos, NF))
    itds = np.zeros(npos)
    ilds = np.zeros(npos)
    for i in range(npos):
        mL = mag_db_on_grid(ir[i, 0], fs, FREQS)
        mR = mag_db_on_grid(ir[i, 1], fs, FREQS)
        mags_L[i] = mL
        mags_R[i] = mR
        itds[i] = xcorr_itd_us(ir[i, 0], ir[i, 1], fs)
        ilds[i] = ild_db(ir[i, 0], ir[i, 1])
        entries.append(
            {
                "az": float(az[i]),
                "el": float(el[i]),
                "L": mL.tolist(),
                "R": mR.tolist(),
                "itd": float(itds[i]),
                "ild": float(ilds[i]),
            }
        )

    # IR bank: int16 interleaved L,R per position, base64
    peak = float(np.max(np.abs(ir)) + 1e-12)
    scale = peak
    q = np.empty((npos * 2, nsamp), dtype=np.int16)
    for i in range(npos):
        q[i * 2] = np.clip(np.round(ir[i, 0] / scale * 32767), -32767, 32767).astype(np.int16)
        q[i * 2 + 1] = np.clip(np.round(ir[i, 1] / scale * 32767), -32767, 32767).astype(np.int16)
    b64 = base64.b64encode(q.tobytes()).decode("ascii")

    ird = {
        "npos": int(npos),
        "nsamp": int(nsamp),
        "scale": scale,
        "b64": b64,
        "itd": itds.astype(float).tolist(),
    }
    tris = sphere_triangles(az, el)
    dif = diffuse_field(mags_L, mags_R, az, el)
    print(f"  {name}: N={npos}, nsamp={nsamp}, tris={len(tris)}, peak={scale:.4f}")
    return entries, tris, dif, ird, mags_L, mags_R


def main():
    print("Loading SOFAs…")
    ear = sofar.read_sofa(str(EARSIM), verify=False)
    bte = sofar.read_sofa(str(BTE), verify=False)
    idx_a, idx_b = common_indices(ear.SourcePosition, bte.SourcePosition)
    print(f"Matched directions: {len(idx_a)}")

    ir_e = np.asarray(ear.Data_IR)[idx_a]
    sp_e = np.asarray(ear.SourcePosition)[idx_a]
    ir_b = np.asarray(bte.Data_IR)[idx_b]
    sp_b = np.asarray(bte.SourcePosition)[idx_b]
    fs = float(ear.Data_SamplingRate)

    print("Packing EarSim (→ meta)…")
    meta, tri_m, dif_m, ird_m, mL_e, mR_e = pack_dataset("earsim", ir_e, sp_e, fs)
    print("Packing BTE (→ soni)…")
    soni, tri_s, dif_s, ird_s, mL_b, mR_b = pack_dataset("bte", ir_b, sp_b, fs)

    # Level offset EarSim − BTE (mean over ears/dirs/freqs in 200–16 kHz)
    fmask = (FREQS >= 200) & (FREQS <= 16000)
    lvl = float(
        np.mean(
            [
                np.mean(mL_e[:, fmask] - mL_b[:, fmask]),
                np.mean(mR_e[:, fmask] - mR_b[:, fmask]),
            ]
        )
    )

    all_mag = np.concatenate([mL_e, mR_e, mL_b, mR_b])
    mag_min = float(np.floor(np.percentile(all_mag, 1) / 5) * 5)
    mag_max = float(np.ceil(np.percentile(all_mag, 99) / 5) * 5)

    data_all = {
        "freqs": FREQS.tolist(),
        "meta": meta,
        "soni": soni,
        "diffuse": {"meta": dif_m, "soni": dif_s},
        "tri_m": tri_m,
        "tri_s": tri_s,
        "ranges": {
            "mag": [mag_min, mag_max],
            "dif": 40.0,
            "norm": 40.0,
            "lvl": lvl,
        },
        "labels": {
            "meta": "EarSim (canal)",
            "soni": "BTE (Quest3 + N5)",
        },
        "fs": fs,
        "note": "Windowed (with ITD) 48 kHz, location-matched; explorer keys meta=EarSim, soni=BTE",
    }
    ird = {"meta": ird_m, "soni": ird_s}

    p1 = OUT / "data_all.json"
    p2 = OUT / "ird.json"
    p1.write_text(json.dumps(data_all, separators=(",", ":")), encoding="utf-8")
    p2.write_text(json.dumps(ird, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {p1} ({p1.stat().st_size/1e6:.2f} MB)")
    print(f"Wrote {p2} ({p2.stat().st_size/1e6:.2f} MB)")
    print(f"Level offset EarSim−BTE (200 Hz–16 kHz): {lvl:.2f} dB")


if __name__ == "__main__":
    main()
