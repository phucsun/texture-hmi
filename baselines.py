#!/usr/bin/env python3
"""
baselines.py — sinh cac texture baseline tu texture baked cua backbone.

Dau vao : results_C4_final/<subj>/<subj>.png   (T_partial, 256x256)
Dau ra  : baselines/<vid>/mirroring.png
          baselines/<vid>/telea.png
          baselines/<vid>/navier_stokes.png
          baselines/<vid>/invalid_mask.png
          baselines/coverage.csv               ty le texel khong duoc quan sat

Vung khong duoc quan sat duoc xac dinh bang nguong toi tren atlas baked: DECA
de nguyen mau den o texel khong co quan sat hop le.

Ghi chu ve mirroring: truc doi xung cua atlas FLAME nam o giua anh (da kiem
chung bang cach do tuong quan giua atlas va ban lat cua no). Sau khi lay guong,
phan con lai chua duoc phu se duoc lap bang Telea ban kinh nho, de baseline nay
cung cho ra mot texture DAY DU — neu khong, so sanh se bi lech vi con lo hong.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DECA_DIR = os.path.join(BASE, "results_C4_final")
OUT_DIR = os.path.join(BASE, "baselines")


def subject_dirs() -> dict:
    out = {}
    if not os.path.isdir(DECA_DIR):
        sys.exit(f"Khong thay {DECA_DIR}")
    for d in sorted(os.listdir(DECA_DIR)):
        if d.startswith("V") and os.path.isdir(os.path.join(DECA_DIR, d)):
            out[d[:7]] = d
    return out


def invalid_mask(bgr: np.ndarray, thr: int) -> np.ndarray:
    """True o texel khong duoc quan sat (gan den tren atlas baked)."""
    return bgr.max(axis=2) < thr


def fill_mirror(bgr: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Lay guong ngang: texel thieu nhan gia tri tu texel doi xung neu no hop le."""
    out = bgr.copy()
    m_img = bgr[:, ::-1]
    m_inv = invalid[:, ::-1]
    take = invalid & (~m_inv)
    out[take] = m_img[take]
    return out


def inpaint(bgr: np.ndarray, invalid: np.ndarray, method: str,
            radius: int = 3) -> np.ndarray:
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(bgr, invalid.astype(np.uint8) * 255, radius, flag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dark-threshold", type=int, default=12,
                    help="nguong coi la texel khong quan sat duoc")
    ap.add_argument("--radius", type=int, default=3, help="ban kinh inpaint")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subs = subject_dirs()
    vids = args.subjects or sorted(subs)
    if args.limit:
        vids = vids[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"Sinh baseline cho {len(vids)} subject")

    rows, n_ok = [], 0
    for i, vid in enumerate(vids, 1):
        sdir = subs.get(vid)
        if sdir is None:
            print(f"[{i}/{len(vids)}] {vid}: khong co -> bo qua")
            continue
        src = os.path.join(DECA_DIR, sdir, f"{sdir}.png")
        if not os.path.isfile(src):
            print(f"[{i}/{len(vids)}] {vid}: thieu {src} -> bo qua")
            continue

        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{i}/{len(vids)}] {vid}: khong doc duoc anh -> bo qua")
            continue

        inv = invalid_mask(bgr, args.dark_threshold)
        frac = float(inv.mean())

        d = os.path.join(args.out, vid)
        os.makedirs(d, exist_ok=True)

        if args.overwrite or not os.path.isfile(os.path.join(d, "mirroring.png")):
            mir = fill_mirror(bgr, inv)
            rest = invalid_mask(mir, args.dark_threshold)
            if rest.any():                       # phan doi xung khong phu duoc
                mir = inpaint(mir, rest, "telea", args.radius)
            cv2.imwrite(os.path.join(d, "mirroring.png"), mir)

        for name in ("telea", "navier_stokes"):
            p = os.path.join(d, f"{name}.png")
            if args.overwrite or not os.path.isfile(p):
                cv2.imwrite(p, inpaint(bgr, inv, name, args.radius))

        cv2.imwrite(os.path.join(d, "invalid_mask.png"),
                    inv.astype(np.uint8) * 255)

        rows.append(dict(vid=vid, unobserved_frac=round(frac, 6),
                         atlas_h=bgr.shape[0], atlas_w=bgr.shape[1]))
        n_ok += 1
        if i % 10 == 0 or i == len(vids):
            print(f"  [{i}/{len(vids)}] {vid}  khong quan sat = {frac*100:.1f}%")

    cov = os.path.join(args.out, "coverage.csv")
    with open(cov, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vid", "unobserved_frac",
                                          "atlas_h", "atlas_w"])
        w.writeheader()
        w.writerows(rows)

    if rows:
        fr = np.array([r["unobserved_frac"] for r in rows])
        print(f"\nXong {n_ok} subject.")
        print(f"Ty le texel khong duoc quan sat: "
              f"trung binh {fr.mean()*100:.1f}%  "
              f"khoang [{fr.min()*100:.1f}%, {fr.max()*100:.1f}%]")
        print(f"  -> dung cho o \\PH{{XX--XX\\%}} trong bai bao")
        print(f"Ghi: {cov}")


if __name__ == "__main__":
    main()
