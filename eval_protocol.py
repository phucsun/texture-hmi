#!/usr/bin/env python3
"""
eval_protocol.py — do PSNR / SSIM / LPIPS / CSIM giua anh render va anh tham chieu.

Nguyen tac: anh render va anh tham chieu duoc dua ve CUNG mot khung bang cach
dung, khong phai bang can chinh hau ky.
  - anh render: camera dung tu 68 landmark 3D cua mesh, khung chuan hoa theo
    khoang cach lien dong tu  (render_views.py da lam)
  - anh tham chieu: detect 68 landmark 2D roi bien doi similarity ve dung vi tri
    cua landmark da chieu cua anh render
  - vung do: giao cua (mask be mat da render) va (bao loi landmark cua anh tham chieu)
    -> loai toc, quan ao, nen — nhung thu mesh khong bieu dien

Dau ra: results.csv theo dung schema ma check_results.py doc duoc
    vid,view,method,psnr,ssim,lpips,csim,region,q_pred

Vi du:
    python3 eval_protocol.py --methods backbone prior_transfer
    python3 eval_protocol.py --region full_crop --out results_fullcrop.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

import lib_face as LF

BASE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(BASE, "polyface_09042026")
RENDER_DIR = os.path.join(BASE, "renders")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SAU
# ─────────────────────────────────────────────────────────────────────────────
_LPIPS = _CSIM = _DEV = None


def _device():
    global _DEV
    if _DEV is None:
        import torch
        if torch.backends.mps.is_available():
            _DEV = torch.device("mps")
        elif torch.cuda.is_available():
            _DEV = torch.device("cuda")
        else:
            _DEV = torch.device("cpu")
        print(f"  thiet bi: {_DEV}")
    return _DEV


def lpips_model():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(_device()).eval()
    return _LPIPS


def csim_model():
    global _CSIM
    if _CSIM is None:
        from facenet_pytorch import InceptionResnetV1
        _CSIM = InceptionResnetV1(pretrained="vggface2").to(_device()).eval()
    return _CSIM


# ─────────────────────────────────────────────────────────────────────────────
# METRIC (chi tinh trong mask)
# ─────────────────────────────────────────────────────────────────────────────

def psnr_masked(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    d = (a[m].astype(np.float64) - b[m].astype(np.float64)) ** 2
    mse = float(d.mean()) if d.size else float("nan")
    if not np.isfinite(mse):
        return float("nan")
    return 100.0 if mse < 1e-8 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim_masked(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    _, smap = ssim_fn(a, b, channel_axis=2, data_range=255, full=True)
    v = smap.mean(axis=2)[m]
    return float(v.mean()) if v.size else float("nan")


def _bbox(m: np.ndarray, pad: int = 4):
    ys, xs = np.where(m)
    if not len(ys):
        return None
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + 1 + pad)
    return y0, y1, x0, x1


def _prep(img: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    """Zero ngoai mask, cat theo bbox, resize — dung cho LPIPS va CSIM."""
    out = img.copy()
    out[~m] = 0
    bb = _bbox(m)
    if bb:
        y0, y1, x0, x1 = bb
        out = out[y0:y1, x0:x1]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def lpips_masked(a, b, m) -> float:
    import torch
    A, B = _prep(a, m, 256), _prep(b, m, 256)

    def t(x):
        y = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return y.permute(2, 0, 1).unsqueeze(0).to(_device())

    with torch.no_grad():
        return float(lpips_model()(t(A), t(B)).item())


def csim_masked(a, b, m) -> float:
    import torch
    A, B = _prep(a, m, 160), _prep(b, m, 160)

    def t(x):
        y = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return y.permute(2, 0, 1).unsqueeze(0).to(_device())

    with torch.no_grad():
        ea = csim_model()(t(A))
        eb = csim_model()(t(B))
        return float(torch.nn.functional.cosine_similarity(ea, eb).item())


# ─────────────────────────────────────────────────────────────────────────────
# ANH THAM CHIEU
# ─────────────────────────────────────────────────────────────────────────────

def gt_path(vid: str, view: str) -> str | None:
    pats = [os.path.join(GT_DIR, vid, f"*_{view}.JPG"),
            os.path.join(GT_DIR, vid, f"*_{view}.jpg"),
            os.path.join(GT_DIR, vid, f"*_{view}.png")]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def align_gt(vid: str, view: str, size: int, cache: dict, downscale: int = 4):
    """
    Tra ve (anh tham chieu da can chinh RGB, mask vung mat) hoac (None, None).
    Ket qua duoc cache theo (vid,view) vi khong phu thuoc phuong phap.
    """
    key = (vid, view)
    if key in cache:
        return cache[key]

    gp = gt_path(vid, view)
    lp = os.path.join(RENDER_DIR, "_lmk", f"{vid}_{view}.json")
    if gp is None or not os.path.isfile(lp):
        cache[key] = (None, None)
        return cache[key]

    bgr = cv2.imread(gp, cv2.IMREAD_COLOR)
    if bgr is None:
        cache[key] = (None, None)
        return cache[key]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # detect tren ban thu nho cho nhanh, roi dua toa do ve kich thuoc goc
    small = cv2.resize(rgb, (rgb.shape[1] // downscale, rgb.shape[0] // downscale),
                       interpolation=cv2.INTER_AREA)
    lm = LF.detect_landmarks2d(small)
    if lm is None:
        cache[key] = (None, None)
        return cache[key]
    lm = lm[:, :2] * downscale

    dst = np.asarray(LF.load_json(lp)["landmarks"], dtype=np.float64)
    idx = LF.IDX_STABLE                       # bo duong ham, vi no doi theo goc
    M = LF.similarity_transform(lm[idx], dst[idx])
    warped = cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    lm_w = (M[:, :2] @ lm.T).T + M[:, 2]
    face = LF.convex_hull_mask(lm_w, size)
    cache[key] = (warped, face)
    return cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--views", nargs="+", default=list(LF.VIEWS.keys()))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=LF.IMG_SIZE)
    ap.add_argument("--renders", default=RENDER_DIR)
    ap.add_argument("--region", choices=["face_mask", "full_crop"],
                    default="face_mask")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--no-csim", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    # subject: lay tu anh da render cua phuong phap dau tien
    first = os.path.join(args.renders, args.methods[0])
    if not os.path.isdir(first):
        sys.exit(f"Chua co anh render trong {first}. Chay render_views.py truoc.")
    vids = sorted({os.path.basename(p).split("_")[0]
                   for p in glob.glob(os.path.join(first, "V*_*.png"))})
    if args.subjects:
        vids = [v for v in vids if v in args.subjects]
    if args.limit:
        vids = vids[:args.limit]
    if not vids:
        sys.exit("Khong tim thay subject nao da render.")

    print(f"Do {len(vids)} subject x {len(args.views)} view x "
          f"{len(args.methods)} method  |  vung do: {args.region}")
    if not args.no_lpips:
        lpips_model()
    if not args.no_csim:
        csim_model()

    cache: dict = {}
    rows = []
    n_miss_gt = 0
    for i, vid in enumerate(vids, 1):
        for view in args.views:
            gt, face = align_gt(vid, view, args.size, cache)
            if gt is None:
                n_miss_gt += 1
                continue
            mpath = os.path.join(args.renders, "_mask", f"{vid}_{view}.png")
            surf = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 127) \
                if os.path.isfile(mpath) else np.ones((args.size,) * 2, bool)

            mask = (surf & face) if args.region == "face_mask" \
                else np.ones_like(surf, dtype=bool)
            if mask.sum() < 500:
                continue

            for method in args.methods:
                rp = os.path.join(args.renders, method, f"{vid}_{view}.png")
                if not os.path.isfile(rp):
                    continue
                ren = cv2.cvtColor(cv2.imread(rp, cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB)
                if ren.shape[:2] != (args.size, args.size):
                    ren = cv2.resize(ren, (args.size, args.size))

                rows.append(dict(
                    vid=vid, view=view, method=method,
                    psnr=round(psnr_masked(ren, gt, mask), 4),
                    ssim=round(ssim_masked(ren, gt, mask), 4),
                    lpips=("" if args.no_lpips
                           else round(lpips_masked(ren, gt, mask), 4)),
                    csim=("" if args.no_csim
                          else round(csim_masked(ren, gt, mask), 4)),
                    region=args.region, q_pred=""))
        print(f"[{i}/{len(vids)}] {vid}  ({len(rows)} dong)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vid", "view", "method", "psnr", "ssim",
                                          "lpips", "csim", "region", "q_pred"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nGhi {len(rows)} dong -> {args.out}")
    if n_miss_gt:
        print(f"Thieu anh tham chieu hoac khong detect duoc mat: {n_miss_gt} (vid,view)")
    print(f"Kiem tra va sinh bang:  python3 check_results.py {args.out} --latex")


if __name__ == "__main__":
    main()
