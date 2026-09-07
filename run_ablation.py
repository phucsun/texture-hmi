"""
run_ablation.py — Chạy ablation study trên N subjects đầu tiên.

Các variant:
  full        : pipeline đầy đủ (baseline của mình)
  no_icp      : --no-icp  (chỉ Procrustes, bỏ 3-pass ICP)
  hard_gate   : --hard-gate 0.03  (hard distance gate thay vì soft-confidence)
  no_gauss    : --no-gauss-blend  (bỏ Pass C Gaussian blending)

Output:
  ablation_output/{vid}/{variant}/deca_texture.png
  ablation_output/ablation_metrics.csv
"""

import os
import subprocess
import sys
import time
import csv
import numpy as np
from PIL import Image

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(BASE_DIR, "results_C4_final")
UVIDM_MESH_DIR = os.path.join(BASE_DIR, "uv-idm-output", "output_vkist_mesh")
UVIDM_TEX_DIR  = os.path.join(BASE_DIR, "uv-idm-output", "output_vkist_uv")
POLYFACE_DIR   = os.path.join(BASE_DIR, "polyface_09042026")
OUTPUT_BASE    = os.path.join(BASE_DIR, "ablation_output")
SCRIPT         = os.path.join(BASE_DIR, "align_and_transfer.py")
PYTHON         = sys.executable

N_SUBJECTS     = None  # None = tất cả subjects
HARD_GATE_DIST = 0.03

VARIANTS = {
    "full":      [],
    "no_icp":    ["--no-icp"],
    "hard_gate": ["--hard-gate", str(HARD_GATE_DIST)],
    "no_gauss":  ["--no-gauss-blend"],
}

# ─────────────────────────────────────────────────────────────────────────────
# TÌM SUBJECTS
# ─────────────────────────────────────────────────────────────────────────────
deca_dirs, uvidm_meshes = {}, {}
for d in sorted(os.listdir(RESULTS_DIR)):
    if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d.startswith("V"):
        deca_dirs[d[:7]] = d
for f in sorted(os.listdir(UVIDM_MESH_DIR)):
    if f.endswith(".obj") and f.startswith("V"):
        uvidm_meshes[f[:7]] = f

common = sorted(set(deca_dirs) & set(uvidm_meshes))
if N_SUBJECTS is not None:
    common = common[:N_SUBJECTS]
print(f"Ablation on {len(common)} subjects: {common}")
os.makedirs(OUTPUT_BASE, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = ((img1.astype(np.float32) - img2.astype(np.float32)) ** 2).mean()
    return float(10 * np.log10(255**2 / (mse + 1e-10)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, channel_axis=2, data_range=255))
    except ImportError:
        return float("nan")


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        import torch
        import lpips
        fn = compute_lpips._fn if hasattr(compute_lpips, "_fn") else None
        if fn is None:
            fn = lpips.LPIPS(net="alex")
            compute_lpips._fn = fn
        to_t = lambda x: torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            return float(fn(to_t(img1), to_t(img2)).item())
    except Exception:
        return float("nan")


def load_gt(vid: str) -> np.ndarray | None:
    """Load frontal (C7) ground-truth image for this subject."""
    subj_dir = os.path.join(POLYFACE_DIR, vid)
    if not os.path.isdir(subj_dir):
        return None
    for f in os.listdir(subj_dir):
        if "C7" in f and f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = np.array(Image.open(os.path.join(subj_dir, f)).convert("RGB"))
            return img
    return None


def render_frontal(obj_path: str, tex_path: str, size: int = 512) -> np.ndarray | None:
    """
    Simple UV-space frontal render: return the texture image itself resized.
    (A full 3-D renderer is not available here; the texture atlas is used as proxy
    for metric comparison, which is consistent with evaluate_metrics.py behaviour.)
    """
    if not os.path.isfile(tex_path):
        return None
    img = np.array(Image.open(tex_path).convert("RGB").resize((size, size)))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# CHẠY ABLATION
# ─────────────────────────────────────────────────────────────────────────────
rows = []

for vid in common:
    deca_dir  = deca_dirs[vid]
    mesh_file = uvidm_meshes[vid]
    tex_file  = mesh_file[:-4]

    src_obj = os.path.join(UVIDM_MESH_DIR, mesh_file)
    src_tex = os.path.join(UVIDM_TEX_DIR,  tex_file)
    tgt_obj = os.path.join(RESULTS_DIR, deca_dir, f"{deca_dir}.obj")

    missing = [p for p in (src_obj, src_tex, tgt_obj) if not os.path.isfile(p)]
    if missing:
        print(f"[SKIP] {vid} — missing: {[os.path.basename(p) for p in missing]}")
        continue

    gt_img = load_gt(vid)
    print(f"\n{'='*60}")
    print(f"  {vid}  (GT: {'found' if gt_img is not None else 'NOT FOUND'})")
    print(f"{'='*60}")

    for variant_name, extra_flags in VARIANTS.items():
        out_dir = os.path.join(OUTPUT_BASE, vid, variant_name)
        os.makedirs(out_dir, exist_ok=True)
        tex_out = os.path.join(out_dir, "deca_texture.png")

        if os.path.isfile(tex_out):
            print(f"  [{variant_name}] already done, skipping run")
        else:
            cmd = [
                PYTHON, SCRIPT,
                "--src-obj", src_obj,
                "--src-tex", src_tex,
                "--tgt-obj", tgt_obj,
                "--out-dir", out_dir,
            ] + extra_flags

            print(f"  [{variant_name}] running: {' '.join(extra_flags) if extra_flags else '(full)'}")
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=BASE_DIR,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True)
            elapsed = time.time() - t0
            if proc.returncode != 0:
                print(f"  [{variant_name}] FAILED in {elapsed:.0f}s")
                print(proc.stdout[-800:] if proc.stdout else "")
                rows.append({"vid": vid, "variant": variant_name,
                             "psnr": "err", "ssim": "err", "lpips": "err"})
                continue
            print(f"  [{variant_name}] OK  {elapsed:.0f}s")

        # ── metrics ────────────────────────────────────────────────────────
        pred_img = render_frontal(
            os.path.join(out_dir, "deca_textured.obj"), tex_out)

        if gt_img is not None and pred_img is not None:
            h = min(gt_img.shape[0], pred_img.shape[0])
            w = min(gt_img.shape[1], pred_img.shape[1])
            gt_r   = np.array(Image.fromarray(gt_img).resize((w, h)))
            pred_r = np.array(Image.fromarray(pred_img).resize((w, h)))
            psnr  = compute_psnr(gt_r, pred_r)
            ssim  = compute_ssim(gt_r, pred_r)
            lpips = compute_lpips(gt_r, pred_r)
            print(f"           PSNR={psnr:.2f}  SSIM={ssim:.3f}  LPIPS={lpips:.3f}")
        else:
            psnr = ssim = lpips = float("nan")

        rows.append({"vid": vid, "variant": variant_name,
                     "psnr": round(psnr, 3), "ssim": round(ssim, 4),
                     "lpips": round(lpips, 4)})

# ─────────────────────────────────────────────────────────────────────────────
# XUẤT CSV
# ─────────────────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_BASE, "ablation_metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["vid", "variant", "psnr", "ssim", "lpips"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nAblation metrics saved → {csv_path}")

# ── tóm tắt theo variant ─────────────────────────────────────────────────────
print("\n── Summary (mean over subjects) ──")
for vname in VARIANTS:
    vrows = [r for r in rows if r["variant"] == vname
             and isinstance(r["psnr"], float) and not np.isnan(r["psnr"])]
    if vrows:
        print(f"  {vname:<12}  PSNR={np.mean([r['psnr'] for r in vrows]):.2f}  "
              f"SSIM={np.mean([r['ssim'] for r in vrows]):.3f}  "
              f"LPIPS={np.mean([r['lpips'] for r in vrows]):.3f}")
    else:
        print(f"  {vname:<12}  (no valid rows)")
