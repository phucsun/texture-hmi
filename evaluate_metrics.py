"""
evaluate_metrics.py — Compare DECA vs HMI generated 2D images against Ground Truth.

Pipeline per subject:
  1. Load generated image from HMI folder.
  2. Load generated image from DECA folder.
  3. Load ground truth image (C7 angle preferred).
  4. Detect face in ground truth and crop to target image size (e.g., 512x512).
  5. Compute PSNR / SSIM / LPIPS.
  6. Save per-subject 3-panel comparison image.
  7. Write CSV + HTML report.
"""

import argparse
import gc
import os
import time
import csv
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.seterr(all="ignore")
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

# Bật tăng tốc phần cứng cho LPIPS
try:
    import torch
    import lpips as lpips_lib
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    LPIPS_NET = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    HAS_LPIPS = True
    print(f"[INFO] LPIPS loaded (AlexNet) on {device}")
except Exception as e:
    HAS_LPIPS = False
    device = None
    print(f"[WARN] LPIPS not available: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--dir-hmi",  default="hmi", help="Directory containing HMI generated images")
_p.add_argument("--dir-deca", default="deca", help="Directory containing DECA generated images")
_p.add_argument("--dir-gt",   default="polyface_09042026", help="Directory containing Ground Truth images")
_p.add_argument("--limit",    type=int,   default=0,   help="Process only N subjects (0=all)")
_p.add_argument("--img-size", type=int,   default=512, help="Evaluation and crop resolution")
_p.add_argument("--out-dir",  default="eval_report",   help="Report output directory")
ARGS = _p.parse_args()

IMG_SIZE    = ARGS.img_size
REPORT_DIR  = ARGS.out_dir
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "subjects"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FACE CROP from ground truth
# ─────────────────────────────────────────────────────────────────────────────

_HAAR = None

def load_haar():
    global _HAAR
    if _HAAR is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _HAAR = cv2.CascadeClassifier(cascade_path)
    return _HAAR

def crop_face_from_image(img_bgr, target_size, padding=0.30):
    """Detect + crop face from BGR image. Falls back to centre-crop if detection fails."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = load_haar()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) == 0:
        # Fallback: crop centre square
        s  = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        crop = img_bgr[y0:y0+s, x0:x0+s]
    else:
        # Use largest detected face
        areas = [w * h for (x, y, w, h) in faces]
        x, y, w, h = faces[int(np.argmax(areas))]
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(W, x + w + pad_w)
        y2 = min(H, y + h + pad_h)
        # Square crop
        sw = x2 - x1; sh = y2 - y1
        s  = max(sw, sh)
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        x1 = max(0, cx - s // 2); y1 = max(0, cy - s // 2)
        x2 = min(W, x1 + s);      y2 = min(H, y1 + s)
        crop = img_bgr[y1:y2, x1:x2]

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    if mse < 1e-8: return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))

def compute_ssim(pred, gt):
    return float(skimage_ssim(pred, gt, channel_axis=2, data_range=255, win_size=11))

def compute_lpips(pred, gt):
    if not HAS_LPIPS: return float("nan")
    def to_tensor(img):
        t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        d = LPIPS_NET(to_tensor(pred), to_tensor(gt))
    return float(d.item())

# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_subjects():
    """Matches subjects across HMI, DECA, and Ground Truth folders."""
    subjects = []
    
    if not os.path.exists(ARGS.dir_hmi):
        print(f"[ERROR] HMI directory not found: {ARGS.dir_hmi}")
        return subjects

    for fname in sorted(os.listdir(ARGS.dir_hmi)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        vid = os.path.splitext(fname)[0]
        hmi_path = os.path.join(ARGS.dir_hmi, fname)
        
        # Look for matching DECA image
        deca_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            cand = os.path.join(ARGS.dir_deca, vid + ext)
            if os.path.isfile(cand):
                deca_path = cand
                break
                
        if not deca_path:
            continue

        # Look for Ground Truth (Prefer C7, fallback to C4)
        gt_path = None
        vid_folder = os.path.join(ARGS.dir_gt, vid)
        if os.path.isdir(vid_folder):
            for cam in ("C7", "C4"): 
                for ext in (".JPG", ".jpg", ".jpeg", ".png"):
                    cand = os.path.join(vid_folder, f"{vid}_S002_L2_E01_{cam}{ext}")
                    if os.path.isfile(cand):
                        gt_path = cand
                        break
                if gt_path: break

        subjects.append({
            "vid": vid,
            "hmi_path": hmi_path,
            "deca_path": deca_path,
            "gt_path": gt_path,
        })
        
    return subjects

# ─────────────────────────────────────────────────────────────────────────────
# PANEL IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def make_panel(deca_img, hmi_img, gt_img, vid, title=""):
    S = IMG_SIZE
    H_pad = 30
    panel = np.ones((S + H_pad, S * 3, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    def place_img(img, col, label):
        x0 = col * S
        if img is not None:
            img_resized = cv2.resize(img, (S, S))
            panel[H_pad:, x0:x0+S] = img_resized
        else:
            panel[H_pad:, x0:x0+S] = 128
        cv2.putText(panel, label, (x0 + 4, H_pad - 8), font, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

    place_img(deca_img, 0, "DECA")
    place_img(hmi_img, 1, "Ours (HMI)")
    place_img(gt_img, 2, "Ground Truth")

    if title:
        cv2.putText(panel, title, (4, 18), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return panel

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    subjects = find_subjects()
    if ARGS.limit > 0:
        subjects = subjects[:ARGS.limit]

    print("=" * 70)
    print(f"  Evaluation: {len(subjects)} subjects  |  img_size={IMG_SIZE}")
    print("=" * 70)

    csv_path  = os.path.join(REPORT_DIR, "metrics.csv")
    csv_rows  = []
    csv_fields = ["vid", "deca_psnr", "deca_ssim", "deca_lpips",
                  "ours_psnr", "ours_ssim", "ours_lpips",
                  "delta_psnr", "delta_ssim", "delta_lpips", "gt_available"]

    t_total = time.time()

    for idx, subj in enumerate(subjects, 1):
        vid = subj["vid"]
        t0  = time.time()
        print(f"\n[{idx:3d}/{len(subjects)}] {vid}", flush=True)

        # ── Load Generated Images ─────────────────────────────────────────────
        try:
            hmi_img = np.array(Image.open(subj["hmi_path"]).convert("RGB"))
            deca_img = np.array(Image.open(subj["deca_path"]).convert("RGB"))
            
            # Ensure size match
            if hmi_img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                hmi_img = cv2.resize(hmi_img, (IMG_SIZE, IMG_SIZE))
            if deca_img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                deca_img = cv2.resize(deca_img, (IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"  [SKIP] Load generated image error: {e}")
            continue

        # ── Ground Truth Crop ─────────────────────────────────────────────────
        gt_rgb = None
        gt_available = False
        if subj["gt_path"] and os.path.isfile(subj["gt_path"]):
            try:
                gt_bgr = cv2.imread(subj["gt_path"])
                gt_crop = crop_face_from_image(gt_bgr, IMG_SIZE)
                gt_rgb = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2RGB)
                gt_available = True
                print(f"  GT loaded and cropped: {os.path.basename(subj['gt_path'])}", flush=True)
            except Exception as e:
                print(f"  [WARN] GT load/crop failed: {e}")

        # ── Compute Metrics ───────────────────────────────────────────────────
        row = {"vid": vid, "gt_available": gt_available}
        if gt_available:
            row["deca_psnr"]  = compute_psnr(deca_img, gt_rgb)
            row["deca_ssim"]  = compute_ssim(deca_img, gt_rgb)
            row["deca_lpips"] = compute_lpips(deca_img, gt_rgb)
            
            row["ours_psnr"]  = compute_psnr(hmi_img, gt_rgb)
            row["ours_ssim"]  = compute_ssim(hmi_img, gt_rgb)
            row["ours_lpips"] = compute_lpips(hmi_img, gt_rgb)
            
            row["delta_psnr"]  = row["ours_psnr"]  - row["deca_psnr"]
            row["delta_ssim"]  = row["ours_ssim"]  - row["deca_ssim"]
            row["delta_lpips"] = row["ours_lpips"] - row["deca_lpips"]
            
            print(f"  PSNR  DECA={row['deca_psnr']:.2f} Ours={row['ours_psnr']:.2f}  Δ={row['delta_psnr']:+.2f}", flush=True)
            print(f"  SSIM  DECA={row['deca_ssim']:.4f} Ours={row['ours_ssim']:.4f}  Δ={row['delta_ssim']:+.4f}", flush=True)
            print(f"  LPIPS DECA={row['deca_lpips']:.4f} Ours={row['ours_lpips']:.4f}  Δ={row['delta_lpips']:+.4f}", flush=True)
        else:
            for k in ["deca_psnr", "deca_ssim", "deca_lpips", "ours_psnr", "ours_ssim", "ours_lpips", "delta_psnr", "delta_ssim", "delta_lpips"]:
                row[k] = float("nan")
            print("  [WARN] No GT → metrics skipped")

        csv_rows.append(row)

        # ── Save Panel Image ──────────────────────────────────────────────────
        title = f"{vid}  |  PSNR: DECA={row.get('deca_psnr', float('nan')):.2f} Ours={row.get('ours_psnr', float('nan')):.2f}"
        panel = make_panel(deca_img, hmi_img, gt_rgb, vid, title)
        panel_path = os.path.join(REPORT_DIR, "subjects", f"{vid}.jpg")
        cv2.imwrite(panel_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  →  {panel_path}", flush=True)
        gc.collect()

    # ── Write CSV & HTML ──────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]) for k in csv_fields if k in row})
    
    valid = [r for r in csv_rows if r["gt_available"]]
    generate_html_report(csv_rows, valid)
    
    print("\n" + "=" * 70)
    print(f"  DONE! Processed {len(csv_rows)} subjects. Time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Reports saved to: {REPORT_DIR}/")
    print("=" * 70)

def generate_html_report(csv_rows, valid_rows):
    agg = {}
    for metric in ["psnr", "ssim", "lpips"]:
        dv = [r[f"deca_{metric}"] for r in valid_rows if not math.isnan(r.get(f"deca_{metric}", float("nan")))]
        ov = [r[f"ours_{metric}"] for r in valid_rows if not math.isnan(r.get(f"ours_{metric}", float("nan")))]
        agg[f"deca_{metric}"] = np.mean(dv) if dv else float("nan")
        agg[f"ours_{metric}"] = np.mean(ov) if ov else float("nan")
        agg[f"delta_{metric}"] = agg[f"ours_{metric}"] - agg[f"deca_{metric}"]

    def fmt(v, decimals=4): return f"{v:.{decimals}f}" if not math.isnan(v) else "—"
    def delta_style(v, higher_better=True):
        if math.isnan(v): return "color:#888"
        return "color:green;font-weight:bold" if (v > 0) == higher_better else "color:red"

    table_rows = ""
    for r in csv_rows:
        gt_tag = "✓" if r["gt_available"] else "✗"
        vid = r["vid"]
        img_rel = f"subjects/{vid}.jpg"
        dP, dS, dL = r.get("delta_psnr", float("nan")), r.get("delta_ssim", float("nan")), r.get("delta_lpips", float("nan"))

        table_rows += f"""
        <tr>
          <td><a href="{img_rel}">{vid}</a></td><td>{gt_tag}</td>
          <td>{fmt(r.get("deca_psnr", float("nan")), 2)}</td><td>{fmt(r.get("deca_ssim", float("nan")), 4)}</td><td>{fmt(r.get("deca_lpips", float("nan")), 4)}</td>
          <td>{fmt(r.get("ours_psnr", float("nan")), 2)}</td><td>{fmt(r.get("ours_ssim", float("nan")), 4)}</td><td>{fmt(r.get("ours_lpips", float("nan")), 4)}</td>
          <td style="{delta_style(dP)}">{fmt(dP,2)}</td><td style="{delta_style(dS)}">{fmt(dS,4)}</td><td style="{delta_style(dL, False)}">{fmt(dL,4)}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>2D Texture Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
    th {{ background: #f2f2f2; text-align: center; }}
    .agg {{ background: #e8f4e8; font-weight: bold; }}
    img.sample {{ max-width: 100%; border: 1px solid #ccc; margin: 4px 0; }}
  </style>
</head>
<body>
<h1>2D Evaluation: DECA vs Ours (HMI)</h1>
<h2>Aggregate Results (n={len(valid_rows)})</h2>
<table>
  <tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>
  <tr><td>DECA</td><td>{fmt(agg.get('deca_psnr', float('nan')), 2)}</td><td>{fmt(agg.get('deca_ssim', float('nan')), 4)}</td><td>{fmt(agg.get('deca_lpips', float('nan')), 4)}</td></tr>
  <tr class="agg"><td>Ours (HMI)</td><td>{fmt(agg.get('ours_psnr', float('nan')), 2)}</td><td>{fmt(agg.get('ours_ssim', float('nan')), 4)}</td><td>{fmt(agg.get('ours_lpips', float('nan')), 4)}</td></tr>
  <tr><td>Δ (Ours − DECA)</td><td style="{delta_style(agg.get('delta_psnr', float('nan')))}">{fmt(agg.get('delta_psnr', float('nan')), 2)}</td><td style="{delta_style(agg.get('delta_ssim', float('nan')))}">{fmt(agg.get('delta_ssim', float('nan')), 4)}</td><td style="{delta_style(agg.get('delta_lpips', float('nan')), False)}">{fmt(agg.get('delta_lpips', float('nan')), 4)}</td></tr>
</table>
<h2>Per-Subject Details</h2>
<table>
  <tr><th>Subject</th><th>GT</th><th colspan="3">DECA</th><th colspan="3">Ours</th><th colspan="3">Δ (Ours−DECA)</th></tr>
  <tr><th></th><th></th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>ΔPSNR</th><th>ΔSSIM</th><th>ΔLPIPS</th></tr>
  {table_rows}
</table>
{''.join(f'<div><img class="sample" src="subjects/{r["vid"]}.jpg"><br><small>{r["vid"]}</small></div>' for r in csv_rows[:20])}
</body>
</html>"""
    with open(os.path.join(REPORT_DIR, "report.html"), "w", encoding="utf-8") as f: f.write(html)

if __name__ == "__main__":
    main()