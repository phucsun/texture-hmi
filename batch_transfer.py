"""
batch_transfer.py — Chạy pipeline align_and_transfer.py hàng loạt
cho tất cả subject có mặt trong cả results_C4_final (DECA) lẫn uv-idm-output.

Cấu trúc đầu vào:
  results_C4_final/V000XXX_v000XXX_s002_l2_e01_c4/   ← DECA mesh (target)
  uv-idm-output/output_vkist_mesh/V000XXX_S002_L2_E01_C4.png.obj  ← UV-IDM mesh (source)
  uv-idm-output/output_vkist/V000XXX_S002_L2_E01_C4.png           ← UV-IDM texture

Đầu ra:
  output_batch/V000XXX/deca_texture.png
  output_batch/V000XXX/deca_textured.obj
  output_batch/V000XXX/deca_textured.mtl
  output_batch/V000XXX/FLAME_albedo_from_BFM.npz
"""

import os
import subprocess
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(BASE_DIR, "results_C4_final")
UVIDM_MESH_DIR = os.path.join(BASE_DIR, "uv-idm-output", "output_vkist_mesh")
UVIDM_TEX_DIR  = os.path.join(BASE_DIR, "uv-idm-output", "output_vkist_uv")
OUTPUT_BASE    = os.path.join(BASE_DIR, "output_batch")
SCRIPT         = os.path.join(BASE_DIR, "align_and_transfer.py")
PYTHON         = sys.executable   # dùng Python đang chạy batch script này

# ─────────────────────────────────────────────────────────────────────────────
# TÌM CÁC SUBJECT KHỚP
# ─────────────────────────────────────────────────────────────────────────────

# DECA: mỗi thư mục có dạng V000XXX_v000XXX_s002_l2_e01_c4
deca_dirs = {}
for d in sorted(os.listdir(RESULTS_DIR)):
    full = os.path.join(RESULTS_DIR, d)
    if os.path.isdir(full) and d.startswith("V"):
        vid = d[:7]   # V000XXX
        deca_dirs[vid] = d

# UV-IDM mesh: V000XXX_S002_L2_E01_C4.png.obj
uvidm_meshes = {}
for f in sorted(os.listdir(UVIDM_MESH_DIR)):
    if f.endswith(".obj") and f.startswith("V"):
        vid = f[:7]
        uvidm_meshes[vid] = f

common = sorted(set(deca_dirs.keys()) & set(uvidm_meshes.keys()))
only_deca  = sorted(set(deca_dirs.keys())  - set(uvidm_meshes.keys()))
only_uvidm = sorted(set(uvidm_meshes.keys()) - set(deca_dirs.keys()))

print("=" * 60)
print(f"  Batch texture transfer  —  {len(common)} subjects")
print("=" * 60)
if only_deca:
    print(f"  [INFO] Chỉ có trong DECA (bỏ qua): {only_deca}")
if only_uvidm:
    print(f"  [INFO] Chỉ có trong UV-IDM (bỏ qua): {only_uvidm}")
print()

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHẠY TỪNG SUBJECT
# ─────────────────────────────────────────────────────────────────────────────
results = {}   # vid → "ok" | "skip" | "error"
t_total = time.time()

for idx, vid in enumerate(common, 1):
    deca_dir  = deca_dirs[vid]          # e.g. V000190_v000190_s002_l2_e01_c4
    mesh_file = uvidm_meshes[vid]       # e.g. V000190_S002_L2_E01_C4.png.obj
    tex_file  = mesh_file[:-4]          # strip .obj → V000190_S002_L2_E01_C4.png

    src_obj = os.path.join(UVIDM_MESH_DIR, mesh_file)
    src_tex = os.path.join(UVIDM_TEX_DIR,  tex_file)
    tgt_obj = os.path.join(RESULTS_DIR, deca_dir, f"{deca_dir}.obj")
    out_dir = os.path.join(OUTPUT_BASE, vid)

    print(f"[{idx:3d}/{len(common)}] {vid}")
    print(f"  src_obj : {os.path.relpath(src_obj, BASE_DIR)}")
    print(f"  src_tex : {os.path.relpath(src_tex, BASE_DIR)}")
    print(f"  tgt_obj : {os.path.relpath(tgt_obj, BASE_DIR)}")
    print(f"  out_dir : {os.path.relpath(out_dir, BASE_DIR)}")

    # Kiểm tra file tồn tại
    missing = [p for p in (src_obj, src_tex, tgt_obj) if not os.path.isfile(p)]
    if missing:
        print(f"  [SKIP] File không tồn tại: {[os.path.basename(p) for p in missing]}")
        results[vid] = "skip"
        print()
        continue

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON, SCRIPT,
        "--src-obj", src_obj,
        "--src-tex", src_tex,
        "--tgt-obj", tgt_obj,
        "--out-dir", out_dir,
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"  [OK] Done in {elapsed:.0f}s")
        results[vid] = "ok"
    else:
        print(f"  [ERROR] returncode={proc.returncode}  elapsed={elapsed:.0f}s")
        results[vid] = "error"
    print()

# ─────────────────────────────────────────────────────────────────────────────
# TỔNG KẾT
# ─────────────────────────────────────────────────────────────────────────────
ok    = [v for v, r in results.items() if r == "ok"]
skip  = [v for v, r in results.items() if r == "skip"]
error = [v for v, r in results.items() if r == "error"]

print("=" * 60)
print("  KẾT QUẢ BATCH")
print("=" * 60)
print(f"  Thành công : {len(ok)}/{len(common)}")
print(f"  Bỏ qua     : {len(skip)}")
print(f"  Lỗi        : {len(error)}")
if error:
    print(f"  Subject lỗi: {error}")
print(f"  Tổng thời gian : {(time.time()-t_total)/60:.1f} phút")
print(f"  Kết quả lưu tại: {OUTPUT_BASE}")
print("=" * 60)
