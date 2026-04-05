#!/usr/bin/env python3
"""
patch_uv.py — Copy UV coordinates from bfm_to_flame/deca.obj into output/deca.obj
              and write output/deca.mtl so the browser viewer can apply the texture.

Usage:
    python3 patch_uv.py

Requires only Python stdlib — no trimesh / numpy needed.
"""

SRC_OBJ = "bfm_to_flame/deca.obj"   # source DECA mesh (has UV)
OUT_OBJ = "output/deca.obj"          # will be overwritten with UV added
OUT_MTL = "output/deca.mtl"          # will be created / overwritten

# ── 1. Parse UV + per-face UV indices from source OBJ ─────────────────────────
vt_lines    = []   # raw "vt u v" strings
src_face_vi  = []  # (F,3) vertex indices  (1-based)
src_face_vti = []  # (F,3) UV indices      (1-based)

with open(SRC_OBJ, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.rstrip()
        if line.startswith("vt "):
            vt_lines.append(line)
        elif line.startswith("f "):
            tokens = line.split()[1:]
            vis, vtis = [], []
            for tok in tokens[:3]:
                parts = tok.split("/")
                vi  = int(parts[0])
                vti = int(parts[1]) if len(parts) > 1 and parts[1] else vi
                vis.append(vi); vtis.append(vti)
            src_face_vi.append(vis)
            src_face_vti.append(vtis)

if not vt_lines:
    raise SystemExit(f"[ERROR] No UV coords (vt lines) found in {SRC_OBJ}")
print(f"  Source UV    : {len(vt_lines):,} coords from {SRC_OBJ}")
print(f"  Source faces : {len(src_face_vti):,}")

# ── 2. Read existing output/deca.obj — keep only vertex lines + face vi ───────
vert_lines  = []   # raw "v x y z" strings
out_face_vi = []   # (F,3) vertex indices  (1-based)

with open(OUT_OBJ, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.rstrip()
        if line.startswith("v ") and not line.startswith("vt"):
            vert_lines.append(line)
        elif line.startswith("f "):
            tokens = line.split()[1:]
            out_face_vi.append([int(tok.split("/")[0]) for tok in tokens[:3]])

if not vert_lines:
    raise SystemExit(f"[ERROR] No vertices in {OUT_OBJ}")
if not out_face_vi:
    raise SystemExit(f"[ERROR] No faces in {OUT_OBJ}")
if len(out_face_vi) != len(src_face_vti):
    raise SystemExit(
        f"[ERROR] Face count mismatch: output has {len(out_face_vi)}, "
        f"source has {len(src_face_vti)}. Meshes are different topology?"
    )
print(f"  Output verts : {len(vert_lines):,}")
print(f"  Output faces : {len(out_face_vi):,}")

# ── 3. Write deca.mtl ─────────────────────────────────────────────────────────
with open(OUT_MTL, "w", encoding="utf-8") as fh:
    fh.write("newmtl deca_mat\n"
             "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n"
             "map_Kd deca_texture.png\n")
print(f"  Written MTL  : {OUT_MTL}")

# ── 4. Write patched deca.obj  (v + vt + f v/vt/vn) ─────────────────────────
out_lines = [
    "# DECA — UV patched by patch_uv.py",
    "mtllib deca.mtl",
    "usemtl deca_mat",
    "o deca",
    "",
]
out_lines.extend(vert_lines)
out_lines.append("")
out_lines.extend(vt_lines)
out_lines.append("")
for vi, vti in zip(out_face_vi, src_face_vti):
    out_lines.append(f"f {vi[0]}/{vti[0]} {vi[1]}/{vti[1]} {vi[2]}/{vti[2]}")

with open(OUT_OBJ, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines) + "\n")

print(f"  Written OBJ  : {OUT_OBJ}")
print("\nDone! Reload output/ in the browser viewer.")
