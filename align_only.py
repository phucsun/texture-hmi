"""
align_only.py — Align BFM → DECA và export 2 mesh để kiểm tra trong browser.
Output: output/bfm_aligned.obj  (BFM sau Procrustes + ICP)
        output/deca.obj          (DECA gốc, không đổi)
"""
import gc, io, os
import numpy as np
import scipy.io
import trimesh


def parse_obj_uvs(path: str):
    """Parse UV coords + per-face UV indices from an OBJ file.
    Returns (vt_arr (N,2), face_vt_idx (F,3)) or (None, None) if no UVs."""
    vt, fvt = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("vt "):
                p = line.split()
                vt.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                row = []
                for tok in parts[:3]:
                    s = tok.split("/")
                    row.append(int(s[1]) - 1 if len(s) > 1 and s[1] else int(s[0]) - 1)
                fvt.append(row)
    if not vt:
        return None, None
    return np.array(vt, dtype=np.float32), np.array(fvt, dtype=np.int32)

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    print("[WARN] open3d không có — bỏ qua ICP")
    HAS_O3D = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_OBJ  = "bfm_to_flame/uv_idm.obj"
TGT_OBJ  = "bfm_to_flame/deca.obj"
LM_BFM   = "landmarks/uv_idm.mat"
LM_DECA  = "landmarks/deca.npy"
OUT_DIR  = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load meshes ───────────────────────────────────────────────────────────────
print("Loading meshes …")
bfm_mesh  = trimesh.load(SRC_OBJ, process=False, force="mesh")
deca_mesh = trimesh.load(TGT_OBJ, process=False, force="mesh")
print(f"  BFM  : {len(bfm_mesh.vertices):,} verts  {len(bfm_mesh.faces):,} faces")
print(f"  DECA : {len(deca_mesh.vertices):,} verts  {len(deca_mesh.faces):,} faces")

# ── Load landmarks ────────────────────────────────────────────────────────────
print("Loading landmarks …")

mat  = scipy.io.loadmat(LM_BFM)
bfm_verts = np.array(bfm_mesh.vertices, dtype=np.float64)
lm_bfm = None
for key in ("keypoints", "pt3d", "landmarks", "lm", "pt2d"):
    if key in mat:
        a = np.array(mat[key], dtype=np.float64)
        if 68 in a.shape:
            a = a.squeeze()
            if a.ndim == 2 and 3 in a.shape and 68 in a.shape:
                if a.shape == (3, 68): a = a.T
                lm_bfm = a[:68, :3]
            else:
                idx = a.flatten().astype(np.int64)[:68]
                lm_bfm = bfm_verts[idx]
            break
assert lm_bfm is not None, "Không đọc được BFM landmarks"

data     = np.load(LM_DECA, allow_pickle=True).item()
face_idx = np.array(data["full_lmk_faces_idx"], dtype=np.int64).flatten()
bary     = np.array(data["full_lmk_bary_coords"], dtype=np.float64).reshape(-1, 3)
V = np.array(deca_mesh.vertices, dtype=np.float64)
F = np.array(deca_mesh.faces,    dtype=np.int64)
vi = F[face_idx]
lm_deca = bary[:,0:1]*V[vi[:,0]] + bary[:,1:2]*V[vi[:,1]] + bary[:,2:3]*V[vi[:,2]]
print(f"  LM BFM range  [{lm_bfm.min():.4f}, {lm_bfm.max():.4f}]")
print(f"  LM DECA range [{lm_deca.min():.4f}, {lm_deca.max():.4f}]")

# ── Pre-centre ────────────────────────────────────────────────────────────────
bfm_mean  = lm_bfm.mean(0)
deca_mean = lm_deca.mean(0)
lm_bfm_c    = lm_bfm  - bfm_mean
lm_deca_c   = lm_deca - deca_mean
bfm_verts_c = bfm_verts - bfm_mean
print(f"  BFM centroid  {bfm_mean.round(4)}")
print(f"  DECA centroid {deca_mean.round(4)}")

# ── Axis check ────────────────────────────────────────────────────────────────
cands = {
    "XYZ" : lm_bfm_c.copy(),
    "XZY" : lm_bfm_c[:, [0,2,1]],
    "X-YZ": lm_bfm_c * [1,-1, 1],
    "XY-Z": lm_bfm_c * [1, 1,-1],
}
mse   = {k: ((v - lm_deca_c)**2).sum(1).mean() for k,v in cands.items()}
best  = min(mse, key=mse.get)
print(f"  Axis MSE: { {k:f'{v:.5f}' for k,v in mse.items()} }  → {best}")
if best != "XYZ":
    lm_bfm_c = cands[best].copy()
    if   best == "XZY" : bfm_verts_c = bfm_verts_c[:, [0,2,1]]
    elif best == "X-YZ": bfm_verts_c = bfm_verts_c * [1,-1, 1]
    elif best == "XY-Z": bfm_verts_c = bfm_verts_c * [1, 1,-1]

# ── Unit normalise ────────────────────────────────────────────────────────────
bfm_diag  = np.linalg.norm(np.ptp(lm_bfm_c,  axis=0))
deca_diag = np.linalg.norm(np.ptp(lm_deca_c, axis=0))
pre       = deca_diag / (bfm_diag + 1e-10)
print(f"  Pre-scale: {pre:.6f}  (BFM diag {bfm_diag:.5f} → DECA diag {deca_diag:.5f})")
lm_bfm_c    *= pre
bfm_verts_c *= pre

# ── Procrustes (Umeyama) ──────────────────────────────────────────────────────
print("Procrustes …")
n  = lm_bfm_c.shape[0]
ms, mt = lm_bfm_c.mean(0), lm_deca_c.mean(0)
sc, tc = lm_bfm_c - ms, lm_deca_c - mt
var_s  = (sc**2).sum() / n
K = tc.T @ sc / n
U, sig, Vt = np.linalg.svd(K)
S = np.eye(3)
if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2,2] = -1
R = U @ S @ Vt
s = float(np.clip(sig.dot(np.diag(S)).sum() / var_s, 1e-3, 1e3))
t = mt - s*(ms @ R.T)

t64 = t.reshape(1,3)
bfm_aligned_v = np.nan_to_num(s * np.matmul(bfm_verts_c, R.T) + t64,
                               nan=0., posinf=1e6, neginf=-1e6)
lm_bfm_aligned = s * np.matmul(lm_bfm_c, R.T) + t64

# Restore world coords
bfm_aligned_v  += deca_mean
lm_bfm_world    = lm_bfm_aligned + deca_mean

rmse = float(np.sqrt(((lm_bfm_world - lm_deca)**2).sum(1).mean()))
print(f"  Procrustes RMSE: {rmse:.6f}")
print(f"  BFM range after: [{bfm_aligned_v.min():.4f}, {bfm_aligned_v.max():.4f}]")
print(f"  DECA range:      [{V.min():.4f}, {V.max():.4f}]")

bfm_aligned = trimesh.Trimesh(vertices=bfm_aligned_v,
                               faces=bfm_mesh.faces, process=False)

# ── ICP 3-pass ────────────────────────────────────────────────────────────────
if HAS_O3D:
    print("ICP …")
    ICP_VOXEL = 0.002

    def to_pcd(m):
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(np.array(m.vertices,       np.float64))
        pcd.normals = o3d.utility.Vector3dVector(np.array(m.vertex_normals, np.float64))
        return pcd

    def safe_T(r, label):
        T = np.nan_to_num(np.array(r.transformation, dtype=np.float64))
        if not np.all(np.isfinite(T)):
            print(f"  [WARN] {label} diverged — identity")
            return np.eye(4, dtype=np.float64)
        return T

    src_full = to_pcd(bfm_aligned)
    tgt_full = to_pcd(deca_mesh)
    src_d = src_full.voxel_down_sample(ICP_VOXEL)
    tgt_d = tgt_full.voxel_down_sample(ICP_VOXEL)
    print(f"  Downsampled: {len(src_d.points):,} / {len(tgt_d.points):,} pts")

    # Pass 1 — coarse Point-to-Plane
    r1 = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    T1 = safe_T(r1, "pass-1")
    v1 = np.matmul(bfm_aligned_v, T1[:3,:3].T) + T1[:3,3]
    print(f"  Pass-1  fitness={r1.fitness:.4f}  RMSE={r1.inlier_rmse:.6f}")

    # Pass 1.5 — Point-to-Point (translation stabilise)
    m15 = trimesh.Trimesh(vertices=v1, faces=bfm_mesh.faces, process=False)
    s15 = to_pcd(m15).voxel_down_sample(ICP_VOXEL)
    r15 = o3d.pipelines.registration.registration_icp(
        s15, tgt_d, 0.03, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    T15 = safe_T(r15, "pass-1.5")
    v15 = np.matmul(v1, T15[:3,:3].T) + T15[:3,3]
    print(f"  Pass-1.5 fitness={r15.fitness:.4f}  RMSE={r15.inlier_rmse:.6f}")
    del src_d, tgt_d, s15, m15; gc.collect()

    # Pass 2 — fine Point-to-Plane
    m2 = trimesh.Trimesh(vertices=v15, faces=bfm_mesh.faces, process=False)
    sf = to_pcd(m2).voxel_down_sample(ICP_VOXEL / 2)
    tf = to_pcd(deca_mesh).voxel_down_sample(ICP_VOXEL / 2)
    r2 = o3d.pipelines.registration.registration_icp(
        sf, tf, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=500, relative_fitness=1e-12, relative_rmse=1e-12))
    T2 = safe_T(r2, "pass-2")
    bfm_final_v = np.matmul(v15, T2[:3,:3].T) + T2[:3,3]
    print(f"  Pass-2  fitness={r2.fitness:.4f}  RMSE={r2.inlier_rmse:.6f}")
    del src_full, tgt_full, sf, tf, m2; gc.collect()
else:
    bfm_final_v = bfm_aligned_v

# Surface RMSE check
from trimesh.proximity import closest_point as cp
bfm_final = trimesh.Trimesh(vertices=bfm_final_v, faces=bfm_mesh.faces, process=False)
samp = np.random.choice(len(V), min(3000, len(V)), replace=False)
_, d, _ = cp(bfm_final, V[samp])
print(f"  Surface RMSE (3k pts): {np.sqrt((d**2).mean()):.6f}")

# ── Export ────────────────────────────────────────────────────────────────────
print("Exporting …")

bfm_out = os.path.join(OUT_DIR, "bfm_aligned.obj")
faces   = np.array(bfm_mesh.faces, np.int32)
with open(bfm_out, "w") as f:
    f.write("# BFM aligned\no bfm_aligned\n")
    for v in bfm_final_v:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for fc in faces:
        f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
print(f"  {bfm_out}  ({len(bfm_final_v):,} verts)")

deca_out   = os.path.join(OUT_DIR, "deca.obj")
deca_faces = np.array(deca_mesh.faces, np.int32)

# Parse UV coordinates from source DECA OBJ (same UV atlas as deca_texture.png)
vt_arr, face_vt_idx = parse_obj_uvs(TGT_OBJ)
has_uv = vt_arr is not None and face_vt_idx is not None and len(vt_arr) > 0

# Write MTL so browser viewer can resolve the texture
mtl_out = os.path.join(OUT_DIR, "deca.mtl")
with open(mtl_out, "w", encoding="utf-8") as f:
    f.write("newmtl deca_mat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd deca_texture.png\n")

# Vectorised OBJ export — no Python per-line loops
_buf = io.StringIO()
_buf.write("# DECA original — exported by align_only.py\n"
           "mtllib deca.mtl\nusemtl deca_mat\no deca\n\n")
np.savetxt(_buf, V, fmt="v  %.6f %.6f %.6f")
_buf.write("\n")

if has_uv:
    np.savetxt(_buf, vt_arr, fmt="vt %.6f %.6f")
    _buf.write("\n")
    vi_1  = deca_faces + 1                             # (F,3) 1-based vertex idx
    vti_1 = face_vt_idx + 1                            # (F,3) 1-based UV idx
    _face_lines = "\n".join(
        f"f {a}/{b}/{a} {c}/{d}/{c} {e}/{g}/{e}"
        for a, b, c, d, e, g in zip(
            vi_1[:,0], vti_1[:,0],
            vi_1[:,1], vti_1[:,1],
            vi_1[:,2], vti_1[:,2],
        )
    )
else:
    print("  [WARN] Không tìm thấy UV trong source DECA — export faces không có UV")
    _face_lines = "\n".join(
        f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}" for fc in deca_faces
    )
_buf.write(_face_lines + "\n")

with open(deca_out, "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())
del _buf, _face_lines

uv_info = f"  +UV ({len(vt_arr):,} coords)" if has_uv else "  (no UV)"
print(f"  {deca_out}  ({len(V):,} verts){uv_info}")
print(f"  {mtl_out}")

# ── Export heatmap: DECA tô màu theo khoảng cách đến BFM ─────────────────────
print("Computing distance heatmap …")
from trimesh.proximity import closest_point as cp_fn

_, dists, _ = cp_fn(bfm_final, V)          # (N,) khoảng cách mỗi đỉnh DECA → BFM

# Jet colormap: xanh lá (0) → vàng → đỏ (max_d)
max_d = np.percentile(dists, 95)           # clip ở p95 để không bị outlier chi phối
t = np.clip(dists / (max_d + 1e-10), 0., 1.)   # (N,) 0..1

def jet(t):
    """t: (N,) 0..1  →  (N,3) RGB float"""
    r = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)
    return np.stack([r, g, b], axis=1)

colors = jet(t)    # (N,3)

heatmap_out = os.path.join(OUT_DIR, "deca_heatmap.obj")
with open(heatmap_out, "w") as f:
    f.write(f"# DECA heatmap — max_dist(p95)={max_d:.5f}\no deca_heatmap\n")
    for i, v in enumerate(V):
        c = colors[i]
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    for fc in deca_faces:
        f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")

print(f"  {heatmap_out}  (xanh=khít  đỏ=lệch, max_dist={max_d*1000:.2f} mm)")
print(f"  Median dist: {np.median(dists)*1000:.2f} mm  |  Mean: {dists.mean()*1000:.2f} mm")
print("\nDone! Mở browser → Load output/ folder để kiểm tra.")
