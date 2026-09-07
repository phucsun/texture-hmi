"""
Visualize each step of the BFM → DECA mesh alignment pipeline.

Panels (front + side view):
  0. Raw (original coordinate spaces)
  1. Pre-centred + axis-corrected
  2. After Procrustes  (with landmark correspondence lines)
  3. After ICP pass-1 + 1.5
  4. After ICP pass-2 / Final  (with per-vertex distance heatmap)
"""

import gc, sys
import numpy as np
import scipy.io
import trimesh
from trimesh.proximity import closest_point as trimesh_closest_point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("[WARN] open3d not found — ICP steps skipped.")

# ── paths ────────────────────────────────────────────────────────────────────
SUBJ      = "V000190"
SRC_OBJ   = f"uv-idm-output/output_vkist_mesh/{SUBJ}_S002_L2_E01_C4.png.obj"
TGT_OBJ   = f"results_C4_final/{SUBJ}_v000190_s002_l2_e01_c4/{SUBJ}_v000190_s002_l2_e01_c4.obj"
LM_BFM    = "landmarks/uv_idm.mat"
LM_DECA   = "landmarks/deca.npy"
ICP_ITERS = 500
ICP_VOXEL = 0.002


# ── landmark loaders (same as align_and_transfer.py) ─────────────────────────
def load_landmarks_bfm(path, bfm_verts):
    mat = scipy.io.loadmat(path)
    for key in ("keypoints", "pt3d", "landmarks", "lm"):
        if key in mat:
            a = np.array(mat[key], dtype=np.float64)
            if 68 in a.shape:
                arr = a.squeeze()
                if arr.ndim == 2 and 3 in arr.shape:
                    if arr.shape == (3, 68): arr = arr.T
                    return arr[:68, :3].astype(np.float64)
                idx = arr.flatten().astype(np.int64)[:68]
                return bfm_verts[idx].astype(np.float64)
    raise ValueError("No 68-pt landmarks in BFM .mat")

def load_landmarks_deca(path, deca_mesh):
    data = np.load(path, allow_pickle=True).item()
    face_idx = np.array(data['full_lmk_faces_idx'], dtype=np.int64).flatten()
    n = len(face_idx)
    bary = np.array(data['full_lmk_bary_coords'], dtype=np.float64).reshape(n, 3)
    V = np.array(deca_mesh.vertices, dtype=np.float64)
    F = np.array(deca_mesh.faces,    dtype=np.int64)
    vi = F[face_idx]
    return bary[:,0:1]*V[vi[:,0]] + bary[:,1:2]*V[vi[:,1]] + bary[:,2:3]*V[vi[:,2]]

def procrustes_align(src, tgt):
    n = src.shape[0]
    ms, mt = src.mean(0), tgt.mean(0)
    sc, tc = src - ms, tgt - mt
    var_s = (sc**2).sum() / n
    K = tc.T @ sc / n
    U, sig, Vt = np.linalg.svd(K)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2,2] = -1
    R = U @ S @ Vt
    s = float(np.clip(sig.dot(np.diag(S)).sum() / var_s, 1e-3, 1e3))
    t = mt - s * (ms @ R.T)
    return R, s, t

def apply_T(v, R, s, t):
    return s * (np.asarray(v, np.float64) @ R.T) + t


# ── rendering helpers ─────────────────────────────────────────────────────────
def depth_shade(verts, view='front'):
    """Return per-vertex brightness [0,1] from orthographic depth shading."""
    if view == 'front': d = verts[:, 2]
    else:               d = verts[:, 0]
    lo, hi = np.percentile(d, 5), np.percentile(d, 95)
    return np.clip((d - lo) / (hi - lo + 1e-8), 0.2, 1.0)

def render_mesh(ax, verts, faces, color, alpha=0.55, view='front',
                zorder=1, subsample=1):
    """Render a mesh as a filled trisurf with depth shading."""
    if view == 'front':   x, y = verts[:, 0], verts[:, 1]
    elif view == 'side':  x, y = verts[:, 2], verts[:, 1]
    else:                 x, y = verts[:, 0], verts[:, 2]

    shade = depth_shade(verts, view)
    # Convert named color to RGBA
    base = np.array(mcolors.to_rgba(color))

    if subsample > 1:
        idx = np.arange(0, len(faces), subsample)
        f = faces[idx]
    else:
        f = faces

    tri = mtri.Triangulation(x, y, f)
    shaded_color = tuple(base[:3] * 0.6 + 0.4)   # fixed mid-shade for fill
    ax.tripcolor(tri, np.ones(len(x)), cmap=None,
                 facecolors=np.ones(len(f)),
                 alpha=alpha, zorder=zorder,
                 edgecolors='none')
    # Overlay as plain triplot for edge hints
    ax.triplot(tri, color=(*base[:3], 0.08), lw=0.2, zorder=zorder+1)

def render_points(ax, verts, color, s=1.5, alpha=0.7, view='front', zorder=5):
    if view == 'front':  x, y = verts[:, 0], verts[:, 1]
    elif view == 'side': x, y = verts[:, 2], verts[:, 1]
    else:                x, y = verts[:, 0], verts[:, 2]
    shade = depth_shade(verts, view)
    ax.scatter(x, y, s=s, c=shade, cmap='Blues' if 'blue' in color else 'Oranges',
               alpha=alpha, zorder=zorder, linewidths=0)

def render_lm(ax, lm, color, view='front', zorder=8):
    if view == 'front':  x, y = lm[:, 0], lm[:, 1]
    elif view == 'side': x, y = lm[:, 2], lm[:, 1]
    else:                x, y = lm[:, 0], lm[:, 2]
    ax.scatter(x, y, s=10, c=color, zorder=zorder, linewidths=0.3,
               edgecolors='white')

def render_lm_lines(ax, lm_src, lm_tgt, view='front', zorder=7):
    """Draw correspondence lines between landmark pairs."""
    for (s, t) in zip(lm_src, lm_tgt):
        if view == 'front':  xs, ys, xt, yt = s[0], s[1], t[0], t[1]
        elif view == 'side': xs, ys, xt, yt = s[2], s[1], t[2], t[1]
        ax.plot([xs, xt], [ys, yt], '-', c='gold', lw=0.5, alpha=0.6, zorder=zorder)

def dist_heatmap(ax, verts_src, verts_tgt, faces_src, view='front'):
    """Color src mesh by distance to tgt surface."""
    _, dists, _ = trimesh_closest_point(
        trimesh.Trimesh(vertices=verts_tgt,
                        faces=trimesh.load(TGT_OBJ, process=False).faces,
                        process=False),
        verts_src
    )
    clip = np.percentile(dists, 95)
    normed = np.clip(dists / clip, 0, 1)

    if view == 'front':  x, y = verts_src[:, 0], verts_src[:, 1]
    elif view == 'side': x, y = verts_src[:, 2], verts_src[:, 1]

    f = faces_src[::2]
    tri = mtri.Triangulation(x, y, f)
    face_colors = normed[f].mean(1)
    ax.tripcolor(tri, face_colors, cmap='RdYlGn_r', vmin=0, vmax=1,
                 alpha=0.85, zorder=3, edgecolors='none')
    return clip


def style_ax(ax, title, xlim=None, ylim=None):
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#111111')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    for sp in ax.spines.values(): sp.set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading meshes …")
bfm_mesh  = trimesh.load(SRC_OBJ, process=False)
deca_mesh = trimesh.load(TGT_OBJ, process=False)
bfm_v0   = np.array(bfm_mesh.vertices,  np.float64)
deca_v   = np.array(deca_mesh.vertices, np.float64)
bfm_f    = np.array(bfm_mesh.faces,     np.int64)
deca_f   = np.array(deca_mesh.faces,    np.int64)

print("Loading landmarks …")
lm_bfm  = load_landmarks_bfm(LM_BFM, bfm_v0)
lm_deca = load_landmarks_deca(LM_DECA, deca_mesh)

# ─────────────────────────────────────────────────────────────────────────────
# RUN ALIGNMENT STEPS  (same logic as align_and_transfer.py)
# ─────────────────────────────────────────────────────────────────────────────

# ── Step A: pre-centre + axis check ──────────────────────────────────────────
bfm_lm_mean  = lm_bfm.mean(0)
deca_lm_mean = lm_deca.mean(0)
lm_bfm_c  = lm_bfm  - bfm_lm_mean
lm_deca_c = lm_deca - deca_lm_mean
bfm_vc    = bfm_v0  - bfm_lm_mean

_cands = {
    'XYZ' : lm_bfm_c.copy(),
    'XZY' : lm_bfm_c[:, [0, 2, 1]],
    'X-YZ': lm_bfm_c * [1., -1., 1.],
    'XY-Z': lm_bfm_c * [1., 1., -1.],
}
_mse  = {k: ((v - lm_deca_c)**2).sum(1).mean() for k, v in _cands.items()}
_best = min(_mse, key=_mse.get)
print(f"  Axis best: {_best}  MSE={_mse[_best]:.5f}")
if _best != 'XYZ':
    lm_bfm_c = _cands[_best].copy()
    if   _best == 'XZY' : bfm_vc = bfm_vc[:, [0, 2, 1]]
    elif _best == 'X-YZ': bfm_vc = bfm_vc * [1., -1., 1.]
    elif _best == 'XY-Z': bfm_vc = bfm_vc * [1., 1., -1.]

_bfm_diag  = np.linalg.norm(np.ptp(lm_bfm_c,  axis=0))
_deca_diag = np.linalg.norm(np.ptp(lm_deca_c, axis=0))
_pre = _deca_diag / (_bfm_diag + 1e-10)
lm_bfm_c = lm_bfm_c * _pre
bfm_vc   = bfm_vc   * _pre

# BFM centred + axis-corrected (for panel 1)
bfm_v_centred = bfm_vc + deca_lm_mean   # bring to DECA world coords

# ── Step B: Procrustes ────────────────────────────────────────────────────────
R, s, t = procrustes_align(lm_bfm_c, lm_deca_c)
bfm_v_proc = apply_T(bfm_vc, R, s, t) + deca_lm_mean
lm_bfm_proc = apply_T(lm_bfm_c, R, s, t) + deca_lm_mean
rmse_proc = float(np.sqrt(((lm_bfm_proc - lm_deca)**2).sum(1).mean()))
print(f"  Procrustes LM RMSE: {rmse_proc:.5f}")

bfm_aligned_mesh = trimesh.Trimesh(vertices=bfm_v_proc, faces=bfm_f, process=False)

# ── Step C: ICP ───────────────────────────────────────────────────────────────
bfm_v_icp1 = bfm_v_proc.copy()
bfm_v_icp15 = bfm_v_proc.copy()
bfm_v_final = bfm_v_proc.copy()
rmse_icp1 = rmse_icp15 = rmse_icp2 = 0.0

if HAS_O3D:
    print("Running ICP …")
    def _pcd(v, n):
        p = o3d.geometry.PointCloud()
        p.points  = o3d.utility.Vector3dVector(v.astype(np.float64))
        p.normals = o3d.utility.Vector3dVector(n.astype(np.float64))
        return p

    src_full = _pcd(bfm_v_proc,
                    np.array(trimesh.Trimesh(vertices=bfm_v_proc, faces=bfm_f, process=False).vertex_normals, np.float64))
    tgt_full = _pcd(deca_v,
                    np.array(deca_mesh.vertex_normals, np.float64))
    src_d = src_full.voxel_down_sample(ICP_VOXEL)
    tgt_d = tgt_full.voxel_down_sample(ICP_VOXEL)

    r1 = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERS))
    T1 = np.array(r1.transformation, np.float64)
    bfm_v_icp1 = (bfm_v_proc @ T1[:3,:3].T) + T1[:3,3]
    rmse_icp1 = r1.inlier_rmse
    print(f"  ICP pass-1 RMSE: {rmse_icp1:.5f}")

    bfm_p15_mesh = trimesh.Trimesh(vertices=bfm_v_icp1, faces=bfm_f, process=False)
    src_p15 = _pcd(bfm_v_icp1, np.array(bfm_p15_mesh.vertex_normals, np.float64)).voxel_down_sample(ICP_VOXEL)
    r15 = o3d.pipelines.registration.registration_icp(
        src_p15, tgt_d, 0.03, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    T15 = np.array(r15.transformation, np.float64)
    bfm_v_icp15 = (bfm_v_icp1 @ T15[:3,:3].T) + T15[:3,3]
    rmse_icp15 = r15.inlier_rmse
    print(f"  ICP pass-1.5 RMSE: {rmse_icp15:.5f}")

    bfm_pass2 = trimesh.Trimesh(vertices=bfm_v_icp15, faces=bfm_f, process=False)
    src_f2 = _pcd(bfm_v_icp15, np.array(bfm_pass2.vertex_normals, np.float64)).voxel_down_sample(ICP_VOXEL/2)
    tgt_f2 = tgt_full.voxel_down_sample(ICP_VOXEL/2)
    r2 = o3d.pipelines.registration.registration_icp(
        src_f2, tgt_f2, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERS))
    T2 = np.array(r2.transformation, np.float64)
    bfm_v_final = (bfm_v_icp15 @ T2[:3,:3].T) + T2[:3,3]
    rmse_icp2 = r2.inlier_rmse
    print(f"  ICP pass-2 RMSE: {rmse_icp2:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
STEPS = 5
VIEWS = 2   # front + side
COLS  = STEPS
ROWS  = VIEWS

fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 2.8, ROWS * 2.8 + 0.6),
                          facecolor='#0d0d0d',
                          gridspec_kw={'hspace': 0.05, 'wspace': 0.04})

step_titles = [
    "Step 0: Raw\n(original spaces)",
    "Step 1: Pre-centred\n+ Axis corrected",
    f"Step 2: Procrustes\n(LM RMSE={rmse_proc*1000:.1f} mm)",
    f"Step 3: ICP pass-1 + 1.5\n(RMSE={rmse_icp15*1000:.1f} mm)" if HAS_O3D else "Step 3: ICP\n(skipped)",
    f"Step 4: ICP pass-2 / Final\n(RMSE={rmse_icp2*1000:.1f} mm)" if HAS_O3D else "Step 4: Final\n(= Procrustes)",
]

COLOR_SRC  = '#5B8DD9'    # blue  = BFM/UV-IDM source
COLOR_TGT  = '#F0914A'    # orange = DECA target
SUBSAMP = 4   # face subsampling for speed

all_steps_src = [bfm_v0, bfm_v_centred, bfm_v_proc, bfm_v_icp15, bfm_v_final]
step_lm_src   = [lm_bfm,  lm_bfm,        lm_bfm_proc, None,        None]

# Compute uniform axis limits from step-2 data (DECA space)
_all = np.concatenate([bfm_v_proc, deca_v])
cx, cy, cz = np.median(deca_v, axis=0)
R_view = np.percentile(np.abs(_all - np.median(_all, 0)), 90) * 1.15

xlim_f = (cx - R_view, cx + R_view)
ylim_f = (cy - R_view, cy + R_view)
xlim_s = (cz - R_view, cz + R_view)

for row, view in enumerate(['front', 'side']):
    for col, (src_v, lm_s, title) in enumerate(zip(all_steps_src, step_lm_src, step_titles)):
        ax = axes[row, col]

        # Determine xlim/ylim per view
        if col == 0:   # raw: use BFM's own range
            _raw = np.concatenate([bfm_v0, deca_v])
            raw_c = np.median(_raw, 0)
            rr = np.percentile(np.abs(_raw - raw_c), 90) * 1.2
            if view == 'front':
                xl = (raw_c[0]-rr, raw_c[0]+rr)
                yl = (raw_c[1]-rr, raw_c[1]+rr)
            else:
                xl = (raw_c[2]-rr, raw_c[2]+rr)
                yl = (raw_c[1]-rr, raw_c[1]+rr)
        else:
            xl = xlim_f if view == 'front' else xlim_s
            yl = ylim_f

        # Render DECA (target, always same)
        render_points(ax, deca_v, COLOR_TGT, s=0.5, alpha=0.4, view=view, zorder=2)

        # Render BFM source
        if col == 4 and HAS_O3D:
            # Final step: distance heatmap
            clip = dist_heatmap(ax, src_v, deca_v, bfm_f, view=view)
            ax_title_extra = f"\nclip={clip*1000:.1f} mm"
        else:
            render_points(ax, src_v, COLOR_SRC, s=0.3, alpha=0.5, view=view, zorder=3)
            ax_title_extra = ""

        # Landmark correspondences (step 2 only)
        if col == 2 and lm_s is not None:
            render_lm_lines(ax, lm_s, lm_deca, view=view)
            render_lm(ax, lm_s,  COLOR_SRC, view=view)
            render_lm(ax, lm_deca, COLOR_TGT, view=view)
        elif lm_s is not None and col < 2:
            render_lm(ax, lm_s,  COLOR_SRC, view=view)
            render_lm(ax, lm_deca, COLOR_TGT, view=view)

        view_label = "(front)" if view == 'front' else "(side)"
        t = (title + ax_title_extra) if row == 0 else ""
        side_label = f"↑Y  ←Z" if (row == 1 and col == 0) else ""
        style_ax(ax, t, xl, yl)

        if col == 0:
            ax.set_ylabel("front" if view == 'front' else "side",
                          color='#aaaaaa', fontsize=7, rotation=90, labelpad=4)

# Legend
legend_elems = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_SRC,
           markersize=6, label=f'Source (UV-IDM, {len(bfm_v0):,} verts)', lw=0),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_TGT,
           markersize=6, label=f'Target (DECA, {len(deca_v):,} verts)', lw=0),
    Line2D([0],[0], color='gold', lw=1.2, label='Landmark correspondences'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3,
           facecolor='#1a1a1a', edgecolor='none',
           labelcolor='white', fontsize=8,
           bbox_to_anchor=(0.5, 0.0))

fig.suptitle("Mesh Alignment Pipeline: BFM/UV-IDM → DECA",
             color='white', fontsize=11, y=1.01)

plt.savefig("alignment_visualization.pdf", dpi=150, bbox_inches='tight',
            facecolor='#0d0d0d')
plt.savefig("alignment_visualization.png", dpi=180, bbox_inches='tight',
            facecolor='#0d0d0d')
print("Saved: alignment_visualization.pdf / .png")
