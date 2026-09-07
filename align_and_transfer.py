"""
align_and_transfer.py — BFM → FLAME/DECA texture transfer  (high-performance)

Pipeline:
  1. Load  meshes / texture / per-face UV (parsed from OBJ)
  2. Procrustes alignment on 68 landmarks
  3. Point-to-Plane ICP  (voxel-downsampled)
  4. Vertex-level colour transfer  (fully vectorized, no per-vertex loop)
  5. Per-texel baking 1024×1024
       – zero-loop UV rasterization  (flat candidate array, pure numpy)
       – batched BFM surface query   (BATCH_SZ=10k, aggressive gc)
  6. Seam dilation + Gaussian AA     (cv2)
  7. Export PNG + OBJ + MTL
"""

import gc
import os
import sys
import time

import numpy as np
np.seterr(all="ignore")
import scipy.io
import trimesh
from trimesh.proximity import closest_point as trimesh_closest_point
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    print("[WARN] opencv-python not found – seam dilation skipped.")
    HAS_CV2 = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    print("[WARN] open3d not found – ICP skipped.")
    HAS_O3D = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (overridable via CLI args for batch processing)
# ─────────────────────────────────────────────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(description="BFM → FLAME/DECA texture transfer",
                              add_help=True)
_parser.add_argument("--src-obj", default="bfm_to_flame/uv_idm.obj",
                     help="Source BFM/UV-IDM mesh (.obj)")
_parser.add_argument("--src-tex", default="bfm_to_flame/uv_idm.png",
                     help="Source texture image (.png)")
_parser.add_argument("--tgt-obj", default="bfm_to_flame/deca.obj",
                     help="Target DECA/FLAME mesh (.obj)")
_parser.add_argument("--lm-bfm",  default="landmarks/uv_idm.mat",
                     help="BFM landmarks (.mat)")
_parser.add_argument("--lm-deca", default="landmarks/deca.npy",
                     help="DECA landmarks (.npy)")
_parser.add_argument("--out-dir", default="output",
                     help="Output directory")
# ── Ablation flags ────────────────────────────────────────────────────────────
_parser.add_argument("--no-icp", action="store_true",
                     help="Ablation: skip ICP, use Procrustes only")
_parser.add_argument("--hard-gate", type=float, default=None, metavar="DIST",
                     help="Ablation: replace soft-confidence with hard distance gate "
                          "(vertices beyond DIST get zero weight / black)")
_parser.add_argument("--no-gauss-blend", action="store_true",
                     help="Ablation: skip Pass-C Gaussian boundary blending")
_args = _parser.parse_args()

SRC_OBJ         = _args.src_obj
SRC_TEX         = _args.src_tex
TGT_OBJ         = _args.tgt_obj
LM_BFM          = _args.lm_bfm
LM_DECA         = _args.lm_deca
OUT_DIR         = _args.out_dir
ABL_NO_ICP      = _args.no_icp
ABL_HARD_GATE   = _args.hard_gate   # float or None
ABL_NO_GAUSS    = _args.no_gauss_blend

TEX_SIZE      = 1024
BATCH_SZ      = 10_000
DIST_CAP      = 0.05
ICP_ITERS     = 500
ICP_VOXEL     = 0.002
RAST_MAX_CAND = 4_000_000

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def progress(msg: str, pct: float = None):
    if pct is None:
        print(f"[....] {msg}", flush=True)
    else:
        bar = "#" * int(pct / 5)
        print(f"\r[{bar:<20s}] {pct:5.1f}%  {msg}   ", end="", flush=True)

def finish_progress():
    print()


def parse_obj_face_uvs(path: str):
    """Parse OBJ → per-face UV.  Handles separate v/vt indices (f a/b c/d e/f)."""
    vt_list, fv_list, fvt_list = [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if line.startswith('vt '):
                p = line.split()
                vt_list.append((float(p[1]), float(p[2])))
            elif line.startswith('f '):
                p = line.split()[1:]
                rv, rvt = [], []
                for tok in p[:3]:
                    s = tok.split('/')
                    vi  = int(s[0]) - 1
                    vti = int(s[1]) - 1 if len(s) > 1 and s[1] else vi
                    rv.append(vi); rvt.append(vti)
                fv_list.append(rv); fvt_list.append(rvt)

    vt_arr  = np.array(vt_list,  dtype=np.float32)   # (N_vt, 2)
    fv_idx  = np.array(fv_list,  dtype=np.int32)      # (F, 3)
    fvt_idx = np.array(fvt_list, dtype=np.int32)      # (F, 3)
    fuvs    = vt_arr[fvt_idx] if len(vt_arr) else np.zeros((len(fv_idx), 3, 2), np.float32)
    return fuvs, fv_idx, fvt_idx, vt_arr   # (F,3,2), (F,3), (F,3), (N_vt,2)


def load_landmarks_bfm(path: str, bfm_verts: np.ndarray) -> np.ndarray:
    """Load BFM landmarks.  Shape (68,3): direct 3-D.  Shape (68,): vertex indices."""
    mat = scipy.io.loadmat(path)
    candidate = ckey = None
    for key in ("keypoints", "pt3d", "landmarks", "lm", "pt2d"):
        if key in mat:
            a = np.array(mat[key], dtype=np.float64)
            if 68 in a.shape:
                candidate, ckey = a, key; break
    if candidate is None:
        for k, v in mat.items():
            if k.startswith('_'): continue
            a = np.array(v, dtype=np.float64)
            if 68 in a.shape:
                candidate, ckey = a, k; break
    if candidate is None:
        raise ValueError(f"No 68-pt landmarks in {path}. Keys: "
                         f"{[k for k in mat if not k.startswith('_')]}")
    arr = candidate.squeeze()
    if arr.ndim == 2 and 3 in arr.shape and 68 in arr.shape:
        if arr.shape == (3, 68): arr = arr.T
        print(f"  BFM lm : '{ckey}' 3-D coords  shape={arr.shape}")
        return arr[:68, :3].astype(np.float64)
    idx = arr.flatten().astype(np.int64)[:68]
    print(f"  BFM lm : '{ckey}' vertex-index lookup  ({arr.shape} → 68×3)")
    return bfm_verts[idx].astype(np.float64)


def load_landmarks_deca(path: str, deca_mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute 68 3-D landmark positions from full_lmk_faces_idx + bary_coords."""
    data = np.load(path, allow_pickle=True).item()
    print(f"  deca.npy keys : {list(data.keys())}")
    face_idx = np.array(data['full_lmk_faces_idx'], dtype=np.int64).flatten()
    n = len(face_idx)
    bary = (np.array(data['full_lmk_bary_coords'], dtype=np.float64).reshape(n, 3)
            if 'full_lmk_bary_coords' in data
            else np.ones((n, 3), np.float64) / 3.0)
    V = np.array(deca_mesh.vertices, dtype=np.float64)
    F = np.array(deca_mesh.faces,    dtype=np.int64)
    vi = F[face_idx]   # (68, 3)
    return (bary[:,0:1]*V[vi[:,0]] + bary[:,1:2]*V[vi[:,1]] + bary[:,2:3]*V[vi[:,2]])


def procrustes_align(src: np.ndarray, tgt: np.ndarray):
    """Umeyama Procrustes → (R, s, t)."""
    n = src.shape[0]
    ms, mt = src.mean(0), tgt.mean(0)
    sc, tc = src - ms, tgt - mt
    var_s = (sc**2).sum() / n
    if var_s < 1e-8:
        raise ValueError(f"[Procrustes] Source landmarks have near-zero variance "
                         f"({var_s:.2e}). Check that BFM landmarks are in 3-D world space.")
    K = tc.T @ sc / n
    U, sig, Vt = np.linalg.svd(K)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    s = sig.dot(np.diag(S)).sum() / var_s
    s = float(np.clip(s, 1e-3, 1e3))          # guard against extreme scale
    t = mt - s * (ms @ R.T)

    # Sanity checks
    assert np.isfinite(s),              f"[Procrustes] scale s={s} is not finite"
    assert np.all(np.isfinite(R)),      "[Procrustes] R contains NaN/Inf"
    assert np.all(np.isfinite(t)),      "[Procrustes] t contains NaN/Inf"
    print(f"  [Procrustes] scale s={s:.5f}  |  var_src={var_s:.5e}")
    return R, s, t

def apply_transform(v: np.ndarray, R, s, t) -> np.ndarray:
    v_64 = np.asarray(v, dtype=np.float64)
    R_64 = np.nan_to_num(np.asarray(R, dtype=np.float64))
    # Validate rotation: det(R) must be ~1.0; re-orthogonalise if not
    det = np.linalg.det(R_64)
    if not (0.9 < abs(det) < 1.1):
        print(f"  [WARN] apply_transform: det(R)={det:.5f} — re-orthogonalising via SVD.")
        U, _, Vt = np.linalg.svd(R_64)
        R_64 = U @ Vt
    s_safe = float(s) if abs(float(s)) > 1e-10 else 1.0
    t_64   = np.asarray(t, dtype=np.float64).reshape(1, 3)   # explicit (1,3) broadcast
    result = s_safe * np.matmul(v_64, R_64.T) + t_64
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)


def sample_texture(uv: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Bilinear sample img (H×W×3 float32).
    BFM UV convention: V=0 at TOP (forehead) → row 0.  No Y-flip needed."""
    H, W = img.shape[:2]
    u  = np.clip(uv[:,0], 0., 1.);  v = np.clip(uv[:,1], 0., 1.)
    px = u*(W-1);                   py = v*(H-1)   # V=0 → row 0 (top)
    x0 = np.floor(px).astype(np.int32); y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0+1, 0, W-1);        y1 = np.clip(y0+1, 0, H-1)
    x0 = np.clip(x0,   0, W-1);        y0 = np.clip(y0,   0, H-1)
    wx = (px-x0).reshape(-1,1);        wy = (py-y0).reshape(-1,1)
    return (img[y0,x0]*(1-wx)*(1-wy) + img[y0,x1]*wx*(1-wy)
          + img[y1,x0]*(1-wx)*wy     + img[y1,x1]*wx*wy).astype(np.float32)


def bary_uv_batch(cp, p0, p1, p2, u0, u1, u2):
    """Vectorized barycentric UV for K points. Returns (uvs (K,2), valid (K,))."""
    e1,e2,ep = p1-p0, p2-p0, cp-p0
    d11=(e1*e1).sum(1); d12=(e1*e2).sum(1); d22=(e2*e2).sum(1)
    dp1=(ep*e1).sum(1); dp2=(ep*e2).sum(1)
    den = d11*d22 - d12*d12
    ok  = np.abs(den) > 1e-12
    d   = np.where(ok, den, 1.)
    bv  = (d22*dp1 - d12*dp2) / d
    bw  = (d11*dp2 - d12*dp1) / d
    bu  = 1. - bv - bw
    bc  = np.clip(np.stack([bu,bv,bw], 1), 0., 1.)
    rs  = bc.sum(1, keepdims=True)
    bc /= np.where(rs > 1e-12, rs, 1.)
    return bc[:,0:1]*u0 + bc[:,1:2]*u1 + bc[:,2:3]*u2, ok


# ─────────────────────────────────────────────────────────────────────────────
# ZERO-LOOP VECTORIZED RASTERIZER
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_uv_triangles(face_uvs: np.ndarray, tex_size: int) -> tuple:
    """
    Fully vectorized UV rasterizer — zero Python pixel or face loops.

    Strategy:
      1. Compute bounding boxes for ALL faces at once.
      2. Enumerate ALL (face, row, col) candidate pixels as flat arrays using
         np.repeat / integer division — no Python loop.
      3. Run edge-function test on the entire flat array in one numpy pass.
      4. Scatter results into face_map / bary_map.

    Faces are processed in chunks of ≤ RAST_MAX_CAND candidates so that
    peak RAM stays bounded regardless of atlas layout.

    Returns
    -------
    face_map : (H, W) int32   — face index at each texel, -1 if empty
    bary_map : (H, W, 3) f32  — barycentric weights
    """
    TS = tex_size - 1
    F  = len(face_uvs)

    face_map = np.full((tex_size, tex_size), -1, dtype=np.int32)
    bary_map = np.zeros((tex_size, tex_size, 3), dtype=np.float32)

    # ── pixel-space vertex coordinates for ALL faces ──────────────────────────
    # face_uvs : (F, 3, 2) float32
    px = face_uvs[:,:,0] * TS                      # (F, 3)
    py = (1.0 - face_uvs[:,:,1]) * TS              # flip Y, (F, 3)

    p0x,p1x,p2x = px[:,0].astype(np.float64), px[:,1].astype(np.float64), px[:,2].astype(np.float64)
    p0y,p1y,p2y = py[:,0].astype(np.float64), py[:,1].astype(np.float64), py[:,2].astype(np.float64)

    areas = (p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x)  # (F,)

    # ── bounding boxes ────────────────────────────────────────────────────────
    bbx0 = np.floor(np.minimum.reduce([p0x,p1x,p2x])).astype(np.int32).clip(0, TS)
    bbx1 = np.ceil( np.maximum.reduce([p0x,p1x,p2x])).astype(np.int32).clip(0, TS)
    bby0 = np.floor(np.minimum.reduce([p0y,p1y,p2y])).astype(np.int32).clip(0, TS)
    bby1 = np.ceil( np.maximum.reduce([p0y,p1y,p2y])).astype(np.int32).clip(0, TS)

    valid = (np.abs(areas) >= 1e-6) & (bbx1 >= bbx0) & (bby1 >= bby0)
    fi_all   = np.where(valid)[0]                      # valid face indices

    rows_all = (bby1[fi_all] - bby0[fi_all] + 1).astype(np.int64)
    cols_all = (bbx1[fi_all] - bbx0[fi_all] + 1).astype(np.int64)
    pix_all  = rows_all * cols_all                     # candidates per face

    n_valid   = len(fi_all)
    chunk_s   = 0
    t0        = time.time()

    while chunk_s < n_valid:
        # ── find chunk_end such that sum(pix_all[chunk_s:chunk_end]) ≤ MAX ───
        cum = np.cumsum(pix_all[chunk_s:])
        chunk_len = int(np.searchsorted(cum, RAST_MAX_CAND, side='left')) + 1
        chunk_len = min(chunk_len, n_valid - chunk_s)
        chunk_e   = chunk_s + chunk_len

        progress(f"rast faces {chunk_s}–{chunk_e}/{n_valid}",
                 chunk_s / n_valid * 100)

        fi_c    = fi_all [chunk_s:chunk_e]             # original face indices (M,)
        rows_c  = rows_all[chunk_s:chunk_e]
        cols_c  = cols_all[chunk_s:chunk_e]
        pix_c   = pix_all [chunk_s:chunk_e]
        M       = len(fi_c)
        n_cand  = int(pix_c.sum())

        # ── build flat candidate arrays — ZERO Python loop ────────────────────
        # lfi_flat : local face index within chunk  (n_cand,)
        lfi_flat = np.repeat(np.arange(M, dtype=np.int32), pix_c)

        # local linear index within each face's bbox
        cum_c = np.zeros(M + 1, dtype=np.int64)
        cum_c[1:] = pix_c.cumsum()
        local = np.arange(n_cand, dtype=np.int64) - cum_c[lfi_flat]

        # (local_row, local_col) from linear index
        cols_flat = cols_c[lfi_flat]
        loc_row   = (local // cols_flat).astype(np.int32)
        loc_col   = (local %  cols_flat).astype(np.int32)
        del local, cum_c

        # absolute pixel coordinates
        abs_y = bby0[fi_c][lfi_flat] + loc_row        # (n_cand,) int32
        abs_x = bbx0[fi_c][lfi_flat] + loc_col
        del loc_row, loc_col, cols_flat

        # pixel centre in float
        ptx = abs_x.astype(np.float64) + 0.5
        pty = abs_y.astype(np.float64) + 0.5

        # original face indices for each candidate
        fi_orig = fi_c[lfi_flat].astype(np.int32)
        del lfi_flat

        # ── edge-function test (vectorized) ──────────────────────────────────
        ia  = 1.0 / areas[fi_orig]                    # (n_cand,)
        _p0x=p0x[fi_orig]; _p1x=p1x[fi_orig]; _p2x=p2x[fi_orig]
        _p0y=p0y[fi_orig]; _p1y=p1y[fi_orig]; _p2y=p2y[fi_orig]

        w0 = ((_p2x-_p1x)*(pty-_p1y) - (_p2y-_p1y)*(ptx-_p1x)) * ia
        w1 = ((_p0x-_p2x)*(pty-_p2y) - (_p0y-_p2y)*(ptx-_p2x)) * ia
        w2 = ((_p1x-_p0x)*(pty-_p0y) - (_p1y-_p0y)*(ptx-_p0x)) * ia
        del ptx, pty, ia, _p0x,_p1x,_p2x, _p0y,_p1y,_p2y

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

        # ── scatter into maps (numpy fancy-index: last write wins per pixel) ──
        if inside.any():
            ay = abs_y[inside]; ax = abs_x[inside]
            face_map[ay, ax] = fi_orig[inside]
            bary_map[ay, ax, 0] = w0[inside].astype(np.float32)
            bary_map[ay, ax, 1] = w1[inside].astype(np.float32)
            bary_map[ay, ax, 2] = w2[inside].astype(np.float32)

        del abs_y, abs_x, fi_orig, w0, w1, w2, inside
        gc.collect()

        chunk_s = chunk_e

    finish_progress()
    print(f"  Rasterization time : {time.time()-t0:.1f}s")
    return face_map, bary_map


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  align_and_transfer.py  —  BFM → FLAME texture transfer")
print("=" * 60)

progress("Loading BFM mesh …")
bfm_mesh = trimesh.load(SRC_OBJ, process=False, force="mesh")
bfm_mesh.fix_normals()
bfm_mesh.fill_holes()
print(f"  BFM  : {len(bfm_mesh.vertices):,} verts, {len(bfm_mesh.faces):,} faces")

progress("Loading DECA mesh …")
deca_mesh = trimesh.load(TGT_OBJ, process=False, force="mesh")
deca_mesh.fix_normals()
deca_mesh.fill_holes()
print(f"  DECA : {len(deca_mesh.vertices):,} verts, {len(deca_mesh.faces):,} faces")

progress("Loading texture …")
tex_img = Image.open(SRC_TEX).convert("RGB")
tex_np  = np.array(tex_img, dtype=np.float32) / 255.0
print(f"  Tex  : {tex_np.shape[1]} × {tex_np.shape[0]}")

progress("Parsing OBJ UV …")
bfm_face_uvs, bfm_face_v_idx, _,      bfm_vt_arr  = parse_obj_face_uvs(SRC_OBJ)
deca_face_uvs, deca_face_v_idx, deca_face_vt_idx, deca_vt_arr = parse_obj_face_uvs(TGT_OBJ)
print(f"  BFM UV : {len(bfm_vt_arr):,} unique  |  DECA UV : {len(deca_vt_arr):,} unique")

if len(deca_vt_arr) == 0:
    sys.exit("[ERROR] DECA mesh has no UV coordinates.")

progress("Loading landmarks …")
lm_bfm  = load_landmarks_bfm(LM_BFM, np.array(bfm_mesh.vertices, dtype=np.float64))
lm_deca = load_landmarks_deca(LM_DECA, deca_mesh)
print(f"  LM BFM {lm_bfm.shape}  |  LM DECA {lm_deca.shape}")

if lm_bfm.shape != lm_deca.shape:
    sys.exit(f"[ERROR] Landmark shape mismatch: {lm_bfm.shape} vs {lm_deca.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROCRUSTES ALIGNMENT  (pre-centre → axis check → unit-normalise → align)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1/5] Procrustes alignment …")
print(f"  LM BFM  range : [{lm_bfm.min():.4f}, {lm_bfm.max():.4f}]")
print(f"  LM DECA range : [{lm_deca.min():.4f}, {lm_deca.max():.4f}]")

# ── 1. Pre-centre: subtract each mesh's landmark mean → both at origin ────────
#    This eliminates large spatial offsets that cause matmul overflow.
bfm_lm_mean  = lm_bfm.mean(0)
deca_lm_mean = lm_deca.mean(0)
lm_bfm_c    = lm_bfm  - bfm_lm_mean                             # (68,3) centred
lm_deca_c   = lm_deca - deca_lm_mean                            # (68,3) centred
bfm_verts_c = np.array(bfm_mesh.vertices, dtype=np.float64) - bfm_lm_mean
print(f"  BFM  lm centroid : {bfm_lm_mean.round(4)}")
print(f"  DECA lm centroid : {deca_lm_mean.round(4)}")

# ── 2. Axis-alignment check (Y-up vs Z-up, axis flips) ───────────────────────
#    Tests 4 common convention mismatches; keeps the one with lowest MSE.
_cands = {
    'XYZ' : lm_bfm_c.copy(),
    'XZY' : lm_bfm_c[:, [0, 2, 1]],
    'X-YZ': lm_bfm_c * np.array([ 1., -1.,  1.]),
    'XY-Z': lm_bfm_c * np.array([ 1.,  1., -1.]),
}
_mse  = {k: ((v - lm_deca_c) ** 2).sum(1).mean() for k, v in _cands.items()}
_best = min(_mse, key=_mse.get)
print(f"  Axis MSE : { {k: f'{v:.5f}' for k, v in _mse.items()} }  → {_best}")
if _best != 'XYZ':
    lm_bfm_c = _cands[_best].copy()
    if   _best == 'XZY' : bfm_verts_c = bfm_verts_c[:, [0, 2, 1]]
    elif _best == 'X-YZ': bfm_verts_c = bfm_verts_c * np.array([ 1., -1.,  1.])
    elif _best == 'XY-Z': bfm_verts_c = bfm_verts_c * np.array([ 1.,  1., -1.])
    print(f"  Applied axis re-ordering: {_best}")
del _cands, _mse, _best

# ── 3. Strict unit normalisation — match bbox diagonals exactly ───────────────
_bfm_diag  = np.linalg.norm(np.ptp(lm_bfm_c,  axis=0))
_deca_diag = np.linalg.norm(np.ptp(lm_deca_c, axis=0))
_pre       = _deca_diag / (_bfm_diag + 1e-10)
print(f"  BFM diag={_bfm_diag:.5f}  DECA diag={_deca_diag:.5f}  pre-scale={_pre:.6f}")
lm_bfm_c    = lm_bfm_c    * _pre
bfm_verts_c = bfm_verts_c * _pre
del _bfm_diag, _deca_diag, _pre

# ── 4. Procrustes on centred + normalised landmarks ───────────────────────────
print(f"  LM BFM_c  first 5 :\n{lm_bfm_c[:5].round(5)}")
print(f"  LM DECA_c first 5 :\n{lm_deca_c[:5].round(5)}")

R, s, t = procrustes_align(lm_bfm_c, lm_deca_c)
# apply_transform → DECA-centred space; add deca_lm_mean to restore world coords
bfm_verts_aligned = apply_transform(bfm_verts_c, R, s, t) + deca_lm_mean
del bfm_verts_c

rmse_p = float(np.sqrt(
    ((apply_transform(lm_bfm_c, R, s, t) + deca_lm_mean - lm_deca) ** 2).sum(1).mean()))
print(f"  Procrustes landmark RMSE : {rmse_p:.6f}")

_has_nan = not np.all(np.isfinite(bfm_verts_aligned))
print(f"  Transformed BFM range : [{bfm_verts_aligned.min():.4f}, {bfm_verts_aligned.max():.4f}]"
      f"  NaN/Inf: {_has_nan}")
if _has_nan:
    sys.exit("[ERROR] Procrustes transform produced NaN/Inf. Check landmark data.")

bfm_aligned = trimesh.Trimesh(vertices=bfm_verts_aligned,
                               faces=bfm_mesh.faces, visual=bfm_mesh.visual, process=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. POINT-TO-PLANE ICP  (voxel-downsampled)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2/5] Point-to-Plane ICP (3-pass) …")
if ABL_NO_ICP:
    bfm_verts_final = bfm_verts_aligned.copy()
    print("  [ABLATION --no-icp] Skipped — using Procrustes result only.")
elif HAS_O3D:
    def _to_pcd(m):
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(np.array(m.vertices,       np.float64))
        pcd.normals = o3d.utility.Vector3dVector(np.array(m.vertex_normals, np.float64))
        return pcd

    def _safe_T(result_obj, label: str) -> np.ndarray:
        T = np.nan_to_num(np.array(result_obj.transformation, dtype=np.float64))
        if not np.all(np.isfinite(T)):
            print(f"  [WARN] {label} Inf/NaN — using identity.")
            return np.eye(4, dtype=np.float64)
        return T

    src_full = _to_pcd(bfm_aligned)
    tgt_full = _to_pcd(deca_mesh)
    src_down = src_full.voxel_down_sample(ICP_VOXEL)
    tgt_down = tgt_full.voxel_down_sample(ICP_VOXEL)
    print(f"  Downsampled : BFM {len(src_down.points):,} pts  |  DECA {len(tgt_down.points):,} pts")

    # ── Pass 1: coarse Point-to-Plane ─────────────────────────────────────────
    r1 = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ICP_ITERS, relative_fitness=1e-7, relative_rmse=1e-7),
    )
    T1 = _safe_T(r1, "ICP pass-1")
    bfm_verts_pass1 = np.matmul(bfm_verts_aligned.astype(np.float64), T1[:3, :3].T) + T1[:3, 3]
    print(f"  ICP pass-1  fitness {r1.fitness:.6f}  |  inlier RMSE {r1.inlier_rmse:.6f}")

    # ── Pass 1.5: Point-to-Point to stabilise translation ─────────────────────
    bfm_p15    = trimesh.Trimesh(vertices=bfm_verts_pass1,
                                  faces=bfm_mesh.faces, visual=bfm_mesh.visual, process=False)
    src_p15    = _to_pcd(bfm_p15).voxel_down_sample(ICP_VOXEL)
    r15 = o3d.pipelines.registration.registration_icp(
        src_p15, tgt_down, 0.03, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200, relative_fitness=1e-9, relative_rmse=1e-9),
    )
    T15 = _safe_T(r15, "ICP pass-1.5")
    bfm_verts_p15 = np.matmul(bfm_verts_pass1, T15[:3, :3].T) + T15[:3, 3]
    print(f"  ICP pass-1.5 fitness {r15.fitness:.6f}  |  inlier RMSE {r15.inlier_rmse:.6f}")
    del src_down, tgt_down, src_p15, bfm_p15; gc.collect()

    # ── Pass 2: fine Point-to-Plane to lock rotation ──────────────────────────
    bfm_pass2  = trimesh.Trimesh(vertices=bfm_verts_p15,
                                  faces=bfm_mesh.faces, visual=bfm_mesh.visual, process=False)
    src_fine   = _to_pcd(bfm_pass2)
    tgt_fine   = _to_pcd(deca_mesh)
    src_fine_d = src_fine.voxel_down_sample(ICP_VOXEL / 2)
    tgt_fine_d = tgt_fine.voxel_down_sample(ICP_VOXEL / 2)
    print(f"  ICP pass-2   downsampled : {len(src_fine_d.points):,} / {len(tgt_fine_d.points):,} pts")

    r2 = o3d.pipelines.registration.registration_icp(
        src_fine_d, tgt_fine_d, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ICP_ITERS, relative_fitness=1e-12, relative_rmse=1e-12),
    )
    T2 = _safe_T(r2, "ICP pass-2")
    bfm_verts_final = np.matmul(bfm_verts_p15, T2[:3, :3].T) + T2[:3, 3]
    print(f"  ICP pass-2   fitness {r2.fitness:.6f}  |  inlier RMSE {r2.inlier_rmse:.6f}")
    del src_full, tgt_full, src_fine, tgt_fine, src_fine_d, tgt_fine_d, bfm_pass2; gc.collect()
else:
    bfm_verts_final = bfm_verts_aligned.copy()
    print("  (skipped – open3d unavailable)")

bfm_aligned = trimesh.Trimesh(vertices=bfm_verts_final,
                               faces=bfm_mesh.faces, visual=bfm_mesh.visual, process=False)
# Ensure outward-facing normals before any dot-product filtering
trimesh.repair.fix_inversion(bfm_aligned, multibody=True)

samp_idx = np.random.choice(len(deca_mesh.vertices), min(5000, len(deca_mesh.vertices)), replace=False)
_, surf_d, _ = trimesh_closest_point(bfm_aligned,
                                     np.array(deca_mesh.vertices, np.float64)[samp_idx])
surf_rmse = np.sqrt((surf_d**2).mean())
print(f"  Surface RMSE (5k pts) : {surf_rmse:.6f}")
DIST_CAP = max(0.03, 2.0 * surf_rmse)
print(f"  Adaptive DIST_CAP     : {DIST_CAP:.5f}")
del samp_idx, surf_d


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERTEX-LEVEL COLOUR TRANSFER  (100% coverage — fully vectorized)
#
#  Strategy: remove ALL hard distance/normal gates from this step.
#  Every vertex receives a colour unconditionally via a 3-level fallback:
#    1. Barycentric UV on closest BFM triangle  (preferred path)
#    2. Nearest-corner UV on degenerate triangles  (vectorized fallback)
#    3. UV clamped to [0,1]  (catches UV atlas edge overshoots)
#  Soft quality weights (dist + normal) are stored in vertex_confidence
#  for use in the baking step — they govern blend amount, not inclusion.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3/5] Vertex-level colour transfer (100% coverage) …")

bfm_aligned._cache.clear()
bfm_verts_f = np.array(bfm_aligned.vertices,    np.float64)
bfm_faces_a = np.array(bfm_aligned.faces,        np.int32)
bfm_face_n  = np.array(bfm_aligned.face_normals, np.float64)

deca_mesh._cache.clear()
deca_verts  = np.array(deca_mesh.vertices,       np.float64)
deca_vn     = np.array(deca_mesh.vertex_normals, np.float64)

N  = len(deca_verts)
t0 = time.time()

# ── (a) Single vectorized query — ALL DECA verts at once ─────────────────────
progress("querying BFM surface for all DECA verts …")
pq_v = trimesh.proximity.ProximityQuery(bfm_aligned)
all_cp, all_dist, all_tri = pq_v.on_surface(deca_verts)   # (N,3), (N,), (N,)

# ── (b) Distance histogram → auto DIST_CAP (informational + confidence) ──────
p50, p95, p99 = (float(np.percentile(all_dist, q)) for q in (50, 95, 99))
DIST_CAP_v    = p99
print(f"  Distances — p50={p50:.5f}  p95={p95:.5f}  p99={p99:.5f}")
print(f"  Would-cover at p95 threshold : {(all_dist <= p95).sum()}/{N} "
      f"({(all_dist <= p95).sum()/N*100:.1f}%)  ← old hard-gate behaviour")

# ── (c) Confidence weights ────────────────────────────────────────────────────
bfm_fn_v  = bfm_face_n[all_tri]                            # (N,3)
dots      = (deca_vn * bfm_fn_v).sum(1)                   # (N,)
print(f"  Normal dots — min={dots.min():.3f}  mean={dots.mean():.3f}  max={dots.max():.3f}")
if dots.mean() < 0:
    print("  [WARN] Mean dot < 0 → BFM normals appear flipped. Inverting bfm_face_n.")
    bfm_face_n = -bfm_face_n
    dots       = -dots

if ABL_HARD_GATE is not None:
    # ABLATION: hard binary gate — vertices beyond threshold get weight 0 (→ black gap)
    gate       = ABL_HARD_GATE
    w_dist_v   = (all_dist <= gate).astype(np.float32)
    w_norm_v   = np.ones(len(dots), dtype=np.float32)
    print(f"  [ABLATION --hard-gate={gate}] hard binary gate. "
          f"  Excluded: {(all_dist > gate).sum()}/{len(all_dist)} vertices "
          f"({(all_dist > gate).sum()/len(all_dist)*100:.1f}% → black gaps expected)")
else:
    # Default: soft confidence — no hard rejection, 100% coverage
    #   w_dist : 1.0 at dist=0, linear falloff to 0.0 at p99
    w_dist_v = np.clip(1.0 - all_dist / (DIST_CAP_v + 1e-10), 0.0, 1.0).astype(np.float32)
    #   w_norm : 0.0 at dot=−0.8 (≈144°), 1.0 at dot=1.0
    w_norm_v = np.clip((dots + 0.8) / 1.8, 0.0, 1.0).astype(np.float32)

vertex_confidence = (w_dist_v * w_norm_v).astype(np.float32)   # (N,) used in baking
del bfm_fn_v, dots, w_dist_v, w_norm_v

# ── (d) Barycentric UV for ALL N vertices (vectorized) ───────────────────────
vf   = bfm_faces_a[all_tri]                                # (N,3)
vp0  = bfm_verts_f[vf[:,0]]; vp1 = bfm_verts_f[vf[:,1]]; vp2 = bfm_verts_f[vf[:,2]]
vu0  = bfm_face_uvs[all_tri,0]; vu1 = bfm_face_uvs[all_tri,1]; vu2 = bfm_face_uvs[all_tri,2]
uvs_all, bary_ok = bary_uv_batch(all_cp, vp0, vp1, vp2, vu0, vu1, vu2)   # (N,2), (N,)
del vp0, vp1, vp2

# ── (e) Fallback: nearest-corner UV for degenerate triangles (vectorized) ────
#   Degenerate = near-zero area triangle where barycentric system is singular.
#   Fix: project query point to the nearest of the 3 corner UVs — no loop needed.
n_degen = int((~bary_ok).sum())
if n_degen:
    fail      = ~bary_ok                                   # (K,) bool
    tf_f      = vf[fail]                                   # (K,3) BFM vert indices
    pts_fail  = deca_verts[fail]                           # (K,3) query points
    d0sq = ((pts_fail - bfm_verts_f[tf_f[:,0]]) ** 2).sum(1)
    d1sq = ((pts_fail - bfm_verts_f[tf_f[:,1]]) ** 2).sum(1)
    d2sq = ((pts_fail - bfm_verts_f[tf_f[:,2]]) ** 2).sum(1)
    nn   = np.argmin(np.stack([d0sq, d1sq, d2sq], axis=1), axis=1)  # (K,) ∈ {0,1,2}
    # numpy advanced indexing: two 1-D arrays index (F,3,2) → (K,2) — zero loops
    uvs_all[fail] = bfm_face_uvs[all_tri[fail], nn]
    del tf_f, pts_fail, d0sq, d1sq, d2sq, nn
    print(f"  Degenerate-tri fallback : {n_degen} verts patched with nearest-corner UV")
del vf, vu0, vu1, vu2, bary_ok

# ── (f) Clamp UV → sample texture for ALL N vertices (zero Python loops) ─────
uvs_clamped   = np.clip(uvs_all, 0.0, 1.0).astype(np.float64)
vertex_colors = sample_texture(uvs_clamped, tex_np)        # (N,3) float32
del uvs_all, uvs_clamped, all_cp; gc.collect()
finish_progress()

# ── (g) Coverage reports ──────────────────────────────────────────────────────
n_coloured = (vertex_colors.sum(1) > 0.01).sum()
_lm_min = lm_bfm.min(0); _lm_max = lm_bfm.max(0)
_in_face = np.all((deca_verts >= _lm_min) & (deca_verts <= _lm_max), axis=1)
n_face   = _in_face.sum()
n_face_c = (vertex_colors[_in_face].sum(1) > 0.01).sum()
del _lm_min, _lm_max, _in_face
print(f"  Vertex transfer  : {time.time()-t0:.1f}s")
print(f"  Coverage before  : (see p95 line above)")
print(f"  Coverage after   : {n_coloured}/{N} ({n_coloured/N*100:.1f}%)  [target: 100%]")
print(f"  Face-region      : {n_face_c}/{n_face} ({n_face_c/max(n_face,1)*100:.1f}%)")
print(f"  High-confidence  : {(vertex_confidence > 0.5).sum()}/{N} "
      f"({(vertex_confidence > 0.5).sum()/N*100:.1f}%)  [confidence > 0.5]")
del all_dist, all_tri


# ─────────────────────────────────────────────────────────────────────────────
# 5. PER-TEXEL BAKING
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 4/5] Per-texel baking ({TEX_SIZE}×{TEX_SIZE}) …")

# Tighten distance cap for baking now that alignment is confirmed good
BAKE_DIST_CAP = max(0.03, DIST_CAP * 0.6)
print(f"  Bake DIST_CAP : {BAKE_DIST_CAP:.5f}  (vertex-transfer used {DIST_CAP:.5f})")

deca_faces  = np.array(deca_mesh.faces,        np.int32)
deca_face_n = np.array(deca_mesh.face_normals, np.float64)

# Use high-confidence vertices only for mean_skin estimate (avoids non-face colour bias)
high_conf = vertex_confidence > 0.5
mean_skin = (vertex_colors[high_conf].mean(0) if high_conf.any()
             else vertex_colors.mean(0))
print(f"  Mean skin (RGB) : {mean_skin.round(3)}  (from {high_conf.sum()} high-conf verts)")

# ── 5a. Zero-loop vectorized rasterization ────────────────────────────────────
print("  Rasterising UV triangles (zero-loop vectorized) …")
face_map, bary_map = rasterize_uv_triangles(deca_face_uvs, TEX_SIZE)

covered_ys, covered_xs = np.where(face_map >= 0)
n_covered = len(covered_ys)
print(f"  Covered texels : {n_covered:,}")

# ── 5b. Vertex-colour UV baking ──────────────────────────────────────────────
# Interpolate per-vertex colours (from Step 4) using barycentric weights from
# the rasteriser — no BFM surface query, no sigmoid blending, no mean_skin wash-out.
print("  Baking vertex colours into UV atlas …")
t_bake = time.time()

fis = face_map[covered_ys, covered_xs]   # (N,) face index per texel
bcs = bary_map[covered_ys, covered_xs]   # (N, 3) barycentric weights
i0  = deca_faces[fis, 0]
i1  = deca_faces[fis, 1]
i2  = deca_faces[fis, 2]

out_tex = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.float32)
out_tex[covered_ys, covered_xs] = np.clip(
    bcs[:, 0:1] * vertex_colors[i0]
    + bcs[:, 1:2] * vertex_colors[i1]
    + bcs[:, 2:3] * vertex_colors[i2],
    0.0, 1.0,
)
print(f"  Baking time : {time.time()-t_bake:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SEAM DILATION + GAUSSIAN ANTI-ALIASING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5/5] Seam dilation + anti-aliasing …")
if HAS_CV2:
    out_u8       = (np.clip(out_tex, 0, 1) * 255).astype(np.uint8)
    covered_mask = (face_map >= 0)                     # (H,W) bool

    # ── Pass A: fine dilation — close UV seam gaps inside face region ─────────
    kernel_sm = np.ones((5, 5), np.uint8)
    dilated_sm = cv2.dilate(out_u8, kernel_sm, iterations=3)
    # Only apply where face_map had coverage nearby (tight seam gaps)
    seam_near = cv2.dilate(covered_mask.astype(np.uint8),
                           kernel_sm, iterations=3).astype(bool) & ~covered_mask
    out_u8[seam_near] = dilated_sm[seam_near]

    # ── Pass B: large dilation — flood ears / neck / scalp background ─────────
    kernel_lg  = np.ones((9, 9), np.uint8)
    dilated_bg = cv2.dilate(out_u8, kernel_lg, iterations=5)
    still_empty = ~covered_mask & ~seam_near          # pixels not yet filled
    out_u8[still_empty] = dilated_bg[still_empty]

    # ── Pass C: soft boundary blur — dissolve face/procedural seam line ───────
    if ABL_NO_GAUSS:
        print("  [ABLATION --no-gauss-blend] Pass C skipped — hard seam preserved.")
        blend_band = np.zeros_like(covered_mask)
    else:
        conf_f32   = covered_mask.astype(np.float32)
        conf_blur  = cv2.GaussianBlur(conf_f32, (11, 11), 3.0)
        blend_band = (conf_blur > 0.05) & (conf_blur < 0.95) & ~covered_mask
        if blend_band.any():
            blurred_full = cv2.GaussianBlur(out_u8, (5, 5), 1.5)
            alpha        = conf_blur[blend_band, None].astype(np.float32)
            out_u8[blend_band] = np.clip(
                alpha * out_u8[blend_band].astype(np.float32)
                + (1.0 - alpha) * blurred_full[blend_band].astype(np.float32),
                0, 255).astype(np.uint8)
            del blurred_full

    out_tex = out_u8.astype(np.float32) / 255.0
    print(f"  Seam near={seam_near.sum():,}  bg_fill={still_empty.sum():,}"
          f"  blend_band={blend_band.sum():,}")
    del dilated_sm, dilated_bg, kernel_sm, kernel_lg
    del seam_near, still_empty, conf_f32, conf_blur, blend_band; gc.collect()
else:
    print("  (skipped – opencv not available)")


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Export] Saving outputs …")

tex_out = os.path.join(OUT_DIR, "deca_texture.png")
Image.fromarray((np.clip(out_tex, 0, 1) * 255).astype(np.uint8)).save(tex_out)
print(f"  Texture : {tex_out}")

mtl_out = os.path.join(OUT_DIR, "deca_textured.mtl")
with open(mtl_out, "w") as f:
    f.write("newmtl deca_mat\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
            "illum 1\nmap_Kd deca_texture.png\n")
print(f"  MTL     : {mtl_out}")

obj_out     = os.path.join(OUT_DIR, "deca_textured.obj")
deca_vn_out = np.array(deca_mesh.vertex_normals, np.float64)
# 1-based indices for OBJ
vi_1  = (deca_face_v_idx  + 1).astype(np.int32)   # (F,3)
vti_1 = (deca_face_vt_idx + 1).astype(np.int32)   # (F,3)
import io as _io
_buf = _io.StringIO()
_buf.write("# Generated by align_and_transfer.py\n"
           "mtllib deca_textured.mtl\nusemtl deca_mat\no deca_textured\n\n")
# Vectorized write: np.savetxt is a single C call — no Python per-line loop
np.savetxt(_buf, deca_verts,   fmt="v  %.6f %.6f %.6f")
_buf.write("\n")
np.savetxt(_buf, deca_vt_arr,  fmt="vt %.6f %.6f")
_buf.write("\n")
np.savetxt(_buf, deca_vn_out,  fmt="vn %.6f %.6f %.6f")
_buf.write("\n")
# Face lines: single join — 1 write call instead of F write calls
_face_lines = "\n".join(
    f"f {a}/{b}/{a} {c}/{d}/{c} {e}/{g}/{e}"
    for a, b, c, d, e, g in zip(
        vi_1[:,0], vti_1[:,0],
        vi_1[:,1], vti_1[:,1],
        vi_1[:,2], vti_1[:,2],
    )
)
_buf.write(_face_lines + "\n")
with open(obj_out, "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())
del _buf, _face_lines, vi_1, vti_1, deca_vn_out
print(f"  OBJ     : {obj_out}")

# Export aligned BFM mesh for visual alignment check in browser
bfm_out = os.path.join(OUT_DIR, "bfm_aligned.obj")
bfm_verts_export = np.array(bfm_aligned.vertices, np.float64)
bfm_faces_export = np.array(bfm_aligned.faces,    np.int32)
with open(bfm_out, "w") as f:
    f.write("# BFM aligned to DECA space — generated by align_and_transfer.py\n")
    f.write("o bfm_aligned\n\n")
    for v in bfm_verts_export:
        f.write(f"v  {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    f.write("\n")
    for face in bfm_faces_export:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print(f"  BFM aligned : {bfm_out}")

# Export structured albedo in MPG FLAME format
npz_out = os.path.join(OUT_DIR, "FLAME_albedo_from_BFM.npz")
# MU = flattened (H*W*3,) float32, matching MPG albedo_mean convention
MU = np.reshape(np.clip(out_tex, 0.0, 1.0).astype(np.float32), [-1])
np.savez(npz_out, MU=MU)
print(f"  Albedo NPZ  : {npz_out}  (MU shape {MU.shape})")
del MU

print("\n" + "=" * 60)
print("  Done!")
print(f"  Output dir : {os.path.abspath(OUT_DIR)}")
print("=" * 60)
