#!/usr/bin/env python3
"""
lib_face.py — tien ich dung chung cho pipeline danh gia.

Gom:
  - doc 68 landmark 3D tu embedding FLAME (landmarks/deca.npy)
  - dung he toa do dau chuan (canonical head frame) tu landmark
  - tinh camera cho mot goc nhin (yaw, pitch) trong he do
  - chieu diem 3D -> pixel (chieu song song, tat dinh)
  - detect 68 landmark 2D tren anh tham chieu (bo boc nhieu backend)
  - bien doi similarity 2D de dua anh tham chieu ve dung khung cua anh render
"""
from __future__ import annotations
import json
import os

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CAU HINH CHUNG
# ─────────────────────────────────────────────────────────────────────────────

# Cau hinh camera cua V-Face. yaw duong = camera dich sang phia trai cua nguoi
# (tuc phia phai cua anh khi nhin chinh dien). Sua lai neu co thong so rig chinh thuc.
VIEWS = {
    "C1":  dict(yaw=-90.0, pitch=  0.0),
    "C4":  dict(yaw=-35.0, pitch=  0.0),   # camera dau vao
    "C7":  dict(yaw=  0.0, pitch=  0.0),
    "C10": dict(yaw= 35.0, pitch=  0.0),
    "C13": dict(yaw= 90.0, pitch=  0.0),
    "C17": dict(yaw=  0.0, pitch= 30.0),   # camera cao, nhin xuong
    "C24": dict(yaw=  0.0, pitch=-20.0),   # camera thap, nhin len
}
INPUT_VIEW = "C4"
NOVEL_VIEWS = [v for v in VIEWS if v != INPUT_VIEW]

IMG_SIZE = 512          # canh anh render (vuong)
IOD_FRAC = 0.26         # khoang cach lien dong tu chiem bao nhieu phan chieu cao anh

# chi so trong bo 68 landmark chuan
IDX_EYE_R = list(range(36, 42))     # mat phai cua doi tuong
IDX_EYE_L = list(range(42, 48))     # mat trai cua doi tuong
IDX_MOUTH = list(range(48, 68))
IDX_STABLE = list(range(17, 68))    # bo duong ham (0-16) vi no doi theo goc nhin


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK 3D TU EMBEDDING FLAME
# ─────────────────────────────────────────────────────────────────────────────

_LMK_EMB = None


def load_lmk_embedding(path: str = "landmarks/deca.npy"):
    """Doc full_lmk_faces_idx + full_lmk_bary_coords (chuan FLAME)."""
    global _LMK_EMB
    if _LMK_EMB is None:
        d = np.load(path, allow_pickle=True).item()
        fi = np.asarray(d["full_lmk_faces_idx"], dtype=np.int64).reshape(-1)
        bc = np.asarray(d["full_lmk_bary_coords"], dtype=np.float64).reshape(-1, 3)
        assert len(fi) == 68, f"mong doi 68 landmark, nhan duoc {len(fi)}"
        _LMK_EMB = (fi, bc)
    return _LMK_EMB


def landmarks3d(vertices: np.ndarray, faces: np.ndarray,
                emb_path: str = "landmarks/deca.npy") -> np.ndarray:
    """68 landmark 3D tren mot mesh FLAME. vertices (N,3), faces (F,3)."""
    fi, bc = load_lmk_embedding(emb_path)
    vi = faces[fi]                                   # (68,3) chi so dinh
    return (bc[:, 0:1] * vertices[vi[:, 0]] +
            bc[:, 1:2] * vertices[vi[:, 1]] +
            bc[:, 2:3] * vertices[vi[:, 2]])


# ─────────────────────────────────────────────────────────────────────────────
# HE TOA DO DAU CHUAN
# ─────────────────────────────────────────────────────────────────────────────

def canonical_frame(lm3d: np.ndarray) -> dict:
    """
    Dung he truc truc giao tu 68 landmark 3D.
      x : tu mat phai sang mat trai cua doi tuong  (sang phai cua anh khi chinh dien)
      y : huong len
      z : huong ra truoc mat
    Tra ve R (3x3, cot la cac truc), center, iod (khoang cach lien dong tu).
    """
    eye_r = lm3d[IDX_EYE_R].mean(0)
    eye_l = lm3d[IDX_EYE_L].mean(0)
    mouth = lm3d[IDX_MOUTH].mean(0)
    eye_c = 0.5 * (eye_r + eye_l)

    x = eye_l - eye_r
    iod = float(np.linalg.norm(x))
    x = x / (iod + 1e-12)

    y_tmp = eye_c - mouth                       # tam thoi: huong len
    y_tmp = y_tmp / (np.linalg.norm(y_tmp) + 1e-12)

    z = np.cross(x, y_tmp)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, x)                          # truc giao hoa
    y = y / (np.linalg.norm(y) + 1e-12)

    center = 0.5 * (eye_c + mouth)              # on dinh hon centroid toan bo
    R = np.stack([x, y, z], axis=1)             # cot = truc
    return dict(R=R, center=center, iod=iod)


def view_basis(frame: dict, yaw_deg: float, pitch_deg: float):
    """
    Tra ve (huong tu tam ra camera, vector up), deu trong toa do the gioi.
    yaw quay quanh truc y, pitch nang len/ha xuong.
    """
    ya, pa = np.radians(yaw_deg), np.radians(pitch_deg)
    d_local = np.array([np.sin(ya) * np.cos(pa),
                        np.sin(pa),
                        np.cos(ya) * np.cos(pa)])
    up_local = np.array([-np.sin(ya) * np.sin(pa),
                          np.cos(pa),
                         -np.cos(ya) * np.sin(pa)])
    R = frame["R"]
    return R @ d_local, R @ up_local


def camera_params(frame: dict, view: str, img_size: int = IMG_SIZE,
                  iod_frac: float = IOD_FRAC, dist_mult: float = 6.0):
    """Tham so camera cho mot view. Dung chieu song song nen khung hinh chinh xac."""
    v = VIEWS[view]
    d, up = view_basis(frame, v["yaw"], v["pitch"])
    center = frame["center"]
    # chieu song song: khoang cach khong anh huong khung hinh, chi can du xa
    position = center + d * (frame["iod"] * dist_mult)
    parallel_scale = frame["iod"] / (2.0 * iod_frac)   # nua chieu cao khung nhin
    return dict(position=position, focal_point=center, up=up,
                parallel_scale=parallel_scale, img_size=img_size,
                direction=d, yaw=v["yaw"], pitch=v["pitch"])


def project(points3d: np.ndarray, frame: dict, cam: dict) -> np.ndarray:
    """
    Chieu song song diem 3D -> pixel (goc trai tren, y huong xuong).
    Khop voi cach VTK dat camera trong render_views.py.
    """
    fwd = cam["focal_point"] - cam["position"]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    up = cam["up"] - np.dot(cam["up"], fwd) * fwd
    up = up / (np.linalg.norm(up) + 1e-12)
    right = np.cross(fwd, up)                      # he thuan tay trai cua man hinh
    right = right / (np.linalg.norm(right) + 1e-12)

    rel = np.asarray(points3d, dtype=np.float64) - cam["focal_point"]
    u = rel @ right
    v = rel @ up
    S = cam["img_size"]
    ps = cam["parallel_scale"]
    px = S / 2.0 + (u / ps) * (S / 2.0)
    py = S / 2.0 - (v / ps) * (S / 2.0)
    return np.stack([px, py], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK 2D TREN ANH THAM CHIEU
# ─────────────────────────────────────────────────────────────────────────────

_DET = None
_DET_KIND = None


def _init_detector():
    """Thu lan luot face_alignment -> dlib. Bao loi ro rang neu khong co."""
    global _DET, _DET_KIND
    if _DET is not None:
        return
    try:
        import face_alignment
        try:
            _DET = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")
        except AttributeError:                      # ban cu dung ten khac
            _DET = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, flip_input=False, device="cpu")
        _DET_KIND = "face_alignment"
        return
    except Exception:
        pass
    try:
        import dlib
        pred_path = os.environ.get("DLIB_LANDMARK_MODEL",
                                   "shape_predictor_68_face_landmarks.dat")
        if os.path.isfile(pred_path):
            _DET = (dlib.get_frontal_face_detector(), dlib.shape_predictor(pred_path))
            _DET_KIND = "dlib"
            return
    except Exception:
        pass
    raise RuntimeError(
        "Khong tim thay bo detect landmark 2D.\n"
        "  Cai mot trong hai:\n"
        "    pip install face-alignment\n"
        "    pip install dlib   (va tai shape_predictor_68_face_landmarks.dat,\n"
        "                        tro toi bang bien moi truong DLIB_LANDMARK_MODEL)")


def detect_landmarks2d(img_rgb: np.ndarray) -> np.ndarray | None:
    """68 landmark 2D tren anh RGB (H,W,3) uint8. None neu khong thay mat."""
    _init_detector()
    if _DET_KIND == "face_alignment":
        out = _DET.get_landmarks(img_rgb)
        if not out:
            return None
        # neu nhieu mat, lay cai co bounding box lon nhat
        best = max(out, key=lambda p: (p[:, 0].ptp() * p[:, 1].ptp()))
        return np.asarray(best, dtype=np.float64)
    else:
        import dlib
        det, pred = _DET
        gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        rects = det(gray, 1)
        if not rects:
            return None
        r = max(rects, key=lambda x: x.width() * x.height())
        sh = pred(gray, r)
        return np.array([[sh.part(i).x, sh.part(i).y] for i in range(68)],
                        dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# CAN CHINH 2D
# ─────────────────────────────────────────────────────────────────────────────

def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Umeyama 2D: tim (s, R, t) sao cho s*R*src + t ~ dst.
    Tra ve ma tran 2x3 dung cho cv2.warpAffine.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    var = (sc ** 2).sum() / len(src)
    cov = (dc.T @ sc) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max(var, 1e-12))
    t = md - s * (R @ ms)
    M = np.zeros((2, 3), np.float64)
    M[:, :2] = s * R
    M[:, 2] = t
    return M


def convex_hull_mask(pts2d: np.ndarray, size: int) -> np.ndarray:
    """Mask bool cua bao loi cac diem landmark, dung lam vung mat."""
    import cv2
    m = np.zeros((size, size), np.uint8)
    hull = cv2.convexHull(np.round(pts2d).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# TIEN ICH I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
