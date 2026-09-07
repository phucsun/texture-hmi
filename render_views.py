#!/usr/bin/env python3
"""
render_views.py — render tat dinh cac goc nhin cho tung phuong phap.

Moi phuong phap dung CHUNG mot mesh; chi khac texture. Nho vay mask be mat va
landmark chieu 2D chi phu thuoc (subject, view) nen duoc tinh mot lan va dung lai.

Dau ra:
    renders/<method>/<vid>_<view>.png     anh render (RGB)
    renders/_mask/<vid>_<view>.png        mask be mat (nhi phan)
    renders/_lmk/<vid>_<view>.json        68 landmark chieu ra pixel

Vi du:
    python3 render_views.py --methods backbone prior_transfer
    python3 render_views.py --methods tessera --tessera-dir tessera_output
    python3 render_views.py --limit 3 --debug-overlay        # kiem tra nhanh
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import trimesh

import lib_face as LF

# ─────────────────────────────────────────────────────────────────────────────
# DUONG DAN DU LIEU
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DECA_DIR = os.path.join(BASE, "results_C4_final")     # mesh + texture baked
BATCH_DIR = os.path.join(BASE, "output_batch")        # ket qua kien truc cu
BASELINE_DIR = os.path.join(BASE, "baselines")        # do baselines.py sinh ra
OUT_DIR = os.path.join(BASE, "renders")

ALL_METHODS = ["backbone", "mirroring", "telea", "navier_stokes",
               "deca_albedo", "prior_transfer", "tessera"]


def subject_dirs() -> dict:
    """vid -> ten thu muc trong results_C4_final."""
    out = {}
    if not os.path.isdir(DECA_DIR):
        sys.exit(f"Khong thay {DECA_DIR}")
    for d in sorted(os.listdir(DECA_DIR)):
        if d.startswith("V") and os.path.isdir(os.path.join(DECA_DIR, d)):
            out[d[:7]] = d
    return out


def texture_path(method: str, vid: str, sdir: str, args) -> str | None:
    if method == "backbone":
        return os.path.join(DECA_DIR, sdir, f"{sdir}.png")
    if method == "prior_transfer":
        return os.path.join(BATCH_DIR, vid, "deca_texture.png")
    if method in ("mirroring", "telea", "navier_stokes"):
        return os.path.join(BASELINE_DIR, vid, f"{method}.png")
    if method == "deca_albedo":
        return os.path.join(args.albedo_dir, f"{vid}.png") if args.albedo_dir else None
    if method == "tessera":
        if not args.tessera_dir:
            return None
        for cand in (os.path.join(args.tessera_dir, vid, "texture.png"),
                     os.path.join(args.tessera_dir, vid, "deca_texture.png"),
                     os.path.join(args.tessera_dir, f"{vid}.png")):
            if os.path.isfile(cand):
                return cand
        return os.path.join(args.tessera_dir, vid, "texture.png")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────

def make_plotter(size: int):
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(size, size))
    p.background_color = "black"
    return p


def set_camera(p, cam):
    p.enable_parallel_projection()
    p.camera.position = tuple(float(x) for x in cam["position"])
    p.camera.focal_point = tuple(float(x) for x in cam["focal_point"])
    p.camera.up = tuple(float(x) for x in cam["up"])
    p.camera.parallel_scale = float(cam["parallel_scale"])


def render_once(mesh_pv, texture, cam, size: int) -> np.ndarray:
    p = make_plotter(size)
    if texture is None:
        p.add_mesh(mesh_pv, color="white", lighting=False)
    else:
        p.add_mesh(mesh_pv, texture=texture, lighting=False)
    set_camera(p, cam)
    img = p.screenshot(return_img=True)
    p.close()
    return np.asarray(img)[:, :, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["backbone", "prior_transfer"],
                    choices=ALL_METHODS)
    ap.add_argument("--views", nargs="+", default=list(LF.VIEWS.keys()))
    ap.add_argument("--subjects", nargs="*", default=None, help="vid cu the")
    ap.add_argument("--limit", type=int, default=0, help="chi lam N subject dau")
    ap.add_argument("--size", type=int, default=LF.IMG_SIZE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--tessera-dir", default=None, help="thu muc texture cua TESSERA")
    ap.add_argument("--albedo-dir", default=None, help="thu muc albedo cua DECA")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug-overlay", action="store_true",
                    help="ve landmark chieu len anh render de kiem tra can chinh")
    args = ap.parse_args()

    import pyvista as pv
    from PIL import Image

    subs = subject_dirs()
    vids = args.subjects or sorted(subs)
    if args.limit:
        vids = vids[:args.limit]
    print(f"Subject: {len(vids)} | method: {args.methods} | view: {args.views}")

    mask_dir = os.path.join(args.out, "_mask")
    lmk_dir = os.path.join(args.out, "_lmk")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(lmk_dir, exist_ok=True)
    for m in args.methods:
        os.makedirs(os.path.join(args.out, m), exist_ok=True)

    n_done = n_skip = 0
    for i, vid in enumerate(vids, 1):
        sdir = subs.get(vid)
        if sdir is None:
            print(f"[{i}/{len(vids)}] {vid}: khong co trong results_C4_final -> bo qua")
            continue
        obj = os.path.join(DECA_DIR, sdir, f"{sdir}.obj")
        if not os.path.isfile(obj):
            print(f"[{i}/{len(vids)}] {vid}: thieu {obj} -> bo qua")
            continue

        # hinh hoc + he toa do chuan (process=False de giu thu tu dinh cua FLAME)
        tm = trimesh.load(obj, process=False)
        frame = LF.canonical_frame(
            LF.landmarks3d(np.asarray(tm.vertices, np.float64),
                           np.asarray(tm.faces, np.int64)))
        lm3d = LF.landmarks3d(np.asarray(tm.vertices, np.float64),
                              np.asarray(tm.faces, np.int64))
        mesh_pv = pv.read(obj)

        # cache texture da nap cho subject nay
        tex_cache: dict[str, object] = {}
        for view in args.views:
            cam = LF.camera_params(frame, view, img_size=args.size)

            # mask + landmark: chi phu thuoc hinh hoc, tinh mot lan
            mpath = os.path.join(mask_dir, f"{vid}_{view}.png")
            lpath = os.path.join(lmk_dir, f"{vid}_{view}.json")
            if args.overwrite or not os.path.isfile(mpath):
                mimg = render_once(mesh_pv, None, cam, args.size)
                mask = (mimg.max(axis=2) > 20).astype(np.uint8) * 255
                Image.fromarray(mask).save(mpath)
            if args.overwrite or not os.path.isfile(lpath):
                pts = LF.project(lm3d, frame, cam)
                LF.save_json(dict(vid=vid, view=view, yaw=cam["yaw"],
                                  pitch=cam["pitch"], size=args.size,
                                  landmarks=pts.tolist()), lpath)

            for method in args.methods:
                dst = os.path.join(args.out, method, f"{vid}_{view}.png")
                if not args.overwrite and os.path.isfile(dst):
                    n_skip += 1
                    continue
                tp = texture_path(method, vid, sdir, args)
                if tp is None or not os.path.isfile(tp):
                    if view == args.views[0]:
                        print(f"    {vid}/{method}: thieu texture ({tp}) -> bo qua")
                    continue
                if tp not in tex_cache:
                    tex_cache[tp] = pv.read_texture(tp)
                img = render_once(mesh_pv, tex_cache[tp], cam, args.size)

                if args.debug_overlay:
                    import cv2
                    pts = LF.project(lm3d, frame, cam)
                    img = img.copy()
                    for (x, y) in pts:
                        cv2.circle(img, (int(round(x)), int(round(y))), 2,
                                   (0, 255, 0), -1)
                Image.fromarray(img).save(dst)
                n_done += 1

        print(f"[{i}/{len(vids)}] {vid}  ok")

    print(f"\nDa render {n_done} anh, bo qua {n_skip} anh da co.")
    print(f"Ket qua: {args.out}/")


if __name__ == "__main__":
    main()
