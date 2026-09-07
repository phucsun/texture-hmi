#!/usr/bin/env python3
"""
run_all.py — chay ca pipeline danh gia bang mot lenh.

    python3 run_all.py                              # 6 phuong phap co san du lieu
    python3 run_all.py --limit 5                    # thu nhanh tren 5 subject
    python3 run_all.py --tessera-dir tessera_output # them dong TESSERA
    python3 run_all.py --skip-render                # da render roi, chi do lai

Cac buoc:
  1. baselines.py      sinh texture mirroring / telea / navier_stokes
  2. render_views.py   render 7 goc nhin cho tung phuong phap
  3. eval_protocol.py  do metric trong mask mat  -> results.csv
  4. eval_protocol.py  do them tren full crop    -> results_fullcrop.csv (phu luc)
  5. check_results.py  kiem tra + in bang + sinh dong LaTeX

Moi buoc deu bo qua viec da lam, nen ngat giua chung roi chay lai duoc.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

CPU_METHODS = ["backbone", "mirroring", "telea", "navier_stokes", "prior_transfer"]


def run(cmd: list[str], title: str) -> bool:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print("  $ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BASE)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  [LOI] ma thoat {r.returncode} sau {dt:.0f}s")
        return False
    print(f"  [OK] {dt:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--tessera-dir", default=None)
    ap.add_argument("--albedo-dir", default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-fullcrop", action="store_true")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    common = []
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.subjects:
        common += ["--subjects"] + args.subjects

    methods = list(CPU_METHODS)
    if args.albedo_dir:
        methods.append("deca_albedo")
    if args.tessera_dir:
        methods.append("tessera")

    if not args.skip_baselines:
        if not run([PY, "baselines.py"] + common, "1/5  Sinh texture baseline"):
            return 1

    if not args.skip_render:
        cmd = [PY, "render_views.py", "--methods"] + methods + common
        if args.tessera_dir:
            cmd += ["--tessera-dir", args.tessera_dir]
        if args.albedo_dir:
            cmd += ["--albedo-dir", args.albedo_dir]
        if not run(cmd, f"2/5  Render 7 goc nhin ({len(methods)} phuong phap)"):
            return 1

    if not run([PY, "eval_protocol.py", "--methods"] + methods +
               common + ["--region", "face_mask", "--out", args.out],
               "3/5  Do metric trong mask mat"):
        return 1

    if not args.skip_fullcrop:
        run([PY, "eval_protocol.py", "--methods"] + methods +
            common + ["--region", "full_crop", "--out", "results_fullcrop.csv"],
            "4/5  Do metric tren full crop (phu luc)")

    run([PY, "check_results.py", args.out, "--latex"],
        "5/5  Kiem tra ket qua + sinh bang LaTeX")

    print("\n" + "=" * 70)
    print("  HOAN TAT")
    print("=" * 70)
    print(f"  ket qua chinh   : {args.out}")
    print(f"  ket qua phu luc : results_fullcrop.csv")
    print(f"  do phu          : baselines/coverage.csv")
    print(f"  anh render      : renders/")
    print("\n  Dan cac dong LaTeX o tren vao bang tab:main trong main.tex.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
