#!/usr/bin/env python3
"""
check_results.py — kiem tra file ket qua theo subject va sinh cac bang cho bai bao.

Cach dung:
    python3 check_results.py results.csv
    python3 check_results.py results.csv --latex     # in ra dong LaTeX de dan vao main.tex

Schema bat buoc (mot dong = mot phep do):
    vid,view,method,psnr,ssim,lpips,csim

Cot tuy chon:
    region   face_mask | full_crop   (mac dinh face_mask)
    q_pred   du bao Q(theta), chi can cho dong tessera — dung cho tuong quan VQP
"""
import argparse, csv, math, statistics as st, sys
from collections import defaultdict

REQUIRED = ["vid", "view", "method", "psnr", "ssim", "lpips", "csim"]

METHODS = [  # thu tu xuat hien trong Bang 1
    ("backbone",       r"Backbone (baked) \cite{feng2021deca}"),
    ("mirroring",      r"Mirroring"),
    ("telea",          r"Telea inpainting"),
    ("navier_stokes",  r"Navier--Stokes inpainting"),
    ("deca_albedo",    r"DECA $+$ albedo prior"),
    ("prior_transfer", r"Generative prior $+$ transfer \cite{li2024uvidm}"),
    ("tessera",        r"\textbf{TESSERA}"),
]
INPUT_VIEW = "C4"
NOVEL_VIEWS = ["C7", "C10", "C1", "C13", "C17", "C24"]
ALL_VIEWS = [INPUT_VIEW] + NOVEL_VIEWS

RANGES = {"psnr": (0, 60), "ssim": (0, 1), "lpips": (0, 1), "csim": (-1, 1)}


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("File rong.")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        sys.exit(f"Thieu cot bat buoc: {missing}\nCot hien co: {list(rows[0])}")
    return rows


def validate(rows):
    problems, warnings = [], []

    for i, r in enumerate(rows, 2):
        for c in ("psnr", "ssim", "lpips", "csim"):
            v = r[c].strip()
            if v == "" or v.lower() in ("nan", "none"):
                problems.append(f"dong {i}: {c} rong")
                continue
            try:
                x = float(v)
            except ValueError:
                problems.append(f"dong {i}: {c}='{v}' khong phai so")
                continue
            lo, hi = RANGES[c]
            if not (lo <= x <= hi):
                problems.append(f"dong {i}: {c}={x} ngoai khoang [{lo},{hi}]")
        if "±" in r["psnr"] or "+/-" in r["psnr"]:
            problems.append(f"dong {i}: gia tri chua '±' — day la bang tong hop, "
                            f"khong phai ket qua theo subject")

    seen_m = {r["method"] for r in rows}
    unknown = seen_m - {m for m, _ in METHODS}
    if unknown:
        warnings.append(f"method la: {sorted(unknown)} (mong doi: {[m for m,_ in METHODS]})")

    seen_v = {r["view"] for r in rows}
    if not seen_v <= set(ALL_VIEWS):
        warnings.append(f"view la: {sorted(seen_v - set(ALL_VIEWS))}")

    vids = sorted({r["vid"] for r in rows})
    print(f"  subject : {len(vids)}")
    print(f"  view    : {sorted(seen_v)}")
    print(f"  method  : {sorted(seen_m)}")
    print(f"  so dong : {len(rows)}")

    # day du to hop
    have = {(r["vid"], r["view"], r["method"]) for r in rows}
    expect = {(v, w, m) for v in vids for w in seen_v for m in seen_m}
    if len(have) != len(rows):
        problems.append(f"co {len(rows)-len(have)} dong trung lap (vid,view,method)")
    miss = expect - have
    if miss:
        warnings.append(f"thieu {len(miss)}/{len(expect)} to hop, vi du: {sorted(miss)[:3]}")

    return problems, warnings


def agg(rows, methods, views, metric):
    """Trung binh theo subject truoc (tren cac view), roi thong ke tren subject."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["view"] in views:
            per[r["method"]][r["vid"]].append(float(r[metric]))
    out = {}
    for m in methods:
        vals = [st.mean(v) for v in per[m].values()]
        if vals:
            out[m] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--latex", action="store_true", help="in dong LaTeX cho Bang 1")
    a = ap.parse_args()

    rows = load(a.csv_path)
    print("=" * 68)
    print("  KIEM TRA FILE KET QUA")
    print("=" * 68)
    problems, warnings = validate(rows)

    if warnings:
        print("\n[CANH BAO]")
        for w in warnings: print("  -", w)
    if problems:
        print("\n[LOI]")
        for p in problems[:20]: print("  -", p)
        if len(problems) > 20: print(f"  ... con {len(problems)-20} loi")
        sys.exit("\nFile chua dung schema. Sua roi chay lai.")
    print("\n  -> schema hop le")

    methods = [m for m, _ in METHODS if any(r["method"] == m for r in rows)]
    views = [v for v in NOVEL_VIEWS if any(r["view"] == v for r in rows)]
    if not views:
        sys.exit("Khong tim thay view held-out nao.")

    print("\n" + "=" * 68)
    print(f"  BANG 1 — trung binh tren {len(views)} view held-out {views}")
    print("=" * 68)
    A = {k: agg(rows, methods, views, k) for k in ("psnr", "ssim", "lpips", "csim")}
    print(f"{'method':22s}{'PSNR':>16s}{'SSIM':>16s}{'LPIPS':>16s}{'CSIM':>16s}")
    for m in methods:
        line = f"{m:22s}"
        for k, d in (("psnr", 2), ("ssim", 3), ("lpips", 3), ("csim", 3)):
            mu, sd, _ = A[k][m]
            line += f"{mu:>10.{d}f} ±{sd:.{d}f}"
        print(line)

    print("\n" + "=" * 68)
    print("  PSNR THEO TUNG CAMERA")
    print("=" * 68)
    hdr = [v for v in ALL_VIEWS if any(r["view"] == v for r in rows)]
    print(f"{'method':22s}" + "".join(f"{v:>9s}" for v in hdr))
    for m in methods:
        line = f"{m:22s}"
        for v in hdr:
            mu = agg(rows, [m], [v], "psnr").get(m)
            line += f"{mu[0]:>9.2f}" if mu else f"{'-':>9s}"
        print(line)

    # kiem dinh ghep cap
    try:
        from scipy.stats import wilcoxon, spearmanr
        if "tessera" in methods:
            others = [m for m in methods if m != "tessera"]
            best = max(others, key=lambda m: A["psnr"][m][0]) if others else None
            if best:
                print("\n" + "=" * 68)
                print(f"  WILCOXON GHEP CAP:  tessera  vs  {best}")
                print("=" * 68)
                for k in ("psnr", "ssim", "lpips", "csim"):
                    x, y = A[k]["tessera"][2], A[k][best][2]
                    if len(x) == len(y) and len(x) > 5:
                        s_, p_ = wilcoxon(x, y)
                        win = sum(1 for i in range(len(x))
                                  if (x[i] < y[i] if k == "lpips" else x[i] > y[i]))
                        print(f"  {k:6s} W={s_:9.1f}  p={p_:.3e}   thang {win}/{len(x)}")
        # tuong quan VQP
        if "q_pred" in rows[0]:
            pairs = [(float(r["q_pred"]), float(r["psnr"]), float(r["ssim"]),
                      float(r["lpips"]), float(r["csim"]))
                     for r in rows if r["method"] == "tessera" and r.get("q_pred", "").strip()]
            if len(pairs) > 5:
                print("\n" + "=" * 68)
                print(f"  TUONG QUAN Q(theta) vs DO THAT  (n={len(pairs)})")
                print("=" * 68)
                q = [p[0] for p in pairs]
                for j, k in enumerate(("psnr", "ssim", "lpips", "csim"), start=1):
                    rho, p_ = spearmanr(q, [p[j] for p in pairs])
                    print(f"  {k:6s} Spearman rho={rho:+.3f}  p={p_:.3g}")
    except ImportError:
        print("\n(scipy khong co — bo qua kiem dinh)")

    # failure case
    if "tessera" in methods:
        _, _, vals = A["csim"]["tessera"]
        per = defaultdict(list)
        for r in rows:
            if r["method"] == "tessera" and r["view"] in views:
                per[r["vid"]].append(float(r["csim"]))
        rank = sorted(((st.mean(v), k) for k, v in per.items()))
        print("\n" + "=" * 68)
        print("  UNG VIEN FAILURE CASE (CSIM thap nhat)")
        print("=" * 68)
        for c, v in rank[:5]:
            print(f"  {v}   CSIM={c:.3f}")

    if a.latex:
        print("\n" + "=" * 68)
        print("  DONG LATEX CHO BANG 1")
        print("=" * 68)
        for m, label in METHODS:
            if m not in methods:
                continue
            cells = []
            for k, d in (("psnr", 2), ("ssim", 3), ("lpips", 3), ("csim", 3)):
                mu, sd, _ = A[k][m]
                cells.append(f"${mu:.{d}f} \\pm {sd:.{d}f}$")
            print(f"    {label:52s} & " + " & ".join(cells) + r" \\")

    print("\nHoan tat.")


if __name__ == "__main__":
    main()
