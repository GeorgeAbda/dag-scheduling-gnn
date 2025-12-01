#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np

Point = Tuple[float, float]

def read_front_csv(path: Path) -> List[Point]:
    if not path.exists():
        return []
    pts: List[Point] = []
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        mk_idx, ae_idx = 0, 1
        if header:
            if "mk_ratio" in header and "ae_ratio" in header:
                mk_idx = header.index("mk_ratio")
                ae_idx = header.index("ae_ratio")
            elif "makespan" in header and "active_energy" in header:
                mk_idx = header.index("makespan")
                ae_idx = header.index("active_energy")
        for row in r:
            try:
                x = float(row[mk_idx]); y = float(row[ae_idx])
                pts.append((x, y))
            except Exception:
                continue
    return pts

def pareto_nd(points: List[Point]) -> List[Point]:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return []
    keep = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        if not keep[i]:
            continue
        p = pts[i]
        dom = (pts <= p + 1e-12).all(axis=1) & (pts < p - 1e-12).any(axis=1)
        if dom.any():
            keep[i] = False
            continue
        dominated_by_p = (pts >= p - 1e-12).all(axis=1) & (pts > p + 1e-12).any(axis=1)
        keep &= ~dominated_by_p
        keep[i] = True
    return [tuple(pts[i]) for i in range(pts.shape[0]) if keep[i]]

def igd(candidate: List[Point], reference_front: List[Point]) -> float:
    if len(reference_front) == 0:
        return float("nan")
    if len(candidate) == 0:
        return float("inf")
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    dists = []
    for r in ref:
        diffs = cand - r[None, :]
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists.append(float(np.min(ds)))
    return float(np.mean(dists))

def igd_plus(candidate: List[Point], reference_front: List[Point]) -> float:
    if len(reference_front) == 0:
        return float("nan")
    if len(candidate) == 0:
        return float("inf")
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    dists = []
    for r in ref:
        diffs = cand - r[None, :]
        diffs = np.maximum(diffs, 0.0)
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists.append(float(np.min(ds)))
    return float(np.mean(dists))

def spacing(front: List[Point]) -> float:
    if len(front) < 2:
        return float("nan")
    from scipy.spatial.distance import cdist
    pts = np.asarray(front, dtype=float)
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    d_min = np.min(D, axis=1)
    d_bar = float(np.mean(d_min))
    return float(np.sqrt(np.mean((d_min - d_bar) ** 2)))

def hypervolume_2d(points: List[Point], reference: Point) -> float:
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(pareto_nd(points), dtype=float)
    if pts.size == 0:
        return 0.0
    mask = (pts[:, 0] <= ref[0] + 1e-12) & (pts[:, 1] <= ref[1] + 1e-12)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    hv = 0.0
    prev_m = ref[0]
    prev_e = ref[1]
    for i in range(pts.shape[0] - 1, -1, -1):
        m, e = pts[i]
        width = max(0.0, prev_m - m)
        height = max(0.0, prev_e - e)
        hv += width * height
        prev_m = m
        prev_e = min(prev_e, e)
    return float(max(0.0, hv))

def main():
    ap = argparse.ArgumentParser(description="Aggregate re-evaluated per-seed fronts to a LaTeX table using metrics vs each seed's NSGA front.")
    ap.add_argument("root", type=str, help="Out-root used by run_nsga_and_reeval.py")
    ap.add_argument("--arches", nargs="*", default=None)
    ap.add_argument("--out-tex", type=str, default="reeval_summary.tex")
    ap.add_argument("--floatfmt", type=str, default=".3f")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    seeds: List[int] = []
    for p in sorted(root.glob("reeval_seed_*")):
        if p.is_dir():
            try:
                seeds.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
    if not seeds:
        raise SystemExit(f"No reeval_seed_* directories found under {root}")
    seeds = sorted(seeds)

    if args.arches:
        arches = args.arches
    else:
        # infer from first seed's fronts_per_seed
        first = root / f"reeval_seed_{seeds[0]}" / "fronts_per_seed"
        arches = [d.name for d in sorted(first.iterdir()) if d.is_dir()]
    if not arches:
        raise SystemExit("No architectures found in fronts_per_seed/")

    def fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "--"
        return f"{x:{args.floatfmt}}"

    cols_per_arch = [
        ("min_mk", "mk"),
        ("min_ae", "ae"),
        ("hv", "hv"),
        ("igd", "igd"),
        ("igd+", "igd+"),
        ("spacing", "spacing"),
    ]

    header_arch = " & ".join([f"\\multicolumn{{{len(cols_per_arch)}}}{{c}}{{{arch}}}" for arch in arches])
    header_metrics_cells: List[str] = []
    for _ in arches:
        header_metrics_cells.extend([lab for _k, lab in cols_per_arch])
    header_metrics = " & ".join(header_metrics_cells)

    lines: List[str] = []
    lines.append("% Auto-generated by aggregate_reeval_to_latex.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    col_spec = "l" + "".join(["rrrrrr" for _ in arches])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(f"Config & {header_arch} \\")
    lines.append(" \\cmidrule(lr){2-" + f"{1 + len(arches)*len(cols_per_arch)}" + "}")
    lines.append(f"  & {header_metrics} \\")
    lines.append("\\midrule")

    for seed in seeds:
        nsga_ref = read_front_csv(root / f"nsga_seed_{seed}" / "reference_front.csv")
        ref_pt = None
        if nsga_ref:
            rx = 1.1 * max(x for (x, _y) in nsga_ref)
            ry = 1.1 * max(y for (_x, y) in nsga_ref)
            ref_pt = (rx, ry)
        cells: List[str] = [str(seed)]
        for arch in arches:
            fr = read_front_csv(root / f"reeval_seed_{seed}" / "fronts_per_seed" / arch / f"seed_{seed}.csv")
            fr_nd = pareto_nd(fr)
            mk_min = min([x for (x, _y) in fr_nd], default=float("nan"))
            ae_min = min([y for (_x, y) in fr_nd], default=float("nan"))
            if ref_pt is None:
                if fr_nd:
                    rx = 1.1 * max(x for (x, _y) in fr_nd)
                    ry = 1.1 * max(y for (_x, y) in fr_nd)
                    ref_pt_use = (rx, ry)
                else:
                    ref_pt_use = (1.0, 1.0)
            else:
                ref_pt_use = ref_pt
            hv = hypervolume_2d(fr_nd, ref_pt_use)
            igd_v = igd(fr_nd, nsga_ref) if nsga_ref else float("nan")
            igd_p = igd_plus(fr_nd, nsga_ref) if nsga_ref else float("nan")
            sp = spacing(fr_nd)
            for v in (mk_min, ae_min, hv, igd_v, igd_p, sp):
                cells.append(fmt(v))
        lines.append(" & ".join(cells) + " \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Re-evaluated per-seed Pareto metrics per architecture. Lower is better for mk, ae, igd, igd+, spacing; higher is better for hv.}")
    lines.append("\\label{tab:reeval_by_config}")
    lines.append("\\end{table}")

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[done] Wrote LaTeX table to {out_path}")

if __name__ == "__main__":
    main()
