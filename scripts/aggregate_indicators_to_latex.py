#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np


def read_last_metrics(metrics_csv: Path) -> Optional[Dict[str, float]]:
    if not metrics_csv.exists():
        return None
    with metrics_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        last: Optional[Dict[str, str]] = None
        for row in r:
            last = row
    if not last:
        return None
    out: Dict[str, float] = {}
    for k in ("hv_cum", "igd_cum", "igd_plus_cum", "spacing_cum"):
        try:
            out[k] = float(last.get(k, "nan"))
        except Exception:
            out[k] = float("nan")
    try:
        out["iter"] = float(last.get("iter", "nan"))
        out["step"] = float(last.get("step", "nan"))
    except Exception:
        pass
    return out


def read_front_extremes(front_csv: Path) -> Tuple[float, float]:
    """Return (min_makespan, min_active_energy) from final_front.csv."""
    if not front_csv.exists():
        return (float("nan"), float("nan"))
    mins = [float("inf"), float("inf")]
    with front_csv.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        mk_idx, ae_idx = 0, 1
        if header:
            # accept either names or plain 2-col
            if "mk_ratio" in header and "ae_ratio" in header:
                mk_idx = header.index("mk_ratio")
                ae_idx = header.index("ae_ratio")
            elif "makespan" in header and "active_energy" in header:
                mk_idx = header.index("makespan")
                ae_idx = header.index("active_energy")
        for row in r:
            try:
                mk = float(row[mk_idx]); ae = float(row[ae_idx])
                mins[0] = min(mins[0], mk)
                mins[1] = min(mins[1], ae)
            except Exception:
                continue
    if mins[0] == float("inf"): mins[0] = float("nan")
    if mins[1] == float("inf"): mins[1] = float("nan")
    return (mins[0], mins[1])


def scan_arches(ind_root: Path) -> List[str]:
    archs: List[str] = []
    for p in sorted(ind_root.iterdir()):
        if p.is_dir():
            archs.append(p.name)
    return archs


def main():
    ap = argparse.ArgumentParser(description="Aggregate indicators per seed and emit a LaTeX table with metrics per architecture.")
    ap.add_argument("root", type=str, help="Out-root produced by run_nsga_and_indicators.py")
    ap.add_argument("--arches", nargs="*", default=None, help="Architectures to include (folder names under indicators_seed_*). If not set, auto-detect.")
    ap.add_argument("--out-tex", type=str, default="indicators_summary.tex", help="Output LaTeX file path")
    ap.add_argument("--floatfmt", type=str, default=".3f", help="Format for floating numbers")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    # Discover seeds
    seeds: List[int] = []
    for p in sorted(root.glob("indicators_seed_*")):
        if p.is_dir():
            try:
                seeds.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
    if not seeds:
        raise SystemExit(f"No indicators_seed_* found under {root}")

    seeds = sorted(seeds)

    # Determine arches
    if args.arches:
        arches = args.arches
    else:
        # use first seed to list arches
        arches = scan_arches(root / f"indicators_seed_{seeds[0]}")
    if not arches:
        raise SystemExit("No architecture subdirectories discovered.")

    # Collect table data: rows=seed, columns=per-arch metrics
    # For each arch and seed, read metrics_over_time.csv last row for hv/igd/igd+/spacing; read final_front.csv for min mk/ae
    data: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed in seeds:
        row: Dict[str, Dict[str, float]] = {}
        ind_root = root / f"indicators_seed_{seed}"
        for arch in arches:
            a_dir = ind_root / arch
            metrics = read_last_metrics(a_dir / "metrics_over_time.csv") or {}
            mk_min, ae_min = read_front_extremes(a_dir / "final_front.csv")
            metrics.update({"min_mk": mk_min, "min_ae": ae_min})
            row[arch] = metrics
        data[seed] = row

    # Build LaTeX tabular
    # Columns: config(seed) | For each arch: mk_min, ae_min, hv, igd, igd+, spacing
    cols_per_arch = [
        ("min_mk", "mk"),
        ("min_ae", "ae"),
        ("hv_cum", "hv"),
        ("igd_cum", "igd"),
        ("igd_plus_cum", "igd+"),
        ("spacing_cum", "spacing"),
    ]

    def fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "--"
        return f"{x:{args.floatfmt}}"

    header_arch = " & ".join([f"\\multicolumn{{{len(cols_per_arch)}}}{{c}}{{{arch}}}" for arch in arches])
    header_metrics = " & ".join([" ".join([lab for _k, lab in cols_per_arch]) for _ in arches])

    lines: List[str] = []
    lines.append("% Auto-generated by aggregate_indicators_to_latex.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    # Column spec: 1 for seed + len(arches)*len(cols)
    col_spec = "l" + "".join(["rrrrrr" for _ in arches])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(f"Config \\ & {header_arch} \\ ")
    lines.append(" \\cmidrule(lr){2-" + f"{1 + len(arches)*len(cols_per_arch)}" + "}")
    # Secondary header: repeated metric labels per arch
    lines.append(f" \\ & {header_metrics} \\ ")
    lines.append("\\midrule")

    for seed in seeds:
        row_cells: List[str] = [str(seed)]
        for arch in arches:
            vals = data.get(seed, {}).get(arch, {})
            for key, _lab in cols_per_arch:
                row_cells.append(fmt(vals.get(key, float('nan'))))
        lines.append(" \\ ".join([" "] + []) )
        lines.append(" ")
        lines[-1] = " ".join([])
        # Proper row assembly
        lines.append(" ")
    # Rebuild rows correctly (fix above assembly)
    lines = lines[:7]  # keep preamble
    for seed in seeds:
        row_cells = [str(seed)]
        for arch in arches:
            vals = data.get(seed, {}).get(arch, {})
            for key, _lab in cols_per_arch:
                row_cells.append(fmt(vals.get(key, float('nan'))))
        lines.append(" \\ ".join([" & ".join(row_cells)]) + " \")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Indicators per configuration (rows) and architecture (groups of columns). Lower is better for mk, ae, igd, igd+, spacing; higher is better for hv.}")
    lines.append("\\label{tab:indicators_by_config}")
    lines.append("\\end{table}")

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[done] Wrote LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
