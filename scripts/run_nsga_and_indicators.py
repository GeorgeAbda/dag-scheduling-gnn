#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import csv
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Metrics helpers to match pareto_compare_models.py logic
try:
    from scheduler.rl_model.robustness.metrics import (
        pareto_non_dominated,
        ideal_nadir,
        normalize_points,
    )
except Exception:
    pareto_non_dominated = None
    ideal_nadir = None
    normalize_points = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
NSGA_SCRIPT = SCRIPTS_DIR / "nsga_schedule_search.py"
INDICATORS_SCRIPT = SCRIPTS_DIR / "pareto_indicator_analysis.py"


def run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


# ============================
# EAF SVG plotting (integrated)
# ============================

Point = Tuple[float, float]


def read_eaf_grid(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_list: List[float] = []
    ys_list: List[float] = []
    alphas: List[float] = []
    with path.open("r") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            xs_list.append(float(row[0]))
            ys_list.append(float(row[1]))
            alphas.append(float(row[2]))
    xs = np.array(xs_list, dtype=float)
    ys = np.array(ys_list, dtype=float)
    a = np.array(alphas, dtype=float)
    xs_unique = np.unique(xs)
    ys_unique = np.unique(ys)
    nx, ny = xs_unique.size, ys_unique.size
    alpha = np.zeros((nx, ny), dtype=float)
    k = 0
    for i in range(nx):
        for j in range(ny):
            alpha[i, j] = a[k]
            k += 1
    return xs_unique, ys_unique, alpha


def _interp(p0: float, p1: float, v0: float, v1: float, level: float) -> float:
    dv = (v1 - v0)
    if abs(dv) < 1e-12:
        return p0
    t = (level - v0) / dv
    t = min(max(t, 0.0), 1.0)
    return p0 + t * (p1 - p0)


def marching_squares(xs: np.ndarray, ys: np.ndarray, alpha: np.ndarray, level: float) -> List[List[Point]]:
    nx, ny = alpha.shape
    segments: List[Tuple[Point, Point]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v_bl = alpha[i, j]
            v_br = alpha[i + 1, j]
            v_tr = alpha[i + 1, j + 1]
            v_tl = alpha[i, j + 1]
            c0 = int(v_bl >= level)
            c1 = int(v_br >= level)
            c2 = int(v_tr >= level)
            c3 = int(v_tl >= level)
            code = (c0 << 0) | (c1 << 1) | (c2 << 2) | (c3 << 3)
            if code == 0 or code == 15:
                continue
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]

            def e_bottom():
                x = _interp(x0, x1, v_bl, v_br, level)
                return (x, y0)

            def e_right():
                y = _interp(y0, y1, v_br, v_tr, level)
                return (x1, y)

            def e_top():
                x = _interp(x0, x1, v_tl, v_tr, level)
                return (x, y1)

            def e_left():
                y = _interp(y0, y1, v_bl, v_tl, level)
                return (x0, y)

            table = {
                1: [(e_left, e_bottom)],
                2: [(e_bottom, e_right)],
                3: [(e_left, e_right)],
                4: [(e_right, e_top)],
                5: [(e_bottom, e_right), (e_left, e_top)],
                6: [(e_bottom, e_top)],
                7: [(e_left, e_top)],
                8: [(e_top, e_left)],
                9: [(e_top, e_bottom)],
                10: [(e_right, e_bottom), (e_top, e_left)],
                11: [(e_top, e_right)],
                12: [(e_right, e_left)],
                13: [(e_right, e_bottom)],
                14: [(e_bottom, e_left)],
            }
            seg_defs = table.get(code, [])
            for a, b in seg_defs:
                p = a(); q = b()
                segments.append((p, q))

    def key(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0] * 1e6)), int(round(pt[1] * 1e6)))

    adj: Dict[Tuple[int, int], List[Point]] = {}
    for p, q in segments:
        adj.setdefault(key(p), []).append(q)
        adj.setdefault(key(q), []).append(p)
    used = set()
    lines: List[List[Point]] = []
    for p, q in segments:
        kp = key(p); kq = key(q)
        if (kp, kq) in used or (kq, kp) in used:
            continue
        line = [p, q]
        used.add((kp, kq))
        cur = q
        while True:
            nbrs = adj.get(key(cur), [])
            nxt = None
            for n in nbrs:
                if len(line) >= 2 and abs(n[0] - line[-2][0]) < 1e-12 and abs(n[1] - line[-2][1]) < 1e-12:
                    continue
                if ((key(cur), key(n)) not in used) and ((key(n), key(cur)) not in used):
                    nxt = n
                    break
            if nxt is None:
                break
            used.add((key(cur), key(nxt)))
            line.append(nxt)
            cur = nxt
        cur = p
        while True:
            nbrs = adj.get(key(cur), [])
            nxt = None
            for n in nbrs:
                if len(line) >= 2 and abs(n[0] - line[1][0]) < 1e-12 and abs(n[1] - line[1][1]) < 1e-12:
                    continue
                if ((key(cur), key(n)) not in used) and ((key(n), key(cur)) not in used):
                    nxt = n
                    break
            if nxt is None:
                break
            used.add((key(cur), key(nxt)))
            line.insert(0, nxt)
            cur = nxt
        lines.append(line)
    return lines


def _to_svg_path(points: List[Point], width: int, height: int, margin: int) -> str:
    def sx(x: float) -> float:
        return margin + x * (width - 2 * margin)
    def sy(y: float) -> float:
        return height - margin - y * (height - 2 * margin)
    if not points:
        return ""
    cmds = []
    x0, y0 = points[0]
    cmds.append(f"M {sx(x0):.2f} {sy(y0):.2f}")
    for (x, y) in points[1:]:
        cmds.append(f"L {sx(x):.2f} {sy(y):.2f}")
    return " ".join(cmds)


def render_eaf_svg(out_path: Path, contours: Dict[str, List[List[Point]]], name_a: str, name_b: str, width: int = 800, height: int = 800, margin: int = 60):
    style_map = {
        f'{name_a}-0.5': ('#1f77b4', '4', 'none'),
        f'{name_a}-0.9': ('#1f77b4', '3', '5,5'),
        f'{name_b}-0.5': ('#d62728', '4', 'none'),
        f'{name_b}-0.9': ('#d62728', '3', '5,5'),
    }
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
    ]
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='black' stroke-width='1.5' />")
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='black' stroke-width='1.5' />")
    for t in np.linspace(0, 1, 6):
        xt = margin + t * (width - 2 * margin)
        parts.append(f"<line x1='{xt:.2f}' y1='{y0}' x2='{xt:.2f}' y2='{y0+5}' stroke='black' stroke-width='1' />")
        parts.append(f"<text x='{xt:.2f}' y='{y0+20}' font-size='12' text-anchor='middle' fill='#444'>{t:.1f}</text>")
        yt = height - margin - t * (height - 2 * margin)
        parts.append(f"<line x1='{x0}' y1='{yt:.2f}' x2='{x0-5}' y2='{yt:.2f}' stroke='black' stroke-width='1' />")
        parts.append(f"<text x='{x0-10}' y='{yt+4:.2f}' font-size='12' text-anchor='end' fill='#444'>{t:.1f}</text>")
    parts.append(f"<text x='{(x0+x1)//2}' y='{height-15}' font-size='14' fill='#000'>makespan (normalized)</text>")
    parts.append(f"<text x='15' y='{(y0+y1)//2}' font-size='14' fill='#000' transform='rotate(-90 15 {(y0+y1)//2})'>active_energy (normalized)</text>")
    for key, lines in contours.items():
        color, width_s, dash = style_map.get(key, ('#000', '2', 'none'))
        for pts in lines:
            d = _to_svg_path(pts, width, height, margin)
            if not d:
                continue
            parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width_s}' stroke-dasharray='{dash}' />")
    lx, ly = width - margin - 200, margin + 10
    legend = [
        f"<rect x='{lx}' y='{ly}' width='190' height='70' fill='white' stroke='#ccc' />",
        f"<path d='M {lx+15} {ly+20} L {lx+55} {ly+20}' stroke='#1f77b4' stroke-width='4' />",
        f"<text x='{lx+60}' y='{ly+24}' font-size='12'>{name_a} 50%</text>",
        f"<path d='M {lx+15} {ly+40} L {lx+55} {ly+40}' stroke='#1f77b4' stroke-width='3' stroke-dasharray='5,5' />",
        f"<text x='{lx+60}' y='{ly+44}' font-size='12'>{name_a} 90%</text>",
        f"<path d='M {lx+15} {ly+60} L {lx+55} {ly+60}' stroke='#d62728' stroke-width='4' />",
        f"<text x='{lx+60}' y='{ly+64}' font-size='12'>{name_b} 50%</text>",
    ]
    parts.extend(legend)
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))

def compute_eaf_from_points(runs: List[List[Point]], grid_n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    alpha = np.zeros((grid_n, grid_n), dtype=float)
    R = max(1, len(runs))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z0, z1 = float(x), float(y)
            attained = 0
            for run in runs:
                if not run:
                    continue
                # any point dominates (minimization)
                if any((p[0] <= z0 + 1e-12 and p[1] <= z1 + 1e-12) for p in run):
                    attained += 1
            alpha[i, j] = attained / R
    return xs, ys, alpha


def render_eaf_svg_multi(out_path: Path, contours_per_model: Dict[str, Dict[float, List[List[Point]]]], width: int = 900, height: int = 900, margin: int = 60):
    # Assign distinct colors (match template ordering: blue, red, green, ...)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    levels_sorted = sorted({lev for m in contours_per_model.values() for lev in m.keys()})
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
    ]
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='black' stroke-width='1.5' />")
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='black' stroke-width='1.5' />")
    for t in np.linspace(0, 1, 6):
        xt = margin + t * (width - 2 * margin)
        parts.append(f"<line x1='{xt:.2f}' y1='{y0}' x2='{xt:.2f}' y2='{y0+5}' stroke='black' stroke-width='1' />")
        parts.append(f"<text x='{xt:.2f}' y='{y0+20}' font-size='12' text-anchor='middle' fill='#444'>{t:.1f}</text>")
        yt = height - margin - t * (height - 2 * margin)
        parts.append(f"<line x1='{x0}' y1='{yt:.2f}' x2='{x0-5}' y2='{yt:.2f}' stroke='black' stroke-width='1' />")
        parts.append(f"<text x='{x0-10}' y='{yt+4:.2f}' font-size='12' text-anchor='end' fill='#444'>{t:.1f}</text>")
    parts.append(f"<text x='{(x0+x1)//2}' y='{height-15}' font-size='14' fill='#000'>makespan (normalized)</text>")
    parts.append(f"<text x='15' y='{(y0+y1)//2}' font-size='14' fill='#000' transform='rotate(-90 15 {(y0+y1)//2})'>active_energy (normalized)</text>")

    # Draw contours per model per level (solid for first level, dashed for others)
    for i, (model, lvldict) in enumerate(contours_per_model.items()):
        color = palette[i % len(palette)]
        for j, level in enumerate(levels_sorted):
            lines = lvldict.get(level, [])
            dash = 'none' if j == 0 else '5,5'
            width_s = '4' if j == 0 else '3'
            for pts in lines:
                d = _to_svg_path(pts, width, height, margin)
                if not d:
                    continue
                parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width_s}' stroke-dasharray='{dash}' />")

    # Legend
    lx, ly = width - margin - 240, margin + 10
    h = 24 * (len(contours_per_model) * len(levels_sorted) + 1)
    parts.append(f"<rect x='{lx}' y='{ly}' width='230' height='{h}' fill='white' stroke='#ccc' />")
    yy = ly + 20
    for i, (model, _lvldict) in enumerate(contours_per_model.items()):
        color = palette[i % len(palette)]
        parts.append(f"<text x='{lx+10}' y='{yy}' font-size='13' fill='#000'>{model}</text>")
        y2 = yy + 6
        xstart = lx + 110
        for j, level in enumerate(levels_sorted):
            dash = 'none' if j == 0 else '5,5'
            width_s = '4' if j == 0 else '3'
            parts.append(f"<path d='M {xstart} {y2} L {xstart+40} {y2}' stroke='{color}' stroke-width='{width_s}' stroke-dasharray='{dash}' />")
            parts.append(f"<text x='{xstart+45}' y='{y2+4}' font-size='12'>{level:g}</text>")
            y2 += 18
        yy = max(yy + 24, y2 + 6)

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description="Run NSGA reference front for a configuration and compute indicators for three architectures using that NSGA front.")
    # Architecture dirs
    ap.add_argument("--arch-dirs", nargs=3, required=True, help="Three per-architecture directories containing *_pareto_*.pt checkpoints")
    ap.add_argument("--out-root", type=str, default=str(REPO_ROOT / "figures" / "config_nsga_indicators"), help="Root output directory")
    # NSGA config
    ap.add_argument("--dag-method", type=str, default="linear")
    ap.add_argument("--gnp-p", type=float, default=None)
    ap.add_argument("--gnp-min-n", type=int, default=20)
    ap.add_argument("--gnp-max-n", type=int, default=20)
    ap.add_argument("--host-count", type=int, default=4)
    ap.add_argument("--vm-count", type=int, default=10)
    ap.add_argument("--workflow-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--seed-start", type=int, default=None, help="If set with --seed-end, sweep seeds in [start,end] inclusive")
    ap.add_argument("--seed-end", type=int, default=None, help="End of seed sweep (inclusive)")
    ap.add_argument("--device", type=str, default="cpu")
    # NSGA hyperparameters
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--key-sigma", type=float, default=0.25)
    ap.add_argument("--key-ratio", type=float, default=0.15)
    ap.add_argument("--vm-flip-prob", type=float, default=0.05)
    ap.add_argument("--cx-prob", type=float, default=0.9)
    # Indicator options
    ap.add_argument("--normalize", action="store_true", help="Normalize objectives for indicator computation")
    # EAF plotting (optional)
    ap.add_argument("--eaf-grid-a", type=str, default=None, help="Path to EAF grid CSV for model A (x,y,alpha)")
    ap.add_argument("--eaf-grid-b", type=str, default=None, help="Path to EAF grid CSV for model B (x,y,alpha)")
    ap.add_argument("--eaf-out", type=str, default=None, help="Output SVG path for EAF contours overlay")
    ap.add_argument("--eaf-levels", type=str, default="0.5,0.9", help="Comma-separated levels, e.g., 0.25,0.5,0.75,0.9")
    ap.add_argument("--eaf-name-a", type=str, default="baseline", help="Label for model A in EAF legend")
    ap.add_argument("--eaf-name-b", type=str, default="hetero", help="Label for model B in EAF legend")
    # EAF 4-way overlay (optional): three architectures + NSGA
    ap.add_argument("--eaf4-grid-mlp", type=str, default=None, help="EAF grid CSV for MLP-only model")
    ap.add_argument("--eaf4-grid-baseline", type=str, default=None, help="EAF grid CSV for baseline model")
    ap.add_argument("--eaf4-grid-hetero", type=str, default=None, help="EAF grid CSV for hetero model")
    ap.add_argument("--eaf4-grid-nsga", type=str, default=None, help="EAF grid CSV for NSGA (if single-run, treat as degenerate EAF)")
    ap.add_argument("--eaf4-name-mlp", type=str, default="mlp_only")
    ap.add_argument("--eaf4-name-baseline", type=str, default="baseline")
    ap.add_argument("--eaf4-name-hetero", type=str, default="hetero")
    ap.add_argument("--eaf4-name-nsga", type=str, default="nsga")
    ap.add_argument("--eaf4-out", type=str, default=None, help="Output SVG path for 4-way EAF overlay")
    ap.add_argument("--eaf4-levels", type=str, default="0.5,0.9", help="Levels for 4-way EAF (comma-separated)")
    # Auto 4-way EAF from this run's outputs
    ap.add_argument("--eaf4-auto", action="store_true", help="Compute a 4-way EAF from this run's final fronts (3 arch) + NSGA ref front")
    ap.add_argument("--eaf4-auto-out", type=str, default=None, help="Output SVG path for auto 4-way EAF")
    ap.add_argument("--eaf4-auto-grid", type=int, default=100, help="Grid resolution for auto EAF")
    ap.add_argument("--eaf4-auto-levels", type=str, default="0.5,0.9", help="Levels for auto 4-way EAF")
    # Combined fronts plot (optional)
    ap.add_argument("--plot-fronts-out", type=str, default=None, help="If set, save a combined plot with 3 architectures' final fronts and the NSGA reference front.")
    ap.add_argument("--plot-fronts-normalize", action="store_true", default=True, help="Normalize fronts using ideal/nadir over the union of all plotted points")
    ap.add_argument("--plot-fronts-pareto", action="store_true", default=True, help="Pareto-filter each series before plotting")
    args = ap.parse_args()

    arch_dirs = [Path(p).resolve() for p in args.arch_dirs]
    for p in arch_dirs:
        if not p.exists():
            raise SystemExit(f"[error] Architecture directory not found: {p}")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare seed list
    if args.seed_start is not None and args.seed_end is not None:
        if args.seed_end < args.seed_start:
            raise SystemExit("[error] --seed-end must be >= --seed-start")
        seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    else:
        seeds = [int(args.seed)]

    # Iterate seeds
    for seed in seeds:
        print(f"[seed] Running pipeline for seed={seed}")
        nsga_out = out_root / f"nsga_seed_{seed}"
        ind_out = out_root / f"indicators_seed_{seed}"
        nsga_out.mkdir(parents=True, exist_ok=True)
        ind_out.mkdir(parents=True, exist_ok=True)

        # 1) Run NSGA for this seed (reproducible)
        nsga_cmd = [
            sys.executable, str(NSGA_SCRIPT),
            "--population", str(args.population),
            "--generations", str(args.generations),
            "--key-sigma", str(args.key_sigma),
            "--key-ratio", str(args.key_ratio),
            "--vm-flip-prob", str(args.vm_flip_prob),
            "--cx-prob", str(args.cx_prob),
            "--dag-method", str(args.dag_method),
            "--gnp-min-n", str(args.gnp_min_n),
            "--gnp-max-n", str(args.gnp_max_n),
            "--host-count", str(args.host_count),
            "--vm-count", str(args.vm_count),
            "--workflow-count", str(args.workflow_count),
            "--seed", str(seed),
            "--device", str(args.device),
            "--out-dir", str(nsga_out),
        ]
        if args.gnp_p is not None:
            nsga_cmd.extend(["--gnp-p", str(args.gnp_p)])
        run(nsga_cmd, cwd=REPO_ROOT)

        ref_csv = nsga_out / "reference_front.csv"
        if not ref_csv.exists():
            raise SystemExit(f"[error] NSGA reference front not found at {ref_csv}")

        # 2) Run indicator analysis for each architecture using the NSGA reference front
        for arch_dir in tqdm(arch_dirs, desc="Indicators"):
            arch_name = arch_dir.name
            out_dir = ind_out / arch_name
            cmd = [
                sys.executable, str(INDICATORS_SCRIPT),
                "--dir", str(arch_dir),
                "--out", str(out_dir),
                "--ref-front-csv", str(ref_csv),
            ]
            if args.normalize:
                cmd.append("--normalize")
            run(cmd, cwd=REPO_ROOT)
            print(f"[done] Indicators for {arch_name} written to: {out_dir}")

    # 3) Optional: EAF contours overlay (if grids are provided)
    if args.eaf_grid_a and args.eaf_grid_b and args.eaf_out:
        grid_a = Path(args.eaf_grid_a)
        grid_b = Path(args.eaf_grid_b)
        if not grid_a.exists() or not grid_b.exists():
            print(f"[warn] EAF grids not found: {grid_a} or {grid_b}. Skipping EAF plot.")
        else:
            xs_a, ys_a, alpha_a = read_eaf_grid(grid_a)
            xs_b, ys_b, alpha_b = read_eaf_grid(grid_b)
            if xs_a.size != xs_b.size or ys_a.size != ys_b.size:
                print("[warn] EAF grids differ in size; proceeding with A grid for contours")
            xs, ys = xs_a, ys_a
            levels = [float(x.strip()) for x in str(args.eaf_levels).split(',') if x.strip()]
            contours: Dict[str, List[List[Point]]] = {}
            for lev in tqdm(levels, desc="EAF levels"):
                lines_a = marching_squares(xs, ys, alpha_a, lev)
                lines_b = marching_squares(xs, ys, alpha_b, lev)
                contours[f"{args.eaf_name_a}-{lev:.1f}"] = lines_a
                contours[f"{args.eaf_name_b}-{lev:.1f}"] = lines_b
            render_eaf_svg(Path(args.eaf_out), contours, args.eaf_name_a, args.eaf_name_b)
            print(f"[done] Wrote EAF SVG: {args.eaf_out}")

    # 3b) Optional: 4-way EAF overlay (MLP, baseline, hetero, NSGA)
    if args.eaf4_out and args.eaf4_grid_mlp and args.eaf4_grid_baseline and args.eaf4_grid_hetero and args.eaf4_grid_nsga:
        grid_paths = {
            args.eaf4_name_mlp: Path(args.eaf4_grid_mlp),
            args.eaf4_name_baseline: Path(args.eaf4_grid_baseline),
            args.eaf4_name_hetero: Path(args.eaf4_grid_hetero),
            args.eaf4_name_nsga: Path(args.eaf4_grid_nsga),
        }
        missing = [name for name, p in grid_paths.items() if not p.exists()]
        if missing:
            print(f"[warn] Missing EAF 4-way grids for: {', '.join(missing)}; skipping 4-way EAF plot")
        else:
            # Use MLP grid as reference axes
            xs_ref, ys_ref, _ = read_eaf_grid(grid_paths[args.eaf4_name_mlp])
            levels = [float(x.strip()) for x in str(args.eaf4_levels).split(',') if x.strip()]
            contours_per_model: Dict[str, Dict[float, List[List[Point]]]] = {}
            for name, p in grid_paths.items():
                xs, ys, alpha = read_eaf_grid(p)
                if xs.size != xs_ref.size or ys.size != ys_ref.size:
                    print(f"[warn] Grid size mismatch for {name}; using its own grid for contours anyway")
                lvldict: Dict[float, List[List[Point]]] = {}
                for lev in levels:
                    lvldict[lev] = marching_squares(xs, ys, alpha, lev)
                contours_per_model[name] = lvldict
            render_eaf_svg_multi(Path(args.eaf4_out), contours_per_model)
            print(f"[done] Wrote 4-way EAF SVG: {args.eaf4_out}")

        # 3c) Optional: Auto-compute 4-way EAF from final fronts of this seed
        if args.eaf4_auto and args.eaf4_auto_out:
            # Collect per-arch final fronts and NSGA ref
            per_model: Dict[str, List[Point]] = {}
            for arch_dir in arch_dirs:
                name = arch_dir.name
                ff = ind_out / name / "final_front.csv"
                pts = []
                if ff.exists():
                    with ff.open("r") as f:
                        r = csv.reader(f)
                        header = next(r, None)
                        mk_idx, ae_idx = 0, 1
                        if header:
                            try:
                                mk_idx = header.index("mk_ratio")
                                ae_idx = header.index("ae_ratio")
                            except Exception:
                                try:
                                    mk_idx = header.index("makespan")
                                    ae_idx = header.index("active_energy")
                                except Exception:
                                    mk_idx, ae_idx = 0, 1
                        for row in r:
                            try:
                                pts.append((float(row[mk_idx]), float(row[ae_idx])))
                            except Exception:
                                continue
                per_model[name] = pts
            # NSGA
            nsga_pts = []
            if Path(ref_csv).exists():
                with Path(ref_csv).open("r") as f:
                    r = csv.reader(f)
                    header = next(r, None)
                    mk_idx, ae_idx = 0, 1
                    if header:
                        try:
                            mk_idx = header.index("mk_ratio")
                            ae_idx = header.index("ae_ratio")
                        except Exception:
                            try:
                                mk_idx = header.index("makespan")
                                ae_idx = header.index("active_energy")
                            except Exception:
                                mk_idx, ae_idx = 0, 1
                    for row in r:
                        try:
                            nsga_pts.append((float(row[mk_idx]), float(row[ae_idx])))
                        except Exception:
                            continue
                per_model["nsga"] = nsga_pts

            # Normalize on union
            all_pts: List[Point] = []
            for ps in per_model.values():
                all_pts.extend(ps)
            if len(all_pts) == 0:
                print("[warn] No points found to compute auto EAF; skipping")
            else:
                if ideal_nadir is not None and normalize_points is not None:
                    ideal, nadir = ideal_nadir(all_pts)
                    per_model_n = {k: normalize_points(v, ideal, nadir) for k, v in per_model.items()}
                else:
                    per_model_n = per_model

                # Build degenerate runs (single run per model)
                levels = [float(x.strip()) for x in str(args.eaf4_auto_levels).split(',') if x.strip()]
                contours_per_model: Dict[str, Dict[float, List[List[Point]]]] = {}
                xs_ref = ys_ref = None
                for name, pts in per_model_n.items():
                    xs, ys, alpha = compute_eaf_from_points([pts], grid_n=int(args.eaf4_auto_grid))
                    if xs_ref is None:
                        xs_ref, ys_ref = xs, ys
                    lvldict: Dict[float, List[List[Point]]] = {}
                    for lev in levels:
                        lvldict[lev] = marching_squares(xs, ys, alpha, lev)
                    contours_per_model[name] = lvldict
                # Seed-suffixed auto EAF out
                auto_out = Path(args.eaf4_auto_out)
                if "{seed}" in str(auto_out):
                    auto_out = Path(str(auto_out).format(seed=seed))
                else:
                    auto_out = auto_out.with_name(auto_out.stem + f"_seed{seed}" + auto_out.suffix)
                render_eaf_svg_multi(auto_out, contours_per_model)
                print(f"[done] Wrote auto 4-way EAF SVG: {auto_out}")

        # 4) Optional: Combined Pareto fronts plot for this seed (3 architectures + NSGA)
        if args.plot_fronts_out:
            if plt is None:
                print("[warn] matplotlib not available; cannot plot combined fronts.")
            else:
                def read_points_csv(path: Path) -> List[Point]:
                    pts: List[Point] = []
                    if not path.exists():
                        return pts
                    with path.open("r") as f:
                        r = csv.reader(f)
                        header = next(r, None)
                        mk_idx, ae_idx = 0, 1
                        if header:
                            # Try common header names
                            try:
                                mk_idx = header.index("mk_ratio")
                                ae_idx = header.index("ae_ratio")
                            except Exception:
                                try:
                                    mk_idx = header.index("makespan")
                                    ae_idx = header.index("active_energy")
                                except Exception:
                                    mk_idx, ae_idx = 0, 1
                        for row in r:
                            try:
                                pts.append((float(row[mk_idx]), float(row[ae_idx])))
                            except Exception:
                                continue
                    return pts

            # Collect per-arch final fronts
            series: List[Tuple[str, List[Point]]] = []
            for arch_dir in arch_dirs:
                arch_name = arch_dir.name
                ff = ind_out / arch_name / "final_front.csv"
                pts = read_points_csv(ff)
                if pts:
                    series.append((arch_name, pts))
                else:
                    print(f"[warn] Missing or empty final_front for {arch_name}: {ff}")
            # NSGA reference front
            nsga_pts = read_points_csv(ref_csv)
            if nsga_pts:
                series.append(("nsga", nsga_pts))
            else:
                print(f"[warn] Missing NSGA reference front points: {ref_csv}")

            if series:
                # Build union to compute ideal/nadir if normalization enabled
                series_proc: List[Tuple[str, List[Point]]] = []
                if args.plot_fronts_pareto and pareto_non_dominated is not None:
                    src_for_norm = []
                    for name, pts in series:
                        nd = pareto_non_dominated(pts)
                        series_proc.append((name, nd))
                        src_for_norm.extend(nd)
                else:
                    series_proc = [(n, p) for (n, p) in series]
                    src_for_norm = [pt for (_n, ps) in series_proc for pt in ps]

                if args.plot_fronts_normalize and ideal_nadir is not None and normalize_points is not None and len(src_for_norm) > 0:
                    ideal, nadir = ideal_nadir(src_for_norm)
                    series_plot = [(n, normalize_points(ps, ideal, nadir)) for (n, ps) in series_proc]
                    xlab, ylab = "makespan (normalized)", "active energy (normalized)"
                else:
                    series_plot = series_proc
                    xlab, ylab = "makespan", "active energy"

                plt.figure(figsize=(6.4, 6.0), dpi=140)
                # Match template: blue, red, green, purple
                colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
                for i, (name, pts) in enumerate(series_plot):
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    # Sort by x for a nicer line connection
                    order = np.argsort(xs)
                    xs = [xs[k] for k in order]
                    ys = [ys[k] for k in order]
                    c = colors[i % len(colors)]
                    plt.plot(xs, ys, '-', lw=1.5, alpha=0.8, color=c)
                    plt.scatter(xs, ys, s=14, alpha=0.9, color=c, label=name)
                plt.xlabel(xlab)
                plt.ylabel(ylab)
                plt.title("Final Pareto fronts (3 models) + NSGA")
                plt.grid(True, alpha=0.3)
                plt.legend(frameon=False)
                out_path = Path(args.plot_fronts_out)
                if "{seed}" in str(out_path):
                    out_path = Path(str(out_path).format(seed=seed))
                else:
                    out_path = out_path.with_name(out_path.stem + f"_seed{seed}" + out_path.suffix)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
                print(f"[done] Wrote combined fronts plot: {out_path}")

    print(f"[all-done] Outputs in: {out_root}")


if __name__ == "__main__":
    main()
