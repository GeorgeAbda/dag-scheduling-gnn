import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
    # Fill matrix assuming rows grouped by x then y (as written)
    k = 0
    for i in range(nx):
        for j in range(ny):
            alpha[i, j] = a[k]
            k += 1
    return xs_unique, ys_unique, alpha


def interp(p0: float, p1: float, v0: float, v1: float, level: float) -> float:
    dv = (v1 - v0)
    if abs(dv) < 1e-12:
        return p0
    t = (level - v0) / dv
    t = min(max(t, 0.0), 1.0)
    return p0 + t * (p1 - p0)


def marching_squares(xs: np.ndarray, ys: np.ndarray, alpha: np.ndarray, level: float) -> List[List[Point]]:
    nx, ny = alpha.shape
    # Build list of line segments
    segments: List[Tuple[Point, Point]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Corner values
            v_bl = alpha[i, j]
            v_br = alpha[i + 1, j]
            v_tr = alpha[i + 1, j + 1]
            v_tl = alpha[i, j + 1]
            # Inside if value >= level
            c0 = int(v_bl >= level)
            c1 = int(v_br >= level)
            c2 = int(v_tr >= level)
            c3 = int(v_tl >= level)
            code = (c0 << 0) | (c1 << 1) | (c2 << 2) | (c3 << 3)
            if code == 0 or code == 15:
                continue
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            # Edge intersections
            # Edges: bottom (bl-br), right (br-tr), top (tl-tr), left (bl-tl)
            pts: List[Point] = []
            # We handle cases by standard table
            def e_bottom():
                x = interp(x0, x1, v_bl, v_br, level)
                return (x, y0)
            def e_right():
                y = interp(y0, y1, v_br, v_tr, level)
                return (x1, y)
            def e_top():
                x = interp(x0, x1, v_tl, v_tr, level)
                return (x, y1)
            def e_left():
                y = interp(y0, y1, v_bl, v_tl, level)
                return (x0, y)
            # Cases following 4-bit code order [bl,br,tr,tl]
            # We decompose ambiguous 5/10 into two segments
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
    # Connect segments into polylines
    # Quantize points to grid to match endpoints
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
        # grow polyline both ways
        line = [p, q]
        used.add((kp, kq))
        # forward
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
        # backward
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


def to_svg_path(points: List[Point], width: int, height: int, margin: int) -> str:
    # Map x in [0,1] to [margin, width-margin]; y in [0,1] to [height-margin, margin]
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


def render_svg(out_path: Path, contours: Dict[str, List[List[Point]]], width: int = 800, height: int = 800, margin: int = 60):
    # Colors and styles per key: e.g., 'baseline-0.5', 'hetero-0.9'
    style_map = {
        'baseline-0.5': ('#1f77b4', '4', 'none'),
        'baseline-0.9': ('#1f77b4', '3', '5,5'),
        'hetero-0.5': ('#d62728', '4', 'none'),
        'hetero-0.9': ('#d62728', '3', '5,5'),
    }
    # SVG header
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
    ]
    # Axes
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='black' stroke-width='1.5' />")
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='black' stroke-width='1.5' />")
    # Ticks
    for t in np.linspace(0, 1, 6):
        xt = margin + t * (width - 2 * margin)
        parts.append(f"<line x1='{xt:.2f}' y1='{y0}' x2='{xt:.2f}' y2='{y0+5}' stroke='black' stroke-width='1' />")
        parts.append(f"<text x='{xt:.2f}' y='{y0+20}' font-size='12' text-anchor='middle' fill='#444'>{t:.1f}</text>")
        yt = height - margin - t * (height - 2 * margin)
        parts.append(f"<line x1='{x0}' y1='{yt:.2f}' x2='{x0-5}' y2='{yt:.2f}' stroke='black' stroke-width='1' />")
        parts.append(f"<text x='{x0-10}' y='{yt+4:.2f}' font-size='12' text-anchor='end' fill='#444'>{t:.1f}</text>")
    parts.append(f"<text x='{(x0+x1)//2}' y='{height-15}' font-size='14' fill='#000'>makespan (normalized)</text>")
    parts.append(f"<text x='15' y='{(y0+y1)//2}' font-size='14' fill='#000' transform='rotate(-90 15 {(y0+y1)//2})'>active_energy (normalized)</text>")
    # Contours
    for key, lines in contours.items():
        color, width_s, dash = style_map.get(key, ('#000', '2', 'none'))
        for pts in lines:
            d = to_svg_path(pts, width, height, margin)
            if not d:
                continue
            parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width_s}' stroke-dasharray='{dash}' />")
    # Legend
    lx, ly = width - margin - 180, margin + 10
    legend = [
        f"<rect x='{lx}' y='{ly}' width='170' height='70' fill='white' stroke='#ccc' />",
        f"<path d='M {lx+15} {ly+20} L {lx+55} {ly+20}' stroke='#1f77b4' stroke-width='4' />",
        f"<text x='{lx+60}' y='{ly+24}' font-size='12'>baseline 50%</text>",
        f"<path d='M {lx+15} {ly+40} L {lx+55} {ly+40}' stroke='#1f77b4' stroke-width='3' stroke-dasharray='5,5' />",
        f"<text x='{lx+60}' y='{ly+44}' font-size='12'>baseline 90%</text>",
        f"<path d='M {lx+15} {ly+60} L {lx+55} {ly+60}' stroke='#d62728' stroke-width='4' />",
        f"<text x='{lx+60}' y='{ly+64}' font-size='12'>hetero 50%</text>",
    ]
    parts.extend(legend)
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description='Plot EAF iso-contours (SVG) for two models from their EAF grids.')
    ap.add_argument('--grid-a', required=True, help='CSV for model A (baseline) EAF grid')
    ap.add_argument('--grid-b', required=True, help='CSV for model B (hetero) EAF grid')
    ap.add_argument('--out', required=True, help='Output SVG file path')
    ap.add_argument('--levels', type=str, default='0.5,0.9', help='Comma-separated levels, e.g., 0.5,0.9')
    args = ap.parse_args()

    xs_a, ys_a, alpha_a = read_eaf_grid(Path(args.grid_a))
    xs_b, ys_b, alpha_b = read_eaf_grid(Path(args.grid_b))
    # Sanity: ensure same grid
    if xs_a.size != xs_b.size or ys_a.size != ys_b.size:
        print('[warn] grids differ in size; proceeding with A grid for marching squares')
    xs, ys = xs_a, ys_a

    levels = [float(x.strip()) for x in args.levels.split(',') if x.strip()]
    contours: Dict[str, List[List[Point]]] = {}

    for lev in levels:
        lines_a = marching_squares(xs, ys, alpha_a, lev)
        lines_b = marching_squares(xs, ys, alpha_b, lev)
        key_a = f'baseline-{lev:.1f}'
        key_b = f'hetero-{lev:.1f}'
        contours[key_a] = lines_a
        contours[key_b] = lines_b

    render_svg(Path(args.out), contours)
    print(f"[done] Wrote SVG contours to {args.out}")


if __name__ == '__main__':
    main()
