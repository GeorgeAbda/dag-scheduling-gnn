#!/usr/bin/env python3
"""
Plot mapped vs original performance across p_target for each source model p using Plotly (HTML).

Reads the CSV produced by cross_p_generalization_ot.py (ot_mapping_summary.csv) and,
for each p_source present, creates a single interactive figure (HTML) with two subplots:
 - Left: Makespan vs p_target
 - Right: Active+Idle Energy (API) vs p_target
Each subplot contains three curves:
 - Target baseline (model trained and evaluated at p_target)
 - Source model evaluated at p_target (unmapped)
 - Source model evaluated at p_target with mapped edge embeddings (mapped)

Output files:
 - <out_dir>/perf_p{p_source}.html (and PNG if kaleido is available)

Usage:
  python -m scheduler.viz_results.decision_boundaries.plot_mapped_performance_plotly \
    --csv scheduler/viz_results/decision_boundaries/cross_p/ot_mapping_summary.csv \
    --out-dir scheduler/viz_results/decision_boundaries/cross_p/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any
import csv as _csv
import math

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:  # pragma: no cover
    print("[error] Plotly is not installed. Install it with:\n  pip install plotly\n(Optional for PNG export)\n  pip install -U kaleido")
    raise


def _ensure_sorted_unique(xs: List[float]) -> List[float]:
    try:
        return sorted(list({float(x) for x in xs}))
    except Exception:
        return xs


def _to_float(val: Any) -> float:
    try:
        if val is None:
            return math.nan
        if isinstance(val, float):
            return val
        if isinstance(val, int):
            return float(val)
        s = str(val).strip()
        if s == '' or s.lower() == 'nan':
            return math.nan
        return float(s)
    except Exception:
        return math.nan


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', newline='') as f:
        r = _csv.DictReader(f)
        return [row for row in r]


def group_by_source(rows: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
    by_src: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        try:
            p_s = _to_float(r.get('p_source'))
            by_src.setdefault(p_s, []).append(r)
        except Exception:
            continue
    return by_src


def plot_for_source(rows: List[Dict[str, Any]], p_source: float, out_dir: Path, ps_order: List[float] | None = None):
    # Determine x order
    pts = [_to_float(r.get('p_target')) for r in rows]
    if ps_order:
        xs = [p for p in ps_order if p in set(pts)]
    else:
        xs = _ensure_sorted_unique(pts)
    if not xs:
        return

    # Aggregate by p_target (mean over duplicates if any)
    def agg(metric: str, p: float) -> float:
        vals = [_to_float(r.get(metric)) for r in rows if _to_float(r.get('p_target')) == p]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return math.nan
        return float(sum(vals) / len(vals))

    mk_tgt = [agg('tgt_mk', p) for p in xs]
    mk_unm = [agg('src_mk_unmapped', p) for p in xs]
    mk_map = [agg('src_mk_mapped', p) for p in xs]
    api_tgt = [agg('tgt_api', p) for p in xs]
    api_unm = [agg('src_api_unmapped', p) for p in xs]
    api_map = [agg('src_api_mapped', p) for p in xs]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Makespan vs p_target (p_source={p_source})",
        f"API vs p_target (p_source={p_source})",
    ))

    # Left: makespan
    fig.add_trace(go.Scatter(x=xs, y=mk_tgt, mode='lines+markers', name='Target baseline', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=mk_unm, mode='lines+markers', name='Original (unmapped)', line=dict(color='#d62728')), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=mk_map, mode='lines+markers', name='Mapped', line=dict(color='#2ca02c')), row=1, col=1)
    fig.update_xaxes(title_text='p_target', row=1, col=1)
    fig.update_yaxes(title_text='Average makespan', row=1, col=1)

    # Right: API
    fig.add_trace(go.Scatter(x=xs, y=api_tgt, mode='lines+markers', name='Target baseline', showlegend=False, line=dict(color='#1f77b4')), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=api_unm, mode='lines+markers', name='Original (unmapped)', showlegend=False, line=dict(color='#d62728')), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=api_map, mode='lines+markers', name='Mapped', showlegend=False, line=dict(color='#2ca02c')), row=1, col=2)
    fig.update_xaxes(title_text='p_target', row=1, col=2)
    fig.update_yaxes(title_text='Active + Idle energy', row=1, col=2)

    fig.update_layout(title_text=f'Performance across p_target for source p={p_source}', legend=dict(orientation='h'))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f'perf_p{p_source:.1f}.html'
    fig.write_html(str(out_html))

    # Optional PNG export if kaleido is available
    try:
        fig.write_image(str(out_html.with_suffix('.png')))
    except Exception:
        pass

    print(f"[plot] Saved {out_html} (and PNG if available)")


def main():
    ap = argparse.ArgumentParser(description='Plot mapped vs original performance for each source model p (Plotly)')
    ap.add_argument('--csv', type=Path, required=True, help='Path to ot_mapping_summary.csv')
    ap.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries/cross_p/plots_html'))
    ap.add_argument('--ps-order', type=str, default='0.8,0.7,0.5,0.3,0.1', help='Comma-separated order for p_target')
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f'CSV not found: {args.csv}')

    rows = read_csv(args.csv)
    grouped = group_by_source(rows)

    ps_order = None
    if args.ps_order:
        try:
            ps_order = [float(x) for x in args.ps_order.split(',') if x.strip()]
        except Exception:
            ps_order = None

    for p_source, rlist in grouped.items():
        try:
            plot_for_source(rlist, float(p_source), args.out_dir, ps_order)
        except Exception as e:
            print(f'[plot] Failed for p_source={p_source}: {e}')


if __name__ == '__main__':
    main()
