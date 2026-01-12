#!/usr/bin/env python3
"""
Plot mapped vs original performance across p_target for each source model p.

Reads the CSV produced by cross_p_generalization_ot.py (ot_mapping_summary.csv) and,
for each p_source present, creates a single figure with two subplots:
 - Left: Makespan vs p_target
 - Right: Active+Idle Energy (API) vs p_target
Each subplot contains three curves:
 - Target baseline (model trained and evaluated at p_target)
 - Source model evaluated at p_target (unmapped)
 - Source model evaluated at p_target with mapped edge embeddings (mapped)

Output files:
 - <out_dir>/perf_p{p_source}.png and .pdf

Usage:
  python -m scheduler.viz_results.decision_boundaries.plot_mapped_performance \
    --csv scheduler/viz_results/decision_boundaries/cross_p/ot_mapping_summary.csv \
    --out-dir scheduler/viz_results/decision_boundaries/cross_p/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_sorted_unique(xs: List[float]) -> List[float]:
    try:
        return sorted(list({float(x) for x in xs}))
    except Exception:
        return xs


def plot_for_source(df: pd.DataFrame, p_source: float, out_dir: Path, ps_order: List[float] | None = None):
    sub = df[df['p_source'] == p_source].copy()
    if sub.empty:
        return
    # Determine x order
    if ps_order is None or not ps_order:
        xs = _ensure_sorted_unique(sub['p_target'].tolist())
    else:
        xs = [p for p in ps_order if p in set(sub['p_target'].tolist())]
    if not xs:
        return

    # Build series for makespan
    mk_tgt = []
    mk_unm = []
    mk_map = []
    api_tgt = []
    api_unm = []
    api_map = []
    for p in xs:
        r = sub[sub['p_target'] == p]
        if r.empty:
            mk_tgt.append(np.nan); mk_unm.append(np.nan); mk_map.append(np.nan)
            api_tgt.append(np.nan); api_unm.append(np.nan); api_map.append(np.nan)
            continue
        # If duplicates, take mean
        mk_tgt.append(float(r['tgt_mk'].mean()))
        mk_unm.append(float(r['src_mk_unmapped'].mean()))
        mk_map_val = float(r['src_mk_mapped'].mean()) if 'src_mk_mapped' in r and r['src_mk_mapped'].notna().any() else np.nan
        mk_map.append(mk_map_val)

        api_tgt.append(float(r['tgt_api'].mean()))
        api_unm.append(float(r['src_api_unmapped'].mean()))
        api_map_val = float(r['src_api_mapped'].mean()) if 'src_api_mapped' in r and r['src_api_mapped'].notna().any() else np.nan
        api_map.append(api_map_val)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    # Makespan
    ax = axes[0]
    ax.plot(xs, mk_tgt, label='Target baseline', color='#1f77b4', marker='o')
    ax.plot(xs, mk_unm, label='Original (unmapped)', color='#d62728', marker='o')
    ax.plot(xs, mk_map, label='Mapped', color='#2ca02c', marker='o')
    ax.set_xlabel('p_target')
    ax.set_ylabel('Average makespan')
    ax.set_title(f'Makespan vs p_target (p_source={p_source})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # API
    ax = axes[1]
    ax.plot(xs, api_tgt, label='Target baseline', color='#1f77b4', marker='o')
    ax.plot(xs, api_unm, label='Original (unmapped)', color='#d62728', marker='o')
    ax.plot(xs, api_map, label='Mapped', color='#2ca02c', marker='o')
    ax.set_xlabel('p_target')
    ax.set_ylabel('Active + Idle energy')
    ax.set_title(f'API vs p_target (p_source={p_source})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f'Performance across p_target for source p={p_source}', fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'perf_p{p_source:.1f}.png'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.pdf'))
    plt.close(fig)
    print(f'[plot] Saved {out_path} and PDF variant')


def main():
    ap = argparse.ArgumentParser(description='Plot mapped vs original performance for each source model p')
    ap.add_argument('--csv', type=Path, required=True, help='Path to ot_mapping_summary.csv')
    ap.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries/cross_p/plots'))
    ap.add_argument('--ps-order', type=str, default='', help='Comma-separated order for p_target, e.g., 0.8,0.7,0.5,0.3,0.1')
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f'CSV not found: {args.csv}')
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit('CSV is empty')

    ps_sources = _ensure_sorted_unique(df['p_source'].tolist())
    ps_order = None
    if args.ps_order:
        try:
            ps_order = [float(x) for x in args.ps_order.split(',') if x.strip()]
        except Exception:
            ps_order = None

    for p in ps_sources:
        try:
            plot_for_source(df, float(p), args.out_dir, ps_order)
        except Exception as e:
            print(f'[plot] Failed for p_source={p}: {e}')


if __name__ == '__main__':
    main()
