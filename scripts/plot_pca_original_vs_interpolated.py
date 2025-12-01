#!/usr/bin/env python3
"""
PCA comparison of original distributions (wide, longcp) vs interpolated DAGs.

- Loads graph-level features for original domains from train_features.csv
  (expected columns: tasks, edges, width_peak, depth_avg, seed)
- Loads interpolated workflows JSON (dataset_interp_all.json) and computes
  the same graph-level features from adjacency:
    - tasks: number of nodes
    - edges: number of directed edges
    - width_peak: max nodes per BFS level from root (0)
    - depth_avg: average BFS level
- Concatenates the three groups and runs PCA on standardized features.
- Saves a scatter plot colored by domain: wide, longcp, interpolated.

Usage:
  python -m scripts.plot_pca_original_vs_interpolated \
    --wide-dir runs/datasets/wide/representativeness \
    --longcp-dir runs/datasets/longcp/representativeness \
    --interp-json runs/datasets/mixed/representativeness_ot/datasets/dataset_interp_all.json \
    --out runs/datasets/mixed/representativeness_ot/pca_original_vs_interp.png

Requirements: numpy, pandas, scikit-learn, matplotlib, seaborn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


FEATS = ["tasks", "edges", "width_peak", "depth_avg"]


def _load_domain_features(base_dir: Path, domain: str) -> pd.DataFrame:
    tf = base_dir / "train_features.csv"
    if not tf.exists():
        raise SystemExit(f"Missing {tf}")
    df = pd.read_csv(tf)
    missing = [c for c in FEATS + ["seed"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{tf} missing columns: {missing}")
    df = df[FEATS + ["seed"]].copy()
    df["domain"] = domain
    return df


def _compute_interp_features(interp_json: Path) -> pd.DataFrame:
    with interp_json.open("r") as f:
        data = json.load(f)
    wfs: List[Dict] = data.get("workflows", [])
    rows: List[Dict[str, float | int | str]] = []
    for k, wf in enumerate(wfs):
        tasks = wf.get("tasks", [])
        n = len(tasks)
        if n == 0:
            continue
        # Build adjacency
        A = np.zeros((n, n), dtype=np.int32)
        for t in tasks:
            i = int(t["id"])
            for j in t.get("child_ids", []):
                A[i, int(j)] = 1
        # tasks, edges
        num_edges = int(A.sum())
        # BFS levels from node 0
        level = np.full(n, -1, dtype=int)
        q = [0]; level[0] = 0
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            for v in np.where(A[u] > 0)[0].tolist():
                if level[v] == -1:
                    level[v] = level[u] + 1
                    q.append(int(v))
        # If some nodes are unreachable from 0, assign minimal level via topo order
        if np.any(level < 0):
            # simple topo: since edges i<j in our generated DAGs, assign level[j] = max(level[parents])+1 or 0
            for j in range(n):
                if level[j] >= 0:
                    continue
                parents = np.where(A[:, j] > 0)[0]
                if parents.size:
                    level[j] = int(level[parents].max() + 1)
                else:
                    level[j] = 0
        # width_peak and depth_avg
        unique, counts = np.unique(level, return_counts=True)
        width_peak = int(counts.max() if counts.size else 1)
        depth_avg = float(level.mean())
        rows.append({
            "tasks": int(n),
            "edges": int(num_edges),
            "width_peak": int(width_peak),
            "depth_avg": float(depth_avg),
            "seed": f"interp_{k}",
            "domain": "interpolated",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-dir", type=Path, required=True)
    ap.add_argument("--longcp-dir", type=Path, required=True)
    ap.add_argument("--interp-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df_wide = _load_domain_features(args.wide_dir, "wide")
    df_long = _load_domain_features(args.longcp_dir, "longcp")
    df_interp = _compute_interp_features(args.interp_json)

    # Concatenate and standardize
    df_all = pd.concat([df_wide, df_long, df_interp], ignore_index=True)
    X = df_all[FEATS].to_numpy(dtype=float)
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)
    df_all["pc1"] = Z[:, 0]
    df_all["pc2"] = Z[:, 1]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_all, x="pc1", y="pc2", hue="domain", alpha=0.7, s=40)
    plt.title("PCA: original (wide, longcp) vs interpolated")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    plt.close()

    print("[done] Wrote:")
    print(" -", args.out)


if __name__ == "__main__":
    main()
