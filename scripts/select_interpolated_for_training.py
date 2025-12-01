#!/usr/bin/env python3
"""
Select a subset of interpolated DAGs to use in training.

Strategy:
- Build anchors from each domain using RL configs (sample a limited number per domain).
- For each interpolated workflow, compute FGW distance to anchors in wide and longcp:
    d_w_min = min FGW(interp, wide_anchor)
    d_l_min = min FGW(interp, long_anchor)
  Compute a balance score b = |d_w_min - d_l_min| (smaller is more "between" domains),
  and centrality c = d_w_min + d_l_min (smaller is more central to both).
- Rank by (b ascending, then c ascending).
- Ensure diversity by k-center over the interpolated set using pairwise FGW distances.
- Output: indices and a dataset JSON with only the selected workflows.

Usage:
  python -m scripts.select_interpolated_for_training \
    --wide-config data/rl_configs/train_wide_p005_seeds.json \
    --longcp-config data/rl_configs/train_long_cp_p08_seeds.json \
    --interp-json runs/datasets/mixed/representativeness_ot/datasets/dataset_interp_all.json \
    --alpha 0.5 \
    --k 6 \
    --anchors-per-domain 20 \
    --out-dir runs/datasets/mixed/representativeness_ot/datasets

Outputs:
- selected_interpolated_indices.json (indices within interp dataset)
- dataset_interp_selected.json (dataset with only selected workflows)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

try:
    import ot  # type: ignore
except Exception:
    ot = None

import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.dataset_generator.core.gen_dataset import generate_dataset


def _load_rl_config(path: Path) -> tuple[list[int], dict]:
    with path.open("r") as f:
        cfg = json.load(f)
    train = cfg.get("train", {})
    seeds = [int(x) for x in train.get("seeds", [])]
    ds = dict(train.get("dataset", {}))
    ds_kwargs = {
        "dag_method": ds.get("dag_method", "gnp"),
        "gnp_min_n": int(ds.get("gnp_min_n", 12)),
        "gnp_max_n": int(ds.get("gnp_max_n", 30)),
        "gnp_p": float(ds.get("gnp_p", 0.1)) if ds.get("gnp_p", None) is not None else None,
        "host_count": int(ds.get("host_count", 4)),
        "vm_count": int(ds.get("vm_count", 10)),
        "max_memory_gb": int(ds.get("max_memory_gb", 10)),
        "min_cpu_speed_mips": int(ds.get("min_cpu_speed", 500)),
        "max_cpu_speed_mips": int(ds.get("max_cpu_speed", 5000)),
        "workflow_count": int(ds.get("workflow_count", 1)),
        "task_length_dist": ds.get("task_length_dist", "uniform"),
        "min_task_length": int(ds.get("min_task_length", 500)),
        "max_task_length": int(ds.get("max_task_length", 100_000)),
        "task_arrival": ds.get("task_arrival", "static"),
        "arrival_rate": float(ds.get("arrival_rate", 0.0)),
    }
    return seeds, ds_kwargs


def _build_graph_from_seed(seed: int, ds_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
    ds = generate_dataset(
        seed=seed,
        host_count=ds_kwargs["host_count"],
        vm_count=ds_kwargs["vm_count"],
        max_memory_gb=ds_kwargs["max_memory_gb"],
        min_cpu_speed_mips=ds_kwargs["min_cpu_speed_mips"],
        max_cpu_speed_mips=ds_kwargs["max_cpu_speed_mips"],
        workflow_count=ds_kwargs.get("workflow_count", 1),
        dag_method=ds_kwargs.get("dag_method", "gnp"),
        gnp_min_n=ds_kwargs.get("gnp_min_n", 12),
        gnp_max_n=ds_kwargs.get("gnp_max_n", 30),
        gnp_p=ds_kwargs.get("gnp_p", None),
        task_length_dist=ds_kwargs.get("task_length_dist", "uniform"),
        min_task_length=ds_kwargs.get("min_task_length", 500),
        max_task_length=ds_kwargs.get("max_task_length", 100_000),
        task_arrival=ds_kwargs.get("task_arrival", "static"),
        arrival_rate=ds_kwargs.get("arrival_rate", 0.0),
        vm_rng_seed=0,
    )
    wf = ds.workflows[0]
    n = len(wf.tasks)
    X = np.zeros((n, 3), dtype=np.float64)
    adj = [[] for _ in range(n)]
    for t in wf.tasks:
        X[t.id, 0] = float(t.length)
        X[t.id, 1] = float(t.req_memory_mb)
        X[t.id, 2] = float(t.req_cpu_cores)
        for v in t.child_ids:
            adj[t.id].append(int(v))
    # Undirected shortest-path distances
    INF = 10**9
    C = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        C[i, i] = 0.0
        q = [i]; dist = {i: 0}; head = 0
        while head < len(q):
            u = q[head]; head += 1
            neighbors = set(adj[u]) | {p for p in range(n) if u in adj[p]}
            for w in neighbors:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        for j, d in dist.items():
            C[i, j] = float(d)
    C = np.minimum(C, C.T)
    finite = C[C < INF]
    if finite.size > 0:
        mx = finite.max()
        if mx > 0:
            C[C >= INF] = mx * 2.0
            C = C / (C.max() + 1e-12)
        else:
            C[C >= INF] = 1.0
    else:
        C[:, :] = 0.0
    return X, C


def _build_graph_from_workflow(wf: Dict) -> Tuple[np.ndarray, np.ndarray]:
    tasks = wf.get("tasks", [])
    n = len(tasks)
    X = np.zeros((n, 3), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    for t in tasks:
        i = int(t["id"])
        X[i, 0] = float(t["length"]) if "length" in t else 0.0
        X[i, 1] = float(t.get("req_memory_mb", 1024))
        X[i, 2] = float(t.get("req_cpu_cores", 1))
        for j in t.get("child_ids", []):
            A[i, int(j)] = 1.0
    # Undirected shortest-path
    INF = 10**9
    C = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        C[i, i] = 0.0
        q = [i]; dist = {i: 0}; head = 0
        while head < len(q):
            u = q[head]; head += 1
            for w in np.where(A[u] > 0)[0].tolist() + np.where(A[:, u] > 0)[0].tolist():
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(int(w))
        for j, d in dist.items():
            C[i, j] = float(d)
    C = np.minimum(C, C.T)
    finite = C[C < INF]
    if finite.size > 0:
        mx = finite.max()
        if mx > 0:
            C[C >= INF] = mx * 2.0
            C = C / (C.max() + 1e-12)
        else:
            C[C >= INF] = 1.0
    else:
        C[:, :] = 0.0
    return X, C


def _fgw_distance(X1: np.ndarray, C1: np.ndarray, X2: np.ndarray, C2: np.ndarray, alpha: float) -> float:
    # feature cost
    x2 = (X1 * X1).sum(axis=1, keepdims=True)
    y2 = (X2 * X2).sum(axis=1, keepdims=True)
    M = x2 + y2.T - 2.0 * (X1 @ X2.T)
    M = np.maximum(M, 0.0)
    n1, n2 = X1.shape[0], X2.shape[0]
    a = np.ones(n1) / max(1, n1)
    b = np.ones(n2) / max(1, n2)
    try:
        fgw2 = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, a, b, alpha=alpha)
    except Exception:
        fgw2 = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, a, b, alpha)
    return float(np.sqrt(max(0.0, fgw2)))


def _kcenter(D: np.ndarray, k: int) -> List[int]:
    n = D.shape[0]
    if k >= n:
        return list(range(n))
    avg = np.mean(D, axis=1)
    centers = [int(np.argmax(avg))]
    min_d = D[centers[0]].copy()
    for _ in range(1, k):
        nxt = int(np.argmax(min_d))
        centers.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
    centers = sorted(set(centers))
    while len(centers) < k:
        mask = np.ones(n, dtype=bool); mask[centers] = False
        # pick farthest from current centers
        d_to_centers = np.min(D[np.ix_(mask, centers)], axis=1)
        idxs = np.where(mask)[0]
        centers.append(int(idxs[int(np.argmax(d_to_centers))]))
    return centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-config", type=Path, required=True)
    ap.add_argument("--longcp-config", type=Path, required=True)
    ap.add_argument("--interp-json", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--anchors-per-domain", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if ot is None:
        raise SystemExit("POT (ot) is required. pip install POT")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds_w, ds_w = _load_rl_config(args.wide_config)
    seeds_l, ds_l = _load_rl_config(args.longcp_config)

    # Build limited anchor sets for speed
    Aw: List[Tuple[np.ndarray, np.ndarray]] = []
    Al: List[Tuple[np.ndarray, np.ndarray]] = []
    for sw in seeds_w[: max(1, args.anchors_per_domain)]:
        X, C = _build_graph_from_seed(int(sw), ds_w)
        Aw.append((X, C))
    for sl in seeds_l[: max(1, args.anchors_per_domain)]:
        X, C = _build_graph_from_seed(int(sl), ds_l)
        Al.append((X, C))

    # Load interpolated workflows
    with args.interp_json.open("r") as f:
        data = json.load(f)
    workflows: List[Dict] = list(data.get("workflows", []))

    # Precompute interpolated representations
    interps: List[Tuple[np.ndarray, np.ndarray]] = []
    for wf in workflows:
        Xi, Ci = _build_graph_from_workflow(wf)
        interps.append((Xi, Ci))

    # Compute per-interp balance and centrality
    scores: List[Tuple[int, float, float, float]] = []  # (idx, b, dsum, dmin)
    for idx, (Xi, Ci) in enumerate(interps):
        d_w = min(_fgw_distance(Xi, Ci, Xw, Cw, args.alpha) for (Xw, Cw) in Aw)
        d_l = min(_fgw_distance(Xi, Ci, Xl, Cl, args.alpha) for (Xl, Cl) in Al)
        b = abs(d_w - d_l)
        dsum = d_w + d_l
        dmin = min(d_w, d_l)
        scores.append((idx, b, dsum, dmin))

    # Rank by balance then centrality
    scores.sort(key=lambda t: (t[1], t[2], t[3]))
    ranked_indices = [idx for (idx, _b, _ds, _dm) in scores]

    # Diversity via k-center on pairwise FGW among interps (small set so OK)
    m = len(interps)
    D = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            d = _fgw_distance(interps[i][0], interps[i][1], interps[j][0], interps[j][1], args.alpha)
            D[i, j] = D[j, i] = d

    # Run k-center on ranked order by restricting candidates progressively
    k = int(max(1, args.k))
    # Prefer top-N ranked to bias towards balanced/central
    topN = min(m, max(2 * k, k + 4))
    cand = ranked_indices[:topN]
    Dc = D[np.ix_(cand, cand)]
    loc = _kcenter(Dc, k)
    selected = [cand[i] for i in loc]

    # Write outputs
    with (args.out_dir / "selected_interpolated_indices.json").open("w") as f:
        json.dump({"indices": selected}, f, indent=2)

    ds_sel = {"workflows": [workflows[i] for i in selected]}
    with (args.out_dir / "dataset_interp_selected.json").open("w") as f:
        json.dump(ds_sel, f)

    print("[select] Done.")
    print(" - selected indices:", selected)
    print(" - wrote:", args.out_dir / "selected_interpolated_indices.json")
    print(" - wrote:", args.out_dir / "dataset_interp_selected.json")


if __name__ == "__main__":
    main()
