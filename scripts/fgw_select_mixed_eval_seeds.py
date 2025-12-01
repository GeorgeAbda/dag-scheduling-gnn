#!/usr/bin/env python3
"""
FGW-based mixed eval seed selection using per-task features and shortest-path structure.

- Builds each DAG using the project's dataset generator for given seeds.
- Node features: [task_length, req_memory_mb, req_cpu_cores] (z-scored across all nodes).
- Structural metric: all-pairs shortest-path distances on the undirected version of the DAG.
- Pairwise distances: POT fused Gromov–Wasserstein (FGW) with weight alpha.
- Selection: k-center greedy over the FGW distance matrix to pick K seeds across both domains.
- Visualization: 
  - MDS on FGW distances (2D), colored by domain and highlighting selected seeds.
  - PCA on graph-level features (from train_features.csv if available; else aggregate node features).

Usage:
  python -m scripts.fgw_select_mixed_eval_seeds \
    --wide-dir runs/datasets/wide/representativeness \
    --longcp-dir runs/datasets/longcp/representativeness \
    --k 10 \
    --alpha 0.5 \
    --out-dir runs/datasets/mixed/representativeness_ot \
    --max-per-domain 40 \
    [--wide-config data/rl_configs/train_wide_p005_seeds.json] \
    [--longcp-config data/rl_configs/train_long_cp_p08_seeds.json] \
    [--k-per-domain 5]

Requirements: numpy, pandas, scikit-learn, matplotlib, POT (ot)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

# Optional dependency: POT
try:
    import ot  # type: ignore
except Exception as e:
    ot = None

# Project imports (dataset generator)
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.dataset_generator.core.gen_dataset import generate_dataset


FEAT_NAMES = ["length", "req_memory_mb", "req_cpu_cores"]


def _read_seeds(json_path: Path) -> List[int]:
    with json_path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "train_seeds" in data and isinstance(data["train_seeds"], list):
            return [int(x) for x in data["train_seeds"]]
        if "selected_eval_seeds" in data and isinstance(data["selected_eval_seeds"], list):
            return [int(x) for x in data["selected_eval_seeds"]]
    return []


def _build_graph_from_seed(seed: int, dag_method: str, gnp_min_n: int, gnp_max_n: int, gnp_p: float | None,
                           host_count: int, vm_count: int, max_memory_gb: int,
                           min_cpu_speed_mips: int, max_cpu_speed_mips: int,
                           min_task_length: int, max_task_length: int,
                           workflow_count: int = 1,
                           task_length_dist: str = "uniform",
                           task_arrival: str = "static",
                           arrival_rate: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, C) for a single-workflow DAG built at 'seed'.
    X: (n, d) node features [length, req_memory_mb, req_cpu_cores]
    C: (n, n) undirected shortest-path distances (float), scaled to [0,1].
    """
    ds = generate_dataset(
        seed=seed,
        host_count=host_count,
        vm_count=vm_count,
        max_memory_gb=max_memory_gb,
        min_cpu_speed_mips=min_cpu_speed_mips,
        max_cpu_speed_mips=max_cpu_speed_mips,
        workflow_count=workflow_count,
        dag_method=dag_method,
        gnp_min_n=gnp_min_n,
        gnp_max_n=gnp_max_n,
        gnp_p=gnp_p,
        task_length_dist=task_length_dist,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        task_arrival=task_arrival,
        arrival_rate=arrival_rate,
        vm_rng_seed=0,
    )
    wf = ds.workflows[0]
    # Node features
    n = len(wf.tasks)
    X = np.zeros((n, 3), dtype=np.float64)
    adj: List[List[int]] = [[] for _ in range(n)]
    for t in wf.tasks:
        X[t.id, 0] = float(t.length)
        X[t.id, 1] = float(t.req_memory_mb)
        X[t.id, 2] = float(t.req_cpu_cores)
        for v in t.child_ids:
            adj[t.id].append(int(v))
            if 0 <= v < n:
                # undirected adjacency for path length
                adj[v].append(int(t.id))
    # All-pairs shortest-path (BFS per node)
    INF = 10**9
    C = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        C[i, i] = 0.0
        # BFS
        q: List[int] = [i]
        dist = {i: 0}
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        for j, d in dist.items():
            C[i, j] = float(d)
    # Symmetrize minimal distances and scale
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


def _zscore_features(list_X: List[np.ndarray]) -> List[np.ndarray]:
    if not list_X:
        return []
    # Stack all nodes to compute global z-score for comparability
    all_nodes = np.concatenate(list_X, axis=0)
    mu = all_nodes.mean(axis=0, keepdims=True)
    sd = all_nodes.std(axis=0, keepdims=True) + 1e-8
    return [ (X - mu) / sd for X in list_X ]


def _fgw_distance(C1: np.ndarray, X1: np.ndarray, C2: np.ndarray, X2: np.ndarray, alpha: float = 0.5) -> float:
    assert ot is not None, "POT (ot) is required. pip install POT"
    n1, n2 = C1.shape[0], C2.shape[0]
    a = np.ones(n1, dtype=np.float64) / max(1, n1)
    b = np.ones(n2, dtype=np.float64) / max(1, n2)
    # Feature cost matrix (squared Euclidean)
    # Efficient computation: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = (X1 * X1).sum(axis=1, keepdims=True)
    y2 = (X2 * X2).sum(axis=1, keepdims=True)
    M = x2 + y2.T - 2.0 * (X1 @ X2.T)
    M = np.maximum(M, 0.0)
    # FGW squared objective value
    try:
        fgw2 = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, a, b, alpha=alpha)
        return float(fgw2)
    except Exception:
        # Fallback: try without alpha kw (older POT)
        fgw2 = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, a, b, alpha)
        return float(fgw2)


def _kcenter(D: np.ndarray, k: int) -> List[int]:
    n = D.shape[0]
    if k >= n:
        return list(range(n))
    # Start with the point with largest average distance
    avg = np.mean(D, axis=1)
    centers = [int(np.argmax(avg))]
    min_d = D[centers[0]].copy()
    for _ in range(1, k):
        nxt = int(np.argmax(min_d))
        centers.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
    centers = sorted(set(centers))
    # If duplicates removed, add farthest remaining until k
    while len(centers) < k:
        mask = np.ones(n, dtype=bool)
        mask[centers] = False
        cand = int(np.argmax(np.min(D[np.ix_(mask, centers)], axis=1)))
        # Map back to original index among masked
        idxs = np.where(mask)[0]
        centers.append(int(idxs[cand]))
    return centers


def _load_graph_features_csv(domain_dir: Path) -> pd.DataFrame | None:
    tf = domain_dir / "train_features.csv"
    if not tf.exists():
        return None
    try:
        df = pd.read_csv(tf)
        return df
    except Exception:
        return None


def _estimate_p_from_edges(n: int, edges: float) -> float:
    """Estimate gnp p for generate_dag_gnp to match target edge count approximately.
    Model: expected edges = p * ((n-1)(n-2)/2) + s(p), where s(p) = sum_{j=1}^{n-1} (1-p)^{j-1} = (1-(1-p)^{n-1})/p.
    Solve via binary search on p in (1e-5, 0.999).
    """
    n = max(2, int(n))
    target = float(max(0.0, edges))
    pair_count = float((n - 1) * (n - 2) / 2.0)
    def exp_edges(p: float) -> float:
        if p <= 0:
            return 0.0
        s = (1.0 - (1.0 - p) ** (n - 1)) / p
        return p * pair_count + s
    lo, hi = 1e-5, 0.999
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        val = exp_edges(mid)
        if val < target:
            lo = mid
        else:
            hi = mid
    p_hat = 0.5 * (lo + hi)
    return float(min(0.999, max(1e-4, p_hat)))


def _load_rl_config(path: Path) -> tuple[list[int], dict]:
    """Load RL config JSON and return (train_seeds, dataset_kwargs).
    The JSON is expected to have keys: { "train": { "seeds": [...], "dataset": { ... } } }.
    We map fields to generate_dataset signature; fill CPU speed defaults if missing.
    """
    with path.open("r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or "train" not in cfg:
        raise ValueError(f"Invalid RL config schema: {path}")
    train = cfg["train"]
    seeds = [int(x) for x in train.get("seeds", [])]
    ds = dict(train.get("dataset", {}))
    # Map keys and defaults
    ds_kwargs = {
        "dag_method": ds.get("dag_method", "gnp"),
        "gnp_min_n": int(ds.get("gnp_min_n", 12)),
        "gnp_max_n": int(ds.get("gnp_max_n", 30)),
        "gnp_p": float(ds.get("gnp_p", 0.1)) if ds.get("gnp_p", None) is not None else None,
        "host_count": int(ds.get("host_count", 4)),
        "vm_count": int(ds.get("vm_count", 10)),
        "max_memory_gb": int(ds.get("max_memory_gb", 10)),
        # CPU speeds may be absent; use sane defaults from prior code
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-dir", type=Path, default=Path("runs/datasets/wide/representativeness"))
    ap.add_argument("--longcp-dir", type=Path, default=Path("runs/datasets/longcp/representativeness"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k-per-domain", type=int, default=0, help="If >0, select k seeds per domain (wide and longcp) using domain-wise k-center; overrides --k for per-domain selection")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/datasets/mixed/representativeness_ot"))
    ap.add_argument("--max-per-domain", type=int, default=40)
    ap.add_argument("--wide-config", type=Path, default=None)
    ap.add_argument("--longcp-config", type=Path, default=None)
    ap.add_argument("--host-count", type=int, default=4)
    ap.add_argument("--vm-count", type=int, default=10)
    ap.add_argument("--max-memory-gb", type=int, default=10)
    ap.add_argument("--min-cpu-speed", type=int, default=500)
    ap.add_argument("--max-cpu-speed", type=int, default=5000)
    ap.add_argument("--min-task-length", type=int, default=500)
    ap.add_argument("--max-task-length", type=int, default=100_000)
    ap.add_argument("--skip-fgw", action="store_true", default=False)
    args = ap.parse_args()

    if (ot is None) and (not args.skip_fgw):
        raise SystemExit("POT (ot) not installed. Please: pip install POT or pass --skip-fgw to bypass FGW")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Domain seeds and dataset kwargs: prefer RL configs if provided
    wide_ds_kwargs = None
    long_ds_kwargs = None
    if args.wide_config is not None and Path(args.wide_config).exists():
        wide_cfg_seeds, wide_ds_kwargs = _load_rl_config(Path(args.wide_config))
        wide_seeds = wide_cfg_seeds
    else:
        wide_json = args.wide_dir / "selected_eval_seeds.json"
        if not wide_json.exists():
            raise SystemExit("Missing wide selected_eval_seeds.json and no --wide-config provided")
        wide_seeds = _read_seeds(wide_json)

    if args.longcp_config is not None and Path(args.longcp_config).exists():
        long_cfg_seeds, long_ds_kwargs = _load_rl_config(Path(args.longcp_config))
        long_seeds = long_cfg_seeds
    else:
        long_json = args.longcp_dir / "selected_eval_seeds.json"
        if not long_json.exists():
            raise SystemExit("Missing longcp selected_eval_seeds.json and no --longcp-config provided")
        long_seeds = _read_seeds(long_json)

    if len(wide_seeds) == 0 or len(long_seeds) == 0:
        raise SystemExit("Empty seed lists from inputs; provide RL configs or seed JSONs with train_seeds/selected_eval_seeds")

    # Subsample for computational tractability
    def _take(lst: List[int], m: int) -> List[int]:
        m = min(m, len(lst))
        # deterministic: pick evenly spaced
        idxs = np.linspace(0, len(lst)-1, num=m, dtype=int)
        return [int(lst[i]) for i in idxs]

    wide_seeds_sub = _take(wide_seeds, int(args.max_per_domain))
    long_seeds_sub = _take(long_seeds, int(args.max_per_domain))

    # Load per-domain train_features to infer n and edges only if RL configs are absent
    feat_map_wide: Dict[int, Tuple[int, float]] = {}
    feat_map_long: Dict[int, Tuple[int, float]] = {}
    if wide_ds_kwargs is None or long_ds_kwargs is None:
        df_wide = _load_graph_features_csv(Path(args.wide_dir))
        df_long = _load_graph_features_csv(Path(args.longcp_dir))
        if df_wide is None or df_long is None:
            print("[warn] Missing train_features.csv; falling back to default n/p ranges (may not match sampled data)")
        if df_wide is not None and {'seed','tasks','edges'}.issubset(df_wide.columns):
            for _, r in df_wide.iterrows():
                try:
                    feat_map_wide[int(r['seed'])] = (int(r['tasks']), float(r['edges']))
                except Exception:
                    pass
        if df_long is not None and {'seed','tasks','edges'}.issubset(df_long.columns):
            for _, r in df_long.iterrows():
                try:
                    feat_map_long[int(r['seed'])] = (int(r['tasks']), float(r['edges']))
                except Exception:
                    pass

    # Build graphs -> (X, C)
    graph_X: List[np.ndarray] = []
    graph_C: List[np.ndarray] = []
    graph_seed: List[int] = []
    graph_domain: List[str] = []

    for s in wide_seeds_sub:
        if wide_ds_kwargs is not None:
            # Use exact domain generator settings from RL config
            ds = wide_ds_kwargs
            X, C = _build_graph_from_seed(
                seed=s,
                dag_method=str(ds.get("dag_method", "gnp")),
                gnp_min_n=int(ds.get("gnp_min_n", 12)),
                gnp_max_n=int(ds.get("gnp_max_n", 30)),
                gnp_p=ds.get("gnp_p", None),
                host_count=int(ds.get("host_count", args.host_count)),
                vm_count=int(ds.get("vm_count", args.vm_count)),
                max_memory_gb=int(ds.get("max_memory_gb", args.max_memory_gb)),
                min_cpu_speed_mips=int(ds.get("min_cpu_speed_mips", args.min_cpu_speed)),
                max_cpu_speed_mips=int(ds.get("max_cpu_speed_mips", args.max_cpu_speed)),
                min_task_length=int(ds.get("min_task_length", args.min_task_length)),
                max_task_length=int(ds.get("max_task_length", args.max_task_length)),
                workflow_count=int(ds.get("workflow_count", 1)),
                task_length_dist=str(ds.get("task_length_dist", "uniform")),
                task_arrival=str(ds.get("task_arrival", "static")),
                arrival_rate=float(ds.get("arrival_rate", 0.0)),
            )
        else:
            if s in feat_map_wide:
                n_i, e_i = feat_map_wide[s]
                p_i = _estimate_p_from_edges(n_i, e_i)
                n_lo = n_hi = int(max(2, n_i))
                p_use = float(p_i)
            else:
                # Fallback: minimal assumptions
                n_lo = 12; n_hi = 30; p_use = 0.25
            X, C = _build_graph_from_seed(
                seed=s,
                dag_method="gnp",
                gnp_min_n=n_lo,
                gnp_max_n=n_hi,
                gnp_p=p_use,
                host_count=args.host_count,
                vm_count=args.vm_count,
                max_memory_gb=args.max_memory_gb,
                min_cpu_speed_mips=args.min_cpu_speed,
                max_cpu_speed_mips=args.max_cpu_speed,
                min_task_length=args.min_task_length,
                max_task_length=args.max_task_length,
            )
        graph_X.append(X)
        graph_C.append(C)
        graph_seed.append(int(s))
        graph_domain.append("wide")

    for s in long_seeds_sub:
        if long_ds_kwargs is not None:
            ds = long_ds_kwargs
            X, C = _build_graph_from_seed(
                seed=s,
                dag_method=str(ds.get("dag_method", "gnp")),
                gnp_min_n=int(ds.get("gnp_min_n", 12)),
                gnp_max_n=int(ds.get("gnp_max_n", 30)),
                gnp_p=ds.get("gnp_p", None),
                host_count=int(ds.get("host_count", args.host_count)),
                vm_count=int(ds.get("vm_count", args.vm_count)),
                max_memory_gb=int(ds.get("max_memory_gb", args.max_memory_gb)),
                min_cpu_speed_mips=int(ds.get("min_cpu_speed_mips", args.min_cpu_speed)),
                max_cpu_speed_mips=int(ds.get("max_cpu_speed_mips", args.max_cpu_speed)),
                min_task_length=int(ds.get("min_task_length", args.min_task_length)),
                max_task_length=int(ds.get("max_task_length", args.max_task_length)),
                workflow_count=int(ds.get("workflow_count", 1)),
                task_length_dist=str(ds.get("task_length_dist", "uniform")),
                task_arrival=str(ds.get("task_arrival", "static")),
                arrival_rate=float(ds.get("arrival_rate", 0.0)),
            )
        else:
            if s in feat_map_long:
                n_i, e_i = feat_map_long[s]
                p_i = _estimate_p_from_edges(n_i, e_i)
                n_lo = n_hi = int(max(2, n_i))
                p_use = float(p_i)
            else:
                n_lo = 12; n_hi = 30; p_use = 0.06
            X, C = _build_graph_from_seed(
                seed=s,
                dag_method="gnp",
                gnp_min_n=n_lo,
                gnp_max_n=n_hi,
                gnp_p=p_use,
                host_count=args.host_count,
                vm_count=args.vm_count,
                max_memory_gb=args.max_memory_gb,
                min_cpu_speed_mips=args.min_cpu_speed,
                max_cpu_speed_mips=args.max_cpu_speed,
                min_task_length=args.min_task_length,
                max_task_length=args.max_task_length,
            )
        graph_X.append(X)
        graph_C.append(C)
        graph_seed.append(int(s))
        graph_domain.append("longcp")

    # Z-score node features across all nodes
    graph_X_z = _zscore_features(graph_X)

    # Optional FGW stage
    m = len(graph_X_z)
    selected: list[int] = []
    if not args.skip_fgw:
        D = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            D[i, i] = 0.0
        for i in range(m):
            for j in range(i+1, m):
                fgw2 = _fgw_distance(graph_C[i], graph_X_z[i], graph_C[j], graph_X_z[j], alpha=float(args.alpha))
                d = float(np.sqrt(max(0.0, fgw2)))
                D[i, j] = d
                D[j, i] = d

        # k-center selection
        kpd = int(max(0, args.k_per_domain))
        if kpd > 0:
            idx_wide = [i for i, d in enumerate(graph_domain) if d == "wide"]
            idx_long = [i for i, d in enumerate(graph_domain) if d == "longcp"]
            if len(idx_wide) < kpd or len(idx_long) < kpd:
                print(f"[warn] Not enough candidates per domain for k-per-domain={kpd}; falling back to global k-center")
                centers_idx = _kcenter(D, int(args.k))
            else:
                Dw = D[np.ix_(idx_wide, idx_wide)]
                Dl = D[np.ix_(idx_long, idx_long)]
                cw_local = _kcenter(Dw, kpd)
                cl_local = _kcenter(Dl, kpd)
                centers_idx = [idx_wide[i] for i in cw_local] + [idx_long[i] for i in cl_local]
        else:
            centers_idx = _kcenter(D, int(args.k))
        selected = [int(graph_seed[i]) for i in centers_idx]

    # Save JSON
    if not args.skip_fgw:
        selected_idx = centers_idx
    else:
        selected_idx = []
    out_json = {
        "selected_eval_seeds": selected,
        "by_domain": {
            "wide": [int(graph_seed[i]) for i in selected_idx if graph_domain[i] == "wide"],
            "longcp": [int(graph_seed[i]) for i in selected_idx if graph_domain[i] == "longcp"],
        },
        "fgw_alpha": float(args.alpha),
        "notes": "FGW (alpha) on per-task z-scored features; C=shortest-path undirected scaled; k-center on FGW distances",
    }
    with (args.out_dir / "fgw_mixed_selected_eval_seeds.json").open("w") as f:
        json.dump(out_json, f, indent=2)

    # Visualization: FGW-MDS
    if not args.skip_fgw:
        try:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
            Z = mds.fit_transform(D)
            plt.figure(figsize=(8, 6))
            colors = {"wide": "tab:orange", "longcp": "tab:blue"}
            for dom in ["wide", "longcp"]:
                idx = [i for i, ddom in enumerate(graph_domain) if ddom == dom]
                plt.scatter(Z[idx, 0], Z[idx, 1], s=35, alpha=0.6, c=colors[dom], label=dom)
            plt.scatter(Z[centers_idx, 0], Z[centers_idx, 1], s=160, facecolors='none', edgecolors='black', linewidths=2, label='selected')
            plt.title("FGW-MDS (2D) of candidate DAGs")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.out_dir / "fgw_mds_selected.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[warn] MDS plotting failed: {e}")

    # Visualization: PCA on graph-level features
    # Try to read domain train_features.csv; else aggregate node features
    def _graph_agg(i: int) -> Dict[str, float]:
        X = graph_X[i]
        return {
            'tasks': float(X.shape[0]),
            'len_mean': float(X[:, 0].mean()),
            'mem_mean': float(X[:, 1].mean()),
            'cpu_mean': float(X[:, 2].mean()),
        }

    df_list: List[pd.DataFrame] = []
    for dom, ddir in [("wide", args.wide_dir), ("longcp", args.longcp_dir)]:
        df = _load_graph_features_csv(Path(ddir))
        if df is not None and 'seed' in df.columns:
            keep = df[df['seed'].isin([s for s, d in zip(graph_seed, graph_domain) if d == dom])].copy()
            keep['domain'] = dom
            df_list.append(keep)
    if not df_list:
        # fallback to aggregates
        rows = []
        for i in range(m):
            r = _graph_agg(i)
            r['seed'] = graph_seed[i]
            r['domain'] = graph_domain[i]
            rows.append(r)
        df_all = pd.DataFrame(rows)
        feat_cols = [c for c in df_all.columns if c not in ('seed','domain')]
    else:
        df_all = pd.concat(df_list, ignore_index=True)
        # Use common original feature names if present
        feat_cols = [c for c in ["tasks","edges","width_peak","depth_avg","Pbar","burstiness","cp_frac"] if c in df_all.columns]
        if not feat_cols:
            feat_cols = [c for c in df_all.columns if c not in ('seed','domain')]

    try:
        Xp = df_all[feat_cols].to_numpy(dtype=float)
        Xp = (Xp - Xp.mean(0, keepdims=True)) / (Xp.std(0, keepdims=True) + 1e-9)
        pca = PCA(n_components=2, random_state=0)
        Zp = pca.fit_transform(Xp)
        plt.figure(figsize=(8, 6))
        for dom in ["wide", "longcp"]:
            idx = df_all['domain'] == dom
            plt.scatter(Zp[idx, 0], Zp[idx, 1], s=35, alpha=0.6, label=dom)
        sel_mask = df_all['seed'].isin(selected)
        plt.scatter(Zp[sel_mask, 0], Zp[sel_mask, 1], s=160, facecolors='none', edgecolors='black', linewidths=2, label='selected')
        plt.title("PCA of graph-level features (selected highlighted)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "pca_graph_features_selected.png", dpi=180)
        plt.close()
        loadings_df = pd.DataFrame(pca.components_.T, index=feat_cols, columns=["PC1", "PC2"]) 
        evr = pca.explained_variance_ratio_
        loadings_df.to_csv(args.out_dir / "pca_loadings.csv")
        pd.Series(evr, index=["PC1", "PC2"]).to_csv(args.out_dir / "pca_explained_variance_ratio.csv")
        try:
            plt.figure(figsize=(8, 6))
            for dom in ["wide", "longcp"]:
                idx = df_all['domain'] == dom
                plt.scatter(Zp[idx, 0], Zp[idx, 1], s=25, alpha=0.5, label=dom)
            arrow_scale = float(np.max(np.abs(Zp)) * 0.5) if Zp.size > 0 else 1.0
            for j, f in enumerate(feat_cols):
                x = float(loadings_df.iloc[j, 0]) * arrow_scale
                y = float(loadings_df.iloc[j, 1]) * arrow_scale
                plt.arrow(0.0, 0.0, x, y, color='k', width=0.001, head_width=0.05, length_includes_head=True)
                plt.text(x * 1.05, y * 1.05, str(f), fontsize=9)
            plt.axhline(0.0, color='gray', lw=0.5)
            plt.axvline(0.0, color='gray', lw=0.5)
            plt.title("PCA biplot (PC1 vs PC2)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.out_dir / "pca_biplot.png", dpi=180)
            plt.close()
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] PCA plotting failed: {e}")

    print("[done] Wrote:")
    print(" -", args.out_dir / "fgw_mixed_selected_eval_seeds.json")
    if not args.skip_fgw:
        print(" -", args.out_dir / "fgw_mds_selected.png")
    print(" -", args.out_dir / "pca_graph_features_selected.png")
    try:
        print(" -", args.out_dir / "pca_loadings.csv")
        print(" -", args.out_dir / "pca_explained_variance_ratio.csv")
        print(" -", args.out_dir / "pca_biplot.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
