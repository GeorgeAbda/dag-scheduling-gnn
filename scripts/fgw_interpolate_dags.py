#!/usr/bin/env python3
"""
Generate interpolated (synthetic) DAGs between two domains using FGW couplings.

High-level procedure per (seed_w, seed_l, lambda):
- Build DAG1 and DAG2 deterministically from RL config DatasetArgs + seeds
- Extract node features X1/X2 = [length, req_memory_mb, req_cpu_cores]
- Build undirected shortest-path distance matrices C1/C2 (scaled to [0,1])
- Compute FGW coupling T between (C1,X1) and (C2,X2)
- Choose a base graph (the one whose node count is closest to n_bar = round((1-λ)*n1 + λ*n2))
- Barycentric node features on base index i:
    X_bar[i] = (1-λ)·X_base[i] + λ·\sum_j ( T_base[i,j] / a_i ) · X_other[j]
  where a_i = \sum_j T_base[i,j] and T_base is T (if base=1) or T.T (if base=2)
- Build a DAG adjacency by mixing base adjacency and transported adjacency from the other graph:
    A_mix = (1-λ)·A_base + λ·( W · A_other · W^T ),  with W[i,j] = T_base[i,j] / a_i
  Keep only i<j to ensure acyclicity. Choose the top E_bar edges by weight with
  E_bar = round((1-λ)·|E1| + λ·|E2|). Ensure all nodes with indegree 0 (i>0) get an in-edge from 0.
- Quantize feature values to valid ints and clip to reasonable ranges.
- Export as a workflows-only JSON: {"workflows": [ ... ]}

Usage:
  python -m scripts.fgw_interpolate_dags \
    --wide-config data/rl_configs/train_wide_p005_seeds.json \
    --longcp-config data/rl_configs/train_long_cp_p08_seeds.json \
    --pairs 5 \
    --lambdas 0.25,0.5,0.75 \
    --alpha 0.5 \
    --max-per-domain 50 \
    --out runs/datasets/mixed/representativeness_ot/datasets

Outputs:
- One aggregated JSON: dataset_interp_all.json with a list of workflows (each a valid DAG)
- Individual per-workflow JSONs under the same folder for inspection

Requirements: numpy, pandas, POT (ot), scikit-learn, matplotlib (optional)
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

# Project imports
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from cogito.dataset_generator.core.gen_dataset import generate_dataset


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


def _build_graph_from_seed(seed: int, ds_kwargs: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        vm_rng_seed=0,  # deterministic VMs
    )
    wf = ds.workflows[0]
    n = len(wf.tasks)
    # Features and adjacency
    X = np.zeros((n, 3), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    adj: List[List[int]] = [[] for _ in range(n)]
    for t in wf.tasks:
        X[t.id, 0] = float(t.length)
        X[t.id, 1] = float(t.req_memory_mb)
        X[t.id, 2] = float(t.req_cpu_cores)
        for v in t.child_ids:
            adj[t.id].append(int(v))
            if 0 <= v < n:
                A[t.id, v] = 1.0
    # Shortest-path undirected distances
    INF = 10**9
    C = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        C[i, i] = 0.0
        # BFS
        q = [i]
        dist = {i: 0}
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            for w in adj[u] + [u2 for u2 in range(n) if A[u2, u] == 1.0]:
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
    return X, C, A


def _fgw_coupling(X1: np.ndarray, C1: np.ndarray, X2: np.ndarray, C2: np.ndarray, alpha: float) -> np.ndarray:
    n1, n2 = X1.shape[0], X2.shape[0]
    a = np.ones(n1, dtype=np.float64) / max(1, n1)
    b = np.ones(n2, dtype=np.float64) / max(1, n2)
    # Feature cost matrix (squared Euclidean)
    x2 = (X1 * X1).sum(axis=1, keepdims=True)
    y2 = (X2 * X2).sum(axis=1, keepdims=True)
    M = x2 + y2.T - 2.0 * (X1 @ X2.T)
    M = np.maximum(M, 0.0)
    # Compute coupling by minimizing FGW^2
    try:
        T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, a, b, alpha=alpha)  # returns coupling
    except Exception:
        # Some POT versions use positional alpha
        T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, a, b, alpha)
    return T.astype(np.float64)


def _barycentric_features(base: int, X1: np.ndarray, X2: np.ndarray, T: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_bar, W) on base index, where W is normalized mapping weights from other->base.
    base=1: base graph is 1, map X2 to base via W[i,j] = T[i,j]/a_i
    base=2: base graph is 2, map X1 via W[j,i] = T[i,j]/b_j
    """
    if base == 1:
        a = T.sum(axis=1, keepdims=True) + 1e-12
        W = T / a  # shape (n1,n2)
        X2_to_1 = W @ X2
        X_bar = (1.0 - lam) * X1 + lam * X2_to_1
        return X_bar, W
    else:
        b = T.sum(axis=0, keepdims=True) + 1e-12
        W = (T / b).T  # shape (n2,n1): map 1->2 weights
        X1_to_2 = W @ X1
        X_bar = (1.0 - lam) * X2 + lam * X1_to_2
        return X_bar, W


def _mix_adjacency(base: int, A1: np.ndarray, A2: np.ndarray, W: np.ndarray, lam: float) -> Tuple[np.ndarray, int]:
    """Return mixed adjacency weights on base index and target edge count E_bar.
    If base=1: A_base=A1, A_other=A2, W maps other->base (shape n1 x n2)
    If base=2: A_base=A2, A_other=A1, W maps other->base (shape n2 x n1)
    """
    if base == 1:
        A_base = A1; A_other = A2
    else:
        A_base = A2; A_other = A1
    # Transport other adjacency into base: W @ A_other @ W^T
    A_other_on_base = W @ A_other @ W.T
    A_mix = (1.0 - lam) * A_base + lam * A_other_on_base
    # Target edge count: convex mix of edge counts (exclude self/illegal)
    E1 = int(A1.sum() - np.trace(A1))
    E2 = int(A2.sum() - np.trace(A2))
    E_bar = int(round((1.0 - lam) * E1 + lam * E2))
    return A_mix, E_bar


def _top_edges_dag(Aw: np.ndarray, E: int) -> np.ndarray:
    """Select top-E edges by weight with DAG constraint i<j. Ensure every node j>0 has indegree>=1 by forcing (0,j) if needed."""
    n = Aw.shape[0]
    M = np.triu(Aw, k=1)  # keep i<j
    # Flatten and pick top E
    idx = np.dstack(np.unravel_index(np.argsort(M, axis=None), M.shape))[0]
    # idx sorted ascending; take last E
    sel = idx[-E:]
    A = np.zeros_like(Aw, dtype=np.float64)
    for i, j in sel:
        if i < j:
            A[i, j] = 1.0
    # Ensure indegree for nodes j>0
    indeg = A.sum(axis=0)
    for j in range(1, n):
        if indeg[j] <= 1e-9:
            A[0, j] = 1.0
    return A


def _quantize_features(X: np.ndarray, min_len: int, max_len: int) -> np.ndarray:
    Xq = X.copy()
    # length
    Xq[:, 0] = np.clip(np.round(Xq[:, 0]), min_len, max_len)
    # memory MB to nearest 1024, at least 1024
    mem = np.maximum(1024.0, np.round(Xq[:, 1] / 1024.0) * 1024.0)
    Xq[:, 1] = mem
    # cpu cores >=1
    Xq[:, 2] = np.maximum(1.0, np.round(Xq[:, 2]))
    return Xq.astype(np.int64)


def _workflow_from_AX(Xq: np.ndarray, A: np.ndarray, wf_id: int) -> dict:
    n = Xq.shape[0]
    tasks = []
    for i in range(n):
        child_ids = [int(j) for j in range(n) if A[i, j] > 0.5]
        tasks.append({
            "id": int(i),
            "workflow_id": int(wf_id),
            "length": int(Xq[i, 0]),
            "req_memory_mb": int(Xq[i, 1]),
            "req_cpu_cores": int(Xq[i, 2]),
            "child_ids": child_ids,
        })
    return {"id": int(wf_id), "tasks": tasks, "arrival_time": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-config", type=Path, required=True)
    ap.add_argument("--longcp-config", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=5, help="number of seed pairs to interpolate")
    ap.add_argument("--lambdas", type=str, default="0.5", help="comma-separated list of interpolation weights (0..1)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-per-domain", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("runs/datasets/mixed/representativeness_ot/datasets"))
    args = ap.parse_args()

    if ot is None:
        raise SystemExit("POT (ot) is required. Please pip install POT")

    args.out.mkdir(parents=True, exist_ok=True)
    lam_list = [float(x) for x in args.lambdas.split(',') if x.strip()]

    seeds_w, ds_w = _load_rl_config(args.wide_config)
    seeds_l, ds_l = _load_rl_config(args.longcp_config)

    # Cap candidate pools
    seeds_w = seeds_w[: max(1, args.max_per_domain)]
    seeds_l = seeds_l[: max(1, args.max_per_domain)]

    # Use first N pairs by index; can be extended to nearest pairing later
    K = int(max(1, args.pairs))
    pairs = list(zip(seeds_w[:K], seeds_l[:K]))

    workflows = []
    wf_id = 0

    for sw, sl in pairs:
        # Build graphs
        X1, C1, A1 = _build_graph_from_seed(int(sw), ds_w)
        X2, C2, A2 = _build_graph_from_seed(int(sl), ds_l)
        # Z-score features jointly for stability (optional)
        allX = np.vstack([X1, X2])
        mu = allX.mean(axis=0, keepdims=True)
        sd = allX.std(axis=0, keepdims=True) + 1e-8
        Z1 = (X1 - mu) / sd
        Z2 = (X2 - mu) / sd

        # Coupling in standardized feature space
        T = _fgw_coupling(Z1, C1, Z2, C2, alpha=float(args.alpha))
        n1, n2 = X1.shape[0], X2.shape[0]

        for lam in lam_list:
            n_bar = int(round((1.0 - lam) * n1 + lam * n2))
            # Choose base side by closeness of size
            base = 1 if abs(n1 - n_bar) <= abs(n2 - n_bar) else 2
            if base == 1:
                X_bar, W = _barycentric_features(1, Z1, Z2, T, lam)
                A_mix, E_bar = _mix_adjacency(1, A1, A2, W, lam)
                # Unstandardize features
                X_bar = X_bar * sd + mu
                X_bar = X_bar.astype(np.float64)
            else:
                X_bar, W = _barycentric_features(2, Z1, Z2, T, lam)
                A_mix, E_bar = _mix_adjacency(2, A1, A2, W, lam)
                X_bar = X_bar * sd + mu
                X_bar = X_bar.astype(np.float64)

            # Keep base node count; enforce DAG and target edge count
            A_bary = _top_edges_dag(A_mix, max(1, E_bar))
            Xq = _quantize_features(X_bar, min_len=int(min(ds_w["min_task_length"], ds_l["min_task_length"])),
                                    max_len=int(max(ds_w["max_task_length"], ds_l["max_task_length"])) )
            wf = _workflow_from_AX(Xq, A_bary, wf_id)
            workflows.append(wf)
            # Save individual
            single = {"workflows": [wf]}
            out_one = args.out / f"workflow_interp_w{sw}_l{sl}_lam{lam:.2f}.json"
            with out_one.open("w") as f:
                json.dump(single, f)
            wf_id += 1

    # Save aggregated
    out_all = args.out / "dataset_interp_all.json"
    with out_all.open("w") as f:
        json.dump({"workflows": workflows}, f)

    print("[done] Wrote:")
    print(" -", out_all)
    print(" -", args.out)


if __name__ == "__main__":
    main()
