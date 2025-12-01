#!/usr/bin/env python3
"""
Predictive geometric analysis of DAG datasets to estimate state distribution and
architecture suitability (MLP-only / baseline-homogeneous GNN / hetero GNN).

Pure-Python (no pandas/sklearn/matplotlib) to avoid env conflicts.

Inputs (existing in this repo):
- runs/datasets/<domain>/train/seed_*.json for each domain in {longcp, wide}
- Optional: runs/representativeness/geometry/boundary_seeds.csv to flag tails

Outputs (under runs/representativeness/geometry):
- state_geometry_features.csv : per-seed feature vector and predicted architecture
- overlaps.json               : domain-wise stats and centroid deltas per feature
- gating_thresholds.json      : the deterministic rules used for prediction
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


@dataclass
class Task:
    id: int
    length: int
    req_memory_mb: int
    req_cpu_cores: int
    child_ids: List[int]


@dataclass
class VM:
    id: int
    memory_mb: int
    cpu_cores: int


# ----------------------------
# DAG utilities
# ----------------------------

def load_dataset_json(path: Path) -> Tuple[List[Task], List[VM]]:
    with path.open("r") as f:
        data = json.load(f)
    workflows = data.get("workflows", [])
    wf = workflows[0] if workflows else {"tasks": []}
    tasks: List[Task] = []
    for t in wf.get("tasks", []):
        tasks.append(
            Task(
                id=int(t.get("id", len(tasks))),
                length=int(t.get("length", 0)),
                req_memory_mb=int(t.get("req_memory_mb", 0)),
                req_cpu_cores=int(t.get("req_cpu_cores", 1)),
                child_ids=[int(x) for x in t.get("child_ids", [])],
            )
        )
    vms: List[VM] = []
    for v in data.get("vms", []):
        vms.append(
            VM(
                id=int(v.get("id", len(vms))),
                memory_mb=int(v.get("memory_mb", 0)),
                cpu_cores=int(v.get("cpu_cores", 1)),
            )
        )
    return tasks, vms


def topo_layers(tasks: List[Task]) -> List[List[int]]:
    n = len(tasks)
    indeg = [0] * n
    children: List[List[int]] = [[] for _ in range(n)]
    for t in tasks:
        for c in t.child_ids:
            if 0 <= c < n:
                indeg[c] += 1
                children[t.id].append(c)
    # Kahn layering
    layer: List[List[int]] = []
    cur: List[int] = [i for i in range(n) if indeg[i] == 0]
    seen: set[int] = set()
    while cur:
        layer.append(cur[:])
        nxt: List[int] = []
        for u in cur:
            seen.add(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
        cur = nxt
    if len(seen) != n:
        rem = [i for i in range(n) if i not in seen]
        if rem:
            layer.append(rem)
    return layer


def longest_path_len(tasks: List[Task]) -> float:
    # DP in reverse topological order using layers
    order: List[int] = [i for L in topo_layers(tasks) for i in L]
    if not order:
        return 0.0
    children = {t.id: list(t.child_ids) for t in tasks}
    L = {t.id: float(t.length) for t in tasks}
    dp: Dict[int, float] = {i: 0.0 for i in order}
    for u in reversed(order):
        best = 0.0
        for v in children.get(u, []):
            best = max(best, dp.get(v, 0.0))
        dp[u] = L.get(u, 0.0) + best
    start = order[0]
    mx = 0.0
    for i in order:
        if dp[i] > mx:
            mx = dp[i]
            start = i
    return float(dp.get(start, 0.0))


def compat_degrees(tasks: List[Task], vms: List[VM]) -> List[int]:
    mems = [vm.memory_mb for vm in vms]
    cores = [max(1, vm.cpu_cores) for vm in vms]
    degs: List[int] = []
    for t in tasks:
        cnt = 0
        for m, c in zip(mems, cores):
            if m >= t.req_memory_mb and c >= t.req_cpu_cores:
                cnt += 1
        degs.append(cnt)
    return degs


def build_phi(tasks: List[Task], vms: List[VM]) -> Dict[str, float]:
    n = len(tasks)
    vm_count = max(1, len(vms))
    Ls = topo_layers(tasks)
    widths = [len(L) for L in Ls]
    width_peak = float(max(widths) if widths else 0.0)
    width_mean = float(mean(widths) if widths else 0.0)
    width_var = float(pstdev(widths) ** 2 if len(widths) > 1 else 0.0)
    frac_layers_over_vm = float(sum(1 for w in widths if w > vm_count) / max(1, len(widths)))

    # Weighted frontier by total MI per layer
    layer_weight: List[float] = []
    for L in Ls:
        layer_weight.append(float(sum(tasks[i].length for i in L)))
    if sum(layer_weight) > 0.0:
        w_frontier_mean = float(sum(w * wt for w, wt in zip(widths, layer_weight)) / sum(layer_weight))
    else:
        w_frontier_mean = width_mean

    cp_len = longest_path_len(tasks)
    tot_len = float(sum(t.length for t in tasks))
    cp_len_ratio = float(cp_len / tot_len) if tot_len > 0 else 0.0

    degs = compat_degrees(tasks, vms)
    deg_mean = float(mean(degs) if degs else 0.0)
    deg_over_vm = float(deg_mean / vm_count)
    deg_var = float(pstdev(degs) ** 2 if len(degs) > 1 else 0.0)

    req_mem_mean = float(mean([t.req_memory_mb for t in tasks]) if tasks else 0.0)
    req_cores_mean = float(mean([t.req_cpu_cores for t in tasks]) if tasks else 0.0)
    len_mean = float(mean([t.length for t in tasks]) if tasks else 0.0)

    # Branching-factor approximation across layers
    # For each layer L, approximate number of valid actions as sum of compat degrees
    # across ready tasks in that layer: B_L = sum_{i in L} deg(i).
    # Also report an upper bound limited by VM count: UB_L = min(B_L, |L| * vm_count).
    B_vals: List[float] = []
    B_ub_vals: List[float] = []
    pair_density_vals: List[float] = []
    for w, L in zip(widths, Ls):
        b = float(sum(degs[i] for i in L)) if L else 0.0
        ub = float(min(b, w * vm_count))
        B_vals.append(b)
        B_ub_vals.append(ub)
        denom = max(1.0, w * vm_count)
        pair_density_vals.append(b / denom)

    def _p(v: List[float], q: float) -> float:
        if not v:
            return 0.0
        s = sorted(v)
        k = int(max(0, min(len(s) - 1, round(q * (len(s) - 1)))))
        return float(s[k])

    B_mean = float(mean(B_vals) if B_vals else 0.0)
    B_p90 = _p(B_vals, 0.90)
    B_max = float(max(B_vals) if B_vals else 0.0)
    Bub_mean = float(mean(B_ub_vals) if B_ub_vals else 0.0)
    pair_dens_mean = float(mean(pair_density_vals) if pair_density_vals else 0.0)

    phi = {
        "n_tasks": float(n),
        "width_peak": width_peak,
        "width_mean": width_mean,
        "width_var": width_var,
        "frontier_over_vm_frac": float(width_mean / vm_count),
        "frontier_over_vm_tail": frac_layers_over_vm,
        "w_frontier_mean": w_frontier_mean,
        "depth_layers": float(len(Ls)),
        "cp_len_ratio": cp_len_ratio,
        "deg_mean": deg_mean,
        "deg_over_vm": deg_over_vm,
        "deg_var": deg_var,
        "req_mem_mean": req_mem_mean,
        "req_cores_mean": req_cores_mean,
        "len_mean": len_mean,
        # Branching-factor features
        "B_mean": B_mean,
        "B_p90": B_p90,
        "B_max": B_max,
        "Bub_mean": Bub_mean,
        "pair_density_mean": pair_dens_mean,
    }
    return phi


# ----------------------------
# Architecture prediction rules
# ----------------------------

def predict_arch(phi: Dict[str, float]) -> str:
    f_over = phi["frontier_over_vm_frac"]
    tail = phi["frontier_over_vm_tail"]
    deg_vm = phi["deg_over_vm"]
    cp = phi["cp_len_ratio"]
    wpk = phi["width_peak"]
    b90 = phi.get("B_p90", 0.0)
    bmax = phi.get("B_max", 0.0)

    # Hetero region: high concurrency, large branching, or high compatibility diversity
    if (f_over >= 0.50) or (tail >= 0.20) or (deg_vm >= 0.35) or (wpk >= 10) or (b90 >= 120) or (bmax >= 200):
        return "hetero"
    # MLP region: strongly CP-dominated, narrow, low compatibility
    if (wpk <= 3) and (deg_vm <= 0.25) and (cp >= 0.55) and (bmax <= 40):
        return "mlp"
    # Intermediate: homogeneous GNN baseline
    return "baseline"


def gating_rules() -> Dict[str, object]:
    return {
        "hetero": {
            "any_of": {
                "frontier_over_vm_frac": ">= 0.50",
                "frontier_over_vm_tail": ">= 0.20",
                "deg_over_vm": ">= 0.35",
                "width_peak": ">= 10",
                "B_p90": ">= 120",
                "B_max": ">= 200",
            }
        },
        "mlp": {
            "all_of": {
                "width_peak": "<= 3",
                "deg_over_vm": "<= 0.25",
                "cp_len_ratio": ">= 0.55",
                "B_max": "<= 40",
            }
        },
        "baseline": "otherwise",
    }


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="runs/representativeness/geometry")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = ["longcp", "wide"]
    rows: List[Dict[str, object]] = []

    for dom in domains:
        train_dir = Path(f"runs/datasets/{dom}/train")
        for fp in sorted(glob(str(train_dir / "seed_*.json"))):
            p = Path(fp)
            seed = int(p.stem.split("_")[-1])
            try:
                tasks, vms = load_dataset_json(p)
                phi = build_phi(tasks, vms)
                phi_row: Dict[str, object] = {k: float(v) for k, v in phi.items()}
                phi_row["seed"] = seed
                phi_row["domain"] = dom
                phi_row["pred_arch"] = predict_arch(phi)
                rows.append(phi_row)
            except Exception as e:
                print(f"[warn] failed {p}: {e}")

    # Merge boundary seeds if present
    boundary_map: Dict[Tuple[str, int], bool] = {}
    bcsv = out_dir / "boundary_seeds.csv"
    if bcsv.exists():
        try:
            with bcsv.open("r", newline="") as f:
                r = csv.DictReader(f)
                for rec in r:
                    dom = str(rec.get("domain", "")).strip()
                    seed = int(rec.get("seed", 0))
                    boundary_map[(dom, seed)] = True
        except Exception:
            pass

    # Write features CSV
    feat_cols = [
        "n_tasks","width_peak","width_mean","width_var","frontier_over_vm_frac",
        "frontier_over_vm_tail","w_frontier_mean","depth_layers","cp_len_ratio",
        "deg_mean","deg_over_vm","deg_var","req_mem_mean","req_cores_mean","len_mean",
        "B_mean","B_p90","B_max","Bub_mean","pair_density_mean",
    ]
    out_csv = out_dir / "state_geometry_features.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = ["domain","seed","pred_arch","is_boundary"] + feat_cols
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            dom = str(r["domain"]) ; seed = int(r["seed"]) ; arch = str(r["pred_arch"])
            ob = {
                "domain": dom,
                "seed": seed,
                "pred_arch": arch,
                "is_boundary": 1 if boundary_map.get((dom, seed), False) else 0,
            }
            for c in feat_cols:
                ob[c] = float(r.get(c, 0.0))
            w.writerow(ob)

    # Overlap and centroids per domain
    def centroid(vals: List[float]) -> float:
        return sum(vals)/max(1, len(vals))

    agg: Dict[str, Dict[str, float]] = {dom: {} for dom in domains}
    # Compute centroids
    for dom in domains:
        dom_rows = [r for r in rows if r["domain"] == dom]
        for c in feat_cols:
            agg[dom][c] = centroid([float(r.get(c, 0.0)) for r in dom_rows])
    # Standardized deltas per feature (|mu1-mu2| / pooled sd)
    deltas: Dict[str, float] = {}
    for c in feat_cols:
        v1 = [float(r.get(c, 0.0)) for r in rows if r["domain"] == "longcp"]
        v2 = [float(r.get(c, 0.0)) for r in rows if r["domain"] == "wide"]
        if len(v1) > 1 and len(v2) > 1:
            m1, m2 = centroid(v1), centroid(v2)
            sd1, sd2 = pstdev(v1), pstdev(v2)
            pooled = math.sqrt((sd1**2 + sd2**2)/2.0) if (sd1>0 or sd2>0) else 1.0
            deltas[c] = abs(m1-m2)/pooled
        else:
            deltas[c] = 0.0

    # Architecture mix per domain
    mix: Dict[str, Dict[str, float]] = {dom: {"mlp":0, "baseline":0, "hetero":0} for dom in domains}
    for dom in domains:
        dom_rows = [r for r in rows if r["domain"] == dom]
        total = max(1, len(dom_rows))
        for a in ("mlp","baseline","hetero"):
            mix[dom][a] = sum(1 for r in dom_rows if r["pred_arch"] == a) / total

    # Save overlaps.json
    out_json = out_dir / "overlaps.json"
    with out_json.open("w") as f:
        json.dump({
            "centroids": agg,
            "std_deltas": deltas,
            "arch_mix": mix,
        }, f, indent=2)

    # Save gating thresholds
    with (out_dir / "gating_thresholds.json").open("w") as f:
        json.dump(gating_rules(), f, indent=2)

    print("[done] Wrote:")
    print(" -", out_csv)
    print(" -", out_json)
    print(" -", out_dir / "gating_thresholds.json")


if __name__ == "__main__":
    main()
