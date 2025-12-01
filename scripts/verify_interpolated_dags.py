#!/usr/bin/env python3
"""
Verify that interpolated workflows form valid DAGs (acyclic) and report stats.

Usage:
  python -m scripts.verify_interpolated_dags \
    --json runs/datasets/mixed/representativeness_ot/datasets/dataset_interp_all.json

Outputs a summary and lists any workflows that fail DAG checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


def _load_workflows(path: Path) -> List[Dict]:
    with path.open("r") as f:
        data = json.load(f)
    return list(data.get("workflows", []))


def _adj_from_workflow(wf: Dict) -> np.ndarray:
    tasks = wf.get("tasks", [])
    n = len(tasks)
    A = np.zeros((n, n), dtype=np.int32)
    for t in tasks:
        i = int(t["id"])
        for j in t.get("child_ids", []):
            j = int(j)
            if 0 <= i < n and 0 <= j < n:
                A[i, j] = 1
    return A


def _is_dag_kahn(A: np.ndarray) -> bool:
    n = A.shape[0]
    indeg = A.sum(axis=0).astype(int)
    q = [i for i in range(n) if indeg[i] == 0]
    visited = 0
    while q:
        u = q.pop()
        visited += 1
        for v in np.where(A[u] > 0)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(int(v))
    return visited == n


def _upper_triangular_only(A: np.ndarray) -> bool:
    # No edges j<=i
    return np.all(A <= np.triu(A, k=0)) and np.count_nonzero(A - np.triu(A, k=1)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    wfs = _load_workflows(args.json)
    if not wfs:
        print("[warn] No workflows found.")
        return

    total_edges = 0
    bad_cycles: List[int] = []
    bad_upper: List[int] = []

    for k, wf in enumerate(wfs):
        A = _adj_from_workflow(wf)
        total_edges += int(A.sum())
        if not _is_dag_kahn(A):
            bad_cycles.append(k)
        if not _upper_triangular_only(A):
            bad_upper.append(k)

    print("[verify]")
    print(f" - file: {args.json}")
    print(f" - workflows: {len(wfs)}")
    print(f" - total_edges: {total_edges}")
    print(f" - acyclic: {len(wfs) - len(bad_cycles)} / {len(wfs)}")
    if bad_cycles:
        print(f" - with_cycles: {bad_cycles}")
    if bad_upper:
        print(f" - with_non_upper_triangular_edges: {bad_upper}")


if __name__ == "__main__":
    main()
