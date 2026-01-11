#!/usr/bin/env python3
"""
Build a combined dataset JSON containing:
- Real DAGs for FGW-mixed selected seeds (generated deterministically from RL configs)
- Selected interpolated DAGs (from dataset_interp_selected.json)

Usage:
  python -m scripts.build_mixed_plus_interpolated_dataset \
    --mixed-seeds runs/datasets/mixed/representativeness_ot/fgw_mixed_selected_eval_seeds.json \
    --wide-config data/rl_configs/train_wide_p005_seeds.json \
    --longcp-config data/rl_configs/train_long_cp_p08_seeds.json \
    --interp-json runs/datasets/mixed/representativeness_ot/datasets/dataset_interp_selected.json \
    --out runs/datasets/mixed/representativeness_ot/datasets/dataset_mixed_plus_interp.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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


def _workflow_from_seed(seed: int, ds_kwargs: dict, wf_id: int) -> dict:
    ds = generate_dataset(
        seed=int(seed),
        host_count=ds_kwargs["host_count"],
        vm_count=ds_kwargs["vm_count"],
        max_memory_gb=ds_kwargs["max_memory_gb"],
        min_cpu_speed_mips=ds_kwargs["min_cpu_speed_mips"],
        max_cpu_speed_mips=ds_kwargs["max_cpu_speed_mips"],
        workflow_count=1,
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
    tasks = []
    for t in wf.tasks:
        tasks.append({
            "id": int(t.id),
            "workflow_id": int(wf_id),
            "length": int(t.length),
            "req_memory_mb": int(t.req_memory_mb),
            "req_cpu_cores": int(t.req_cpu_cores),
            "child_ids": [int(c) for c in t.child_ids],
        })
    return {"id": int(wf_id), "tasks": tasks, "arrival_time": 0}


def _load_mixed_seeds(path: Path) -> List[int]:
    with path.open("r") as f:
        data = json.load(f)
    seeds = data.get("selected_eval_seeds", [])
    return [int(x) for x in seeds]


def _load_interp_selected(path: Path) -> List[dict]:
    with path.open("r") as f:
        data = json.load(f)
    return list(data.get("workflows", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed-seeds", type=Path, required=True)
    ap.add_argument("--wide-config", type=Path, required=True)
    ap.add_argument("--longcp-config", type=Path, required=True)
    ap.add_argument("--interp-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mixed_seeds = _load_mixed_seeds(args.mixed_seeds)
    seeds_wide, ds_w = _load_rl_config(args.wide_config)
    seeds_long, ds_l = _load_rl_config(args.longcp_config)

    workflows: List[dict] = []
    wf_id = 0

    # For each seed, decide domain by numeric range (200xxx -> wide, 100xxx -> longcp)
    for s in mixed_seeds:
        if s >= 200000:
            wf = _workflow_from_seed(s, ds_w, wf_id)
        else:
            wf = _workflow_from_seed(s, ds_l, wf_id)
        workflows.append(wf)
        wf_id += 1

    # Append selected interpolated workflows, reindexing workflow_id
    interp_wfs = _load_interp_selected(args.interp_json)
    for wf in interp_wfs:
        # rewrite workflow_id and possibly task ids to 0..n-1 (they already are)
        n = len(wf.get("tasks", []))
        for t in wf.get("tasks", []):
            t["workflow_id"] = int(wf_id)
        wf_out = {"id": int(wf_id), "tasks": wf.get("tasks", []), "arrival_time": int(wf.get("arrival_time", 0))}
        workflows.append(wf_out)
        wf_id += 1

    # Save combined
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"workflows": workflows}, f)

    print("[build] Combined dataset written:")
    print(" -", args.out)
    print(" - workflows:", len(workflows))


if __name__ == "__main__":
    main()
