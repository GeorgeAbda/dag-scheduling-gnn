#!/usr/bin/env python3
"""Generate 3D energy fitness landscapes for long_cp-like and wide-like domains.

This script:
- Reads the RL configs for long_cp and wide:
    - data/rl_configs/train_long_cp_p08_seeds.json
    - data/rl_configs/train_wide_p005_seeds.json
- Builds small, tractable EnumArgs for each domain (few tasks, few VMs) that
  approximate the long_cp / wide regimes.
- Uses the existing Hilbert-curve enumeration and enhanced 3D surface
  visualization utilities to render the *energy* landscape over the
  scheduling search space.

The resulting figures are saved under logs/landscape/ as PNG and PDF.

Note: For computational reasons, we use much smaller host_count, vm_count,
      and DAG sizes than the full training configs, but we keep the other
      dataset characteristics (e.g., gnp_p, task length distribution) aligned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import replace as _dc_replace

import numpy as np

# Make project imports work when running this file directly from the repo root.
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from cogito.viz_results.decision_boundaries.solution_landscape import (
    EnumArgs,
    build_env,
    enumerate_spacefill_once,
)
from enhanced_3d_viz import create_enhanced_3d_surface, create_difference_3d_surface
DATA_DIR = ROOT / "data"
RL_CFG_LONG = DATA_DIR / "rl_configs" / "train_long_cp_p08_seeds.json"
RL_CFG_WIDE = DATA_DIR / "rl_configs" / "train_wide_p005_seeds.json"
OUT_DIR = ROOT / "logs" / "landscape"


def _load_train_dataset_cfg(path: Path) -> dict:
    """Return the training dataset sub-config from an RL JSON config.

    The RL config has the structure {"train": {"dataset": {...}, ...}, ...}.
    We only care about the dataset dict here; seeds are not used for the
    exhaustive landscape (we instead fix a seed explicitly for reproducible
    enumeration).
    """
    with path.open("r") as f:
        cfg = json.load(f)
    train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    return dict(train.get("dataset", {}))


def _dataset_args_from_cfg(dataset_cfg: dict, seed: int) -> DatasetArgs:
    """Mirror the keys used in the training configs into DatasetArgs.

    This mirrors scheduler/rl_model/collect_random_state_space._dataset_args_from_cfg
    but is kept local to avoid importing that training script.
    """
    return DatasetArgs(
        seed=int(seed),
        host_count=int(dataset_cfg.get("host_count", 4)),
        vm_count=int(dataset_cfg.get("vm_count", 10)),
        max_memory_gb=int(dataset_cfg.get("max_memory_gb", 10)),
        min_cpu_speed=int(dataset_cfg.get("min_cpu_speed", 500)),
        max_cpu_speed=int(dataset_cfg.get("max_cpu_speed", 5000)),
        workflow_count=int(dataset_cfg.get("workflow_count", 1)),
        dag_method=str(dataset_cfg.get("dag_method", "gnp")),
        gnp_min_n=int(dataset_cfg.get("gnp_min_n", 10)),
        gnp_max_n=int(dataset_cfg.get("gnp_max_n", 40)),
        task_length_dist=str(dataset_cfg.get("task_length_dist", "normal")),
        min_task_length=int(dataset_cfg.get("min_task_length", 500)),
        max_task_length=int(dataset_cfg.get("max_task_length", 100_000)),
        task_arrival=str(dataset_cfg.get("task_arrival", "static")),
        arrival_rate=float(dataset_cfg.get("arrival_rate", 3.0)),
        style=str(dataset_cfg.get("style", "generic")),
        gnp_p=dataset_cfg.get("gnp_p", None),
    )


def _compute_optimal_req_divisor_from_cfg(dataset_cfg: dict, seed: int) -> int:
    """Mirror ablation_gnn_traj_main._compute_optimal_req_divisor using JSON cfg.

    This computes a conservative req_divisor so that repeating per-task
    demands across all tasks does not exceed the smallest VM's capacity.
    """
    host_count = int(dataset_cfg.get("host_count", 4))
    vm_count = int(dataset_cfg.get("vm_count", 10))
    max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
    min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
    max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))
    n_tasks = int(dataset_cfg.get("gnp_max_n", 40))
    if n_tasks <= 0:
        n_tasks = 1

    rng = np.random.RandomState(int(seed))
    hosts = generate_hosts(n=host_count, rng=rng)
    vms = generate_vms(
        n=vm_count,
        max_memory_gb=max_memory_gb,
        min_cpu_speed_mips=min_cpu_speed,
        max_cpu_speed_mips=max_cpu_speed,
        rng=rng,
    )
    allocate_vms(vms, hosts, rng)
    if not vms:
        return 1

    mem_caps = [int(getattr(vm, "memory_mb", 0)) for vm in vms]
    core_caps = [int(max(1, getattr(vm, "cpu_cores", 1))) for vm in vms]
    min_mem = max(1, min(mem_caps))
    max_mem = max(mem_caps)
    min_cores = max(1, min(core_caps))
    max_cores = max(core_caps)

    max_safe_mem_per_task = max(1024, min_mem // n_tasks)
    max_safe_cores_per_task = max(1, min_cores // n_tasks)
    req_div_mem = max(1, max_mem // max_safe_mem_per_task)
    req_div_core = max(1, max_cores // max_safe_cores_per_task)
    print(f"[req_div][{dataset_cfg.get('style', 'unknown')}] mem_div={req_div_mem}, core_div={req_div_core}, "
          f"max_mem={max_mem}, max_cores={max_cores}, n_tasks={n_tasks}")
    return int(max(req_div_mem, req_div_core))


def _make_enum_args_from_dataset(ds_args: DatasetArgs, *, small_host_count: int, small_vm_count: int,
                                 small_gnp_n: int) -> EnumArgs:
    """Construct a small EnumArgs that approximates the given DatasetArgs.

    We keep distributional characteristics (task lengths, DAG method) but
    shrink host_count, vm_count, and DAG size to make exhaustive enumeration
    tractable.
    """
    # For host/VM topology, respect the RL config / host_specs-driven values
    # in ds_args. We only shrink the DAG size (gnp_*_n) to keep enumeration
    # tractable, but keep the full host_count and vm_count.
    return EnumArgs(
        host_count=ds_args.host_count,
        vm_count=ds_args.vm_count,
        workflow_count=1,
        gnp_min_n=small_gnp_n,
        gnp_max_n=small_gnp_n,
        dag_method=ds_args.dag_method,
        style=str(getattr(ds_args, "style", "generic")),
        gnp_p=getattr(ds_args, "gnp_p", None),
        req_divisor=getattr(ds_args, "req_divisor", None),
        min_task_length=ds_args.min_task_length,
        max_task_length=ds_args.max_task_length,
        min_cpu_speed=ds_args.min_cpu_speed,
        max_cpu_speed=ds_args.max_cpu_speed,
        max_memory_gb=ds_args.max_memory_gb,
        seed=ds_args.seed,
        score="energy",
        out_dir=str(OUT_DIR),
        show_progress=True,
    )


def _generate_landscape_for_domain(name: str, rl_cfg_path: Path,
                                   small_host_count: int,
                                   small_vm_count: int,
                                   small_gnp_n: int,
                                   sf_limit: int | None = None):
    print(f"[landscape] Domain={name}, cfg={rl_cfg_path}")

    ds_cfg = _load_train_dataset_cfg(rl_cfg_path)
    # Use a fixed seed for reproducible, domain-consistent enumeration.
    # (The RL configs contain many seeds for training, but for the purposes
    #  of this exhaustive landscape we just want a single representative
    #  realization of each domain.)
    seed = 12345

    # Compute a conservative req_divisor using the same logic as ablation.
    req_div = _compute_optimal_req_divisor_from_cfg(ds_cfg, seed=seed)

    # Build DatasetArgs (includes style, gnp_p, etc.), then attach req_divisor
    # and shrink to EnumArgs for enumeration.
    ds_args = _dataset_args_from_cfg(ds_cfg, seed=seed)
    ds_args = _dc_replace(ds_args, req_divisor=int(req_div))
    enum_args = _make_enum_args_from_dataset(
        ds_args,
        small_host_count=small_host_count,
        small_vm_count=small_vm_count,
        small_gnp_n=small_gnp_n,
    )

    print(f"[landscape] Using EnumArgs: host_count={enum_args.host_count}, vm_count={enum_args.vm_count}, "
          f"gnp_min_n={enum_args.gnp_min_n}, gnp_max_n={enum_args.gnp_max_n}, seed={enum_args.seed}")

    # Build environment and enumerate solution space
    env = build_env(enum_args)
    env.reset(seed=enum_args.seed)

    grids, feas_mask, visited, side = enumerate_spacefill_once(env, enum_args, sf_limit)

    # Report feasibility statistics for this domain so we can compare longcp vs wide.
    total_visited = int(visited.sum())
    total_feasible = int(feas_mask.sum())
    frac = (total_feasible / total_visited) if total_visited > 0 else 0.0
    print(
        f"[landscape][{name}] feasible cells: {total_feasible} / {total_visited} "
        f"({frac * 100.0:.4f}% of visited)"
    )

    # Render enhanced 3D surface for energy
    metric = "energy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[landscape] Creating enhanced 3D surface for {name} ({metric})...")
    create_enhanced_3d_surface(
        grids,
        feas_mask,
        visited,
        side,
        dag_type=name,
        metric=metric,
        out_dir=str(OUT_DIR),
        vmin_global=None,
        vmax_global=None,
        paper_mode=False,
    )

    # Return raw grids and masks so caller can compare domains (e.g., longcp vs wide).
    return grids, feas_mask, visited, side


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Choose small but non-trivial settings for exhaustive enumeration.
    # You may adjust these if enumeration is too slow or too trivial.
    small_host_count = 10
    small_vm_count = 10
    # Slightly smaller DAG size to keep enumeration tractable.
    small_gnp_n = 20

    # Cap the number of enumerated sequences to avoid very long runtimes.
    # This is an upper bound on sequences; the Hilbert grid side is computed
    # from this cap.
    sf_limit = 50_000

    # Long critical path-like domain
    grids_long, feas_long, visited_long, side_long = _generate_landscape_for_domain(
        name="longcp",
        rl_cfg_path=RL_CFG_LONG,
        small_host_count=small_host_count,
        small_vm_count=small_vm_count,
        small_gnp_n=small_gnp_n,
        sf_limit=sf_limit,
    )

    # Wide-like domain
    grids_wide, feas_wide, visited_wide, side_wide = _generate_landscape_for_domain(
        name="wide",
        rl_cfg_path=RL_CFG_WIDE,
        small_host_count=small_host_count,
        small_vm_count=small_vm_count,
        small_gnp_n=small_gnp_n,
        sf_limit=sf_limit,
    )

    # Sanity check: both should use the same grid side for a fair comparison.
    if side_long == side_wide:
        print("[landscape] Creating 3D difference surface (longcp - wide) for energy over common feasible region...")
        create_difference_3d_surface(
            grids_long,
            feas_long,
            visited_long,
            grids_wide,
            feas_wide,
            visited_wide,
            side_long,
            metric="energy",
            out_dir=str(OUT_DIR),
            label_a="longcp",
            label_b="wide",
        )
    else:
        print(f"[landscape][warn] side mismatch between longcp ({side_long}) and wide ({side_wide}); skipping diff surface.")


if __name__ == "__main__":
    main()
