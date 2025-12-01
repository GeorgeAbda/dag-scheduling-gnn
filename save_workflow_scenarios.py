#!/usr/bin/env python3
"""
Generate and save three workflow datasets:
1) baseline (low bottleneck)
2) medium bottleneck
3) heavy bottleneck (the fast-heavy setup we trained with)

Outputs JSON files under data/saved_workflows/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scheduler.dataset_generator.core.gen_dataset import generate_dataset


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def build_and_save(
    *,
    out_dir: Path,
    filename: str,
    seed: int,
    host_count: int,
    vm_count: int,
    max_memory_gb: int,
    min_cpu_speed: int,
    max_cpu_speed: int,
    workflow_count: int,
    dag_method: str,
    gnp_min_n: int,
    gnp_max_n: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    task_arrival: str,
    arrival_rate: float,
) -> Path:
    dataset = generate_dataset(
        seed=seed,
        host_count=host_count,
        vm_count=vm_count,
        max_memory_gb=max_memory_gb,
        min_cpu_speed_mips=min_cpu_speed,
        max_cpu_speed_mips=max_cpu_speed,
        workflow_count=workflow_count,
        dag_method=dag_method,
        gnp_min_n=gnp_min_n,
        gnp_max_n=gnp_max_n,
        task_length_dist=task_length_dist,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        task_arrival=task_arrival,
        arrival_rate=arrival_rate,
        vm_rng_seed=0,
    )
    out_path = out_dir / f"{filename}.json"
    save_json(dataset.to_json(), out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Save three workflow scenarios (baseline/medium/heavy)")
    p.add_argument("--out-dir", type=Path, default=Path("data/saved_workflows"))
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    seed = int(args.seed)

    outputs = {}

    # 1) Baseline (matches your first config used earlier)
    outputs["baseline"] = build_and_save(
        out_dir=out_dir,
        filename="baseline",
        seed=seed,
        host_count=4,
        vm_count=10,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        workflow_count=10,
        dag_method="linear",
        gnp_min_n=12,
        gnp_max_n=24,
        task_length_dist="normal",
        min_task_length=500,
        max_task_length=100_000,
        task_arrival="static",
        arrival_rate=3.0,
    )

    # 2) Medium bottleneck (fewer VMs and longer tasks, moderate workflows)
    outputs["medium_bottleneck"] = build_and_save(
        out_dir=out_dir,
        filename="medium_bottleneck",
        seed=seed + 1,
        host_count=1,
        vm_count=2,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        workflow_count=12,
        dag_method="linear",
        gnp_min_n=48,
        gnp_max_n=96,
        task_length_dist="normal",
        min_task_length=50_000,
        max_task_length=200_000,
        task_arrival="static",
        arrival_rate=3.0,
    )

    # 3) Heavy bottleneck (the fast-heavy used in training comparisons)
    outputs["heavy_bottleneck"] = build_and_save(
        out_dir=out_dir,
        filename="heavy_bottleneck",
        seed=seed + 2,
        host_count=1,
        vm_count=1,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        workflow_count=10,
        dag_method="linear",
        gnp_min_n=80,
        gnp_max_n=120,
        task_length_dist="normal",
        min_task_length=150_000,
        max_task_length=400_000,
        task_arrival="static",
        arrival_rate=3.0,
    )

    print("Saved scenarios:")
    for k, v in outputs.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
