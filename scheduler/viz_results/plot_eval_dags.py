import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

from scheduler.config.settings import ROOT_PATH
from scheduler.dataset_generator.core.gen_dataset import (
    generate_dataset_long_cp_queue_free,
    generate_dataset_wide_queue_free,
    _topological_layers,
)


# Utilities
# ----------------------------------------------------------------------------------------------------------------------

def _load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        return json.load(f)


def _generate_datasets_from_eval(eval_block: dict) -> List[Tuple[int, object]]:
    """
    Given an eval block from RL config (with keys: seeds, dataset), generate a dataset per seed.
    Returns list of (seed, Dataset).
    """
    seeds: List[int] = list(eval_block.get("seeds", []))
    ds = eval_block.get("dataset", {})
    style = str(ds.get("style", "generic")).lower()

    # Common params
    host_count = int(ds.get("host_count", 10))
    vm_count = int(ds.get("vm_count", 10))
    max_memory_gb = int(ds.get("max_memory_gb", 128))
    workflow_count = int(ds.get("workflow_count", 1))
    gnp_min_n = int(ds.get("gnp_min_n", 12))
    gnp_max_n = int(ds.get("gnp_max_n", 24))
    task_length_dist = str(ds.get("task_length_dist", "normal"))
    min_task_length = int(ds.get("min_task_length", 500))
    max_task_length = int(ds.get("max_task_length", 100000))
    task_arrival = str(ds.get("task_arrival", "static"))
    arrival_rate = float(ds.get("arrival_rate", 3.0))
    # VM speed range (used only for VM generation; does not affect DAG structure)
    min_cpu_speed_mips = 500
    max_cpu_speed_mips = 5000

    # Pin p via (p, p) if provided
    gnp_p = ds.get("gnp_p", None)
    p_range = None
    if gnp_p is not None:
        p = float(gnp_p)
        p_range = (p, p)

    out: List[Tuple[int, object]] = []
    for seed in seeds:
        if style == "long_cp":
            dataset = generate_dataset_long_cp_queue_free(
                seed=int(seed),
                host_count=host_count,
                vm_count=vm_count,
                max_memory_gb=max_memory_gb,
                min_cpu_speed_mips=min_cpu_speed_mips,
                max_cpu_speed_mips=max_cpu_speed_mips,
                workflow_count=workflow_count,
                gnp_min_n=gnp_min_n,
                gnp_max_n=gnp_max_n,
                task_length_dist=task_length_dist,
                min_task_length=min_task_length,
                max_task_length=max_task_length,
                task_arrival=task_arrival,
                arrival_rate=arrival_rate,
                vm_rng_seed=0,
                p_range=p_range or (0.70, 0.95),
                alpha_range=(0.8, 0.95),
            )
        elif style == "wide":
            dataset = generate_dataset_wide_queue_free(
                seed=int(seed),
                host_count=host_count,
                vm_count=vm_count,
                max_memory_gb=max_memory_gb,
                min_cpu_speed_mips=min_cpu_speed_mips,
                max_cpu_speed_mips=max_cpu_speed_mips,
                workflow_count=workflow_count,
                gnp_min_n=gnp_min_n,
                gnp_max_n=gnp_max_n,
                task_length_dist=task_length_dist,
                min_task_length=min_task_length,
                max_task_length=max_task_length,
                task_arrival=task_arrival,
                arrival_rate=arrival_rate,
                vm_rng_seed=0,
                p_range=p_range or (0.02, 0.20),
                alpha_range=(0.8, 0.95),
            )
        else:
            raise ValueError(f"Unsupported style for eval plotting: {style}")
        out.append((int(seed), dataset))
    return out


def _draw_workflow(ax, workflow) -> None:
    """
    Draw a single workflow DAG into the provided matplotlib Axes using a simple layered layout.
    """
    tasks = getattr(workflow, "tasks", [])
    children = {t.id: set(t.child_ids) for t in tasks}
    layers = _topological_layers(children)

    # Assign node positions by layer and index within layer
    positions: dict[int, Tuple[float, float]] = {}
    for li, layer_nodes in enumerate(layers):
        k = max(1, len(layer_nodes))
        # Centered x positions
        xs = [j - (k - 1) / 2 for j in range(k)]
        xs = [x * 1.4 for x in xs]  # spread a bit
        y = -li
        for x, nid in zip(xs, layer_nodes):
            positions[int(nid)] = (float(x), float(y))

    # Draw edges first (behind nodes)
    for u, vs in children.items():
        x0, y0 = positions[u]
        for v in vs:
            x1, y1 = positions[v]
            ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=1.5, zorder=1)

    # Draw nodes
    xs = [positions[t.id][0] for t in tasks]
    ys = [positions[t.id][1] for t in tasks]
    ax.scatter(xs, ys, s=160, c="#2a9d8f", edgecolors="#1b4d4a", linewidths=1.2, zorder=2)

    # Node labels
    for t in tasks:
        x, y = positions[t.id]
        ax.text(x, y, str(t.id), ha="center", va="center", fontsize=8, color="white", zorder=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _plot_eval_set(cfg_path: Path, out_path: Path) -> None:
    cfg = _load_config(cfg_path)
    eval_block = cfg.get("eval", {})
    # Determine style for title heuristics
    ds = eval_block.get("dataset", {})
    style = str(ds.get("style", "")).upper()

    items = _generate_datasets_from_eval(eval_block)

    # Layout params
    n = len(items)
    cols = 5 if n >= 5 else n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, (seed, dataset) in enumerate(items):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[0][c]
        wf = dataset.workflows[0] if hasattr(dataset, "workflows") and dataset.workflows else None
        if wf is None:
            ax.set_axis_off()
            ax.set_title(f"Seed {seed}: (no workflow)")
            continue
        _draw_workflow(ax, wf)
        n_nodes = len(getattr(wf, "tasks", []))
        n_edges = sum(len(getattr(t, "child_ids", []) or []) for t in wf.tasks)
        ax.set_title(f"{style} seed {seed}\n|V|={n_nodes} |E|={n_edges}", fontsize=9)

    # Hide any unused axes
    for extra in range(n, rows * cols):
        r, c = divmod(extra, cols)
        ax = axes[r][c] if rows > 1 else axes[0][c]
        ax.set_axis_off()

    fig.suptitle(f"Eval DAGs: {style}", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # Default config paths provided by the repo
    cfg_long = ROOT_PATH / "data" / "rl_configs" / "train_long_cp_p08_seeds.json"
    cfg_wide = ROOT_PATH / "data" / "rl_configs" / "train_wide_p005_seeds.json"

    out_dir = ROOT_PATH / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_eval_set(cfg_long, out_dir / "eval_dags_longcp.png")
    _plot_eval_set(cfg_wide, out_dir / "eval_dags_wide.png")
    print(f"Saved figures to: {out_dir / 'eval_dags_longcp.png'} and {out_dir / 'eval_dags_wide.png'}")


if __name__ == "__main__":
    main()
