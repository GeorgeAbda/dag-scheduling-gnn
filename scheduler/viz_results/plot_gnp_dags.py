#!/usr/bin/env python3
"""
Generate a figure of one or more randomly generated DAGs using a G(n, p) style process
for a list of probabilities p (e.g., 0.0, 0.3, 0.5, 0.8), suitable for ablation study visualization.

We construct a DAG by sampling a random permutation of nodes and then adding a directed edge
from i -> j (where i precedes j in the permutation) with probability p. This guarantees acyclicity
and approximates an Erdos–Renyi-like density controlled by p.

A simple layered layout is used based on the longest-path distance from source nodes. This avoids
extra dependencies (like Graphviz) while providing a clean DAG structure visualization.

Usage:
  python scheduler/viz_results/plot_gnp_dags.py \
      --n 16 \
      --p-values 0.0,0.3,0.5,0.8 \
      --seed 42 \
      --out scheduler/viz_results/gnp_dags_multi.png

"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def make_random_dag(n: int, p: float, seed: int | None = None) -> nx.DiGraph:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    pos_in_order = {node: idx for idx, node in enumerate(order)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            # directed from lower order to higher order
            u = order[i]
            v = order[j]
            if rng.random() < p:
                G.add_edge(u, v)
    return G


def longest_path_levels(G: nx.DiGraph) -> Dict[int, int]:
    # Compute a level for each node as the length of the longest path from any source to the node
    topo = list(nx.topological_sort(G))
    level: Dict[int, int] = {u: 0 for u in topo}
    for u in topo:
        for v in G.successors(u):
            level[v] = max(level[v], level[u] + 1)
    return level


def layered_layout(G: nx.DiGraph, h_gap: float = 1.0, v_gap: float = 1.0) -> Dict[int, Tuple[float, float]]:
    level = longest_path_levels(G)
    # group nodes by level
    levels: Dict[int, List[int]] = {}
    for u, l in level.items():
        levels.setdefault(l, []).append(u)
    # sort nodes within each level for deterministic placement
    for l in levels:
        levels[l].sort()

    # assign positions: x by level, y spread within level
    pos: Dict[int, Tuple[float, float]] = {}
    for l, nodes in sorted(levels.items()):
        k = len(nodes)
        if k == 1:
            ys = [0.0]
        else:
            # spread centered around 0
            ys = [v_gap * (i - (k - 1) / 2.0) for i in range(k)]
        x = h_gap * l
        for y, u in zip(ys, nodes):
            pos[u] = (x, y)
    return pos


def draw_dag(ax, G: nx.DiGraph, title: str) -> None:
    pos = layered_layout(G, h_gap=1.5, v_gap=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    # draw edges first
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="#999999", arrows=True, arrowsize=12, width=1.2)
    # draw nodes
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=250, node_color="#1f77b4", alpha=0.9)
    # draw labels as small
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=8, font_color="white")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random G(n,p) DAGs for multiple p-values.")
    parser.add_argument("--n", type=int, default=16, help="Number of nodes")
    parser.add_argument("--p-values", type=str, default="0.3,0.8", help="Comma-separated probabilities, e.g., '0.0,0.3,0.5,0.8'")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--out", type=Path, default=Path("scheduler/viz_results/gnp_dags_multi.png"), help="Output image path")
    args = parser.parse_args()

    p_vals = [float(s.strip()) for s in args.p_values.split(",") if s.strip()]
    if len(p_vals) < 1:
        raise SystemExit("Please provide at least one p-value, e.g., --p-values 0.3 or 0.0,0.3,0.5,0.8")

    # Build one DAG per p, varying the seed slightly for deterministic variety
    dags = [make_random_dag(args.n, p, seed=args.seed + i) for i, p in enumerate(p_vals)]

    # Choose a compact grid: up to 3 columns for readability
    k = len(p_vals)
    cols = min(3, k)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=300)
    if rows == 1 and cols == 1:
        axes_grid = [[axes]]
    elif rows == 1:
        axes_grid = [axes]  # single row list
    else:
        axes_grid = axes
    fig.patch.set_facecolor("white")

    # Draw each DAG
    for idx, (p, G) in enumerate(zip(p_vals, dags)):
        r, c = divmod(idx, cols)
        ax = axes_grid[r][c] if rows > 1 else axes_grid[r][c]
        draw_dag(ax, G, title=f"G(n={args.n}, p={p:.2f})")

    # Hide any unused axes
    total_cells = rows * cols
    for extra in range(k, total_cells):
        r, c = divmod(extra, cols)
        ax = axes_grid[r][c] if rows > 1 else axes_grid[r][c]
        ax.axis("off")

    fig.tight_layout(pad=1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved DAG figure to: {args.out}")


if __name__ == "__main__":
    main()
