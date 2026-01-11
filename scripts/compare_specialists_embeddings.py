#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# Optional plotting deps
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

# Ensure project root (one level up from scripts/) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class CfgPaths:
    longcp_config: str
    wide_config: str


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    # Conservative reuse of logic from eval script: scale per-task demand by job size
    from scheduler.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
    rng = np.random.RandomState(int(seed))
    host_count = int(dataset_cfg.get("host_count", 10))
    vm_count = int(dataset_cfg.get("vm_count", 10))
    max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
    min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
    max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))
    n_tasks = int(dataset_cfg.get("gnp_max_n", 40)) or 1
    hosts = generate_hosts(n=host_count, rng=rng)
    vms = generate_vms(n=vm_count, max_memory_gb=max_memory_gb, min_cpu_speed_mips=min_cpu_speed, max_cpu_speed_mips=max_cpu_speed, rng=rng)
    allocate_vms(vms, hosts, rng)
    mem_caps = [int(getattr(vm, "memory_mb", 0)) for vm in vms]
    core_caps = [int(max(1, getattr(vm, "cpu_cores", 1))) for vm in vms]
    min_mem, max_mem = max(1, min(mem_caps)), max(mem_caps)
    min_cores, max_cores = max(1, min(core_caps)), max(core_caps)
    max_safe_mem_per_task = max(1024, min_mem // n_tasks)
    max_safe_cores_per_task = max(1, min_cores // n_tasks)
    req_div_mem = max(1, max_mem // max_safe_mem_per_task)
    req_div_core = max(1, max_cores // max_safe_cores_per_task)
    return int(max(req_div_mem, req_div_core))


def _dataset_args_from_cfg(dataset_cfg: dict, seed: int, override_req_divisor: int | None = None) -> DatasetArgs:
    req_div = int(override_req_divisor) if override_req_divisor is not None else _compute_optimal_req_divisor(dataset_cfg, seed)
    return DatasetArgs(
        seed=int(seed),
        host_count=int(dataset_cfg.get("host_count", 10)),
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
        style=str(dataset_cfg.get("style", "generic")),
        gnp_p=dataset_cfg.get("gnp_p", None),
        req_divisor=req_div,
    )


def _build_fixed_dataset(dataset_cfg: dict, seed: int, override_req_divisor: int | None = None) -> Any:
    ds_args = _dataset_args_from_cfg(dataset_cfg, seed, override_req_divisor)
    return CloudSchedulingGymEnvironment.gen_dataset(seed, ds_args)


def _load_agent_auto(ckpt: Path, device: torch.device) -> AblationGinAgent:
    var = AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    state = torch.load(str(ckpt), map_location=device)
    hidden_dim = 64
    embedding_dim = 32
    # infer dims from encoder weights if available
    if isinstance(state, dict):
        td = state
        if "actor.network.task_encoder.0.weight" in td:
            hidden_dim = int(td["actor.network.task_encoder.0.weight"].shape[0])
        if "actor.network.task_encoder.6.weight" in td:
            embedding_dim = int(td["actor.network.task_encoder.6.weight"].shape[0])
    agent = AblationGinAgent(device, var, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    if isinstance(state, dict):
        agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def _collect_embeddings_for_domain(
    agent: AblationGinAgent,
    domain_cfg: dict,
    seeds: List[int],
    frames_per_ep: int,
    max_nodes_per_ep: int,
    device: torch.device,
    override_req_divisor: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    node_list: List[np.ndarray] = []
    graph_list: List[np.ndarray] = []
    # Use a subset of seeds for speed
    use_seeds = seeds if len(seeds) <= 10 else list(seeds[:10])
    for s in use_seeds:
        # fixed dataset per seed for determinism
        ds = _build_fixed_dataset(domain_cfg, int(s), override_req_divisor)
        env = GinAgentWrapper(CloudSchedulingGymEnvironment(dataset=ds, collect_timelines=False, compute_metrics=False, profile=False, fixed_env_seed=True))
        obs_np, _ = env.reset(seed=int(s))
        with torch.no_grad():
            frames = 0
            while frames < frames_per_ep:
                x = torch.tensor(obs_np, dtype=torch.float32, device=device)
                go = agent.mapper.unmap(x)
                node_Z, _edge_Z, graph_Z = agent.actor.network(go)
                node_Z_np = node_Z.detach().cpu().numpy()
                if node_Z_np.shape[0] > max_nodes_per_ep:
                    idx = rng.choice(node_Z_np.shape[0], size=max_nodes_per_ep, replace=False)
                    node_Z_np = node_Z_np[idx]
                node_list.append(node_Z_np)
                graph_list.append(graph_Z.detach().cpu().numpy().reshape(1, -1))
                # step greedily to move state
                act, _, _, _ = agent.get_action_and_value(x.unsqueeze(0), deterministic=True)
                nxt, _, term, trunc, _info = env.step(int(act.item()))
                obs_np = nxt
                frames += 1
                if bool(term) or bool(trunc):
                    break
        env.close()
    nodes = np.concatenate(node_list, axis=0) if node_list else np.zeros((0, agent.actor.network.embedding_dim), dtype=np.float32)
    graphs = np.concatenate(graph_list, axis=0) if graph_list else np.zeros((0, agent.actor.network.embedding_dim), dtype=np.float32)
    return nodes, graphs


def _plot_2d_domains_for_agent(
    X_long: np.ndarray,
    X_wide: np.ndarray,
    title: str,
    out_path: Path,
    method: str = "pca",
) -> None:
    """Project one agent's embeddings to 2D and color-code long_cp vs wide jobs.

    X_long, X_wide are embeddings from the SAME agent (node or graph) collected on
    long_cp vs wide domains respectively.
    """
    if plt is None:
        print("[emb] matplotlib not available; skipping plot", title)
        return
    if X_long.size == 0 or X_wide.size == 0:
        print("[emb] One of the domain embedding sets is empty; skipping plot", title)
        return

    # Ensure same feature dimension by zero-padding the smaller one
    d_l = X_long.shape[1]
    d_w = X_wide.shape[1]
    if d_l != d_w:
        d = max(d_l, d_w)
        Xl_pad = np.zeros((X_long.shape[0], d), dtype=X_long.dtype)
        Xw_pad = np.zeros((X_wide.shape[0], d), dtype=X_wide.dtype)
        Xl_pad[:, :d_l] = X_long
        Xw_pad[:, :d_w] = X_wide
        X_long_use, X_wide_use = Xl_pad, Xw_pad
    else:
        X_long_use, X_wide_use = X_long, X_wide

    Z = np.vstack([X_long_use, X_wide_use])
    if method == "tsne" and TSNE is not None:
        projector = TSNE(n_components=2, perplexity=30.0, random_state=0, init="pca")
        Z2 = projector.fit_transform(Z)
    else:
        if PCA is None:
            print("[emb] PCA not available; skipping plot", title)
            return
        projector = PCA(n_components=2, random_state=0)
        Z2 = projector.fit_transform(Z)
    Xl2 = Z2[: X_long_use.shape[0]]
    Xw2 = Z2[X_long_use.shape[0] :]
    plt.figure(figsize=(7, 6), dpi=140)
    plt.scatter(Xl2[:, 0], Xl2[:, 1], s=8, c="#1f77b4", alpha=0.6, label="long_cp jobs")
    plt.scatter(Xw2[:, 0], Xw2[:, 1], s=8, c="#d62728", alpha=0.6, label="wide jobs")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Compare GNN embeddings of long_cp and wide specialists on both domains")
    p.add_argument("--longcp_config", type=str, default="data/rl_configs/train_long_cp_p08_seeds.json")
    p.add_argument("--wide_config", type=str, default="data/rl_configs/train_wide_p005_seeds.json")
    p.add_argument("--longcp_ckpt", type=str, required=True)
    p.add_argument("--wide_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--episodes_per_domain", type=int, default=4)
    p.add_argument("--frames_per_ep", type=int, default=2)
    p.add_argument("--max_nodes_per_ep", type=int, default=256)
    p.add_argument("--dataset_req_divisor", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="logs/embeddings_compare")
    p.add_argument("--tsne", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs and seeds
    cfg_long = json.loads(Path(args.longcp_config).read_text())
    cfg_wide = json.loads(Path(args.wide_config).read_text())
    tr_long = cfg_long.get("train", {})
    tr_wide = cfg_wide.get("train", {})
    seeds_long: List[int] = [int(s) for s in tr_long.get("seeds", [])]
    seeds_wide: List[int] = [int(s) for s in tr_wide.get("seeds", [])]
    ds_long = dict(tr_long.get("dataset", {}))
    ds_wide = dict(tr_wide.get("dataset", {}))

    # Load agents
    long_agent = _load_agent_auto(Path(args.longcp_ckpt), device)
    wide_agent = _load_agent_auto(Path(args.wide_ckpt), device)

    # Collect embeddings: for each domain, collect from both agents
    print("[emb] Collecting embeddings on long_cp domain ...")
    long_nodes_long, long_graphs_long = _collect_embeddings_for_domain(
        long_agent, ds_long, seeds_long[:args.episodes_per_domain], args.frames_per_ep, args.max_nodes_per_ep, device, args.dataset_req_divisor
    )
    wide_nodes_long, wide_graphs_long = _collect_embeddings_for_domain(
        wide_agent, ds_long, seeds_long[:args.episodes_per_domain], args.frames_per_ep, args.max_nodes_per_ep, device, args.dataset_req_divisor
    )

    print("[emb] Collecting embeddings on wide domain ...")
    long_nodes_wide, long_graphs_wide = _collect_embeddings_for_domain(
        long_agent, ds_wide, seeds_wide[:args.episodes_per_domain], args.frames_per_ep, args.max_nodes_per_ep, device, args.dataset_req_divisor
    )
    wide_nodes_wide, wide_graphs_wide = _collect_embeddings_for_domain(
        wide_agent, ds_wide, seeds_wide[:args.episodes_per_domain], args.frames_per_ep, args.max_nodes_per_ep, device, args.dataset_req_divisor
    )

    # Save npy dumps
    np.save(out_dir / "long_agent_nodes_on_long.npy", long_nodes_long)
    np.save(out_dir / "wide_agent_nodes_on_long.npy", wide_nodes_long)
    np.save(out_dir / "long_agent_nodes_on_wide.npy", long_nodes_wide)
    np.save(out_dir / "wide_agent_nodes_on_wide.npy", wide_nodes_wide)

    np.save(out_dir / "long_agent_graph_on_long.npy", long_graphs_long)
    np.save(out_dir / "wide_agent_graph_on_long.npy", wide_graphs_long)
    np.save(out_dir / "long_agent_graph_on_wide.npy", long_graphs_wide)
    np.save(out_dir / "wide_agent_graph_on_wide.npy", wide_graphs_wide)

    # PCA / t-SNE per specialist: long_cp jobs vs wide jobs in that agent's space
    # Long specialist
    _plot_2d_domains_for_agent(
        long_nodes_long,
        long_nodes_wide,
        "Long specialist: node embeddings (long_cp vs wide)",
        out_dir / "long_specialist_nodes_pca.png",
        method="pca",
    )
    _plot_2d_domains_for_agent(
        long_graphs_long,
        long_graphs_wide,
        "Long specialist: graph embeddings (long_cp vs wide)",
        out_dir / "long_specialist_graphs_pca.png",
        method="pca",
    )

    # Wide specialist
    _plot_2d_domains_for_agent(
        wide_nodes_long,
        wide_nodes_wide,
        "Wide specialist: node embeddings (long_cp vs wide)",
        out_dir / "wide_specialist_nodes_pca.png",
        method="pca",
    )
    _plot_2d_domains_for_agent(
        wide_graphs_long,
        wide_graphs_wide,
        "Wide specialist: graph embeddings (long_cp vs wide)",
        out_dir / "wide_specialist_graphs_pca.png",
        method="pca",
    )

    if args.tsne:
        _plot_2d_domains_for_agent(
            long_nodes_long,
            long_nodes_wide,
            "Long specialist: node embeddings (long_cp vs wide, t-SNE)",
            out_dir / "long_specialist_nodes_tsne.png",
            method="tsne",
        )
        _plot_2d_domains_for_agent(
            wide_nodes_long,
            wide_nodes_wide,
            "Wide specialist: node embeddings (long_cp vs wide, t-SNE)",
            out_dir / "wide_specialist_nodes_tsne.png",
            method="tsne",
        )
        _plot_2d_domains_for_agent(
            long_graphs_long,
            long_graphs_wide,
            "Long specialist: graph embeddings (long_cp vs wide, t-SNE)",
            out_dir / "long_specialist_graphs_tsne.png",
            method="tsne",
        )
        _plot_2d_domains_for_agent(
            wide_graphs_long,
            wide_graphs_wide,
            "Wide specialist: graph embeddings (long_cp vs wide, t-SNE)",
            out_dir / "wide_specialist_graphs_tsne.png",
            method="tsne",
        )

    print(f"[emb] Saved embeddings and plots under {out_dir}")


if __name__ == "__main__":
    main()
