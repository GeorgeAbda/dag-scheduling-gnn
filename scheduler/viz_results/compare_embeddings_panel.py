#!/usr/bin/env python3
"""
Generate publication-quality comparison plots of edge embeddings across three architectures:
- Homogeneous GNN (No-Global)
- Heterogeneous GNN 
- MLP-only

Creates two panel figures:
1. VM ID clustering comparison (3x1 subplots)
2. Ready task mask comparison (3x1 subplots)

Each model gets its own t-SNE projection for fair comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def register_spectral_11_cmap():
    """Register the custom 'Spectral-11' colormap from user-provided JSON."""
    spectral_11_rgba = [
        (158/255, 1/255, 66/255, 1),
        (213/255, 62/255, 79/255, 1),
        (244/255, 109/255, 67/255, 1),
        (253/255, 174/255, 97/255, 1),
        (254/255, 224/255, 139/255, 1),
        (255/255, 255/255, 191/255, 1),
        (230/255, 245/255, 152/255, 1),
        (171/255, 221/255, 164/255, 1),
        (102/255, 194/255, 165/255, 1),
        (50/255, 136/255, 189/255, 1),
        (94/255, 79/255, 162/255, 1),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('Spectral-11', spectral_11_rgba)
    plt.colormaps.register(cmap)


register_spectral_11_cmap()


# Local imports from project
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as AblArgs,
    _make_test_env,
)
from scheduler.dataset_generator.gen_dataset import DatasetArgs


# ----------------------------
# Configuration and helpers
# ----------------------------

def build_args_gnp(p: float, hosts: int, vms: int, workflows: int, min_tasks: int, max_tasks: int, device: str) -> AblArgs:
    """Build evaluation arguments for GNP dataset."""
    a = AblArgs()
    a.device = device
    a.dataset = DatasetArgs(
        host_count=hosts,
        vm_count=vms,
        workflow_count=workflows,
        dag_method="gnp",
        gnp_min_n=min_tasks,
        gnp_max_n=max_tasks,
        gnp_p=p,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        min_task_length=500,
        max_task_length=100_000,
        task_arrival="static",
    )
    a.test_iterations = 1
    return a


def pick_variant(name: str) -> AblationVariant:
    """Create variant configuration by name."""
    n = name.strip().lower()    
    if n in ("noglobal", "no_global_actor", "no_global"):
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    elif n == "homo":
        return AblationVariant(name="homogeneous")
    elif n in ("mlp_only", "mlp"):
        return AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False)
    elif n == "hetero":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    elif n == "hetero_noglobal":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True, use_actor_global_embedding=False)
    else:
        return AblationVariant(name="baseline")


def load_agent(ckpt_path: str, device: torch.device, variant_name: str) -> AblationGinAgent:
    """Load trained agent from checkpoint."""
    var = pick_variant(variant_name)
    # Use embedding_dim=16 for hetero, 8 for others (matching training setup)
    emb_dim = 8
    agent = AblationGinAgent(device, var, embedding_dim=emb_dim)
    state = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def extract_embeddings_and_labels(agent: AblationGinAgent, args: AblArgs, episodes: int = 3, seed_base: int = 1_000_000_001) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract edge embeddings and labels across multiple episodes (snapshots).
    Set episodes=1 for a single state. Use seed_base to control which state(s) are sampled.
    """
    X_list = []
    lab_collect = []
    
    for episode in range(episodes):
        env = _make_test_env(args)
        obs_np, _ = env.reset(seed=seed_base + episode)
        obs_tensor = torch.tensor(np.asarray(obs_np, dtype=np.float32))
        obs = agent.mapper.unmap(obs_tensor)
        print(f'we are extracting embeddings for episode {episode}')
        # Get edge embeddings
        with torch.no_grad():
            _, edge_embeddings, _ = agent.actor.network(obs)
        
        E = int(obs.compatibilities.shape[1])
        edge_emb = edge_embeddings[:E].cpu().numpy()
        
        # Extract labels
        task_ids = obs.compatibilities[0][:E].cpu().numpy().astype(int)
        vm_ids = obs.compatibilities[1][:E].cpu().numpy().astype(int)
        ready_mask = obs.task_state_ready.cpu().numpy().astype(bool)
        
        labels = {
            'vm': vm_ids,
            'task': task_ids,
            'ready': ready_mask[task_ids],
        }
        
        X_list.append(edge_emb)
        lab_collect.append(labels)
        env.close()
    
    # Concatenate across episodes
    X = np.concatenate(X_list, axis=0)
    combined_labels = {
        'vm': np.concatenate([d['vm'] for d in lab_collect], axis=0),
        'task': np.concatenate([d['task'] for d in lab_collect], axis=0),
        'ready': np.concatenate([d['ready'] for d in lab_collect], axis=0),
    }
    
    return X, combined_labels


def compute_tsne(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Compute t-SNE projection with appropriate perplexity."""
    n = X.shape[0]
    perplexity = max(5, min(30, n // 3))
    # Use only widely supported TSNE parameters for sklearn compatibility
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='random',
        learning_rate='auto',
        random_state=random_state,
    )
    return tsne.fit_transform(X)


# ----------------------------
# Single-state probability coloring
# ----------------------------

def extract_state_embeddings_and_probs(agent: AblationGinAgent, args: AblArgs, seed: int = 1_000_000_001,
                                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one environment state, return:
    - X: edge embeddings for compatible task→VM edges (E, D)
    - Y: t-SNE projection of X (E, 2)
    - P: per-edge action probabilities under the agent's actor for this state (E,)

    We mirror the actor forward pass to compute edge scores, then apply the same readiness/scheduled mask
    the actor uses when building the action-score matrix. Probabilities are softmax-normalized across valid edges.
    """
    # Build one test environment and get a single observation
    env = _make_test_env(args)
    try:
        obs_np, _ = env.reset(seed=seed)
        obs_tensor = torch.tensor(np.asarray(obs_np, dtype=np.float32))
        obs = agent.mapper.unmap(obs_tensor)

        with torch.no_grad():
            # Get raw edge embeddings and graph embedding from the actor's backbone
            _, edge_embeddings, graph_embedding = agent.actor.network(obs)
            E = int(obs.compatibilities.shape[1])
            edge_embeddings = edge_embeddings[:E]

            # Construct the scorer input exactly as in actor.forward
            scorer_in = edge_embeddings
            if getattr(agent.variant, 'use_actor_global_embedding', False):
                rep_graph = graph_embedding.expand(edge_embeddings.shape[0], agent.actor.embedding_dim)
                scorer_in = torch.cat([scorer_in, rep_graph], dim=1)

            # Edge scores (logits) for the compatible edges only
            scores = agent.actor.edge_scorer(scorer_in).flatten()
            print(f'we have {scores.shape[0]} scores')
            # Apply readiness/scheduled mask at edge level (actor masks per-task rows to -1e8)
            t_idx = obs.compatibilities[0][:E].to(torch.long)
            valid_task_mask = (obs.task_state_ready[t_idx] == 1) & (obs.task_state_scheduled[t_idx] == 0)
            masked_scores = scores.clone()
            masked_scores[~valid_task_mask] = -1e8
            print(f'we have {masked_scores.shape[0]} masked scores')
            # Probabilities across valid edges
            probs = torch.softmax(masked_scores, dim=0)
            P = probs.cpu().numpy()

        # t-SNE for this single state
        X = edge_embeddings.cpu().numpy()
        print(f'we will compute the t-SNE for {X.shape[0]} embeddings')
        Y = compute_tsne(X, random_state=random_state)
        try:
            _model_name = getattr(getattr(agent, 'variant', None), 'name', 'model')
        except Exception:
            _model_name = 'model'
        print(f'Computing probability plot for {_model_name}')
        return X, Y, P
    finally:
        try:
            env.close()
        except Exception:
            pass


# ----------------------------
# Publication-quality plotting
# ----------------------------

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    # Match the style from plot_models_comparison.py
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    # Customizations for panel plots
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False


def plot_vm_comparison(results: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]], 
                      out_path: Path, cmap_name: str = 'Spectral-11'):
    """Create VM ID clustering comparison plot."""
    setup_publication_style()
    # Build a consistent VM color map across all models
    all_vm_ids: list[int] = []
    for key, (X, Y, labels) in results.items():
        all_vm_ids.extend(list(np.unique(labels['vm'])))
    uniq_vm = sorted(list(set(all_vm_ids)))
    try:
        base_cmap = plt.colormaps.get_cmap(cmap_name)(np.linspace(0, 1, max(1, len(uniq_vm))))
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Falling back to 'tab20'.")
        base_cmap = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, max(1, len(uniq_vm))))
    vm_color_map = {vm_id: base_cmap[i % len(base_cmap)] for i, vm_id in enumerate(uniq_vm)}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    model_names = ['Homogeneous GNN', 'Heterogeneous GNN', 'MLP-only']
    model_keys = ['homogeneous', 'heterogeneous', 'mlp']
    panel_labels = ['(a)', '(b)', '(c)']
    
    for i, (ax, model_name, key) in enumerate(zip(axes, model_names, model_keys)):
        if key not in results:
            ax.text(0.5, 0.5, f'{model_name}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
            
        X, Y, labels = results[key]
        vm_ids = labels['vm']
        
        # Compute silhouette score
        try:
            sil_score = silhouette_score(X, vm_ids)
            sil_text = f'Silhouette: {sil_score:.3f}'
        except:
            sil_text = 'Silhouette: N/A'
        
        # Create scatter plot with consistent VM colors
        unique_vms = np.unique(vm_ids)
        for vm_id in unique_vms:
            mask = vm_ids == vm_id
            color = vm_color_map.get(int(vm_id), (0.3, 0.3, 0.3, 1.0))
            ax.scatter(Y[mask, 0], Y[mask, 1], c=[color], s=9, alpha=0.75,
                       label=f'VM {vm_id}', edgecolors='none')
        
        # Panel label and title
        ax.text(0.01, 0.98, panel_labels[i], transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a compact legend below the row for readability (handled after loop)

    # Create a shared legend using the global VM palette
    legend_patches = [mpatches.Patch(color=vm_color_map[vm], label=f'VM {vm}') for vm in uniq_vm[:10]]
    if legend_patches:
        fig.legend(handles=legend_patches, loc='lower center', ncol=min(10, len(legend_patches)), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved VM comparison plot: {out_path}")


def plot_ready_comparison(results: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]], 
                         out_path: Path):
    """Create ready task mask comparison plot."""
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    model_names = ['Homogeneous GNN', 'Heterogeneous GNN', 'MLP-only']
    model_keys = ['homogeneous', 'heterogeneous', 'mlp']
    panel_labels = ['(a)', '(b)', '(c)']
    
    for i, (ax, model_name, key) in enumerate(zip(axes, model_names, model_keys)):
        if key not in results:
            ax.text(0.5, 0.5, f'{model_name}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
            
        X, Y, labels = results[key]
        ready_mask = labels['ready']
        
        # Plot ready vs not ready tasks
        ready_idx = ready_mask == True
        not_ready_idx = ready_mask == False
        
        if np.any(not_ready_idx):
            ax.scatter(Y[not_ready_idx, 0], Y[not_ready_idx, 1], 
                      c='#E14434', s=9, alpha=0.65, label='Not Ready', edgecolors='none')
        
        if np.any(ready_idx):
            ax.scatter(Y[ready_idx, 0], Y[ready_idx, 1], 
                      c='#5EABD6', s=9, alpha=0.8, label='Ready', edgecolors='none')
        
        # Compute separation metric
        if np.any(ready_idx) and np.any(not_ready_idx):
            try:
                sil_score = silhouette_score(X, ready_mask.astype(int))
                sep_text = f'Silhouette: {sil_score:.3f}'
            except:
                sep_text = 'Silhouette: N/A'
        else:
            sep_text = 'Single class'
        
        # Panel label and title
        ax.text(0.01, 0.98, panel_labels[i], transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left')
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Shared legend centered below
    handles = [mpatches.Patch(color='#5EABD6', label='Ready'), mpatches.Patch(color='#E14434', label='Not Ready')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ready task comparison plot: {out_path}")


# ----------------------------
# Probability-colored panel
# ----------------------------

def plot_probability_comparison(results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                out_path: Path,
                                cmap: str = 'viridis'):
    """Create a 3x1 panel where each subplot shows one model's edge embeddings colored by
    the per-edge action probability in a single chosen state.
    """
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    model_names = ['Homogeneous GNN', 'Heterogeneous GNN', 'MLP-only']
    model_keys = ['homogeneous', 'heterogeneous', 'mlp']
    panel_labels = ['(a)', '(b)', '(c)']

    # Shared color normalization across models
    vmax = 0.0
    for key in model_keys:
        if key in results:
            _X, _Y, P = results[key]
            if P.size > 0:
                vmax = max(vmax, float(np.max(P)))
    norm = mcolors.Normalize(vmin=0.0, vmax=max(1e-9, vmax))
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(cmap), norm=norm)
    sm.set_array([])

    for i, (ax, name, key) in enumerate(zip(axes, model_names, model_keys)):
        if key not in results:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        X, Y, P = results[key]
        colors = plt.colormaps.get_cmap(cmap)(norm(P))
        # Match scatter style with other panels (size, alpha, no edge)
        ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=9, alpha=0.8, edgecolors='none')
        ax.text(0.01, 0.98, panel_labels[i], transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left')
        ax.set_xticks([]); ax.set_yticks([])

    # Place a shared horizontal colorbar below the row to match the legend placement style
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.2, fraction=0.05, aspect=30)
    cbar.set_label('Action probability')
    # Align layout margins with other panels (space for bottom bar)
    plt.tight_layout(rect=[0, 0.15, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved probability-colored comparison plot: {out_path}")


# ----------------------------
# Main execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality embedding comparison plots')
    
    # Model paths
    parser.add_argument('--homogeneous-model', type=str, required=True,
                       help='Path to homogeneous (no-global) model checkpoint')
    parser.add_argument('--heterogeneous-model', type=str, required=True,
                       help='Path to heterogeneous model checkpoint')
    parser.add_argument('--mlp-model', type=str, required=True,
                       help='Path to MLP-only model checkpoint')
    
    # Dataset parameters
    parser.add_argument('--gnp-p', type=float, default=0.8, help='GNP edge probability')
    parser.add_argument('--hosts', type=int, default=4, help='Number of hosts')
    parser.add_argument('--vms', type=int, default=10, help='Number of VMs')
    parser.add_argument('--workflows', type=int, default=10, help='Number of workflows')
    parser.add_argument('--min-tasks', type=int, default=12, help='Minimum tasks per workflow')
    parser.add_argument('--max-tasks', type=int, default=24, help='Maximum tasks per workflow')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to sample')
    
    # Output parameters
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--out-dir', type=Path, default=Path('csv_ablation/gnp_0.8/plots/comparison'),
                       help='Output directory for plots')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for t-SNE')
    parser.add_argument('--panel-seed-base', type=int, default=1_000_000_001, help='Seed base for VM/Ready panels; with --episodes 1, the state seed is exactly this value')
    parser.add_argument('--prob-seed', type=int, default=1_000_000_001, help='Seed for the single state used in probability coloring')
    parser.add_argument('--vm-cmap', type=str, default='Spectral-11', help='Matplotlib colormap for VM coloring')
    parser.add_argument('--prob-cmap', type=str, default='Spectral-11', help='Matplotlib colormap for probability coloring')
    
    args = parser.parse_args()
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup evaluation arguments
    device = torch.device(args.device)
    eval_args = build_args_gnp(args.gnp_p, args.hosts, args.vms, args.workflows, 
                              args.min_tasks, args.max_tasks, args.device)
    
    # Load models and extract embeddings
    results = {}
    
    model_configs = [
        ('homogeneous', args.homogeneous_model, 'homogeneous'),
        ('heterogeneous', args.heterogeneous_model, 'hetero'),
        ('mlp', args.mlp_model, 'mlp_only')
    ]
    
    for key, model_path, variant in model_configs:
        if model_path and Path(model_path).exists():
            print(f"Processing {key} model...")
            try:
                agent = load_agent(model_path, device, variant)
                X, labels = extract_embeddings_and_labels(agent, eval_args, args.episodes, seed_base=args.panel_seed_base)
                Y = compute_tsne(X, random_state=args.random_seed)
                results[key] = (X, Y, labels)
                print(f"  Extracted {X.shape[0]} edge embeddings")
            except Exception as e:
                print(f"  Error processing {key} model: {e}")
        else:
            print(f"Skipping {key} model (path not found: {model_path})")
    
    if not results:
        print("No models successfully loaded. Exiting.")
        return
    
    # Generate comparison plots
    vm_plot_path = args.out_dir / f"embedding_comparison_vm_gnp{args.gnp_p:.1f}.pdf"
    ready_plot_path = args.out_dir / f"embedding_comparison_ready_gnp{args.gnp_p:.1f}.pdf"

    plot_vm_comparison(results, vm_plot_path, cmap_name=args.vm_cmap)
    plot_ready_comparison(results, ready_plot_path)

    # Probability-colored plot: single shared state across models
    prob_results = {}
    for key, model_path, variant in model_configs:
        if model_path and Path(model_path).exists():
            try:
                agent = load_agent(model_path, device, variant)
                Xp, Yp, P = extract_state_embeddings_and_probs(agent, eval_args, seed=args.prob_seed, random_state=args.random_seed)
                print(f"  Extracted {Xp.shape[0]} edge embeddings for probability plot")
                prob_results[key] = (Xp, Yp, P)
            except Exception as e:
                print(f"  Error computing probabilities for {key}: {e}")
        else:
            print(f"Skipping probability plot for {key} (path not found: {model_path})")
    print(f'we have {len(prob_results)} models for probability comparison')
    if prob_results:
        prob_plot_path = args.out_dir / f"embedding_comparison_prob_gnp{args.gnp_p:.1f}_seed{args.prob_seed}.pdf"
        plot_probability_comparison(prob_results, prob_plot_path, cmap=args.prob_cmap)

    print(f"\nComparison plots saved to: {args.out_dir}")
    print("Files generated:")
    print(f"  - {vm_plot_path.name}")
    print(f"  - {ready_plot_path.name}")
    if 'prob_plot_path' in locals():
        print(f"  - {prob_plot_path.name}")


if __name__ == '__main__':
    main()
