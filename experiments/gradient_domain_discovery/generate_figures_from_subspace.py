"""
Generate publication figures from Gradient Subspace Discovery results.

Outputs (all to experiments/gradient_domain_discovery/figures/):
- fig5_theoretical_validation.png
- gradient_flow_HETERO.png
- gradient_subspace_pca_HETERO.png
- gradient_trajectory.png (uses multiple training checkpoints if available)

Usage:
  python experiments/gradient_domain_discovery/generate_figures_from_subspace.py \
      --host-specs data/host_specs_NAL.json \
      --model logs/generalist_mixed/generalist_final.pt \
      --results logs/gradient_subspace/subspace_results.json \
      [--checkpoints-dir logs/generalist_mixed] \
      [--objectives 20] [--replicates 2] [--steps 192]

Notes:
- If subspace artifacts are not available, this script will recompute gradients
  for the specified MDP set using the provided model to fit PCA and generate plots.
- It will be slower on CPU.
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# White background, publication style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titleweight': 'bold'
})

# Reference multi‑hue palette
PALETTE = [
    '#00FFFF',  # cyan
    '#FF00FF',  # magenta
    '#32CD32',  # lime
    '#FFA500',  # orange
    '#8A2BE2',  # violet
    '#008080',  # teal
    '#FF0000',  # red
    '#0000FF',  # blue
]

# Ensure repository root is on sys.path so we can import sibling modules
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import helpers from the discovery script
from gradient_subspace_discovery import (
    AblationGinAgent, AblationVariant,
    create_mdp_from_params,
    compute_gradient_for_objective_weights,
)


def load_agent(model_path: str, device: torch.device):
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)

    # Initialize lazy params with dummy obs
    dummy_env, _ = create_mdp_from_params([0.5, 20, 30, 1000, 5000], os.environ.get('HOST_SPECS_PATH', 'data/host_specs_NAL.json'))
    obs, _ = dummy_env.reset(seed=42)
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_t)
    dummy_env.close()

    # Load weights
    state = torch.load(model_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def collect_gradients(agent, mdp_configs, host_specs_file, objectives, replicates=2, steps=192, device=torch.device('cpu')):
    all_gradients = []
    gradient_matrix = []

    pbar = tqdm(mdp_configs, desc='Collecting gradients', ncols=100)
    for mdp_idx, cfg in enumerate(pbar):
        params = [cfg['edge_prob'], cfg['min_tasks'], cfg['max_tasks'], cfg['min_length'], cfg['max_length']]
        env, _ = create_mdp_from_params(params, host_specs_file)

        mdp_grads = []
        for obj_idx, (am, ae) in enumerate(objectives):
            grads = []
            for r in range(replicates):
                g = compute_gradient_for_objective_weights(
                    agent, env, am, ae, num_steps=steps, device=device,
                    seed=12345 + mdp_idx * 1000 + obj_idx * 10 + r
                )
                if g is not None:
                    grads.append(g)
            if grads:
                gmean = np.mean(np.stack(grads, 0), 0)
                mdp_grads.append(gmean)
                all_gradients.append(gmean)
        env.close()
        if mdp_grads:
            gradient_matrix.append(np.stack(mdp_grads, 0))

    return np.array(all_gradients), np.array(gradient_matrix)


def make_fig5_theoretical_validation(figdir, explained_variance, true_labels, clusters, edge_probs, sil_score):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # EV bar
    ax = axes[0, 0]
    ax.bar(["PC1", "PC2"], explained_variance, color=[PALETTE[5], PALETTE[4]])
    ax.set_ylim(0, 1)
    ax.set_title('Explained Variance (PCA)')

    # Confusion matrix
    ax = axes[0, 1]
    cm = confusion_matrix(true_labels, clusters)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title('Confusion Matrix (True vs Pred)')
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Label')

    # Cluster vs edge probability
    ax = axes[1, 0]
    x = np.array(edge_probs)
    y = np.array(clusters)
    ax.scatter(x[y == 0], np.zeros_like(x[y == 0]), color=PALETTE[2], alpha=0.8, label='Cluster 0')
    ax.scatter(x[y == 1], np.ones_like(x[y == 1]), color=PALETTE[6], alpha=0.8, label='Cluster 1')
    ax.set_yticks([0, 1])
    ax.set_xlabel('Edge Probability')
    ax.set_title('Clusters vs Edge Probability')
    ax.legend()

    # Silhouette
    ax = axes[1, 1]
    ax.bar(['Silhouette'], [sil_score], color=PALETTE[1])
    ax.set_ylim(0, 1)
    ax.set_title('Silhouette Score')

    plt.tight_layout()
    out = figdir / 'fig5_theoretical_validation.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def make_gradient_subspace_pca(figdir, pca, all_gradients, gradient_matrix, random_objectives, true_labels):
    # Projections of all gradients
    proj = pca.transform(all_gradients)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by alpha_makespan
    ax = axes[0]
    alphas_m = np.array([am for (am, ae) in random_objectives for _ in range(gradient_matrix.shape[0])])
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=alphas_m, cmap='viridis', s=18, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('alpha_makespan')
    ax.set_title('All Gradients in PCA Space (colored by alpha_makespan)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Right: per-MDP mean projections colored by true label
    ax = axes[1]
    # Mean across objectives per MDP
    mdp_means = pca.transform(gradient_matrix.mean(axis=1))
    colors = np.where(np.array(true_labels) == 0, PALETTE[2], PALETTE[6])
    ax.scatter(mdp_means[:, 0], mdp_means[:, 1], c=colors, s=80, alpha=0.9, edgecolors='black', linewidths=0.6)
    ax.set_title('MDP Mean Gradient Projections')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    plt.tight_layout()
    out = figdir / 'gradient_subspace_pca_HETERO.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def make_gradient_flow(figdir, pca, gradient_matrix, objectives, edge_probs):
    # Need projections for the extremes only (energy, makespan)
    # Find indices of (1,0) and (0,1) in objectives
    idx_makespan = None
    idx_energy = None
    for i, (am, ae) in enumerate(objectives):
        if abs(am - 1.0) < 1e-6 and abs(ae - 0.0) < 1e-6:
            idx_makespan = i
        if abs(am - 0.0) < 1e-6 and abs(ae - 1.0) < 1e-6:
            idx_energy = i
    if idx_makespan is None or idx_energy is None:
        raise RuntimeError('Objectives must include (1,0) and (0,1) for flow plot')

    grad_ms = gradient_matrix[:, idx_makespan, :]
    grad_en = gradient_matrix[:, idx_energy, :]

    proj_ms = pca.transform(grad_ms)
    proj_en = pca.transform(grad_en)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color by edge probability
    edge_probs = np.array(edge_probs)
    norm = (edge_probs - edge_probs.min()) / (edge_probs.ptp() + 1e-9)
    colors = plt.cm.plasma(norm)

    # Draw arrows from energy to makespan for each MDP
    for i in range(len(edge_probs)):
        x0, y0 = proj_en[i]
        dx, dy = (proj_ms[i] - proj_en[i])
        ax.arrow(x0, y0, dx, dy, head_width=0.02, head_length=0.03, length_includes_head=True,
                 color=colors[i], alpha=0.8)

    ax.set_title('Gradient Flow in PCA Space (energy → makespan)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    sm = plt.cm.ScalarMappable(cmap='plasma')
    sm.set_array(edge_probs)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Edge Probability')

    plt.tight_layout()
    out = figdir / 'gradient_flow_HETERO.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def find_checkpoints(checkpoints_dir: Path):
    if not checkpoints_dir.exists():
        return []
    cps = []
    for p in checkpoints_dir.glob('checkpoint_*.pt'):
        m = re.search(r'checkpoint_(\d+)\.pt', p.name)
        if m:
            cps.append((int(m.group(1)), p))
    cps.sort()
    return cps


def make_gradient_trajectory(figdir, host_specs_file, mdp_configs, pca, objectives, checkpoints_dir, device):
    cps = find_checkpoints(Path(checkpoints_dir))
    if not cps:
        print('[warn] No checkpoints found, will plot trajectory using final model only')
        cps = []

    # Add final model as last point if not already present
    # The caller will pass the final model separately for the other figures

    # Choose two representative MDPs: lowest and highest edge_prob
    idx_low = int(np.argmin([c['edge_prob'] for c in mdp_configs]))
    idx_high = int(np.argmax([c['edge_prob'] for c in mdp_configs]))
    chosen = [idx_low, idx_high]

    # Use pure objectives for trajectory
    traj_objectives = [(1.0, 0.0), (0.0, 1.0)]  # makespan, energy

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, mdp_idx in enumerate(chosen):
        ax = axes[ax_idx]
        params = [
            mdp_configs[mdp_idx]['edge_prob'],
            mdp_configs[mdp_idx]['min_tasks'],
            mdp_configs[mdp_idx]['max_tasks'],
            mdp_configs[mdp_idx]['min_length'],
            mdp_configs[mdp_idx]['max_length'],
        ]
        env, _ = create_mdp_from_params(params, host_specs_file)

        # Iterate through checkpoints chronologically
        steps_list = []
        proj_points = {obj: [] for obj in traj_objectives}

        for steps, ckpt_path in cps:
            agent = load_agent(str(ckpt_path), device)
            for obj in traj_objectives:
                g = compute_gradient_for_objective_weights(agent, env, obj[0], obj[1], num_steps=160, device=device, seed=7777)
                proj = pca.transform(g.reshape(1, -1))[0]
                proj_points[obj].append(proj)
            steps_list.append(steps)

        env.close()

        # Plot trajectories
        for obj, color in zip(traj_objectives, [PALETTE[6], PALETTE[2]]):  # red/blue-ish
            pts = np.array(proj_points[obj])
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], '-o', color=color, alpha=0.9, label=f'alpha=({obj[0]:.1f},{obj[1]:.1f})')

        title = f'MDP #{mdp_idx+1} (edge_p={mdp_configs[mdp_idx]["edge_prob"]:.2f})'
        ax.set_title(f'Gradient Trajectory over Training\n{title}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = figdir / 'gradient_trajectory.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-specs', type=str, required=True)
    ap.add_argument('--model', type=str, required=True, help='Path to final trained model (.pt)')
    ap.add_argument('--results', type=str, required=True, help='Path to subspace_results.json')
    ap.add_argument('--checkpoints-dir', type=str, default='logs/generalist_mixed')
    ap.add_argument('--objectives', type=int, default=20)
    ap.add_argument('--replicates', type=int, default=2)
    ap.add_argument('--steps', type=int, default=192)
    args = ap.parse_args()

    os.environ['HOST_SPECS_PATH'] = args.host_specs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load subspace results for metadata
    with open(args.results, 'r') as f:
        subres = json.load(f)
    mdp_configs = subres['mdp_configs']
    true_labels = np.array(subres['true_labels'])
    clusters = np.array(subres['clusters'])
    explained_variance = np.array(subres['explained_variance'])
    edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
    sil_score = float(subres.get('silhouette', 0.0))

    # Prepare objectives (include extremes for flow)
    rng = np.random.default_rng(42)
    alphas = rng.uniform(0, 1, size=args.objectives - 2)  # reserve 2 for extremes
    objectives = [(1.0, 0.0), (0.0, 1.0)] + [(float(a), float(1-a)) for a in alphas]

    # Load agent
    agent = load_agent(args.model, device)

    # Collect gradients
    print('[i] Collecting gradients to fit PCA for figures...')
    all_gradients, gradient_matrix = collect_gradients(
        agent, mdp_configs, args.host_specs, objectives,
        replicates=args.replicates, steps=args.steps, device=device
    )

    # Fit PCA on all gradients
    pca = PCA(n_components=2)
    pca.fit(all_gradients)
    print(f'[i] PCA EV ratio: {pca.explained_variance_ratio_}')

    # Output dir
    figdir = Path('experiments/gradient_domain_discovery/figures')
    figdir.mkdir(parents=True, exist_ok=True)

    # Make figures
    p1 = make_fig5_theoretical_validation(figdir, explained_variance, true_labels, clusters, edge_probs, sil_score)
    p2 = make_gradient_subspace_pca(figdir, pca, all_gradients, gradient_matrix, objectives, true_labels)
    p3 = make_gradient_flow(figdir, pca, gradient_matrix, objectives, edge_probs)
    p4 = make_gradient_trajectory(figdir, args.host_specs, mdp_configs, pca, objectives, args.checkpoints_dir, device)

    print('\nSaved:')
    for p in [p1, p2, p3, p4]:
        print('  -', p)


if __name__ == '__main__':
    main()
