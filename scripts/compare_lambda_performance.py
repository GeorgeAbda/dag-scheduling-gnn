#!/usr/bin/env python3
"""
Compare in-domain scheduling performance across lambda models.

Evaluates each lambda model on the training domain (e.g., n=24, p=0.8) and collects:
- Makespan
- Total energy (active + idle)
- Active energy
- Idle energy

Outputs:
- CSV with metrics per lambda
- Bar plots comparing metrics

Example:
  python -m scripts.compare_lambda_performance \
    --architecture hetero \
    --lambda-models "0:logs/.../hetero_best_model.pt,1e-5:logs/.../hetero_best_model.pt,1e-4:logs/.../hetero_best_model.pt,1e-3:logs/.../hetero_best_model.pt" \
    --n-min 24 --n-max 24 \
    --p 0.8 \
    --hosts 4 --vms 10 --workflows 10 \
    --episodes 10 \
    --device cpu \
    --out-dir scheduler/viz_results/decision_boundaries/lambda_performance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
_proj_root = Path(__file__).resolve().parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from cogito.viz_results.compare_embeddings_panel import load_agent, build_args_gnp
from cogito.gnn_deeprl_model.ablation_gnn import _test_agent, AblationGinAgent


def load_lambda_models(lambda_paths_str: str, device: str, architecture: str = "hetero") -> Dict[float, AblationGinAgent]:
    """Parse 'lambda:path,lambda:path,...' and load agents."""
    models = {}
    for entry in lambda_paths_str.split(','):
        lam_str, path_str = entry.split(':', 1)
        lam = float(lam_str)
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: checkpoint not found for lambda={lam}: {path}")
            continue
        agent = load_agent(str(path), device=device, variant_name=architecture)
        models[lam] = agent
        print(f"[load] Lambda={lam}: {path}")
    return models


def evaluate_agent_metrics(agent: AblationGinAgent, eval_args, episodes: int, seed_base: int = 1_000_000_001) -> Dict:
    """Run agent for multiple episodes and return mean metrics."""
    makespans = []
    total_energies = []
    active_energies = []
    idle_energies = []
    
    for ep in range(episodes):
        eval_args.seed = seed_base + ep
        try:
            mk, eobs, etot, mm = _test_agent(agent, eval_args)
            ea = (mm or {}).get('avg_active_energy', 0.0)
            ei = (mm or {}).get('avg_idle_energy', 0.0)
            
            makespans.append(mk)
            total_energies.append(etot if etot > 0 else eobs)
            active_energies.append(ea)
            idle_energies.append(ei)
        except Exception as e:
            print(f"[warn] Episode {ep} failed: {e}")
            continue
    
    if not makespans:
        return {
            'makespan_mean': np.nan, 'makespan_std': np.nan,
            'total_energy_mean': np.nan, 'total_energy_std': np.nan,
            'active_energy_mean': np.nan, 'active_energy_std': np.nan,
            'idle_energy_mean': np.nan, 'idle_energy_std': np.nan,
        }
    
    return {
        'makespan_mean': float(np.mean(makespans)),
        'makespan_std': float(np.std(makespans)),
        'total_energy_mean': float(np.mean(total_energies)),
        'total_energy_std': float(np.std(total_energies)),
        'active_energy_mean': float(np.mean(active_energies)),
        'active_energy_std': float(np.std(active_energies)),
        'idle_energy_mean': float(np.mean(idle_energies)),
        'idle_energy_std': float(np.std(idle_energies)),
    }


def plot_comparison(df: pd.DataFrame, out_dir: Path):
    """Generate bar plots comparing metrics across lambda."""
    sns.set_style("whitegrid")
    metrics = [
        ('makespan', 'Makespan'),
        ('total_energy', 'Total Energy (Active + Idle)'),
        ('active_energy', 'Active Energy'),
        ('idle_energy', 'Idle Energy'),
    ]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    for ax, (metric, title) in zip(axes, metrics):
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        x = df['lambda'].astype(str)
        y = df[mean_col]
        yerr = df[std_col]
        ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.8, edgecolor='black')
        ax.set_xlabel('λ (low-pass regularization)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} (mean±std)')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    out_path = out_dir / 'lambda_performance_comparison.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare in-domain performance across lambda models')
    parser.add_argument('--architecture', type=str, default='hetero', help='Architecture name')
    parser.add_argument('--lambda-models', type=str, required=True,
                       help='Comma-separated lambda:path pairs, e.g., "0:path1,1e-5:path2"')
    
    # Evaluation domain (training domain)
    parser.add_argument('--n-min', type=int, default=24, help='Min tasks (training domain)')
    parser.add_argument('--n-max', type=int, default=24, help='Max tasks (training domain)')
    parser.add_argument('--p', type=float, default=0.8, help='Edge probability (training domain)')
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per model for mean/std')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out-dir', type=str, default='scheduler/viz_results/decision_boundaries/lambda_performance')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    models = load_lambda_models(args.lambda_models, args.device, args.architecture)
    if not models:
        print("Error: no models loaded")
        return
    
    # Build evaluation args (training domain)
    eval_args = build_args_gnp(
        args.p, args.hosts, args.vms, args.workflows,
        args.n_min, args.n_max, args.device
    )
    
    # Evaluate each lambda model
    rows = []
    for lam, agent in sorted(models.items()):
        print(f"\n[eval] Lambda={lam}")
        metrics = evaluate_agent_metrics(agent, eval_args, args.episodes)
        metrics['lambda'] = lam
        rows.append(metrics)
        print(f"  Makespan: {metrics['makespan_mean']:.2f} ± {metrics['makespan_std']:.2f}")
        print(f"  Total Energy: {metrics['total_energy_mean']:.2f} ± {metrics['total_energy_std']:.2f}")
        print(f"  Active Energy: {metrics['active_energy_mean']:.2f} ± {metrics['active_energy_std']:.2f}")
        print(f"  Idle Energy: {metrics['idle_energy_mean']:.2f} ± {metrics['idle_energy_std']:.2f}")
    
    # Save CSV
    df = pd.DataFrame(rows)
    df = df[['lambda', 'makespan_mean', 'makespan_std', 'total_energy_mean', 'total_energy_std',
             'active_energy_mean', 'active_energy_std', 'idle_energy_mean', 'idle_energy_std']]
    csv_path = out_dir / 'lambda_performance_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n[csv] Saved: {csv_path}")
    
    # Plot
    plot_comparison(df, out_dir)
    
    print(f"\n[done] Results in: {out_dir}")


if __name__ == '__main__':
    main()
