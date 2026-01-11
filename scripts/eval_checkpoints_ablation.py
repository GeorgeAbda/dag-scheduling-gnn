#!/usr/bin/env python3
"""
Evaluate training checkpoints for ablation variants and (optionally) compare
best models across variants.

This script supersedes scripts/eval_checkpoints_grokking.py. It generalizes the
use-case to ablation evaluation, plotting learning progress across checkpoints,
and providing an option to compare the best saved models from different
variants (e.g., baseline vs no_global_actor, etc.).

Outputs:
- A CSV summary with metrics per checkpoint and config
- Plots of metric vs iteration to visualize improvement during training
- Optional CSV/plots for best-models comparison across variants

Example usages:

1) Evaluate a series of checkpoints for a single variant

  python scripts/eval_checkpoints_ablation.py \
    --checkpoints "logs/.../ablation/per_variant/baseline_iter*.pt" \
    --variant baseline \
    --bases gnp linear \
    --gnp_p 0.3 \
    --test_iterations 15 \
    --out_dir logs/ablation_eval_baseline

2) Compare best models across variants

  python scripts/eval_checkpoints_ablation.py \
    --compare_best \
    --best_globs "logs/**/ablation/per_variant/*_best_model.pt" \
    --bases gnp linear \
    --gnp_p 0.3 \
    --test_iterations 30 \
    --out_dir logs/ablation_eval_compare_best

3) One-shot: evaluate multiple variants' checkpoints then compare best models

  python scripts/eval_checkpoints_ablation.py \
    --variants baseline no_global_actor mlp_only \
    --checkpoint_globs \\
      "logs/**/ablation/per_variant/baseline_iter*.pt" \\
      "logs/**/ablation/per_variant/no_global_actor_iter*.pt" \\
      "logs/**/ablation/per_variant/mlp_only_iter*.pt" \
    --also_compare_best \
    --bases gnp linear \
    --gnp_p 0.3 \
    --test_iterations 20 \
    --out_dir logs/ablation_eval_all

"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch

# Local imports (mirror style used in ablation_gnn.py)
import sys as _sys, os as _os
_grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _grandparent_dir not in _sys.path:
    _sys.path.insert(0, _grandparent_dir)

from cogito.gnn_deeprl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args,
    _test_agent,
)


def _make_variant(name: str) -> AblationVariant:
    name = name.strip().lower()
    if name == "baseline":
        return AblationVariant(name="baseline")
    if name == "no_global_actor":
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    if name == "no_bn":
        return AblationVariant(name="no_bn", use_batchnorm=False)
    if name == "no_task_deps":
        return AblationVariant(name="no_task_deps", use_task_dependencies=False)
    if name == "shallow_gnn":
        return AblationVariant(name="shallow_gnn", gin_num_layers=1)
    if name == "mlp_only":
        return AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False)
    # Generic fallback: assume a custom variant name that maps 1:1 to files
    return AblationVariant(name=name)


def _extract_iteration(ckpt_path: Path) -> int:
    # Expect patterns like *_iter00025.pt
    m = re.search(r"iter(\d+)\.pt$", ckpt_path.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1


def build_eval_args(base: str, test_iterations: int, device: str, seed_base: int, gnp_p: float | None) -> Args:
    args = Args()
    args.device = device
    args.test_iterations = int(test_iterations)
    setattr(args, 'eval_seed_base', int(seed_base))
    args.dataset.dag_method = base
    # Optional: enforce higher connectivity on GNP
    if base == 'gnp' and gnp_p is not None:
        try:
            args.dataset.gnp_p = float(gnp_p)
        except Exception:
            pass
    return args


def evaluate_checkpoints(
    checkpoints: List[Path],
    variant_name: str,
    bases: List[str],
    out_dir: Path,
    device: str = 'cpu',
    test_iterations: int = 10,
    seed_base: int = 12345,
    gnp_p: float | None = None,
    # Optional: also evaluate on the exact training configuration
    train_cfg_enable: bool = False,
    train_seed: int | None = None,
    train_gnp_p: float | None = None,
    train_gnp_min_n: int | None = None,
    train_gnp_max_n: int | None = None,
    train_workflow_count: int | None = None,
    train_host_count: int | None = None,
    train_vm_count: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prepare agent template
    variant = _make_variant(variant_name)

    # Write CSV header
    csv_path = out_dir / f"ablation_{variant_name}.csv"
    import csv as _csv
    with csv_path.open('w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(["iteration", "base", "config", "makespan", "total_energy", "avg_active", "avg_idle", "active_plus_idle", "checkpoint"])  # noqa

    # Evaluate
    for ckpt in sorted(checkpoints, key=lambda p: _extract_iteration(p)):
        iteration = _extract_iteration(ckpt)
        print(f"[ablation-eval] Evaluating {ckpt} (iter={iteration}) for bases={bases}")
        # Load
        agent = AblationGinAgent(torch.device(device), variant)
        state = torch.load(str(ckpt), map_location=device)
        agent.load_state_dict(state, strict=False)
        agent.eval()
        # Evaluate across bases
        for base in bases:
            eval_args = build_eval_args(base, test_iterations, device, seed_base, gnp_p)
            mk, eobs, etot, mm = _test_agent(agent, eval_args)
            ea = (mm or {}).get('avg_active_energy')
            ei = (mm or {}).get('avg_idle_energy')
            api = (ea or 0.0) + (ei or 0.0)
            with csv_path.open('a', newline='') as f:
                w = _csv.writer(f)
                w.writerow([iteration, base, "default", mk, (etot if etot > 0 else eobs), (ea or 0.0), (ei or 0.0), api, str(ckpt)])

        # Optional: evaluate exact training configuration (GNP with training params and seed)
        if train_cfg_enable:
            base = 'gnp'
            eval_args = build_eval_args(base, test_iterations, device, seed_base=(train_seed if train_seed is not None else seed_base), gnp_p=(train_gnp_p if train_gnp_p is not None else gnp_p))
            # Override dataset sizes/resources if provided
            if train_gnp_min_n is not None:
                eval_args.dataset.gnp_min_n = int(train_gnp_min_n)
            if train_gnp_max_n is not None:
                eval_args.dataset.gnp_max_n = int(train_gnp_max_n)
            if train_workflow_count is not None:
                eval_args.dataset.workflow_count = int(train_workflow_count)
            if train_host_count is not None:
                eval_args.dataset.host_count = int(train_host_count)
            if train_vm_count is not None:
                eval_args.dataset.vm_count = int(train_vm_count)
            # Evaluate
            mk, eobs, etot, mm = _test_agent(agent, eval_args)
            ea = (mm or {}).get('avg_active_energy')
            ei = (mm or {}).get('avg_idle_energy')
            api = (ea or 0.0) + (ei or 0.0)
            with csv_path.open('a', newline='') as f:
                w = _csv.writer(f)
                w.writerow([iteration, base, "train_cfg", mk, (etot if etot > 0 else eobs), (ea or 0.0), (ei or 0.0), api, str(ckpt)])
    print(f"[ablation-eval] Wrote {csv_path}")
    return csv_path


def plot_checkpoint_curves(csv_path: Path, out_dir: Path, title: str = "Ablation Checkpoint Curves") -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ablation-eval] Skipping plots (pandas/matplotlib missing): {e}")
        return
    df = pd.read_csv(csv_path)
    if not {'iteration', 'base'}.issubset(set(df.columns)):
        print(f"[ablation-eval] CSV missing required columns, skipping plots: {csv_path}")
        return
    # Handle non-iter checkpoints by dropping iteration==-1 from lines
    dfl = df[df['iteration'] >= 0].copy()
    metrics = [
        ("makespan", "Avg makespan"),
        ("total_energy", "Avg total energy"),
        ("avg_active", "Avg active energy"),
        ("avg_idle", "Avg idle energy"),
        ("active_plus_idle", "Avg (active+idle) energy"),
    ]
    # Default comparisons across bases
    for metric, ylabel in metrics:
        plt.figure(figsize=(7.5, 4.0))
        for base, g in dfl[dfl['config'] == 'default'].groupby('base'):
            g2 = g.sort_values('iteration')
            plt.plot(g2['iteration'], g2[metric], marker='o', label=str(base))
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"ablation_{metric}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"[ablation-eval] Wrote {out_path}")

    # If train_cfg rows exist, also plot train configuration curves
    if 'config' in dfl.columns and (dfl['config'] == 'train_cfg').any():
        df_train = dfl[dfl['config'] == 'train_cfg']
        for metric, ylabel in metrics:
            plt.figure(figsize=(7.5, 4.0))
            g2 = df_train.sort_values('iteration')
            plt.plot(g2['iteration'], g2[metric], marker='o', label='gnp_train_cfg')
            plt.xlabel('Iteration')
            plt.ylabel(ylabel)
            plt.title(f"{title} (Training Config)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_path = out_dir / f"ablation_traincfg_{metric}.png"
            plt.savefig(out_path, dpi=180)
            plt.close()
            print(f"[ablation-eval] Wrote {out_path}")


def _infer_variant_from_path(p: Path) -> str:
    # Expect names like baseline_best_model.pt or no_global_actor_best_model.pt
    stem = p.stem  # e.g., baseline_best_model
    if stem.endswith("_best_model"):
        return stem[:-len("_best_model")]  # strip suffix
    # Fallback to directory name if possible
    return p.name.split("_best_model")[0] if "_best_model" in p.name else stem


def compare_best_models(
    model_paths: List[Path],
    bases: List[str],
    out_dir: Path,
    device: str = 'cpu',
    test_iterations: int = 20,
    seed_base: int = 12345,
    gnp_p: float | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    csv_path = out_dir / "ablation_compare_best.csv"
    with csv_path.open('w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(["variant", "base", "makespan", "total_energy", "avg_active", "avg_idle", "active_plus_idle", "model"])

    for mp in model_paths:
        if not mp.exists():
            print(f"[ablation-eval] Warning: model not found: {mp}")
            continue
        variant_name = _infer_variant_from_path(mp)
        variant = _make_variant(variant_name)
        print(f"[ablation-eval] Evaluating best model for variant '{variant_name}': {mp}")
        # Load agent
        agent = AblationGinAgent(torch.device(device), variant)
        state = torch.load(str(mp), map_location=device)
        agent.load_state_dict(state, strict=False)
        agent.eval()
        # Evaluate across bases
        for base in bases:
            eval_args = build_eval_args(base, test_iterations, device, seed_base, gnp_p)
            mk, eobs, etot, mm = _test_agent(agent, eval_args)
            ea = (mm or {}).get('avg_active_energy')
            ei = (mm or {}).get('avg_idle_energy')
            api = (ea or 0.0) + (ei or 0.0)
            with csv_path.open('a', newline='') as f:
                w = _csv.writer(f)
                w.writerow([variant_name, base, mk, (etot if etot > 0 else eobs), (ea or 0.0), (ei or 0.0), api, str(mp)])

    print(f"[ablation-eval] Wrote {csv_path}")
    return csv_path


def plot_compare_best(csv_path: Path, out_dir: Path, title: str = "Ablation: Best Models Comparison") -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ablation-eval] Skipping best-model plots (pandas/matplotlib missing): {e}")
        return
    df = pd.read_csv(csv_path)
    if not {'variant', 'base'}.issubset(set(df.columns)):
        print(f"[ablation-eval] CSV missing required columns, skipping plots: {csv_path}")
        return
    metrics = [
        ("makespan", "Avg makespan"),
        ("total_energy", "Avg total energy"),
        ("avg_active", "Avg active energy"),
        ("avg_idle", "Avg idle energy"),
        ("active_plus_idle", "Avg (active+idle) energy"),
    ]
    # Bar plots per base comparing variants
    for base, g in df.groupby('base'):
        for metric, ylabel in metrics:
            try:
                import matplotlib.pyplot as plt
                g2 = g.sort_values('variant')
                plt.figure(figsize=(7.5, 4.0))
                xs = np.arange(len(g2['variant']))
                plt.bar(xs, g2[metric])
                plt.xticks(xs, g2['variant'], rotation=45, ha='right')
                plt.xlabel('Variant')
                plt.ylabel(ylabel)
                plt.title(f"{title} ({base})")
                plt.tight_layout()
                out_path = out_dir / f"ablation_best_{base}_{metric}.png"
                plt.savefig(out_path, dpi=180)
                plt.close()
                print(f"[ablation-eval] Wrote {out_path}")
            except Exception as e:
                print(f"[ablation-eval] Plot failed for base={base}, metric={metric}: {e}")


def main():
    ap = argparse.ArgumentParser()
    # Mode selection
    ap.add_argument('--compare_best', action='store_true', help='If set, compare best models across variants instead of per-checkpoint curves')
    ap.add_argument('--also_compare_best', action='store_true', help='When using --variants/--checkpoint_globs, also compare *_best_model.pt across those variants in one run')

    # Common eval params
    ap.add_argument('--bases', type=str, nargs='+', default=['gnp', 'linear'], help='List of DAG bases to evaluate (gnp, linear)')
    ap.add_argument('--gnp_p', type=float, default=None, help='Optional p for GNP during eval (e.g., 0.3)')
    ap.add_argument('--device', type=str, default='cpu', help='cpu|cuda|mps')
    ap.add_argument('--test_iterations', type=int, default=10)
    ap.add_argument('--seed_base', type=int, default=12345)
    ap.add_argument('--out_dir', type=str, default='logs/ablation_eval')
    ap.add_argument('--no_plots', action='store_true', help='Skip plots')

    # Per-checkpoint evaluation args (non-compare_best)
    ap.add_argument('--checkpoints', type=str, default='', help='Glob pattern to checkpoint files, e.g., logs/.../per_variant/baseline_iter*.pt')
    ap.add_argument('--variant', type=str, default='baseline', help='Variant name (baseline, no_global_actor, etc.)')
    # One-shot multi-variant evaluation (paired lists)
    ap.add_argument('--variants', type=str, nargs='*', default=None, help='List of variant names to evaluate in one shot')
    ap.add_argument('--checkpoint_globs', type=str, nargs='*', default=None, help='List of glob patterns, one per variant, matching iter*.pt checkpoints')
    # Training configuration overrides (optional)
    ap.add_argument('--train_cfg_enable', action='store_true', help='Also evaluate on exact training configuration (GNP)')
    ap.add_argument('--train_seed', type=int, default=None, help='Training seed to use for eval (eval_seed_base)')
    ap.add_argument('--train_gnp_p', type=float, default=None, help='GNP p used in training (e.g., 0.3)')
    ap.add_argument('--train_gnp_min_n', type=int, default=None)
    ap.add_argument('--train_gnp_max_n', type=int, default=None)
    ap.add_argument('--train_workflow_count', type=int, default=None)
    ap.add_argument('--train_host_count', type=int, default=None)
    ap.add_argument('--train_vm_count', type=int, default=None)

    # Best-model comparison args (compare_best)
    ap.add_argument('--best_globs', type=str, nargs='*', default=None, help='List of glob patterns that resolve to *_best_model.pt files')

    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # 1) Dedicated compare-best mode
    if args.compare_best:
        # Collect best model paths
        model_paths: List[Path] = []
        if args.best_globs:
            for pat in args.best_globs:
                for p in glob.glob(pat):
                    if p.endswith('_best_model.pt'):
                        model_paths.append(Path(p))
        if not model_paths:
            raise SystemExit("No *_best_model.pt files found via --best_globs.")
        csv_path = compare_best_models(
            model_paths=model_paths,
            bases=args.bases,
            out_dir=out_dir,
            device=args.device,
            test_iterations=args.test_iterations,
            seed_base=args.seed_base,
            gnp_p=args.gnp_p,
        )
        if not args.no_plots:
            plot_compare_best(csv_path, out_dir, title="Ablation: Best Models Comparison")
        return

    # 2) One-shot multi-variant evaluation + optional best-compare
    if args.variants and args.checkpoint_globs:
        if len(args.variants) != len(args.checkpoint_globs):
            raise SystemExit('--variants and --checkpoint_globs must have the same length')
        per_variant_csvs: Dict[str, Path] = {}
        for vname, pat in zip(args.variants, args.checkpoint_globs):
            ckpts = sorted([Path(p) for p in glob.glob(pat)], key=lambda p: _extract_iteration(Path(p)))
            if not ckpts:
                print(f"[ablation-eval] Warning: no checkpoints for variant={vname} with pattern: {pat}")
                continue
            v_out = out_dir / vname
            v_out.mkdir(parents=True, exist_ok=True)
            csv_path = evaluate_checkpoints(
                ckpts,
                vname,
                args.bases,
                v_out,
                device=args.device,
                test_iterations=args.test_iterations,
                seed_base=args.seed_base,
                gnp_p=args.gnp_p,
                train_cfg_enable=args.train_cfg_enable,
                train_seed=args.train_seed,
                train_gnp_p=args.train_gnp_p,
                train_gnp_min_n=args.train_gnp_min_n,
                train_gnp_max_n=args.train_gnp_max_n,
                train_workflow_count=args.train_workflow_count,
                train_host_count=args.train_host_count,
                train_vm_count=args.train_vm_count,
            )
            per_variant_csvs[vname] = csv_path
            if not args.no_plots:
                plot_checkpoint_curves(csv_path, v_out, title=f"Ablation Checkpoints: {vname}")
        # Optionally compare *_best_model.pt for the same set of variants
        if args.also_compare_best:
            # Find best models for provided variants
            bests: List[Path] = []
            for vname in (args.variants or []):
                # Search under out_dir's parent logs by default
                candidates = glob.glob(f"**/ablation/per_variant/{vname}_best_model.pt", recursive=True)
                for c in candidates:
                    bests.append(Path(c))
            if not bests:
                print('[ablation-eval] Warning: no *_best_model.pt found for provided variants to compare')
            else:
                best_csv = compare_best_models(
                    model_paths=bests,
                    bases=args.bases,
                    out_dir=out_dir / 'compare_best',
                    device=args.device,
                    test_iterations=args.test_iterations,
                    seed_base=args.seed_base,
                    gnp_p=args.gnp_p,
                )
                if not args.no_plots:
                    plot_compare_best(best_csv, out_dir / 'compare_best', title='Ablation: Best Models Comparison')
        return

    # 3) Per-checkpoint single-variant evaluation mode
    ckpts: List[Path] = []
    if args.checkpoints:
        ckpts = sorted([Path(p) for p in glob.glob(args.checkpoints)], key=lambda p: _extract_iteration(Path(p)))
    if not ckpts:
        # Allow single file path
        p = Path(args.checkpoints)
        if p.exists():
            ckpts = [p]
        else:
            raise SystemExit(f"No checkpoints match pattern: {args.checkpoints}")

    csv_path = evaluate_checkpoints(
        ckpts,
        args.variant,
        args.bases,
        out_dir,
        device=args.device,
        test_iterations=args.test_iterations,
        seed_base=args.seed_base,
        gnp_p=args.gnp_p,
        train_cfg_enable=args.train_cfg_enable,
        train_seed=args.train_seed,
        train_gnp_p=args.train_gnp_p,
        train_gnp_min_n=args.train_gnp_min_n,
        train_gnp_max_n=args.train_gnp_max_n,
        train_workflow_count=args.train_workflow_count,
        train_host_count=args.train_host_count,
        train_vm_count=args.train_vm_count,
    )
    if not args.no_plots:
        plot_checkpoint_curves(csv_path, out_dir, title=f"Ablation Checkpoints: {args.variant}")


if __name__ == '__main__':
    main()
