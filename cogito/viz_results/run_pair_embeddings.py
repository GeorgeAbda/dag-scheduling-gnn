#!/usr/bin/env python3
"""
Run pairwise (task, VM) embedding visualization for a GIN agent and save to HTML.

Examples:
  python -m scheduler.viz_results.run_pair_embeddings \
    --labeling characteristics \
    --reducer tsne \
    --use_all_pairs false \
    --n_clusters 6 \
    --seed 1 \
    --output figures/gin_embeddings.html

  # TSNE-only mode
  python -m scheduler.viz_results.run_pair_embeddings \
    --labeling kmeans \
    --reducer tsne \
    --n_clusters 6 \
    --output figures/gin_embeddings_tsne.html

If you have a trained checkpoint for the ablation GIN agent, pass it via --checkpoint.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
import os
from pathlib import Path
import sys
import torch
import numpy as np
import plotly.io as pio

from cogito.viz_results.embeddings import GINInterpreter
from cogito.gnn_deeprl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as TrainArgs,
    _make_test_env,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GIN pairwise (task, VM) visualization and save HTML")
    p.add_argument("--labeling", type=str, default="characteristics", choices=["characteristics", "kmeans"],
                   help="Labeling method for (task, VM) pairs")
    p.add_argument("--reducer", type=str, default="tsne", choices=["tsne"],
                   help="Dimensionality reduction method")
    p.add_argument("--use_all_pairs", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=False,
                   help="If true, use all task-VM pairs (cartesian); otherwise, only compatibilities")
    p.add_argument("--n_clusters", type=int, default=6, help="Number of clusters when labeling=kmeans")
    p.add_argument("--seed", type=int, default=1, help="Environment reset seed")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda","mps"], help="Torch device")
    p.add_argument("--output", type=str, default="figures/gin_embeddings.html", help="Output HTML path")
    p.add_argument("--compare_models", nargs='+', default=None, help="List of model paths to compare")
    p.add_argument("--checkpoint", type=str, default="", help="Optional path to a trained agent checkpoint (.pt)")
    p.add_argument("--model1", type=str, default="", help="Path to first model checkpoint for highlighted plot")
    p.add_argument("--model2", type=str, default="", help="Path to second model checkpoint for highlighted plot")
    p.add_argument("--model3", type=str, default="", help="Path to third model checkpoint for highlighted plot")
    p.add_argument("--top_k_highlight", type=int, default=3, help="Top-K (task, VM) pairs to highlight by action score for each model")
    # Auto-train fallback when no model paths are provided
    p.add_argument("--auto_train_if_missing", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=True,
                   help="If true and no --modelN provided, run a short ablation_gnn training to obtain a checkpoint")
    p.add_argument("--train_total_timesteps", type=int, default=200000, help="Timesteps for auto-train fallback")
    p.add_argument("--train_exp_name", type=str, default="gnn_ablation_linear", help="Experiment name for auto-train run")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Prepare env and one observation
    targs = TrainArgs()
    env = _make_test_env(targs)
    next_obs, _ = env.reset(seed=int(args.seed))
    obs_tensor = torch.from_numpy(np.asarray(next_obs, dtype=np.float32)).unsqueeze(0)

    # Pick device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")
    elif args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("MPS requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Create agent (Ablation GIN default variant); load checkpoint if provided
    variant = AblationVariant(name="viz_gin_linear")
    agent = AblationGinAgent(device=device, variant=variant)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=device)
            missing = agent.load_state_dict(state, strict=False)
            if getattr(missing, 'missing_keys', None):
                print(f"Loaded checkpoint with missing keys (strict=False): {missing.missing_keys}")
            print(f"Loaded checkpoint: {ckpt_path} (strict=False)")
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}; proceeding with untrained weights", file=sys.stderr)

    # Handle model comparison if requested
    if args.compare_models:
        interpreter = GINInterpreter(agent, device=str(device))
        fig = interpreter.compare_models_top_pairs(
            obs_tensor, 
            model_paths=args.compare_models
        )
        return 0

    # Interpreter and visualization
    output_base = Path(args.output).stem
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    interpreter = GINInterpreter(agent, device=str(device))
    # Generate all combinations of labeling methods and reducers (baseline, no highlighting)
    all_metrics = {}
    for labeling_method in ['characteristics', 'kmeans']:
        for reducer in ['tsne']:
            fig, metrics = interpreter.visualize_task_vm_pairs(
                obs_tensor=obs_tensor,
                labeling_method=labeling_method,
                reducer=reducer,
                use_all_pairs=bool(args.use_all_pairs),
                n_clusters=int(args.n_clusters),
                title_prefix=f"Task-VM Pairs ({labeling_method})",
            )
            out_path = output_dir / f"{output_base}_{labeling_method}_{reducer}.html"
            pio.write_html(fig, file=str(out_path), auto_open=False)
            print(f"Saved: {out_path}")
            all_metrics[f"{labeling_method}_{reducer}"] = metrics

    # Helper to compute top-K (task, VM) pairs from a model's action scores
    def topk_pairs_for_model(agt: AblationGinAgent, K: int) -> list[tuple[int, int]]:
        with torch.no_grad():
            obs = agt.mapper.unmap(obs_tensor.squeeze(0))
            action_scores: torch.Tensor = agt.actor(obs)  # [T, V]
            # Use compatibilities to enumerate valid pairs
            comp = obs.compatibilities
            t_idx = comp[0].long()
            v_idx = comp[1].long()
            scores = action_scores[t_idx, v_idx]
            K = max(1, min(int(K), scores.numel()))
            topk = torch.topk(scores, k=K).indices
            sel = [(int(t_idx[i].item()), int(v_idx[i].item())) for i in topk]
            return sel

    # If no specific models provided and auto-train is enabled, invoke trainer to obtain a checkpoint
    if not any([args.model1, args.model2, args.model3]) and not args.checkpoint and bool(args.auto_train_if_missing):
        print("No model paths provided. Auto-training a GIN agent via ablation_gnn to obtain a checkpoint...")
        cmd = [
            sys.executable, "-m", "scheduler.rl_model.ablation_gnn",
            "--exp_name", str(args.train_exp_name),
            "--train_only_baseline",
            "--dataset.dag_method", "linear",
            "--dataset.gnp_min_n", "12",
            "--dataset.gnp_max_n", "24",
            "--dataset.workflow_count", "10",
            "--dataset.host_count", "4",
            "--dataset.vm_count", "10",
            "--total_timesteps", str(int(args.train_total_timesteps)),
            "--test_every_iters", "10",
            "--test_iterations", "4",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: auto-train failed with error {e}. Proceeding without a trained checkpoint.", file=sys.stderr)
        # Attempt to locate a checkpoint under logs/
        logs_dir = Path("logs")
        found_ckpt = None
        if logs_dir.exists():
            # Search for most recent model.pt within last modified run
            newest_time = 0
            for root, dirs, files in os.walk(logs_dir):
                for f in files:
                    if f.endswith(".pt") and f in {"model.pt", "agent.pt", "checkpoint.pt"}:
                        fp = Path(root) / f
                        mtime = fp.stat().st_mtime
                        if mtime > newest_time:
                            newest_time = mtime
                            found_ckpt = fp
        if found_ckpt is not None:
            print(f"Auto-train produced checkpoint: {found_ckpt}")
            args.model1 = str(found_ckpt)
        else:
            print("Warning: no checkpoint file was found after training. Highlighted plots will use the base agent.", file=sys.stderr)

    # For each provided model checkpoint, load model, compute top-K pairs, and generate highlighted plots
    model_paths = [("model1", args.model1), ("model2", args.model2), ("model3", args.model3)]
    for tag, mpath in model_paths:
        if not mpath:
            continue
        ckpt = Path(mpath)
        if not ckpt.exists():
            print(f"Warning: {tag} checkpoint not found at {ckpt}", file=sys.stderr)
            continue
        # Fresh agent to avoid weight contamination
        model_agent = AblationGinAgent(device=device, variant=variant)
        state = torch.load(str(ckpt), map_location=device)
        try:
            missing = model_agent.load_state_dict(state, strict=False)
            if getattr(missing, 'missing_keys', None):
                print(f"Loaded {tag} with missing keys (strict=False): {missing.missing_keys}")
        except Exception as e:
            print(f"Warning: failed to load {tag} ({ckpt}): {e}", file=sys.stderr)
            continue

        # Compute top-K highlight pairs
        highlight_pairs = topk_pairs_for_model(model_agent, args.top_k_highlight)
        pretty_tag = ckpt.parent.name  # e.g., '1757444537_energy'

        for labeling_method in ['characteristics']:
            for reducer in ['tsne']:
                fig_h, metrics_h = interpreter.visualize_task_vm_pairs(
                    obs_tensor=obs_tensor,
                    labeling_method=labeling_method,
                    reducer=reducer,
                    use_all_pairs=False,
                    n_clusters=int(args.n_clusters),
                    title_prefix=f"Task-VM Pairs ({labeling_method}) — {pretty_tag}",
                    highlight_pairs=highlight_pairs,
                )
                out_path_h = output_dir / f"{output_base}_{labeling_method}_{reducer}_{pretty_tag}_highlight.html"
                pio.write_html(fig_h, file=str(out_path_h), auto_open=False)
                print(f"Saved: {out_path_h}")

    print("\nAll Metrics:")
    for config, metrics in all_metrics.items():
        print(f"\n{config}:")
        if metrics:
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print("  No metrics available")
    
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())