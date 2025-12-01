#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Project imports
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


def _load_train_features(domain_dir: Path) -> pd.DataFrame | None:
    tf = domain_dir / 'train_features.csv'
    if tf.exists():
        try:
            return pd.read_csv(tf)
        except Exception:
            return None
    return None


def _read_seeds(json_path: Path) -> List[int]:
    if not json_path.exists():
        return []
    try:
        with json_path.open('r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'train_seeds' in data and isinstance(data['train_seeds'], list):
                return [int(x) for x in data['train_seeds']]
            if 'selected_eval_seeds' in data and isinstance(data['selected_eval_seeds'], list):
                return [int(x) for x in data['selected_eval_seeds']]
    except Exception:
        pass
    return []


def _load_rl_config(path: Path) -> tuple[list[int], dict]:
    with path.open('r') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or 'train' not in cfg:
        raise ValueError(f'Invalid RL config schema: {path}')
    train = cfg['train']
    seeds = [int(x) for x in train.get('seeds', [])]
    ds = dict(train.get('dataset', {}))
    ds_kwargs = {
        'dag_method': ds.get('dag_method', 'gnp'),
        'gnp_min_n': int(ds.get('gnp_min_n', 12)),
        'gnp_max_n': int(ds.get('gnp_max_n', 30)),
        'gnp_p': float(ds.get('gnp_p', 0.1)) if ds.get('gnp_p', None) is not None else None,
        'host_count': int(ds.get('host_count', 4)),
        'vm_count': int(ds.get('vm_count', 10)),
        'max_memory_gb': int(ds.get('max_memory_gb', 10)),
        'min_cpu_speed_mips': int(ds.get('min_cpu_speed', 500)),
        'max_cpu_speed_mips': int(ds.get('max_cpu_speed', 5000)),
        'workflow_count': int(ds.get('workflow_count', 1)),
        'task_length_dist': ds.get('task_length_dist', 'uniform'),
        'min_task_length': int(ds.get('min_task_length', 500)),
        'max_task_length': int(ds.get('max_task_length', 100_000)),
        'task_arrival': ds.get('task_arrival', 'static'),
        'arrival_rate': float(ds.get('arrival_rate', 0.0)),
    }
    return seeds, ds_kwargs


def _choose_action_first_ready(env: GinAgentWrapper) -> int:
    vm_count = len(env.prev_obs.vm_observations)
    compat_set = set(env.prev_obs.compatibilities)
    for t_id, t in enumerate(env.prev_obs.task_observations):
        if t_id in (0, len(env.prev_obs.task_observations) - 1):
            continue
        if t.assigned_vm_id is not None or not t.is_ready:
            continue
        for vm_id in range(vm_count):
            if (t_id, vm_id) in compat_set:
                return int(t_id * vm_count + vm_id)
    return 0


def _run_once_collect_metrics(ds_kwargs: dict, seed: int) -> Dict[str, float]:
    dataset = generate_dataset(
        seed=seed,
        host_count=int(ds_kwargs['host_count']),
        vm_count=int(ds_kwargs['vm_count']),
        max_memory_gb=int(ds_kwargs['max_memory_gb']),
        min_cpu_speed_mips=int(ds_kwargs['min_cpu_speed_mips']),
        max_cpu_speed_mips=int(ds_kwargs['max_cpu_speed_mips']),
        workflow_count=int(ds_kwargs['workflow_count']),
        dag_method=str(ds_kwargs['dag_method']),
        gnp_min_n=int(ds_kwargs['gnp_min_n']),
        gnp_max_n=int(ds_kwargs['gnp_max_n']),
        gnp_p=ds_kwargs.get('gnp_p', None),
        task_length_dist=str(ds_kwargs['task_length_dist']),
        min_task_length=int(ds_kwargs['min_task_length']),
        max_task_length=int(ds_kwargs['max_task_length']),
        task_arrival=str(ds_kwargs['task_arrival']),
        arrival_rate=float(ds_kwargs['arrival_rate']),
        vm_rng_seed=0,
    )
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=True, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=seed)
    final_info = None
    while True:
        a = _choose_action_first_ready(env)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            final_info = info
            break
    metrics: Dict[str, float] = {}
    if isinstance(final_info, dict):
        # Prefer timeline-derived metrics if available
        t = final_info.get('timeline_t', None)
        r = final_info.get('timeline_ready', None)
        s = final_info.get('timeline_schedulable', None)
        if t is not None and r is not None and s is not None and len(t) == len(r) == len(s) and len(t) > 0:
            r_np = np.asarray(r, dtype=float)
            s_np = np.asarray(s, dtype=float)
            gap = np.maximum(0.0, r_np - s_np)
            metrics['decision_steps'] = float(len(t))
            metrics['sum_ready_tasks'] = float(np.sum(r_np))
            metrics['sum_bottleneck_ready_tasks'] = float(np.sum(gap))
            metrics['bottleneck_steps'] = float(np.sum((gap > 0).astype(float)))
        # Also copy any direct metrics if present
        for k in ['bottleneck_steps','decision_steps','sum_ready_tasks','sum_bottleneck_ready_tasks','cumulative_wait_time']:
            if k in final_info and not np.isfinite(metrics.get(k, np.nan)):
                try:
                    metrics[k] = float(final_info[k])
                except Exception:
                    pass
    # Ensure all keys exist
    for k in ['bottleneck_steps','decision_steps','sum_ready_tasks','sum_bottleneck_ready_tasks','cumulative_wait_time']:
        if k not in metrics:
            metrics[k] = float('nan')
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide-dir', type=Path, default=Path('runs/datasets/wide/representativeness'))
    ap.add_argument('--longcp-dir', type=Path, default=Path('runs/datasets/longcp/representativeness'))
    ap.add_argument('--wide-config', type=Path, default=None)
    ap.add_argument('--longcp-config', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=Path('runs/datasets/mixed/representativeness_ot'))
    ap.add_argument('--max-per-domain', type=int, default=40)
    ap.add_argument('--num-pca-components', type=int, default=2)
    ap.add_argument('--seeds-per-domain', type=int, default=30)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load domain features or seeds
    df_wide = _load_train_features(args.wide_dir)
    df_long = _load_train_features(args.longcp_dir)
    if df_wide is None or df_long is None:
        wide_seeds = _read_seeds(args.wide_dir / 'selected_eval_seeds.json')
        long_seeds = _read_seeds(args.longcp_dir / 'selected_eval_seeds.json')
        if not wide_seeds or not long_seeds:
            raise SystemExit('Need train_features.csv or selected_eval_seeds.json in both domain dirs')
        # Build minimal placeholder features
        df_wide = pd.DataFrame({'seed': wide_seeds})
        df_long = pd.DataFrame({'seed': long_seeds})
        feat_cols: List[str] = []
    else:
        feat_cols = [c for c in [
            'tasks','edges','width_peak','depth_avg','Pbar','burstiness','cp_frac'
        ] if c in df_wide.columns and c in df_long.columns]
        if not feat_cols:
            raise SystemExit('No common feature columns found in train_features.csv')

    # Harmonize and label
    df_wide['domain'] = 'wide'
    df_long['domain'] = 'longcp'
    df_all = pd.concat([df_wide, df_long], ignore_index=True)

    # PCA on graph-level features (if available)
    if feat_cols:
        Xp = df_all[feat_cols].to_numpy(dtype=float)
        Xp = (Xp - Xp.mean(0, keepdims=True)) / (Xp.std(0, keepdims=True) + 1e-9)
        pca = PCA(n_components=int(args.num_pca_components), random_state=0)
        Zp = pca.fit_transform(Xp)
        coords = pd.DataFrame(Zp[:, :2], columns=['PC1','PC2'])
    else:
        # If no features, set NaNs
        coords = pd.DataFrame({'PC1': np.nan, 'PC2': np.nan}, index=df_all.index)
    coords_df = pd.concat([df_all[['seed','domain']].reset_index(drop=True), coords.reset_index(drop=True)], axis=1)
    coords_path = args.out_dir / 'pca_coords_by_seed.csv'
    coords_df.to_csv(coords_path, index=False)

    # Dataset kwargs from RL configs (preferred)
    ds_wide = ds_long = None
    if args.wide_config is not None and args.wide_config.exists():
        _, ds_wide = _load_rl_config(args.wide_config)
    if args.longcp_config is not None and args.longcp_config.exists():
        _, ds_long = _load_rl_config(args.longcp_config)
    if ds_wide is None or ds_long is None:
        raise SystemExit('Please provide --wide-config and --longcp-config to generate datasets consistently')

    # Select seeds to evaluate per domain
    def _take(df: pd.DataFrame, k: int) -> List[int]:
        vals = df['seed'].tolist()
        if len(vals) <= k:
            return [int(x) for x in vals]
        idxs = np.linspace(0, len(vals)-1, num=k, dtype=int)
        return [int(vals[i]) for i in idxs]

    seeds_w = _take(coords_df[coords_df['domain']=='wide'], int(args.seeds_per_domain))
    seeds_l = _take(coords_df[coords_df['domain']=='longcp'], int(args.seeds_per_domain))

    # Collect metrics per seed
    rows: List[Dict[str, float]] = []
    for s in seeds_w:
        m = _run_once_collect_metrics(ds_wide, seed=int(s))
        rows.append({'seed': int(s), 'domain': 'wide', **m})
    for s in seeds_l:
        m = _run_once_collect_metrics(ds_long, seed=int(s))
        rows.append({'seed': int(s), 'domain': 'longcp', **m})
    metrics_df = pd.DataFrame(rows)
    metrics_path = args.out_dir / 'episode_metrics_by_seed.csv'
    metrics_df.to_csv(metrics_path, index=False)

    # Join with PCA coords and compute correlations
    joined = pd.merge(coords_df, metrics_df, on=['seed','domain'], how='inner')

    def _corr(a: pd.Series, b: pd.Series) -> float:
        a2 = a.to_numpy(dtype=float)
        b2 = b.to_numpy(dtype=float)
        m = (~np.isnan(a2)) & (~np.isnan(b2))
        if m.sum() < 2:
            return float('nan')
        return float(np.corrcoef(a2[m], b2[m])[0,1])

    corr_rows = []
    for metric in ['bottleneck_steps','sum_ready_tasks']:
        corr_rows.append({'metric': metric, 'corr_PC1': _corr(joined['PC1'], joined[metric]), 'corr_PC2': _corr(joined['PC2'], joined[metric])})
    corr_df = pd.DataFrame(corr_rows)
    corr_path = args.out_dir / 'correlations_PC_vs_metrics.csv'
    corr_df.to_csv(corr_path, index=False)

    # Scatter plots
    try:
        for metric in ['bottleneck_steps','sum_ready_tasks']:
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            for dom in ['wide','longcp']:
                sub = joined[joined['domain']==dom]
                plt.scatter(sub['PC1'], sub[metric], s=25, alpha=0.7, label=dom)
            plt.xlabel('PC1')
            plt.ylabel(metric)
            plt.title(f'{metric} vs PC1')
            plt.legend()
            plt.subplot(1,2,2)
            for dom in ['wide','longcp']:
                sub = joined[joined['domain']==dom]
                plt.scatter(sub['PC2'], sub[metric], s=25, alpha=0.7, label=dom)
            plt.xlabel('PC2')
            plt.ylabel(metric)
            plt.title(f'{metric} vs PC2')
            plt.tight_layout()
            plt.savefig(args.out_dir / f'scatter_{metric}_vs_PC12.png', dpi=180)
            plt.close()
    except Exception:
        pass

    print('[done] Wrote:')
    print(' -', coords_path)
    print(' -', metrics_path)
    print(' -', corr_path)


if __name__ == '__main__':
    main()
