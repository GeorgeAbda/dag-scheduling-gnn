import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import List, Tuple

import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as TrainArgs,
    _make_test_env,
    _test_agent,
)
from scheduler.viz_results.embeddings import GINInterpreter


@dataclass
class ModelSpec:
    label: str
    path: str
    variant: str  # 'baseline' or 'no_global_actor' (extends as needed)


def parse_model_arg(s: str) -> ModelSpec:
    # format: label=/path/to/ckpt.pt:variant
    # variant defaults to 'baseline' if omitted
    if ':' in s:
        lp, variant = s.rsplit(':', 1)
    else:
        lp, variant = s, 'baseline'
    if '=' in lp:
        label, path = lp.split('=', 1)
    else:
        # derive label from directory name
        path = lp
        label = Path(path).parent.name
    return ModelSpec(label=label, path=path, variant=variant)


def build_obs(seed: int, device: str, workflow_count: int, dag_method: str):
    targs = TrainArgs()
    targs.seed = int(seed)
    targs.device = device
    targs.dataset.workflow_count = int(workflow_count)
    targs.dataset.dag_method = dag_method
    # Make GNP test graphs more diverse/long-wide if applicable

    env = _make_test_env(targs)
    obs, _ = env.reset()
    env.close()
    return torch.from_numpy(obs).unsqueeze(0).to(dtype=torch.float32)


def quick_eval(agent: AblationGinAgent, dag_method: str, iterations: int = 3, seed_base: int | None = None) -> dict:
    args = TrainArgs()
    args.device = str(agent.device)
    args.dataset.dag_method = dag_method
    args.test_iterations = int(iterations)
    if seed_base is not None:
        args.seed = int(seed_base)
        # Use same seed base for eval episodes to mirror training/eval configs
        setattr(args, 'eval_seed_base', int(seed_base))
    mk, eobs, etot, mm = _test_agent(agent, args)
    return {
        'makespan': float(mk),
        'total_energy': float(etot if etot > 0 else eobs),
        'avg_active_energy': float(mm.get('avg_active_energy')) if mm and (mm.get('avg_active_energy') is not None) else None,
        'avg_idle_energy': float(mm.get('avg_idle_energy')) if mm and (mm.get('avg_idle_energy') is not None) else None,
        'avg_active_plus_idle_energy': float(mm.get('avg_active_energy') + mm.get('avg_idle_energy')) if mm and (mm.get('avg_active_energy') is not None) and (mm.get('avg_idle_energy') is not None) else None,
    }


def load_agent(spec: ModelSpec, device: torch.device) -> AblationGinAgent:
    # Build the agent architecture according to variant
    if spec.variant == 'baseline':
        variant = AblationVariant(name='baseline')
    elif spec.variant == 'no_global_actor':
        variant = AblationVariant(name='no_global_actor', use_actor_global_embedding=False)
    else:
        raise ValueError(f"Unknown variant '{spec.variant}' for model {spec.label}")
    agent = AblationGinAgent(device=device, variant=variant)
    state = torch.load(spec.path, map_location=device)
    agent.load_state_dict(state, strict=False)
    return agent


def reduce_2d(X: np.ndarray, method: str):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    if method == 'pca':
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T
    perp = min(30, max(2, X.shape[0]-1))
    return TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)


def build_fig_edge_only(models: List[Tuple[ModelSpec, AblationGinAgent]], obs_list, reducer: str, top_k: int, width: int, height: int, title: str):
    # Two rows: original (raw feature) space on row 1, latent (learned embedding) space on row 2
    cols = len(models)
    fig = make_subplots(
        rows=2,
        cols=cols,
        subplot_titles=[m.label for m, _ in models]
    )

    # Collect per-model entries for CSVs (unchanged behavior)
    entries = []  # (label, Z_emb2d, S, texts, top_idx)

    for col_idx, (spec, agent) in enumerate(models, start=1):
        inter = GINInterpreter(agent, device=str(agent.device))
        learned_all, scores_all, texts_all = [], [], []
        rawfeat_all = []

        for obs_name, obs_tensor in obs_list:
            # Learned embeddings and scores over feasible pairs (edge-only)
            learned, scores, pairs = inter.model_learned_pair_embeddings(obs_tensor, include_graph=False)
            learned_all.append(learned)
            scores_all.append(scores)
            texts_all += [f"{obs_name}: t{ti}-v{vj}" for (ti, vj) in pairs]

            # Build simple raw feature vector per pair for the same pairs
            emb = inter.extract_embeddings_and_features(obs_tensor)
            t_len = emb['task_features']['length']
            t_mem = emb['task_features']['memory_req']
            v_speed = emb['vm_features']['speed']
            v_mem = emb['vm_features']['memory_total']
            for (ti, vj) in pairs:
                raw_vec = [
                    float(t_len[ti]),
                    float(t_mem[ti]),
                    float(v_speed[vj]),
                    float(v_mem[vj]),
                    float(t_len[ti] / (v_speed[vj] + 1e-8)),
                ]
                rawfeat_all.append(raw_vec)

        # Concatenate across observations
        L = np.concatenate(learned_all, axis=0) if learned_all else np.zeros((0, 2))
        S = np.concatenate(scores_all, axis=0) if scores_all else np.zeros((0,))
        R = np.asarray(rawfeat_all, dtype=float) if rawfeat_all else np.zeros((0, 2))

        # Reduce to 2D
        Z_emb = reduce_2d(L, reducer) if L.shape[0] else np.zeros((0, 2))
        Z_raw = reduce_2d(R, reducer) if R.shape[0] else np.zeros((0, 2))

        # Compute top-k indices for opacity highlighting and CSVs
        k = max(1, min(int(top_k), len(S)))
        s_for_rank = np.nan_to_num(S, nan=-1e9)
        top_idx = np.argsort(-s_for_rank)[:k]
        mask = np.zeros(len(S), dtype=bool)
        mask[top_idx] = True
        lo_idx = np.where(~mask)[0]
        hi_idx = top_idx

        # Colors and styles
        base_color = 'rgba(31,119,180,1.0)'
        bg_opacity = 0.15
        hi_opacity = 1.0
        hi_size = 9
        bg_size = 6

        # Plot original/raw space (row 1) with opacity layering
        if Z_raw.size:
            # Background
            if lo_idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=Z_raw[lo_idx, 0],
                        y=Z_raw[lo_idx, 1],
                        mode='markers',
                        name=f'{spec.label} raw bg',
                        marker=dict(size=bg_size, color=base_color, opacity=bg_opacity),
                        text=[texts_all[i] for i in lo_idx],
                        hoverinfo='skip',
                    ),
                    row=1, col=col_idx,
                )
            # Top-K foreground
            if hi_idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=Z_raw[hi_idx, 0],
                        y=Z_raw[hi_idx, 1],
                        mode='markers',
                        name=f'{spec.label} raw top-{k}',
                        marker=dict(size=hi_size, color=base_color, opacity=hi_opacity, line=dict(width=1, color='black')),
                        text=[f"{texts_all[i]}<br>score: {float(S[i]):.3f}" for i in hi_idx],
                        hoverinfo='text',
                    ),
                    row=1, col=col_idx,
                )

        # Plot latent/embedding space (row 2) with opacity layering
        if Z_emb.size:
            if lo_idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=Z_emb[lo_idx, 0],
                        y=Z_emb[lo_idx, 1],
                        mode='markers',
                        name=f'{spec.label} emb bg',
                        marker=dict(size=bg_size, color=base_color, opacity=bg_opacity),
                        text=[texts_all[i] for i in lo_idx],
                        hoverinfo='skip',
                    ),
                    row=2, col=col_idx,
                )
            if hi_idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=Z_emb[hi_idx, 0],
                        y=Z_emb[hi_idx, 1],
                        mode='markers',
                        name=f'{spec.label} emb top-{k}',
                        marker=dict(size=hi_size, color=base_color, opacity=hi_opacity, line=dict(width=1, color='black')),
                        text=[f"{texts_all[i]}<br>score: {float(S[i]):.3f}" for i in hi_idx],
                        hoverinfo='text',
                    ),
                    row=2, col=col_idx,
                )

        entries.append((spec.label, Z_emb, S, texts_all, top_idx))

    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    # Axes titles per row
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text=f"{reducer.upper()} 1 (raw)", row=1, col=c)
        fig.update_yaxes(title_text=f"{reducer.upper()} 2 (raw)", row=1, col=c)
        fig.update_xaxes(title_text=f"{reducer.upper()} 1 (emb)", row=2, col=c)
        fig.update_yaxes(title_text=f"{reducer.upper()} 2 (emb)", row=2, col=c)
    return fig, entries


def build_fig_spotlight(spec: ModelSpec, agent: AblationGinAgent, obs_list, reducer: str, top_k: int, width: int, height: int, title: str):
    """Elegant single-model figure: Row 1 = raw feature projection, Row 2 = learned embedding.
    Top-K pairs emphasized; others faded. No per-point text. Stacked layout with a simple legend.
    """
    fig = make_subplots(rows=2, cols=1)
    inter = GINInterpreter(agent, device=str(agent.device))

    learned_all, scores_all, texts_all, rawfeat_all = [], [], [], []
    for obs_name, obs_tensor in obs_list:
        L, S, pairs = inter.model_learned_pair_embeddings(obs_tensor, include_graph=False)
        learned_all.append(L); scores_all.append(S)
        texts_all += [f"{obs_name}: t{ti}-v{vj}" for (ti, vj) in pairs]
        emb = inter.extract_embeddings_and_features(obs_tensor)
        t_len = emb['task_features']['length']; t_mem = emb['task_features']['memory_req']
        v_speed = emb['vm_features']['speed']; v_mem = emb['vm_features']['memory_total']
        for (ti, vj) in pairs:
            rawfeat_all.append([
                float(t_len[ti]), float(t_mem[ti]), float(v_speed[vj]), float(v_mem[vj]), float(t_len[ti]/(v_speed[vj]+1e-8))
            ])

    L = np.concatenate(learned_all, axis=0) if learned_all else np.zeros((0,2))
    S = np.concatenate(scores_all, axis=0) if scores_all else np.zeros((0,))
    R = np.asarray(rawfeat_all, dtype=float) if rawfeat_all else np.zeros((0,2))
    Z_emb = reduce_2d(L, reducer) if L.shape[0] else np.zeros((0,2))
    Z_raw = reduce_2d(R, reducer) if R.shape[0] else np.zeros((0,2))

    k = max(1, min(int(top_k), len(S)))
    s_for_rank = np.nan_to_num(S, nan=-1e9)
    top_idx = np.argsort(-s_for_rank)[:k]
    mask = np.zeros(len(S), dtype=bool); mask[top_idx] = True
    lo_idx = np.where(~mask)[0]; hi_idx = top_idx

    # Styling: same color family, use opacity to distinguish
    base_color = 'rgba(33, 150, 243, 1.0)'  # blue
    bg_color = base_color
    bg_opacity = 0.15
    hi_opacity = 1.0
    hi_size = 12
    bg_size = 6

    # Row 1: Raw space
    if Z_raw.size:
        if lo_idx.size:
            fig.add_trace(go.Scatter(x=Z_raw[lo_idx,0], y=Z_raw[lo_idx,1], mode='markers', name='Others',
                                     marker=dict(size=bg_size, color=bg_color, opacity=bg_opacity), hoverinfo='skip', showlegend=True), row=1, col=1)
        if hi_idx.size:
            fig.add_trace(go.Scatter(x=Z_raw[hi_idx,0], y=Z_raw[hi_idx,1], mode='markers', name=f'Top-{k}',
                                     marker=dict(size=hi_size, color=base_color, opacity=hi_opacity, line=dict(width=1, color='black')),
                                     hoverinfo='skip', showlegend=True), row=1, col=1)

    # Row 2: Embedding space
    if Z_emb.size:
        if lo_idx.size:
            fig.add_trace(go.Scatter(x=Z_emb[lo_idx,0], y=Z_emb[lo_idx,1], mode='markers', name='Others',
                                     marker=dict(size=bg_size, color=bg_color, opacity=bg_opacity), hoverinfo='skip', showlegend=False), row=2, col=1)
        if hi_idx.size:
            fig.add_trace(go.Scatter(x=Z_emb[hi_idx,0], y=Z_emb[hi_idx,1], mode='markers', name=f'Top-{k}',
                                     marker=dict(size=hi_size, color=base_color, opacity=hi_opacity, line=dict(width=1, color='black')),
                                     hoverinfo='skip', showlegend=False), row=2, col=1)

    # Clean layout: no main title or subplot titles; single legend
    fig.update_layout(height=height, width=width, showlegend=True,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(title_text=f"{reducer.upper()} 1 (raw)", row=1, col=1)
    fig.update_yaxes(title_text=f"{reducer.upper()} 2 (raw)", row=1, col=1)
    fig.update_xaxes(title_text=f"{reducer.upper()} 1 (emb)", row=2, col=1)
    fig.update_yaxes(title_text=f"{reducer.upper()} 2 (emb)", row=2, col=1)
    return fig

def write_topk_csvs(entries, out_base: Path):
    # Per-model CSV
    pat = re.compile(r"^([^:]+): t(\d+)-v(\d+)$")
    for (label, Z, S, texts, top_idx) in entries:
        out_csv = out_base.with_name(out_base.name + f"_{label}_topk.csv")
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model','rank','obs','task','vm','pair','score'])
            writer.writeheader()
            for rank, i in enumerate(top_idx, start=1):
                txt = texts[i]
                m = pat.match(txt)
                obs_name, ti, vj = (m.group(1), int(m.group(2)), int(m.group(3))) if m else ("?", -1, -1)
                writer.writerow({'model': label, 'rank': rank, 'obs': obs_name, 'task': ti, 'vm': vj, 'pair': txt, 'score': float(S[i])})
    # Combined union CSV with presence flags
    union_pairs = {}
    labels = [e[0] for e in entries]
    for (label, Z, S, texts, top_idx) in entries:
        for i in top_idx:
            union_pairs.setdefault(texts[i], {}).update({label: float(S[i])})
    union_csv = out_base.with_name(out_base.name + "_union_topk.csv")
    with open(union_csv, 'w', newline='') as f:
        fieldnames = ['pair'] + [f'in_{lab}' for lab in labels] + [f'score_{lab}' for lab in labels]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair, present in union_pairs.items():
            row = {'pair': pair}
            for lab in labels:
                row[f'in_{lab}'] = lab in present
                row[f'score_{lab}'] = present.get(lab, '')
            writer.writerow(row)


def write_metrics_csv(metrics_by_label: dict, out_base: Path):
    """Write a separate CSV summarizing evaluation metrics per model for this category."""
    out_csv = out_base.with_name(out_base.name + "_metrics.csv")
    with open(out_csv, 'w', newline='') as f:
        fieldnames = ['model','makespan','total_energy','avg_active_energy','avg_idle_energy','avg_active_plus_idle_energy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_label, m in metrics_by_label.items():
            writer.writerow({
                'model': model_label,
                'makespan': m.get('makespan'),
                'total_energy': m.get('total_energy'),
                'avg_active_energy': m.get('avg_active_energy'),
                'avg_idle_energy': m.get('avg_idle_energy'),
                'avg_active_plus_idle_energy': m.get('avg_active_plus_idle_energy')
            })


def main():
    ap = argparse.ArgumentParser(description="Compare 4 models (edge-only embeddings) on linear/gnp/mixed and export images + CSVs")
    ap.add_argument('--models', type=str, nargs='+', required=True,
                    help="Repeatable model specs: label=/path/to/model.pt:variant. Variant in {baseline,no_global_actor}.")
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda','mps'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--workflow_count', type=int, default=10)
    ap.add_argument('--top_k', type=int, default=3)
    ap.add_argument('--reducer', type=str, default='tsne', choices=['tsne','pca'])
    ap.add_argument('--out_prefix', type=str, default='figures/compare4_edge', help='Prefix for output files')
    ap.add_argument('--formats', nargs='+', default=['png'], choices=['png','svg','pdf','jpg','jpeg','webp'])
    ap.add_argument('--width', type=int, default=1800)
    ap.add_argument('--height', type=int, default=520)
    ap.add_argument('--scale', type=float, default=3.0)
    ap.add_argument('--also_html', action='store_true')
    # Spotlight options
    ap.add_argument('--spotlight_label', type=str, default=None, help='Label of the model to render a single-model spotlight figure')
    ap.add_argument('--spotlight_top_k', type=int, default=5, help='Top-K to emphasize in the spotlight figure')
    args = ap.parse_args()

    device = torch.device(args.device)
    specs = [parse_model_arg(s) for s in args.models]
    if len(specs) != 4:
        raise SystemExit(f"Please provide exactly 4 --models (got {len(specs)}).")

    # Load models
    models: List[Tuple[ModelSpec, AblationGinAgent]] = []
    for spec in specs:
        agent = load_agent(spec, device)
        models.append((spec, agent))

    # Prepare observations
    obs_linear = [('linear', build_obs(args.seed, args.device, args.workflow_count, 'linear'))]
    obs_gnp = [('gnp', build_obs(args.seed, args.device, args.workflow_count, 'gnp'))]
    obs_mixed = [
        ('linear', build_obs(args.seed, args.device, max(1, args.workflow_count//2), 'linear')),
        ('gnp', build_obs(args.seed+1, args.device, max(1, args.workflow_count - args.workflow_count//2), 'gnp')),
    ]

    def save_all(obs_list, suffix: str, title: str):
        fig, entries = build_fig_edge_only(models, obs_list, args.reducer, args.top_k, args.width, args.height, title)
        base = Path(args.out_prefix).with_name(Path(args.out_prefix).stem + f'_{suffix}')
        base.parent.mkdir(parents=True, exist_ok=True)
        # Compute quick eval metrics per model on provided obs_list (average if multiple obs, e.g., mixed)
        dag_methods = [name for name, _ in obs_list]
        metrics_by_label = {}
        for (spec, agent) in models:
            vals = []
            for dm in dag_methods:
                try:
                    vals.append(quick_eval(agent, dm, iterations=3, seed_base=args.seed))
                except Exception:
                    pass
            if vals:
                # average numeric fields
                mk = np.mean([v['makespan'] for v in vals if v.get('makespan') is not None]) if any(v.get('makespan') is not None for v in vals) else None
                te = np.mean([v['total_energy'] for v in vals if v.get('total_energy') is not None]) if any(v.get('total_energy') is not None for v in vals) else None
                ae = np.mean([v['avg_active_energy'] for v in vals if v.get('avg_active_energy') is not None]) if any(v.get('avg_active_energy') is not None for v in vals) else None
                ie = np.mean([v['avg_idle_energy'] for v in vals if v.get('avg_idle_energy') is not None]) if any(v.get('avg_idle_energy') is not None for v in vals) else None
                ia = np.mean([v['avg_active_plus_idle_energy'] for v in vals if v.get('avg_active_plus_idle_energy') is not None]) if any(v.get('avg_active_plus_idle_energy') is not None for v in vals) else None

                metrics_by_label[spec.label] = {
                    'makespan': mk,
                    'total_energy': te,
                    'avg_active_energy': ae,
                    'avg_idle_energy': ie,
                    'avg_active_plus_idle_energy': ia,
                }
            else:
                metrics_by_label[spec.label] = {}
        # Images
        for fmt in args.formats:
            out_path = base.with_suffix('.' + fmt)
            try:
                fig.write_image(str(out_path), format=fmt, width=args.width, height=args.height, scale=args.scale)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Warning: could not write {out_path} ({e}). Install 'kaleido' for static image export: pip install -U kaleido")
        if args.also_html:
            fig.write_html(str(base.with_suffix('.html')))
        # CSVs
        write_topk_csvs(entries, base)
        write_metrics_csv(metrics_by_label, base)

    save_all(obs_linear, 'linear', f'Top-{args.top_k} pairs — Edge embeddings — linear (4 models)')
    save_all(obs_gnp, 'gnp', f'Top-{args.top_k} pairs — Edge embeddings — gnp (4 models)')
    save_all(obs_mixed, 'mixed', f'Top-{args.top_k} pairs — Edge embeddings — mixed (4 models)')

    # Optional single-model spotlight for GNP
    if args.spotlight_label is not None:
        # Find the model by label
        model_map = {spec.label: agent for (spec, agent) in models}
        if args.spotlight_label in model_map:
            agent = model_map[args.spotlight_label]
            spec = next(s for (s, a) in models if s.label == args.spotlight_label)
            fig_sp = build_fig_spotlight(spec, agent, obs_gnp, args.reducer, args.spotlight_top_k, 900, 900,
                                         title=f"{spec.label} — GNP (Top-{args.spotlight_top_k})")
            base_sp = Path(args.out_prefix).with_name(Path(args.out_prefix).stem + f'_gnp_spotlight_{spec.label}')
            try:
                fig_sp.write_image(str(base_sp.with_suffix('.png')), width=900, height=900, scale=3)
                print(f"Saved: {base_sp.with_suffix('.png')}")
            except Exception as e:
                print(f"Warning: could not write {base_sp.with_suffix('.png')} ({e}). Install 'kaleido' for static export: pip install -U kaleido")
            if args.also_html:
                fig_sp.write_html(str(base_sp.with_suffix('.html')))
        else:
            print(f"Warning: --spotlight_label '{args.spotlight_label}' not found among labels {[s.label for (s,_) in models]}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
