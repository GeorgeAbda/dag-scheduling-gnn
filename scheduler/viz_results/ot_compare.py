import argparse
import json
import sys
import math
import csv
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional dependency: POT (Python Optimal Transport)
try:
    import ot  # pip install POT
    POT_AVAILABLE = True
except Exception:
    POT_AVAILABLE = False

from scheduler.viz_results.embeddings import GINInterpreter
from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as TrainArgs,
    _make_test_env,
    _test_agent,
)


def pairwise_sqeuclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    XX = (X ** 2).sum(axis=1)[:, None]
    YY = (Y ** 2).sum(axis=1)[None, :]
    return np.maximum(XX + YY - 2.0 * (X @ Y.T), 0.0)


def softmax_scores_to_weights(scores: np.ndarray, temperature: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    s = np.nan_to_num(scores.astype(float), nan=-1e9)
    s = s / max(temperature, 1e-6)
    s = s - np.max(s)
    p = np.exp(s)
    p = p + eps  # avoid zeros
    z = np.sum(p)
    if z <= 0:
        return np.ones_like(p) / max(1, len(p))
    return p / z


def sinkhorn_ot_cost(Xa: np.ndarray, Xb: np.ndarray, wa: np.ndarray, wb: np.ndarray, reg: float = 0.05, num_iter: int = 5000):
    if not POT_AVAILABLE:
        raise RuntimeError("POT (Python Optimal Transport) is required. Install with: pip install POT")
    C = pairwise_sqeuclidean(Xa, Xb)
    # Normalize cost for numeric stability
    std = C.std()
    if std > 0:
        C = C / std
    # Increase iterations and set a stricter stop threshold for stability
    G = ot.sinkhorn(wa, wb, C, reg, numItermax=int(num_iter), stopThr=1e-9, verbose=False)
    cost = float((G * C).sum())
    return cost, G, C


def fgw_cost(Xa: np.ndarray, Xb: np.ndarray, wa: np.ndarray, wb: np.ndarray, alpha: float = 0.5, num_iter: int = 100, armijo: bool = False):
    """Compute Fused Gromov–Wasserstein (FGW) cost and coupling.
    Xa, Xb: feature embeddings (Nxa x d), (Nxb x d)
    wa, wb: distributions (sum to 1)
    alpha: trade-off between structure (Gromov) and feature (Wasserstein) costs
    """
    if not POT_AVAILABLE:
        raise RuntimeError("POT (Python Optimal Transport) is required. Install with: pip install POT")
    # Feature cost matrix between spaces
    M = pairwise_sqeuclidean(Xa, Xb)
    # Structural cost matrices within each space
    C1 = pairwise_sqeuclidean(Xa, Xa)
    C2 = pairwise_sqeuclidean(Xb, Xb)
    # Normalize costs for numerical stability
    for mat in (M, C1, C2):
        std = mat.std()
        if std > 0:
            mat /= std
    # POT FGW solver
    G = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, wa, wb, alpha=alpha, armijo=armijo, log=False, numItermax=num_iter
    )
    # Compute FGW objective value (approximate) as <G, M> + alpha * GW_term
    # POT provides fgw loss via fused_gromov_wasserstein2 (returns loss, G)
    loss, G2 = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, wa, wb, alpha=alpha, armijo=armijo, log=False, numItermax=num_iter
    )
    return float(loss), G2


def build_env_and_obs(seed: int = 1, device: str = "cpu", dag_method: str = "linear", workflow_count: int = 8):
    targs = TrainArgs()
    targs.seed = int(seed)
    targs.device = device
    targs.dataset.workflow_count = int(workflow_count)
    targs.dataset.dag_method = dag_method  # choose obs type to probe graph context
    env = _make_test_env(targs)
    obs, _ = env.reset()
    # One step to get a non-trivial obs if desired
    # obs, _, _, _, _ = env.step(env.action_space.sample())
    env.close()
    # Map to agent tensor format: AblationGinAgent expects mapper input shape [B, ...]
    # Ensure float32 dtype to match model weights (avoid Double vs Float errors)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(dtype=torch.float32)
    return obs_tensor, targs


def load_agent(ckpt_path: Path, device: torch.device, variant: AblationVariant | None = None) -> AblationGinAgent:
    """Load an AblationGinAgent checkpoint, auto-detecting whether the actor used
    the global graph embedding (affects edge_scorer input dimension).
    """
    state = torch.load(str(ckpt_path), map_location=device)
    # Heuristic: inspect first Linear weight of edge_scorer
    in_features = None
    for k, v in state.items():
        if k.endswith('actor.edge_scorer.0.weight') and hasattr(v, 'shape') and len(v.shape) == 2:
            in_features = int(v.shape[1])
            break
    # Default embedding_dim in AblationGinAgent is 8, so:
    # in_features == 16 -> no_global_actor (2*8)
    # in_features == 24 -> baseline with global (3*8)
    use_global = True if in_features is None else (in_features % 8 == 0 and (in_features // 8) >= 3)

    # Allow explicit override via provided variant
    if variant is None:
        if use_global:
            variant = AblationVariant(name="baseline", use_actor_global_embedding=True)
        else:
            variant = AblationVariant(name="no_global_actor", use_actor_global_embedding=False)

    agent = AblationGinAgent(device=device, variant=variant)
    try:
        agent.load_state_dict(state, strict=False)
    except RuntimeError:
        # Retry with flipped global flag if mismatch persists
        alt = AblationVariant(name="no_global_actor" if variant.use_actor_global_embedding else "baseline",
                              use_actor_global_embedding=not variant.use_actor_global_embedding,
                              gin_num_layers=variant.gin_num_layers,
                              use_batchnorm=variant.use_batchnorm,
                              use_task_dependencies=variant.use_task_dependencies,
                              graph_type=variant.graph_type,
                              gat_heads=variant.gat_heads,
                              gat_dropout=variant.gat_dropout)
        agent = AblationGinAgent(device=device, variant=alt)
        agent.load_state_dict(state, strict=False)
    return agent


def extract_pairs_via_interpreter(agent: AblationGinAgent, obs_tensor: torch.Tensor, device: torch.device, include_graph: bool, obs_tag: str):
    """Use GINInterpreter.model_learned_pair_embeddings to obtain per-pair embeddings and scores.
    Returns labeled pairs in the form f"{obs_tag}: t{ti}-v{vj}" for uniqueness across mixed observations.
    """
    interpreter = GINInterpreter(agent, device=str(device))
    learned_pairs, scores, pairs = interpreter.model_learned_pair_embeddings(obs_tensor, include_graph=include_graph)
    labels = [f"{obs_tag}: t{ti}-v{vj}" for (ti, vj) in pairs]
    return labels, np.asarray(learned_pairs, dtype=float), np.asarray(scores, dtype=float)


def quick_eval(agent: AblationGinAgent, base_args: TrainArgs, dag_method: str, iterations: int = 3) -> dict:
    args = TrainArgs(**vars(base_args))
    args.dataset.dag_method = dag_method
    args.test_iterations = int(iterations)
    mk, eobs, etot, mm = _test_agent(agent, args)
    return {
        'makespan': float(mk),
        'total_energy': float(etot if etot > 0 else eobs),
        'avg_active_energy': float(mm.get('avg_active_energy')) if mm and (mm.get('avg_active_energy') is not None) else None,
        'avg_idle_energy': float(mm.get('avg_idle_energy')) if mm and (mm.get('avg_idle_energy') is not None) else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-model OT between learned pair embeddings and action distributions (edge vs edge+graph), with decision-aware metrics and CSVs")
    ap.add_argument("--model_a", type=str, required=True, help="Path to first model checkpoint (e.g., baseline)")
    ap.add_argument("--model_b", type=str, required=True, help="Path to second model checkpoint (e.g., no_global_actor)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device")
    ap.add_argument("--seed", type=int, default=1, help="Seed for env/obs")
    ap.add_argument("--reg", type=float, default=0.05, help="Sinkhorn entropic regularization")
    ap.add_argument("--sinkhorn_iter", type=int, default=5000, help="Max iterations for Sinkhorn")
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax weights (higher = smoother)")
    ap.add_argument("--dim_reducer", type=str, default="pca", choices=["pca","tsne"], help="2D reducer for visualization")
    ap.add_argument("--top_edges", type=int, default=200, help="Max transport edges to draw in plot")
    ap.add_argument("--top_k", type=int, default=3, help="Top-K pairs for decision-aware metrics")
    ap.add_argument("--modes", nargs='+', default=["edge","edge_graph"], choices=["edge","edge_graph"], help="Which embedding modes to run")
    ap.add_argument("--categories", nargs='+', default=["linear","gnp","mixed"], choices=["linear","gnp","mixed"], help="Observation categories to compare")
    ap.add_argument("--workflow_count", type=int, default=10, help="Workflows per observation (per category element)")
    ap.add_argument("--out_prefix", type=str, default="figures/ot_compare", help="Prefix for outputs (plots and CSVs)")
    ap.add_argument("--save_html", action="store_true", help="Also save interactive HTML plots")
    # FGW optional
    ap.add_argument("--alpha", type=float, default=0.5, help="FGW alpha in [0,1]")
    ap.add_argument("--fgw_iter", type=int, default=200, help="Max iterations for FGW solver")
    args = ap.parse_args()

    if not POT_AVAILABLE:
        print("Error: POT (Python Optimal Transport) not installed. Run: pip install POT", file=sys.stderr)
        return 2

    device = torch.device(args.device)

    # Load both agents
    mA = load_agent(Path(args.model_a), device)
    mB = load_agent(Path(args.model_b), device)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def build_obs_list(category: str):
        if category == 'mixed':
            obs1, _ = build_env_and_obs(seed=args.seed, device=args.device, dag_method='linear', workflow_count=int(args.workflow_count//2 or 1))
            obs2, _ = build_env_and_obs(seed=args.seed+1, device=args.device, dag_method='gnp', workflow_count=int(args.workflow_count - args.workflow_count//2 or 1))
            return [('linear', obs1), ('gnp', obs2)]
        else:
            obs, _ = build_env_and_obs(seed=args.seed, device=args.device, dag_method=category, workflow_count=int(args.workflow_count))
            return [(category, obs)]

    def decision_metrics(weightsA: np.ndarray, weightsB: np.ndarray, G: np.ndarray, labels: list[str], K: int):
        # Top-K indices per model by weights
        k = max(1, min(K, len(weightsA), len(weightsB)))
        idxA = np.argsort(-weightsA)[:k]
        idxB = np.argsort(-weightsB)[:k]
        setA = set(labels[i] for i in idxA)
        setB = set(labels[i] for i in idxB)
        overlap = len(setA.intersection(setB)) / float(k)
        # OT mass alignment from A_topK to B_topK
        maskA = np.zeros(len(labels), dtype=bool); maskA[idxA] = True
        maskB = np.zeros(len(labels), dtype=bool); maskB[idxB] = True
        mAB = float(G[np.ix_(maskA, maskB)].sum())
        # Entropy of transport rows from A_topK
        entropies = []
        row_sums = G.sum(axis=1) + 1e-12
        for i in idxA:
            q = G[i] / row_sums[i]
            q = np.where(q > 0, q, 1e-12)
            ent = -float((q * np.log(q)).sum())
            entropies.append(ent)
        H_A = float(np.mean(entropies)) if entropies else float('nan')
        return {
            'topk_overlap': overlap,
            'ot_mass_A_to_B_topk': mAB,
            'avg_entropy_from_A_topk': H_A,
            'k': k,
        }, idxA, idxB

    def write_csv(path: Path, header: list[str], rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    summary = {
        'model_a': str(args.model_a),
        'model_b': str(args.model_b),
        'reg': float(args.reg),
        'temperature': float(args.temperature),
        'results': {}
    }

    # Run across categories and modes
    for category in args.categories:
        obs_list = build_obs_list(category)
        for mode in args.modes:
            include_graph = (mode == 'edge_graph')
            # Extract and concatenate across obs_list
            labels = []
            XA_list = []; XB_list = []
            sA_list = []; sB_list = []
            for obs_tag, obs_tensor in obs_list:
                la, Xa, sa = extract_pairs_via_interpreter(mA, obs_tensor, device, include_graph=include_graph, obs_tag=obs_tag)
                lb, Xb, sb = extract_pairs_via_interpreter(mB, obs_tensor, device, include_graph=include_graph, obs_tag=obs_tag)
                # align by labels
                map_b = {lab: i for i, lab in enumerate(lb)}
                common = [lab for lab in la if lab in map_b]
                if not common:
                    continue
                idx_a = [la.index(l) for l in common]
                idx_b = [map_b[l] for l in common]
                labels += common
                XA_list.append(Xa[idx_a])
                XB_list.append(Xb[idx_b])
                sA_list.append(sa[idx_a])
                sB_list.append(sb[idx_b])
            if not labels:
                print(f"Warning: no common pairs for {category}/{mode}")
                continue
            XA = np.vstack(XA_list); XB = np.vstack(XB_list)
            sA = np.concatenate(sA_list); sB = np.concatenate(sB_list)
            wA = softmax_scores_to_weights(sA, temperature=float(args.temperature))
            wB = softmax_scores_to_weights(sB, temperature=float(args.temperature))

            # Sinkhorn OT
            cost, G, C = sinkhorn_ot_cost(XA, XB, wA, wB, reg=float(args.reg), num_iter=int(args.sinkhorn_iter))
            # Decision-aware metrics
            dec_metrics, idxA, idxB = decision_metrics(wA, wB, G, labels, int(args.top_k))

            # Barycentric projection of A's top-K into B space and displacement wrt aligned XB
            # Compute bary_i = sum_j T[i,j] XB[j] / sum_j T[i,j]
            row_sums = G.sum(axis=1) + 1e-12
            top_rows = G[idxA]
            denom = row_sums[idxA][:, None]
            bary = (top_rows @ XB) / denom
            # Displacement vectors in B space: from the aligned XB[i] to its barycentric image
            XB_ref = XB[idxA]
            disp = bary - XB_ref
            disp_sq = (disp * disp).sum(axis=1)
            disp_norm = np.sqrt(disp_sq)
            bary_msd = float(disp_sq.mean()) if disp_sq.size > 0 else float('nan')
            bary_mean_norm = float(disp_norm.mean()) if disp_norm.size > 0 else float('nan')
            # Angle relative to global PCA axes of XB
            Xc = XB - XB.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            pc = Vt[:2]  # [2, d]
            comps = disp @ pc.T  # [k, 2]
            angles = np.degrees(np.arctan2(comps[:, 1], comps[:, 0])) if comps.shape[1] >= 2 else np.zeros(disp.shape[0])
            mean_abs_angle = float(np.mean(np.abs(angles))) if angles.size > 0 else float('nan')

            # Store summary
            key = f"{category}_{mode}"
            summary['results'][key] = {
                'sinkhorn_cost': float(cost),
                **dec_metrics,
                'num_pairs': int(len(labels)),
                'bary_msd': bary_msd,
                'bary_mean_norm': bary_mean_norm,
                'bary_mean_abs_angle_deg': mean_abs_angle,
            }

            # CSVs
            base = out_prefix.with_name(out_prefix.stem + f"_{category}_{mode}")
            rows = []
            for i, lab in enumerate(labels):
                rows.append({'pair': lab, 'w_model_a': float(wA[i]), 'w_model_b': float(wB[i]), 'rank_a': int(np.where(np.argsort(-wA)==i)[0][0])+1, 'rank_b': int(np.where(np.argsort(-wB)==i)[0][0])+1})
            write_csv(base.with_name(base.name + "_weights.csv"), ['pair','w_model_a','w_model_b','rank_a','rank_b'], rows)
            write_csv(base.with_name(base.name + "_summary.csv"), list(summary['results'][key].keys()), [summary['results'][key]])
            # Per-topK barycentric CSV
            bary_rows = []
            for local_idx, i in enumerate(idxA):
                lab = labels[i]
                bary_rows.append({
                    'pair': lab,
                    'disp_sq': float(disp_sq[local_idx]),
                    'disp_norm': float(disp_norm[local_idx]),
                    'angle_deg': float(angles[local_idx]) if angles.size > local_idx else float('nan'),
                    'pc1_comp': float(comps[local_idx, 0]) if comps.shape[1] >= 1 else float('nan'),
                    'pc2_comp': float(comps[local_idx, 1]) if comps.shape[1] >= 2 else float('nan'),
                })
            write_csv(base.with_name(base.name + "_barycentric_topk.csv"), ['pair','disp_sq','disp_norm','angle_deg','pc1_comp','pc2_comp'], bary_rows)

            # Visualization (2D projection + transport)
            def reduce_both(A: np.ndarray, B: np.ndarray, method: str = "pca"):
                Xcat = np.vstack([A, B])
                if method == "tsne":
                    try:
                        from sklearn.manifold import TSNE
                        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, Xcat.shape[0]-1))).fit_transform(Xcat)
                    except Exception:
                        method = "pca"
                if method == "pca":
                    Xc = Xcat - Xcat.mean(axis=0, keepdims=True)
                    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                    Z = Xc @ Vt[:2].T
                return Z[:A.shape[0]], Z[A.shape[0]:]

            ZA, ZB = reduce_both(XA, XB, method=args.dim_reducer)
            fig = make_subplots(rows=1, cols=1, subplot_titles=(f"OT mapping — {category} — {mode}",))
            fig.add_trace(go.Scatter(x=ZA[:,0], y=ZA[:,1], mode='markers', name='A',
                                     marker=dict(color='rgba(31,119,180,0.65)', size=6)))
            fig.add_trace(go.Scatter(x=ZB[:,0], y=ZB[:,1], mode='markers', name='B',
                                     marker=dict(color='rgba(255,127,14,0.65)', size=6)))
            # Draw top transport edges
            try:
                # pick largest entries of G
                flat_idx = np.argsort(G.flatten())[::-1][:int(args.top_edges)]
                Ia = flat_idx // G.shape[1]
                Jb = flat_idx % G.shape[1]
                xs = []; ys = []
                for i, j in zip(Ia, Jb):
                    xs += [ZA[i,0], ZB[j,0], None]
                    ys += [ZA[i,1], ZB[j,1], None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(0,0,0,0.15)', width=1), showlegend=False))
            except Exception:
                pass
            fig.update_layout(height=600, width=1000)
            out_img = base.with_suffix('.png')
            try:
                fig.write_image(str(out_img), width=1000, height=600, scale=2)
                print(f"Saved: {out_img}")
            except Exception as e:
                print(f"Warning: could not write {out_img} ({e}). Install 'kaleido' for static export.")
            if args.save_html:
                fig.write_html(str(base.with_suffix('.html')))
                print(f"Saved: {base.with_suffix('.html')}")


    # Save summary JSON next to out_prefix
    out_json = out_prefix.with_suffix('.json')
    with out_json.open('w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_json}")

    # 9) Print console summary
    print("\nCross-model OT summary:")
    print(f"  model_a: {args.model_a}")
    print(f"  model_b: {args.model_b}")
    for k, v in summary['results'].items():
        print(f"  {k}: Sinkhorn {v['sinkhorn_cost']:.6f}, overlap {v['topk_overlap']:.3f}, mAB {v['ot_mass_A_to_B_topk']:.3f}, H_A {v['avg_entropy_from_A_topk']:.3f}")

if __name__ == "__main__":
    raise SystemExit(main())
