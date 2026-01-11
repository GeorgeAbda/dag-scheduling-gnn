import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
import scheduler.config.settings as _cfg
import scheduler.dataset_generator.core.gen_vm as _gen_vm


LONG_CP_CFG = "data/rl_configs/train_long_cp_p08_seeds.json"
WIDE_CFG = "data/rl_configs/train_wide_p005_seeds.json"
OUT_DIR = os.environ.get("OUT_DIR", "logs/dags")
OUT_PNG = os.path.join(OUT_DIR, "state_visitation_pi_ei_agent.png")


# -------------------------------
# Dataset helpers
# -------------------------------

def _load_seedset(path: str, source: str = "eval") -> Tuple[List[int], dict]:
    with open(path, "r") as f:
        cfg = json.load(f)
    section = cfg.get(source, {}) if isinstance(cfg, dict) else {}
    seeds: List[int] = list(section.get("seeds", []))
    ds: dict = dict(section.get("dataset", {}))
    return seeds, ds


def _mk_dataset(seed: int, ds: dict):
    return generate_dataset(
        seed=seed,
        host_count=int(ds.get("host_count", 10)),
        vm_count=int(ds.get("vm_count", 10)),
        max_memory_gb=int(ds.get("max_memory_gb", 128)),
        min_cpu_speed_mips=int(ds.get("min_cpu_speed", 500)),
        max_cpu_speed_mips=int(ds.get("max_cpu_speed", 5000)),
        workflow_count=int(ds.get("workflow_count", 1)),
        dag_method=str(ds.get("dag_method", "gnp")),
        gnp_min_n=int(ds.get("gnp_min_n", 12)),
        gnp_max_n=int(ds.get("gnp_max_n", 24)),
        task_length_dist=str(ds.get("task_length_dist", "normal")),
        min_task_length=int(ds.get("min_task_length", 500)),
        max_task_length=int(ds.get("max_task_length", 100000)),
        task_arrival=str(ds.get("task_arrival", "static")),
        arrival_rate=float(ds.get("arrival_rate", 3)),
        vm_rng_seed=0,
        gnp_p=float(ds.get("gnp_p")) if ds.get("gnp_p") is not None else None,
    )


def _topo_layers(children: Dict[int, List[int]]) -> List[List[int]]:
    indeg: Dict[int, int] = {u: 0 for u in children}
    for u, vs in children.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    frontier: List[int] = [u for u in children if indeg.get(u, 0) == 0]
    layers: List[List[int]] = []
    seen: set[int] = set()
    while frontier:
        cur = list(frontier)
        layers.append(cur)
        frontier = []
        for u in cur:
            if u in seen:
                continue
            seen.add(u)
            for v in children.get(u, []):
                indeg[v] = indeg.get(v, 0) - 1
                if indeg[v] == 0:
                    frontier.append(v)
    if not layers:
        return [list(children.keys())]
    return layers


def _max_width_from_dataset(dataset) -> int:
    wf = dataset.workflows[0]
    ch: Dict[int, List[int]] = {int(t.id): list(t.child_ids) for t in wf.tasks}
    layers = _topo_layers(ch)
    return max((len(L) for L in layers), default=1)


# -------------------------------
# Metrics from wrapper.prev_obs
# -------------------------------

def _ready_count(prev_obs) -> int:
    T = len(prev_obs.task_observations)
    cnt = 0
    for idx, t in enumerate(prev_obs.task_observations):
        if idx == 0 or idx == T - 1:
            continue
        if bool(t.is_ready) and (t.assigned_vm_id is None):
            cnt += 1
    return int(cnt)


def _energy_intensity_index_prev(prev_obs) -> float:
    total_span = 0.0
    active = 0.0
    for v in prev_obs.vm_observations:
        idle = float(getattr(v, "host_power_idle_watt", 0.0))
        peak = float(getattr(v, "host_power_peak_watt", idle))
        span = max(0.0, peak - idle)
        total_span += span
        cores = max(1.0, float(getattr(v, "cpu_cores", 1.0)))
        avail = float(getattr(v, "available_cpu_cores", cores))
        used = max(0.0, cores - avail)
        u = min(1.0, max(0.0, used / cores))
        active += span * u
    if total_span <= 0.0:
        return 0.0
    return float(active / total_span)


# Instantaneous active power rate (Watts-equivalent, up to a constant factor), i.e., numerator of EI
def _active_power_rate(prev_obs) -> float:
    total = 0.0
    for v in prev_obs.vm_observations:
        idle = float(getattr(v, "host_power_idle_watt", 0.0))
        peak = float(getattr(v, "host_power_peak_watt", idle))
        span = max(0.0, peak - idle)
        cores = max(1.0, float(getattr(v, "cpu_cores", 1.0)))
        avail = float(getattr(v, "available_cpu_cores", cores))
        used = max(0.0, cores - avail)
        u = min(1.0, max(0.0, used / cores))
        total += span * u
    return float(total)

# -------------------------------
# Agent helpers
# -------------------------------

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "cpu") and torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cpu")


def _load_agent(model_path: str | None, init_seed: int | None = None, jitter_std: float = 0.0) -> GinAgent:
    device = _pick_device()
    # Optional deterministic init per episode
    if init_seed is not None:
        try:
            torch.manual_seed(int(init_seed))
            np.random.seed(int(init_seed))
        except Exception:
            pass

    def _filter_and_load(m: torch.nn.Module, sd_raw: dict) -> tuple[int, int]:
        target_sd = m.state_dict()
        ok = {}
        for k, v in sd_raw.items():
            if k in target_sd and isinstance(v, torch.Tensor) and target_sd[k].shape == v.shape:
                ok[k] = v
        missing = len(set(target_sd.keys()) - set(ok.keys()))
        m.load_state_dict(ok, strict=False)
        return len(ok), missing

    def _try_load_gin(sd_raw: dict) -> tuple[bool, GinAgent | None]:
        try:
            m = GinAgent(device=device)
            n_ok, n_miss = _filter_and_load(m, sd_raw)
            print(f"[load] GinAgent matched tensors: {n_ok}, missing: {n_miss}")
            return True, m
        except Exception as e:
            print(f"[load] GinAgent load failed: {e}")
            return False, None

    def _try_load_ablation(sd_raw: dict, path_hint: str) -> tuple[bool, GinAgent | None]:
        try:
            from scheduler.rl_model import ablation_gnn as AG
            # Heuristic variant detection from path and keys
            vname = "baseline"
            for cand in ["hetero", "gatv2", "sage", "pna", "transformer", "nnconv", "edgeconv"]:
                if cand in (path_hint or "").lower():
                    vname = cand
                    break
            # Infer conv layer count from keys
            import re
            max_idx = -1
            for k in sd_raw.keys():
                m = re.search(r"graph_network\.convs\.(\d+)\.", k)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
            gin_layers = max(1, max_idx + 1) if max_idx >= 0 else 3
            use_attnpool = any("_attn_pool" in k for k in sd_raw.keys())
            variant = AG.AblationVariant(name=vname, gin_num_layers=gin_layers, use_attention_pool=use_attnpool, graph_type=(vname if vname in {"hetero","gatv2","sage","pna","transformer","nnconv","edgeconv"} else "gin"))
            m = AG.AblationGinAgent(device=device, variant=variant, hidden_dim=32, embedding_dim=16)
            n_ok, n_miss = _filter_and_load(m, sd_raw)
            print(f"[load] AblationGinAgent[{variant.name}|layers={gin_layers}|attnpool={use_attnpool}] matched tensors: {n_ok}, missing: {n_miss}")
            return True, m
        except Exception as e:
            print(f"[load] Ablation load failed: {e}")
            return False, None

    # Default fresh agent
    agent: GinAgent | None = GinAgent(device=device)
    if model_path and os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd and all(isinstance(k, str) for k in sd["state_dict"].keys()):
            sd = sd["state_dict"]
        # Try ablation first if path hints it
        hint = (model_path or "")
        prefer_ablation = any(x in hint.lower() for x in ["hetero", "gatv2", "sage", "pna", "transformer", "nnconv", "edgeconv"]) or any("_attn_pool" in k for k in (sd.keys() if isinstance(sd, dict) else []))
        ok = False
        loaded: GinAgent | None = None
        if prefer_ablation:
            ok, loaded = _try_load_ablation(sd, hint)
        if not ok:
            ok, loaded = _try_load_gin(sd)
        if not ok:
            # Last attempt: ablation fallback even if not hinted
            ok, loaded = _try_load_ablation(sd, hint)
        if ok and loaded is not None:
            agent = loaded
        else:
            print("[load] Warning: using randomly initialized agent (could not load checkpoint).")
        # Optional: create small variants by adding Gaussian noise to weights
        if jitter_std and jitter_std > 0.0 and agent is not None:
            with torch.no_grad():
                for p in agent.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * float(jitter_std))
        print(f"Loaded model: {model_path} (jitter_std={jitter_std})")
    assert agent is not None
    agent.eval()
    return agent


# -------------------------------
# Rollout and plotting
# -------------------------------

def _rollout_collect_points_with_agent(dataset, agent: GinAgent) -> Tuple[List[float], List[float], List[Tuple[float, float]], List[float], List[float], List[float], List[np.ndarray], List[float], List[float], List[float], float]:
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(dataset=dataset, compute_metrics=False, dataset_episode_mode="single"))
    obs_np, _ = env.reset()

    width = max(1, _max_width_from_dataset(dataset))

    pis: List[float] = []
    eis: List[float] = []
    traj: List[Tuple[float, float]] = []
    ms_series: List[float] = []
    apr_series: List[float] = []
    cum_active_energy_series: List[float] = []
    feat_list: List[np.ndarray] = []
    ent_list: List[float] = []
    energy_reward_list: List[float] = []
    makespan_reward_list: List[float] = []
    cum_e = 0.0
    last_ms = None

    step_guard = 0
    while True and step_guard < 1000000:
        step_guard += 1
        prev = env.prev_obs  # EnvObservation
        pi = float(_ready_count(prev)) / float(width)
        ei = _energy_intensity_index_prev(prev)
        pis.append(pi)
        eis.append(ei)
        traj.append((pi, ei))
        # Collect mapped observation for PCA at decision time
        try:
            feat_list.append(np.array(obs_np, dtype=np.float32))
        except Exception:
            pass
        # Series (per decision)
        ms = float(prev.makespan())
        apr = _active_power_rate(prev)
        ms_series.append(ms)
        apr_series.append(apr)
        if last_ms is None:
            last_ms = ms
        dt = max(0.0, ms - last_ms)
        cum_e += apr * dt
        cum_active_energy_series.append(cum_e)
        last_ms = ms

        # Agent action
        x = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            a, _lp, ent, _v = agent.get_action_and_value(x)
            try:
                ent_list.append(float(ent.squeeze().cpu().numpy()))
            except Exception:
                try:
                    ent_list.append(float(ent))
                except Exception:
                    ent_list.append(float('nan'))
        obs_np, _, terminated, truncated, _info = env.step(int(a.item()))
        try:
            energy_reward_list.append(float(_info.get("dbg_energy_reward", np.nan)))
            makespan_reward_list.append(float(_info.get("dbg_makespan_reward", np.nan)))
        except Exception:
            energy_reward_list.append(float('nan'))
            makespan_reward_list.append(float('nan'))
        if terminated or truncated:
            break
    # Realized makespan at episode end: max completion time across all tasks
    try:
        final_ms = 0.0
        tobs = getattr(env.prev_obs, 'task_observations', [])
        for t in tobs:
            if t.completion_time is not None:
                final_ms = max(final_ms, float(t.completion_time))
    except Exception:
        final_ms = float('nan')

    return pis, eis, traj, ms_series, apr_series, cum_active_energy_series, feat_list, ent_list, energy_reward_list, makespan_reward_list, final_ms


def _plot_pca(
    X_list: List[np.ndarray],
    labels: List[str],
    pis: List[float],
    eis: List[float],
    out_style: str,
    out_pi: str,
    out_ei: str,
    max_points: int = 6000,
    ei_transform: str = "log1p",
):
    if not X_list:
        return
    X = np.asarray(X_list, dtype=np.float32)
    n = X.shape[0]
    labels_arr = np.asarray(labels)
    pis_arr = np.asarray(pis, dtype=float)
    eis_arr = np.asarray(eis, dtype=float)
    if n != labels_arr.shape[0] or n != pis_arr.shape[0] or n != eis_arr.shape[0]:
        m = min(n, labels_arr.shape[0], pis_arr.shape[0], eis_arr.shape[0])
        X = X[:m]
        labels_arr = labels_arr[:m]
        pis_arr = pis_arr[:m]
        eis_arr = eis_arr[:m]
        n = m
    if n == 0:
        return
    # Downsample if too many points
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        X = X[idx]
        labels_arr = labels_arr[idx]
        pis_arr = pis_arr[idx]
        eis_arr = eis_arr[idx]
        n = X.shape[0]
    # Remove zero-variance columns
    var = X.var(axis=0)
    keep = var > 1e-12
    X2 = X[:, keep] if np.any(keep) else X
    # Standardize
    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X2)
    # PCA -> 2D
    pca = PCA(n_components=2, random_state=0)
    Y = pca.fit_transform(Xz)
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {evr}")

    # Colors for styles
    style_colors = {"long_cp": "#2E7D32", "wide": "#1565C0"}
    # 1) Colored by style
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=200)
    for style, col in style_colors.items():
        mask = labels_arr == style
        if np.any(mask):
            ax.scatter(Y[mask, 0], Y[mask, 1], s=8, alpha=0.6, label=style, c=col)
    ax.set_title(f"PCA space by style (var={evr[0]:.2f}+{evr[1]:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_style)
    print(f"Saved: {out_style}")

    # 2) Colored by PI
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=200)
    sc = ax.scatter(Y[:, 0], Y[:, 1], s=8, c=pis_arr, cmap="viridis", alpha=0.7, vmin=0.0, vmax=1.0)
    ax.set_title("PCA space colored by PI")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="PI")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pi)
    print(f"Saved: {out_pi}")

    # 3) Colored by EI (optionally transformed)
    def _tx(y: np.ndarray) -> np.ndarray:
        if ei_transform == "log1p":
            return np.log1p(y)
        if ei_transform == "sqrt":
            return np.sqrt(y)
        return y
    ei_c = _tx(eis_arr)
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=200)
    sc = ax.scatter(Y[:, 0], Y[:, 1], s=8, c=ei_c, cmap="magma", alpha=0.7)
    ax.set_title(f"PCA space colored by EI [{ei_transform}]")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, shrink=0.8, label=f"EI [{ei_transform}]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_ei)
    print(f"Saved: {out_ei}")


def _dataset_structure_stats(dataset) -> tuple[int, int]:
    wf = dataset.workflows[0]
    children: Dict[int, List[int]] = {int(t.id): list(t.child_ids) for t in wf.tasks}
    layers = _topo_layers(children)
    max_w = max((len(L) for L in layers), default=1)
    n_layers = len(layers)
    return int(max_w), int(n_layers)


def _plot_structure_state_behavior_triptych(
    X_list: List[np.ndarray],
    labels: List[str],
    entropies: List[float],
    struct_long: List[tuple[int, int]],
    struct_wide: List[tuple[int, int]],
    out_path: str,
    max_points: int = 6000,
    bw_adjust: float = 1.0,
):
    if not X_list:
        return
    X = np.asarray(X_list, dtype=np.float32)
    labs = np.asarray(labels)
    ents = np.asarray(entropies, dtype=float)
    n = X.shape[0]
    if n != labs.shape[0] or n != ents.shape[0]:
        m = min(n, labs.shape[0], ents.shape[0])
        X = X[:m]; labs = labs[:m]; ents = ents[:m]
        n = m
    if n == 0:
        return
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        X = X[idx]; labs = labs[idx]; ents = ents[idx]
        n = X.shape[0]
    var = X.var(axis=0)
    keep = var > 1e-12
    X2 = X[:, keep] if np.any(keep) else X
    Xz = StandardScaler().fit_transform(X2)
    pca = PCA(n_components=2, random_state=0)
    Y = pca.fit_transform(Xz)
    evr = pca.explained_variance_ratio_

    # Prepare canvas
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 4.8), dpi=200)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.0, 1.2])

    # Panel A: Structure scatter (width vs layers)
    axA = fig.add_subplot(gs[0, 0])
    if struct_long:
        wL, lL = zip(*struct_long)
        axA.scatter(wL, lL, c="#2E7D32", s=20, alpha=0.6, label="LongCP")
    if struct_wide:
        wW, lW = zip(*struct_wide)
        axA.scatter(wW, lW, c="#1565C0", s=20, alpha=0.6, label="Wide")
    axA.set_title("Structure: max width vs #layers")
    axA.set_xlabel("Max layer width")
    axA.set_ylabel("# layers (CP length in layers)")
    axA.legend(frameon=False)
    axA.grid(True, alpha=0.3)

    # Panel B: PCA density overlay (states)
    axB = fig.add_subplot(gs[0, 1])
    maskL = (labs == "long_cp")
    maskW = (labs == "wide")
    if np.any(maskL):
        sns.kdeplot(x=Y[maskL, 0], y=Y[maskL, 1], ax=axB, fill=True, levels=20, cmap="Greens", alpha=0.7, bw_adjust=bw_adjust, thresh=0.02)
    if np.any(maskW):
        sns.kdeplot(x=Y[maskW, 0], y=Y[maskW, 1], ax=axB, fill=True, levels=20, cmap="Blues", alpha=0.5, bw_adjust=bw_adjust, thresh=0.02)
    axB.set_title(f"States in PCA space (var={evr[0]:.2f}+{evr[1]:.2f})")
    axB.set_xlabel("PC1")
    axB.set_ylabel("PC2")
    axB.grid(True, alpha=0.3)

    # Panel C: Behavior difference in PCA (policy entropy)
    axC = fig.add_subplot(gs[0, 2])
    # Grid the space
    x_min, x_max = np.quantile(Y[:, 0], [0.01, 0.99])
    y_min, y_max = np.quantile(Y[:, 1], [0.01, 0.99])
    nx, ny = 120, 120
    xg = np.linspace(x_min, x_max, nx)
    yg = np.linspace(y_min, y_max, ny)
    HX_L = np.zeros((ny, nx)); C_L = np.zeros((ny, nx))
    HX_W = np.zeros((ny, nx)); C_W = np.zeros((ny, nx))
    # Bin indices
    xi = np.clip(np.searchsorted(xg, Y[:, 0], side="right") - 1, 0, nx - 1)
    yi = np.clip(np.searchsorted(yg, Y[:, 1], side="right") - 1, 0, ny - 1)
    for k in range(n):
        if labs[k] == "long_cp":
            HX_L[yi[k], xi[k]] += ents[k]
            C_L[yi[k], xi[k]] += 1.0
        else:
            HX_W[yi[k], xi[k]] += ents[k]
            C_W[yi[k], xi[k]] += 1.0
    mean_L = np.divide(HX_L, np.maximum(C_L, 1e-9))
    mean_W = np.divide(HX_W, np.maximum(C_W, 1e-9))
    diff = mean_W - mean_L
    vmax = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
    im = axC.imshow(diff, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axC.set_title("Behavior shift: Δ policy entropy (wide − long)")
    axC.set_xlabel("PC1")
    axC.set_ylabel("PC2")
    fig.colorbar(im, ax=axC, shrink=0.8, label="Δ mean entropy")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

# -------------------------------
# Cross-eval: invalid vs valid edge scores
# -------------------------------

def _collect_ready_split_edge_scores(dataset, agent: GinAgent) -> tuple[list[float], list[float]]:
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(dataset=dataset, compute_metrics=False, dataset_episode_mode="single"))
    obs_np, _ = env.reset()
    valid_scores: list[float] = []
    invalid_scores: list[float] = []  # edges where task is not-ready or already scheduled
    while True:
        # Decode observation into tensor struct used by the GIN
        x = torch.from_numpy(obs_np.astype(np.float32))
        with torch.no_grad():
            decoded = agent.mapper.unmap(x.to(agent.device))
            # Edge embeddings and scores (pre readiness/scheduled masking)
            _, edge_embeddings, graph_embedding = agent.actor.network(decoded)
            # Build scorer input depending on whether the actor expects global graph embedding
            try:
                rep_graph = graph_embedding.expand(edge_embeddings.shape[0], agent.actor.network.embedding_dim)
                e = torch.cat([edge_embeddings, rep_graph], dim=1)
                edge_scores = agent.actor.edge_scorer(e).flatten()
            except Exception:
                # Fallback for actors that do not use global graph embedding
                edge_scores = agent.actor.edge_scorer(edge_embeddings).flatten()
            # Keep only compat edge scores (drop dep edges)
            E = int(decoded.compatibilities.shape[1]) if decoded.compatibilities.ndim == 2 else 0
            if E > 0 and edge_scores.shape[0] >= E:
                edge_scores = edge_scores[:E]
            elif E == 0:
                edge_scores = edge_scores[:0]
            # Edge endpoints (task, vm) only for compat edges
            comp_t = decoded.compatibilities[0].to(torch.long)
            # comp_v = decoded.compatibilities[1].to(torch.long)
            # Readiness mask per task (also excludes already-scheduled)
            ready_mask = decoded.task_state_ready.bool() & (decoded.task_state_scheduled == 0)
            is_ready_edge = ready_mask[comp_t]
            # Align lengths (mask must match scores length)
            if edge_scores.shape[0] != is_ready_edge.shape[0]:
                L = min(edge_scores.shape[0], is_ready_edge.shape[0])
                edge_scores = edge_scores[:L]
                is_ready_edge = is_ready_edge[:L]
            vs = edge_scores[is_ready_edge].detach().cpu().numpy().tolist()
            ivs = edge_scores[~is_ready_edge].detach().cpu().numpy().tolist()
            valid_scores += vs
            invalid_scores += ivs
            # Advance rollout with actual action
            a, _, _, _ = agent.get_action_and_value(x.unsqueeze(0))
            obs_np, _, term, trunc, _ = env.step(int(a.item()))
            if term or trunc:
                break
    return valid_scores, invalid_scores


def _plot_invalid_vs_valid_scores(results: dict, out_path: str):
    # results: {label: {"valid": [...], "invalid": [...]}}
    keys = list(results.keys())
    n = len(keys)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.8 * rows), dpi=200)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, k in enumerate(keys):
        r = idx // cols; c = idx % cols
        ax = axes[r, c]
        v = np.asarray(results[k].get("valid", []), dtype=float)
        iv = np.asarray(results[k].get("invalid", []), dtype=float)
        if v.size > 0:
            ax.hist(v, bins=60, color="#2E7D32", alpha=0.6, label="valid (ready)")
        if iv.size > 0:
            ax.hist(iv, bins=60, color="#C62828", alpha=0.6, label="invalid (not-ready/assigned)")
        ax.set_title(k)
        ax.set_xlabel("Pre-mask edge score (logit)")
        ax.set_ylabel("count")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def _plot_reward_mix(data: dict, out_path: str, ew: float, mw: float):
    styles = ["long_cp", "wide"]
    vals_unw = []
    vals_w = []
    for st in styles:
        er = np.array([v for (v, s) in zip(data.get("er", []), data.get("style", [])) if s == st], dtype=float)
        mr = np.array([v for (v, s) in zip(data.get("mr", []), data.get("style", [])) if s == st], dtype=float)
        er = er[np.isfinite(er)]
        mr = mr[np.isfinite(mr)]
        a_er = float(np.mean(np.abs(er))) if er.size else 0.0
        a_mr = float(np.mean(np.abs(mr))) if mr.size else 0.0
        aw_er = float(np.mean(np.abs(ew * er))) if er.size else 0.0
        aw_mr = float(np.mean(np.abs(mw * mr))) if mr.size else 0.0
        vals_unw.append((a_er, a_mr))
        vals_w.append((aw_er, aw_mr))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    x = np.arange(len(styles))
    w = 0.35
    axes[0].bar(x - w/2, [v[0] for v in vals_unw], width=w, label="|energy|", color="#29B6F6")
    axes[0].bar(x + w/2, [v[1] for v in vals_unw], width=w, label="|makespan|", color="#EF6C00")
    axes[0].set_xticks(x, styles)
    axes[0].set_title("Mean abs reward components (unweighted)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x - w/2, [v[0] for v in vals_w], width=w, label="|wE·energy|", color="#29B6F6")
    axes[1].bar(x + w/2, [v[1] for v in vals_w], width=w, label="|wM·makespan|", color="#EF6C00")
    axes[1].set_xticks(x, styles)
    axes[1].set_title("Mean abs reward components (weighted)")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path); print(f"Saved: {out_path}")


def _plot_reward_dominance_maps(
    data: dict,
    out_long: str,
    out_wide: str,
    y_max: float,
    transform: str = "log1p",
    bins: int = 60,
):
    def _tx(y: np.ndarray) -> np.ndarray:
        if transform == "log1p":
            return np.log1p(y)
        if transform == "sqrt":
            return np.sqrt(y)
        return y
    def arr(style: str):
        x = np.array([p for (p, s) in zip(data.get("pi", []), data.get("style", [])) if s == style], dtype=float)
        y = np.array([e for (e, s) in zip(data.get("ei", []), data.get("style", [])) if s == style], dtype=float)
        er = np.array([r for (r, s) in zip(data.get("er", []), data.get("style", [])) if s == style], dtype=float)
        mr = np.array([r for (r, s) in zip(data.get("mr", []), data.get("style", [])) if s == style], dtype=float)
        m = min(x.size, y.size, er.size, mr.size)
        return x[:m], _tx(y[:m]), er[:m], mr[:m]
    def make_map(style: str, out_path: str):
        x, y, er, mr = arr(style)
        if x.size == 0:
            return
        xg = np.linspace(0.0, 1.0, bins)
        yg = np.linspace(0.0, float(_tx(np.array([y_max]))[0]), bins)
        X, Y = np.meshgrid(xg, yg)
        se = np.zeros_like(X); sm = np.zeros_like(X); c = np.zeros_like(X)
        xi = np.clip(np.searchsorted(xg, x, side="right") - 1, 0, bins - 1)
        yi = np.clip(np.searchsorted(yg, y, side="right") - 1, 0, bins - 1)
        for i in range(x.size):
            se[yi[i], xi[i]] += abs(er[i])
            sm[yi[i], xi[i]] += abs(mr[i])
            c[yi[i], xi[i]] += 1.0
        me = np.divide(se, np.maximum(c, 1e-9))
        mm = np.divide(sm, np.maximum(c, 1e-9))
        diff = mm - me
        vmax = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6), dpi=200)
        im = ax.imshow(diff, origin="lower", extent=[0, 1.0, 0, float(_tx(np.array([y_max]))[0])], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Reward dominance (makespan − energy): {style}")
        ax.set_xlabel("PI"); ax.set_ylabel(f"EI [{transform}]")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Δ mean |reward|")
        plt.tight_layout(); plt.savefig(out_path); print(f"Saved: {out_path}")
    make_map("long_cp", out_long)
    make_map("wide", out_wide)


def _plot_panels(
    data: dict,
    traj_long: List[Tuple[float, float]],
    traj_wide: List[Tuple[float, float]],
    out_path: str,
    y_max: float | None = None,
    transform: str = "none",
    bw: float = 1.0,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200, sharex=True, sharey=True)

    def _tx(y: np.ndarray) -> np.ndarray:
        if transform == "log1p":
            return np.log1p(y)
        if transform == "sqrt":
            return np.sqrt(y)
        return y

    def _label() -> str:
        if transform == "log1p":
            return "Energy Intensity Index [log1p]"
        if transform == "sqrt":
            return "Energy Intensity Index [sqrt]"
        return "Energy Intensity Index (active power rate)"

    def get_arrays(style: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([p for (p, s) in zip(data["pi"], data["style"]) if s == style]) if data["pi"] else np.array([])
        y = np.array([e for (e, s) in zip(data["ei"], data["style"]) if s == style]) if data["ei"] else np.array([])
        return x, _tx(y)

    long_pi, long_ei = get_arrays("long_cp")
    wide_pi, wide_ei = get_arrays("wide")

    # LongCP panel
    ax = axes[0]
    if long_pi.size > 1 and long_ei.size > 1:
        sns.kdeplot(x=long_pi, y=long_ei, fill=True, levels=20, cmap="Greens", thresh=0.02, bw_adjust=bw, ax=ax)
    ax.plot([p for (p, _e) in traj_long], [_tx(np.array([e]))[0] for (_p, e) in traj_long], color="#E67E22", linewidth=1.2, alpha=0.9)
    ax.set_title("LongCP — policy visitation")
    ax.set_xlabel("Parallelism Index (ready / width)")
    ax.set_ylabel(_label())
    ax.set_xlim(0, 1.0)
    if y_max is not None:
        y_plot_max = float(_tx(np.array([y_max]))[0])
        ax.set_ylim(0, y_plot_max)
    else:
        ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # Wide panel
    ax = axes[1]
    if wide_pi.size > 1 and wide_ei.size > 1:
        sns.kdeplot(x=wide_pi, y=wide_ei, fill=True, levels=20, cmap="Greens", thresh=0.02, bw_adjust=bw, ax=ax)
    ax.plot([p for (p, _e) in traj_wide], [_tx(np.array([e]))[0] for (_p, e) in traj_wide], color="#E67E22", linewidth=1.2, alpha=0.9)
    ax.set_title("Wide — policy visitation")
    ax.set_xlabel("Parallelism Index (ready / width)")
    ax.set_ylabel(_label())
    ax.set_xlim(0, 1.0)
    if y_max is not None:
        y_plot_max = float(_tx(np.array([y_max]))[0])
        ax.set_ylim(0, y_plot_max)
    else:
        ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

def _plot_ms_ei_maps(
    data_em: dict,
    out_hist: str,
    out_diff: str,
    ms_max: float,
    ei_max: float,
    transform: str = "log1p",
    bw: float = 1.0,
    bins: int = 40,
):
    def _tx(y: np.ndarray) -> np.ndarray:
        if transform == "log1p":
            return np.log1p(y)
        if transform == "sqrt":
            return np.sqrt(y)
        return y

    def arr(style: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([m for (m, s) in zip(data_em["ms"], data_em["style"]) if s == style], dtype=float)
        y = np.array([e for (e, s) in zip(data_em["ei"], data_em["style"]) if s == style], dtype=float)
        return x, _tx(y)

    long_x, long_y = arr("long_cp")
    wide_x, wide_y = arr("wide")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=200)
    # Long KDE
    ax = axes[0]
    if long_x.size > 1:
        sns.kdeplot(x=long_x, y=long_y, fill=True, levels=20, cmap="Greens", thresh=0.02, bw_adjust=bw, ax=ax)
    ax.set_title("LongCP — KDE (MS vs EI)")
    ax.set_xlabel("Makespan (realized)")
    ax.set_ylabel(f"EI [{transform}]")
    ax.set_xlim(0, ms_max)
    ax.set_ylim(0, _tx(np.array([ei_max]))[0])
    # Long Hist
    ax = axes[1]
    if long_x.size > 1:
        ax.hist2d(long_x, long_y, bins=bins, range=[[0, ms_max], [0, _tx(np.array([ei_max]))[0]]], cmap="Greens")
    ax.set_title("LongCP — 2D histogram")
    ax.set_xlabel("Makespan (realized)")
    ax.set_ylabel(f"EI [{transform}]")
    # Wide KDE
    ax = axes[2]
    if wide_x.size > 1:
        sns.kdeplot(x=wide_x, y=wide_y, fill=True, levels=20, cmap="Greens", thresh=0.02, bw_adjust=bw, ax=ax)
    ax.set_title("Wide — KDE (MS vs EI)")
    ax.set_xlabel("Makespan (realized)")
    ax.set_ylabel(f"EI [{transform}]")
    ax.set_xlim(0, ms_max)
    ax.set_ylim(0, _tx(np.array([ei_max]))[0])
    # Wide Hist
    ax = axes[3]
    if wide_x.size > 1:
        ax.hist2d(wide_x, wide_y, bins=bins, range=[[0, ms_max], [0, _tx(np.array([ei_max]))[0]]], cmap="Greens")
    ax.set_title("Wide — 2D histogram")
    ax.set_xlabel("Makespan (realized)")
    ax.set_ylabel(f"EI [{transform}]")
    plt.tight_layout()
    plt.savefig(out_hist)
    print(f"Saved: {out_hist}")

    # Difference map (wide − long)
    xg = np.linspace(0.0, ms_max, 140)
    yg_lin = np.linspace(0.0, float(_tx(np.array([ei_max]))[0]), 120)
    X, Y = np.meshgrid(xg, yg_lin)
    grid_points = np.vstack([X.ravel(), Y.ravel()])

    def kde_eval(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.size < 2:
            return np.zeros_like(X)
        kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
        z = kde(grid_points).reshape(X.shape)
        return z / (z.sum() + 1e-12)

    z_long = kde_eval(long_x, long_y)
    z_wide = kde_eval(wide_x, wide_y)
    z_diff = z_wide - z_long

    fig2, axd = plt.subplots(1, 1, figsize=(6.0, 4.5), dpi=200)
    im = axd.imshow(
        z_diff,
        origin="lower",
        extent=[0, ms_max, 0, float(_tx(np.array([ei_max]))[0])],
        cmap="RdBu_r",
        vmin=-np.max(np.abs(z_diff)) if z_diff.size else -1,
        vmax=np.max(np.abs(z_diff)) if z_diff.size else 1,
        aspect="auto",
    )
    axd.set_title("Density difference (wide − long): MS vs EI")
    axd.set_xlabel("Makespan (realized)")
    axd.set_ylabel(f"EI [{transform}]")
    fig2.colorbar(im, ax=axd, shrink=0.8, label="Δ density")
    plt.tight_layout()
    plt.savefig(out_diff)
    print(f"Saved: {out_diff}")


def _plot_kde_hist_and_diff(
    data: dict,
    out_path_hist: str,
    out_path_diff: str,
    y_max: float,
    transform: str = "log1p",
    bw: float = 1.0,
    bins: int = 40,
):
    def _tx(y: np.ndarray) -> np.ndarray:
        if transform == "log1p":
            return np.log1p(y)
        if transform == "sqrt":
            return np.sqrt(y)
        return y

    def arr(style: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([p for (p, s) in zip(data["pi"], data["style"]) if s == style], dtype=float)
        y = np.array([e for (e, s) in zip(data["ei"], data["style"]) if s == style], dtype=float)
        return x, _tx(y)

    long_x, long_y = arr("long_cp")
    wide_x, wide_y = arr("wide")

    # 1) KDE and 2D hist side-by-side per style
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=200, sharex=False, sharey=False)
    # Long KDE
    ax = axes[0]
    if long_x.size > 1:
        sns.kdeplot(x=long_x, y=long_y, fill=True, levels=20, cmap="Greens", thresh=0.02, bw_adjust=bw, ax=ax)
    ax.set_title("LongCP — KDE")
    ax.set_xlabel("PI")
    ax.set_ylabel(f"EI [{transform}]")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, _tx(np.array([y_max]))[0])
    # Long Hist
    ax = axes[1]
    if long_x.size > 1:
        ax.hist2d(long_x, long_y, bins=bins, range=[[0, 1.0], [0, _tx(np.array([y_max]))[0]]], cmap="Greens")
    ax.set_title("LongCP — 2D histogram")
    ax.set_xlabel("PI")
    ax.set_ylabel(f"EI [{transform}]")
    # Wide KDE
    ax = axes[2]
    if wide_x.size > 1:
        sns.kdeplot(x=wide_x, y=wide_y, fill=True, levels=20, cmap="Greens", thresh=0.02, bw_adjust=bw, ax=ax)
    ax.set_title("Wide — KDE")
    ax.set_xlabel("PI")
    ax.set_ylabel(f"EI [{transform}]")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, _tx(np.array([y_max]))[0])
    # Wide Hist
    ax = axes[3]
    if wide_x.size > 1:
        ax.hist2d(wide_x, wide_y, bins=bins, range=[[0, 1.0], [0, _tx(np.array([y_max]))[0]]], cmap="Greens")
    ax.set_title("Wide — 2D histogram")
    ax.set_xlabel("PI")
    ax.set_ylabel(f"EI [{transform}]")
    plt.tight_layout()
    plt.savefig(out_path_hist)
    print(f"Saved: {out_path_hist}")

    # 2) Difference map: wide − long on a fixed grid
    xg = np.linspace(0.0, 1.0, 120)
    yg_lin = np.linspace(0.0, float(_tx(np.array([y_max]))[0]), 120)
    X, Y = np.meshgrid(xg, yg_lin)
    grid_points = np.vstack([X.ravel(), Y.ravel()])

    def kde_eval(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.size < 2:
            return np.zeros_like(X)
        kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
        z = kde(grid_points).reshape(X.shape)
        return z / (z.sum() + 1e-12)  # normalize to a density-like surface

    z_long = kde_eval(long_x, long_y)
    z_wide = kde_eval(wide_x, wide_y)
    z_diff = z_wide - z_long

    fig2, axd = plt.subplots(1, 1, figsize=(6.0, 4.5), dpi=200)
    im = axd.imshow(
        z_diff,
        origin="lower",
        extent=[0, 1.0, 0, float(_tx(np.array([y_max]))[0])],
        cmap="RdBu_r",
        vmin=-np.max(np.abs(z_diff)) if z_diff.size else -1,
        vmax=np.max(np.abs(z_diff)) if z_diff.size else 1,
        aspect="auto",
    )
    axd.set_title("Density difference (wide − long)")
    axd.set_xlabel("PI")
    axd.set_ylabel(f"EI [{transform}]")
    fig2.colorbar(im, ax=axd, shrink=0.8, label="Δ density")
    plt.tight_layout()
    plt.savefig(out_path_diff)
    print(f"Saved: {out_path_diff}")


def _plot_marginals(data: dict, out_path_marg: str, y_max: float, transform: str = "log1p"):
    def _tx(y: np.ndarray) -> np.ndarray:
        if transform == "log1p":
            return np.log1p(y)
        if transform == "sqrt":
            return np.sqrt(y)
        return y

    def arr(style: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([p for (p, s) in zip(data["pi"], data["style"]) if s == style], dtype=float)
        y = np.array([e for (e, s) in zip(data["ei"], data["style"]) if s == style], dtype=float)
        return x, _tx(y)

    long_x, long_y = arr("long_cp")
    wide_x, wide_y = arr("wide")

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12, 5), dpi=200)
    gs = gridspec.GridSpec(2, 4, width_ratios=[4, 1, 4, 1], height_ratios=[1, 4])

    def plot_block(col: int, x: np.ndarray, y: np.ndarray, title: str):
        ax_main = fig.add_subplot(gs[1, col])
        ax_top = fig.add_subplot(gs[0, col], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, col + 1], sharey=ax_main)
        if x.size > 1:
            sns.kdeplot(x=x, y=y, fill=True, levels=20, cmap="Greens", thresh=0.02, ax=ax_main)
        ax_main.set_title(title)
        ax_main.set_xlabel("PI")
        ax_main.set_ylabel(f"EI [{transform}]")
        ax_main.set_xlim(0, 1.0)
        ax_main.set_ylim(0, _tx(np.array([y_max]))[0])
        # Top marginal
        if x.size > 0:
            ax_top.hist(x, bins=40, range=(0, 1.0), color="#6AA84F", edgecolor="white")
        ax_top.set_ylabel("count")
        plt.setp(ax_top.get_xticklabels(), visible=False)
        # Right marginal
        if y.size > 0:
            ax_right.hist(y, bins=40, range=(0, _tx(np.array([y_max]))[0]), orientation="horizontal", color="#6AA84F", edgecolor="white")
        ax_right.set_xlabel("count")
        plt.setp(ax_right.get_yticklabels(), visible=False)

    plot_block(0, long_x, long_y, "LongCP — KDE + marginals")
    plot_block(2, wide_x, wide_y, "Wide — KDE + marginals")
    plt.tight_layout()
    plt.savefig(out_path_marg)
    print(f"Saved: {out_path_marg}")


def _plot_series(
    ms_long: List[float], apr_long: List[float], cum_long: List[float],
    ms_wide: List[float], apr_wide: List[float], cum_wide: List[float],
    out_path: str,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=200, sharex=True)
    # Active power rate
    axes[0].plot(apr_long, label="LongCP", color="#2E7D32")
    axes[0].plot(apr_wide, label="Wide", color="#1565C0")
    axes[0].set_ylabel("Active power rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Cumulative active energy (approx)
    axes[1].plot(cum_long, label="LongCP", color="#2E7D32")
    axes[1].plot(cum_wide, label="Wide", color="#1565C0")
    axes[1].set_ylabel("Cumulative active energy (approx)")
    axes[1].grid(True, alpha=0.3)
    # Makespan estimate
    axes[2].plot(ms_long, label="LongCP", color="#2E7D32")
    axes[2].plot(ms_wide, label="Wide", color="#1565C0")
    axes[2].set_ylabel("Makespan estimate")
    axes[2].set_xlabel("Decision step")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_path = os.environ.get("MODEL_PATH", None)
    # Optional: override host specs path for this run (no global file edits)
    _host_override = os.environ.get("HOST_SPECS_PATH", "").strip()
    if _host_override:
        try:
            _cfg.HOST_SPECS_PATH = Path(_host_override)
            # Also override in gen_vm (it imported the constant at module import time)
            _gen_vm.HOST_SPECS_PATH = _cfg.HOST_SPECS_PATH
            print(f"[override] Using host specs from: {_cfg.HOST_SPECS_PATH}")
        except Exception as _e:
            print(f"[override] Failed to set HOST_SPECS_PATH='{_host_override}': {_e}")
    # Base agent (legacy fallback)
    agent = _load_agent(model_path)

    seed_source = os.environ.get("SEED_SOURCE", "eval").strip().lower()  # 'eval' or 'train'
    num_seeds = int(os.environ.get("NUM_SEEDS", "10"))
    episodes_per_seed = int(os.environ.get("EPISODES_PER_SEED", "10"))
    num_agents = int(os.environ.get("NUM_AGENTS", "1"))
    do_shuffle = os.environ.get("SHUFFLE_SEEDS", "0") == "1"
    reinit_per_ep = os.environ.get("REINIT_AGENT_PER_EPISODE", "0") == "1"
    agent_seed_base = int(os.environ.get("AGENT_INIT_SEED_BASE", "0"))
    agent_seed_stride = int(os.environ.get("AGENT_INIT_SEED_STRIDE", "1000"))
    agent_jitter_std = float(os.environ.get("AGENT_JITTER_STD", "0.0"))

    long_seeds, long_ds = _load_seedset(LONG_CP_CFG, seed_source)
    wide_seeds, wide_ds = _load_seedset(WIDE_CFG, seed_source)

    if do_shuffle:
        import random as _rnd
        _rnd.shuffle(long_seeds)
        _rnd.shuffle(wide_seeds)

    long_seeds = long_seeds[:num_seeds]
    wide_seeds = wide_seeds[:num_seeds]

    # Fast path: do only cross-evaluation of invalid vs valid scores
    if os.environ.get("CROSS_ONLY", "0") == "1":
        mp_long = os.environ.get("MODEL_PATH_LONG", "").strip()
        mp_wide = os.environ.get("MODEL_PATH_WIDE", "").strip()
        try:
            cross_n = int(os.environ.get("CROSS_NUM_SEEDS", "5"))
        except Exception:
            cross_n = 5
        results: dict[str, dict[str, list[float]]] = {}
        if mp_long and os.path.exists(mp_long):
            agent_long = _load_agent(mp_long)
            v_all: list[float] = []; iv_all: list[float] = []
            for s in long_seeds[:cross_n]:
                ds = _mk_dataset(int(s), long_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_long)
                v_all += v; iv_all += iv
            results["agent_long→long"] = {"valid": v_all, "invalid": iv_all}
            v_all = []; iv_all = []
            for s in wide_seeds[:cross_n]:
                ds = _mk_dataset(int(s), wide_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_long)
                v_all += v; iv_all += iv
            results["agent_long→wide"] = {"valid": v_all, "invalid": iv_all}
        else:
            print("[cross-only] MODEL_PATH_LONG not provided or not found; skipping long agent.")
        if mp_wide and os.path.exists(mp_wide):
            agent_wide = _load_agent(mp_wide)
            v_all = []; iv_all = []
            for s in wide_seeds[:cross_n]:
                ds = _mk_dataset(int(s), wide_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_wide)
                v_all += v; iv_all += iv
            results["agent_wide→wide"] = {"valid": v_all, "invalid": iv_all}
            v_all = []; iv_all = []
            for s in long_seeds[:cross_n]:
                ds = _mk_dataset(int(s), long_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_wide)
                v_all += v; iv_all += iv
            results["agent_wide→long"] = {"valid": v_all, "invalid": iv_all}
        else:
            print("[cross-only] MODEL_PATH_WIDE not provided or not found; skipping wide agent.")
        if results:
            out_scores = os.path.join(OUT_DIR, "invalid_vs_valid_edge_scores.png")
            _plot_invalid_vs_valid_scores(results, out_scores)
        return

    data = {"style": [], "pi": [], "ei": []}
    data_em = {"style": [], "ms": [], "ei": []}
    data_rewards = {"style": [], "er": [], "mr": []}
    X_all: List[np.ndarray] = []
    labels_all: List[str] = []
    pi_all: List[float] = []
    ei_all: List[float] = []
    ent_all: List[float] = []
    agent_ids_all: List[int] = []
    struct_stats_long: List[tuple[int, int]] = []
    struct_stats_wide: List[tuple[int, int]] = []
    traj_long: List[Tuple[float, float]] = []
    traj_wide: List[Tuple[float, float]] = []
    ms_long: List[float] | None = None
    apr_long: List[float] | None = None
    cum_long: List[float] | None = None
    ms_wide: List[float] | None = None
    apr_wide: List[float] | None = None
    cum_wide: List[float] | None = None

    for agent_idx in range(max(1, num_agents)):
        # Build a base agent per agent_idx if not reinitializing per episode
        if reinit_per_ep:
            agent_base = agent  # will reinit per episode
        else:
            agent_base = _load_agent(model_path, init_seed=agent_seed_base + agent_idx, jitter_std=agent_jitter_std)

        for s in long_seeds:
            ds = _mk_dataset(int(s), long_ds)
            try:
                if agent_idx == 0:
                    struct_stats_long.append(_dataset_structure_stats(ds))
            except Exception:
                pass
            for epi in range(max(1, episodes_per_seed)):
                # Agent per episode
                agent_ep = agent_base
                if reinit_per_ep:
                    init_seed = agent_seed_base + agent_idx * agent_seed_stride + epi
                    agent_ep = _load_agent(model_path, init_seed=init_seed, jitter_std=agent_jitter_std)
                pis, eis, traj, ms_s, apr_s, cum_s, feats, ents, er_s, mr_s, final_ms = _rollout_collect_points_with_agent(ds, agent_ep)
                data["style"] += ["long_cp"] * len(pis)
                data["pi"] += pis
                data["ei"] += eis
                data_rewards["style"] += ["long_cp"] * len(er_s)
                data_rewards["er"] += er_s
                data_rewards["mr"] += mr_s
                # MS vs EI aggregate (align series lengths)
                n = min(len(ms_s), len(eis))
                if n > 0:
                    data_em["style"] += ["long_cp"] * n
                    data_em["ms"] += [float(final_ms)] * n
                    data_em["ei"] += eis[:n]
                # PCA aggregates
                if feats:
                    X_all += feats
                    labels_all += ["long_cp"] * len(feats)
                    pi_all += pis[:len(feats)]
                    ei_all += eis[:len(feats)]
                    ent_all += ents[:len(feats)] if ents else []
                    agent_ids_all += [agent_idx] * len(feats)
                if not traj_long:
                    traj_long = traj
                if ms_long is None:
                    ms_long, apr_long, cum_long = ms_s, apr_s, cum_s

        for s in wide_seeds:
            ds = _mk_dataset(int(s), wide_ds)
            try:
                if agent_idx == 0:
                    struct_stats_wide.append(_dataset_structure_stats(ds))
            except Exception:
                pass
            for epi in range(max(1, episodes_per_seed)):
                agent_ep = agent_base
                if reinit_per_ep:
                    init_seed = agent_seed_base + agent_idx * agent_seed_stride + epi
                    agent_ep = _load_agent(model_path, init_seed=init_seed, jitter_std=agent_jitter_std)
                pis, eis, traj, ms_s, apr_s, cum_s, feats, ents, er_s, mr_s, final_ms = _rollout_collect_points_with_agent(ds, agent_ep)
                data["style"] += ["wide"] * len(pis)
                data["pi"] += pis
                data["ei"] += eis
                data_rewards["style"] += ["wide"] * len(er_s)
                data_rewards["er"] += er_s
                data_rewards["mr"] += mr_s
                n = min(len(ms_s), len(eis))
                if n > 0:
                    data_em["style"] += ["wide"] * n
                    data_em["ms"] += [float(final_ms)] * n
                    data_em["ei"] += eis[:n]
                if feats:
                    X_all += feats
                    labels_all += ["wide"] * len(feats)
                    pi_all += pis[:len(feats)]
                    ei_all += eis[:len(feats)]
                    ent_all += ents[:len(feats)] if ents else []
                    agent_ids_all += [agent_idx] * len(feats)
                if not traj_wide:
                    traj_wide = traj
                if ms_wide is None:
                    ms_wide, apr_wide, cum_wide = ms_s, apr_s, cum_s

    # Base (full-scale) figure
    bw = float(os.environ.get("KDE_BW", "1.0"))
    _plot_panels(data, traj_long, traj_wide, OUT_PNG, y_max=None, transform="none", bw=bw)

    # Zoom: y-axis clipped to EI quantile (default 98th percentile)
    try:
        q = float(os.environ.get("EI_Q", "0.98"))
    except Exception:
        q = 0.98
    ei_all = np.asarray(data["ei"], dtype=float)
    ei_q = float(np.quantile(ei_all, q)) if ei_all.size > 0 else 1.0
    out_zoom = os.path.join(OUT_DIR, "state_visitation_pi_ei_agent_zoom.png")
    _plot_panels(data, traj_long, traj_wide, out_zoom, y_max=ei_q, transform="none", bw=bw)

    # Log-scaled EI (log1p) to expand small values
    if os.environ.get("LOG_Y", "1") == "1":
        out_log = os.path.join(OUT_DIR, "state_visitation_pi_ei_agent_log.png")
        _plot_panels(data, traj_long, traj_wide, out_log, y_max=ei_q, transform="log1p", bw=bw)

    # KDE + 2D histograms and density difference map
    out_hist = os.path.join(OUT_DIR, "state_visitation_pi_ei_agent_kde_hist.png")
    out_diff = os.path.join(OUT_DIR, "state_visitation_pi_ei_agent_diff.png")
    _plot_kde_hist_and_diff(data, out_hist, out_diff, y_max=ei_q, transform="log1p", bw=bw, bins=int(os.environ.get("HIST_BINS", "40")))

    # Marginal histograms
    out_marg = os.path.join(OUT_DIR, "state_visitation_pi_ei_agent_marginals.png")
    _plot_marginals(data, out_marg, y_max=ei_q, transform="log1p")

    # Per-step series (first episode per style)
    if ms_long is not None and ms_wide is not None and apr_long is not None and apr_wide is not None and cum_long is not None and cum_wide is not None:
        out_series = os.path.join(OUT_DIR, "state_series_ms_active.png")
        _plot_series(ms_long, apr_long, cum_long, ms_wide, apr_wide, cum_wide, out_series)

    # Makespan vs Energy-Intensity maps
    if len(data_em["ms"]) > 1:
        try:
            ms_q = float(os.environ.get("MS_Q", "0.98"))
        except Exception:
            ms_q = 0.98
        ms_all = np.asarray(data_em["ms"], dtype=float)
        ms_cap = float(np.quantile(ms_all, ms_q)) if ms_all.size > 0 else float(ms_all.max() if ms_all.size > 0 else 1.0)
        out_em_hist = os.path.join(OUT_DIR, "state_visitation_ms_ei_agent_kde_hist.png")
        out_em_diff = os.path.join(OUT_DIR, "state_visitation_ms_ei_agent_diff.png")
        _plot_ms_ei_maps(data_em, out_em_hist, out_em_diff, ms_max=ms_cap, ei_max=ei_q, transform="log1p", bw=bw, bins=int(os.environ.get("HIST_BINS", "40")))

    # Reward mix summary and dominance maps
    try:
        ew = float(os.environ.get("GIN_ENERGY_WEIGHT", "1.0"))
    except Exception:
        ew = 1.0
    try:
        mw = float(os.environ.get("GIN_MAKESPAN_WEIGHT", "1.0"))
    except Exception:
        mw = 1.0
    if data_rewards["er"]:
        out_mix = os.path.join(OUT_DIR, "reward_mix_by_style.png")
        _plot_reward_mix({"style": data_rewards["style"], "er": data_rewards["er"], "mr": data_rewards["mr"]}, out_mix, ew, mw)
        out_dom_long = os.path.join(OUT_DIR, "reward_dominance_longcp.png")
        out_dom_wide = os.path.join(OUT_DIR, "reward_dominance_wide.png")
        _plot_reward_dominance_maps({"style": data["style"], "pi": data["pi"], "ei": data["ei"], "er": data_rewards["er"], "mr": data_rewards["mr"]}, out_dom_long, out_dom_wide, y_max=ei_q, transform="log1p", bins=int(os.environ.get("HIST_BINS", "60")))
    # PCA of visited states
    if X_all:
        try:
            pca_max = int(os.environ.get("PCA_MAX_POINTS", "6000"))
        except Exception:
            pca_max = 6000
        out_pca_style = os.path.join(OUT_DIR, "state_visitation_pca_by_style.png")
        out_pca_pi = os.path.join(OUT_DIR, "state_visitation_pca_by_pi.png")
        out_pca_ei = os.path.join(OUT_DIR, "state_visitation_pca_by_ei.png")
        _plot_pca(X_all, labels_all, pi_all, ei_all, out_pca_style, out_pca_pi, out_pca_ei, max_points=pca_max, ei_transform="log1p")

        # PCA colored by agent id
        try:
            import matplotlib.cm as cm
        except Exception:
            cm = None
        def _plot_pca_by_agent(X_list: List[np.ndarray], agent_ids: List[int], out_path: str, max_points: int = 6000):
            if not X_list:
                return
            X = np.asarray(X_list, dtype=np.float32)
            aid = np.asarray(agent_ids, dtype=int)
            n = X.shape[0]
            if aid.shape[0] != n:
                m = min(n, aid.shape[0])
                X = X[:m]; aid = aid[:m]; n = m
            if n == 0:
                return
            if n > max_points:
                idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
                X = X[idx]; aid = aid[idx]; n = X.shape[0]
            var = X.var(axis=0)
            keep = var > 1e-12
            X2 = X[:, keep] if np.any(keep) else X
            Xz = StandardScaler().fit_transform(X2)
            pca = PCA(n_components=2, random_state=0)
            Y = pca.fit_transform(Xz)
            evr = pca.explained_variance_ratio_
            uniq = np.unique(aid)
            palette = sns.color_palette("tab10", n_colors=max(10, len(uniq)))
            color_map = {int(u): palette[i % len(palette)] for i, u in enumerate(uniq)}
            fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=200)
            for u in uniq:
                m = aid == u
                ax.scatter(Y[m, 0], Y[m, 1], s=8, alpha=0.6, c=[color_map[int(u)]], label=f"agent {int(u)}")
            ax.set_title(f"PCA by agent (var={evr[0]:.2f}+{evr[1]:.2f})")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.legend(frameon=False, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(out_path); print(f"Saved: {out_path}")

        out_pca_agent = os.path.join(OUT_DIR, "state_visitation_pca_by_agent.png")
        _plot_pca_by_agent(X_all, agent_ids_all if agent_ids_all else [0]*len(X_all), out_pca_agent, max_points=pca_max)

        # Triptych: Structure → State → Behavior
        try:
            pca_bw = float(os.environ.get("PCA_BW", str(bw)))
        except Exception:
            pca_bw = bw
        out_triptych = os.path.join(OUT_DIR, "structure_state_behavior_triptych.png")
        _plot_structure_state_behavior_triptych(
            X_all,
            labels_all,
            ent_all if ent_all else [0.0] * len(X_all),
            struct_stats_long,
            struct_stats_wide,
            out_triptych,
            max_points=pca_max,
            bw_adjust=pca_bw,
        )

    # Cross-evaluation: invalid vs valid scores (pre-mask) per agent and style
    if os.environ.get("CROSS_EVAL", "0") == "1":
        mp_long = os.environ.get("MODEL_PATH_LONG", "").strip()
        mp_wide = os.environ.get("MODEL_PATH_WIDE", "").strip()
        try:
            cross_n = int(os.environ.get("CROSS_NUM_SEEDS", "5"))
        except Exception:
            cross_n = 5
        results: dict[str, dict[str, list[float]]] = {}
        if mp_long and os.path.exists(mp_long):
            agent_long = _load_agent(mp_long)
            # long agent on long seeds
            v_all: list[float] = []; iv_all: list[float] = []
            for s in long_seeds[:cross_n]:
                ds = _mk_dataset(int(s), long_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_long)
                v_all += v; iv_all += iv
            results["agent_long→long"] = {"valid": v_all, "invalid": iv_all}
            # long agent on wide seeds
            v_all = []; iv_all = []
            for s in wide_seeds[:cross_n]:
                ds = _mk_dataset(int(s), wide_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_long)
                v_all += v; iv_all += iv
            results["agent_long→wide"] = {"valid": v_all, "invalid": iv_all}
        else:
            print("[cross] MODEL_PATH_LONG not provided or not found; skipping.")
        if mp_wide and os.path.exists(mp_wide):
            agent_wide = _load_agent(mp_wide)
            # wide agent on wide seeds
            v_all = []; iv_all = []
            for s in wide_seeds[:cross_n]:
                ds = _mk_dataset(int(s), wide_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_wide)
                v_all += v; iv_all += iv
            results["agent_wide→wide"] = {"valid": v_all, "invalid": iv_all}
            # wide agent on long seeds
            v_all = []; iv_all = []
            for s in long_seeds[:cross_n]:
                ds = _mk_dataset(int(s), long_ds)
                v, iv = _collect_ready_split_edge_scores(ds, agent_wide)
                v_all += v; iv_all += iv
            results["agent_wide→long"] = {"valid": v_all, "invalid": iv_all}
        else:
            print("[cross] MODEL_PATH_WIDE not provided or not found; skipping.")
        if results:
            out_scores = os.path.join(OUT_DIR, "invalid_vs_valid_edge_scores.png")
            _plot_invalid_vs_valid_scores(results, out_scores)


if __name__ == "__main__":
    main()
