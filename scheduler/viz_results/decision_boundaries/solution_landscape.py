if True:
    def shifted_cmap(name: str, start: float = 0.1, end: float = 1.0, bad: str | None = None) -> "ListedColormap":
        """Return a ListedColormap sampling a base cmap between [start, end].
        This ensures value 0 maps to a real color (not white) when using Reds.
        """
        base = plt.colormaps.get_cmap(name)
        colors = base(np.linspace(start, end, 256))
        cm = ListedColormap(colors)
        if bad is not None:
            cm = cm.with_extremes(bad=bad)
        return cm

    import math
    import os
    from dataclasses import dataclass
    from typing import List, Tuple, Dict, Any

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import seaborn as sns
    from tqdm import tqdm
    import argparse
    import itertools
    from contextlib import nullcontext

    # Make project imports work when running this file directly
    import sys
    from pathlib import Path
    THIS_DIR = Path(__file__).resolve().parent
    # Match train.py style: add the 'scheduler' directory itself to sys.path
    SCHEDULER_DIR = THIS_DIR.parents[2]
    if str(SCHEDULER_DIR) not in sys.path:
        sys.path.insert(0, str(SCHEDULER_DIR))

    from scheduler.config.settings import MIN_TESTING_DS_SEED
    from scheduler.dataset_generator.gen_dataset import DatasetArgs
    from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
    from scheduler.rl_model.core.env.action import EnvAction


    @dataclass
    class EnumArgs:
        # Dataset size (keep small for exhaustive enumeration)
        host_count: int = 2
        vm_count: int = 3
        workflow_count: int = 2
        gnp_min_n: int = 6
        gnp_max_n: int = 6
        dag_method: str = "gnp"
        min_task_length: int = 500
        max_task_length: int = 5_000
        min_cpu_speed: int = 500
        max_cpu_speed: int = 3_000
        max_memory_gb: int = 4

        # Enumeration control
        seed: int = MIN_TESTING_DS_SEED
        limit_solutions: int | None = None  # hard cap if the tree explodes
        score: str = "makespan"           # or "energy"
        show_progress: bool = True
        estimate_samples: int = 64         # Monte Carlo samples to estimate total solutions for ETA

        # Plotting
        cmap: str = "Reds"  # red sequential map for clearer boundaries
        dpi: int = 250
        out_dir: str = "logs/landscape"
        title: str | None = None
        paper: bool = False  # if True, suppress titles/labels for publication-ready figures

        # Optional fixed color scale limits for normalized maps/panels
        fixed_vmin_makespan: float | None = None
        fixed_vmax_makespan: float | None = None
        fixed_vmin_energy: float | None = None
        fixed_vmax_energy: float | None = None


    def build_env(a: EnumArgs) -> CloudSchedulingGymEnvironment:
        ds_args = DatasetArgs(
            host_count=a.host_count,
            vm_count=a.vm_count,
            workflow_count=a.workflow_count,
            gnp_min_n=a.gnp_min_n,
            gnp_max_n=a.gnp_max_n,
            max_memory_gb=a.max_memory_gb,
            min_cpu_speed=a.min_cpu_speed,
            max_cpu_speed=a.max_cpu_speed,
            min_task_length=a.min_task_length,
            max_task_length=a.max_task_length,
            task_arrival="static",
            dag_method=a.dag_method,
        )
        return CloudSchedulingGymEnvironment(dataset_args=ds_args)


    def ready_tasks(env: CloudSchedulingGymEnvironment) -> List[int]:
        assert env.state is not None
        ts = env.state.task_states
        # exclude dummy 0 and last task
        last_id = len(ts) - 1
        return [i for i, s in enumerate(ts)
                if s.is_ready and s.assigned_vm_id is None and i not in (0, last_id)]


    def compatible_vms(env: CloudSchedulingGymEnvironment, task_id: int) -> List[int]:
        assert env.state is not None
        vms = env.state.static_state.vms
        return [vm_id for vm_id in range(len(vms))
                if (task_id, vm_id) in env.state.static_state.compatibilities]


    def run_sequence(env: CloudSchedulingGymEnvironment, seq: List[Tuple[int, int]]) -> tuple[bool, Dict[str, Any]]:
        """Play a whole sequence on a fresh env. Return (feasible, metrics)."""
        obs, _ = env.reset(seed=None)
        info_last: Dict[str, Any] | None = None
        for (t, v) in seq:
            # Feasibility gate: must be ready and compatible at this step
            if t not in ready_tasks(env):
                return False, {"reason": "task_not_ready"}
            if v not in compatible_vms(env, t):
                return False, {"reason": "incompatible"}
            obs, r, term, trunc, info = env.step(EnvAction(task_id=t, vm_id=v))
            if term or trunc:
                info_last = info
                break
        # If not terminated yet, keep scheduling greedily until end (choose first ready, first compatible)
        while True:
            rt = ready_tasks(env)
            if not rt:
                break
            t = rt[0]
            cvs = compatible_vms(env, t)
            v = cvs[0]
            obs, r, term, trunc, info = env.step(EnvAction(task_id=t, vm_id=v))
            if term or trunc:
                info_last = info
                break

        # Collect final stats
        assert env.state is not None
        per_vm_ct = [vm.completion_time for vm in env.state.vm_states]
        makespan = float(max(per_vm_ct) if per_vm_ct else 0.0)
        energy = float(info_last.get("total_energy", 0.0)) if isinstance(info_last, dict) else 0.0
        return True, {"makespan": makespan, "energy": energy}


    def enumerate_all(env: CloudSchedulingGymEnvironment, a: EnumArgs) -> List[Tuple[List[Tuple[int, int]], Dict[str, Any]]]:
        """
        Exhaustively DFS over all (task, vm) choices respecting readiness and compatibility.
        Returns list of (sequence, metrics).
        Warning: combinatorial explosion. Keep dataset tiny!
        """
        results: List[Tuple[List[Tuple[int, int]], Dict[str, Any]]] = []

        # Helper: number of available choices in a state
        def num_choices(e: CloudSchedulingGymEnvironment) -> int:
            rt = ready_tasks(e)
            total = 0
            for t in rt:
                total += len(compatible_vms(e, t))
            return total

        # Estimate total leaves via Monte Carlo playouts to provide ETA
        est_total: int | None = None
        if a.show_progress and a.limit_solutions is None and a.estimate_samples > 0:
            choices_by_depth: Dict[int, list[int]] = {}
            for _ in range(a.estimate_samples):
                e = build_env(a)
                e.reset(seed=a.seed)
                depth = 0
                while True:
                    c = num_choices(e)
                    choices_by_depth.setdefault(depth, []).append(c)
                    if c == 0:
                        break
                    # Pick a random (task, vm)
                    rts = ready_tasks(e)
                    tv_pairs: list[Tuple[int, int]] = []
                    for t in rts:
                        for v in compatible_vms(e, t):
                            tv_pairs.append((t, v))
                    t, v = tv_pairs[np.random.randint(len(tv_pairs))]
                    _, _, term, trunc, _ = e.step(EnvAction(task_id=t, vm_id=v))
                    depth += 1
                    if term or trunc:
                        break
            avg_choices = []
            for d in sorted(choices_by_depth.keys()):
                arr = np.array(choices_by_depth[d], dtype=float)
                if len(arr) == 0:
                    continue
                avg_choices.append(max(1.0, float(arr.mean())))
            if avg_choices:
                est = 1.0
                for v in avg_choices:
                    est *= v
                # cap to avoid silly huge numbers
                est_total = int(min(est, 10_000_000))

        # Progress bar
        pbar = None
        total_for_pbar = a.limit_solutions if a.limit_solutions is not None else est_total
        if a.show_progress:
            pbar = tqdm(total=total_for_pbar, desc="Enumerating", unit="sol")

        # Work by replaying from root for each branch using the sequence prefix.
        # This avoids deep-copying the environment state.

        def dfs(prefix: List[Tuple[int, int]]):
            if a.limit_solutions is not None and len(results) >= a.limit_solutions:
                return
            # Recreate env and simulate prefix to know current frontier
            env_local = build_env(a)
            env_local.reset(seed=a.seed)
            feasible = True
            for (t, v) in prefix:
                if t not in ready_tasks(env_local) or v not in compatible_vms(env_local, t):
                    feasible = False
                    break
                _, _, term, trunc, _ = env_local.step(EnvAction(task_id=t, vm_id=v))
                if term or trunc:
                    break
            if not feasible:
                return

            # If done, score the sequence
            if env_local.state is not None and env_local.state.task_states[-1].assigned_vm_id is not None:
                _, metrics = run_sequence(build_env(a), prefix)
                results.append((prefix.copy(), metrics))
                if pbar is not None:
                    pbar.update(1)
                return

            # Expand choices from this state
            frontier = ready_tasks(env_local)
            if not frontier:
                # Stuck (should not happen) -> score greedy fill-in
                _, metrics = run_sequence(build_env(a), prefix)
                results.append((prefix.copy(), metrics))
                if pbar is not None:
                    pbar.update(1)
                return
            for t in frontier:
                for v in compatible_vms(env_local, t):
                    prefix.append((t, v))
                    dfs(prefix)
                    prefix.pop()

        dfs([])
        if pbar is not None:
            pbar.close()
        return results


    def layout_to_grid(values: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Pack a 1D array into a near-square 2D grid with padding using NaN."""
        n = len(values)
        if n == 0:
            return np.zeros((1, 1)), 1, 1
        side = int(math.ceil(math.sqrt(n)))
        grid = np.full((side, side), np.nan, dtype=float)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < n:
                    grid[i, j] = float(values[idx])
                    idx += 1
        return grid, side, side


    def plot_landscape(results: List[Tuple[List[Tuple[int, int]], Dict[str, Any]]], a: EnumArgs):
        if not results:
            print("No results to plot.")
            return
        # Extract chosen score
        vals = np.array([r[1][a.score] for r in results], dtype=float)
        # Normalize to [0,1] with robust scaling
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if vmax > vmin:
            norm_vals = (vals - vmin) / (vmax - vmin)
        else:
            norm_vals = np.zeros_like(vals)

        # Optional: shuffle to create a more textured pattern
        rng = np.random.default_rng(12345)
        order = rng.permutation(len(norm_vals))
        norm_vals = norm_vals[order]

        grid, H, W = layout_to_grid(norm_vals)

        os.makedirs(a.out_dir, exist_ok=True)
        title = a.title or f"Fitness landscape ({a.score}); N={len(results)}"

        sns.set_theme(context="paper", style="white", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=a.dpi)
        cmap = shifted_cmap(a.cmap, start=0.1, end=1.0, bad="#FDF6F6")
        im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Normalized {a.score}")
        out_png = os.path.join(a.out_dir, f"landscape_{a.score}.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved landscape to {out_png}; {len(results)} solutions")


    def pca_2d(X: np.ndarray) -> np.ndarray:
        """Simple PCA to 2D via SVD (no sklearn dependency)."""
        Xc = X - X.mean(axis=0, keepdims=True)
        # economy SVD
        U, S, VT = np.linalg.svd(Xc, full_matrices=False)
        W = VT[:2].T  # d x 2
        return Xc @ W


    def enumerate_all_sequences_vectors(env: CloudSchedulingGymEnvironment,
                                        a: EnumArgs,
                                        all_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Enumerate all sequences (task order permutation × VM choice per position), independent of feasibility.
        Vectorize each as [task_id_0, vm_id_0, task_id_1, vm_id_1, ...].
        Label feasibility by rolling the env and checking if the env accepts the sequence in order.
        Returns (X, y), where X is N×(2K) and y is boolean mask feasible.
        """
        # Build tiny env to read static info
        env.reset(seed=a.seed)
        assert env.state is not None
        K = len(env.state.task_states) - 2  # exclude dummy start/end
        if K <= 0:
            return np.zeros((0, 2)), np.zeros((0,), dtype=bool)
        real_tasks = [i for i in range(1, 1 + K)]
        V = len(env.state.static_state.vms)

        seq_count_est = math.factorial(K) * (V ** K)
        cap = all_limit if all_limit is not None else min(seq_count_est, 200000)

        X_list: List[List[int]] = []
        y_list: List[bool] = []

        total = 0
        with tqdm(total=cap, desc="All sequences", unit="seq") if cap is not None else nullcontext():
            # Iterate permutations of tasks
            for perm in itertools.permutations(real_tasks, K):
                # Iterate all VM choices per position
                for vm_choice in itertools.product(range(V), repeat=K):
                    seq = list(zip(perm, vm_choice))
                    # vector
                    vec: List[int] = []
                    for t, v in seq:
                        vec.extend([int(t), int(v)])
                    X_list.append(vec)
                    # label feasibility by trying to execute
                    e = build_env(a)
                    e.reset(seed=a.seed)
                    feasible = True
                    for (t, v) in seq:
                        if t not in ready_tasks(e) or v not in compatible_vms(e, t):
                            feasible = False
                            break
                        _o, _r, term, trunc, _ = e.step(EnvAction(task_id=t, vm_id=v))
                        if term or trunc:
                            break
                    y_list.append(feasible)
                    total += 1
                    if cap is not None and total >= cap:
                        break
                if cap is not None and total >= cap:
                    break

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=bool)
        return X, y


    def plot_all_sequences_2d(env: CloudSchedulingGymEnvironment, a: EnumArgs,
                            all_limit: int | None = None, proj: str = "pca"):
        X, y = enumerate_all_sequences_vectors(env, a, all_limit)
        if X.size == 0:
            print("No sequences to plot.")
            return
        if proj == "pca":
            Z = pca_2d(X)
        else:
            Z = pca_2d(X)
        # Plot
        os.makedirs(a.out_dir, exist_ok=True)
        sns.set_theme(context="paper", style="white", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=a.dpi)
        ax.scatter(Z[~y, 0], Z[~y, 1], s=8, c="#BBBBBB", alpha=0.5, label="infeasible")
        ax.scatter(Z[y, 0], Z[y, 1], s=10, c="#E84A5F", alpha=0.8, label="feasible")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=True)
        ax.set_title("All sequences (PCA), feasible highlighted")
        out_png = os.path.join(a.out_dir, "all_sequences_pca.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 2D projection of all sequences to {out_png}; N={len(X)} (feasible={int(y.sum())})")


    # -------------------- Space-filling (Hilbert) landscape over ALL sequences --------------------

    def lehmer_index(perm: Tuple[int, ...], K: int) -> int:
        """Convert permutation of 1..K to Lehmer code index in [0, K!-1]."""
        L = list(range(1, K + 1))
        idx = 0
        fact = 1
        # precompute factorials
        facts = [1] * (K + 1)
        for i in range(2, K + 1):
            facts[i] = facts[i - 1] * i
        for i, x in enumerate(perm):
            pos = L.index(x)
            idx += pos * facts[K - 1 - i]
            L.pop(pos)
        return idx


    def vm_index(vm_choice: Tuple[int, ...], V: int) -> int:
        """Interpret VM digits (base-V) as integer."""
        val = 0
        for d in vm_choice:
            val = val * V + int(d)
        return val


    def seq_to_linear_index(perm: Tuple[int, ...], vm_choice: Tuple[int, ...], K: int, V: int) -> int:
        """Combine Lehmer(perm) and base-V index into a single lexicographic integer."""
        facts = 1
        # compute V^K
        vpow = 1
        for _ in range(K):
            vpow *= V
        return lehmer_index(perm, K) * vpow + vm_index(vm_choice, V)


    def hilbert_side_for_count(N: int) -> int:
        side = int(1 << math.ceil(math.log2(max(1, int(math.ceil(math.sqrt(max(1, N))))))))
        return side


    def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            # Swap x and y
            x, y = y, x
        return x, y


    def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
        """Map Hilbert distance d to (x,y) for grid size n (power of two)."""
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = _rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y


    def plot_space_filling_landscape_all(env: CloudSchedulingGymEnvironment, a: EnumArgs,
                                        sf_limit: int | None = None, score: str = "makespan"):
        """Create a Hilbert-ordered 2D landscape over ALL sequences. Feasible cells colored by score; infeasible as background."""
        env.reset(seed=a.seed)
        assert env.state is not None
        K = len(env.state.task_states) - 2  # exclude dummy start/end
        if K <= 0:
            print("No real tasks to enumerate.")
            return
        real_tasks = tuple(range(1, 1 + K))
        V = len(env.state.static_state.vms)

        # Total sequences
        total_perm = 1
        for i in range(2, K + 1):
            total_perm *= i
        vpow = 1
        for _ in range(K):
            vpow *= V
        N_total = total_perm * vpow

        # Decide cap
        cap = sf_limit if sf_limit is not None else N_total

        # Prepare grid
        side = hilbert_side_for_count(N_total if cap is None else cap)
        grid = np.full((side, side), np.nan, dtype=float)
        mask = np.zeros((side, side), dtype=bool)

        counted = 0
        pbar = tqdm(total=cap, desc="SpaceFill", unit="seq") if cap is not None else None
        e = build_env(a)  # reuse env; reset per sequence
        for perm in itertools.permutations(real_tasks, K):
            for vm_choice in itertools.product(range(V), repeat=K):
                idx = seq_to_linear_index(perm, vm_choice, K, V)
                if cap is not None and counted >= cap:
                    break
                seq = list(zip(perm, vm_choice))
                # Simulate this sequence once
                e.reset(seed=a.seed)
                feasible = True
                info_last: Dict[str, Any] | None = None
                for (t, v) in seq:
                    if t not in ready_tasks(e) or v not in compatible_vms(e, t):
                        feasible = False
                        break
                    _o, _r, term, trunc, info = e.step(EnvAction(task_id=t, vm_id=v))
                    if term or trunc:
                        info_last = info
                        break
                # If feasible so far, greedily finish to termination to read metrics
                if feasible:
                    while True:
                        if getattr(e, "state", None) is None:
                            break
                        rt = ready_tasks(e)
                        if not rt:
                            break
                        t_next = rt[0]
                        cvs = compatible_vms(e, t_next)
                        if not cvs:
                            break
                        v_next = cvs[0]
                        _o, _r, term, trunc, info = e.step(EnvAction(task_id=t_next, vm_id=v_next))
                        if term or trunc:
                            info_last = info
                            break
                    # Collect desired score
                    assert e.state is not None
                    if score == "makespan":
                        per_vm_ct = [vm.completion_time for vm in e.state.vm_states]
                        val = float(max(per_vm_ct) if per_vm_ct else 0.0)
                    elif score == "energy":
                        val = float(info_last.get("total_energy", 0.0)) if isinstance(info_last, dict) else 0.0
                    else:
                        val = float("nan")
                    x, y = hilbert_d2xy(side, idx % (side * side))
                    grid[y, x] = val
                    mask[y, x] = True
                counted += 1
                if pbar is not None:
                    pbar.update(1)
            if cap is not None and counted >= cap:
                break
        if pbar is not None:
            pbar.close()

        vals = grid[mask]
        if vals.size == 0:
            print("No feasible sequences with scores to plot.")
            return
        # Normalize to [0,1] with robust percentiles for vivid gradient
        vmin = float(np.nanpercentile(vals, 2.0))
        vmax = float(np.nanpercentile(vals, 98.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
        norm_grid = (grid - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(grid)

        # Plot with proximity shading: infeasible cells shaded by distance to nearest feasible cell
        os.makedirs(a.out_dir, exist_ok=True)
        sns.set_theme(context="paper", style="white", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=a.dpi)
        # Compose an explicit RGBA image: infeasible -> red by proximity, feasible -> Blues by score
        H, W = norm_grid.shape
        rgba = np.ones((H, W, 4), dtype=float)
        # Base: light pink
        rgba[..., 0] = 0xF6/255.0; rgba[..., 1] = 0xE6/255.0; rgba[..., 2] = 0xE6/255.0; rgba[..., 3] = 1.0
        # map feasible cells with Blues
        cmap_blues = shifted_cmap("Blues", start=0.2, end=1.0)
        blues_rgba = cmap_blues(np.clip(norm_grid, 0.0, 1.0))
        feas_idx = mask
        rgba[feas_idx] = blues_rgba[feas_idx]
        # Infeasible proximity shading (Manhattan distance to nearest feasible)
        infeas_idx = ~mask
        if feas_idx.any() and infeas_idx.any():
            inf = np.full((H, W), np.inf, dtype=float)
            inf[feas_idx] = 0.0
            # two-pass manhattan distance transform
            for i in range(H):
                for j in range(W):
                    if i > 0:
                        inf[i, j] = min(inf[i, j], inf[i-1, j] + 1)
                    if j > 0:
                        inf[i, j] = min(inf[i, j], inf[i, j-1] + 1)
            for i in range(H-1, -1, -1):
                for j in range(W-1, -1, -1):
                    if i+1 < H:
                        inf[i, j] = min(inf[i, j], inf[i+1, j] + 1)
                    if j+1 < W:
                        inf[i, j] = min(inf[i, j], inf[i, j+1] + 1)
            d = inf
            d = d[infeas_idx]
            if d.size > 0 and np.isfinite(d).any():
                q = np.nanpercentile(d[np.isfinite(d)], 95.0)
                q = max(q, 1.0)
                prox = np.clip((inf / q), 0.0, 1.0)
                # invert so near boundary -> darker red
                prox = 1.0 - prox
                reds = shifted_cmap("Reds", start=0.3, end=0.95)
                reds_rgba = reds(prox)
                # apply only on infeasible cells
                rgba[infeas_idx] = reds_rgba[infeas_idx]
        overlay = ax.imshow(rgba, interpolation="nearest", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        if not a.paper:
            ax.set_title(f"Space-filling landscape ({score}); K={K}, V={V}")
        # Colorbar for feasible values only
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_blues)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if not a.paper:
            cbar.set_label(f"Normalized {score} (feasible, blue)")
        out_png = os.path.join(a.out_dir, f"spacefill_{score}.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved space-filling landscape to {out_png}; side={side}, filled={int(mask.sum())}/{side*side}")


    def plot_space_filling_feasibility_map(env: CloudSchedulingGymEnvironment, a: EnumArgs,
                                        sf_limit: int | None = None):
        """Create a Hilbert-ordered 2D map over ALL sequences, highlighting only feasibility.
        - Each cell corresponds to a full ordered sequence of (task, vm) pairs of length K.
        - Feasible = bright color; infeasible = dark background.
        """
        env.reset(seed=a.seed)
        assert env.state is not None
        K = len(env.state.task_states) - 2  # exclude dummy start/end
        if K <= 0:
            print("No real tasks to enumerate.")
            return
        real_tasks = tuple(range(1, 1 + K))
        V = len(env.state.static_state.vms)

        # Total sequences
        total_perm = 1
        for i in range(2, K + 1):
            total_perm *= i
        vpow = 1
        for _ in range(K):
            vpow *= V
        N_total = total_perm * vpow

        cap = sf_limit if sf_limit is not None else N_total
        side = hilbert_side_for_count(N_total if cap is None else cap)
        # Ensure output directory exists
        os.makedirs(a.out_dir, exist_ok=True)
        grid = np.full((side, side), np.nan, dtype=float)

        counted = 0
        pbar = tqdm(total=cap, desc="SpaceFillFeasible", unit="seq") if cap is not None else None
        e = build_env(a)  # reuse env; reset per sequence
        for perm in itertools.permutations(real_tasks, K):
            for vm_choice in itertools.product(range(V), repeat=K):
                if cap is not None and counted >= cap:
                    break
                idx = seq_to_linear_index(perm, vm_choice, K, V)
                seq = list(zip(perm, vm_choice))
                # Check feasibility by stepping through the exact ordered pairs
                e.reset(seed=a.seed)
                feasible = True
                for (t, v) in seq:
                    if t not in ready_tasks(e) or v not in compatible_vms(e, t):
                        feasible = False
                        break
                    _o, _r, term, trunc, _ = e.step(EnvAction(task_id=t, vm_id=v))
                    if term or trunc:
                        # Termination before consuming full sequence is still considered feasible
                        break
                x, y = hilbert_d2xy(side, idx % (side * side))
                grid[y, x] = 1.0 if feasible else 0.0
                counted += 1
                if pbar is not None:
                    pbar.update(1)
            if cap is not None and counted >= cap:
                break
        if pbar is not None:
            pbar.close()

        # Plot feasibility only: feasible vs infeasible in distinct colors
        os.makedirs(a.out_dir, exist_ok=True)
        sns.set_theme(context="paper", style="white", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=a.dpi)
        # Two-color discrete map: infeasible -> very light pink, feasible -> deep red
        cmap = ListedColormap(["#F6E6E6", "#B31B34"])  # 0: infeasible, 1: feasible
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        # Replace remaining NaNs (unfilled due to cap padding) with np.nan to show as background
        # We already filled infeasible as 0.0 and feasible as 1.0
        im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        if not a.paper:
            ax.set_title(f"Space-filling feasibility map; K={K}, V={V}")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        if not a.paper:
            cbar.set_ticklabels(["infeasible", "feasible"])
        out_png = os.path.join(a.out_dir, "spacefill_feasible.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        total_cells = side * side
        feasible_cells = int(np.sum(grid == 1.0))
        infeasible_cells = int(np.sum(grid == 0.0))
        print(f"Saved feasibility map to {out_png}; side={side}, feasible={feasible_cells}, infeasible={infeasible_cells}, total_cells={total_cells}")


    def plot_space_filling_both(env: CloudSchedulingGymEnvironment, a: EnumArgs,
                                sf_limit: int | None):
        """Enumerate sequences once and render two plots: makespan and energy.
        Infeasible cells are medium red; feasible cells are blue shades with per-plot normalization.
        """
        env.reset(seed=a.seed)
        assert env.state is not None
        # Ensure output directory exists for saves
        os.makedirs(a.out_dir, exist_ok=True)
        K = len(env.state.task_states) - 2
        if K <= 0:
            print("No real tasks to enumerate.")
            return
        real_tasks = tuple(range(1, 1 + K))
        V = len(env.state.static_state.vms)

        # Total sequences
        total_perm = 1
        for i in range(2, K + 1):
            total_perm *= i
        vpow = 1
        for _ in range(K):
            vpow *= V
        N_total = total_perm * vpow

        cap = sf_limit if sf_limit is not None else N_total
        side = hilbert_side_for_count(N_total if cap is None else cap)
        # Ensure output directory exists for surface outputs
        os.makedirs(a.out_dir, exist_ok=True)
        scores = ["makespan", "energy"]
        grids: dict[str, np.ndarray] = {s: np.full((side, side), np.nan, dtype=float) for s in scores}
        feas_mask = np.zeros((side, side), dtype=bool)
        visited = np.zeros((side, side), dtype=bool)

        counted = 0
        pbar = tqdm(total=cap, desc="SpaceFillBoth", unit="seq") if cap is not None else None
        # Reuse a single environment object; reset per sequence
        e = build_env(a)
        for perm in itertools.permutations(real_tasks, K):
            for vm_choice in itertools.product(range(V), repeat=K):
                idx = seq_to_linear_index(perm, vm_choice, K, V)
                if cap is not None and counted >= cap:
                    break
                seq = list(zip(perm, vm_choice))
                # Simulate this sequence once
                e.reset(seed=a.seed)
                info_last: Dict[str, Any] | None = None
                feasible = True
                for (t, v) in seq:
                    if t not in ready_tasks(e) or v not in compatible_vms(e, t):
                        feasible = False
                        break
                    _obs, _r, term, trunc, info_last = e.step(EnvAction(task_id=t, vm_id=v))
                    if term or trunc:
                        break
                x, y = hilbert_d2xy(side, idx % (side * side))
                visited[y, x] = True
                if feasible and isinstance(info_last, dict):
                    feas_mask[y, x] = True
                    vals = {}
                    # Collect metrics
                    per_vm_ct = [vm.completion_time for vm in e.state.vm_states] if e.state is not None else []
                    vals["makespan"] = float(max(per_vm_ct) if per_vm_ct else 0.0)
                    vals["energy"] = float(info_last.get("total_energy", 0.0))
                    for k, v in vals.items():
                        grids[k][y, x] = v
                counted += 1
                if pbar is not None:
                    pbar.update(1)
            if cap is not None and counted >= cap:
                break
        if pbar is not None:
            pbar.close()

        # Normalize each grid independently to [0,1]
        for k, G in grids.items():
            mask = ~np.isnan(G)
            if mask.any():
                vmin = float(np.nanmin(G))
                vmax = float(np.nanmax(G))
                if vmax > vmin:
                    G[mask] = (G[mask] - vmin) / (vmax - vmin)
                else:
                    G[mask] = 0.0

        # Compute proximity (Manhattan) distance to feasible region for boundary band shading
        H, W = side, side
        dist = np.full((H, W), np.inf, dtype=float)
        dist[feas_mask] = 0.0
        if feas_mask.any():
            for i in range(H):
                for j in range(W):
                    if i > 0:
                        dist[i, j] = min(dist[i, j], dist[i-1, j] + 1)
                    if j > 0:
                        dist[i, j] = min(dist[i, j], dist[i, j-1] + 1)
            for i in range(H-1, -1, -1):
                for j in range(W-1, -1, -1):
                    if i+1 < H:
                        dist[i, j] = min(dist[i, j], dist[i+1, j] + 1)
                    if j+1 < W:
                        dist[i, j] = min(dist[i, j], dist[i, j+1] + 1)
        # Render two separate images with the red/blue scheme and boundary band transparency (no remapping)
        os.makedirs(a.out_dir, exist_ok=True)
        sns.set_theme(context="paper", style="white", font_scale=1.1)
        for k in scores:
            G = grids[k]
            mask_vals = ~np.isnan(G)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=a.dpi)
            # Base light background
            rgba = np.ones((H, W, 4), dtype=float)
            rgba[..., 0] = 0xFA/255.0; rgba[..., 1] = 0xFA/255.0; rgba[..., 2] = 0xFA/255.0; rgba[..., 3] = 1.0
            # Feasible as blue shades
            cmap_blues = shifted_cmap("Blues", start=0.2, end=1.0)
            blues_rgba = cmap_blues(np.clip(G, 0.0, 1.0))
            rgba[mask_vals] = blues_rgba[mask_vals]
            # Infeasible as red with alpha decaying beyond a boundary band
            band = 3.0
            reds = shifted_cmap("Reds", start=0.3, end=0.95)
            infeas_idx = ~feas_mask & visited
            if np.any(infeas_idx):
                # alpha proportional to (band - dist)/band
                alpha = np.clip((band - dist) / band, 0.0, 1.0)
                red_rgba = reds(np.clip(1.0 - (dist / max(band, 1.0)), 0.0, 1.0))
                # Apply color and alpha on infeasible
                rgba[infeas_idx] = red_rgba[infeas_idx]
                rgba[..., 3] = np.where(infeas_idx, alpha, rgba[..., 3])
            ax.imshow(rgba, interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if not a.paper:
                ax.set_title(f"Space-filling {k}; K={K}, V={V}")
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_blues)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            if not a.paper:
                cbar.set_label(f"Normalized {k} (feasible, blue)")
            out_png = os.path.join(a.out_dir, f"spacefill_{k}.png")
            plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close(fig)
            print(f"Saved space-filling {k} to {out_png}")


    def enumerate_spacefill_once(env: CloudSchedulingGymEnvironment, a: EnumArgs,
                                 sf_limit: int | None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
        """
        Enumerate sequences and return per-metric grids, feasibility mask, visited mask, and grid side length.
        Grids hold RAW metric values (not normalized). Metrics: 'makespan', 'energy'.
        """
        env.reset(seed=a.seed)
        assert env.state is not None
        K = len(env.state.task_states) - 2
        if K <= 0:
            return {"makespan": np.zeros((1, 1)), "energy": np.zeros((1, 1))}, np.zeros((1, 1), dtype=bool), np.zeros((1, 1), dtype=bool), 1
        real_tasks = tuple(range(1, 1 + K))
        V = len(env.state.static_state.vms)

        # Total sequences and grid side
        total_perm = 1
        for i in range(2, K + 1):
            total_perm *= i
        vpow = 1
        for _ in range(K):
            vpow *= V
        N_total = total_perm * vpow
        cap = sf_limit if sf_limit is not None else N_total
        side = hilbert_side_for_count(N_total if cap is None else cap)

        grids: dict[str, np.ndarray] = {s: np.full((side, side), np.nan, dtype=float) for s in ("makespan", "energy")}
        feas_mask = np.zeros((side, side), dtype=bool)
        visited = np.zeros((side, side), dtype=bool)

        counted = 0
        pbar = tqdm(total=cap, desc="SpaceFillEnum", unit="seq") if cap is not None and a.show_progress else None
        e = build_env(a)
        for perm in itertools.permutations(real_tasks, K):
            for vm_choice in itertools.product(range(V), repeat=K):
                idx = seq_to_linear_index(perm, vm_choice, K, V)
                if cap is not None and counted >= cap:
                    break
                seq = list(zip(perm, vm_choice))
                e.reset(seed=a.seed)
                info_last: Dict[str, Any] | None = None
                feasible = True
                for (t, v) in seq:
                    if t not in ready_tasks(e) or v not in compatible_vms(e, t):
                        feasible = False
                        break
                    _obs, _r, term, trunc, info_last = e.step(EnvAction(task_id=t, vm_id=v))
                    if term or trunc:
                        break
                x, y = hilbert_d2xy(side, idx % (side * side))
                visited[y, x] = True
                if feasible and isinstance(info_last, dict):
                    feas_mask[y, x] = True
                    per_vm_ct = [vm.completion_time for vm in e.state.vm_states] if e.state is not None else []
                    grids["makespan"][y, x] = float(max(per_vm_ct) if per_vm_ct else 0.0)
                    grids["energy"][y, x] = float(info_last.get("total_energy", 0.0))
                counted += 1
                if pbar is not None:
                    pbar.update(1)
            if cap is not None and counted >= cap:
                break
        if pbar is not None:
            pbar.close()
        return grids, feas_mask, visited, side


    def compose_linear_vs_gnp_panels(a: EnumArgs, sf_limit: int | None, metric: str = "makespan") -> None:
        """Compose a 1x2 panel (linear vs. GNP) with matched color scales for the given metric.
        Saves PNG and PDF under a.out_dir as spacefill_panel_<metric>.{png,pdf}.
        """
        assert metric in ("makespan", "energy")
        # Build args for both DAG types
        a_linear = EnumArgs(**{**a.__dict__})
        a_linear.dag_method = "linear"
        a_gnp = EnumArgs(**{**a.__dict__})
        a_gnp.dag_method = "gnp"

        env_lin = build_env(a_linear)
        env_gnp = build_env(a_gnp)
        grids_lin, feas_lin, visited_lin, side_lin = enumerate_spacefill_once(env_lin, a_linear, sf_limit)
        grids_gnp, feas_gnp, visited_gnp, side_gnp = enumerate_spacefill_once(env_gnp, a_gnp, sf_limit)

        G_lin_raw = grids_lin[metric]
        G_gnp_raw = grids_gnp[metric]
        m_lin = feas_lin & ~np.isnan(G_lin_raw)
        m_gnp = feas_gnp & ~np.isnan(G_gnp_raw)

        # Determine normalization bounds
        fixed_vmin = None
        fixed_vmax = None
        if metric == "makespan":
            fixed_vmin = a.fixed_vmin_makespan
            fixed_vmax = a.fixed_vmax_makespan
        else:
            fixed_vmin = a.fixed_vmin_energy
            fixed_vmax = a.fixed_vmax_energy

        if fixed_vmin is not None and fixed_vmax is not None and fixed_vmax > fixed_vmin:
            vmin = float(fixed_vmin)
            vmax = float(fixed_vmax)
        else:
            vals = []
            if np.any(m_lin):
                vals.append(G_lin_raw[m_lin])
            if np.any(m_gnp):
                vals.append(G_gnp_raw[m_gnp])
            if len(vals) == 0:
                print("No feasible values found to compose panels.")
                return
            allv = np.concatenate(vals)
            vmin = float(np.nanmin(allv))
            vmax = float(np.nanmax(allv))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                print("Invalid vmin/vmax for panel normalization.")
                return

        # Normalize to [0,1] with shared bounds
        def norm_with_bounds(G: np.ndarray, mask: np.ndarray) -> np.ndarray:
            N = np.full_like(G, np.nan, dtype=float)
            if np.any(mask):
                N[mask] = (G[mask] - vmin) / (vmax - vmin)
                N[mask] = np.clip(N[mask], 0.0, 1.0)
            return N

        G_lin = norm_with_bounds(G_lin_raw, m_lin)
        G_gnp = norm_with_bounds(G_gnp_raw, m_gnp)

        # Compute proximity fields for infeasible shading
        def manhattan_dist_to_mask(feas_mask: np.ndarray) -> np.ndarray:
            H, W = feas_mask.shape
            dist = np.full((H, W), np.inf, dtype=float)
            dist[feas_mask] = 0.0
            if feas_mask.any():
                for i in range(H):
                    for j in range(W):
                        if i > 0:
                            dist[i, j] = min(dist[i, j], dist[i-1, j] + 1)
                        if j > 0:
                            dist[i, j] = min(dist[i, j], dist[i, j-1] + 1)
                for i in range(H-1, -1, -1):
                    for j in range(W-1, -1, -1):
                        if i+1 < H:
                            dist[i, j] = min(dist[i, j], dist[i+1, j] + 1)
                        if j+1 < W:
                            dist[i, j] = min(dist[i, j], dist[i, j+1] + 1)
            return dist

        dist_lin = manhattan_dist_to_mask(feas_lin)
        dist_gnp = manhattan_dist_to_mask(feas_gnp)

        # Render side-by-side 2D panels with matched blue colormap
        os.makedirs(a.out_dir, exist_ok=True)
        sns.set_theme(context="paper", style="white", font_scale=1.1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=a.dpi, constrained_layout=True)
        panels = [
            (G_lin, feas_lin, visited_lin, dist_lin, side_lin, "Linear"),
            (G_gnp, feas_gnp, visited_gnp, dist_gnp, side_gnp, "GNP"),
        ]
        cmap_blues = shifted_cmap("Blues", start=0.2, end=1.0)
        reds = shifted_cmap("Reds", start=0.3, end=0.95)
        band = 3.0

        # Statistics for annotations
        def counts(feas: np.ndarray, visited: np.ndarray) -> tuple[int, int]:
            visited_count = int(np.sum(visited))
            feasible_count = int(np.sum(feas & visited))
            return feasible_count, visited_count

        for ax, (G, feas, visited, dist_map, side, title) in zip(axes, panels):
            H, W = G.shape
            rgba = np.ones((H, W, 4), dtype=float)
            rgba[..., 0] = 0xFA/255.0; rgba[..., 1] = 0xFA/255.0; rgba[..., 2] = 0xFA/255.0; rgba[..., 3] = 1.0
            mask_vals = ~np.isnan(G)
            blues_rgba = cmap_blues(np.clip(G, 0.0, 1.0))
            rgba[mask_vals] = blues_rgba[mask_vals]
            infeas_idx = ~feas & visited
            if np.any(infeas_idx):
                alpha = np.clip((band - dist_map) / band, 0.0, 1.0)
                red_rgba = reds(np.clip(1.0 - (dist_map / max(band, 1.0)), 0.0, 1.0))
                rgba[infeas_idx] = red_rgba[infeas_idx]
                rgba[..., 3] = np.where(infeas_idx, alpha, rgba[..., 3])
            ax.imshow(rgba, interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if not a.paper:
                ax.set_title(f"{title}")
            # Terse annotation in paper mode, detailed otherwise
            feas_c, vis_c = counts(feas, visited)
            if a.paper:
                terse = f"{title}: {feas_c}/{vis_c} feas"
                ax.text(0.02, 0.98, terse, transform=ax.transAxes, va='top', ha='left', fontsize=7.5, color='black')
            else:
                feas_frac = (feas_c / vis_c) if vis_c > 0 else 0.0
                textstr = f"{title}\nFeasible: {feas_c}/{vis_c} ({feas_frac:.1%})"
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, va='top', ha='left', fontsize=9.5, weight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.08', facecolor='white', alpha=0.95, edgecolor='darkred', linewidth=1.0))

        # Shared colorbar for the normalized metric
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_blues)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        if not a.paper:
            cbar.set_label(f"Normalized {metric} (feasible, blue)")

        out_png = os.path.join(a.out_dir, f"spacefill_panel_{metric}.png")
        out_pdf = os.path.join(a.out_dir, f"spacefill_panel_{metric}.pdf")
        plt.savefig(out_png, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved side-by-side panel to {out_png} and {out_pdf}; metric={metric}, vmin={vmin:.3f}, vmax={vmax:.3f}")

        return fig


def single_3d_surface(a: EnumArgs, sf_limit: int | None, metric: str, dag_type: str):
    """Generate individual 3D surface plot for either linear or GNP DAG."""
    # Build environment with the specified DAG type
    a_single = EnumArgs(
        seed=a.seed, vm_count=a.vm_count, workflow_count=a.workflow_count,
        gnp_min_n=a.gnp_min_n, gnp_max_n=a.gnp_max_n,
        dag_method=dag_type, min_task_length=a.min_task_length,
        max_task_length=a.max_task_length, min_cpu_speed=a.min_cpu_speed,
        max_cpu_speed=a.max_cpu_speed, max_memory_gb=a.max_memory_gb,
        limit_solutions=None, score=metric, show_progress=a.show_progress,
        estimate_samples=0, out_dir=a.out_dir, dpi=a.dpi, paper=a.paper,
        fixed_vmin_makespan=a.fixed_vmin_makespan, fixed_vmax_makespan=a.fixed_vmax_makespan,
        fixed_vmin_energy=a.fixed_vmin_energy, fixed_vmax_energy=a.fixed_vmax_energy
    )
    
    env = build_env(a_single)
    env.reset(seed=a_single.seed)
    
    # Enumerate and collect data
    grids_raw, feas_mask, visited, grid_size = enumerate_spacefill_once(env, a_single, sf_limit)
    
    # Get the specific metric data
    G_raw = grids_raw[metric]
    
    # Normalize scores
    def norm(G: np.ndarray):
        N = np.zeros_like(G)
        mask = feas_mask & visited
        if np.any(mask):
            vmin, vmax = G[mask].min(), G[mask].max()
            if vmax > vmin:
                N[mask] = (G[mask] - vmin) / (vmax - vmin)
                N[mask] = np.clip(N[mask], 0.0, 1.0)
        return N
    
    Z = norm(G_raw)
    
    # Set up single 3D plot
    os.makedirs(a.out_dir, exist_ok=True)
    sns.set_theme(context="paper", style="white", font_scale=1.0)
    fig = plt.figure(figsize=(7, 6), dpi=a.dpi, constrained_layout=False)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Use same colormap as the side-by-side version
    try:
        from matplotlib import cm
        cmap_surface = cm.get_cmap('RdYlGn')
    except Exception:
        cmap_surface = plt.get_cmap('RdYlGn')
    
    # Compute tight bounding box around feasible cells to zoom-in
    H, W = Z.shape
    feas_idx = np.argwhere(feas_mask)
    if feas_idx.size > 0:
        pad = 2
        iy_min = max(0, int(feas_idx[:, 0].min()) - pad)
        iy_max = min(H - 1, int(feas_idx[:, 0].max()) + pad)
        ix_min = max(0, int(feas_idx[:, 1].min()) - pad)
        ix_max = min(W - 1, int(feas_idx[:, 1].max()) + pad)
    else:
        iy_min, iy_max, ix_min, ix_max = 0, H - 1, 0, W - 1

    # Create cropped mesh and values
    y_range = np.arange(iy_min, iy_max + 1)
    x_range = np.arange(ix_min, ix_max + 1)
    X, Y = np.meshgrid(x_range, y_range)
    Z_plot = np.array(Z[np.ix_(y_range, x_range)], copy=True)
    feas_crop = feas_mask[np.ix_(y_range, x_range)]
    # Mask infeasible as gaps in the surface
    Z_plot[~feas_crop] = np.nan
    
    # Convert normalized Z to facecolors
    facecolors = cmap_surface(np.clip(Z_plot, 0.0, 1.0))
    
    # Draw surface
    surf = ax.plot_surface(
        X, Y, Z_plot,
        facecolors=facecolors,
        linewidth=0.2,
        edgecolor=(0, 0, 0, 0.35),
        antialiased=True,
        rstride=1, cstride=1, shade=False
    )
    
    # Strong wireframe overlay
    try:
        ax.plot_wireframe(X, Y, Z_plot, rstride=2, cstride=2, color=(1, 1, 1, 0.65), linewidth=0.35)
    except Exception:
        pass
    
    # Show cube grid and axes
    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_edgecolor((0.2, 0.2, 0.2, 1.0))
            axis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    except Exception:
        pass
    ax.grid(True)
    ax.tick_params(axis='both', which='both', labelsize=7, pad=2, length=2)
    
    # Tight limits to the cropped region
    ax.set_xlim(ix_min, ix_max)
    ax.set_ylim(iy_min, iy_max)
    try:
        ax.set_xticks(np.linspace(ix_min, ix_max, num=4))
        ax.set_yticks(np.linspace(iy_min, iy_max, num=4))
        ax.set_zticks(np.linspace(0.0, 1.0, num=4))
    except Exception:
        pass
    
    # Adjust viewpoint and aspect
    try:
        ax.set_box_aspect((1.0, 1.0, 0.55))
        ax.set_proj_type('persp')
    except Exception:
        pass
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=30, azim=-70)
    
    # Camera distance
    try:
        ax.dist = 5.6
    except Exception:
        pass
    
    if not a.paper:
        ax.set_title(f"{dag_type.upper()} DAG - {metric.capitalize()}")
    
    # Feasibility annotation
    visited_count = int(np.sum(visited))
    feasible_count = int(np.sum(feas_mask & visited))
    feas_frac = (feasible_count / visited_count) if visited_count > 0 else 0.0
    if a.paper:
        ax.text2D(0.02, 0.98, f"{dag_type.upper()}: {feasible_count}/{visited_count} feas", 
                  transform=ax.transAxes, fontsize=7.5)
    else:
        ax.text2D(0.015, 0.985, f"{dag_type.upper()}\nFeasible: {feasible_count}/{visited_count} ({feas_frac:.1%})",
                  transform=ax.transAxes, va='top', ha='left', fontsize=9.0, weight='bold', color='darkred',
                  bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.08', facecolor='white', alpha=0.95, 
                           edgecolor='darkred', linewidth=1.0))
    
    # Annotate global optimum
    try:
        if np.any(~np.isnan(Z_plot)):
            flat_idx = int(np.nanargmin(Z_plot))
            iyc, ixc = np.unravel_index(flat_idx, Z_plot.shape)
            gx = x_range[ixc]; gy = y_range[iyc]
            ax.scatter([gx], [gy], [Z_plot[iyc, ixc]], c=['cyan'], s=24, depthshade=True, 
                      edgecolors='k', linewidths=0.3)
            if not a.paper:
                ax.text(gx, gy, min(1.0, Z_plot[iyc, ixc] + 0.05), "Global optimum", 
                       color='black', fontsize=7)
    except Exception:
        pass
    
    # Add colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_surface)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label(f'Normalized {metric.capitalize()}', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Save files
    base_name = f"spacefill_3d_{dag_type}_{metric}"
    png_path = os.path.join(a.out_dir, f"{base_name}.png")
    pdf_path = os.path.join(a.out_dir, f"{base_name}.pdf")
    
    fig.savefig(png_path, dpi=a.dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved individual 3D surface to {png_path} and {pdf_path}; dag_type={dag_type}, metric={metric}")


def compose_linear_vs_gnp_3d(a: EnumArgs, sf_limit: int | None, metric: str):
    """Generate side-by-side 3D surface plots for linear vs GNP DAGs with matched color scales."""
    # Build args for both DAG types
    a_linear = EnumArgs(**{**a.__dict__}); a_linear.dag_method = "linear"
    a_gnp = EnumArgs(**{**a.__dict__}); a_gnp.dag_method = "gnp"

    env_lin = build_env(a_linear)
    env_gnp = build_env(a_gnp)
    grids_lin, feas_lin, visited_lin, side_lin = enumerate_spacefill_once(env_lin, a_linear, sf_limit)
    grids_gnp, feas_gnp, visited_gnp, side_gnp = enumerate_spacefill_once(env_gnp, a_gnp, sf_limit)

    G_lin_raw = grids_lin[metric]; G_gnp_raw = grids_gnp[metric]
    m_lin = feas_lin & ~np.isnan(G_lin_raw)
    m_gnp = feas_gnp & ~np.isnan(G_gnp_raw)

    # Determine normalization bounds
    fixed_vmin = a.fixed_vmin_makespan if metric == "makespan" else a.fixed_vmin_energy
    fixed_vmax = a.fixed_vmax_makespan if metric == "makespan" else a.fixed_vmax_energy
    if fixed_vmin is not None and fixed_vmax is not None and fixed_vmax > fixed_vmin:
        vmin = float(fixed_vmin); vmax = float(fixed_vmax)
    else:
        vals = []
        if np.any(m_lin): vals.append(G_lin_raw[m_lin])
        if np.any(m_gnp): vals.append(G_gnp_raw[m_gnp])
        if len(vals) == 0:
            print("No feasible values for 3D panels.")
            return
        allv = np.concatenate(vals)
        vmin = float(np.nanmin(allv)); vmax = float(np.nanmax(allv))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            print("Invalid vmin/vmax for 3D panel normalization.")
            return

    # Normalize to [0,1] with shared bounds; NaN for infeasible
    def norm(G: np.ndarray, mask: np.ndarray) -> np.ndarray:
        N = np.full_like(G, np.nan, dtype=float)
        if np.any(mask):
            N[mask] = (G[mask] - vmin) / (vmax - vmin)
            N[mask] = np.clip(N[mask], 0.0, 1.0)
        return N

    Z_lin = norm(G_lin_raw, m_lin)
    Z_gnp = norm(G_gnp_raw, m_gnp)

    # Set up figure with two 3D subplots
    os.makedirs(a.out_dir, exist_ok=True)
    sns.set_theme(context="paper", style="white", font_scale=1.0)
    fig = plt.figure(figsize=(11.8, 6.0), dpi=a.dpi, constrained_layout=False)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Reduce whitespace between subplots to bring surfaces visually closer and larger
    fig.subplots_adjust(left=0.06, right=0.98, top=0.985, bottom=0.045, wspace=0.14)
    # Use a green-yellow-red colormap similar to the reference (green high, red low)
    try:
        from matplotlib import cm
        cmap_surface = cm.get_cmap('RdYlGn')
    except Exception:
        cmap_surface = plt.get_cmap('RdYlGn')

    def plot_surface(ax, Z: np.ndarray, title: str, feas: np.ndarray):
        H, W = Z.shape
        # Compute tight bounding box around feasible cells to zoom-in
        feas_idx = np.argwhere(feas)
        if feas_idx.size > 0:
            pad = 2
            iy_min = max(0, int(feas_idx[:, 0].min()) - pad)
            iy_max = min(H - 1, int(feas_idx[:, 0].max()) + pad)
            ix_min = max(0, int(feas_idx[:, 1].min()) - pad)
            ix_max = min(W - 1, int(feas_idx[:, 1].max()) + pad)
        else:
            iy_min, iy_max, ix_min, ix_max = 0, H - 1, 0, W - 1

        # Create cropped mesh and values
        y_range = np.arange(iy_min, iy_max + 1)
        x_range = np.arange(ix_min, ix_max + 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z_plot = np.array(Z[np.ix_(y_range, x_range)], copy=True)
        feas_crop = feas[np.ix_(y_range, x_range)]
        # Mask infeasible as gaps in the surface
        Z_plot[~feas_crop] = np.nan
        # Convert normalized Z to facecolors using the rainbow colormap
        facecolors = cmap_surface(np.clip(Z_plot, 0.0, 1.0))
        # Draw a shaded surface with subtle mesh lines (wireframe-like look)
        surf = ax.plot_surface(
            X, Y, Z_plot,
            facecolors=facecolors,
            linewidth=0.2,
            edgecolor=(0, 0, 0, 0.35),
            antialiased=True,
            rstride=1, cstride=1, shade=False
        )
        # Strong wireframe overlay to mimic the dense grid mesh
        try:
            ax.plot_wireframe(X, Y, Z_plot, rstride=2, cstride=2, color=(1, 1, 1, 0.65), linewidth=0.35)
        except Exception:
            pass
        # Show cube grid and axes like the reference
        try:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_edgecolor((0.2, 0.2, 0.2, 1.0))
                axis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        except Exception:
            pass
        ax.grid(True)
        # Keep ticks but small for a clean look
        ax.tick_params(axis='both', which='both', labelsize=7, pad=2, length=2)
        # Tight limits to the cropped region so feasible area fills the frame
        ax.set_xlim(ix_min, ix_max)
        ax.set_ylim(iy_min, iy_max)
        # Reduce number of ticks inside the crop
        try:
            ax.set_xticks(np.linspace(ix_min, ix_max, num=4))
            ax.set_yticks(np.linspace(iy_min, iy_max, num=4))
            ax.set_zticks(np.linspace(0.0, 1.0, num=4))
        except Exception:
            pass
        # Adjust viewpoint and aspect to match the dramatic perspective
        try:
            ax.set_box_aspect((1.0, 1.0, 0.55))
            ax.set_proj_type('persp')
        except Exception:
            pass
        ax.set_zlim(0.0, 1.0)
        ax.view_init(elev=30, azim=-70)
        # Bring camera closer for a more immersive, highlighted 3D feel
        try:
            ax.dist = 5.6  # slightly farther to reduce the cropped feel
        except Exception:
            pass
        if not a.paper:
            ax.set_title(title)
        # Feasibility annotation
        visited_count = int(np.sum(visited_lin if ax is ax1 else visited_gnp))
        feasible_count = int(np.sum(feas & (visited_lin if ax is ax1 else visited_gnp)))
        feas_frac = (feasible_count / visited_count) if visited_count > 0 else 0.0
        if a.paper:
            ax.text2D(0.02, 0.98, f"{title}: {feasible_count}/{visited_count} feas", transform=ax.transAxes, fontsize=7.5)
        else:
            ax.text2D(0.015, 0.985, f"{title}\nFeasible: {feasible_count}/{visited_count} ({feas_frac:.1%})",
                          transform=ax.transAxes, va='top', ha='left', fontsize=9.0, weight='bold', color='darkred',
                          bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.08', facecolor='white', alpha=0.95, edgecolor='darkred', linewidth=1.0))
        # Annotate global optimum (min normalized value within the crop)
        try:
            if np.any(~np.isnan(Z_plot)):
                flat_idx = int(np.nanargmin(Z_plot))
                iyc, ixc = np.unravel_index(flat_idx, Z_plot.shape)
                gx = x_range[ixc]; gy = y_range[iyc]
                ax.scatter([gx], [gy], [Z_plot[iyc, ixc]], c=['cyan'], s=24, depthshade=True, edgecolors='k', linewidths=0.3)
                if not a.paper:
                    ax.text(gx, gy, min(1.0, Z_plot[iyc, ixc] + 0.05), "Global optimum", color='black', fontsize=7)
        except Exception:
            pass
        return surf

    surf1 = plot_surface(ax1, Z_lin, "Linear", feas_lin)
    surf2 = plot_surface(ax2, Z_gnp, "GNP", feas_gnp)

    # Shared colorbar (normalized 0..1)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_surface)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    if not a.paper:
        cbar.set_label(f"Normalized {metric} (feasible valleys)")

    out_png = os.path.join(a.out_dir, f"spacefill_panel3d_{metric}.png")
    out_pdf = os.path.join(a.out_dir, f"spacefill_panel3d_{metric}.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D side-by-side panel to {out_png} and {out_pdf}; metric={metric}, vmin={vmin:.3f}, vmax={vmax:.3f}")



def plot_space_filling_surface(env: CloudSchedulingGymEnvironment, a: EnumArgs,
                            sf_limit: int | None):
        """Render publication-quality 3D surface landscapes using the Hilbert grid as X-Y.
        Feasible cells form valleys (normalized metric), with enhanced styling for ML conferences.
        Saves one surface per metric (makespan, energy) as both PNG and PDF.
        """
        env.reset(seed=a.seed)
        assert env.state is not None
        os.makedirs(a.out_dir, exist_ok=True)
        K = len(env.state.task_states) - 2
        if K <= 0:
            print("No real tasks to enumerate.")
            return
        real_tasks = tuple(range(1, 1 + K))
        V = len(env.state.static_state.vms)

        # Total sequences
        total_perm = 1
        for i in range(2, K + 1):
            total_perm *= i
        vpow = 1
        for _ in range(K):
            vpow *= V
        N_total = total_perm * vpow

        cap = sf_limit if sf_limit is not None else N_total
        side = hilbert_side_for_count(N_total if cap is None else cap)
        scores = ["makespan", "energy"]
        grids: dict[str, np.ndarray] = {s: np.full((side, side), np.nan, dtype=float) for s in scores}
        feas_mask = np.zeros((side, side), dtype=bool)
        visited = np.zeros((side, side), dtype=bool)

        counted = 0
        pbar = tqdm(total=cap, desc="SpaceFillSurface", unit="seq") if cap is not None else None
        e = build_env(a)
        for perm in itertools.permutations(real_tasks, K):
            for vm_choice in itertools.product(range(V), repeat=K):
                idx = seq_to_linear_index(perm, vm_choice, K, V)
                if cap is not None and counted >= cap:
                    break
                seq = list(zip(perm, vm_choice))
                e.reset(seed=a.seed)
                info_last: Dict[str, Any] | None = None
                feasible = True
                for (t, v) in seq:
                    if t not in ready_tasks(e) or v not in compatible_vms(e, t):
                        feasible = False
                        break
                    _obs, _r, term, trunc, info_last = e.step(EnvAction(task_id=t, vm_id=v))
                    if term or trunc:
                        break
                x, y = hilbert_d2xy(side, idx % (side * side))
                visited[y, x] = True
                if feasible and isinstance(info_last, dict):
                    feas_mask[y, x] = True
                    per_vm_ct = [vm.completion_time for vm in e.state.vm_states] if e.state is not None else []
                    grids["makespan"][y, x] = float(max(per_vm_ct) if per_vm_ct else 0.0)
                    grids["energy"][y, x] = float(info_last.get("total_energy", 0.0))
                counted += 1
                if pbar is not None:
                    pbar.update(1)
            if cap is not None and counted >= cap:
                break
        if pbar is not None:
            pbar.close()

        # Normalize feasible values per metric to [0,1]
        for k, G in grids.items():
            m = feas_mask & ~np.isnan(G)
            if m.any():
                vmin = float(np.nanmin(G[m]))
                vmax = float(np.nanmax(G[m]))
                if vmax > vmin:
                    G[m] = (G[m] - vmin) / (vmax - vmin)
                else:
                    G[m] = 0.0

        # Distance to feasible (for ridge height)
        H, W = side, side
        dist = np.full((H, W), np.inf, dtype=float)
        dist[feas_mask] = 0.0
        if feas_mask.any():
            for i in range(H):
                for j in range(W):
                    if i > 0:
                        dist[i, j] = min(dist[i, j], dist[i-1, j] + 1)
                    if j > 0:
                        dist[i, j] = min(dist[i, j], dist[i, j-1] + 1)
            for i in range(H-1, -1, -1):
                for j in range(W-1, -1, -1):
                    if i+1 < H:
                        dist[i, j] = min(dist[i, j], dist[i+1, j] + 1)
                    if j+1 < W:
                        dist[i, j] = min(dist[i, j], dist[i, j+1] + 1)

        # Prepare coordinates
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        band = 3.0

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        # Compute feasible stats for annotation
        visited_count = int(np.sum(visited))
        feasible_count = int(np.sum(feas_mask & visited))
        feas_frac = (feasible_count / visited_count) if visited_count > 0 else 0.0

        # Estimate critical path length using DAG and per-task minimal processing time
        def estimate_cp_length() -> float:
            assert env.state is not None
            static = env.state.static_state
            tasks = static.tasks
            vms = static.vms
            compat = set(static.compatibilities)
            # Real tasks 1..K
            real_ids = list(range(1, 1 + K))
            # min proc time per task on fastest compatible VM
            min_time = {t: float('inf') for t in real_ids}
            for t in real_ids:
                for vm_id in range(len(vms)):
                    if (t, vm_id) in compat:
                        cpu = max(1.0, float(vms[vm_id].cpu_speed_mips))
                        l = float(tasks[t].length)
                        min_time[t] = min(min_time[t], l / cpu)
                if not np.isfinite(min_time[t]):
                    # Fallback if none compatible (should be rare due to relax)
                    cpu_max = max(1.0, max(float(v.cpu_speed_mips) for v in vms))
                    min_time[t] = float(tasks[t].length) / cpu_max
            # Build DAG over real tasks
            edges = [(p, c) for (p, c) in env.state.task_dependencies if 1 <= p <= K and 1 <= c <= K]
            indeg = {t: 0 for t in real_ids}
            adj: dict[int, list[int]] = {t: [] for t in real_ids}
            for p, c in edges:
                adj[p].append(c)
                indeg[c] += 1
            # Kahn topo + longest path DP
            from collections import deque
            q = deque([t for t in real_ids if indeg[t] == 0])
            dist_cp = {t: min_time[t] for t in real_ids}
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist_cp[v] < dist_cp[u] + min_time[v]:
                        dist_cp[v] = dist_cp[u] + min_time[v]
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            return float(max(dist_cp.values()) if dist_cp else 0.0)

        cp_est = estimate_cp_length()
        
        # Store raw values for colorbar and optima marking
        raw_values = {k: [] for k in scores}
        for k in scores:
            G = grids[k]
            feas = feas_mask & ~np.isnan(G)
            if np.any(feas):
                raw_values[k] = G[feas]
        
        for k in scores:
            G = grids[k]
            Z = np.full((H, W), np.nan, dtype=float)
            # Feasible only: valleys by normalized metric; hide infeasible completely
            feas = feas_mask & ~np.isnan(G)
            Z[feas] = G[feas]
            
            # Enhanced figure setup for publication
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'axes.linewidth': 1.2,
                'figure.dpi': 300,
                'figure.facecolor': 'white'
            })
            
            # Square export in paper mode for tighter composition
            if a.paper:
                fig = plt.figure(figsize=(8, 8), dpi=300)
            else:
                fig = plt.figure(figsize=(10, 8), dpi=300)
            ax = fig.add_subplot(111, projection='3d')
            # Minimize whitespace: make axes fill the canvas
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            try:
                ax.set_position([0.0, 0.0, 1.0, 1.0])
            except Exception:
                pass
            # Ensure square XY aspect in paper mode (with modest Z scale)
            if a.paper:
                try:
                    ax.set_box_aspect((1, 1, 0.7))
                except Exception:
                    pass
            
            # Create beautiful gradient colormap with enhanced reds
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['#FFF5F5', '#FFE5E5', '#FFB3B3', '#FF8080', '#FF4D4D', '#FF1A1A', '#CC0000', '#990000']
            n_bins = 256
            cmap_beautiful_reds = LinearSegmentedColormap.from_list('beautiful_reds', colors, N=n_bins)
            
            # Enhanced surface coloring with smooth gradients
            rgba = np.zeros((H, W, 4), dtype=float)  # default transparent
            if np.any(feas):
                # Apply slight smoothing to values for better gradients
                G_smooth = G.copy()
                # Create beautiful color mapping with enhanced contrast
                reds_rgba = cmap_beautiful_reds(np.clip(G_smooth, 0.0, 1.0))
                rgba[feas] = reds_rgba[feas]
                rgba[feas, 3] = 0.95  # slightly transparent for depth
            
            # Create surface with enhanced visual properties
            surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, facecolors=rgba,
                                linewidth=0, antialiased=True, shade=True, alpha=0.95)

            # Feasibility gradient halos (underlay): highlight proximity to infeasible boundary
            # Use precomputed distance map `dist` and band parameter to modulate halo opacity
            try:
                dist_local = dist.copy()
            except NameError:
                dist_local = np.zeros_like(Z)
            # Halo strongest near boundary (small distance), fades inward
            halo_alpha = np.zeros((H, W), dtype=float)
            with np.errstate(invalid='ignore'):
                halo_alpha[feas] = np.exp(-(dist_local[feas] / max(1e-6, band)))
            # Golden halo color
            halo_rgb = np.array([1.0, 0.85, 0.3], dtype=float)
            halo_rgba = np.zeros((H, W, 4), dtype=float)
            halo_rgba[..., :3] = halo_rgb
            halo_rgba[..., 3] = np.clip(0.6 * halo_alpha, 0.0, 0.6)
            # Place halos on a shallow base plane below the main surface
            z_min = np.nanmin(Z[feas]) if np.any(feas) else 0.0
            z_max = np.nanmax(Z[feas]) if np.any(feas) else 1.0
            z_span = max(1e-6, (z_max - z_min))
            Z_halo = np.full_like(Z, z_min - 0.08 * z_span)
            ax.plot_surface(X, Y, Z_halo, rstride=2, cstride=2, facecolors=halo_rgba,
                            linewidth=0, antialiased=True, shade=False)
            
            # Removed global optimum marker for cleaner presentation
            
            # Compact colorbar (also in paper mode) to explain the metric scale
            if len(raw_values[k]) > 0:
                vmin, vmax = np.min(raw_values[k]), np.max(raw_values[k])
                sm = plt.cm.ScalarMappable(cmap=cmap_beautiful_reds,
                                        norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                try:
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    if a.paper:
                        axins = inset_axes(ax, width="2.2%", height="26%", loc='upper right',
                                           borderpad=0.4)
                    else:
                        axins = inset_axes(ax, width="3%", height="32%", loc='upper right',
                                           borderpad=0.6)
                    cbar = plt.colorbar(sm, cax=axins)
                except Exception:
                    cbar = plt.colorbar(sm, ax=ax, shrink=0.65 if a.paper else 0.7, aspect=25, pad=0.02)
                cbar.set_label(f'{k.capitalize()}', fontsize=8 if a.paper else 10, labelpad=8 if a.paper else 10, weight='bold')
                cbar.ax.tick_params(labelsize=7 if a.paper else 8, pad=1 if a.paper else 2)
                # Enhanced ticks with better formatting
                cbar.set_ticks([0, 0.5, 1.0])
                cbar.set_ticklabels([f'{vmin:.3f}', f'{(vmin+vmax)/2:.3f}', f'{vmax:.3f}'])
                # Add subtle border to colorbar
                cbar.outline.set_linewidth(0.8 if a.paper else 1.0)
                cbar.outline.set_edgecolor('gray')
            
            # Beautiful axis styling
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            
            # Create elegant transparent background
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.grid(False)
            
            # Set background to gradient
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            # Perfect viewing angle for dramatic effect (stronger relief in paper mode)
            if a.paper:
                ax.view_init(elev=42, azim=60)
            else:
                ax.view_init(elev=32, azim=47)
            
            # Add subtle lighting effect
            from matplotlib.colors import LightSource
            ls = LightSource(azdeg=315, altdeg=45)
            
            # Enhance the surface with contour lines for depth
            if np.any(feas):
                # Add subtle contour lines
                Z_contour = Z.copy()
                Z_contour[~feas] = np.nan
                levels = np.linspace(np.nanmin(Z_contour), np.nanmax(Z_contour), 8)
                ax.contour(X, Y, Z_contour, levels=levels, colors='darkred', 
                        alpha=0.3, linewidths=0.8, zdir='z', offset=np.nanmin(Z_contour)-0.05)

            # Small legend explaining elements (surface, halo)
            try:
                import matplotlib.patches as mpatches
                surface_color = cmap_beautiful_reds(0.75)
                halo_color = (1.0, 0.85, 0.3, 0.4)
                surface_patch = mpatches.Patch(facecolor=surface_color, edgecolor='darkred', label='Feasible surface')
                halo_patch = mpatches.Patch(facecolor=halo_color, edgecolor='goldenrod', label='Feasibility halo')
                ax.legend(handles=[surface_patch, halo_patch], loc='lower left',
                          bbox_to_anchor=(0.02, 0.02), frameon=True, framealpha=0.9,
                          fontsize=7 if a.paper else 8, borderpad=0.3, handlelength=1.2)
            except Exception:
                pass
            
            # Elegant annotation with compact styling to reduce margins
            # Always show feasibility statistics; in paper mode use a terse form.
            if not a.paper:
                textstr = (
                    f"Feasible: {feasible_count}/{visited_count} ({feas_frac:.1%})\n"
                    f"Coverage: {visited_count}/{N_total} ({visited_count / max(1, N_total):.1%})\n"
                    f"CP bound: {cp_est:.3f}"
                )
                ax.text2D(0.015, 0.985, textstr, transform=ax.transAxes, va='top', ha='left',
                          fontsize=9.5, weight='bold', color='darkred',
                          bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.08', facecolor='white',
                                    alpha=0.95, edgecolor='darkred', linewidth=1.0))
            else:
                # Terse, publication-friendly annotation
                terse = (
                    f"{feasible_count}/{visited_count} feas • {visited_count}/{N_total} cov"
                )
                ax.text2D(0.02, 0.98, terse, transform=ax.transAxes, va='top', ha='left',
                          fontsize=7.5, color='black')
            
            # Save with enhanced quality settings
            out_png = os.path.join(a.out_dir, f"spacefill_surface_{k}.png")
            out_pdf = os.path.join(a.out_dir, f"spacefill_surface_{k}.pdf")
            
            # Use minimal padding around the figure when not in paper mode.
            # In paper mode, avoid tight_layout and tight bbox due to inset_axes + tight bbox renderer issues.
            if not a.paper:
                plt.tight_layout(pad=0.2)
                bbox_opt = "tight"
            else:
                bbox_opt = None
            # Ensure renderer is initialized before saving
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass
            # High-quality PNG with anti-aliasing
            if bbox_opt:
                plt.savefig(out_png, bbox_inches=bbox_opt, dpi=300, facecolor='white',
                        edgecolor='none', transparent=False)
                # Vector PDF for publications
                plt.savefig(out_pdf, bbox_inches=bbox_opt, format='pdf', facecolor='white',
                        edgecolor='none', transparent=False)
            else:
                plt.savefig(out_png, dpi=300, facecolor='white', edgecolor='none', transparent=False)
                plt.savefig(out_pdf, format='pdf', facecolor='white', edgecolor='none', transparent=False)
            plt.close(fig)
            
            print(f"Saved surface space-filling {k} to {out_png} and {out_pdf}")
            
            # Reset matplotlib params
            plt.rcdefaults()


def main():
    # CLI
    parser = argparse.ArgumentParser(description="Surface landscapes: 3D feasible-only valleys (infeasible removed)")
    parser.add_argument("--seed", type=int, default=MIN_TESTING_DS_SEED)
    parser.add_argument("--host-count", type=int, default=2)
    parser.add_argument("--vm-count", type=int, default=3)
    parser.add_argument("--workflow-count", type=int, default=2)
    parser.add_argument("--gnp-min-n", type=int, default=6)
    parser.add_argument("--gnp-max-n", type=int, default=6)
    parser.add_argument("--dag-method", type=str, default="gnp", choices=["gnp", "linear"])
    parser.add_argument("--min-task-length", type=int, default=500)
    parser.add_argument("--max-task-length", type=int, default=5000)
    parser.add_argument("--min-cpu-speed", type=int, default=500)
    parser.add_argument("--max-cpu-speed", type=int, default=3000)
    parser.add_argument("--max-memory-gb", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out-dir", type=str, default="logs/landscape")
    parser.add_argument("--dpi", type=int, default=250)
    # Enumeration cap
    parser.add_argument("--sf-limit", type=int, default=None, help="Cap for enumerated sequences")
    parser.add_argument("--paper", action="store_true", help="Publication-ready: suppress titles/labels")
    # Fixed scale options for panels
    parser.add_argument("--fixed-vmin-makespan", type=float, default=None)
    parser.add_argument("--fixed-vmax-makespan", type=float, default=None)
    parser.add_argument("--fixed-vmin-energy", type=float, default=None)
    parser.add_argument("--fixed-vmax-energy", type=float, default=None)
    # Composition mode
    parser.add_argument("--compose-linear-gnp", action="store_true", help="Compose side-by-side linear vs GNP panels")
    parser.add_argument('--compose-linear-gnp-3d', action='store_true',
                    help='Generate side-by-side 3D surface plots for linear vs GNP DAGs with matched color scales')
    parser.add_argument('--single-3d-linear', action='store_true',
                    help='Generate individual 3D surface plot for linear DAG only')
    parser.add_argument('--single-3d-gnp', action='store_true',
                    help='Generate individual 3D surface plot for GNP DAG only')
    parser.add_argument("--metric", type=str, default="makespan", choices=["makespan", "energy"], help="Metric for composition panels")
    args = parser.parse_args()

    a = EnumArgs(
        seed=args.seed,
        vm_count=args.vm_count,
        workflow_count=args.workflow_count,
        gnp_min_n=args.gnp_min_n,
        gnp_max_n=args.gnp_max_n,
        dag_method=args.dag_method,
        min_task_length=args.min_task_length,
        max_task_length=args.max_task_length,
        min_cpu_speed=args.min_cpu_speed,
        max_cpu_speed=args.max_cpu_speed,
        max_memory_gb=args.max_memory_gb,
        limit_solutions=None,
        score="makespan",
        show_progress=(not args.no_progress),
        estimate_samples=0,
        out_dir=args.out_dir,
        cmap="Reds",
        dpi=args.dpi,
        paper=args.paper,
        fixed_vmin_makespan=args.fixed_vmin_makespan,
        fixed_vmax_makespan=args.fixed_vmax_makespan,
        fixed_vmin_energy=args.fixed_vmin_energy,
        fixed_vmax_energy=args.fixed_vmax_energy
    )
    env = build_env(a)
    env.reset(seed=a.seed)

    if args.compose_linear_gnp_3d:
        compose_linear_vs_gnp_3d(a, args.sf_limit, args.metric)
        return
    
    if args.single_3d_linear:
        single_3d_surface(a, args.sf_limit, args.metric, dag_type="linear")
        return
    
    if args.single_3d_gnp:
        single_3d_surface(a, args.sf_limit, args.metric, dag_type="gnp")
        return
    
    if args.compose_linear_gnp:
        # Compose side-by-side panels with matched scales
        compose_linear_vs_gnp_panels(a, sf_limit=args.sf_limit, metric=args.metric)
    else:
        # 3D surface landscapes
        plot_space_filling_surface(env, a, sf_limit=args.sf_limit)


if __name__ == "__main__":
    main()
