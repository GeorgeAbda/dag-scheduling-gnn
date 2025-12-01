import numpy as np

from scheduler.dataset_generator.core.gen_workflow import generate_workflows
from scheduler.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from scheduler.dataset_generator.core.models import Dataset


def generate_dataset(
    seed: int,
    host_count: int,
    vm_count: int,
    max_memory_gb: int,
    min_cpu_speed_mips: int,
    max_cpu_speed_mips: int,
    workflow_count: int,
    dag_method: str,
    gnp_min_n: int,
    gnp_max_n: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    task_arrival: str,
    arrival_rate: float,
    vm_rng_seed: int | None = 0,
    gnp_p: float | tuple[float, float] | None = None,
    req_divisor: int = 20,
) -> Dataset:
    """
    Generate a dataset.
    """

    rng = np.random.RandomState(seed)
    vm_rng = rng
    if vm_rng_seed is not None:
        vm_rng = np.random.RandomState(vm_rng_seed)

    hosts = generate_hosts(host_count, vm_rng)
    vms = generate_vms(vm_count, max_memory_gb, min_cpu_speed_mips, max_cpu_speed_mips, vm_rng)
    allocate_vms(vms, hosts, vm_rng)

    workflows = generate_workflows(
        workflow_count=workflow_count,
        dag_method=dag_method,
        gnp_min_n=gnp_min_n,
        gnp_max_n=gnp_max_n,
        task_length_dist=task_length_dist,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        # Make sure that the problem is feasible
        max_req_memory_mb=max(1024, max(vm.memory_mb for vm in vms)//req_divisor),
        max_req_cpu_cores=max(1, max(vm.cpu_cores for vm in vms)//req_divisor),
        task_arrival=task_arrival,
        arrival_rate=arrival_rate,
        rng=rng,
        gnp_p=gnp_p,
    )

    return Dataset(workflows=workflows, vms=vms, hosts=hosts)


def _topological_layers(children: dict[int, set[int]]) -> list[list[int]]:
    n = len(children)
    indeg = {u: 0 for u in children}
    for u, vs in children.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    frontier = [u for u in children if indeg[u] == 0]
    layers: list[list[int]] = []
    seen: set[int] = set()
    while frontier:
        cur = list(frontier)
        layers.append(cur)
        frontier = []
        for u in cur:
            if u in seen:
                continue
            seen.add(u)
            for v in children.get(u, ()): 
                indeg[v] -= 1
                if indeg[v] == 0:
                    frontier.append(v)
    if not layers:
        return [list(children.keys())]
    return layers


def _enforce_queue_free(
    workflows, vms, rng: np.random.RandomState, alpha_low: float = 0.8, alpha_high: float = 0.95, beta: float = 0.95
):
    mem_total = int(sum(vm.memory_mb for vm in vms))
    cores_total = int(sum(max(1, vm.cpu_cores) for vm in vms))
    max_vm_mem = int(max(vm.memory_mb for vm in vms))
    max_vm_cores = int(max(max(1, vm.cpu_cores) for vm in vms))
    # Precompute VM capacity pairs (cores, mem)
    vm_caps = [(int(max(1, vm.cpu_cores)), int(vm.memory_mb)) for vm in vms]
    for wf in workflows:
        graph: dict[int, set[int]] = {t.id: set(t.child_ids) for t in wf.tasks}
        layers = _topological_layers(graph)
        W = max(len(L) for L in layers) if layers else 1
        if W <= 0:
            W = 1
        alpha = float(rng.uniform(alpha_low, alpha_high))
        tgt_mean_mem = max(1024.0, alpha * mem_total / float(W))
        tgt_mean_cpu = max(1.0, alpha * cores_total / float(W))
        cur_mean_mem = max(1.0, float(np.mean([t.req_memory_mb for t in wf.tasks])))
        cur_mean_cpu = max(1.0, float(np.mean([t.req_cpu_cores for t in wf.tasks])))
        s_mem = float(tgt_mean_mem / cur_mean_mem)
        s_cpu = float(tgt_mean_cpu / cur_mean_cpu)
        # Helper: fit scaled (cpu, mem) jointly to one VM capacity vector to ensure feasibility
        def fit_to_vm(req_cpu: float, req_mem: float) -> tuple[int, int]:
            # Scale to integers and 1024MB blocks after selecting best VM
            best_idx = -1
            best_scale = -1.0
            # Evaluate downscale needed per VM; choose VM that requires minimal downscale (max scale)
            for (cores_cap, mem_cap) in vm_caps:
                # Feasible scale factor along both dims
                s1 = float(mem_cap) / max(1.0, float(req_mem))
                s2 = float(cores_cap) / max(1.0, float(req_cpu))
                scale = min(s1, s2)
                if scale > best_scale:
                    best_scale = scale
            if best_scale <= 0.0:
                best_scale = 1.0
            # Apply scale if needed (only downscale when scale<1)
            eff_cpu = float(req_cpu) * min(1.0, best_scale)
            eff_mem = float(req_mem) * min(1.0, best_scale)
            # Snap to capacities: cap by per-dimension maxima as safety
            eff_cpu = min(eff_cpu, float(max_vm_cores))
            eff_mem = min(eff_mem, float(max_vm_mem))
            cpu_int = int(max(1, round(eff_cpu)))
            mem_int = int(max(1024, round(eff_mem / 1024.0) * 1024))
            # Final guard: ensure at least one VM satisfies (cpu_int, mem_int)
            ok = any((cpu_int <= c and mem_int <= m) for (c, m) in vm_caps)
            if not ok:
                # If still infeasible due to rounding, reduce minimally
                # Try decreasing memory blocks until feasible, then CPU
                mem_tmp = mem_int
                cpu_tmp = cpu_int
                tried = 0
                while not any((cpu_tmp <= c and mem_tmp <= m) for (c, m) in vm_caps) and tried < 64:
                    if mem_tmp > 1024:
                        mem_tmp = int(max(1024, mem_tmp - 1024))
                    elif cpu_tmp > 1:
                        cpu_tmp = int(max(1, cpu_tmp - 1))
                    else:
                        break
                    tried += 1
                cpu_int, mem_int = cpu_tmp, mem_tmp
            return cpu_int, mem_int

        # First pass: mean-level scaling then joint fit per task
        for t in wf.tasks:
            scaled_mem = float(t.req_memory_mb) * s_mem
            scaled_cpu = float(t.req_cpu_cores) * s_cpu
            new_cpu, new_mem = fit_to_vm(scaled_cpu, scaled_mem)
            t.req_memory_mb = int(new_mem)
            t.req_cpu_cores = int(new_cpu)

        # Second pass: prefix IFR check on peak layer and global downscale if needed
        # Identify peak layer
        L_peak = max(layers, key=len) if layers else []
        if L_peak:
            mem_reqs = sorted([next(t.req_memory_mb for t in wf.tasks if t.id == u) for u in L_peak], reverse=True)
            cpu_reqs = sorted([next(t.req_cpu_cores for t in wf.tasks if t.id == u) for u in L_peak], reverse=True)
            vm_mems = sorted([int(vm.memory_mb) for vm in vms], reverse=True)
            vm_cores = sorted([int(max(1, vm.cpu_cores)) for vm in vms], reverse=True)
            k_mem = min(len(mem_reqs), len(vm_mems))
            k_cpu = min(len(cpu_reqs), len(vm_cores))
            if k_mem > 0:
                S = np.cumsum(mem_reqs[:k_mem]).astype(float)
                P = np.cumsum(vm_mems[:k_mem]).astype(float)
                IFR_mem = float(np.max(S / np.maximum(P, 1e-9)))
            else:
                IFR_mem = 0.0
            if k_cpu > 0:
                Sc = np.cumsum(cpu_reqs[:k_cpu]).astype(float)
                Pc = np.cumsum(vm_cores[:k_cpu]).astype(float)
                IFR_cpu = float(np.max(Sc / np.maximum(Pc, 1e-9)))
            else:
                IFR_cpu = 0.0
            IFR = max(IFR_mem, IFR_cpu)
            if IFR > beta:
                down = float(beta / IFR)
                for t in wf.tasks:
                    # Downscale jointly and refit to VM
                    new_cpu, new_mem = fit_to_vm(float(t.req_cpu_cores) * down, float(t.req_memory_mb) * down)
                    t.req_memory_mb = int(new_mem)
                    t.req_cpu_cores = int(new_cpu)


def _peak_layer_metrics(wf, vms):
    graph: dict[int, set[int]] = {t.id: set(t.child_ids) for t in wf.tasks}
    layers = _topological_layers(graph)
    L_peak = max(layers, key=len) if layers else []
    if not L_peak:
        return {
            "width": 0,
            "ifr_mem": 0.0,
            "ifr_cpu": 0.0,
            "ifr": 0.0,
            "match_size": 0,
            "layer_size": 0,
            "alpha_mem": 0.0,
            "alpha_cpu": 0.0,
        }
    mem_reqs = sorted([next(t.req_memory_mb for t in wf.tasks if t.id == u) for u in L_peak], reverse=True)
    cpu_reqs = sorted([next(t.req_cpu_cores for t in wf.tasks if t.id == u) for u in L_peak], reverse=True)
    vm_mems = sorted([int(v.memory_mb) for v in vms], reverse=True)
    vm_cores = sorted([int(max(1, v.cpu_cores)) for v in vms], reverse=True)
    k = min(len(L_peak), len(vm_mems), len(vm_cores))
    if k <= 0:
        k = len(L_peak)
    S_mem = np.cumsum(mem_reqs[:k]).astype(float) if k > 0 else np.array([0.0])
    P_mem = np.cumsum(vm_mems[:k]).astype(float) if k > 0 else np.array([1.0])
    S_cpu = np.cumsum(cpu_reqs[:k]).astype(float) if k > 0 else np.array([0.0])
    P_cpu = np.cumsum(vm_cores[:k]).astype(float) if k > 0 else np.array([1.0])
    ifr_mem = float(np.max(S_mem / np.maximum(P_mem, 1e-9))) if k > 0 else 0.0
    ifr_cpu = float(np.max(S_cpu / np.maximum(P_cpu, 1e-9))) if k > 0 else 0.0
    ifr = max(ifr_mem, ifr_cpu)

    topk_vm_mems = P_mem[-1] / max(1, k) if k > 0 else 1.0
    topk_vm_cores = P_cpu[-1] / max(1, k) if k > 0 else 1.0
    alpha_mem = float(np.mean(mem_reqs[:k])) / max(1.0, float(topk_vm_mems)) if k > 0 else 0.0
    alpha_cpu = float(np.mean(cpu_reqs[:k])) / max(1.0, float(topk_vm_cores)) if k > 0 else 0.0

    # Multi-tenant greedy packing on peak layer: pack tasks across VMs s.t. per-VM (mem, cores) capacities are respected
    tasks = [next(t for t in wf.tasks if t.id == u) for u in L_peak]
    vm_caps = [(int(max(1, vm.cpu_cores)), int(vm.memory_mb)) for vm in vms]
    # Try multiple orderings to reduce false negatives
    def try_pack(order: list[int]) -> tuple[int, bool, float, float]:
        # residual capacities
        cap_cores = [c for (c, _m) in vm_caps]
        cap_mem = [m for (_c, m) in vm_caps]
        res_cores = cap_cores.copy()
        res_mem = cap_mem.copy()
        used = [False] * len(vm_caps)
        placed = 0
        for idx in order:
            t = tasks[idx]
            rc = int(max(1, t.req_cpu_cores))
            rm = int(max(1, t.req_memory_mb))
            best_vm = -1
            best_slack = None
            for j in range(len(vm_caps)):
                if rc <= res_cores[j] and rm <= res_mem[j]:
                    # favor tighter fit to avoid saturating VMs
                    slack = (res_cores[j] - rc) / max(1.0, float(cap_cores[j])) + (res_mem[j] - rm) / max(1.0, float(cap_mem[j]))
                    if best_slack is None or slack < best_slack:
                        best_slack = slack
                        best_vm = j
            if best_vm >= 0:
                res_cores[best_vm] -= rc
                res_mem[best_vm] -= rm
                used[best_vm] = True
                placed += 1
            else:
                continue
        # Compute strict slack ratios across used VMs (min across used)
        used_idxs = [j for j, u in enumerate(used) if u]
        if used_idxs:
            min_slack_core = min((res_cores[j] / max(1.0, float(cap_cores[j]))) for j in used_idxs)
            min_slack_mem = min((res_mem[j] / max(1.0, float(cap_mem[j]))) for j in used_idxs)
        else:
            min_slack_core = 1.0
            min_slack_mem = 1.0
        # sat_free means no VM is exactly saturated in either dimension when used (strict positive slack)
        sat_free = all((res_cores[j] > 0 and res_mem[j] > 0) for j in used_idxs) if used_idxs else True
        return placed, sat_free, float(min_slack_mem), float(min_slack_core)

    idxs = list(range(len(tasks)))
    # Sort keys
    by_mem = sorted(idxs, key=lambda i: int(tasks[i].req_memory_mb), reverse=True)
    by_cpu = sorted(idxs, key=lambda i: int(tasks[i].req_cpu_cores), reverse=True)
    # Combined pressure ratio w.r.t. max VM
    maxC = float(max((c for (c, _m) in vm_caps), default=1))
    maxM = float(max((m for (_c, m) in vm_caps), default=1))
    by_ratio = sorted(idxs, key=lambda i: max(float(tasks[i].req_cpu_cores)/max(1.0,maxC), float(tasks[i].req_memory_mb)/max(1.0,maxM)), reverse=True)
    cand = [try_pack(by_mem), try_pack(by_cpu), try_pack(by_ratio)]
    # pick best candidate: prioritize placing all with sat_free True; then max placed; then best slack
    def key_fn(t):
        placed, sat_free_i, s_mem, s_core = t
        return (
            int(placed == len(tasks) and sat_free_i),  # 1 if perfect & sat-free
            placed,                                     # placed count
            min(s_mem, s_core),                         # minimal slack ratio
        )
    best = max(cand, key=key_fn) if cand else (0, True, 1.0, 1.0)
    pack_size = int(best[0])
    sat_free_peak = bool(best[1]) and (pack_size == len(tasks))
    min_slack_mem = float(best[2])
    min_slack_core = float(best[3])

    return {
        "width": len(L_peak),
        "ifr_mem": ifr_mem,
        "ifr_cpu": ifr_cpu,
        "ifr": ifr,
        "pack_size": pack_size,
        "layer_size": len(L_peak),
        "alpha_mem": alpha_mem,
        "alpha_cpu": alpha_cpu,
        "sat_free": sat_free_peak,
        "min_slack_mem_ratio": min_slack_mem,
        "min_slack_core_ratio": min_slack_core,
    }


def verify_queue_free(workflows, vms, tol: float = 0.0) -> dict:
    res = []
    ok_all = True
    max_ifr = 0.0
    max_alpha = 0.0
    for wf in workflows:
        m = _peak_layer_metrics(wf, vms)
        ratio = float(m.get("pack_size", 0)) / max(1, m["layer_size"]) if m["layer_size"] > 0 else 1.0
        ok = (ratio >= 1.0 - 1e-9) and bool(m.get("sat_free", False))
        ok_all = ok_all and ok
        max_ifr = max(max_ifr, float(m["ifr"]))
        max_alpha = max(max_alpha, float(max(m["alpha_mem"], m["alpha_cpu"])) )
        res.append({"workflow_id": wf.id, **m, "ok": ok})
    return {"queue_free": ok_all, "max_ifr": max_ifr, "alpha_empirical": max_alpha, "details": res}


def classify_queue_regime(workflows, vms, alpha_divisor: float | None = None, tol: float = 0.05) -> tuple[str, dict]:
    v = verify_queue_free(workflows, vms, tol=tol)
    labels = []
    for d in v["details"]:
        ratio = float(d.get("pack_size", 0)) / max(1, d["layer_size"]) if d["layer_size"] > 0 else 1.0
        if ratio >= 1.0 - 1e-9 and bool(d.get("sat_free", False)):
            labels.append("queue_free")
        elif ratio >= 1.0 - 1e-9:
            labels.append("queue")
        else:
            labels.append("queue_high")
    # Dataset-level label as the worst-case
    order = {"queue_free": 0, "queue": 1, "queue_high": 2}
    label = max(labels, key=lambda x: order[x]) if labels else "queue_free"
    info = {"per_workflow": labels, **v}
    if alpha_divisor is not None and alpha_divisor > 0:
        info["alpha_divisor"] = float(alpha_divisor)
        info["alpha_divisor_inverse"] = 1.0 / float(alpha_divisor)
    return label, info


def generate_dataset_long_cp_queue_free(
    seed: int,
    host_count: int,
    vm_count: int,
    max_memory_gb: int,
    min_cpu_speed_mips: int,
    max_cpu_speed_mips: int,
    workflow_count: int,
    gnp_min_n: int,
    gnp_max_n: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    task_arrival: str,
    arrival_rate: float,
    vm_rng_seed: int | None = 0,
    p_range: tuple[float, float] = (0.70, 0.95),
    alpha_range: tuple[float, float] = (0.8, 0.95),
) -> Dataset:
    rng = np.random.RandomState(seed)
    vm_rng = rng if vm_rng_seed is None else np.random.RandomState(vm_rng_seed)
    hosts = generate_hosts(host_count, vm_rng)
    vms = generate_vms(vm_count, max_memory_gb, min_cpu_speed_mips, max_cpu_speed_mips, vm_rng)
    allocate_vms(vms, hosts, vm_rng)
    gnp_p = (float(p_range[0]), float(p_range[1]))
    workflows = generate_workflows(
        workflow_count=workflow_count,
        dag_method="gnp",
        gnp_min_n=gnp_min_n,
        gnp_max_n=gnp_max_n,
        task_length_dist=task_length_dist,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        max_req_memory_mb=max(vm.memory_mb for vm in vms),
        max_req_cpu_cores=max(vm.cpu_cores for vm in vms),
        task_arrival=task_arrival,
        arrival_rate=arrival_rate,
        rng=rng,
        gnp_p=gnp_p,
    )
    _enforce_queue_free(workflows, vms, rng, alpha_low=alpha_range[0], alpha_high=alpha_range[1])
    return Dataset(workflows=workflows, vms=vms, hosts=hosts)


def generate_dataset_wide_queue_free(
    seed: int,
    host_count: int,
    vm_count: int,
    max_memory_gb: int,
    min_cpu_speed_mips: int,
    max_cpu_speed_mips: int,
    workflow_count: int,
    gnp_min_n: int,
    gnp_max_n: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    task_arrival: str,
    arrival_rate: float,
    vm_rng_seed: int | None = 0,
    p_range: tuple[float, float] = (0.02, 0.20),
    alpha_range: tuple[float, float] = (0.8, 0.95),
) -> Dataset:
    rng = np.random.RandomState(seed)
    vm_rng = rng if vm_rng_seed is None else np.random.RandomState(vm_rng_seed)
    hosts = generate_hosts(host_count, vm_rng)
    vms = generate_vms(vm_count, max_memory_gb, min_cpu_speed_mips, max_cpu_speed_mips, vm_rng)
    allocate_vms(vms, hosts, vm_rng)
    gnp_p = (float(p_range[0]), float(p_range[1]))
    workflows = generate_workflows(
        workflow_count=workflow_count,
        dag_method="gnp",
        gnp_min_n=gnp_min_n,
        gnp_max_n=gnp_max_n,
        task_length_dist=task_length_dist,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        max_req_memory_mb=max(vm.memory_mb for vm in vms),
        max_req_cpu_cores=max(vm.cpu_cores for vm in vms),
        task_arrival=task_arrival,
        arrival_rate=arrival_rate,
        rng=rng,
        gnp_p=gnp_p,
    )
    _enforce_queue_free(workflows, vms, rng, alpha_low=alpha_range[0], alpha_high=alpha_range[1])
    return Dataset(workflows=workflows, vms=vms, hosts=hosts)
