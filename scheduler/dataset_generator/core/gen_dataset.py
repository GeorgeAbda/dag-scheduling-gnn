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
        max_req_memory_mb=max(vm.memory_mb for vm in vms),
        max_req_cpu_cores=max(vm.cpu_cores for vm in vms),
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
