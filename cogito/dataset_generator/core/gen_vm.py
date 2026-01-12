import json

import numpy as np

from cogito.config.settings import HOST_SPECS_PATH
from cogito.dataset_generator.core.models import Host, Vm


# Generating Hosts
# ----------------------------------------------------------------------------------------------------------------------


def generate_hosts(n: int, rng: np.random.RandomState) -> list[Host]:
    """
    Generate a list of hosts with the specified number of hosts.
    Uses the host specifications from data/host_specs.json.
    """

    # Check for runtime override via environment variable
    host_specs_path = os.environ.get("HOST_SPECS_PATH", str(HOST_SPECS_PATH))
    with open(host_specs_path) as f:
        available_hosts: list = json.load(f)
    # Deterministic: sort by memory (desc) and pick sequentially
    available_hosts_sorted = sorted(available_hosts, key=lambda s: int(s.get("memory_gb", 0)), reverse=True)

    hosts: list[Host] = []
    for i in range(n):
        spec = available_hosts_sorted[i % len(available_hosts_sorted)]
        hosts.append(
            Host(
                id=i,
                cores=int(spec["cores"]),
                cpu_speed_mips=int(spec["cpu_speed_gips"] * 1e3),
                memory_mb=int(spec["memory_gb"] * 1024),
                disk_mb=int(spec["disk_tb"] * 1e6),
                bandwidth_mbps=int(spec["bandwidth_gbps"] * 1024),
                power_idle_watt=int(spec["power_idle_watt"]),
                power_peak_watt=int(spec["power_peak_watt"]),
            )
        )
    return hosts


# Generating and Allocating VMs
# ----------------------------------------------------------------------------------------------------------------------


def generate_vms(
    n: int, max_memory_gb: int, min_cpu_speed_mips: int, max_cpu_speed_mips: int, rng: np.random.RandomState
) -> list[Vm]:
    """
    Generate a list of VMs with the specified number of VMs.
    """

    vms = []
    for i in range(n):
        ram_mb = rng.randint(1, max_memory_gb + 1) * 1024
        cpu_speed = rng.randint(min_cpu_speed_mips, max_cpu_speed_mips + 1)
        host_id = -1  # Unallocated
        vms.append(Vm(i, host_id, cpu_speed, memory_mb=ram_mb, disk_mb=1024, bandwidth_mbps=50, vmm="Xen"))
    return vms


def allocate_vms(vms: list[Vm], hosts: list[Host], rng: np.random.RandomState):
    """
    Allocate VMs to hosts randomly.
    """
    if len(hosts) == 0:
        return

    # Deterministic round-robin VM -> host mapping
    for i, vm in enumerate(vms):
        host = hosts[i % len(hosts)]
        vm.host_id = host.id
        # Mirror host capacities on VM to match fixed-dataset behavior
        vm.cpu_cores = int(host.cores)
        vm.cpu_speed_mips = int(host.cpu_speed_mips)
        vm.memory_mb = int(host.memory_mb)
        vm.disk_mb = int(host.disk_mb)
        vm.bandwidth_mbps = int(host.bandwidth_mbps)
