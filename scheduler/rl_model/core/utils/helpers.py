from scheduler.rl_model.core.env.observation import VmObservation
from scheduler.rl_model.core.types import VmDto, TaskDto


def is_suitable(vm: VmDto, task: TaskDto):
    """Check if the VM is suitable for the task."""
    # print(f'Checking suitability of VM {vm.id} for Task {task.id}')
    # print(f'VM Memory: {vm.memory_mb} MB, Task Memory Requirement: {task.req_memory_mb} MB')
    # print(f'VM CPU Cores: {vm.cpu_cores}, Task CPU Requirement: {task.req_cpu_cores} cores')
    # print(f'VM Memory: {vm.memory_mb} MB, Task Memory Requirement: {task.req_memory_mb} MB')
    # print(f'VM CPU Cores: {vm.cpu_cores}, Task CPU Requirement: {task.req_cpu_cores} cores')
    return (vm.memory_mb >= task.req_memory_mb) and (vm.cpu_cores >= task.req_cpu_cores)



def active_energy_consumption_per_mi(vm: VmDto | VmObservation, cpu_fraction: float = 1.0):
    """
     Return energy per MI (Joules per million instructions) consumed at a given CPU utilization.

     We model instantaneous power as:
         P(cpu_fraction) = idle + (peak - idle) * cpu_fraction   [Watts]

     Converting to energy-per-MI so that energy can be computed as:
         energy = rate_per_mi * (vm.cpu_speed_mips * dt) = P * dt

     This keeps unit consistency across code paths that multiply this rate by MI processed.
     """
    # print(f'cpu_fraction = {cpu_fraction}')
    # print(f'vm.host_power_idle_watt = {vm.host_power_idle_watt}')

    # Instantaneous power in Watts at the given CPU fraction
    power_watt = vm.host_power_idle_watt + (vm.host_power_peak_watt - vm.host_power_idle_watt) * cpu_fraction
    # Convert to energy per MI by dividing by the VM's execution rate (MI per second)
    vm_mips = getattr(vm, "cpu_speed_mips", None)
    if vm_mips is None or vm_mips <= 0:
        # Fallback: avoid division by zero; treat as zero additional energy-per-MI
        return 0.0
    return power_watt / vm_mips
