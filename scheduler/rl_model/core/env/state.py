from dataclasses import dataclass

from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.rl_model.core.utils.task_mapper import TaskMapper


@dataclass
class EnvState:
    static_state: "StaticState"
    task_states: list["TaskState"]
    vm_states: list["VmState"]
    task_dependencies: set[tuple[int, int]]
    current_time: float = 0.0


@dataclass
class VmState:
    assigned_task_id: int | None = None
    completion_time: float = 0
    used_memory_mb: int = 0
    used_cpu_cores: int = 0
    # active_tasks: list of (task_id, completion_time, mem_mb, cpu_cores)
    active_tasks: list[tuple[int, float, int, int]] | None = None
    # Sorted list of (time, delta_mem, delta_cores) for incremental capacity checks on this VM
    # Maintained in gym_env.step() using bisect insertions. None means uninitialized.
    active_events: list[tuple[float, int, int]] | None = None
    # Cached usage snapshot at a reference time (optional optimization; may be None if unused)
    used_mem_snapshot: int | None = None
    used_cores_snapshot: int | None = None
    snapshot_time: float | None = None


@dataclass
class TaskState:
    is_ready: bool = False
    assigned_vm_id: int | None = None
    start_time: float = 0
    completion_time: float = 0
    energy_consumption: float = 0


@dataclass
class StaticState:
    task_mapper: TaskMapper
    tasks: list[TaskDto]
    vms: list[VmDto]
    compatibilities: list[tuple[int, int]]
