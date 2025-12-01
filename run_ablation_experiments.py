import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Args:
    exp_name: str
    device: str
    num_envs: int
    total_timesteps: int
    update_epochs: int
    test_every_iters: int
    test_iterations: int
    dataset: dict

def _dc_replace(obj, **kwargs):
    return dataclasses.replace(obj, **kwargs)

def _clone_args_with_dataset(base: Args, dag_method: str, is_eval: bool = False) -> Args:
    new = _dc_replace(base)
    if is_eval:
        # Evaluation config (different from training)
        new.dataset = _dc_replace(
            base.dataset,
            workflow_count=8,  # Fewer workflows for faster eval
            dag_method=dag_method,
            gnp_min_n=15,  # More complex workflows
            gnp_max_n=30,
            host_count=6,  # More hosts
            vm_count=15,   # More VMs
        )
    else:
        # Training config (as specified by user)
        new.dataset = _dc_replace(
            base.dataset,
            workflow_count=10,
            dag_method=dag_method,
            gnp_min_n=base.dataset.gnp_min_n,
            gnp_max_n=base.dataset.gnp_max_n,
        )
    new.test_iterations = 5  # More test iterations for stable metrics
    return new

def run_experiment(dag_method: str, output_suffix: str):
    base_cmd = [
        "python", "-m", "scheduler.rl_model.ablation_gnn",
        "--exp_name", "gnn_ablation_full",
        "--device", "cuda:0",  # or "cpu" if no GPU
        "--num_envs", "4",
        "--total_timesteps", "1000000",
        "--update_epochs", "4",
        "--test_every_iters", "10",
        "--test_iterations", "5",
        "--dataset.gnp_min_n", "12",
        "--dataset.gnp_max_n", "24",
        "--dataset.workflow_count", "10",
        "--dataset.host_count", "4",
        "--dataset.vm_count", "10",
    ]
    cmd = base_cmd + [
        "--dataset.dag_method", dag_method,
        "--exp_name", f"gnn_ablation_{output_suffix}",
    ]
    print(f"\n{'='*80}\nRunning: {' '.join(cmd)}\n{'='*80}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Run GNP only
    run_experiment("gnp", "gnp_only")
    
    # Run Linear only
    run_experiment("linear", "linear_only")
    
    # Run Mixed (5 GNP + 5 Linear)
    # The ablation_gnn.py script already handles the mixed case automatically
    # by running both and combining metrics
    run_experiment("gnp", "mixed")  # The script will handle the mixed case
    print("\nAll experiments completed!")
