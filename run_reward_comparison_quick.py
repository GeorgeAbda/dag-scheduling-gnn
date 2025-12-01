#!/usr/bin/env python3
"""
Quick version of reward comparison script with reduced training time for faster results.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def modify_wrapper_for_reward_config(config_name: str):
    """
    Modify the GinAgentWrapper to use different reward configurations.
    """
    wrapper_path = project_root / "scheduler/rl_model/agents/gin_agent/wrapper.py"
    
    # Read the current wrapper file
    with open(wrapper_path, 'r') as f:
        lines = f.readlines()
    
    # Find the reward calculation section
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if "eps = 1e-8" in line and start_line is None:
            start_line = i
        if "self.prev_obs = obs" in line and start_line is not None:
            end_line = i
            break
    
    if start_line is None or end_line is None:
        raise ValueError("Could not find reward calculation section")
    
    # Define the three reward configurations
    if config_name == "energy+makespan":
        reward_lines = [
            "        eps = 1e-8\n",
            "        makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / max(obs.makespan(), eps)\n",
            "        energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / max(obs.energy_consumption(), eps)\n",
            "        reward = self.energy_weight * energy_reward + self.makespan_weight * makespan_reward\n",
            "        if (terminated or truncated) and isinstance(info, dict) and (\"total_energy\" in info):\n",
            "            total_energy = float(info[\"total_energy\"])\n",
            "            if self.energy_ref is None:\n",
            "                self.energy_ref = max(total_energy, 1e-6)\n",
            "            else:\n",
            "                self.energy_ref = (\n",
            "                    self.energy_ref_alpha * self.energy_ref + (1.0 - self.energy_ref_alpha) * max(total_energy, 1e-6)\n",
            "                )\n",
            "            terminal_term = - total_energy / max(self.energy_ref, 1e-6)\n",
            "            reward += self.terminal_energy_weight * terminal_term\n"
        ]
    elif config_name == "energy":
        reward_lines = [
            "        eps = 1e-8\n",
            "        energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / max(obs.energy_consumption(), eps)\n",
            "        reward = self.energy_weight * energy_reward\n",
            "        if (terminated or truncated) and isinstance(info, dict) and (\"total_energy\" in info):\n",
            "            total_energy = float(info[\"total_energy\"])\n",
            "            if self.energy_ref is None:\n",
            "                self.energy_ref = max(total_energy, 1e-6)\n",
            "            else:\n",
            "                self.energy_ref = (\n",
            "                    self.energy_ref_alpha * self.energy_ref + (1.0 - self.energy_ref_alpha) * max(total_energy, 1e-6)\n",
            "                )\n",
            "            terminal_term = - total_energy / max(self.energy_ref, 1e-6)\n",
            "            reward += self.terminal_energy_weight * terminal_term\n"
        ]
    elif config_name == "makespan":
        reward_lines = [
            "        eps = 1e-8\n",
            "        makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / max(obs.makespan(), eps)\n",
            "        reward = self.makespan_weight * makespan_reward\n"
        ]
    else:
        raise ValueError(f"Unknown reward config: {config_name}")
    
    # Replace the reward calculation section
    new_lines = lines[:start_line] + reward_lines + lines[end_line:]
    
    # Write back to file
    with open(wrapper_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Modified wrapper for {config_name} reward configuration")

def run_training(config_name: str, exp_name: str):
    """
    Run training with the specified configuration.
    """
    print(f"\n{'='*60}")
    print(f"Running training for {config_name} reward configuration")
    print(f"{'='*60}")
    
    # Modify the wrapper for this reward configuration
    modify_wrapper_for_reward_config(config_name)
    
    # Prepare training command with reduced timesteps for quick demo
    cmd = [
        sys.executable, 
        str(project_root / "scheduler/rl_model/train.py"),
        "--exp_name", exp_name,
        "--dataset.dag_method", "linear",
        "--dataset.gnp_min_n", "80",
        "--dataset.gnp_max_n", "120", 
        "--dataset.workflow_count", "10",
        "--dataset.host_count", "1",
        "--dataset.vm_count", "1",
        "--dataset.min_task_length", "150000",
        "--dataset.max_task_length", "400000",
        "--dataset.task_arrival", "static",
        "--num_envs", "1",
        "--env_mode", "sync",
        "--total_timesteps", "4000",
        "--test_every_iters", "1",
        "--csv_reward_tag", config_name,
        "--csv_dir", str(project_root / "csv")
    ]
    
    # Run the training and stream logs
    print(f"[runner] Launch: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(f"[{config_name}] {line}", end="")
    finally:
        proc.wait()

    if proc.returncode != 0:
        print(f"Training failed for {config_name} (exit {proc.returncode})")
        return None
    else:
        print(f"Training completed successfully for {config_name}")
        return "OK"

def extract_metrics_to_csv(logs_dir: Path, config_name: str, csv_dir: Path):
    """
    Extract metrics from tensorboard logs and save to CSV files.
    """
    print(f"Extracting metrics for {config_name}...")
    
    # Find the most recent log directory for this configuration
    pattern = f"*{config_name}*"
    matching_dirs = list(logs_dir.glob(pattern))
    if not matching_dirs:
        print(f"No log directories found for pattern {pattern}")
        return
    
    # Get the most recent directory
    log_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Using log directory: {log_dir}")
    
    # Use tensorboard to extract data
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Extract metrics we're interested in
        metrics_to_extract = {
            'tests/total_energy': 'total_energy',
            'tests/active_energy': 'active_energy', 
            'tests/idle_energy': 'idle_energy',
            'tests/makespan': 'makespan'
        }
        
        csv_dir.mkdir(exist_ok=True)
        
        for tb_tag, metric_name in metrics_to_extract.items():
            if tb_tag in ea.Tags()['scalars']:
                scalar_events = ea.Scalars(tb_tag)
                
                # Create CSV data
                csv_data = []
                for event in scalar_events:
                    csv_data.append({
                        'Wall time': event.wall_time,
                        'Step': event.step,
                        'Value': event.value
                    })
                
                if csv_data:
                    import pandas as pd
                    df = pd.DataFrame(csv_data)
                    csv_file = csv_dir / f"{metric_name}_{config_name}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"Saved {len(csv_data)} data points to {csv_file}")
                else:
                    print(f"No data found for {tb_tag}")
            else:
                print(f"Tag {tb_tag} not found in tensorboard logs")
                
    except ImportError:
        print("tensorboard not available for metric extraction")
        print("You can manually extract metrics from the tensorboard logs")

def generate_comparison_plots(csv_dir: Path):
    """
    Generate comparison plots using the plot_metrics_grid script.
    """
    print(f"\n{'='*60}")
    print("Generating comparison plots")
    print(f"{'='*60}")
    
    plot_script = project_root / "scheduler/viz_results/plot_metrics_grid.py"
    output_file = csv_dir / "reward_comparison_grid_quick.png"
    
    cmd = [
        sys.executable,
        str(plot_script),
        "--csv-dir", str(csv_dir),
        "--out", str(output_file),
        "--window", "3",
        "--xaxis", "Episode"
    ]
    
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Plot generation failed:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    else:
        print(f"Comparison plots saved to: {output_file}")

def main():
    """
    Main function to run the quick reward comparison experiment.
    """
    print("Starting QUICK reward configuration comparison experiment")
    print("This will train models with 3 different reward configurations (reduced timesteps):")
    print("1. Energy + Makespan")
    print("2. Energy only") 
    print("3. Makespan only")
    
    # Create directories
    logs_dir = project_root / "logs"
    csv_dir = project_root / "csv"
    csv_dir.mkdir(exist_ok=True)
    
    # Backup original wrapper
    wrapper_path = project_root / "scheduler/rl_model/agents/gin_agent/wrapper.py"
    backup_path = wrapper_path.with_suffix('.py.backup_quick')
    shutil.copy2(wrapper_path, backup_path)
    print(f"Backed up original wrapper to {backup_path}")
    
    try:
        # Run training for each reward configuration
        configs = [
            ("energy+makespan", "quick_reward_both"),
            ("energy", "quick_reward_energy"), 
            ("makespan", "quick_reward_makespan")
        ]
        
        for config_name, exp_name in configs:
            run_training(config_name, exp_name)
            extract_metrics_to_csv(logs_dir, exp_name, csv_dir)
            time.sleep(1)
        
        # Generate comparison plots
        generate_comparison_plots(csv_dir)
        
        print(f"\n{'='*60}")
        print("Quick experiment completed!")
        print(f"Results saved in: {csv_dir}")
        print(f"Comparison plots: {csv_dir}/reward_comparison_grid_quick.png")
        print(f"{'='*60}")
        
    finally:
        # Restore original wrapper
        shutil.copy2(backup_path, wrapper_path)
        print(f"Restored original wrapper from backup")
        backup_path.unlink()

if __name__ == "__main__":
    main()
