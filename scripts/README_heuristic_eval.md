# Heuristic Evaluation Script

## Overview

`eval_heuristics_on_seeds.py` evaluates two scheduling heuristics on initial observations:

1. **Makespan Heuristic**: Greedy earliest-completion-time scheduling
2. **Energy Heuristic**: Greedy minimum-energy-rate scheduling

For each seed in the config, the script:
- Creates an environment with the initial observation
- Runs both heuristics to completion
- Computes final makespan and active energy for each solution
- Saves results to CSV and generates a LaTeX table with mean values

## Usage

### Basic Usage

```bash
python scripts/eval_heuristics_on_seeds.py \
    --config-path data/rl_configs/train_long_cp_p08_seeds.json \
    --host-specs-path data/host_specs.json \
    --out-csv logs/heuristic_eval_results.csv \
    --out-tex logs/heuristic_eval_table.tex
```

### Arguments

- `--config-path`: Path to the seed config JSON (default: `data/rl_configs/train_long_cp_p08_seeds.json`)
- `--host-specs-path`: Path to host specs JSON (default: `data/host_specs.json`)
- `--out-csv`: Output CSV file path (default: `logs/heuristic_eval_results.csv`)
- `--out-tex`: Output LaTeX table file path (default: `logs/heuristic_eval_table.tex`)
- `--dataset-req-divisor`: Optional override for req_divisor (default: auto-computed)

### Example with Different Config

```bash
python scripts/eval_heuristics_on_seeds.py \
    --config-path data/rl_configs/train_wide_p005_seeds.json \
    --host-specs-path data/host_specs.json \
    --out-csv logs/heuristic_eval_wide.csv \
    --out-tex logs/heuristic_eval_wide_table.tex
```

## Output

### CSV File

The CSV file contains per-seed results:

```
seed,mk_heuristic_makespan,mk_heuristic_active_energy,en_heuristic_makespan,en_heuristic_active_energy
100001,1234.56,5678.90,1345.67,5234.12
100002,1256.78,5712.34,1367.89,5289.45
...
```

### LaTeX Table

The LaTeX table contains mean values across all seeds:

```latex
\begin{table}[h]
\centering
\caption{Heuristic Performance on Initial Observations}
\label{tab:heuristic_eval}
\begin{tabular}{lcc}
\hline
\textbf{Heuristic} & \textbf{Makespan} & \textbf{Active Energy} \\
\hline
Makespan Heuristic & 1245.67 & 5645.62 \\
Energy Heuristic & 1356.78 & 5261.78 \\
\hline
\end{tabular}
\end{table}
```

## How It Works

1. **Load Configuration**: Reads the seed config and extracts training seeds and dataset parameters
2. **For Each Seed**:
   - Creates a fresh environment with the dataset
   - **Makespan Heuristic**: At each step, picks the (task, VM) pair with earliest completion time
   - **Energy Heuristic**: At each step, picks the (task, VM) pair with minimum energy rate
   - Records final makespan and active energy from the environment's metrics
3. **Aggregate Results**: Computes mean values across all seeds
4. **Save Outputs**: Writes CSV with per-seed results and LaTeX table with means

## Notes

- The script uses the same environment and metric computation as `eval_hetero_agents_over_seed_configs.py`
- Both heuristics respect task dependencies and VM compatibility constraints
- Active energy is computed from the environment's detailed energy tracking
- The script is deterministic: running with the same config produces identical results
