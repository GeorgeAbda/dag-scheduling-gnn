#!/usr/bin/env python3
"""
Merge heterogeneous agent evaluation results with heuristic results
and generate a comprehensive LaTeX table.
"""

import pandas as pd
import os
from pathlib import Path


def load_rl_results(case_label: str, base_dir: str) -> pd.DataFrame:
    """Load RL agent results for a given case."""
    summary_path = os.path.join(base_dir, f"{case_label}_hetero_eval.summary.csv")
    df = pd.read_csv(summary_path)
    return df


def load_heuristic_results(config_label: str, case_label: str, base_dir: str) -> dict:
    """Load heuristic results for a given config and case."""
    csv_path = os.path.join(base_dir, f"heuristic_eval_{config_label}_{case_label}.csv")
    df = pd.read_csv(csv_path)
    
    # Compute means
    mk_makespan = df['mk_heuristic_makespan'].mean()
    mk_energy = df['mk_heuristic_active_energy'].mean()
    en_makespan = df['en_heuristic_makespan'].mean()
    en_energy = df['en_heuristic_active_energy'].mean()
    
    return {
        'mk_makespan': mk_makespan,
        'mk_energy': mk_energy,
        'en_makespan': en_makespan,
        'en_energy': en_energy
    }


def format_value(val: float, is_best: bool, scale: float = 1.0) -> str:
    """Format a value with optional bold for best."""
    scaled = val / scale
    formatted = f"{scaled:.2f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(rl_dir: str, heuristic_dir: str, output_path: str):
    """Generate the complete LaTeX table merging RL and heuristic results."""
    
    # Case configurations
    cases = [
        ('HS', 'HS'),
        ('HP', 'HP'),
        ('AL', 'AL'),
        ('NAL', 'NA')  # RL uses NAL, display as NA
    ]
    
    # Energy scaling factor (to convert to millions)
    energy_scale = 1e7
    
    lines = []
    lines.append("\\begin{table*}[htp]")
    lines.append("\\caption{Cross-domain evaluation of heterogeneous agents and heuristics across host configurations.")
    lines.append("         Best results for each evaluation domain within a host configuration are highlighted in bold.}")
    lines.append("\\label{tab:hetero-all-hosts}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{} lll rr @{}}")
    lines.append("\\toprule")
    lines.append("Host cfg & Method & Eval Domain & Makespan & \\makecell{Active\\\\Energy (10$^7$ J)} \\\\")
    lines.append("\\midrule")
    lines.append("")
    
    for case_rl, case_display in cases:
        lines.append(f"% ---------- {case_display} configuration ----------")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\itshape {case_display} host configuration}} \\\\")
        
        # Load RL results
        rl_df = load_rl_results(case_rl, rl_dir)
        
        # Load heuristic results for both configs
        heur_longcp = load_heuristic_results('longcp', case_display, heuristic_dir)
        heur_wide = load_heuristic_results('wide', case_display, heuristic_dir)
        
        # Process long_cp eval domain
        longcp_eval = rl_df[rl_df['eval_domain'] == 'long_cp'].copy()
        
        # Collect all methods for long_cp eval
        methods_longcp = []
        
        # long_cp agent on long_cp
        row = longcp_eval[longcp_eval['agent_train_domain'] == 'long_cp'].iloc[0]
        methods_longcp.append({
            'method': 'long\\_cp',
            'makespan': row['mean_makespan'],
            'energy': row['mean_energy_active'],
            'color': 'blue!8'
        })
        
        # wide agent on long_cp
        row = longcp_eval[longcp_eval['agent_train_domain'] == 'wide'].iloc[0]
        methods_longcp.append({
            'method': 'wide',
            'makespan': row['mean_makespan'],
            'energy': row['mean_energy_active'],
            'color': 'green!8'
        })
        
        # Makespan heuristic on long_cp
        methods_longcp.append({
            'method': 'mk\\_heur',
            'makespan': heur_longcp['mk_makespan'],
            'energy': heur_longcp['mk_energy'],
            'color': 'yellow!8'
        })
        
        # Energy heuristic on long_cp
        methods_longcp.append({
            'method': 'en\\_heur',
            'makespan': heur_longcp['en_makespan'],
            'energy': heur_longcp['en_energy'],
            'color': 'orange!8'
        })
        
        # Find best for long_cp eval
        best_makespan_longcp = min(m['makespan'] for m in methods_longcp)
        best_energy_longcp = min(m['energy'] for m in methods_longcp)
        
        # Output long_cp eval rows
        for m in methods_longcp:
            is_best_mk = abs(m['makespan'] - best_makespan_longcp) < 1e-6
            is_best_en = abs(m['energy'] - best_energy_longcp) < 1e-6
            
            mk_str = format_value(m['makespan'], is_best_mk)
            en_str = format_value(m['energy'], is_best_en, energy_scale)
            
            lines.append(f"\\rowcolor{{{m['color']}}}")
            lines.append(f"{case_display} & {m['method']} & long\\_cp & {mk_str} & {en_str} \\\\")
        
        lines.append("")
        
        # Process wide eval domain
        wide_eval = rl_df[rl_df['eval_domain'] == 'wide'].copy()
        
        # Collect all methods for wide eval
        methods_wide = []
        
        # long_cp agent on wide
        row = wide_eval[wide_eval['agent_train_domain'] == 'long_cp'].iloc[0]
        methods_wide.append({
            'method': 'long\\_cp',
            'makespan': row['mean_makespan'],
            'energy': row['mean_energy_active'],
            'color': 'blue!8'
        })
        
        # wide agent on wide
        row = wide_eval[wide_eval['agent_train_domain'] == 'wide'].iloc[0]
        methods_wide.append({
            'method': 'wide',
            'makespan': row['mean_makespan'],
            'energy': row['mean_energy_active'],
            'color': 'green!8'
        })
        
        # Makespan heuristic on wide
        methods_wide.append({
            'method': 'mk\\_heur',
            'makespan': heur_wide['mk_makespan'],
            'energy': heur_wide['mk_energy'],
            'color': 'yellow!8'
        })
        
        # Energy heuristic on wide
        methods_wide.append({
            'method': 'en\\_heur',
            'makespan': heur_wide['en_makespan'],
            'energy': heur_wide['en_energy'],
            'color': 'orange!8'
        })
        
        # Find best for wide eval
        best_makespan_wide = min(m['makespan'] for m in methods_wide)
        best_energy_wide = min(m['energy'] for m in methods_wide)
        
        # Output wide eval rows
        for m in methods_wide:
            is_best_mk = abs(m['makespan'] - best_makespan_wide) < 1e-6
            is_best_en = abs(m['energy'] - best_energy_wide) < 1e-6
            
            mk_str = format_value(m['makespan'], is_best_mk)
            en_str = format_value(m['energy'], is_best_en, energy_scale)
            
            lines.append(f"\\rowcolor{{{m['color']}}}")
            lines.append(f"{case_display} & {m['method']} & wide & {mk_str} & {en_str} \\\\")
        
        lines.append("\\addlinespace[0.6em]")
        lines.append("")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table written to: {output_path}")
    print("\nTable preview:")
    print('\n'.join(lines))


def main():
    # Directories
    rl_dir = "logs/hetero_eval_all_cases"
    heuristic_dir = "logs/heuristic_multi_cases"
    output_path = "logs/merged_hetero_heuristics_table.tex"
    
    generate_latex_table(rl_dir, heuristic_dir, output_path)


if __name__ == "__main__":
    main()
