#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

# Ensure project root (one level up from scripts/) is on sys.path so that 'scheduler' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.dataset_generator.core import gen_vm
from scheduler.viz_results.decision_boundaries import score_correlation_agents as corr


def _load_train_cfg(cfg_path: Path) -> tuple[Dict, List[int]]:
    cfg = json.loads(cfg_path.read_text())
    train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    seeds = [int(s) for s in train.get("seeds", [])]
    return dict(train.get("dataset", {})), seeds


def run_case_two_seeds(
    case_name: str,
    model_a_path: Path,
    model_b_path: Path,
    host_specs_path: Path,
    wide_cfg_path: Path,
    long_cfg_path: Path,
    wide_eval_seeds: list[int],
    long_eval_seeds: list[int],
    out_root: Path,
    max_decision_steps: int = 200,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override host specs for this case
    gen_vm.HOST_SPECS_PATH = host_specs_path

    domains = {
        "wide": (wide_cfg_path, wide_eval_seeds),
        "long_cp": (long_cfg_path, long_eval_seeds),
    }

    for domain, (cfg_path, eval_seeds) in domains.items():
        ds_cfg, _ = _load_train_cfg(cfg_path)
        sel_seeds = [int(s) for s in (eval_seeds or [])]
        if not sel_seeds:
            raise SystemExit(f"No eval seeds provided for domain={domain} (case={case_name})")

        all_A: List[np.ndarray] = []
        all_B: List[np.ndarray] = []

        for s in sel_seeds:
            ds = corr._build_fixed_dataset(ds_cfg, s, override_req_divisor=None)
            args = corr.Args(
                model_a_path=str(model_a_path),
                model_b_path=str(model_b_path),
                agent_type="hetero",
                max_decision_steps=max_decision_steps,
                driver="A",
            )
            A_s, B_s = corr.collect_scores(args, device, dataset_override=ds, job_seed=s)
            if len(A_s) > 0 and len(B_s) > 0:
                all_A.append(A_s)
                all_B.append(B_s)

        if not all_A or not all_B:
            raise SystemExit(f"No scores collected for case={case_name}, domain={domain}")

        A = np.concatenate(all_A, axis=0)
        B = np.concatenate(all_B, axis=0)

        out_dir = out_root / case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{case_name}_{domain}_two_seeds.png"

        a_label = f"{case_name}_A"
        b_label = f"{case_name}_B"
        corr.plot_correlation(A, B, str(out_path), dpi=300, label_a=a_label, label_b=b_label)
        print(f"[corr-two-seeds] Saved {case_name} {domain} figure to: {out_path}")


def main() -> None:
    project_root = Path(PROJECT_ROOT)

    wide_cfg = project_root / "data/rl_configs/train_wide_p005_seeds.json"
    long_cfg = project_root / "data/rl_configs/train_long_cp_p08_seeds.json"

    # Load the 10 representative seeds per domain
    long_sel_path = project_root / "runs/datasets/longcp/representativeness_new/selected_eval_seeds_longcp_k10.json"
    wide_sel_path = project_root / "runs/datasets/wide/representativeness/selected_eval_seeds_wide_k10.json"

    long_sel_data = json.loads(long_sel_path.read_text())
    wide_sel_data = json.loads(wide_sel_path.read_text())

    long_eval_seeds = [int(s) for s in long_sel_data.get("selected_eval_seeds", [])]
    wide_eval_seeds = [int(s) for s in wide_sel_data.get("selected_eval_seeds", [])]

    out_root = project_root / "logs" / "corr_three_cases_two_seeds"

    # Case 1: HP homopower-controlled agents + homopower host specs
    run_case_two_seeds(
        case_name="HP_homopower",
        model_a_path=project_root / "logs/HP_controlled/hetero_wide_homopower_controlled/ablation/per_variant/hetero/hetero_best_return.pt",
        model_b_path=project_root / "logs/HP_controlled/long_cp_specialist_homopower_controlled/ablation/per_variant/hetero/hetero_best_return.pt",
        host_specs_path=project_root / "data/host_specs_homoPower.json",
        wide_cfg_path=wide_cfg,
        long_cfg_path=long_cfg,
        wide_eval_seeds=wide_eval_seeds,
        long_eval_seeds=long_eval_seeds,
        out_root=out_root,
    )

    # Case 2: HS homospeed-controlled agents + homospeed host specs
    run_case_two_seeds(
        case_name="HS_homospeed",
        model_a_path=project_root / "logs/HS_controlled/hetero_wide_homospeed_controlled/ablation/per_variant/hetero/hetero_best_return.pt",
        model_b_path=project_root / "logs/HS_controlled/long_cp_specialist_traj_homospeed_controlled/ablation/per_variant/hetero/hetero_best_return.pt",
        host_specs_path=project_root / "data/host_specs_homospeed.json",
        wide_cfg_path=wide_cfg,
        long_cfg_path=long_cfg,
        wide_eval_seeds=wide_eval_seeds,
        long_eval_seeds=long_eval_seeds,
        out_root=out_root,
    )

    # Case 3: Original wide/long_cp specialists + NAL host specs
    run_case_two_seeds(
        case_name="NAL_baseline",
        model_a_path=project_root / "logs/wide_specialist_traj/ablation/per_variant/hetero/hetero_best_return.pt",
        model_b_path=project_root / "logs/long_cp_specialist_traj/ablation/per_variant/hetero/hetero_best_return.pt",
        host_specs_path=project_root / "logs/NAL_case/host_specs.json",
        wide_cfg_path=wide_cfg,
        long_cfg_path=long_cfg,
        wide_eval_seeds=wide_eval_seeds,
        long_eval_seeds=long_eval_seeds,
        out_root=out_root,
    )


if __name__ == "__main__":
    main()
