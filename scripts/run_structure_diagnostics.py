from pathlib import Path
import os
import sys
import torch

# Ensure project root is on sys.path so 'scheduler' package is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args,
    run_structure_diagnostics,
)

# Edit these to your exact checkpoints and variants if paths change
MODELS = [
    ("noglobal_gnp", 
     "/Users/anashattay/Documents/GitHub/PERFECT/logs/1758035010_noglobal_gnp/ablation/per_variant/no_global_actor_model.pt", 
     "no_global_actor"),
    ("noglobal_linear", 
     "/Users/anashattay/Documents/GitHub/PERFECT/logs/1758048011_noglobal_linear/ablation/per_variant/no_global_actor_model.pt", 
     "no_global_actor"),
    ("baseline_linear", 
     "/Users/anashattay/Documents/GitHub/PERFECT/logs/1758052085_baseline_linear/model.pt", 
     "baseline"),
    ("baseline_gnp", 
     "/Users/anashattay/Documents/GitHub/PERFECT/logs/1758052297_baseline_gnp/model.pt", 
     "baseline"),
]

def make_variant(variant_name: str) -> AblationVariant:
    if variant_name == "no_global_actor":
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    elif variant_name == "baseline":
        return AblationVariant(name="baseline", use_actor_global_embedding=True)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


def main():
    device = torch.device("cpu")  # change to 'cuda' if available and desired

    # Common eval args template
    base_template = Args()
    base_template.device = "cpu"
    base_template.test_iterations = 30  # adjust as needed for stability/speed
    # Align evaluation seeds with training default (1)
    setattr(base_template, "eval_seed_base", base_template.seed)

    out_root = Path("logs/diagnostics")
    out_root.mkdir(parents=True, exist_ok=True)

    for label, ckpt_path, variant_name in MODELS:
        variant = make_variant(variant_name)
        agent = AblationGinAgent(device, variant)
        state = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(state, strict=False)
        agent.eval()

        for base in ["gnp", "linear"]:
            print(f"[diagnostics] Running for {label} base={base}")
            base_args = Args()
            # copy from template
            base_args.device = base_template.device
            base_args.test_iterations = base_template.test_iterations
            setattr(base_args, "eval_seed_base", getattr(base_template, "eval_seed_base"))
            base_args.dataset.dag_method = base
            # Force higher GNP connectivity to stress non-linear structure
            if base == "gnp":
                try:
                    setattr(base_args.dataset, "gnp_p", 0.3)
                except Exception:
                    pass

            out_csv = out_root / f"struct_diag_{label}_{base}.csv"
            run_structure_diagnostics(agent, base_args, out_csv)
            print(f"[diagnostics] Wrote {out_csv}")


if __name__ == "__main__":
    main()
