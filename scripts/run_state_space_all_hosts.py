import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

HOST_VARIANTS = [
    {
        "name": "hetero",
        "host_specs": ROOT / "data" / "host_specs.json",
    },
    {
        "name": "homoPower",
        "host_specs": ROOT / "data" / "host_specs_homoPower.json",
    },
    {
        "name": "homospeed",
        "host_specs": ROOT / "data" / "host_specs_homospeed.json",
    },
    {
        "name": "NAL_case",
        "host_specs": ROOT / "logs" / "NAL_case" / "host_specs.json",
    },
]


def run_for_variant(variant: dict) -> None:
    name = variant["name"]
    host_specs_path = str(variant["host_specs"])

    env = os.environ.copy()
    env["HOST_SPECS_PATH"] = host_specs_path

    base_out = ROOT / "runs" / "state_space_random" / name
    base_out.mkdir(parents=True, exist_ok=True)

    # 1) Collect random state-space for wide + longcp
    print(f"[run] Collecting random state space for variant={name} using {host_specs_path}")
    subprocess.run(
        [
            "python",
            "-m",
            "scheduler.rl_model.collect_random_state_space",
            "--output_dir",
            str(base_out),
        ],
        cwd=str(ROOT),
        check=True,
        env=env,
    )

    # 2) Plot PCA & t-SNE for this variant
    plots_out = base_out / "plots"
    print(f"[run] Plotting PCA/t-SNE for variant={name}")
    subprocess.run(
        [
            "python",
            "scripts/plot_state_space_pca_tsne.py",
            "--input_dir",
            str(base_out),
            "--output_dir",
            str(plots_out),
            "--save_pdf",
            "--kde_contours",
            "--grid",
            "--legend_outside",
            "--equal_aspect",
            "--marker_size",
            "8",
            "--alpha",
            "0.7",
        ],
        cwd=str(ROOT),
        check=True,
        env=env,
    )


def main() -> None:
    for v in HOST_VARIANTS:
        run_for_variant(v)


if __name__ == "__main__":
    main()
