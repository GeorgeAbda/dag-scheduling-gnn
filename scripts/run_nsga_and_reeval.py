#!/usr/bin/env python3
import argparse
import subprocess
import os
import csv
from pathlib import Path
from typing import List, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


def run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def run_seed_pipeline(static: dict, filtered_dirs: List[str], seed: int) -> int:
    """Worker: run NSGA for this seed and then re-evaluate architectures.
    static holds CLI values captured in the parent; filtered_dirs are pre-built allowlisted dirs.
    """
    out_root = Path(static["out_root"])  # str path
    # 1) NSGA reference for this seed
    nsga_out = out_root / f"nsga_seed_{seed}"
    nsga_cmd = [
        "python3", "scripts/nsga_schedule_search.py",
        "--population", str(static["population"]),
        "--generations", str(static["generations"]),
        "--key-sigma", "0.25", "--key-ratio", "0.15", "--vm-flip-prob", "0.05", "--cx-prob", "0.9",
        "--dag-method", str(static["dag_method"]),
    ]
    if static.get("gnp_p") is not None:
        nsga_cmd += ["--gnp-p", str(static["gnp_p"])]
    nsga_cmd += [
        "--gnp-min-n", str(static["gnp_min_n"]), "--gnp-max-n", str(static["gnp_max_n"]),
        "--host-count", str(static["host_count"]), "--vm-count", str(static["vm_count"]), "--workflow-count", str(static["workflow_count"]),
        "--seed", str(seed),
        "--device", str(static["device"]),
        "--out-dir", str(nsga_out),
    ]
    run(nsga_cmd)

    # 2) Re-evaluate per-arch checkpoints for this seed
    reeval_out = out_root / f"reeval_seed_{seed}"
    cap_cmd = [
        "python3", "scripts/compare_arch_pareto.py",
        "--dirs", *filtered_dirs,
        "--episodes", str(static["episodes"]),
        "--seed-base", str(seed),
        "--device", str(static["device"]),
        "--out-dir", str(reeval_out),
        "--dag-method", str(static["dag_method"]),
    ]
    if static.get("gnp_p") is not None:
        cap_cmd += ["--gnp-p", str(static["gnp_p"])]
    cap_cmd += [
        "--gnp-min-n", str(static["gnp_min_n"]), "--gnp-max-n", str(static["gnp_max_n"]),
        "--host-count", str(static["host_count"]), "--vm-count", str(static["vm_count"]), "--workflow-count", str(static["workflow_count"]),
        "--ref-front-csv", str(nsga_out / "reference_front.csv"),
        "--front-scope", "per_seed",
    ]
    if static.get("style_ga"):
        cap_cmd.append("--style-ga")
    if static.get("plot_per_seed"):
        cap_cmd.append("--plot-per-seed")
    run(cap_cmd)
    return int(seed)


def main():
    ap = argparse.ArgumentParser(description="Run NSGA per seed, then re-evaluate all checkpoints of each architecture on that configuration and emit per-seed fronts and plots. Optionally aggregate to a LaTeX table.")
    ap.add_argument("--arch-dirs", nargs="+", required=True, help="Architecture directories containing *_pareto_*.pt checkpoints")
    ap.add_argument("--out-root", required=True, help="Output root directory")
    # Config shared with nsga_schedule_search and compare_arch_pareto eval
    ap.add_argument("--dag-method", required=True)
    ap.add_argument("--gnp-p", type=float, default=None)
    ap.add_argument("--gnp-min-n", type=int, required=True)
    ap.add_argument("--gnp-max-n", type=int, required=True)
    ap.add_argument("--host-count", type=int, required=True)
    ap.add_argument("--vm-count", type=int, required=True)
    ap.add_argument("--workflow-count", type=int, required=True)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=10, help="Episodes per-checkpoint during re-evaluation")
    ap.add_argument("--style-ga", action="store_true", help="Use GA-style plotting aesthetics in compare_arch_pareto")
    ap.add_argument("--plot-per-seed", action="store_true", help="Emit per-seed overlay plots of reevaluated fronts")
    # Optional allowlist directory (deprecated; auto-detect by default)
    ap.add_argument("--allowlist-dir", type=str, default=None, help="[Optional] Legacy: directory containing per-architecture CSV files named <arch>.csv. If not provided, the script will auto-detect <arch>_pareto.csv near each architecture directory.")
    # Parallelism
    ap.add_argument("--jobs", type=int, default=1, help="Number of seeds to process in parallel inside this job")
    # LaTeX aggregation (skippable)
    ap.add_argument("--build-latex", action="store_true", help="If set, build a LaTeX table from re-evaluated fronts at the end")
    ap.add_argument("--latex-out", type=str, default=None, help="Output path for LaTeX table (defaults to ROOT/reeval_summary.tex)")
    ap.add_argument("--latex-arches", nargs="*", default=None, help="Architectures to include in LaTeX (defaults to auto-detect)")
    ap.add_argument("--latex-floatfmt", type=str, default=".3f", help="Float format in LaTeX table")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    def read_allowlist(csv_path: Path) -> Set[str]:
        allow: Set[str] = set()
        try:
            with csv_path.open("r", newline="") as f:
                # Try DictReader first to use 'checkpoint' column
                f.seek(0)
                dr = csv.DictReader(f)
                used_dict = False
                if dr.fieldnames:
                    lower_fields = [fn.strip().lower() for fn in dr.fieldnames]
                    if "checkpoint" in lower_fields:
                        used_dict = True
                        key = dr.fieldnames[lower_fields.index("checkpoint")]
                        for row in dr:
                            tok = str(row.get(key, "")).strip()
                            if not tok:
                                continue
                            allow.add(tok)
                            if tok.endswith('.pt'):
                                allow.add(Path(tok).stem)
                if not used_dict:
                    # Fallback to simple CSV rows
                    f.seek(0)
                    reader = csv.reader(f)
                    for row in reader:
                        for cell in row:
                            tok = str(cell).strip()
                            if not tok or tok.lower() in {"makespan", "active_energy", "iteration", "global_step", "checkpoint"}:
                                continue
                            allow.add(tok)
                            if tok.endswith('.pt'):
                                allow.add(Path(tok).stem)
        except FileNotFoundError:
            pass
        return allow

    def find_arch_csv(arch_dir: Path, arch_name: str) -> Optional[Path]:
        # Check arch_dir/<arch>_pareto.csv
        cand1 = arch_dir / f"{arch_name}_pareto.csv"
        if cand1.exists():
            return cand1
        # Check parent/<arch>_pareto.csv (e.g., ablation level)
        parent = arch_dir.parent
        cand2 = parent / f"{arch_name}_pareto.csv"
        if cand2.exists():
            return cand2
        # Check one level higher
        gp = parent.parent
        cand3 = gp / f"{arch_name}_pareto.csv"
        if gp.exists() and cand3.exists():
            return cand3
        return None

    def build_filtered_dirs(arch_dirs: List[str], allowlist_root: Optional[Path], out_root: Path, tmp_suffix: Optional[str] = None) -> List[str]:
        tmp_dir_name = f"tmp_allow_{tmp_suffix}" if tmp_suffix else "tmp_allow"
        tmp_root = out_root / tmp_dir_name
        tmp_root.mkdir(parents=True, exist_ok=True)
        filtered: List[str] = []
        for d in arch_dirs:
            src = Path(d)
            arch = src.name.rstrip('/\\')
            # Resolve CSV path: prefer auto-detected <arch>_pareto.csv near src, otherwise legacy allowlist-root/<arch>.csv
            csv_file = find_arch_csv(src, arch)
            if csv_file is None and allowlist_root is not None:
                legacy = allowlist_root / f"{arch}.csv"
                csv_file = legacy if legacy.exists() else None
            allow = read_allowlist(csv_file) if csv_file else set()
            # If no allowlist for this arch, keep original dir
            if not allow:
                filtered.append(str(src))
                continue
            dest = tmp_root / arch
            dest.mkdir(parents=True, exist_ok=True)
            # link only allowed checkpoints
            for f in src.glob("*.pt"):
                name = f.name
                stem = f.stem
                if (name in allow) or (stem in allow):
                    link_path = dest / name
                    try:
                        if link_path.exists():
                            link_path.unlink()
                        os.symlink(os.path.abspath(str(f)), str(link_path))
                    except Exception:
                        # fallback: copy file if symlink not permitted
                        import shutil
                        shutil.copy2(str(f), str(link_path))
            filtered.append(str(dest))
        return filtered

    # Build filtered directories once and reuse across seeds (allowlist is seed-independent)
    # Avoid tmp collisions across concurrent jobs by suffixing with job-id/pid
    tmp_suffix = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("JOB_ID") or str(os.getpid())
    filtered_dirs_once = build_filtered_dirs([str(p) for p in args.arch_dirs], Path(args.allowlist_dir) if args.allowlist_dir else None, out_root, tmp_suffix=tmp_suffix)

    # Seeds list
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))

    # Prepare static config for workers
    static = {
        "out_root": str(out_root),
        "population": int(args.population),
        "generations": int(args.generations),
        "dag_method": str(args.dag_method),
        "gnp_p": float(args.gnp_p) if args.gnp_p is not None else None,
        "gnp_min_n": int(args.gnp_min_n),
        "gnp_max_n": int(args.gnp_max_n),
        "host_count": int(args.host_count),
        "vm_count": int(args.vm_count),
        "workflow_count": int(args.workflow_count),
        "device": str(args.device),
        "episodes": int(args.episodes),
        "style_ga": bool(args.style_ga),
        "plot_per_seed": bool(args.plot_per_seed),
    }

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for s in seeds:
            run_seed_pipeline(static, filtered_dirs_once, s)
    else:
        print(f"[parallel] Running {len(seeds)} seeds with jobs={jobs}")
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_seed_pipeline, static, filtered_dirs_once, s): s for s in seeds}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    _ = fut.result()
                    print(f"[parallel] Seed {s} done")
                except Exception as e:
                    print(f"[parallel] Seed {s} failed: {e}")

    # Optional LaTeX aggregation
    if args.build_latex:
        tex_out = args.latex_out or str(out_root / "reeval_summary.tex")
        cmd = [
            "python3", "scripts/aggregate_reeval_to_latex.py",
            str(out_root),
            "--out-tex", tex_out,
            "--floatfmt", str(args.latex_floatfmt),
        ]
        if args.latex_arches:
            cmd.extend(["--arches", *args.latex_arches])
        run(cmd)

    print(f"[all-done] Outputs in: {out_root}")


if __name__ == "__main__":
    main()
