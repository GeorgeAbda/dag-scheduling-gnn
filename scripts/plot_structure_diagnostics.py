import argparse
from pathlib import Path
import re
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


FNAME_PAT = re.compile(r"^struct_diag_(?P<label>.+?)_(?P<base>linear|gnp)\.csv$")


def load_diagnostics(diag_dir: Path) -> pd.DataFrame:
    rows = []
    for p in diag_dir.glob("struct_diag_*.csv"):
        m = FNAME_PAT.match(p.name)
        if not m:
            continue
        label = m.group("label")
        base = m.group("base")
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # Expect columns: setting,makespan,total_energy,avg_active,avg_idle
        if not set(["setting","makespan","total_energy","avg_active","avg_idle"]).issubset(df.columns):
            continue
        df = df.copy()
        df["model"] = label
        df["base"] = base
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["model","base","setting","makespan","total_energy","avg_active","avg_idle"]) 
    return pd.concat(rows, ignore_index=True)


def plot_grouped_by_setting(df: pd.DataFrame, metric: str, base: str, out_dir: Path, formats: List[str]):
    sub = df[df["base"] == base].copy()
    if sub.empty:
        print(f"[plot] No diagnostics for base={base}")
        return
    settings = list(sub["setting"].unique())
    models = list(sub["model"].unique())
    settings.sort()
    models.sort()

    pivot = sub.pivot_table(index="setting", columns="model", values=metric, aggfunc="mean")[models].reindex(settings)

    plt.figure(figsize=(10, 5))
    ax = pivot.plot(kind="bar")
    ax.set_title(f"{metric} — base={base}")
    ax.set_xlabel("Setting")
    ax.set_ylabel(metric)
    plt.xticks(rotation=0)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_dir / f"diag_{base}_{metric}.{fmt}"
        plt.savefig(out_path, dpi=200)
        print(f"[plot] Saved {out_path}")
    plt.close()


def plot_grouped_by_model(df: pd.DataFrame, metric: str, base: str, out_dir: Path, formats: List[str]):
    sub = df[df["base"] == base].copy()
    if sub.empty:
        print(f"[plot] No diagnostics for base={base}")
        return
    models = list(sub["model"].unique())
    settings = list(sub["setting"].unique())
    models.sort()
    settings.sort()

    pivot = sub.pivot_table(index="model", columns="setting", values=metric, aggfunc="mean")[settings].reindex(models)

    plt.figure(figsize=(10, 5))
    ax = pivot.plot(kind="bar")
    ax.set_title(f"{metric} — base={base}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_dir / f"diag_{base}_{metric}_by_model.{fmt}"
        plt.savefig(out_path, dpi=200)
        print(f"[plot] Saved {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot structural diagnostics across models")
    ap.add_argument("--diag_dir", type=str, default="logs/diagnostics", help="Directory with struct_diag_*.csv files")
    ap.add_argument("--out_dir", type=str, default="logs/diagnostics/plots", help="Where to save plots")
    ap.add_argument("--metrics", nargs="+", default=["makespan","total_energy","avg_active","avg_idle","active_plus_idle"],
                    choices=["makespan","total_energy","avg_active","avg_idle","active_plus_idle"])
    ap.add_argument("--bases", nargs="+", default=["linear","gnp"], choices=["linear","gnp"])
    ap.add_argument("--formats", nargs="+", default=["png"], choices=["png","svg","pdf","jpg","jpeg","webp"]) 
    args = ap.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)

    df = load_diagnostics(diag_dir)
    if df.empty:
        print(f"[plot] No diagnostics found under {diag_dir}")
        return 0

    # Save an aggregated CSV for convenience
    agg_csv = out_dir.parent / "diagnostics_aggregated.csv"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(agg_csv, index=False)
    print(f"[plot] Wrote aggregated CSV to {agg_csv}")

    for base in args.bases:
        for metric in args.metrics:
            plot_grouped_by_setting(df, metric, base, out_dir, args.formats)
            plot_grouped_by_model(df, metric, base, out_dir, args.formats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
