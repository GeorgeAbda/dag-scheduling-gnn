import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import csv

def load_jsons(paths: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        try:
            with p.open('r') as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out

METRICS = [
    "sinkhorn_cost",
    "topk_overlap",
    "ot_mass_A_to_B_topk",
    "avg_entropy_from_A_topk",
    "bary_msd",
    "bary_mean_norm",
    "bary_mean_abs_angle_deg",
    "num_pairs",
]

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    # results is list of JSON dicts with a 'results' mapping: key -> metrics
    # We will aggregate per key across the list
    accum: Dict[str, Dict[str, List[float]]] = {}
    for js in results:
        rd = js.get("results", {})
        for key, vals in rd.items():
            if key not in accum:
                accum[key] = {m: [] for m in METRICS}
            for m in METRICS:
                v = vals.get(m)
                if v is None:
                    continue
                try:
                    accum[key][m].append(float(v))
                except Exception:
                    pass
    # Compute mean/std/ci
    out: Dict[str, Dict[str, float]] = {}
    for key, md in accum.items():
        out[key] = {}
        for m, arr in md.items():
            if not arr:
                continue
            a = np.asarray(arr, dtype=float)
            mean = float(np.mean(a))
            std = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(max(1, a.size))) if a.size > 1 else 0.0
            out[key][f"{m}_mean"] = mean
            out[key][f"{m}_std"] = std
            out[key][f"{m}_ci95"] = ci95
            out[key]["n"] = a.size
    return out


def write_csv(path: Path, agg: Dict[str, Dict[str, float]]):
    # collect headers
    headers = sorted({h for v in agg.values() for h in v.keys()})
    headers = ["result_key"] + headers
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for key, vals in sorted(agg.items()):
            row = {"result_key": key}
            row.update(vals)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Aggregate multiple ot_compare JSONs into mean/std/CI CSV")
    ap.add_argument("--inputs", nargs='+', required=True, help="List of JSON result files to aggregate")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    files = [Path(p) for p in args.inputs]
    js = load_jsons(files)
    if not js:
        print("No valid JSON files provided.")
        return 2
    agg = aggregate(js)
    write_csv(Path(args.out_csv), agg)
    print(f"Wrote {args.out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
