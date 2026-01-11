from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tyro

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cogito.dataset_generator.core.models import Dataset, Workflow
from cogito.gnn_deeprl_model.representative_eval import _gen_ds_for_seed, _extract_features


from dataclasses import dataclass


@dataclass
class Args:
    wide_config: str
    longcp_config: str
    rep_dataset_json: str
    out_dir: str = "runs/representativeness"
    out_png: str | None = None


def _load_cfg(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _features_for_train_seeds(cfg: Dict[str, Any], domain: str) -> Tuple[np.ndarray, List[str]]:
    tr = cfg.get("train", {})
    seeds: List[int] = [int(s) for s in tr.get("seeds", [])]
    if not seeds:
        raise RuntimeError(f"No train.seeds found in config for domain={domain}")
    feats: List[np.ndarray] = []
    for s in seeds:
        ds = _gen_ds_for_seed(int(s), tr)
        f, names = _extract_features(ds)
        feats.append(f)
    return np.stack(feats, axis=0), names


def _features_for_each_workflow_in_dataset(ds: Dataset) -> Tuple[np.ndarray, List[str]]:
    rows: List[np.ndarray] = []
    names: List[str] | None = None
    for wf in ds.workflows:
        tmp = Dataset(workflows=[wf], vms=ds.vms, hosts=ds.hosts)
        f, names = _extract_features(tmp)
        rows.append(f)
    assert names is not None
    return np.stack(rows, axis=0), names  # (k, D)


def main(a: Args) -> None:
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_w = _load_cfg(a.wide_config)
    cfg_l = _load_cfg(a.longcp_config)

    Fw, names = _features_for_train_seeds(cfg_w, domain="wide")
    Fl, _ = _features_for_train_seeds(cfg_l, domain="longcp")

    rep_data = json.loads(Path(a.rep_dataset_json).read_text())
    rep_ds = Dataset.from_json(rep_data)
    Fr, _ = _features_for_each_workflow_in_dataset(rep_ds)  # (k, D)

    # Build labeled dataframe
    df_w = pd.DataFrame(Fw, columns=names)
    df_w["domain"] = "wide"
    df_w["set"] = "original"

    df_l = pd.DataFrame(Fl, columns=names)
    df_l["domain"] = "longcp"
    df_l["set"] = "original"

    # Representatives domain labels: prefer exact metadata if present
    labels_rep: List[str]
    meta = rep_data.get("meta", {}) if isinstance(rep_data, dict) else {}
    wf_domains = meta.get("workflow_domains", None)
    if isinstance(wf_domains, list) and len(wf_domains) == Fr.shape[0]:
        labels_rep = [str(d) for d in wf_domains]
    else:
        # Fallback: nearest centroid labeling
        mu_w = Fw.mean(axis=0)
        mu_l = Fl.mean(axis=0)
        labels_rep = []
        for r in Fr:
            dw = float(np.sum((r - mu_w) ** 2))
            dl = float(np.sum((r - mu_l) ** 2))
            labels_rep.append("wide" if dw <= dl else "longcp")
    df_r = pd.DataFrame(Fr, columns=names)
    df_r["domain"] = labels_rep
    df_r["set"] = "representative"

    df_all = pd.concat([df_w, df_l, df_r], ignore_index=True)
    df_all.to_csv(out_dir / "pca_reps_vs_original_features.csv", index=False)

    # PCA
    from sklearn.decomposition import PCA
    X = df_all[names].values.astype(float)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)
    df_all["PC1"] = Z[:, 0]
    df_all["PC2"] = Z[:, 1]

    # Plot
    plt.figure(figsize=(8, 6))
    # Originals
    mask_w_o = (df_all["domain"] == "wide") & (df_all["set"] == "original")
    mask_l_o = (df_all["domain"] == "longcp") & (df_all["set"] == "original")
    plt.scatter(df_all.loc[mask_w_o, "PC1"], df_all.loc[mask_w_o, "PC2"], s=20, alpha=0.35, c="tab:orange", label="wide (orig)")
    plt.scatter(df_all.loc[mask_l_o, "PC1"], df_all.loc[mask_l_o, "PC2"], s=20, alpha=0.35, c="tab:blue", label="longcp (orig)")
    # Representatives (highlight)
    mask_w_r = (df_all["domain"] == "wide") & (df_all["set"] == "representative")
    mask_l_r = (df_all["domain"] == "longcp") & (df_all["set"] == "representative")
    plt.scatter(df_all.loc[mask_w_r, "PC1"], df_all.loc[mask_w_r, "PC2"], s=120, marker="X", edgecolors="black", linewidths=1.5, c="tab:orange", label="wide (rep)")
    plt.scatter(df_all.loc[mask_l_r, "PC1"], df_all.loc[mask_l_r, "PC2"], s=120, marker="X", edgecolors="black", linewidths=1.5, c="tab:blue", label="longcp (rep)")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = Path(a.out_png) if a.out_png else (out_dir / "pca_reps_vs_original.png")
    plt.savefig(out_png, dpi=180)
    plt.close()

    # Save PCA loadings and EVR
    loadings = pd.DataFrame(pca.components_.T, index=names, columns=["PC1", "PC2"]) 
    evr = pd.Series(pca.explained_variance_ratio_, index=["PC1", "PC2"]) 
    loadings.to_csv(out_dir / "pca_reps_vs_original_loadings.csv")
    evr.to_csv(out_dir / "pca_reps_vs_original_evr.csv")

    print(f"Saved: {out_png}")
    print(f"Saved: {out_dir / 'pca_reps_vs_original_features.csv'}")
    print(f"Saved: {out_dir / 'pca_reps_vs_original_loadings.csv'}")
    print(f"Saved: {out_dir / 'pca_reps_vs_original_evr.csv'}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Args))
