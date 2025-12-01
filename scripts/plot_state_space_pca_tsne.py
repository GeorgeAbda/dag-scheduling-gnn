import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_domain_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    domain = data["domain"]
    seed = data["seed"]
    agent = data["agent"]
    step = data["step"]
    return X, domain, seed, agent, step


def main():
    parser = argparse.ArgumentParser(description="PCA & t-SNE of random-agent state spaces (wide vs longcp)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="runs/state_space_random",
        help="Directory containing *_random_states.npz",
    )
    parser.add_argument(
        "--max_samples_per_domain",
        type=int,
        default=20000,
        help="Optional cap on number of states per domain for visualization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/state_space_random/plots",
        help="Directory to save PCA/t-SNE plots",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_npz = in_dir / "wide_random_states.npz"
    long_npz = in_dir / "longcp_random_states.npz"

    X_wide, dom_w, seed_w, agent_w, step_w = load_domain_npz(wide_npz)
    X_long, dom_l, seed_l, agent_l, step_l = load_domain_npz(long_npz)

    # Subsample if desired to keep t-SNE feasible
    def subsample(X, domain_labels, max_n):
        n = X.shape[0]
        if n <= max_n:
            return X, domain_labels
        idx = np.random.RandomState(0).choice(n, size=max_n, replace=False)
        return X[idx], domain_labels[idx]

    X_wide, dom_w = subsample(X_wide, dom_w, args.max_samples_per_domain)
    X_long, dom_l = subsample(X_long, dom_l, args.max_samples_per_domain)

    X = np.concatenate([X_wide, X_long], axis=0)
    labels = np.concatenate([np.zeros(len(X_wide), dtype=int), np.ones(len(X_long), dtype=int)])

    # PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    mask_w = labels == 0
    mask_l = labels == 1
    plt.scatter(Z_pca[mask_w, 0], Z_pca[mask_w, 1], s=6, alpha=0.4, label="wide")
    plt.scatter(Z_pca[mask_l, 0], Z_pca[mask_l, 1], s=6, alpha=0.4, label="longcp")
    plt.legend()
    plt.title("State space PCA: wide vs longcp (random agents)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    pca_path = out_dir / "state_space_pca_wide_vs_longcp.png"
    plt.savefig(pca_path, dpi=300)
    plt.close()

    # t-SNE (use a moderate perplexity; can be tuned)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
    Z_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_tsne[mask_w, 0], Z_tsne[mask_w, 1], s=6, alpha=0.4, label="wide")
    plt.scatter(Z_tsne[mask_l, 0], Z_tsne[mask_l, 1], s=6, alpha=0.4, label="longcp")
    plt.legend()
    plt.title("State space t-SNE: wide vs longcp (random agents)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    tsne_path = out_dir / "state_space_tsne_wide_vs_longcp.png"
    plt.savefig(tsne_path, dpi=300)
    plt.close()

    print(f"Saved PCA plot to {pca_path}")
    print(f"Saved t-SNE plot to {tsne_path}")


if __name__ == "__main__":
    main()
