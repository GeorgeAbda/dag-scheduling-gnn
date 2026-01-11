import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
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
    parser.add_argument(
        "--wide_color",
        type=str,
        default="#2E7D32",
        help="Color for 'wide' domain points (hex or named color)",
    )
    parser.add_argument(
        "--longcp_color",
        type=str,
        default="#66BB6A",
        help="Color for 'longcp' domain points (hex or named color)",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=6.5,
        help="Figure width (inches)",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=5.5,
        help="Figure height (inches)",
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=6,
        help="Marker size for scatter points",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Alpha (transparency) for scatter points",
    )
    parser.add_argument(
        "--legend_outside",
        action="store_true",
        help="Place the legend outside the axes (right side)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Enable a very light grid",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for PNG",
    )
    parser.add_argument(
        "--save_pdf",
        action="store_true",
        help="Also save PDF versions of the figures",
    )
    parser.add_argument(
        "--save_svg",
        action="store_true",
        help="Also save SVG versions of the figures",
    )
    parser.add_argument(
        "--title_suffix",
        type=str,
        default="(random agents)",
        help="Suffix appended to plot titles",
    )
    parser.add_argument(
        "--equal_aspect",
        action="store_true",
        help="Use equal aspect ratio for axes",
    )
    parser.add_argument(
        "--kde_contours",
        action="store_true",
        help="Overlay 2D KDE contours for each domain (publication style)",
    )
    parser.add_argument(
        "--kde_levels",
        type=int,
        default=6,
        help="Number of contour levels for KDE",
    )
    parser.add_argument(
        "--kde_bw",
        type=float,
        default=1.0,
        help="Bandwidth adjust (bw_adjust) for KDE",
    )
    parser.add_argument(
        "--kde_linewidth",
        type=float,
        default=1.2,
        help="Line width for KDE contours",
    )
    parser.add_argument(
        "--kde_alpha",
        type=float,
        default=0.9,
        help="Alpha for KDE contour lines",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
    })

    wide_npz = in_dir / "wide_random_states.npz"
    long_npz = in_dir / "longcp_random_states.npz"

    pbar = tqdm(total=4, desc="State-space plotting", unit="stage")

    # Stage 1: load data
    X_wide, dom_w, seed_w, agent_w, step_w = load_domain_npz(wide_npz)
    X_long, dom_l, seed_l, agent_l, step_l = load_domain_npz(long_npz)
    pbar.update(1)

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

    pbar.update(1)  # finished preprocessing / subsampling

    # PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    mask_w = labels == 0
    mask_l = labels == 1
    ax.scatter(
        Z_pca[mask_w, 0], Z_pca[mask_w, 1],
        s=args.marker_size, alpha=args.alpha, label="wide", c=args.wide_color,
        linewidths=0.0, edgecolors="none"
    )
    ax.scatter(
        Z_pca[mask_l, 0], Z_pca[mask_l, 1],
        s=args.marker_size, alpha=args.alpha, label="longcp", c=args.longcp_color,
        linewidths=0.0, edgecolors="none"
    )
    if args.kde_contours:
        sns.kdeplot(x=Z_pca[mask_w, 0], y=Z_pca[mask_w, 1],
                    levels=args.kde_levels, color=args.wide_color,
                    bw_adjust=args.kde_bw, linewidths=args.kde_linewidth,
                    alpha=args.kde_alpha, fill=False, ax=ax)
        sns.kdeplot(x=Z_pca[mask_l, 0], y=Z_pca[mask_l, 1],
                    levels=args.kde_levels, color=args.longcp_color,
                    bw_adjust=args.kde_bw, linewidths=args.kde_linewidth,
                    alpha=args.kde_alpha, fill=False, ax=ax)
    if args.legend_outside:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        ax.legend(frameon=False)
    ax.set_title(f"State space PCA: wide vs longcp {args.title_suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if args.equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.grid(args.grid, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    pca_path = out_dir / "state_space_pca_wide_vs_longcp.png"
    fig.savefig(pca_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    if args.save_pdf:
        fig.savefig(pca_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    if args.save_svg:
        fig.savefig(pca_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    pbar.update(1)  # finished PCA plot

    # t-SNE (use a moderate perplexity; can be tuned)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=0, init="pca")
    Z_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.scatter(
        Z_tsne[mask_w, 0], Z_tsne[mask_w, 1],
        s=args.marker_size, alpha=args.alpha, label="wide", c=args.wide_color,
        linewidths=0.0, edgecolors="none"
    )
    ax.scatter(
        Z_tsne[mask_l, 0], Z_tsne[mask_l, 1],
        s=args.marker_size, alpha=args.alpha, label="longcp", c=args.longcp_color,
        linewidths=0.0, edgecolors="none"
    )
    if args.kde_contours:
        sns.kdeplot(x=Z_tsne[mask_w, 0], y=Z_tsne[mask_w, 1],
                    levels=args.kde_levels, color=args.wide_color,
                    bw_adjust=args.kde_bw, linewidths=args.kde_linewidth,
                    alpha=args.kde_alpha, fill=False, ax=ax)
        sns.kdeplot(x=Z_tsne[mask_l, 0], y=Z_tsne[mask_l, 1],
                    levels=args.kde_levels, color=args.longcp_color,
                    bw_adjust=args.kde_bw, linewidths=args.kde_linewidth,
                    alpha=args.kde_alpha, fill=False, ax=ax)
    if args.legend_outside:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        ax.legend(frameon=False)
    # ax.set_title("State space t-SNE: wide vs longcp")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if args.equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.grid(args.grid, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    tsne_path = out_dir / "state_space_tsne_wide_vs_longcp.png"
    fig.savefig(tsne_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    if args.save_pdf:
        fig.savefig(tsne_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    if args.save_svg:
        fig.savefig(tsne_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    pbar.update(1)  # finished t-SNE plot
    pbar.close()

    print(f"Saved PCA plot to {pca_path}")
    print(f"Saved t-SNE plot to {tsne_path}")


if __name__ == "__main__":
    main()
