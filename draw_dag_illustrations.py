import os
import math
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

OUT_DIR = "logs/landscape"
random.seed(42)


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_linear_dag(n: int = 8, fname_prefix: str = "dag_linear") -> Tuple[str, str]:
    ensure_out_dir(OUT_DIR)
    # Node positions along a line
    xs = [i for i in range(n)]
    ys = [0 for _ in range(n)]
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)

    # Edges i -> i+1
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))

    # Nodes
    for i, (x, y) in enumerate(zip(xs, ys)):
        circle = plt.Circle((x, y), 0.12, color="#1f77b4")
        ax.add_patch(circle)
        ax.text(x, y+0.28, f"T{i+1}", ha="center", va="center", fontsize=9, color="#2c3e50")

    ax.set_aspect('equal')
    ax.set_xlim(-0.6, n-0.4)
    ax.set_ylim(-0.6, 0.8)
    ax.axis('off')
    ax.set_title("Linear DAG (Chain)", fontsize=12, pad=6, color="#2c3e50")

    png = os.path.join(OUT_DIR, f"{fname_prefix}.png")
    pdf = os.path.join(OUT_DIR, f"{fname_prefix}.pdf")
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def sample_gnp_dag(n: int = 10, p: float = 0.25) -> List[Tuple[int, int]]:
    """Generate a simple GNP DAG by orienting edges i<j: i -> j to ensure acyclicity."""
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                edges.append((i, j))
    return edges


def layout_circle(n: int) -> List[Tuple[float, float]]:
    R = 1.8
    pts: List[Tuple[float, float]] = []
    for k in range(n):
        ang = 2*math.pi * k / n
        pts.append((R*math.cos(ang), R*math.sin(ang)))
    return pts


def to_tikz_dag(n: int, edges: List[Tuple[int, int]], pos: List[Tuple[float, float]]) -> str:
    """Converts a DAG to a TikZ string for LaTeX."""
    tikz_lines = [
        "\\documentclass[tikz, border=5mm]{standalone}",
        "\\usepackage{tikz}",
        "\\begin{document}",
        "\\begin{tikzpicture}[",
        "    task/.style={circle, draw, fill=blue!20, minimum size=20pt, inner sep=0pt},",
        "    edge/.style={->, >=latex, thick}",
        "]"
    ]

    # Node definitions
    for i in range(n):
        x, y = pos[i]
        tikz_lines.append(f"    \\node[task] (T{i}) at ({x:.2f}, {y:.2f}) {{T$_{i}$}};")

    # Edge definitions
    for u, v in edges:
        tikz_lines.append(f"    \\draw[edge] (T{u}) -- (T{v});")

    tikz_lines.extend([
        "\\end{tikzpicture}",
        "\\end{document}"
    ])
    return "\n".join(tikz_lines)

def draw_gnp_dag_tikz(n: int = 10, p: float = 0.25, fname_prefix: str = "dag_gnp_tikz") -> str:
    ensure_out_dir(OUT_DIR)
    edges = sample_gnp_dag(n, p)
    pos = layout_circle(n)

    tikz_content = to_tikz_dag(n, edges, pos)

    tex_file_path = os.path.join(OUT_DIR, f"{fname_prefix}_p{p:.1f}.tex")
    with open(tex_file_path, "w") as f:
        f.write(tikz_content)

    return tex_file_path


if __name__ == "__main__":
    draw_linear_dag(n=8)
    
    # Generate TikZ files for GNP DAGs
    gnp_p03_path = draw_gnp_dag_tikz(n=12, p=0.3)
    gnp_p08_path = draw_gnp_dag_tikz(n=12, p=0.8)

    print("Saved DAG illustrations to:")
    print(" -", os.path.join(OUT_DIR, "dag_linear.png"))
    print(" -", os.path.join(OUT_DIR, "dag_linear.pdf"))
    print(f" - {gnp_p03_path}")
    print(f" - {gnp_p08_path}")

