import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Paths to the four panels and their titles
paths = {
    "a": ("figs/AL_longCP_active_energy.jpg", "a) Active energy, AL Long-CP"),
    "b": ("figs/NA_longCP_active_energy.jpg", "b) Active energy, NA Long-CP"),
    "c": ("figs/AL_longCP_makespan.jpg",      "c) Makespan, AL Long-CP"),
    "d": ("figs/NA_longCP_makespan.jpg",      "d) Makespan, NA Long-CP"),
}

def read_raster(path: Path):
    """Read a raster image and fail clearly for non-raster files.
    Supports .png, .jpg, .jpeg.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(
            f"{path} is not a raster image (.png/.jpg). "
            "Export to PNG/JPG (or convert) and retry."
        )
    try:
        return mpimg.imread(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path}: {e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, key in enumerate(["a", "b", "c", "d"]):
    img_path_str, title = paths[key]
    img_path = Path(img_path_str)
    img = read_raster(img_path)
    axes[i].imshow(img)
    axes[i].set_title(title, fontsize=12, loc="left", pad=8)
    axes[i].axis("off")

plt.tight_layout()
out_path = Path("figs/actor_landscape_2x2.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path.resolve()}")
