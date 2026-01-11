#!/usr/bin/env python3
"""Compose AL and NAL t-SNE state-space images side by side.

Usage example:
    python scripts/compose_state_space_tsne_cases.py \
      --al_img runs/state_space_random/AL_case/plots/state_space_tsne_wide_vs_longcp.png \
      --nal_img runs/state_space_random/NAL_case/plots/state_space_tsne_wide_vs_longcp.png \
      --out runs/state_space_random/plots/state_space_tsne_AL_vs_NAL_side_by_side.png
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def compose_side_by_side(al_img_path: Path, nal_img_path: Path, out_path: Path) -> None:
    img_al = Image.open(al_img_path).convert("RGBA")
    img_nal = Image.open(nal_img_path).convert("RGBA")

    # Match heights by resizing NAL to AL height (keep aspect ratio)
    if img_nal.height != img_al.height:
        new_width = int(img_nal.width * (img_al.height / img_nal.height))
        img_nal = img_nal.resize((new_width, img_al.height), Image.LANCZOS)

    total_width = img_al.width + img_nal.width
    max_height = max(img_al.height, img_nal.height)

    canvas = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))
    canvas.paste(img_al, (0, 0))
    canvas.paste(img_nal, (img_al.width, 0))

    # Add (a) and (b) labels
    draw = ImageDraw.Draw(canvas)
    try:
        # Try to use a nice font, fallback to default if not available
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except:
        font = ImageFont.load_default()

    # Label positions (top-left corner with some padding)
    label_padding = 20
    text_color = (0, 0, 0, 255)  # Black
    bg_color = (255, 255, 255, 200)  # Semi-transparent white background

    # (a) for AL image
    label_a = "(a)"
    bbox_a = draw.textbbox((0, 0), label_a, font=font)
    text_width_a = bbox_a[2] - bbox_a[0]
    text_height_a = bbox_a[3] - bbox_a[1]
    pos_a = (label_padding, label_padding)
    draw.rectangle(
        [pos_a[0] - 5, pos_a[1] - 5, pos_a[0] + text_width_a + 5, pos_a[1] + text_height_a + 5],
        fill=bg_color
    )
    draw.text(pos_a, label_a, fill=text_color, font=font)

    # (b) for NAL image
    label_b = "(b)"
    bbox_b = draw.textbbox((0, 0), label_b, font=font)
    text_width_b = bbox_b[2] - bbox_b[0]
    text_height_b = bbox_b[3] - bbox_b[1]
    pos_b = (img_al.width + label_padding, label_padding)
    draw.rectangle(
        [pos_b[0] - 5, pos_b[1] - 5, pos_b[0] + text_width_b + 5, pos_b[1] + text_height_b + 5],
        fill=bg_color
    )
    draw.text(pos_b, label_b, fill=text_color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    print(f"Saved composed t-SNE figure to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose AL and NAL t-SNE images side by side.")
    parser.add_argument("--al_img", type=Path, required=True, help="Path to AL t-SNE PNG")
    parser.add_argument("--nal_img", type=Path, required=True, help="Path to NAL t-SNE PNG")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    compose_side_by_side(args.al_img, args.nal_img, args.out)


if __name__ == "__main__":
    main()
