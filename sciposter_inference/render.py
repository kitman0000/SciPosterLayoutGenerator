"""Render normalized layout predictions without loading the model."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schema import LABELS

COLORS = ("#93c5fd", "#86efac", "#fde68a", "#fca5a5", "#c4b5fd")


def render_layouts(document, output_dir, width=1920, height=1080):
    """Save one PNG per poster and an index.json mapping filenames to IDs."""
    if type(width) is not int or type(height) is not int or not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("Canvas dimensions must be integers in [1, 8192]")
    if document.get("bbox_format") != "cxcywh" or document.get("coordinate_space") != "normalized":
        raise ValueError("Expected normalized cxcywh predictions from sciposter-infer")
    posters = document.get("posters")
    if not isinstance(posters, list):
        raise ValueError("Expected a posters list")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_size = max(12, min(width, height) // 45)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    manifest = []
    for index, poster in enumerate(posters):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        skipped = 0
        for panel_index, panel in enumerate(poster["panels"]):
            if not panel.get("valid", False):
                skipped += 1
                continue
            category = panel["category"]
            bbox = panel["bbox"]
            if type(category) is not int or not 0 <= category < len(LABELS):
                raise ValueError("Panel category must be an integer in [0, 4]")
            if len(bbox) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
                raise ValueError("bbox must contain four finite numbers")
            cx, cy, w, h = bbox
            if w <= 0 or h <= 0:
                raise ValueError("Valid panels must have positive width and height")
            # Clip only for display; the original predictions remain unchanged.
            left, top = max(0, (cx - w / 2) * width), max(0, (cy - h / 2) * height)
            right, bottom = min(width - 1, (cx + w / 2) * width), min(height - 1, (cy + h / 2) * height)
            if right <= left or bottom <= top:
                skipped += 1
                continue
            draw.rectangle((left, top, right, bottom), fill=COLORS[category], outline="#334155", width=max(1, width // 640))
            label = f"{panel_index + 1}. {LABELS[category]}"
            # Avoid text spilling into neighboring panels.
            while label and draw.textbbox((0, 0), label, font=font)[2] > right - left - 8:
                label = label[:-1]
            if label and bottom - top >= font_size + 8:
                draw.text((left + 4, top + 4), label, font=font, fill="#0f172a")
        # Numeric names avoid path traversal, illegal filename characters and duplicate IDs.
        filename = f"poster_{index:05d}.png"
        canvas.save(output_dir / filename)
        manifest.append({"id": poster["id"], "file": filename, "skipped_panels": skipped})
    (output_dir / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Draw previously generated layout JSON as PNGs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    results = render_layouts(document, args.output_dir, args.width, args.height)
    print(f"Rendered {len(results)} posters to {args.output_dir}")


if __name__ == "__main__":
    main()
