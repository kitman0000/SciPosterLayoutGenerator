import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Standalone Sci-PosterLayout BART inference")
    parser.add_argument("--config", type=Path, required=True, help="Training config.yaml or standalone model config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact .pt state_dict path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON feature input")
    source.add_argument("--dataset-dir", type=Path, help="Evaluation dataset root or raw directory")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--poster-id", action="append", help="Select an ID from the split; repeatable")
    parser.add_argument("--limit", type=int, help="Limit selected dataset posters")
    parser.add_argument("--features-output", type=Path, help="Save the exact input features as JSON")
    parser.add_argument("--output", type=Path, required=True, help="JSON predictions")
    parser.add_argument("--npy-output", type=Path, help="Optional [N,10,5] category,cx,cy,w,h array")
    parser.add_argument("--image-dir", type=Path, help="Optional directory for layout PNGs")
    parser.add_argument("--width", type=int, default=1920, help="PNG canvas width")
    parser.add_argument("--height", type=int, default=1080, help="PNG canvas height")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling", choices=("greedy", "random"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.input and (args.poster_id or args.limit is not None or args.split != "test"):
        parser.error("--split, --poster-id and --limit require --dataset-dir")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.image_dir and not (1 <= args.width <= 8192 and 1 <= args.height <= 8192):
        parser.error("Canvas dimensions must be in [1, 8192]")

    # Keep --help usable before dependencies have been installed.
    import torch
    import yaml
    from .schema import ModelConfig, validate_input
    from .model import load_checkpoint
    from .predictor import generate

    with args.config.open(encoding="utf-8-sig") as stream:
        config = ModelConfig.from_mapping(yaml.safe_load(stream))
    if args.dataset_dir:
        from .dataset import load_dataset
        document = load_dataset(args.dataset_dir, args.split, args.poster_id, args.limit)
    else:
        document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    validate_input(document, config)
    if args.features_output:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        args.features_output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, config, device)
    result = generate(model, document, args.batch_size, args.sampling, args.temperature, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.npy_output:
        import numpy as np
        values = np.zeros((len(result["posters"]), 10, 5), dtype=np.float32)
        values[:, :, 0] = 5
        for i, poster in enumerate(result["posters"]):
            for j, panel in enumerate(poster["panels"]):
                if panel["valid"]:
                    values[i, j] = [panel["category"], *panel["bbox"]]
        args.npy_output.parent.mkdir(parents=True, exist_ok=True)
        with args.npy_output.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
    if args.image_dir:
        from .render import render_layouts
        render_layouts(result, args.image_dir, args.width, args.height)
    print(f"Wrote {len(result['posters'])} posters to {args.output}")
