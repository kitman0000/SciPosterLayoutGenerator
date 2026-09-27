"""Read the raw evaluation dataset without importing trainer or PyG."""

import argparse
import json
from pathlib import Path

import numpy as np


def _ids(path):
    values = path.read_text(encoding="utf-8-sig").splitlines()
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate IDs in {path}")
    if any(not value or "/" in value or "\\" in value or value in (".", "..") for value in values):
        raise ValueError(f"Invalid poster ID in {path}")
    return values


def find_raw_dir(path):
    path = Path(path)
    for candidate in (path / "posterGlobal-max10/raw", path / "raw", path):
        if (candidate / "poster_id_list.txt").is_file():
            return candidate
    raise FileNotFoundError(f"No raw dataset under {path}")


def load_dataset(path, split="test", poster_ids=None, limit=None):
    """Match default posterGlobalDataset.process feature values and sample order.

    Read current raw data, not potentially stale processed .pt caches. Coordinates
    are not fed to inference; the position array supplies category/order only.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    raw = find_raw_dir(path)
    ids = _ids(raw / "poster_id_list.txt")
    selected = set(_ids(raw / f"{split}_poster_id.txt"))
    unknown = selected - set(ids)
    if unknown:
        raise ValueError(f"Split contains unknown IDs: {sorted(unknown)[:5]}")
    if poster_ids:
        requested = set(poster_ids)
        if requested - selected:
            raise ValueError(f"IDs not in {split} split: {sorted(requested - selected)}")
        selected &= requested
    positions = np.load(raw / "poster_panel_position.npy", allow_pickle=False)
    if positions.ndim != 3 or positions.shape[0] != len(ids) or positions.shape[2] != 5:
        raise ValueError("Expected numeric [poster count, panel count, 5] position array matching the ID list")

    sections = []
    for name in ids:
        metadata = json.loads((raw / "poster_meta" / f"{name}.json").read_text(encoding="utf-8-sig"))
        rows = list(metadata["section"].values())
        if not rows:
            raise ValueError(f"No metadata sections for {name}")
        sections.append(rows)
    # Deliberately include all IDs and all sections, even when inferring one poster.
    max_text = max(row["textlen"] for rows in sections for row in rows)
    max_figures = max(row["figures"] for rows in sections for row in rows)
    if max_text <= 0 or max_figures <= 0:
        raise ValueError("Dataset normalization maxima must be positive")

    posters = []
    for name, layout, rows in zip(ids, positions, sections):
        if name not in selected:
            continue
        labels = layout[layout[:, 0] != 5, 0]
        if not 1 <= len(labels) <= 10 or not np.all(np.isin(labels, np.arange(5))):
            raise ValueError(f"{name}: expected 1-10 panels with categories 0-4")
        features = {}
        for output, source, maximum in (
            ("text_len", "textlen", max_text), ("text_ratio", "textRatio", None),
            ("figure_count", "figures", max_figures), ("figure_ratio", "figRatio", None),
        ):
            values = np.asarray([row[source] for row in rows], dtype=np.float32)
            if maximum is not None:
                values = values / np.float32(maximum)
            # Original code converts float32 to float64 BEFORE multiplying by 10.
            values = np.rint(values.astype(np.float64) * 10)
            if not np.isfinite(values).all() or ((values < 0) | (values > 10)).any():
                raise ValueError(f"{name}: invalid {source} values")
            padded = np.zeros(10, dtype=np.int64)
            padded[:min(len(values), 10)] = values[:10].astype(np.int64)
            features[output] = padded
        panels = [dict(category=int(label), **{key: int(values[i]) for key, values in features.items()})
                  for i, label in enumerate(labels)]
        posters.append({"id": name, "panels": panels})
        if limit is not None and len(posters) >= limit:
            break
    if not posters:
        raise ValueError("No posters selected")
    return {"posters": posters}


def main():
    parser = argparse.ArgumentParser(description="Export evaluation dataset features to inference JSON")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--poster-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = load_dataset(args.dataset_dir, args.split, args.poster_id, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(document['posters'])} posters to {args.output}")


if __name__ == "__main__":
    main()
