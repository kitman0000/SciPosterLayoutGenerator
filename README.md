# Standalone Sci-PosterLayout inference

[](./outputs/render-check/cli/poster_00000.png)

## Install and run

```bash
python -m pip install .
sciposter-infer --config config.yaml --checkpoint best_model.pt --input examples/input.json --output outputs/layouts.json --device cpu
```

Use the config.yaml saved by the NEW training code and its matching checkpoint.
The simplified fusion layer has 6*d_model inputs; older feature-rich checkpoints
are incompatible and require retraining. Loading checks every parameter strictly.
examples/model.yaml is an example for backbone d_model=256, feedforward=2048,
4 layers; its dimensions already include the training-time 21/32 shrink.
Training YAML must contain explicit values (no interpolation or composition).

Optional flags: --npy-output outputs/layouts.npy, --batch-size 16,
--device auto (default), --sampling random --temperature 1.0 --seed 1.
Greedy sampling is the default. Random reproducibility assumes the same device,
software environment and batch partition; it is not bitwise identical to legacy sampling.

## Input

JSON: {"posters": [{"id": "poster-1", "panels": [...]}]}.
Each poster has a unique string ID and 1-10 panels in training order.
Each panel requires:

- category: 0-4 or Title, Introduction, Method, Result, Discussion.
- text_len, text_ratio, figure_count, figure_ratio: quantized integers 0-10.

For text_len and figure_count, normalize using the TRAINING maximum, multiply
by 10 and round. For ratios, multiply by 10 and round. Do not estimate new
normalization maxima from inference inputs. Panel IDs are generated sequentially.
No original text/images, ground-truth boxes, AOV labels, type annotations or
keyword vectors are needed. See examples/input.json for a complete input.

## Output

JSON preserves IDs, panel order and categories. bbox is normalized [cx,cy,w,h].
A PAD/EOS coordinate prediction yields valid=false and a zero box.
No additional clipping is applied. Multiply x/width by canvas width and y/height
by canvas height for pixels. NPY output is float32 [N,10,5], with rows
[category,cx,cy,w,h]; absent or invalid panels are [5,0,0,0,0].

## Compatibility

Supported: BART, 10 panels, 32 linear bins, shared bbox vocabulary, c-w-h-x-y,
default positional embeddings, sort_by=none. Unsupported configurations fail.
The historical encoder residual calculation is preserved. Only the C decoder
executes because A/O/V predictions do not influence it; auxiliary weights are
retained for strict checkpoint validation. Encoder memory is cached per batch.

## Tests and packaging

```bash
python inference/sciposter_inference/cli.py --config PATH_TO_CONFIG --checkpoint PATH_TO_CHECKPOINT --image-dir inference/examples/png --input inference/examples/input.json
```

Download the model checkpoint here
https://huggingface.co/kitman0000/SciPosterLayoutGenerator

Please cite us if this work helps you.

```
@article{ZHONG2025111507,
title = {Scientific poster generation: A new dataset and approach},
journal = {Pattern Recognition},
volume = {164},
pages = {111507},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111507},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325001670},
author = {Xinyi Zhong and Zusheng Tan and Jing Li and Shen Gao and Jing Ma and Shanshan Feng and Billy Chiu}
}
```

