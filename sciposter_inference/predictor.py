"""Feature preparation and autoregressive category-conditioned generation."""

import torch

from .schema import STAT_FIELDS, validate_input

PAD, BOS, EOS, MASK = 38, 39, 40, 41


def prepare_batch(posters, device):
    count = len(posters)
    seq = torch.full((count, 51), PAD, dtype=torch.long, device=device)
    seq[:, 0] = BOS
    features = torch.zeros((count, 5, 10), dtype=torch.long, device=device)
    features[:, 4] = 10  # padded unique ids
    for b, poster in enumerate(posters):
        for i, panel in enumerate(poster["panels"]):
            start = 1 + 5 * i
            seq[b, start] = panel["category"]
            seq[b, start + 1:start + 5] = MASK
            features[b, :4, i] = torch.tensor([panel[k] for k in STAT_FIELDS], device=device)
            features[b, 4, i] = i
    features = features.repeat_interleave(5, dim=2)
    features = torch.cat((features.new_full((count, 5, 1), 10), features), dim=2)
    return seq, features


@torch.inference_mode()
def generate(model, document, batch_size=16, sampling="greedy", temperature=1.0, seed=1):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampling not in ("greedy", "random"):
        raise ValueError("sampling must be greedy or random")
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be positive and finite")
    posters = validate_input(document, model.cfg)
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    result = []
    for offset in range(0, len(posters), batch_size):
        items = posters[offset:offset + batch_size]
        seq, features = prepare_batch(items, device)
        memory = model.encode(seq, features)  # cached once per batch
        target = seq[:, :1].clone()
        for i in range(50):
            fixed = seq[:, i + 1]
            if (fixed != MASK).all():
                predicted = fixed
            else:
                logits = model.decode_next(target, memory)
                # Original bbox token mask allows bins plus PAD/EOS.
                allowed = torch.zeros(42, dtype=torch.bool, device=device)
                allowed[6:39] = True
                allowed[EOS] = True
                logits = logits.masked_fill(~allowed, float("-inf"))
                if sampling == "greedy":
                    predicted = logits.argmax(dim=-1)
                else:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    predicted = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
                predicted = torch.where(fixed == MASK, predicted, fixed)
            target = torch.cat((target, predicted.unsqueeze(1)), dim=1)
        tokens = target[:, 1:].reshape(-1, 10, 5).cpu()
        for poster, rows in zip(items, tokens):
            panels = []
            for panel, row in zip(poster["panels"], rows):
                w, h, x, y = (row[1:] - 6).tolist()
                valid = all(0 <= v < 32 for v in (w, h, x, y))
                bbox = [x / 32, y / 32, (w + 1) / 32, (h + 1) / 32] if valid else [0.0] * 4
                panels.append({"category": panel["category"], "bbox": bbox, "valid": valid})
            result.append({"id": poster["id"], "panels": panels})
    return {"bbox_format": "cxcywh", "coordinate_space": "normalized", "posters": result}
