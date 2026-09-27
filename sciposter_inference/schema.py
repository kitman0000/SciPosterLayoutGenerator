"""JSON contract and training-config conversion (no tensor dependencies)."""

from dataclasses import dataclass

LABELS = ["Title", "Introduction", "Method", "Result", "Discussion"]
STAT_FIELDS = ("text_len", "text_ratio", "figure_count", "figure_ratio")


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class ModelConfig:
    # Dimensions are AFTER the training code's 21/32 shrink.
    d_model: int
    nhead: int
    dim_feedforward: int
    num_layers: int
    dropout: float = 0.1
    activation: str = "relu"
    multi_decoder: bool = True

    def __post_init__(self):
        for name in ("d_model", "nhead", "dim_feedforward", "num_layers"):
            integer(getattr(self, name), 1, 100000, name)
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not isinstance(self.dropout, (float, int)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("Only relu/gelu checkpoints are supported")
        for name in ("multi_decoder",):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("Config must be a mapping")
        if "backbone" not in raw:
            return cls(**raw)
        data, dataset, model = raw["data"], raw["dataset"], raw["model"]
        expected = {
            "num_bin_bboxes": 32, "bbox_quantization": "linear",
            "shared_bbox_vocab": "xywh", "var_order": "c-w-h-x-y",
            "pad_until_max": True, "special_tokens": ["pad", "bos", "eos", "mask"],
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"Unsupported data.{key}: expected {value!r}")
        if dataset.get("max_seq_length") != 10 or not dataset.get("_target_", "").endswith(".posterGlobalDataset"):
            raise ValueError("Only posterGlobalDataset with max_seq_length=10 is supported")
        if model.get("_target_") != "trainer.models.bart.BART":
            raise ValueError("Only BART checkpoints are supported")
        extra = {k: v for k, v in raw.get("extra_model_params", {}).items() if v is not None and v != "???"}
        options = dict(model, **extra)
        if options.get("use_size_position_decoders", False):
            raise ValueError("The unfinished size/position decoder is unsupported")
        if options.get("pos_emb", "default") != "default" or options.get("sort_by", "none") not in ("none", "None", None):
            raise ValueError("Only default positional embeddings and sort_by=none are supported")
        if raw["backbone"].get("norm") is not None:
            raise ValueError("Encoder final normalization is unsupported")
        layer = raw["backbone"]["encoder_layer"]
        if layer.get("_target_") != "trainer.models.transformer_utils.Block":
            raise ValueError("Unsupported encoder layer")
        if not layer.get("norm_first", False) or not layer.get("batch_first", False) or layer.get("timestep_type") is not None:
            raise ValueError("Expected batch_first/norm_first encoder without timestep conditioning")
        return cls(
            d_model=int(layer["d_model"] * 21 / 32), nhead=layer["nhead"],
            dim_feedforward=int(layer["dim_feedforward"] * 21 / 32),
            num_layers=raw["backbone"]["num_layers"], dropout=layer.get("dropout", 0.1),
            activation=layer.get("activation", "relu"),
            **{k: options.get(k, default) for k, default in (
                ("multi_decoder", True),)},
        )


def validate_input(document, config):
    """Return canonical posters; require all learned features, never invent embeddings."""
    if not isinstance(document, dict) or not isinstance(document.get("posters"), list) or not document["posters"]:
        raise ValueError("Input must contain a nonempty posters list")
    result, ids = [], set()
    for index, poster in enumerate(document["posters"]):
        if not isinstance(poster, dict):
            raise ValueError(f"posters[{index}] must be an object")
        name = poster.get("id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Each poster needs a unique, nonempty string id")
        ids.add(name)
        panels = poster.get("panels")
        if not isinstance(panels, list) or not 1 <= len(panels) <= 10:
            raise ValueError(f"{name}: expected 1 to 10 panels")
        normalized = []
        for i, panel in enumerate(panels):
            if not isinstance(panel, dict):
                raise ValueError(f"{name}.panels[{i}] must be an object")
            prefix = f"{name}.panels[{i}]"
            label = panel.get("category")
            if isinstance(label, str) and label in LABELS:
                label = LABELS.index(label)
            item = {"category": integer(label, 0, 4, f"{prefix}.category")}
            for key in STAT_FIELDS:
                item[key] = integer(panel.get(key), 0, 10, f"{prefix}.{key}")
            normalized.append(item)
        result.append({"id": name, "panels": normalized})
    return result
