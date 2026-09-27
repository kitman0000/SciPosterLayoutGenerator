"""Checkpoint-compatible BART weights and the layout-only inference graph."""

import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.activation = F.relu if cfg.activation == "relu" else F.gelu

    def forward(self, x):
        # Historical encoder uses the NORMALIZED tensor as its residual.
        # nn.TransformerEncoderLayer(norm_first=True) is not equivalent.
        x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x, x, x, need_weights=False)[0])
        return x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(51, dim))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, length):
        return self.pos_emb[:length].unsqueeze(0)


class LayoutModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = Encoder(cfg)
        # Keep every original parameter so checkpoint validation remains strict.
        # Auxiliary branches cannot affect C logits and need not run at inference.
        for name in ("c", "a", "o", "v"):
            decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout, activation=cfg.activation, batch_first=True, norm_first=True,
                ), cfg.num_layers,
            )
            setattr(self, name + "_decoder", decoder)
        for name, size in (
            ("input_cat_emb", 42), ("target_cat_emb", 42), ("action_emb", 4),
            ("unique_id_emb", 11), ("value_emb", 5), ("cat_emb_additional", 42),
            ("tl_emb_additional", 11), ("tr_emb_additional", 11),
            ("fc_emb_additional", 11), ("fr_emb_additional", 11),
            ("uid_emb_additional", 11),
        ):
            setattr(self, name, nn.Embedding(size, d))
        self.input_pos_emb = PositionEmbedding(d)
        self.target_pos_emb = PositionEmbedding(d)
        self.self_cond_fuser = nn.Sequential(nn.Linear(6 * d, d), nn.ReLU())
        self.drop = nn.Dropout(0.1)
        for name, size in (("c", 42), ("a", 4), ("o", 42), ("v", 5)):
            setattr(self, name + "_head", nn.Sequential(nn.LayerNorm(d), nn.Linear(d, size, bias=False)))

    def encode(self, seq, features):
        parts = [self.input_cat_emb(seq)]
        names = ("tl", "tr", "fc", "fr", "uid")
        for i, name in enumerate(names):
            parts.append(getattr(self, name + "_emb_additional")(features[:, i]))
        x = self.self_cond_fuser(torch.cat(parts, dim=-1))
        return self.encoder(self.drop(x + self.input_pos_emb(seq.shape[1])))

    def decode_next(self, target, memory):
        x = self.target_cat_emb(target) + self.target_pos_emb(target.shape[1])
        mask = torch.full((target.shape[1], target.shape[1]), float("-inf"), device=x.device).triu(1)
        return self.c_head(self.c_decoder(x, memory, tgt_mask=mask))[:, -1]


def load_checkpoint(path, config, device):
    # Restricted tensor-only loading; no dependency on pickled trainer classes.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Expected a tensor state_dict saved by save_model")
    prefix = "model.module."
    if all(k.startswith(prefix) for k in state):
        state = {k[len(prefix):]: v for k, v in state.items()}
    model = LayoutModel(config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
