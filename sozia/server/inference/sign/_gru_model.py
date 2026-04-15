"""ActionGRU — inference-only copy of the GRU classifier from sozia-research.

This avoids a runtime dependency on sozia-research while remaining checkpoint-
compatible with ``tsl_recognition.models.gru.ActionGRU``.  The architecture
and weight names match exactly so ``state_dict`` loads without remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_SIZES = {
    "small":  {"gru_hidden": 256,  "gru_layers": 4, "fc": [512, 256]},
    "large":  {"gru_hidden": 512,  "gru_layers": 5, "fc": [1024, 512]},
    "xlarge": {"gru_hidden": 1024, "gru_layers": 6, "fc": [2048, 1024]},
}


class ActionGRU(nn.Module):
    """GRU-based sign language classifier (inference mirror)."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        model_size: str = "small",
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        cfg = _SIZES.get(model_size, _SIZES["small"])

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=cfg["gru_hidden"],
            num_layers=cfg["gru_layers"],
            batch_first=True,
            dropout=dropout if cfg["gru_layers"] > 1 else 0,
            bidirectional=False,
        )

        fc_sizes = cfg["fc"]
        self.head = nn.Sequential(
            nn.Linear(cfg["gru_hidden"], fc_sizes[0]),
            nn.BatchNorm1d(fc_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.BatchNorm1d(fc_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_sizes[1], num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True,
            )
        else:
            gru_out, _ = self.gru(x)

        if lengths is not None:
            batch_idx = torch.arange(x.size(0), device=x.device)
            encoded = gru_out[batch_idx, lengths - 1, :]
        else:
            encoded = gru_out[:, -1, :]

        return self.head(encoded)
