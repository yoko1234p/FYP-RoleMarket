"""
Experiment #11v2 Hybrid Transformer 模型

此模型架構與 kaggle_train_lulu_exp11v2.py 訓練的模型兼容。

架構特點:
- seq_embedding: 時間序列 embedding
- static_fc: 2層 (781 → 256 → 128)
- output_fc: 3層 (192 → 128 → 64 → 1)
- nhead: 8 (vs 4 in legacy)
- seq_length: 4

性能指標 (目標):
- R² > 0.60
- MAE < 350

Author: Developer (James)
Date: 2025-11-11
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncodingExp11v2(nn.Module):
    """Positional Encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class HybridTransformerExp11v2(nn.Module):
    """
    Experiment #11v2 架構的 Hybrid Transformer。

    差異（與 Legacy 相比）：
    - seq_embedding 而非 ts_embedding
    - static_fc: 2層 (781 → 256 → 128)
    - output_fc: 3層 (192 → 128 → 64 → 1)
    - nhead: 8（更多注意力頭）
    - 使用 mean pooling 而非 CLS token
    """

    def __init__(
        self,
        seq_len: int = 4,
        static_dim: int = 781,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.seq_len = seq_len
        self.static_dim = static_dim
        self.d_model = d_model

        # ============================================================
        # 時間序列分支
        # ============================================================
        self.seq_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncodingExp11v2(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ============================================================
        # 靜態特徵分支 (2層)
        # ============================================================
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ============================================================
        # 輸出層 (3層)
        # ============================================================
        combined_dim = d_model + 128  # 64 + 128 = 192
        self.output_fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        logger.info("HybridTransformerExp11v2 initialized (Experiment #11v2 architecture)")
        logger.info(f"  seq_len: {seq_len}, static_dim: {static_dim}")
        logger.info(f"  d_model: {d_model}, nhead: {nhead}")

    def forward(self, seq_input: torch.Tensor, static_input: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            seq_input: (batch_size, seq_len, 1) - 時間序列輸入
            static_input: (batch_size, static_dim) - 靜態特徵

        Returns:
            output: (batch_size, 1) - 銷量預測
        """
        # ============================================================
        # 時間序列分支
        # ============================================================
        # 1. Embedding
        seq_embedded = self.seq_embedding(seq_input)  # (batch, seq_len, d_model)

        # 2. Positional Encoding
        seq_embedded = self.pos_encoder(seq_embedded)

        # 3. Transformer Encoder
        seq_output = self.transformer_encoder(seq_embedded)  # (batch, seq_len, d_model)

        # 4. Mean Pooling (取序列平均)
        seq_pooled = seq_output.mean(dim=1)  # (batch, d_model)

        # ============================================================
        # 靜態特徵分支
        # ============================================================
        static_output = self.static_fc(static_input)  # (batch, 128)

        # ============================================================
        # Combine and Output
        # ============================================================
        combined = torch.cat([seq_pooled, static_output], dim=1)  # (batch, 192)
        output = self.output_fc(combined)  # (batch, 1)

        return output
