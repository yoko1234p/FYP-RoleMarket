"""
舊版 Hybrid Transformer 模型（用於載入已訓練模型）

此模型架構與 2025-10-28 訓練的模型兼容。
新版本請參考 hybrid_transformer_model.py。

Author: Developer (James)
Date: 2025-11-11
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncodingLegacy(nn.Module):
    """Positional Encoding for Transformer (legacy format)."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super(PositionalEncodingLegacy, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # IMPORTANT: Legacy model uses shape [max_len, 1, d_model] not [1, max_len, d_model]
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # pe: (max_len, 1, d_model)
        # Need to slice pe to match seq_len and broadcast to batch
        seq_len = x.size(1)
        # pe[:seq_len, :, :] -> (seq_len, 1, d_model)
        # Squeeze middle dim and add batch dim: -> (1, seq_len, d_model)
        pe_slice = self.pe[:seq_len, 0, :].unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pe_slice
        return self.dropout(x)


class HybridTransformerLegacy(nn.Module):
    """
    舊版 Hybrid Transformer（與已訓練模型兼容）。

    架構差異（與新版本相比）：
    - static_fc: 3層 Linear (781 → 256 → 128 → 64)
    - fusion_fc: 3層 Linear (128 → 128 → 64 → 1)
    - 使用 ts_embedding 而非 input_embedding
    """

    def __init__(
        self,
        ts_input_dim: int = 1,
        static_input_dim: int = 781,  # 舊版使用 781
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        static_hidden_dim: int = 256,
        static_hidden_dim_2: int = 128,  # 舊版額外的層
        fusion_hidden_dim: int = 128,     # 舊版: 128 而非 64
        fusion_hidden_dim_2: int = 64,    # 舊版額外的層
        dropout: float = 0.3,
        max_seq_len: int = 4  # 舊版使用 4
    ):
        super(HybridTransformerLegacy, self).__init__()

        self.ts_input_dim = ts_input_dim
        self.static_input_dim = static_input_dim
        self.d_model = d_model

        # ============================================================
        # 時間序列分支 (Transformer Branch)
        # ============================================================
        self.ts_embedding = nn.Linear(ts_input_dim, d_model)
        self.pos_encoder = PositionalEncodingLegacy(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ============================================================
        # 靜態特徵分支 (層索引必須匹配儲存的模型)
        # 0, 3, 6 是 Linear 層（與儲存模型的 keys 匹配）
        # ============================================================
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),      # 0: 781 → 256
            nn.ReLU(),                                           # 1
            nn.Dropout(dropout),                                 # 2
            nn.Linear(static_hidden_dim, static_hidden_dim_2),   # 3: 256 → 128
            nn.ReLU(),                                           # 4
            nn.Dropout(dropout),                                 # 5
            nn.Linear(static_hidden_dim_2, d_model),             # 6: 128 → 64
            nn.ReLU(),                                           # 7
        )

        # ============================================================
        # Fusion 層 (層索引必須匹配儲存的模型)
        # 0, 3, 6 是 Linear 層（與儲存模型的 keys 匹配）
        # ============================================================
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model + d_model, fusion_hidden_dim),  # 0: 128 → 128
            nn.ReLU(),                                        # 1
            nn.Dropout(dropout),                              # 2
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim_2),  # 3: 128 → 64
            nn.ReLU(),                                        # 4
            nn.Dropout(dropout),                              # 5
            nn.Linear(fusion_hidden_dim_2, 1)                 # 6: 64 → 1
        )

        logger.info("HybridTransformerLegacy initialized (compatible with old trained model)")

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x_ts: (batch_size, seq_len, ts_input_dim)
            x_static: (batch_size, static_input_dim)

        Returns:
            predictions: (batch_size, 1)
        """
        batch_size = x_ts.size(0)

        # ============================================================
        # 時間序列分支
        # ============================================================
        # 1. Embedding
        ts_embedded = self.ts_embedding(x_ts)  # (batch, seq_len, d_model)

        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        ts_embedded = torch.cat([cls_tokens, ts_embedded], dim=1)  # (batch, seq_len+1, d_model)

        # 3. Positional Encoding
        ts_embedded = self.pos_encoder(ts_embedded)

        # 4. Transformer Encoder
        ts_encoded = self.transformer_encoder(ts_embedded)  # (batch, seq_len+1, d_model)

        # 5. Extract CLS token
        ts_features = ts_encoded[:, 0, :]  # (batch, d_model)

        # ============================================================
        # 靜態特徵分支
        # ============================================================
        static_features = self.static_fc(x_static)  # (batch, d_model)

        # ============================================================
        # Fusion
        # ============================================================
        combined = torch.cat([ts_features, static_features], dim=1)  # (batch, 2*d_model)
        output = self.fusion_fc(combined)  # (batch, 1)

        return output
