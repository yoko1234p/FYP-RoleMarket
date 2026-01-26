"""
Hybrid Transformer 模型

使用 Transformer 架構替代 LSTM 處理時間序列特徵。
結合 Self-Attention 機制與靜態特徵進行銷量預測。

優勢:
- Self-Attention 可捕捉長距離依賴
- 並行計算，訓練速度更快
- 更強的表達能力

Author: Product Manager (John)
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.

    為序列中的每個位置添加位置信息。
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding 維度
            max_len: 最大序列長度
            dropout: Dropout 率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 創建位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridTransformer(nn.Module):
    """
    Hybrid Transformer 模型，結合時間序列和靜態特徵。

    架構:
    1. 時間序列分支:
       - Input Embedding
       - Positional Encoding
       - Transformer Encoder
       - Pooling (取 CLS token 或 mean pooling)

    2. 靜態特徵分支:
       - Fully Connected layers

    3. Fusion:
       - Concatenate 兩個分支
       - Final prediction layers
    """

    def __init__(
        self,
        ts_input_dim: int = 1,          # 時間序列輸入維度
        static_input_dim: int = 772,    # 靜態特徵維度（CLIP 768 + 季節 4）
        d_model: int = 64,              # Transformer embedding 維度
        nhead: int = 4,                 # Multi-head attention 數量
        num_encoder_layers: int = 2,    # Transformer encoder 層數
        dim_feedforward: int = 256,     # Feedforward 維度
        static_hidden_dim: int = 256,   # Static 分支 hidden dim
        fusion_hidden_dim: int = 64,    # Fusion 層 hidden dim
        dropout: float = 0.3,           # Dropout 率
        max_seq_len: int = 10           # 最大序列長度
    ):
        """
        初始化 Hybrid Transformer 模型。

        Args:
            ts_input_dim: 時間序列輸入維度
            static_input_dim: 靜態特徵輸入維度
            d_model: Transformer embedding 維度（必須能被 nhead 整除）
            nhead: Multi-head attention 頭數
            num_encoder_layers: Encoder 層數
            dim_feedforward: Feedforward 網絡維度
            static_hidden_dim: Static 分支 hidden dimension
            fusion_hidden_dim: Fusion 層 hidden dimension
            dropout: Dropout 率
            max_seq_len: 最大序列長度
        """
        super(HybridTransformer, self).__init__()

        self.ts_input_dim = ts_input_dim
        self.static_input_dim = static_input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

        # ============================================================
        # 時間序列分支 (Transformer Branch)
        # ============================================================

        # Input Embedding: 將 ts_input_dim 投影到 d_model
        self.input_embedding = nn.Linear(ts_input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用 (batch, seq, feature) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # CLS token (可選，用於聚合序列信息)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ============================================================
        # 靜態特徵分支 (Static Features Branch)
        # ============================================================
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(static_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim, d_model),  # 輸出維度與 Transformer 一致
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )

        # ============================================================
        # Fusion 層 (Combine Transformer + Static features)
        # ============================================================
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(fusion_hidden_dim, 1)  # 輸出: 預測銷量
        )

        # 初始化權重
        self._init_weights()

        logger.info("HybridTransformer initialized")
        logger.info(f"  Time-series input dim: {ts_input_dim}")
        logger.info(f"  Static input dim: {static_input_dim}")
        logger.info(f"  d_model: {d_model}")
        logger.info(f"  nhead: {nhead}")
        logger.info(f"  num_encoder_layers: {num_encoder_layers}")
        logger.info(f"  dim_feedforward: {dim_feedforward}")
        logger.info(f"  static_hidden_dim: {static_hidden_dim}")
        logger.info(f"  fusion_hidden_dim: {fusion_hidden_dim}")
        logger.info(f"  dropout: {dropout}")

    def _init_weights(self):
        """初始化模型權重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x_ts: 時間序列輸入 (batch_size, seq_len, ts_input_dim)
            x_static: 靜態特徵輸入 (batch_size, static_input_dim)

        Returns:
            sales_pred: 預測銷量 (batch_size, 1)
        """
        batch_size = x_ts.size(0)

        # ============================================================
        # Transformer 分支: 處理時間序列
        # ============================================================

        # 1. Input Embedding
        # x_ts: (batch_size, seq_len, ts_input_dim) -> (batch_size, seq_len, d_model)
        x_embedded = self.input_embedding(x_ts)

        # 2. Add CLS token (optional)
        # cls_tokens: (batch_size, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x_embedded: (batch_size, seq_len+1, d_model)
        x_embedded = torch.cat([cls_tokens, x_embedded], dim=1)

        # 3. Positional Encoding
        x_embedded = self.pos_encoder(x_embedded)

        # 4. Transformer Encoder
        # transformer_out: (batch_size, seq_len+1, d_model)
        transformer_out = self.transformer_encoder(x_embedded)

        # 5. 取 CLS token 的輸出作為序列特徵表示
        # transformer_features: (batch_size, d_model)
        transformer_features = transformer_out[:, 0, :]  # CLS token

        # 備選方案: 使用 mean pooling
        # transformer_features = transformer_out.mean(dim=1)

        # ============================================================
        # 靜態特徵分支
        # ============================================================
        static_features = self.static_fc(x_static)

        # ============================================================
        # Fusion 層: 結合兩個分支
        # ============================================================
        # Concatenate: (batch_size, d_model + d_model)
        combined = torch.cat([transformer_features, static_features], dim=1)

        # 最終預測
        sales_pred = self.fusion(combined)

        return sales_pred

    def get_model_summary(self) -> Dict:
        """
        獲取模型摘要信息。

        Returns:
            Dictionary with model architecture info
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'HybridTransformer',
            'ts_input_dim': self.ts_input_dim,
            'static_input_dim': self.static_input_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


class HybridTransformerV2(nn.Module):
    """
    Hybrid Transformer V2 - 簡化版本

    不使用 CLS token，直接對序列進行 mean pooling。
    更適合小數據集。
    """

    def __init__(
        self,
        ts_input_dim: int = 1,
        static_input_dim: int = 772,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        static_hidden_dim: int = 256,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
        max_seq_len: int = 10
    ):
        """初始化簡化版 Transformer。"""
        super(HybridTransformerV2, self).__init__()

        self.ts_input_dim = ts_input_dim
        self.static_input_dim = static_input_dim
        self.d_model = d_model

        # Input Embedding
        self.input_embedding = nn.Linear(ts_input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
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

        # Static branch
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(static_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim, d_model),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(fusion_hidden_dim, 1)
        )

        logger.info("HybridTransformerV2 (simplified) initialized")

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor
    ) -> torch.Tensor:
        """前向傳播（使用 mean pooling）。"""
        # Embed and encode
        x_embedded = self.input_embedding(x_ts)
        x_embedded = self.pos_encoder(x_embedded)
        transformer_out = self.transformer_encoder(x_embedded)

        # Mean pooling (不使用 CLS token)
        transformer_features = transformer_out.mean(dim=1)

        # Static features
        static_features = self.static_fc(x_static)

        # Fusion
        combined = torch.cat([transformer_features, static_features], dim=1)
        sales_pred = self.fusion(combined)

        return sales_pred

    def get_model_summary(self) -> Dict:
        """獲取模型摘要。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'HybridTransformerV2',
            'ts_input_dim': self.ts_input_dim,
            'static_input_dim': self.static_input_dim,
            'd_model': self.d_model,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


def demo():
    """Demo function."""
    print("\n" + "="*80)
    print("Hybrid Transformer Model Demo")
    print("="*80 + "\n")

    # 創建模型
    model = HybridTransformer(
        ts_input_dim=1,
        static_input_dim=772,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        static_hidden_dim=256,
        fusion_hidden_dim=64,
        dropout=0.3,
        max_seq_len=10
    )

    # 顯示模型摘要
    summary = model.get_model_summary()
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 測試前向傳播
    print("\n" + "="*80)
    print("Testing Forward Pass")
    print("="*80 + "\n")

    batch_size = 8
    seq_len = 4  # 4 季的 Trends 歷史
    ts_input_dim = 1
    static_input_dim = 772

    # 創建隨機測試數據
    x_ts = torch.randn(batch_size, seq_len, ts_input_dim)
    x_static = torch.randn(batch_size, static_input_dim)

    print(f"Input shapes:")
    print(f"  x_ts: {x_ts.shape}")
    print(f"  x_static: {x_static.shape}")

    # 前向傳播
    model.eval()
    with torch.no_grad():
        predictions = model(x_ts, x_static)

    print(f"\nOutput shape:")
    print(f"  predictions: {predictions.shape}")
    print(f"\nSample predictions:")
    print(f"  {predictions.squeeze()[:5]}")

    # 測試簡化版本
    print("\n" + "="*80)
    print("Testing HybridTransformerV2 (Simplified)")
    print("="*80 + "\n")

    model_v2 = HybridTransformerV2(
        ts_input_dim=1,
        static_input_dim=772,
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )

    v2_summary = model_v2.get_model_summary()
    print("Model V2 Summary:")
    for key, value in v2_summary.items():
        print(f"  {key}: {value}")

    # 測試前向傳播
    model_v2.eval()
    with torch.no_grad():
        v2_predictions = model_v2(x_ts, x_static)

    print(f"\nV2 Output shape:")
    print(f"  predictions: {v2_predictions.shape}")

    # 比較參數量
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80 + "\n")

    print(f"HybridTransformer (with CLS):  {summary['total_params']:,} params")
    print(f"HybridTransformerV2 (no CLS):   {v2_summary['total_params']:,} params")
    print(f"Difference: {summary['total_params'] - v2_summary['total_params']:,} params")


if __name__ == '__main__':
    demo()
