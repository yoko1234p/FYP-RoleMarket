"""
Hybrid LSTM 模型

結合時間序列特徵（Google Trends）和靜態特徵（CLIP embeddings + 季節）
進行銷量預測的混合 LSTM 模型。

模型架構:
- LSTM 分支: 處理時間序列特徵（Google Trends 歷史）
- Static 分支: 處理靜態特徵（CLIP embeddings + 季節 one-hot）
- Fusion 層: 結合兩個分支的輸出進行最終預測

Author: Product Manager (John)
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridLSTM(nn.Module):
    """
    Hybrid LSTM 模型，結合時間序列和靜態特徵。

    輸入:
    - x_ts: 時間序列特徵 (batch_size, seq_len, ts_input_dim)
           例如: Google Trends 過去 4 季的歷史 (batch_size, 4, 1)
    - x_static: 靜態特徵 (batch_size, static_input_dim)
               例如: CLIP embeddings (768) + 季節 one-hot (4) = 772

    輸出:
    - sales_pred: 預測銷量 (batch_size, 1)
    """

    def __init__(
        self,
        ts_input_dim: int = 1,          # 時間序列特徵維度（Trends 分數）
        static_input_dim: int = 772,    # 靜態特徵維度（CLIP 768 + 季節 4）
        lstm_hidden_dim: int = 128,     # LSTM hidden dimension
        lstm_num_layers: int = 2,       # LSTM 層數
        static_hidden_dim: int = 256,   # Static 分支的 hidden dimension
        fusion_hidden_dim: int = 64,    # Fusion 層的 hidden dimension
        dropout: float = 0.3            # Dropout 率
    ):
        """
        初始化 Hybrid LSTM 模型。

        Args:
            ts_input_dim: 時間序列輸入維度
            static_input_dim: 靜態特徵輸入維度
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: LSTM 層數
            static_hidden_dim: Static 分支 hidden dimension
            fusion_hidden_dim: Fusion 層 hidden dimension
            dropout: Dropout 率
        """
        super(HybridLSTM, self).__init__()

        self.ts_input_dim = ts_input_dim
        self.static_input_dim = static_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # ============================================================
        # LSTM 分支 (Time-Series Branch)
        # ============================================================
        self.lstm = nn.LSTM(
            input_size=ts_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0  # Dropout 只在多層時使用
        )

        # ============================================================
        # 靜態特徵分支 (Static Features Branch)
        # ============================================================
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(static_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim, lstm_hidden_dim),  # 輸出維度與 LSTM 一致
            nn.ReLU(),
            nn.BatchNorm1d(lstm_hidden_dim),
        )

        # ============================================================
        # Fusion 層 (Combine LSTM + Static features)
        # ============================================================
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden_dim + lstm_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # 較低的 dropout
            nn.Linear(fusion_hidden_dim, 1)  # 輸出: 預測銷量
        )

        logger.info("HybridLSTM initialized")
        logger.info(f"  Time-series input dim: {ts_input_dim}")
        logger.info(f"  Static input dim: {static_input_dim}")
        logger.info(f"  LSTM hidden dim: {lstm_hidden_dim}")
        logger.info(f"  LSTM num layers: {lstm_num_layers}")
        logger.info(f"  Static hidden dim: {static_hidden_dim}")
        logger.info(f"  Fusion hidden dim: {fusion_hidden_dim}")
        logger.info(f"  Dropout: {dropout}")

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
        # ============================================================
        # LSTM 分支: 處理時間序列
        # ============================================================
        # lstm_out: (batch_size, seq_len, hidden_dim)
        # hn: (num_layers, batch_size, hidden_dim)
        # cn: (num_layers, batch_size, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x_ts)

        # 取最後一層的 hidden state 作為時間序列特徵表示
        # hn[-1]: (batch_size, hidden_dim)
        lstm_features = hn[-1]

        # ============================================================
        # 靜態特徵分支
        # ============================================================
        static_features = self.static_fc(x_static)

        # ============================================================
        # Fusion 層: 結合兩個分支
        # ============================================================
        # Concatenate: (batch_size, hidden_dim + hidden_dim)
        combined = torch.cat([lstm_features, static_features], dim=1)

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
            'model_name': 'HybridLSTM',
            'ts_input_dim': self.ts_input_dim,
            'static_input_dim': self.static_input_dim,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_num_layers': self.lstm_num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


class HybridGRU(nn.Module):
    """
    Hybrid GRU 模型（備案）。

    與 HybridLSTM 類似，但使用 GRU 替代 LSTM。
    GRU 較簡單，訓練速度更快，但表達能力可能略弱。
    """

    def __init__(
        self,
        ts_input_dim: int = 1,
        static_input_dim: int = 772,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        static_hidden_dim: int = 256,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """初始化 Hybrid GRU 模型。"""
        super(HybridGRU, self).__init__()

        self.ts_input_dim = ts_input_dim
        self.static_input_dim = static_input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers

        # GRU 分支 (替代 LSTM)
        self.gru = nn.GRU(
            input_size=ts_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # 靜態特徵分支（與 LSTM 版本相同）
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(static_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim, gru_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gru_hidden_dim),
        )

        # Fusion 層
        self.fusion = nn.Sequential(
            nn.Linear(gru_hidden_dim + gru_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(fusion_hidden_dim, 1)
        )

        logger.info("HybridGRU initialized")
        logger.info(f"  Time-series input dim: {ts_input_dim}")
        logger.info(f"  Static input dim: {static_input_dim}")
        logger.info(f"  GRU hidden dim: {gru_hidden_dim}")
        logger.info(f"  GRU num layers: {gru_num_layers}")

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor
    ) -> torch.Tensor:
        """前向傳播。"""
        # GRU 分支
        gru_out, hn = self.gru(x_ts)
        gru_features = hn[-1]  # 最後一層的 hidden state

        # 靜態特徵分支
        static_features = self.static_fc(x_static)

        # Fusion
        combined = torch.cat([gru_features, static_features], dim=1)
        sales_pred = self.fusion(combined)

        return sales_pred

    def get_model_summary(self) -> Dict:
        """獲取模型摘要信息。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'HybridGRU',
            'ts_input_dim': self.ts_input_dim,
            'static_input_dim': self.static_input_dim,
            'gru_hidden_dim': self.gru_hidden_dim,
            'gru_num_layers': self.gru_num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


def demo():
    """Demo function."""
    print("\n" + "="*80)
    print("Hybrid LSTM Model Demo")
    print("="*80 + "\n")

    # 創建模型
    model = HybridLSTM(
        ts_input_dim=1,
        static_input_dim=772,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        static_hidden_dim=256,
        fusion_hidden_dim=64,
        dropout=0.3
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

    # 測試 GRU 模型
    print("\n" + "="*80)
    print("Testing Hybrid GRU Model")
    print("="*80 + "\n")

    gru_model = HybridGRU(
        ts_input_dim=1,
        static_input_dim=772,
        gru_hidden_dim=128,
        gru_num_layers=2
    )

    gru_summary = gru_model.get_model_summary()
    print("GRU Model Summary:")
    for key, value in gru_summary.items():
        print(f"  {key}: {value}")

    # 測試 GRU 前向傳播
    gru_model.eval()
    with torch.no_grad():
        gru_predictions = gru_model(x_ts, x_static)

    print(f"\nGRU Output shape:")
    print(f"  predictions: {gru_predictions.shape}")


if __name__ == '__main__':
    demo()
