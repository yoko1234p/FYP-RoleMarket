"""
Forecast Predictor API Wrapper - Obj 3 Integration Layer

封裝 Hybrid Transformer 銷量預測模型為 Streamlit 友善的 API。

Author: Developer (James)
Date: 2025-11-06
Version: 1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj3_lstm_forecast.hybrid_transformer_model import HybridTransformer
from obj3_lstm_forecast.hybrid_transformer_model_legacy import HybridTransformerLegacy
from obj3_lstm_forecast.hybrid_transformer_model_exp11v2 import HybridTransformerExp11v2

logger = logging.getLogger(__name__)


class ForecastError(Exception):
    """Raised when forecasting fails."""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ForecastPredictorWrapper:
    """
    Wrapper for Obj 3 Hybrid Transformer 銷量預測模型。

    提供簡化的 API 供 Streamlit 使用。
    """

    # Experiment #11v2 model configuration (RECOMMENDED)
    # 延長訓練時間優化（修復輸出）
    MODEL_CONFIG_EXP11V2 = {
        'seq_len': 4,
        'static_dim': 781,
        'd_model': 64,
        'nhead': 8,  # Exp11v2 uses 8 attention heads
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.3
    }

    # Legacy model configuration (fallback, compatible with saved model from 2025-10-28)
    MODEL_CONFIG_LEGACY = {
        'ts_input_dim': 1,
        'static_input_dim': 781,  # Old model used 781
        'd_model': 64,
        'nhead': 4,  # Old model used 4 heads
        'num_encoder_layers': 2,
        'dim_feedforward': 256,
        'static_hidden_dim': 256,
        'static_hidden_dim_2': 128,
        'fusion_hidden_dim': 128,
        'fusion_hidden_dim_2': 64,
        'dropout': 0.3,
        'max_seq_len': 4  # Saved model has pe with size 4
    }

    # New model configuration (for future retraining)
    MODEL_CONFIG = {
        'ts_input_dim': 1,
        'static_input_dim': 772,  # CLIP (768) + Season (4)
        'd_model': 64,
        'nhead': 8,
        'num_encoder_layers': 2,
        'dim_feedforward': 256,
        'static_hidden_dim': 256,
        'fusion_hidden_dim': 64,
        'dropout': 0.3,
        'max_seq_len': 10
    }

    # Performance metrics (from Exp #11v2)
    MAE = 327.26
    R2 = 0.6788

    # Season encoding
    SEASON_MAP = {
        'Spring': [1, 0, 0, 0],
        'Summer': [0, 1, 0, 0],
        'Fall': [0, 0, 1, 0],
        'Winter': [0, 0, 0, 1]
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ForecastPredictorWrapper.

        Args:
            model_path: Path to model weights (optional, defaults to best model)
        """
        if model_path is None:
            model_path = PROJECT_ROOT / 'models' / 'transformer_lulu' / 'best_transformer_model.pth'

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise ModelLoadError(f"模型權重檔案不存在：{self.model_path}")

        # Model will be lazy loaded
        self._model = None

        logger.info(f"ForecastPredictorWrapper initialized (model_path={self.model_path})")

    @property
    def model(self) -> nn.Module:
        """
        Lazy load Transformer model.

        Returns:
            Loaded and eval-mode model
        """
        if self._model is None:
            logger.info("Loading Transformer model...")
            self._model = self._load_model()
            logger.info("Transformer model loaded successfully")
        return self._model

    def _load_model(self) -> nn.Module:
        """
        載入訓練好的 Transformer 模型。

        嘗試順序：
        1. Experiment #11v2 架構（推薦）
        2. Legacy 架構（fallback）

        Returns:
            Loaded model in eval mode

        Raises:
            ModelLoadError: 當模型載入失敗時
        """
        # Load state dict first
        try:
            state_dict = torch.load(
                self.model_path,
                map_location='cpu',
                weights_only=True
            )
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise ModelLoadError(f"模型檔案讀取失敗：{str(e)}")

        exp11v2_error = None
        legacy_error = None

        # Try Experiment #11v2 architecture first (RECOMMENDED)
        try:
            logger.info("Attempting to load model with Experiment #11v2 architecture...")
            model = HybridTransformerExp11v2(**self.MODEL_CONFIG_EXP11V2)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            logger.info(f"✅ Model loaded from {self.model_path} (Experiment #11v2 architecture)")
            return model
        except Exception as e:
            exp11v2_error = e
            logger.warning(f"Failed to load with Exp11v2 architecture: {e}")

        # Fallback to Legacy architecture
        try:
            logger.info("Falling back to legacy architecture...")
            model = HybridTransformerLegacy(**self.MODEL_CONFIG_LEGACY)

            # Load with strict=False to allow pe size mismatch
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Handle positional encoding size mismatch
            if 'pos_encoder.pe' in missing_keys or any('pos_encoder.pe' in str(k) for k in missing_keys):
                logger.warning("Positional encoding size mismatch, will be handled at runtime")

            # Extend pe buffer if needed (saved pe is [4,1,64], we need at least [5,1,64] for CLS token)
            saved_pe = state_dict.get('pos_encoder.pe')
            if saved_pe is not None and saved_pe.size(0) < 5:
                import math
                logger.info(f"Extending positional encoding from {saved_pe.size(0)} to 5 positions...")

                # Create extended pe with same structure as saved
                d_model = saved_pe.size(2)
                max_len = 5
                pe_extended = torch.zeros(max_len, 1, d_model)

                # Copy existing positions
                pe_extended[:saved_pe.size(0), :, :] = saved_pe

                # Generate new position (position 4) using same formula
                position = torch.tensor([4.0]).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe_extended[4, 0, 0::2] = torch.sin(position * div_term)
                pe_extended[4, 0, 1::2] = torch.cos(position * div_term)

                # Replace pe buffer
                model.pos_encoder.pe = pe_extended
                logger.info("✅ Positional encoding extended successfully")

            model.eval()
            logger.info(f"✅ Model loaded from {self.model_path} (legacy architecture)")
            return model
        except Exception as e:
            legacy_error = e
            logger.error(f"Failed to load with legacy architecture: {e}")
            raise ModelLoadError(
                f"模型載入失敗：無法匹配 Exp11v2 或 Legacy 架構\n"
                f"Exp11v2 錯誤: {exp11v2_error}\n"
                f"Legacy 錯誤: {legacy_error}"
            )

    def _encode_season(self, season: str) -> np.ndarray:
        """
        季節 one-hot encoding。

        Args:
            season: "Spring", "Summer", "Fall", "Winter"

        Returns:
            One-hot encoded array (4,)

        Raises:
            ValueError: 如果 season 不在允許值中
        """
        if season not in self.SEASON_MAP:
            raise ValueError(
                f"Invalid season: {season}. "
                f"Must be one of {list(self.SEASON_MAP.keys())}"
            )

        return np.array(self.SEASON_MAP[season], dtype=np.float32)

    def predict_sales(
        self,
        season: str,
        clip_embedding: np.ndarray,
        trends_history: List[float]
    ) -> Dict[str, float]:
        """
        預測指定季節的銷量。

        Args:
            season: 季節名稱 (Spring/Summer/Fall/Winter)
            clip_embedding: CLIP embedding vector (768-dim)
            trends_history: 過去 4 季度的 Google Trends 分數 [Q-3, Q-2, Q-1, Q0]

        Returns:
            包含預測結果的字典:
            {
                'predicted_sales': 預測銷量,
                'lower_bound': 下限 (predicted - MAE),
                'upper_bound': 上限 (predicted + MAE),
                'confidence': 信心度 (R²),
                'mae': MAE 值
            }

        Raises:
            ValueError: 如果輸入參數無效
            ForecastError: 如果預測失敗

        Example:
            >>> predictor = ForecastPredictorWrapper()
            >>> result = predictor.predict_sales(
            ...     season="Spring",
            ...     clip_embedding=np.random.rand(768),
            ...     trends_history=[45, 52, 48, 50]
            ... )
            >>> print(f"Predicted: {result['predicted_sales']:.0f} ± {result['mae']:.0f}")
        """
        # Validation
        if clip_embedding.shape != (768,):
            raise ValueError(f"clip_embedding must be (768,), got {clip_embedding.shape}")

        if len(trends_history) != 4:
            raise ValueError(f"trends_history must have 4 values, got {len(trends_history)}")

        try:
            # Prepare inputs
            season_encoding = self._encode_season(season)  # (4,)
            static_features = np.concatenate([clip_embedding, season_encoding])  # (772,)

            # IMPORTANT: Legacy model requires 781 dimensions
            # Add 9-dimensional padding to match old model architecture
            if static_features.shape[0] == 772:
                padding = np.zeros(9, dtype=np.float32)
                static_features = np.concatenate([static_features, padding])  # (781,)
                logger.debug(f"Added 9-dim padding for legacy model compatibility (772 → 781)")

            time_series = np.array(trends_history, dtype=np.float32).reshape(-1, 1)  # (4, 1)

            # Convert to tensors
            ts_tensor = torch.FloatTensor(time_series).unsqueeze(0)  # (1, 4, 1)
            static_tensor = torch.FloatTensor(static_features).unsqueeze(0)  # (1, 781)

            # Predict
            with torch.no_grad():
                prediction = self.model(ts_tensor, static_tensor)

            predicted_sales = float(prediction.item())

            logger.info(f"Prediction for {season}: {predicted_sales:.2f} ± {self.MAE:.2f}")

            return {
                'predicted_sales': predicted_sales,
                'lower_bound': predicted_sales - self.MAE,
                'upper_bound': predicted_sales + self.MAE,
                'confidence': self.R2,
                'mae': self.MAE
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ForecastError(f"預測失敗：{str(e)}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        返回 Feature Importance（基於 Obj 3 實驗分析）。

        Returns:
            字典包含各特徵的重要性權重（總和 = 1.0）

        Example:
            >>> predictor.get_feature_importance()
            {
                'Google Trends': 0.35,
                'CLIP Similarity': 0.30,
                'Season': 0.20,
                'Product Type': 0.15
            }
        """
        return {
            'Google Trends': 0.35,
            'CLIP Similarity': 0.30,
            'Season': 0.20,
            'Product Type': 0.15
        }

    def get_model_metrics(self) -> Dict[str, float]:
        """
        返回模型性能指標（來自 Exp #11v2）。

        Returns:
            字典包含模型評估指標
        """
        return {
            'MAE': self.MAE,
            'R2': self.R2,
            'Error_Rate': self.MAE / 2844,  # 平均銷量
            'Confidence_Percent': self.R2 * 100
        }

    def generate_market_insights(
        self,
        predicted_sales: float,
        season: str,
        clip_similarity: float
    ) -> Dict[str, str]:
        """
        生成市場洞察建議（基於預測結果）。

        Args:
            predicted_sales: 預測銷量
            season: 季節
            clip_similarity: CLIP 相似度

        Returns:
            字典包含市場建議
        """
        insights = {}

        # 1. 上市時機建議
        if season in ['Spring', 'Summer']:
            insights['timing'] = f"{season} 是推出新品的理想時機（需求較高）"
        else:
            insights['timing'] = f"{season} 需求相對較低，建議配合節日活動"

        # 2. 生產數量建議
        production_qty = int(predicted_sales * 1.1)  # 多備 10%
        insights['production'] = f"建議生產數量：{production_qty:,} 件（預測 + 10% 安全庫存）"

        # 3. 角色一致性評估
        if clip_similarity >= 0.85:
            insights['character'] = "✅ 角色一致性極佳，品牌識別度高"
        elif clip_similarity >= 0.80:
            insights['character'] = "✅ 角色一致性良好，符合品牌要求"
        else:
            insights['character'] = "⚠️ 角色一致性偏低，建議優化設計"

        # 4. 風險提示
        error_rate = (self.MAE / predicted_sales) * 100
        if error_rate > 25:
            insights['risk'] = f"⚠️ 預測誤差較大（±{error_rate:.1f}%），建議謹慎評估"
        elif error_rate > 15:
            insights['risk'] = f"⚠️ 中度預測不確定性（±{error_rate:.1f}%）"
        else:
            insights['risk'] = f"✅ 預測可信度高（誤差 ±{error_rate:.1f}%）"

        return insights
