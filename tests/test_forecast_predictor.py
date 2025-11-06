"""
Unit Tests for ForecastPredictorWrapper

測試 Obj 3 API Wrapper 功能。

Author: Developer (James)
Date: 2025-11-06
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.forecast_predictor import (
    ForecastPredictorWrapper,
    ForecastError,
    ModelLoadError
)


class TestForecastPredictorWrapper(unittest.TestCase):
    """Test cases for ForecastPredictorWrapper."""

    @patch('obj4_web_app.utils.forecast_predictor.Path.exists')
    @patch('obj4_web_app.utils.forecast_predictor.torch.load')
    @patch('obj4_web_app.utils.forecast_predictor.HybridTransformer')
    def test_initialization_success(self, mock_transformer, mock_load, mock_exists):
        """Test successful initialization."""
        mock_exists.return_value = True

        predictor = ForecastPredictorWrapper()

        self.assertIsNotNone(predictor)
        self.assertIsNotNone(predictor.model_path)

    @patch('obj4_web_app.utils.forecast_predictor.Path.exists')
    def test_initialization_model_not_found(self, mock_exists):
        """Test initialization with missing model file."""
        mock_exists.return_value = False

        with self.assertRaises(ModelLoadError):
            ForecastPredictorWrapper(model_path="/nonexistent/model.pth")

    def test_encode_season_valid(self):
        """Test season encoding with valid season."""
        with patch('obj4_web_app.utils.forecast_predictor.Path.exists', return_value=True):
            predictor = ForecastPredictorWrapper()

            spring = predictor._encode_season('Spring')
            self.assertTrue(np.array_equal(spring, [1, 0, 0, 0]))

            summer = predictor._encode_season('Summer')
            self.assertTrue(np.array_equal(summer, [0, 1, 0, 0]))

            fall = predictor._encode_season('Fall')
            self.assertTrue(np.array_equal(fall, [0, 0, 1, 0]))

            winter = predictor._encode_season('Winter')
            self.assertTrue(np.array_equal(winter, [0, 0, 0, 1]))

    def test_encode_season_invalid(self):
        """Test season encoding with invalid season."""
        with patch('obj4_web_app.utils.forecast_predictor.Path.exists', return_value=True):
            predictor = ForecastPredictorWrapper()

            with self.assertRaises(ValueError):
                predictor._encode_season('InvalidSeason')

    def test_get_feature_importance(self):
        """Test get_feature_importance returns correct structure."""
        with patch('obj4_web_app.utils.forecast_predictor.Path.exists', return_value=True):
            predictor = ForecastPredictorWrapper()

            importance = predictor.get_feature_importance()

            # Check structure
            self.assertIsInstance(importance, dict)
            self.assertIn('Google Trends', importance)
            self.assertIn('CLIP Similarity', importance)
            self.assertIn('Season', importance)
            self.assertIn('Product Type', importance)

            # Check sum is approximately 1.0
            total = sum(importance.values())
            self.assertAlmostEqual(total, 1.0, places=2)

    def test_get_model_metrics(self):
        """Test get_model_metrics returns correct metrics."""
        with patch('obj4_web_app.utils.forecast_predictor.Path.exists', return_value=True):
            predictor = ForecastPredictorWrapper()

            metrics = predictor.get_model_metrics()

            self.assertIsInstance(metrics, dict)
            self.assertIn('MAE', metrics)
            self.assertIn('R2', metrics)
            self.assertEqual(metrics['MAE'], 327.26)
            self.assertEqual(metrics['R2'], 0.6788)

    def test_generate_market_insights(self):
        """Test generate_market_insights returns insights."""
        with patch('obj4_web_app.utils.forecast_predictor.Path.exists', return_value=True):
            predictor = ForecastPredictorWrapper()

            insights = predictor.generate_market_insights(
                predicted_sales=2500,
                season='Spring',
                clip_similarity=0.85
            )

            self.assertIsInstance(insights, dict)
            self.assertIn('timing', insights)
            self.assertIn('production', insights)
            self.assertIn('character', insights)
            self.assertIn('risk', insights)

    @patch('obj4_web_app.utils.forecast_predictor.Path.exists')
    @patch('obj4_web_app.utils.forecast_predictor.torch.load')
    @patch('obj4_web_app.utils.forecast_predictor.HybridTransformer')
    def test_predict_sales_invalid_clip_embedding(self, mock_transformer, mock_load, mock_exists):
        """Test predict_sales with invalid CLIP embedding shape."""
        mock_exists.return_value = True

        predictor = ForecastPredictorWrapper()

        # Invalid shape (not 768)
        invalid_embedding = np.random.rand(100)

        with self.assertRaises(ValueError):
            predictor.predict_sales(
                season='Spring',
                clip_embedding=invalid_embedding,
                trends_history=[45, 52, 48, 50]
            )

    @patch('obj4_web_app.utils.forecast_predictor.Path.exists')
    @patch('obj4_web_app.utils.forecast_predictor.torch.load')
    @patch('obj4_web_app.utils.forecast_predictor.HybridTransformer')
    def test_predict_sales_invalid_trends_length(self, mock_transformer, mock_load, mock_exists):
        """Test predict_sales with invalid trends history length."""
        mock_exists.return_value = True

        predictor = ForecastPredictorWrapper()

        valid_embedding = np.random.rand(768)
        invalid_trends = [45, 52]  # Only 2 values, need 4

        with self.assertRaises(ValueError):
            predictor.predict_sales(
                season='Spring',
                clip_embedding=valid_embedding,
                trends_history=invalid_trends
            )


class TestForecastPredictorWrapperIntegration(unittest.TestCase):
    """Integration tests (requires model weights)."""

    def test_model_config_consistency(self):
        """Test model config matches Exp #11v2."""
        with patch('obj4_web_app.utils.forecast_predictor.Path.exists', return_value=True):
            predictor = ForecastPredictorWrapper()

            self.assertEqual(predictor.MODEL_CONFIG['d_model'], 64)
            self.assertEqual(predictor.MODEL_CONFIG['nhead'], 8)
            self.assertEqual(predictor.MODEL_CONFIG['num_encoder_layers'], 2)
            self.assertEqual(predictor.MODEL_CONFIG['static_input_dim'], 772)  # 768 + 4


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
