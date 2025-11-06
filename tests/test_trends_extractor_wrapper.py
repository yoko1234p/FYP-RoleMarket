"""
Unit Tests for TrendsExtractorWrapper

測試 Google Trends 自動提取功能。

Author: Developer (James)
Date: 2025-11-06
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.trends_extractor_wrapper import (
    TrendsExtractorWrapper,
    TrendsExtractionError
)


class TestTrendsExtractorWrapper(unittest.TestCase):
    """Test cases for TrendsExtractorWrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.wrapper = TrendsExtractorWrapper(region='HK', lang='zh-TW')

    def test_initialization(self):
        """Test successful initialization."""
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.region, 'HK')
        self.assertEqual(self.wrapper.lang, 'zh-TW')

    def test_supported_themes(self):
        """Test supported themes constant."""
        themes = TrendsExtractorWrapper.SUPPORTED_THEMES
        self.assertEqual(len(themes), 7)
        self.assertIn('Halloween', themes)
        self.assertIn('Christmas', themes)
        self.assertIn('Spring Festival', themes)

    def test_theme_display_names(self):
        """Test theme display names mapping."""
        display_names = TrendsExtractorWrapper.THEME_DISPLAY_NAMES
        self.assertEqual(display_names['Halloween'], '🎃 萬聖節')
        self.assertEqual(display_names['Christmas'], '🎄 聖誕節')

    def test_get_all_themes(self):
        """Test get_all_themes returns correct structure."""
        themes = self.wrapper.get_all_themes()

        self.assertIsInstance(themes, list)
        self.assertEqual(len(themes), 7)

        # Check structure of first theme
        first_theme = themes[0]
        self.assertIn('value', first_theme)
        self.assertIn('display', first_theme)
        self.assertIsInstance(first_theme['value'], str)
        self.assertIsInstance(first_theme['display'], str)

    def test_format_keywords_for_prompt(self):
        """Test keyword formatting for prompt."""
        keywords = ['萬聖節', '南瓜', 'trick or treat']
        formatted = self.wrapper.format_keywords_for_prompt(keywords)

        self.assertEqual(formatted, '萬聖節, 南瓜, trick or treat')

    def test_format_keywords_empty(self):
        """Test formatting empty keyword list."""
        keywords = []
        formatted = self.wrapper.format_keywords_for_prompt(keywords)

        self.assertEqual(formatted, '')

    def test_get_theme_suggestions_october(self):
        """Test theme suggestions for October."""
        suggestions = self.wrapper.get_theme_suggestions(10)

        self.assertIsInstance(suggestions, list)
        self.assertIn('Halloween', suggestions)

    def test_get_theme_suggestions_december(self):
        """Test theme suggestions for December."""
        suggestions = self.wrapper.get_theme_suggestions(12)

        self.assertIsInstance(suggestions, list)
        self.assertIn('Christmas', suggestions)
        self.assertIn('New Year', suggestions)

    def test_get_theme_suggestions_april(self):
        """Test theme suggestions for April (no specific themes)."""
        suggestions = self.wrapper.get_theme_suggestions(4)

        self.assertIsInstance(suggestions, list)
        self.assertEqual(len(suggestions), 0)

    @patch('obj4_web_app.utils.trends_extractor_wrapper.TrendsExtractor')
    def test_get_trending_keywords_success(self, mock_extractor_class):
        """Test successful keyword extraction."""
        # Mock TrendsExtractor
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        # Mock trends DataFrame
        mock_df = pd.DataFrame({
            'keyword': ['萬聖節', '南瓜', 'trick or treat'],
            'trend_score': [95.0, 78.5, 65.2],
            'theme': ['Halloween', 'Halloween', 'Halloween']
        })
        mock_extractor.extract_keywords.return_value = mock_df

        # Create new wrapper to trigger mock
        wrapper = TrendsExtractorWrapper()

        # Extract keywords
        keywords = wrapper.get_trending_keywords('Halloween', top_n=3)

        # Assertions
        self.assertEqual(len(keywords), 3)
        self.assertEqual(keywords[0]['keyword'], '萬聖節')
        self.assertEqual(keywords[0]['trend_score'], 95.0)
        self.assertEqual(keywords[0]['rank'], 1)
        self.assertTrue(keywords[0]['is_high_trend'])  # >= 70

        self.assertEqual(keywords[2]['keyword'], 'trick or treat')
        self.assertFalse(keywords[2]['is_high_trend'])  # < 70

    @patch('obj4_web_app.utils.trends_extractor_wrapper.TrendsExtractor')
    def test_get_trending_keywords_empty(self, mock_extractor_class):
        """Test keyword extraction with empty results."""
        # Mock TrendsExtractor
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        # Mock empty DataFrame
        mock_df = pd.DataFrame(columns=['keyword', 'trend_score', 'theme'])
        mock_extractor.extract_keywords.return_value = mock_df

        # Create new wrapper
        wrapper = TrendsExtractorWrapper()

        # Extract keywords
        keywords = wrapper.get_trending_keywords('Halloween')

        # Should return empty list
        self.assertEqual(len(keywords), 0)

    def test_get_trending_keywords_invalid_theme(self):
        """Test keyword extraction with invalid theme."""
        with self.assertRaises(ValueError):
            self.wrapper.get_trending_keywords('InvalidTheme')

    @patch('obj4_web_app.utils.trends_extractor_wrapper.TrendsExtractor')
    def test_get_keywords_dataframe(self, mock_extractor_class):
        """Test get_keywords_dataframe returns DataFrame."""
        # Mock TrendsExtractor
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        # Mock trends DataFrame
        mock_df = pd.DataFrame({
            'keyword': ['萬聖節', '南瓜'],
            'trend_score': [95.0, 78.5],
            'theme': ['Halloween', 'Halloween']
        })
        mock_extractor.extract_keywords.return_value = mock_df

        # Create new wrapper
        wrapper = TrendsExtractorWrapper()

        # Get DataFrame
        df = wrapper.get_keywords_dataframe('Halloween', top_n=2)

        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('keyword', df.columns)
        self.assertIn('trend_score', df.columns)
        self.assertIn('rank', df.columns)
        self.assertIn('is_high_trend', df.columns)


class TestTrendsExtractorWrapperIntegration(unittest.TestCase):
    """Integration tests (requires Google Trends API access)."""

    def test_lazy_loading(self):
        """Test extractor is lazy loaded."""
        wrapper = TrendsExtractorWrapper()

        # Should not be loaded yet
        self.assertIsNone(wrapper._extractor)

        # Access extractor property
        extractor = wrapper.extractor

        # Should now be loaded
        self.assertIsNotNone(extractor)
        self.assertIsNotNone(wrapper._extractor)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
