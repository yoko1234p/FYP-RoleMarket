"""
Trends Extractor Wrapper - Obj 1 Google Trends Integration for Streamlit

封裝 TrendsExtractor 為 Streamlit 友善的 API，支援自動提取熱門關鍵字。

Author: Developer (James)
Date: 2025-11-06
Version: 1.1
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj1_nlp_prompt.trends_extractor import TrendsExtractor, TrendsExtractionError

logger = logging.getLogger(__name__)


class TrendsExtractorWrapper:
    """
    Wrapper for TrendsExtractor 以支援 Streamlit 自動關鍵字提取。

    提供簡化的 API 供 Streamlit 使用。
    """

    # Supported themes (7 major themes)
    SUPPORTED_THEMES = [
        "Halloween",
        "Christmas",
        "Spring Festival",
        "Summer",
        "Valentine's Day",
        "Mid-Autumn Festival",
        "New Year"
    ]

    # Theme display names (Traditional Chinese)
    THEME_DISPLAY_NAMES = {
        "Halloween": "🎃 萬聖節",
        "Christmas": "🎄 聖誕節",
        "Spring Festival": "🧧 春節",
        "Summer": "☀️ 夏天",
        "Valentine's Day": "💝 情人節",
        "Mid-Autumn Festival": "🥮 中秋節",
        "New Year": "🎆 新年"
    }

    def __init__(self, region: str = 'HK', lang: str = 'zh-TW'):
        """
        Initialize TrendsExtractorWrapper.

        Args:
            region: Google Trends region code (default: 'HK')
            lang: Language code (default: 'zh-TW')
        """
        self.region = region
        self.lang = lang
        self._extractor = None  # Lazy load

        logger.info(f"TrendsExtractorWrapper initialized (region={region}, lang={lang})")

    @property
    def extractor(self) -> TrendsExtractor:
        """
        Lazy load TrendsExtractor.

        Returns:
            Loaded TrendsExtractor instance
        """
        if self._extractor is None:
            logger.info("Loading TrendsExtractor...")
            self._extractor = TrendsExtractor(region=self.region, lang=self.lang)
            logger.info("TrendsExtractor loaded successfully")
        return self._extractor

    def get_trending_keywords(
        self,
        theme: str,
        timeframe: str = 'today 12-m',
        top_n: int = 10
    ) -> List[Dict[str, any]]:
        """
        獲取指定主題的熱門關鍵字。

        Args:
            theme: 主題名稱（必須在 SUPPORTED_THEMES 中）
            timeframe: 時間範圍（default: 'today 12-m' 過去 12 個月）
            top_n: 返回前 N 個關鍵字（default: 10）

        Returns:
            List of dictionaries:
            [
                {
                    'keyword': '萬聖節',
                    'trend_score': 95.0,
                    'rank': 1,
                    'is_high_trend': True  # trend_score >= 70
                },
                ...
            ]

        Raises:
            TrendsExtractionError: 當提取失敗時

        Example:
            >>> wrapper = TrendsExtractorWrapper()
            >>> keywords = wrapper.get_trending_keywords('Halloween', top_n=10)
            >>> print(keywords[0])
            {'keyword': '萬聖節', 'trend_score': 95.0, 'rank': 1, 'is_high_trend': True}
        """
        # Validation
        if theme not in self.SUPPORTED_THEMES:
            raise ValueError(
                f"Unsupported theme: {theme}. "
                f"Must be one of {self.SUPPORTED_THEMES}"
            )

        try:
            # Extract trends using TrendsExtractor
            trends_df = self.extractor.extract_keywords(
                theme=theme,
                timeframe=timeframe,
                top_n=top_n
            )

            if trends_df.empty:
                logger.warning(f"No trends found for theme: {theme}")
                return []

            # Convert to list of dicts with additional metadata
            keywords = []
            for i, row in trends_df.iterrows():
                keyword_data = {
                    'keyword': row['keyword'],
                    'trend_score': round(float(row['trend_score']), 2),
                    'rank': len(keywords) + 1,
                    'is_high_trend': float(row['trend_score']) >= 70.0
                }
                keywords.append(keyword_data)

            logger.info(f"Extracted {len(keywords)} keywords for theme: {theme}")
            return keywords

        except Exception as e:
            logger.error(f"Failed to extract trends for {theme}: {e}")
            raise TrendsExtractionError(f"趨勢提取失敗：{str(e)}")

    def get_all_themes(self) -> List[Dict[str, str]]:
        """
        獲取所有支援的主題。

        Returns:
            List of dictionaries:
            [
                {'value': 'Halloween', 'display': '🎃 萬聖節'},
                {'value': 'Christmas', 'display': '🎄 聖誕節'},
                ...
            ]

        Example:
            >>> wrapper = TrendsExtractorWrapper()
            >>> themes = wrapper.get_all_themes()
            >>> print(themes[0])
            {'value': 'Halloween', 'display': '🎃 萬聖節'}
        """
        return [
            {
                'value': theme,
                'display': self.THEME_DISPLAY_NAMES.get(theme, theme)
            }
            for theme in self.SUPPORTED_THEMES
        ]

    def format_keywords_for_prompt(self, selected_keywords: List[str]) -> str:
        """
        格式化選中的關鍵字為 Prompt 輸入格式。

        Args:
            selected_keywords: 選中的關鍵字列表

        Returns:
            逗號分隔的關鍵字字串

        Example:
            >>> wrapper = TrendsExtractorWrapper()
            >>> keywords = ['萬聖節', '南瓜', 'trick or treat']
            >>> formatted = wrapper.format_keywords_for_prompt(keywords)
            >>> print(formatted)
            '萬聖節, 南瓜, trick or treat'
        """
        return ", ".join(selected_keywords)

    def get_theme_suggestions(self, current_month: int) -> List[str]:
        """
        根據當前月份建議相關主題。

        Args:
            current_month: 當前月份 (1-12)

        Returns:
            List of suggested theme names

        Example:
            >>> wrapper = TrendsExtractorWrapper()
            >>> suggestions = wrapper.get_theme_suggestions(10)  # October
            >>> print(suggestions)
            ['Halloween', 'Mid-Autumn Festival']
        """
        # Month to theme mapping
        month_themes = {
            1: ["New Year"],
            2: ["Valentine's Day", "Spring Festival"],
            3: ["Spring Festival"],
            4: [],
            5: [],
            6: ["Summer"],
            7: ["Summer"],
            8: ["Mid-Autumn Festival"],
            9: ["Mid-Autumn Festival"],
            10: ["Halloween"],
            11: ["Christmas"],
            12: ["Christmas", "New Year"]
        }

        return month_themes.get(current_month, [])

    def get_keywords_dataframe(
        self,
        theme: str,
        timeframe: str = 'today 12-m',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        獲取關鍵字 DataFrame（用於視覺化）。

        Args:
            theme: 主題名稱
            timeframe: 時間範圍
            top_n: 前 N 個關鍵字

        Returns:
            pandas DataFrame with columns: keyword, trend_score, rank, is_high_trend

        Example:
            >>> wrapper = TrendsExtractorWrapper()
            >>> df = wrapper.get_keywords_dataframe('Halloween', top_n=5)
            >>> print(df.head())
               keyword  trend_score  rank  is_high_trend
            0  萬聖節    95.0         1     True
            1  南瓜      78.5         2     True
        """
        keywords = self.get_trending_keywords(theme, timeframe, top_n)
        return pd.DataFrame(keywords)
