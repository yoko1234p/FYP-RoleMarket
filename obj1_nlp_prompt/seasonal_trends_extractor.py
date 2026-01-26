"""
季節性趨勢提取器 - 根據節日時段動態查詢 Google Trends

功能：
- 自動判斷當前是否在節日時段
- 使用節日專屬時段查詢趨勢（例如：聖誕節用 11-12月數據）
- 如果不在時段內，使用去年同期數據
- 獲取當時最熱門的新興關鍵字

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
Story: Enhanced - Seasonal Timeframe Trends
"""

from pytrends.request import TrendReq
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeasonalTrendsExtractor:
    """
    根據節日時段提取 Google Trends 數據.

    自動判斷當前日期是否在節日高峰期，使用對應時段查詢趨勢。
    例如：如果現在是 12 月，查詢聖誕節趨勢時會使用 11-12 月數據。

    Usage:
        >>> extractor = SeasonalTrendsExtractor()
        >>> trends = extractor.extract_seasonal_trends('Christmas')
        >>> print(f"Found {len(trends)} trending keywords for Christmas")
    """

    def __init__(
        self,
        config_path: str = 'config/seasonal_timeframes.json',
        region: str = 'HK',
        lang: str = 'zh-TW'
    ):
        """
        Initialize seasonal trends extractor.

        Args:
            config_path: Path to seasonal timeframes config
            region: Google Trends region code
            lang: Language code
        """
        self.region = region
        self.lang = lang
        self.pytrend = TrendReq(hl=lang, tz=480)

        # Load seasonal timeframes config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        logger.info(f"SeasonalTrendsExtractor initialized: region={region}")

    def get_timeframe_for_theme(
        self,
        theme: str,
        current_date: Optional[datetime] = None
    ) -> str:
        """
        根據節日和當前日期，返回最佳查詢時段.

        邏輯：
        1. 如果當前月份在節日高峰期內 → 使用今年該時段
        2. 如果不在高峰期 → 使用去年同期數據
        3. 如果找不到配置 → 使用默認 12 個月

        Args:
            theme: 節日主題（例如 'Christmas'）
            current_date: 當前日期（用於測試，默認為今天）

        Returns:
            Google Trends timeframe 格式字串

        Example:
            >>> extractor = SeasonalTrendsExtractor()
            >>> # 如果現在是 12 月
            >>> timeframe = extractor.get_timeframe_for_theme('Christmas')
            >>> print(timeframe)  # "2024-11-01 2024-12-31"
        """
        if current_date is None:
            current_date = datetime.now()

        current_month = current_date.month
        current_year = current_date.year

        # Get theme config
        theme_config = self.config['seasonal_timeframes'].get(theme)

        if not theme_config:
            logger.warning(f"No seasonal config found for {theme}, using default")
            return self.config['default_timeframe']

        peak_months = theme_config['peak_months']

        # Check if current month is in peak period
        if current_month in peak_months:
            # 在高峰期內，使用今年該時段
            timeframe = theme_config['search_period']
            logger.info(f"✅ {theme} is in peak season! Using current year: {timeframe}")
        else:
            # 不在高峰期，使用去年同期
            # 計算去年的時段
            last_year = current_year - 1
            original_period = theme_config['search_period']

            # Replace year with last year
            timeframe = original_period.replace(str(current_year), str(last_year))
            logger.info(f"⏮️  Not in peak season. Using last year data: {timeframe}")

        return timeframe

    def extract_seasonal_trends(
        self,
        theme: str,
        top_n: int = 20,
        current_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        提取節日的季節性趨勢數據.

        自動使用最佳時段查詢 Google Trends。

        Args:
            theme: 節日主題
            top_n: 返回前 N 個關鍵字
            current_date: 當前日期（用於測試）

        Returns:
            DataFrame with columns: keyword, trend_score, theme, timeframe

        Example:
            >>> extractor = SeasonalTrendsExtractor()
            >>> christmas_trends = extractor.extract_seasonal_trends('Christmas')
            >>> print(christmas_trends.head())
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"提取節日趨勢: {theme}")
        logger.info(f"{'='*60}\n")

        # Get optimal timeframe
        timeframe = self.get_timeframe_for_theme(theme, current_date)

        # Get theme keywords
        theme_keywords = self._get_theme_keywords(theme)
        logger.info(f"種子關鍵字: {theme_keywords}")
        logger.info(f"查詢時段: {timeframe}\n")

        try:
            # Build payload
            self.pytrend.build_payload(
                theme_keywords,
                cat=0,
                timeframe=timeframe,
                geo=self.region
            )

            # Get interest over time
            interest_df = self.pytrend.interest_over_time()

            if interest_df.empty:
                logger.warning(f"No trends data found for {theme}")
                return pd.DataFrame(columns=['keyword', 'trend_score', 'theme', 'timeframe'])

            # Calculate trend scores
            trend_scores = []
            for keyword in theme_keywords:
                if keyword in interest_df.columns:
                    avg_score = interest_df[keyword].mean()
                    max_score = interest_df[keyword].max()
                    trend_scores.append({
                        'keyword': keyword,
                        'trend_score': avg_score,
                        'peak_score': max_score,
                        'theme': theme,
                        'timeframe': timeframe
                    })

            # Get related queries
            logger.info("正在獲取相關查詢...")
            related_queries = self.pytrend.related_queries()

            for keyword in theme_keywords:
                if keyword in related_queries and related_queries[keyword]['top'] is not None:
                    top_related = related_queries[keyword]['top'].head(10)
                    for _, row in top_related.iterrows():
                        trend_scores.append({
                            'keyword': row['query'],
                            'trend_score': row['value'],
                            'peak_score': row['value'],
                            'theme': theme,
                            'timeframe': timeframe
                        })

            # Convert to DataFrame
            trends_df = pd.DataFrame(trend_scores)

            # Remove duplicates and sort
            trends_df = trends_df.drop_duplicates(subset=['keyword'])
            trends_df = trends_df.sort_values('trend_score', ascending=False)
            trends_df = trends_df.head(top_n)

            logger.info(f"\n✅ 成功提取 {len(trends_df)} 個關鍵字")
            logger.info(f"Top 5:")
            for idx, row in trends_df.head(5).iterrows():
                logger.info(f"  {idx+1}. {row['keyword']:<30} (分數: {row['trend_score']:.1f})")

            return trends_df

        except Exception as e:
            logger.error(f"❌ 提取趨勢失敗: {e}")
            return pd.DataFrame(columns=['keyword', 'trend_score', 'theme', 'timeframe'])

    def extract_all_themes_seasonal(
        self,
        themes: List[str],
        top_n: int = 20,
        output_dir: str = 'data/trends_seasonal'
    ) -> Dict[str, pd.DataFrame]:
        """
        提取所有節日的季節性趨勢.

        Args:
            themes: 節日主題列表
            top_n: 每個主題的關鍵字數量
            output_dir: 輸出目錄

        Returns:
            Dictionary mapping theme to trends DataFrame
        """
        all_trends = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, theme in enumerate(themes, 1):
            logger.info(f"\n處理主題 {i}/{len(themes)}: {theme}")

            trends_df = self.extract_seasonal_trends(theme, top_n)
            all_trends[theme] = trends_df

            # Save immediately
            filename = output_path / f"{theme.lower().replace(' ', '_')}_seasonal_trends.csv"
            trends_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"💾 已儲存: {filename}\n")

            # Rate limiting
            import time
            time.sleep(2)

        logger.info(f"\n{'='*60}")
        logger.info(f"完成所有主題提取")
        logger.info(f"{'='*60}")
        logger.info(f"總主題數: {len(all_trends)}")
        logger.info(f"總關鍵字數: {sum(len(df) for df in all_trends.values())}")
        logger.info(f"輸出目錄: {output_dir}")
        logger.info(f"{'='*60}\n")

        return all_trends

    def _get_theme_keywords(self, theme: str) -> List[str]:
        """Get seed keywords for theme."""
        theme_map = {
            'Halloween': ['Halloween', '萬聖節', '南瓜', 'trick or treat', 'costume'],
            'Christmas': ['Christmas', '聖誕節', '聖誕老人', 'Xmas', 'christmas tree'],
            'Spring Festival': ['Spring Festival', '春節', '農曆新年', 'Lunar New Year', '紅包'],
            'Summer': ['Summer', '夏天', '暑假', 'beach', '游泳'],
            "Valentine's Day": ['Valentine', '情人節', 'love', '浪漫', 'roses'],
            'Mid-Autumn Festival': ['Mid-Autumn', '中秋節', '月餅', 'mooncake', '賞月'],
            'Dragon Boat Festival': ['Dragon Boat', '端午節', '粽子', 'dragon boat race']
        }
        return theme_map.get(theme, [theme])


def main():
    """
    測試季節性趨勢提取.

    提取所有節日的季節性趨勢數據。
    """
    import json

    # Load themes
    with open('config/character_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    themes = config['themes']

    logger.info(f"\n{'='*60}")
    logger.info(f"季節性趨勢提取測試")
    logger.info(f"{'='*60}\n")
    logger.info(f"測試主題: {themes}")
    logger.info(f"當前日期: {datetime.now().strftime('%Y-%m-%d')}\n")

    # Initialize extractor
    extractor = SeasonalTrendsExtractor()

    # Extract all themes
    all_trends = extractor.extract_all_themes_seasonal(themes, top_n=20)

    return all_trends


if __name__ == '__main__':
    main()
