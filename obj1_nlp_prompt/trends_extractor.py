"""
Google Trends Data Extraction Module

Extracts trending keywords from Google Trends for Hong Kong market (Traditional Chinese).
Supports seasonal themes for character IP design generation.

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
Story: 2.1 - Google Trends Data Extraction
"""

from pytrends.request import TrendReq
import pandas as pd
import time
from typing import List, Dict
from pathlib import Path
import logging
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Regional configurations for Google Trends
REGION_CONFIGS = {
    'HK': {'geo': 'HK', 'hl': 'zh-TW', 'tz': 480},  # Hong Kong
    'TW': {'geo': 'TW', 'hl': 'zh-TW', 'tz': 480},  # Taiwan
    'US': {'geo': 'US', 'hl': 'en-US', 'tz': 360},  # United States
    'CN': {'geo': 'CN', 'hl': 'zh-CN', 'tz': 480},  # China
}


def retry_with_backoff(max_retries=3, base_delay=2):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be exponentially increased)

    Example:
        @retry_with_backoff(max_retries=3, base_delay=2)
        def fetch_data():
            # API call that might fail
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay ** attempt  # Exponential: 2, 4, 8 seconds
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}"
                        )
                        logger.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {str(e)}"
                        )
                        raise
            return None
        return wrapper
    return decorator


class TrendsExtractor:
    """
    Extract trending keywords from Google Trends for seasonal themes.

    Features:
    - Hong Kong market focus (zh-TW)
    - 7 seasonal themes support
    - Rate limit handling
    - CSV export

    Usage:
        >>> extractor = TrendsExtractor(region='HK', lang='zh-TW')
        >>> trends = extractor.extract_keywords('Halloween', timeframe='today 12-m')
        >>> extractor.save_trends(trends, 'Halloween')
    """

    def __init__(self, region: str = 'HK', lang: str = None, tz: int = None):
        """
        Initialize Google Trends client with regional configuration.

        Args:
            region: Google Trends region code (default: 'HK')
                   Supported: 'HK', 'TW', 'US', 'CN'
            lang: Language code (optional, auto-detected from region config)
            tz: Timezone offset in minutes (optional, auto-detected from region config)
        """
        # Get regional configuration
        if region in REGION_CONFIGS:
            config = REGION_CONFIGS[region]
            self.region = config['geo']
            self.lang = lang or config['hl']
            self.tz = tz or config['tz']
            logger.info(f"Using regional config for {region}: {config}")
        else:
            # Fallback to manual configuration
            self.region = region
            self.lang = lang or 'zh-TW'
            self.tz = tz or 480
            logger.warning(
                f"Region {region} not in predefined configs. "
                f"Using manual config: geo={self.region}, hl={self.lang}, tz={self.tz}"
            )

        # Initialize pytrends client
        try:
            self.pytrend = TrendReq(hl=self.lang, tz=self.tz)
            logger.info(
                f"✅ TrendsExtractor initialized successfully: "
                f"region={self.region}, lang={self.lang}, tz={self.tz}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize TrendReq: {str(e)}")
            raise

    @retry_with_backoff(max_retries=3, base_delay=2)
    def _fetch_trends_data(
        self,
        theme_keywords: List[str],
        timeframe: str
    ) -> tuple[pd.DataFrame, dict]:
        """
        Fetch trends data from Google Trends API with retry logic.

        Args:
            theme_keywords: List of keywords to query
            timeframe: Google Trends timeframe

        Returns:
            Tuple of (interest_df, related_queries)

        Raises:
            Exception: If all retry attempts fail
        """
        logger.debug(f"Querying Google Trends API...")
        logger.debug(f"  Keywords: {theme_keywords}")
        logger.debug(f"  Timeframe: {timeframe}")
        logger.debug(f"  Region: {self.region}")

        # Build payload
        self.pytrend.build_payload(
            theme_keywords,
            cat=0,  # All categories
            timeframe=timeframe,
            geo=self.region
        )

        # Get interest over time
        interest_df = self.pytrend.interest_over_time()
        logger.debug(f"  Interest over time shape: {interest_df.shape}")

        # Get related queries
        related_queries = self.pytrend.related_queries()
        logger.debug(f"  Related queries retrieved: {len(related_queries)} keywords")

        return interest_df, related_queries

    def extract_keywords(
        self,
        theme: str,
        timeframe: str = 'today 12-m',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Extract trending keywords for a specific theme.

        Args:
            theme: Seasonal theme (e.g., 'Halloween', 'Christmas')
            timeframe: Google Trends timeframe (default: 'today 12-m' for 12 months)
            top_n: Number of top keywords to extract (default: 20)

        Returns:
            DataFrame with columns: keyword, trend_score, theme

        Example:
            >>> extractor = TrendsExtractor()
            >>> trends = extractor.extract_keywords('Halloween')
            >>> print(trends.head())
               keyword  trend_score     theme
            0  萬聖節      95.0        Halloween
            1  南瓜       78.5        Halloween
        """
        logger.info(f"📊 Starting trend extraction for theme: {theme}")
        start_time = time.time()

        try:
            # Build keyword list for theme (English + Chinese)
            theme_keywords = self._get_theme_keywords(theme)
            logger.info(f"  Using {len(theme_keywords)} seed keywords: {theme_keywords}")

            # Fetch trends data (with retry logic)
            try:
                interest_df, related_queries = self._fetch_trends_data(
                    theme_keywords,
                    timeframe
                )
            except Exception as e:
                # Improved error message
                error_msg = (
                    f"⚠️ 未找到相關趨勢數據：{theme}\n\n"
                    f"可能原因：\n"
                    f"1. Google Trends API 限流（請稍後重試）\n"
                    f"2. 主題關鍵字 '{theme}' 未找到相關數據\n"
                    f"3. 網絡連接問題\n\n"
                    f"💡 建議：\n"
                    f"- 請稍等 1-2 分鐘後重試\n"
                    f"- 或使用「✍️ 手動輸入」標籤頁手動輸入關鍵字\n"
                    f"- 嘗試其他主題（如：🎄 聖誕節、🎃 萬聖節）\n\n"
                    f"技術細節：{str(e)}"
                )
                logger.error(error_msg)
                raise TrendsExtractionError(error_msg)

            # Check if data is empty
            if interest_df.empty:
                logger.warning(f"⚠️ No trends data found for theme: {theme}")
                logger.warning(f"  This could mean the keywords have very low search volume")
                return pd.DataFrame(columns=['keyword', 'trend_score', 'theme'])

            # Calculate average trend score
            trend_scores = []
            for keyword in theme_keywords:
                if keyword in interest_df.columns:
                    avg_score = interest_df[keyword].mean()
                    trend_scores.append({
                        'keyword': keyword,
                        'trend_score': avg_score,
                        'theme': theme
                    })
                    logger.debug(f"  Keyword '{keyword}' avg score: {avg_score:.2f}")

            # Get related queries
            related_count = 0
            for keyword in theme_keywords:
                if keyword in related_queries and related_queries[keyword]['top'] is not None:
                    top_related = related_queries[keyword]['top'].head(5)
                    for _, row in top_related.iterrows():
                        trend_scores.append({
                            'keyword': row['query'],
                            'trend_score': row['value'],
                            'theme': theme
                        })
                        related_count += 1

            logger.info(f"  Found {related_count} related keywords")

            # Convert to DataFrame and sort
            trends_df = pd.DataFrame(trend_scores)
            if trends_df.empty:
                logger.warning(f"⚠️ No valid trend scores for theme: {theme}")
                return pd.DataFrame(columns=['keyword', 'trend_score', 'theme'])

            trends_df = trends_df.sort_values('trend_score', ascending=False)
            trends_df = trends_df.head(top_n)

            elapsed = time.time() - start_time
            logger.info(
                f"✅ Extracted {len(trends_df)} keywords for {theme} "
                f"in {elapsed:.2f}s"
            )

            # Rate limiting (avoid Google Trends 429 errors)
            time.sleep(2)

            return trends_df

        except TrendsExtractionError:
            # Re-raise our custom error with improved message
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error extracting trends for {theme}: {str(e)}", exc_info=True)
            raise TrendsExtractionError(
                f"趨勢提取過程中發生錯誤：{str(e)}\n\n"
                f"請稍後重試或使用手動輸入模式。"
            )


    def extract_all_themes(
        self,
        themes: List[str],
        timeframe: str = 'today 12-m',
        top_n: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract trends for all themes.

        Args:
            themes: List of seasonal themes
            timeframe: Google Trends timeframe
            top_n: Number of keywords per theme

        Returns:
            Dictionary mapping theme to trends DataFrame

        Example:
            >>> extractor = TrendsExtractor()
            >>> themes = ['Halloween', 'Christmas', 'Spring Festival']
            >>> all_trends = extractor.extract_all_themes(themes)
            >>> print(f"Total themes: {len(all_trends)}")
        """
        all_trends = {}
        total_start = time.time()

        for i, theme in enumerate(themes, 1):
            logger.info(f"Processing theme {i}/{len(themes)}: {theme}")
            trends_df = self.extract_keywords(theme, timeframe, top_n)
            all_trends[theme] = trends_df

            # Save immediately to avoid data loss
            self.save_trends(trends_df, theme)

        total_elapsed = time.time() - total_start
        logger.info(f"Completed all themes in {total_elapsed:.2f}s (target: <30s per theme)")

        return all_trends

    def save_trends(self, trends_df: pd.DataFrame, theme: str, output_dir: str = 'data/trends'):
        """
        Save trends data to CSV.

        Args:
            trends_df: Trends DataFrame
            theme: Theme name for filename
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / f"{theme.lower().replace(' ', '_')}_trends.csv"
        trends_df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Saved trends to: {filename}")

    def _get_theme_keywords(self, theme: str) -> List[str]:
        """
        Get seed keywords for theme (English + Chinese).

        Args:
            theme: Seasonal theme

        Returns:
            List of seed keywords for Google Trends query
        """
        # Theme-specific keywords (English + Chinese)
        theme_map = {
            'Halloween': ['Halloween', '萬聖節', '南瓜', 'trick or treat'],
            'Christmas': ['Christmas', '聖誕節', '聖誕老人', 'Xmas'],
            'Spring Festival': ['Spring Festival', '春節', '農曆新年', 'Lunar New Year'],
            'Summer': ['Summer', '夏天', '暑假', 'beach'],
            "Valentine's Day": ['Valentine', '情人節', 'love', '浪漫'],
            'Mid-Autumn Festival': ['Mid-Autumn', '中秋節', '月餅', 'mooncake'],
            'New Year': ['New Year', '新年', '元旦', 'countdown']
        }

        return theme_map.get(theme, [theme])


class TrendsExtractionError(Exception):
    """
    Custom exception for trends extraction failures.

    Raised when Google Trends API calls fail after all retry attempts.
    """
    pass


def main():
    """
    Main execution for Story 2.1.

    Extract trends for all 7 themes and save to data/trends/.
    """
    # Load themes from character config
    import json
    with open('config/character_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    themes = config['themes']
    logger.info(f"Loaded {len(themes)} themes from config: {themes}")

    # Initialize extractor
    extractor = TrendsExtractor(region='HK', lang='zh-TW')

    # Extract all trends
    all_trends = extractor.extract_all_themes(themes, timeframe='today 12-m', top_n=20)

    # Summary
    total_keywords = sum(len(df) for df in all_trends.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"Story 2.1 Completion Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Themes processed: {len(all_trends)}")
    logger.info(f"Total keywords extracted: {total_keywords}")
    logger.info(f"Target: {len(themes)} themes × 20 keywords = {len(themes) * 20}")
    logger.info(f"Output directory: data/trends/")
    logger.info(f"{'='*60}\n")

    return all_trends


if __name__ == '__main__':
    main()
