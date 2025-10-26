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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __init__(self, region: str = 'HK', lang: str = 'zh-TW', tz: int = 480):
        """
        Initialize Google Trends client.

        Args:
            region: Google Trends region code (default: 'HK' for Hong Kong)
            lang: Language code (default: 'zh-TW' for Traditional Chinese)
            tz: Timezone offset in minutes (default: 480 for HKT)
        """
        self.region = region
        self.lang = lang
        self.tz = tz
        self.pytrend = TrendReq(hl=lang, tz=tz)
        logger.info(f"TrendsExtractor initialized: region={region}, lang={lang}")

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
        logger.info(f"Extracting trends for theme: {theme}")
        start_time = time.time()

        try:
            # Build keyword list for theme (English + Chinese)
            theme_keywords = self._get_theme_keywords(theme)

            # Query Google Trends
            self.pytrend.build_payload(
                theme_keywords,
                cat=0,  # All categories
                timeframe=timeframe,
                geo=self.region
            )

            # Get interest over time
            interest_df = self.pytrend.interest_over_time()

            if interest_df.empty:
                logger.warning(f"No trends data found for theme: {theme}")
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

            # Get related queries
            related_queries = self.pytrend.related_queries()
            for keyword in theme_keywords:
                if keyword in related_queries and related_queries[keyword]['top'] is not None:
                    top_related = related_queries[keyword]['top'].head(5)
                    for _, row in top_related.iterrows():
                        trend_scores.append({
                            'keyword': row['query'],
                            'trend_score': row['value'],
                            'theme': theme
                        })

            # Convert to DataFrame and sort
            trends_df = pd.DataFrame(trend_scores)
            trends_df = trends_df.sort_values('trend_score', ascending=False)
            trends_df = trends_df.head(top_n)

            elapsed = time.time() - start_time
            logger.info(f"Extracted {len(trends_df)} keywords for {theme} in {elapsed:.2f}s")

            # Rate limiting (avoid Google Trends 429 errors)
            time.sleep(2)

            return trends_df

        except Exception as e:
            logger.error(f"Error extracting trends for {theme}: {e}")
            return pd.DataFrame(columns=['keyword', 'trend_score', 'theme'])

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
