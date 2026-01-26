"""
Category-Based Trends Extractor using trendspyg

Enhanced version using trendspyg library for real-time Google Trends RSS data.

Key Advantages over trendspy/pytrends:
- Ultra-fast RSS feed (0.2-0.5 seconds)
- No rate limiting issues
- Rich media: news articles, images, headlines
- Real-time trending searches
- No API quota limitations
- Zero configuration needed

Author: Developer (James)
Date: 2025-11-10
Version: 3.0 - trendspyg Integration
"""

from trendspyg import download_google_trends_rss
import pandas as pd
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryTrendsExtractorTrendspyg:
    """
    Extract real-time trends using trendspyg RSS feed.

    Advantages:
    - Ultra-fast (0.2-0.5 seconds)
    - No rate limiting
    - Rich media data (news, images)
    - Zero configuration
    """

    def __init__(
        self,
        region: str = 'HK',
        lang: str = 'zh-TW',
        include_images: bool = True,
        include_articles: bool = True,
        max_articles_per_trend: int = 5
    ):
        """
        Initialize category-based trends extractor with trendspyg.

        Args:
            region: Google Trends region code (default: HK)
            lang: Language code (default: zh-TW) - not used by RSS but kept for compatibility
            include_images: Include trend images (default: True)
            include_articles: Include news articles (default: True)
            max_articles_per_trend: Max news articles per trend (default: 5)
        """
        self.region = region
        self.lang = lang
        self.include_images = include_images
        self.include_articles = include_articles
        self.max_articles_per_trend = max_articles_per_trend

        logger.info(f"CategoryTrendsExtractorTrendspyg initialized: region={region}, "
                   f"images={include_images}, articles={include_articles} (max={max_articles_per_trend})")

    def extract_trending_now(
        self,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Extract current trending searches using RSS feed.

        Args:
            top_n: Return top N trends (default: 20)
                Note: RSS typically returns 10-15 trends

        Returns:
            DataFrame with columns: keyword, trend_score, traffic, published,
                                   news_count, image_source

        Example:
            >>> extractor = CategoryTrendsExtractorTrendspyg()
            >>> trends = extractor.extract_trending_now(top_n=10)
            >>> print(trends.head())
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"trendspyg RSS Feed - Real-time Trending Searches")
        logger.info(f"{'='*60}\n")
        logger.info(f"Region: {self.region}")
        logger.info(f"Fetching real-time trends...")

        try:
            # Get RSS trends with configured options
            trends = download_google_trends_rss(
                geo=self.region,
                include_images=self.include_images,
                include_articles=self.include_articles,
                max_articles_per_trend=self.max_articles_per_trend
            )

            if not trends:
                logger.warning("No trends found!")
                return pd.DataFrame(columns=['keyword', 'trend_score', 'traffic',
                                           'published', 'news_count', 'image_source'])

            # Process trends
            all_keywords = []
            for trend in trends[:top_n]:
                # Parse traffic (e.g., "200+", "1000+", "500+")
                traffic_str = trend.get('traffic', '0+')
                traffic_num = self._parse_traffic(traffic_str)

                # Count news articles
                news_count = len(trend.get('news_articles', []))

                # Get image source
                image_source = None
                if 'image' in trend and trend['image']:
                    image_source = trend['image'].get('source')

                all_keywords.append({
                    'keyword': trend.get('trend', ''),
                    'trend_score': traffic_num,  # Use traffic as trend score
                    'traffic': traffic_str,
                    'published': trend.get('published', ''),
                    'news_count': news_count,
                    'image_source': image_source,
                    'type': 'trending_now'
                })

            df = pd.DataFrame(all_keywords)

            # Sort by trend score
            df = df.sort_values('trend_score', ascending=False)

            # Add timeframe
            df['timeframe'] = 'now'

            # Reset index
            df = df.reset_index(drop=True)

            logger.info(f"\n✅ Extracted {len(df)} trending searches")
            logger.info(f"{'='*60}\n")

            # Display top 10
            logger.info("Top 10 trending searches:")
            for idx, row in df.head(10).iterrows():
                news_info = f", {row['news_count']} news" if row['news_count'] > 0 else ""
                logger.info(f"  {idx+1:2d}. {row['keyword']:30s} "
                          f"({row['traffic']}{news_info})")

            return df

        except Exception as e:
            logger.error(f"❌ Error fetching trends: {e}")
            return pd.DataFrame(columns=['keyword', 'trend_score', 'traffic',
                                       'published', 'news_count', 'image_source'])

    def _parse_traffic(self, traffic_str: str) -> float:
        """
        Parse traffic string to numeric score.

        Args:
            traffic_str: Traffic string (e.g., "200+", "1000+", "50+")

        Returns:
            Numeric traffic score
        """
        try:
            # Remove '+' and convert to number
            return float(traffic_str.replace('+', '').replace(',', ''))
        except:
            return 0.0

    def extract_category_trends(
        self,
        timeframe: str,
        categories: List[str] = None,
        broad_seeds: List[str] = None,
        top_n: int = 50,
        theme_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract trending keywords - uses RSS feed (no category filtering available).

        Note: RSS feed doesn't support category filtering, timeframe filtering,
        or seed keywords. This method wraps extract_trending_now() for compatibility
        with existing pipeline.

        Args:
            timeframe: Not used (RSS returns current trends only)
            categories: Not used (RSS doesn't support category filtering)
            broad_seeds: Not used (RSS doesn't support seed keywords)
            top_n: Return top N keywords
            theme_filter: Optional theme filter regex (post-processing)

        Returns:
            DataFrame with keywords and trend scores

        Example:
            >>> extractor = CategoryTrendsExtractorTrendspyg()
            >>> trends = extractor.extract_category_trends(
            ...     timeframe='now',  # Ignored
            ...     top_n=20
            ... )
        """
        logger.info(f"\nNote: trendspyg RSS feed returns real-time trends only")
        logger.info(f"Timeframe, categories, and seeds are not applicable\n")

        # Get trending now
        df = self.extract_trending_now(top_n=top_n)

        # Optional theme filtering
        if theme_filter and not df.empty:
            logger.info(f"Applying theme filter: {theme_filter}")
            original_count = len(df)
            df = df[df['keyword'].str.contains(theme_filter, case=False, na=False)]
            logger.info(f"After filtering: {len(df)}/{original_count} keywords")

        return df

    def extract_timeframe_trends(
        self,
        timeframe: str = 'now',
        discovery_mode: str = 'rss',
        categories: List[str] = None,
        theme_filter: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        High-level API: Extract trends - wraps extract_trending_now().

        Args:
            timeframe: Not used (RSS returns current trends only)
            discovery_mode: Not used (only 'rss' mode available)
            categories: Not used
            theme_filter: Optional theme filter
            top_n: Return top N keywords

        Returns:
            DataFrame with keywords and trend scores
        """
        return self.extract_category_trends(
            timeframe=timeframe,
            categories=categories,
            top_n=top_n,
            theme_filter=theme_filter
        )


def main():
    """
    Test trendspyg Category-Based Trends Extraction.
    """
    extractor = CategoryTrendsExtractorTrendspyg(region='HK')

    # Test: Real-time trending searches
    logger.info("\n" + "="*80)
    logger.info("TEST: Real-time Trending Searches (Hong Kong)")
    logger.info("="*80)

    trends = extractor.extract_trending_now(top_n=15)

    print("\n" + "="*80)
    print("Real-time Trending Searches:")
    print("="*80)
    print(trends[['keyword', 'traffic', 'news_count', 'image_source']].to_string(index=False))

    # Save
    output_file = 'data/trends_seasonal/trendspyg_realtime_hk.csv'
    trends.to_csv(output_file, index=False)
    logger.info(f"\n💾 Saved to: {output_file}")


if __name__ == '__main__':
    main()
