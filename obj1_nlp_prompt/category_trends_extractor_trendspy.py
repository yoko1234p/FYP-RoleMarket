"""
Category-Based Trends Extractor using TrendsPy

Enhanced version using trendspy library for better performance and rate limiting handling.

Key Improvements over pytrends version:
- Simpler API (no build_payload required)
- Better rate limiting handling
- Proxy support to avoid 429 errors
- Batch processing for multiple keywords
- Access to real-time trending searches

Author: Developer (James)
Date: 2025-11-09
Version: 2.0 - TrendsPy Integration
"""

from trendspy import Trends
import pandas as pd
from typing import List, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryTrendsExtractorTrendsPy:
    """
    Extract trends using TrendsPy with Google Trends Categories + broad seed keywords.

    Advantages over pytrends:
    - Simpler API calls
    - Built-in proxy support
    - Better error handling
    - Batch processing capabilities
    """

    # Google Trends Category IDs
    # Reference: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    CATEGORIES = {
        'all': 0,                           # All categories
        'shopping': 18,                     # Shopping
        'arts_entertainment': 3,            # Arts & Entertainment
        'hobbies_leisure': 8,               # Hobbies & Leisure
        'games': 8,                         # Games
        'holidays': 19,                     # Holidays & Seasonal Events
        'gifts': 251,                       # Gifts & Special Event Items
        'toys': 237,                        # Toys
    }

    def __init__(
        self,
        region: str = 'HK',
        lang: str = 'zh-TW',
        proxy: Optional[str] = None,
        request_delay: float = 3.0
    ):
        """
        Initialize category-based trends extractor with TrendsPy.

        Args:
            region: Google Trends region code (default: HK)
            lang: Language code (default: zh-TW)
            proxy: Optional proxy URL (e.g., 'http://10.10.1.10:3128')
            request_delay: Delay between requests in seconds (default: 3.0)
                Increase this to avoid rate limiting (recommended: 2.0-5.0)
        """
        self.region = region
        self.lang = lang

        # Initialize TrendsPy with optional proxy and request delay
        if proxy:
            self.trends = Trends(proxy=proxy, request_delay=request_delay)
            logger.info(f"CategoryTrendsExtractorTrendsPy initialized with proxy: {proxy}, delay: {request_delay}s")
        else:
            self.trends = Trends(request_delay=request_delay)
            logger.info(f"CategoryTrendsExtractorTrendsPy initialized: region={region}, delay: {request_delay}s")

    def extract_category_trends(
        self,
        timeframe: str,
        categories: List[str] = ['holidays', 'gifts', 'toys'],
        broad_seeds: List[str] = None,
        top_n: int = 50,
        theme_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract trending keywords using categories and broad seed keywords.

        Args:
            timeframe: Time range (e.g., '2024-11-01 2024-12-31', 'today 12-m')
            categories: Category list (default: ['holidays', 'gifts', 'toys'])
            broad_seeds: Broad seed keywords (default: social media + entertainment)
            top_n: Return top N keywords
            theme_filter: Optional theme filter regex (e.g., 'christmas|xmas')

        Returns:
            DataFrame with columns: keyword, trend_score, category, timeframe

        Workflow:
            1. For each category, query with broad seeds
            2. Get related queries to expand keyword pool
            3. Merge all related queries
            4. Sort by trend score
            5. Optional theme filtering
            6. Return top N
        """
        if broad_seeds is None:
            # Social media + entertainment seed keywords (bilingual)
            broad_seeds = [
                # Social media platforms
                'instagram', '小紅書',
                # Trending topics
                'trending', '熱門', '爆紅',
                # Entertainment content
                '明星', 'viral'
            ]

        logger.info(f"\n{'='*60}")
        logger.info(f"TrendsPy Category-Based Trends Extraction")
        logger.info(f"{'='*60}\n")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Categories: {categories}")
        logger.info(f"Broad seeds: {broad_seeds}")
        logger.info(f"Theme filter: {theme_filter or 'None (all trends)'}\n")

        all_keywords = []

        # Process each category
        for category_name in categories:
            category_id = self.CATEGORIES.get(category_name, 0)
            logger.info(f"Processing category: {category_name} (ID: {category_id})")

            # Process seeds in batches (trendspy can handle more at once)
            # But we'll still be conservative to avoid rate limiting
            batch_size = 5
            seed_batches = [broad_seeds[i:i+batch_size]
                          for i in range(0, len(broad_seeds), batch_size)]

            for batch_idx, seed_batch in enumerate(seed_batches):
                logger.info(f"  Batch {batch_idx+1}/{len(seed_batches)}: {seed_batch}")

                try:
                    # Get related queries for each seed in the batch
                    for seed in seed_batch:
                        try:
                            # Get related queries using trendspy
                            related = self.trends.related_queries(
                                seed,
                                timeframe=timeframe,
                                geo=self.region,
                                cat=str(category_id)
                            )

                            # Extract top queries
                            if 'top' in related and related['top'] is not None:
                                top_df = related['top']
                                if not top_df.empty:
                                    for _, row in top_df.iterrows():
                                        all_keywords.append({
                                            'keyword': row['query'],
                                            'trend_score': row['value'],
                                            'category': category_name,
                                            'type': 'top'
                                        })

                            # Extract rising queries
                            if 'rising' in related and related['rising'] is not None:
                                rising_df = related['rising']
                                if not rising_df.empty:
                                    for _, row in rising_df.iterrows():
                                        # Handle rising values (may be "Breakout" or percentage)
                                        value = row['value']
                                        if isinstance(value, str):
                                            if value == 'Breakout':
                                                value = 1000  # High score for breakout
                                            else:
                                                # Try to parse percentage
                                                try:
                                                    value = float(value.replace('%', '').replace('+', ''))
                                                except:
                                                    value = 100  # Default value

                                        all_keywords.append({
                                            'keyword': row['query'],
                                            'trend_score': value,
                                            'category': category_name,
                                            'type': 'rising'
                                        })

                            time.sleep(1)  # Small delay between individual queries

                        except Exception as e:
                            logger.warning(f"    ⚠️ Error with seed '{seed}': {e}")
                            continue

                    logger.info(f"    → Batch completed")
                    time.sleep(2)  # Rate limiting between batches

                except Exception as e:
                    logger.error(f"    ❌ Error processing batch: {e}")
                    continue

            logger.info(f"  ✅ Category {category_name} completed\n")
            time.sleep(3)  # Rate limiting between categories

        # Merge and deduplicate
        if not all_keywords:
            logger.warning("No keywords found!")
            return pd.DataFrame(columns=['keyword', 'trend_score', 'category', 'timeframe'])

        df = pd.DataFrame(all_keywords)

        # Deduplicate (keep highest score)
        df = df.sort_values('trend_score', ascending=False).drop_duplicates('keyword')

        logger.info(f"\n{'='*60}")
        logger.info(f"Total unique keywords: {len(df)}")

        # Optional theme filtering
        if theme_filter:
            logger.info(f"Applying theme filter: {theme_filter}")
            df = df[df['keyword'].str.contains(theme_filter, case=False, na=False)]
            logger.info(f"After filtering: {len(df)} keywords")

        # Sort by score and get top N
        df = df.sort_values('trend_score', ascending=False).head(top_n)

        # Add timeframe
        df['timeframe'] = timeframe

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"\n✅ Extracted Top {len(df)} keywords")
        logger.info(f"{'='*60}\n")

        # Display top 10
        logger.info("Top 10 keywords:")
        for idx, row in df.head(10).iterrows():
            logger.info(f"  {idx+1:2d}. {row['keyword']:30s} "
                       f"(Score: {row['trend_score']:6.1f}, {row['category']})")

        return df

    def get_trending_now(
        self,
        geo: str = None,
        max_trends: int = 20
    ) -> List[Dict]:
        """
        Get real-time trending searches (NEW feature from trendspy).

        This is a unique feature of trendspy not available in pytrends.

        Args:
            geo: Region code (default: self.region)
            max_trends: Maximum number of trends to return

        Returns:
            List of trending topics with metadata
        """
        if geo is None:
            geo = self.region

        logger.info(f"Fetching real-time trending searches for {geo}...")

        try:
            trends = self.trends.trending_now(geo=geo)

            trending_list = []
            for i, trend in enumerate(trends[:max_trends]):
                trending_list.append({
                    'keyword': trend.title,
                    'traffic': trend.traffic if hasattr(trend, 'traffic') else None,
                    'rank': i + 1
                })

            logger.info(f"✅ Found {len(trending_list)} trending topics")
            return trending_list

        except Exception as e:
            logger.error(f"❌ Error fetching trending searches: {e}")
            return []

    def extract_timeframe_trends(
        self,
        timeframe: str,
        discovery_mode: str = 'category',
        categories: List[str] = ['holidays', 'gifts', 'toys'],
        theme_filter: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        High-level API: Extract trends for a specific timeframe.

        Args:
            timeframe: Time range (e.g., '2024-11-01 2024-12-31', 'today 12-m')
            discovery_mode: Discovery mode ('category' or 'broad')
                - 'category': Use B+C hybrid approach (recommended)
                - 'broad': Use only broad seeds (no category filtering)
            categories: Category list
            theme_filter: Optional theme filter
            top_n: Return top N keywords

        Returns:
            DataFrame with keywords and trend scores
        """
        if discovery_mode == 'category':
            return self.extract_category_trends(
                timeframe=timeframe,
                categories=categories,
                top_n=top_n,
                theme_filter=theme_filter
            )
        else:
            # Broad mode: no category filtering
            return self.extract_category_trends(
                timeframe=timeframe,
                categories=['all'],
                top_n=top_n,
                theme_filter=theme_filter
            )


def main():
    """
    Test TrendsPy Category-Based Trends Extraction.
    """
    # Initialize extractor (with optional proxy)
    extractor = CategoryTrendsExtractorTrendsPy()
    # For proxy: extractor = CategoryTrendsExtractorTrendsPy(proxy='http://10.10.1.10:3128')

    # Test 1: Social media trends (Nov-Dec) - No filter
    logger.info("\n" + "="*80)
    logger.info("TEST 1: TrendsPy Social Media Trends (Nov-Dec) - All Topics")
    logger.info("="*80)

    social_media_seeds = [
        # Social media platforms
        'instagram', 'tiktok', '小紅書',
        # Trending topics
        'trending', 'viral', '熱門', '爆紅',
        # Entertainment
        '明星', 'meme'
    ]

    try:
        social_trends = extractor.extract_category_trends(
            timeframe='2024-11-01 2024-12-31',
            categories=['arts_entertainment', 'shopping', 'toys'],
            broad_seeds=social_media_seeds,
            theme_filter=None,
            top_n=30
        )

        print("\n" + "="*80)
        print("TrendsPy Social Media Trends (All Topics):")
        print("="*80)
        print(social_trends.head(20).to_string(index=False))

        # Save
        social_trends.to_csv('data/trends_seasonal/trendspy_nov_dec_all.csv', index=False)
        logger.info(f"\n💾 Saved to: data/trends_seasonal/trendspy_nov_dec_all.csv")

    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")

    # Test 2: Real-time trending searches (NEW feature)
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Real-time Trending Searches (TrendsPy Exclusive)")
    logger.info("="*80)

    try:
        trending_now = extractor.get_trending_now(max_trends=10)

        print("\n" + "="*80)
        print("Real-time Trending Searches:")
        print("="*80)
        for trend in trending_now:
            print(f"  {trend['rank']:2d}. {trend['keyword']}")

    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")


if __name__ == '__main__':
    main()
