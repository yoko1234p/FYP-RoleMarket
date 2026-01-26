"""
Category-Based Trends Extractor using trendspyg CSV Mode

Enhanced version using trendspyg CSV download for comprehensive Google Trends data.

Key Advantages over RSS mode:
- Comprehensive dataset (~480 trends vs ~10-20)
- Category filtering (20 categories)
- Time period filtering (4h, 24h, 48h, 7d)
- Active trends only toggle
- Search volume data
- Trend breakdown analysis
- Keyword search capability

Trade-offs:
- Slower than RSS (~10s vs 0.2-0.5s)
- Requires Chrome browser + Selenium
- No rich media (images, news articles)

Author: Developer (James)
Date: 2025-11-10
Version: 4.0 - trendspyg CSV Integration
"""

from trendspyg import download_google_trends_csv
from trendspyg.config import CATEGORIES, TIME_PERIODS
import pandas as pd
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryTrendsExtractorTrendspygCSV:
    """
    Extract comprehensive trends using trendspyg CSV download.

    Advantages:
    - Large dataset (~480 trends)
    - Category filtering (20 categories)
    - Time period filtering (4h, 24h, 48h, 7d)
    - Search volume data
    - Keyword search capability
    """

    def __init__(
        self,
        region: str = 'HK',
        category: str = 'all',
        hours: int = 24,
        active_only: bool = False,
        sort_by: str = 'relevance',
        headless: bool = True
    ):
        """
        Initialize category-based trends extractor with trendspyg CSV mode.

        Args:
            region: Google Trends region code (default: HK)
            category: Filter by category (default: 'all')
                     Options: 'all', 'autos', 'beauty', 'business', 'climate',
                             'entertainment', 'food', 'games', 'health', 'hobbies',
                             'jobs', 'law', 'other', 'pets', 'politics', 'science',
                             'shopping', 'sports', 'technology', 'travel'
            hours: Time period in hours (default: 24)
                  Options: 4, 24, 48, 168 (7 days)
            active_only: Only return rising/active trends (default: False)
            sort_by: Sort order (default: 'relevance')
                    Options: 'relevance', 'title', 'volume', 'recency'
            headless: Run browser in headless mode (default: True)
        """
        self.region = region
        self.category = category
        self.hours = hours
        self.active_only = active_only
        self.sort_by = sort_by
        self.headless = headless

        # Validate inputs
        if category not in CATEGORIES:
            logger.warning(f"Invalid category '{category}', using 'all'")
            self.category = 'all'

        if hours not in [4, 24, 48, 168]:
            logger.warning(f"Invalid hours {hours}, using 24")
            self.hours = 24

        logger.info(f"CategoryTrendsExtractorTrendspygCSV initialized:")
        logger.info(f"  Region: {region}")
        logger.info(f"  Category: {category}")
        logger.info(f"  Time Period: {hours}h")
        logger.info(f"  Active Only: {active_only}")
        logger.info(f"  Sort By: {sort_by}")

    def extract_trending_now(
        self,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Extract comprehensive trending searches using CSV download.

        Args:
            top_n: Return top N trends (default: 50)
                  Note: CSV typically returns 400-500 trends

        Returns:
            DataFrame with columns: keyword, trend_score, traffic,
                                   search_volume, breakdown, timeframe

        Example:
            >>> extractor = CategoryTrendsExtractorTrendspygCSV(
            ...     region='US',
            ...     category='sports',
            ...     hours=24
            ... )
            >>> trends = extractor.extract_trending_now(top_n=50)
            >>> print(trends.head())
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"trendspyg CSV Download - Comprehensive Trends")
        logger.info(f"{'='*60}\n")
        logger.info(f"Region: {self.region}")
        logger.info(f"Category: {self.category}")
        logger.info(f"Time Period: {self.hours}h")
        logger.info(f"Active Only: {self.active_only}")
        logger.info(f"Downloading trends (this may take ~10 seconds)...")

        try:
            # Download CSV trends with configured options
            df = download_google_trends_csv(
                geo=self.region,
                hours=self.hours,
                category=self.category,
                active_only=self.active_only,
                sort_by=self.sort_by,
                headless=self.headless,
                output_format='dataframe'
            )

            if df is None or df.empty:
                logger.warning("No trends found!")
                return pd.DataFrame(columns=['keyword', 'trend_score', 'traffic',
                                           'search_volume', 'breakdown', 'timeframe'])

            logger.info(f"✅ Downloaded {len(df)} trends")

            # Rename columns to match pipeline expectations
            df_processed = pd.DataFrame()
            df_processed['keyword'] = df['Trends']

            # Parse search volume to numeric score
            df_processed['search_volume'] = df['Search volume']
            df_processed['trend_score'] = df['Search volume'].apply(self._parse_search_volume)

            # Keep original search volume string for display
            df_processed['traffic'] = df['Search volume']

            # Trend breakdown (related searches)
            df_processed['breakdown'] = df.get('Trend breakdown', '')

            # Add timeframe info
            time_label = self._get_time_label(self.hours)
            df_processed['timeframe'] = time_label
            df_processed['type'] = 'trending_now'

            # Sort by trend score
            df_processed = df_processed.sort_values('trend_score', ascending=False)

            # Limit to top_n
            df_processed = df_processed.head(top_n)

            # Reset index
            df_processed = df_processed.reset_index(drop=True)

            logger.info(f"\n✅ Processed {len(df_processed)} trends")
            logger.info(f"{'='*60}\n")

            # Display top 10
            logger.info("Top 10 trending searches:")
            for idx, row in df_processed.head(10).iterrows():
                logger.info(f"  {idx+1:2d}. {row['keyword']:35s} ({row['traffic']})")

            return df_processed

        except Exception as e:
            logger.error(f"❌ Error downloading trends: {e}")
            logger.error(f"Make sure Chrome browser is installed")

            # Provide helpful timeout suggestions
            error_str = str(e)
            if 'timeout' in error_str.lower() or 'TimeoutException' in error_str:
                logger.error("\n" + "="*60)
                logger.error("⏱️ CSV Mode Timeout - Possible Solutions:")
                logger.error("="*60)
                logger.error("1. 🌐 Check internet connection and speed")
                logger.error("2. 🔄 Try again - Google Trends might be temporarily slow")
                logger.error("3. ⚡ Use trendspyg RSS mode instead (0.5s, no browser needed)")
                logger.error("4. 🌍 Try different region (some regions load faster)")
                logger.error("5. ⏱️ Try shorter time period (4h instead of 168h)")
                logger.error("\n💡 Recommendation: Switch to trendspyg (RSS) backend for")
                logger.error("   ultra-fast extraction without browser automation")
                logger.error("="*60 + "\n")

            return pd.DataFrame(columns=['keyword', 'trend_score', 'traffic',
                                       'search_volume', 'breakdown', 'timeframe'])

    def _parse_search_volume(self, volume_str: str) -> float:
        """
        Parse search volume string to numeric score.

        Args:
            volume_str: Search volume string (e.g., "50K+", "100K+", "2M+")

        Returns:
            Numeric search volume score
        """
        try:
            # Remove '+' and whitespace
            volume_str = str(volume_str).replace('+', '').replace(',', '').strip()

            # Handle K (thousands) and M (millions)
            if 'K' in volume_str:
                return float(volume_str.replace('K', '')) * 1000
            elif 'M' in volume_str:
                return float(volume_str.replace('M', '')) * 1000000
            else:
                return float(volume_str)
        except:
            return 0.0

    def _get_time_label(self, hours: int) -> str:
        """Get time period label."""
        if hours == 4:
            return 'past_4h'
        elif hours == 24:
            return 'past_24h'
        elif hours == 48:
            return 'past_48h'
        elif hours == 168:
            return 'past_7d'
        else:
            return f'past_{hours}h'

    def search_keywords(
        self,
        keywords: List[str],
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Search for specific keywords in trending data.

        Args:
            keywords: List of keywords to search for
            top_n: Return top N matching trends

        Returns:
            DataFrame with matching trends

        Example:
            >>> extractor = CategoryTrendsExtractorTrendspygCSV(region='US')
            >>> results = extractor.search_keywords(['nba', 'football'], top_n=20)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Keyword Search: {', '.join(keywords)}")
        logger.info(f"{'='*60}\n")

        # Get all trends
        df = self.extract_trending_now(top_n=500)  # Get more for searching

        if df.empty:
            return df

        # Search for keywords (case-insensitive)
        pattern = '|'.join(keywords)
        matches = df[df['keyword'].str.contains(pattern, case=False, na=False)]

        logger.info(f"Found {len(matches)} matching trends")

        # Limit to top_n
        matches = matches.head(top_n)

        return matches

    def extract_category_trends(
        self,
        timeframe: str,
        categories: List[str] = None,
        broad_seeds: List[str] = None,
        top_n: int = 50,
        theme_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract trending keywords with optional filtering.

        Args:
            timeframe: Time period ('now', 'today', 'this_week')
                      Note: Actual filtering is done via hours parameter
            categories: Not used (category set in __init__)
            broad_seeds: Optional keyword seeds for filtering
            top_n: Return top N keywords
            theme_filter: Optional theme filter regex

        Returns:
            DataFrame with keywords and trend scores

        Example:
            >>> extractor = CategoryTrendsExtractorTrendspygCSV(
            ...     region='US',
            ...     category='sports',
            ...     hours=24
            ... )
            >>> trends = extractor.extract_category_trends(
            ...     timeframe='today',
            ...     broad_seeds=['basketball', 'football'],
            ...     top_n=30
            ... )
        """
        # Get trending now
        df = self.extract_trending_now(top_n=top_n * 2)  # Get extra for filtering

        if df.empty:
            return df

        # Apply seed keyword filtering
        if broad_seeds:
            logger.info(f"Filtering by seed keywords: {', '.join(broad_seeds)}")
            pattern = '|'.join(broad_seeds)
            original_count = len(df)
            df = df[df['keyword'].str.contains(pattern, case=False, na=False)]
            logger.info(f"After seed filtering: {len(df)}/{original_count} keywords")

        # Apply theme filtering
        if theme_filter:
            logger.info(f"Applying theme filter: {theme_filter}")
            original_count = len(df)
            df = df[df['keyword'].str.contains(theme_filter, case=False, na=False)]
            logger.info(f"After theme filtering: {len(df)}/{original_count} keywords")

        # Limit to top_n
        df = df.head(top_n)

        return df

    def extract_timeframe_trends(
        self,
        timeframe: str = 'today',
        discovery_mode: str = 'csv',
        categories: List[str] = None,
        theme_filter: Optional[str] = None,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        High-level API: Extract trends - wraps extract_category_trends().

        Args:
            timeframe: Time period (mapped to hours in __init__)
            discovery_mode: Not used (only 'csv' mode available)
            categories: Not used (category set in __init__)
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
    Test trendspyg CSV Category-Based Trends Extraction.
    """
    # Test 1: Sports trends from US
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Sports Trends (United States, Past 24h)")
    logger.info("="*80)

    extractor = CategoryTrendsExtractorTrendspygCSV(
        region='US',
        category='sports',
        hours=24,
        active_only=False
    )

    trends = extractor.extract_trending_now(top_n=20)

    print("\n" + "="*80)
    print("Sports Trends (US):")
    print("="*80)
    print(trends[['keyword', 'traffic', 'breakdown']].to_string(index=False))

    # Test 2: Keyword search
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Keyword Search - NBA, Football")
    logger.info("="*80)

    results = extractor.search_keywords(['nba', 'football'], top_n=10)

    print("\n" + "="*80)
    print("Keyword Search Results:")
    print("="*80)
    print(results[['keyword', 'traffic']].to_string(index=False))

    # Save
    output_file = 'data/trends_seasonal/trendspyg_csv_us_sports.csv'
    trends.to_csv(output_file, index=False)
    logger.info(f"\n💾 Saved to: {output_file}")


if __name__ == '__main__':
    main()
