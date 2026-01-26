"""
Enhanced Trends Pipeline Wrapper for Streamlit Integration

Integrates EnhancedTrendsPipeline (Solution 3 + CSV Mode) into Streamlit Web App.

Key Features:
- Real-time trending searches (trendspyg RSS backend - ultra-fast)
- Comprehensive trends data (trendspyg CSV backend - advanced filtering)
- Theme only used for LLM prompt generation
- Support for 4 backends: trendspyg, trendspyg_csv, trendspy, pytrends

Author: Developer (James)
Date: 2025-11-10
Version: 6.0 - trendspyg CSV Integration
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj1_nlp_prompt.enhanced_trends_pipeline import EnhancedTrendsPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTrendsError(Exception):
    """Enhanced trends pipeline error."""
    pass


class EnhancedTrendsWrapper:
    """
    Wrapper for EnhancedTrendsPipeline to integrate with Streamlit.

    Features:
        - Real-time trending searches (trendspyg RSS - ultra-fast, 10-20 trends)
        - Comprehensive trends data (trendspyg CSV - 480 trends, category/time filtering)
        - No rate limiting issues
        - Keyword optimization and filtering
        - Theme only used for prompt generation context
        - Support for 4 backends: trendspyg, trendspyg_csv, trendspy, pytrends

    Backend Comparison:
        - trendspyg: RSS mode, 0.2-0.5s, 10-20 trends, no filtering (best for speed)
        - trendspyg_csv: CSV mode, ~10s, 480 trends, category/time filtering (best for data)
        - trendspy: Better rate limiting than pytrends
        - pytrends: Original backend (archived)
    """

    def __init__(
        self,
        region: str = 'HK',
        lang: str = 'zh-TW',
        backend: str = 'trendspyg',
        proxy: Optional[str] = None,
        request_delay: float = 3.0,
        # trendspyg RSS specific options
        include_images: bool = True,
        include_articles: bool = True,
        max_articles_per_trend: int = 5,
        # trendspyg CSV specific options
        category: str = 'all',
        hours: int = 24,
        active_only: bool = False,
        sort_by: str = 'relevance',
        headless: bool = True
    ):
        """
        Initialize enhanced trends wrapper.

        Args:
            region: Google Trends region code (default: 'HK')
            lang: Language code (default: 'zh-TW')
            backend: Trends backend
                - 'trendspyg': RSS mode - ultra-fast, real-time, 10-20 trends (best for speed)
                - 'trendspyg_csv': CSV mode - comprehensive, 480 trends, filtering (best for data)
                - 'trendspy': Better rate limiting than pytrends
                - 'pytrends': Original backend (archived)
            proxy: Optional proxy URL (only for trendspy)
            request_delay: Delay between requests in seconds (only for trendspy)

            # RSS Mode Options (backend='trendspyg'):
            include_images: Include images in trends (default: True)
            include_articles: Include news articles (default: True)
            max_articles_per_trend: Max news articles per trend (default: 5)

            # CSV Mode Options (backend='trendspyg_csv'):
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
        self.lang = lang
        self.backend = backend

        try:
            self.pipeline = EnhancedTrendsPipeline(
                region=region,
                lang=lang,
                backend=backend,
                proxy=proxy,
                request_delay=request_delay,
                # RSS options
                include_images=include_images,
                include_articles=include_articles,
                max_articles_per_trend=max_articles_per_trend,
                # CSV options
                category=category,
                hours=hours,
                active_only=active_only,
                sort_by=sort_by,
                headless=headless
            )
            logger.info(f"EnhancedTrendsWrapper initialized: "
                       f"region={region}, backend={backend}")
            if backend == 'trendspyg_csv':
                logger.info(f"  CSV mode: category={category}, hours={hours}, "
                          f"active_only={active_only}")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedTrendsPipeline: {e}")
            raise EnhancedTrendsError(f"Pipeline initialization failed: {e}")

    def extract_trends(
        self,
        timeframe: str,
        top_n: int = 20,
        categories: List[str] = None,
        theme_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract trending keywords for a given timeframe.

        Core Logic:
        1. Use broad social media seed keywords
        2. Filter through Google Trends Categories
        3. Expand via related_queries
        4. Optimize keywords and calculate visual scores
        5. Optional theme filtering

        Args:
            timeframe: Time range (e.g., 'today 12-m', '2024-11-01 2024-12-31')
            top_n: Return top N keywords
            categories: Google Trends categories (default: ['arts_entertainment', 'shopping', 'toys'])
            theme_filter: Optional theme filter regex (e.g., 'christmas|xmas')

        Returns:
            List of keyword dictionaries with:
                - keyword: Keyword string
                - trend_score: Trend score
                - visual_score: Visual relevance score
                - combined_score: Combined score
                - rank: Ranking

        Raises:
            EnhancedTrendsError: If extraction fails
        """
        if categories is None:
            categories = ['arts_entertainment', 'shopping', 'toys']

        logger.info(f"Extracting trends for timeframe: {timeframe}")
        logger.info(f"Categories: {categories}")
        logger.info(f"Theme filter: {theme_filter or 'None (all trends)'}")

        # Adjust min_visual_score based on backend
        # trendspyg (RSS) returns news/event-oriented trends → lower threshold
        # trendspyg_csv returns comprehensive trends → lower threshold
        # Other backends return design-oriented trends → higher threshold
        if self.backend in ['trendspyg', 'trendspyg_csv']:
            min_visual_score = 0.5  # Lower for real-time/comprehensive trends
            logger.info(f"Using {self.backend} backend: lowered visual score threshold to 0.5")
        else:
            min_visual_score = 1.5  # Higher for design-oriented trends

        try:
            # Extract and optimize trends
            result = self.pipeline.extract_and_optimize_trends(
                timeframe=timeframe,
                theme_filter=theme_filter,
                categories=categories,
                top_n_raw=50,  # Extract more for better filtering
                top_n_optimized=top_n,
                min_visual_score=min_visual_score,
                output_dir='data/trends_seasonal'
            )

            if result['optimized_trends'].empty:
                logger.warning("No trends extracted")
                return []

            # Format for frontend
            optimized_df = result['optimized_trends']
            keywords = []

            for idx, row in optimized_df.iterrows():
                keywords.append({
                    'keyword': row['keyword'],
                    'trend_score': float(row['trend_score']),
                    'visual_score': float(row['visual_score']),
                    'combined_score': float(row['combined_score']),
                    'rank': idx + 1,
                    'is_high_trend': row['combined_score'] >= 7.0  # High combined score
                })

            logger.info(f"✅ Extracted {len(keywords)} optimized keywords")
            return keywords

        except Exception as e:
            logger.error(f"Trend extraction failed: {e}")

            # Check if it's a rate limiting error (429)
            error_str = str(e)
            is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'too many' in error_str.lower()

            if is_rate_limit:
                raise EnhancedTrendsError(
                    f"⚠️ Google Trends Rate Limiting Detected (Error 429)\n\n"
                    f"Google Trends has temporarily blocked requests due to too many API calls.\n\n"
                    f"📋 Quick Solutions:\n"
                    f"1. ✍️ Use 'Manual Input' method instead (recommended now)\n"
                    f"2. ⏰ Wait 2-3 minutes and try 'Auto Extract' again\n"
                    f"3. 🔄 Reduce the number of keywords (Top N)\n\n"
                    f"💡 Why this happens:\n"
                    f"- Google limits: ~10-15 requests/minute per IP\n"
                    f"- Each extraction sends multiple requests (3+ categories)\n"
                    f"- Previous extractions may still count toward limit\n\n"
                    f"🎯 Recommended: Switch to 'Manual Input' tab to continue immediately!"
                )
            else:
                raise EnhancedTrendsError(
                    f"Trend extraction failed: {str(e)}\n\n"
                    f"Possible causes:\n"
                    f"1. Insufficient trend data in selected timeframe\n"
                    f"2. Network connection issues\n"
                    f"3. Invalid timeframe format\n\n"
                    f"Suggestions:\n"
                    f"- Try a different timeframe\n"
                    f"- Check network connection\n"
                    f"- Or use 'Manual Input' method"
                )

    def generate_prompt_with_theme(
        self,
        keywords: List[str],
        theme: str,
        character_name: str = "Lulu Pig",
        character_desc: str = "Cute pink pig"
    ) -> str:
        """
        Generate AI prompt using keywords and theme.

        Note: Theme is only used for LLM prompt generation, does not affect trend extraction.

        Args:
            keywords: List of keywords
            theme: Theme name (e.g., 'Christmas', 'Cozy Winter')
            character_name: Character name
            character_desc: Character description

        Returns:
            Generated prompt string

        Raises:
            EnhancedTrendsError: If generation fails
        """
        logger.info(f"Generating prompt with theme: {theme}")
        logger.info(f"Keywords: {keywords}")

        try:
            # Format keywords for pipeline
            formatted_keywords = {
                'keywords': keywords,
                'concepts': [],  # Concepts extracted during optimization
                'summary': {
                    'total_keywords': len(keywords),
                    'theme': theme
                }
            }

            # Generate prompts (n=1 for Streamlit)
            prompts = self.pipeline.generate_prompts_from_trends(
                formatted_keywords=formatted_keywords,
                theme=theme,
                n_variations=1,
                output_dir='data/prompts_enhanced'
            )

            if not prompts:
                raise EnhancedTrendsError("Prompt generation returned empty result")

            # Return first prompt
            generated_prompt = prompts[0]['prompt']
            logger.info(f"✅ Prompt generated successfully ({len(generated_prompt)} chars)")

            return generated_prompt

        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            raise EnhancedTrendsError(f"Prompt generation failed: {str(e)}")


def main():
    """
    Test EnhancedTrendsWrapper.
    """
    import json

    print("="*80)
    print("EnhancedTrendsWrapper Test")
    print("="*80)

    # Initialize wrapper
    wrapper = EnhancedTrendsWrapper()

    # Test 1: Extract trends for a timeframe (no theme filter)
    print("\nTest 1: Extract trends for Nov-Dec 2024 (no filter)")
    print("-"*80)

    trends = wrapper.extract_trends(
        timeframe='2024-11-01 2024-12-31',
        top_n=15,
        theme_filter=None
    )

    print(f"\nExtracted {len(trends)} trends:")
    for i, trend in enumerate(trends[:10], 1):
        print(f"  {i:2d}. {trend['keyword']:30s} "
              f"(Trend: {trend['trend_score']:5.1f}, "
              f"Visual: {trend['visual_score']:4.1f}, "
              f"Total: {trend['combined_score']:4.1f})")

    # Test 2: Generate prompt with theme
    if trends:
        print("\n" + "="*80)
        print("Test 2: Generate prompt with theme")
        print("-"*80)

        selected_keywords = [t['keyword'] for t in trends[:8]]

        prompt = wrapper.generate_prompt_with_theme(
            keywords=selected_keywords,
            theme='Cozy Christmas',
            character_name='Lulu Pig',
            character_desc='Cute pink pig with big eyes'
        )

        print(f"\nGenerated Prompt:")
        print("-"*80)
        print(prompt)

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == '__main__':
    main()
