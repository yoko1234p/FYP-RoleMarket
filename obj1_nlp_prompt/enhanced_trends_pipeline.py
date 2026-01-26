"""
Enhanced Trends Pipeline - Complete trend extraction and optimization workflow

Integrations:
    1. CategoryTrendsExtractor (social media seed keywords - pytrends)
    2. CategoryTrendsExtractorTrendsPy (trendspy backend)
    3. CategoryTrendsExtractorTrendspyg (trendspyg RSS - ultra-fast)
    4. CategoryTrendsExtractorTrendspygCSV (trendspyg CSV - comprehensive)
    5. KeywordOptimizer (keyword filtering and optimization)
    6. PromptGenerator (LLM prompt generation)

Author: Product Manager (John), Developer (James)
Date: 2025-11-10
Version: 4.0 - trendspyg CSV Integration

Workflow:
    Real-time trends extraction → Keyword optimization → Prompt generation → Image generation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from obj1_nlp_prompt.category_trends_extractor import CategoryTrendsExtractor
from obj1_nlp_prompt.category_trends_extractor_trendspy import CategoryTrendsExtractorTrendsPy
from obj1_nlp_prompt.category_trends_extractor_trendspyg import CategoryTrendsExtractorTrendspyg
from obj1_nlp_prompt.category_trends_extractor_trendspyg_csv import CategoryTrendsExtractorTrendspygCSV
from obj1_nlp_prompt.keyword_optimizer import KeywordOptimizer
from obj1_nlp_prompt.prompt_generator import PromptGenerator
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTrendsPipeline:
    """
    Complete trends extraction and prompt generation pipeline.

    Features:
        - Real-time trending searches (trendspyg RSS - ultra-fast)
        - Comprehensive trends data (trendspyg CSV - advanced filtering)
        - Social media seed keyword auto-discovery
        - Keyword filtering and optimization
        - LLM-based prompt generation
        - Support for 4 backends: trendspyg, trendspyg_csv, trendspy, pytrends

    Backend Comparison:
        - trendspyg (RSS): Ultra-fast (0.2-0.5s), 10-20 trends, no filtering
        - trendspyg_csv (CSV): Slower (~10s), 480 trends, category/time filtering
        - trendspy: Better rate limiting than pytrends
        - pytrends: Original backend (archived, may have issues)
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
        Initialize enhanced trends pipeline.

        Args:
            region: Google Trends region code (default: 'HK')
            lang: Language code (default: 'zh-TW')
            backend: Trends backend
                - 'trendspyg': RSS mode - ultra-fast, real-time, 10-20 trends (RECOMMENDED for speed)
                - 'trendspyg_csv': CSV mode - comprehensive, 480 trends, filtering (RECOMMENDED for data)
                - 'trendspy': Better rate limiting than pytrends
                - 'pytrends': Original backend (archived, may have issues)
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
        self.backend = backend

        # Initialize trends extractor based on backend
        if backend == 'trendspyg':
            self.trends_extractor = CategoryTrendsExtractorTrendspyg(
                region=region,
                lang=lang,
                include_images=include_images,
                include_articles=include_articles,
                max_articles_per_trend=max_articles_per_trend
            )
            logger.info("EnhancedTrendsPipeline initialized with trendspyg RSS backend (ultra-fast)")
        elif backend == 'trendspyg_csv':
            self.trends_extractor = CategoryTrendsExtractorTrendspygCSV(
                region=region,
                category=category,
                hours=hours,
                active_only=active_only,
                sort_by=sort_by,
                headless=headless
            )
            logger.info(f"EnhancedTrendsPipeline initialized with trendspyg CSV backend")
            logger.info(f"  Category: {category}, Hours: {hours}, Active Only: {active_only}")
        elif backend == 'trendspy':
            self.trends_extractor = CategoryTrendsExtractorTrendsPy(
                region=region,
                lang=lang,
                proxy=proxy,
                request_delay=request_delay
            )
            logger.info(f"EnhancedTrendsPipeline initialized with trendspy backend (delay: {request_delay}s)")
        else:
            self.trends_extractor = CategoryTrendsExtractor(region=region, lang=lang)
            logger.info("EnhancedTrendsPipeline initialized with pytrends backend")

        self.keyword_optimizer = KeywordOptimizer()

        # Use absolute paths for PromptGenerator
        base_dir = Path(__file__).parent.parent
        self.prompt_generator = PromptGenerator(
            template_path=str(base_dir / 'obj1_nlp_prompt' / 'templates' / 'prompt_template.txt'),
            character_desc_path=str(base_dir / 'data' / 'character_descriptions' / 'lulu_pig.txt')
        )

    def extract_and_optimize_trends(
        self,
        timeframe: str,
        theme_filter: str = None,
        categories: List[str] = ['arts_entertainment', 'shopping', 'toys'],
        top_n_raw: int = 50,
        top_n_optimized: int = 15,
        min_visual_score: float = 2.0,
        output_dir: str = 'data/trends_seasonal'
    ) -> Dict:
        """
        提取並優化趨勢關鍵字。

        Args:
            timeframe: 時間範圍 (e.g., '2024-11-01 2024-12-31')
            theme_filter: 主題過濾 (e.g., 'christmas|聖誕')
            categories: Google Trends categories
            top_n_raw: 原始提取關鍵字數量
            top_n_optimized: 優化後保留數量
            min_visual_score: 最低視覺化分數
            output_dir: 輸出目錄

        Returns:
            Dictionary with:
                - raw_trends: 原始趨勢 DataFrame
                - optimized_trends: 優化後趨勢 DataFrame
                - formatted_keywords: 格式化關鍵字（準備提交 GPT）
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Enhanced Trends Pipeline - 提取與優化")
        logger.info(f"{'='*80}\n")

        # Step 1: 提取社交媒體趨勢
        logger.info("Step 1: 提取社交媒體趨勢...")
        raw_trends = self.trends_extractor.extract_category_trends(
            timeframe=timeframe,
            categories=categories,
            theme_filter=theme_filter,
            top_n=top_n_raw
        )

        if raw_trends.empty:
            logger.warning("❌ 未提取到任何趨勢關鍵字！")
            return {
                'raw_trends': raw_trends,
                'optimized_trends': pd.DataFrame(),
                'formatted_keywords': {'keywords': [], 'concepts': [], 'summary': {}}
            }

        # Save raw trends
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        raw_path = f"{output_dir}/raw_trends_{timeframe.replace(' ', '_').replace('-', '')}.csv"
        raw_trends.to_csv(raw_path, index=False)
        logger.info(f"💾 Raw trends saved: {raw_path}\n")

        # Step 2: 優化關鍵字
        logger.info("Step 2: 優化關鍵字...")
        optimized_trends = self.keyword_optimizer.optimize_keywords(
            raw_trends,
            top_n=top_n_optimized,
            min_visual_score=min_visual_score
        )

        # Save optimized trends
        opt_path = f"{output_dir}/optimized_trends_{timeframe.replace(' ', '_').replace('-', '')}.csv"
        optimized_trends.to_csv(opt_path, index=False)
        logger.info(f"💾 Optimized trends saved: {opt_path}\n")

        # Step 3: 格式化為 GPT 輸入
        logger.info("Step 3: 格式化關鍵字...")
        formatted_keywords = self.keyword_optimizer.format_for_prompt(optimized_trends)

        logger.info(f"\n{'='*80}")
        logger.info(f"Pipeline 完成")
        logger.info(f"{'='*80}")
        logger.info(f"原始關鍵字: {len(raw_trends)}")
        logger.info(f"優化後關鍵字: {len(optimized_trends)}")
        logger.info(f"準備提交 GPT: {len(formatted_keywords['keywords'])} keywords")
        logger.info(f"{'='*80}\n")

        return {
            'raw_trends': raw_trends,
            'optimized_trends': optimized_trends,
            'formatted_keywords': formatted_keywords
        }

    def generate_prompts_from_trends(
        self,
        formatted_keywords: Dict,
        theme: str,
        n_variations: int = 4,
        output_dir: str = 'data/prompts_enhanced'
    ) -> List[Dict]:
        """
        使用優化後的關鍵字生成 Prompts。

        Args:
            formatted_keywords: 格式化的關鍵字（從 extract_and_optimize_trends 獲得）
            theme: 主題名稱 (e.g., 'Christmas')
            n_variations: Prompt 變體數量
            output_dir: 輸出目錄

        Returns:
            List of prompt dictionaries
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Prompt 生成")
        logger.info(f"{'='*80}\n")

        keywords = formatted_keywords['keywords']
        concepts = formatted_keywords['concepts']

        logger.info(f"Theme: {theme}")
        logger.info(f"Keywords ({len(keywords)}): {', '.join(keywords)}")
        logger.info(f"Visual Concepts: {', '.join(concepts[:5])}...")

        # Generate prompts
        prompts = self.prompt_generator.generate_variations(
            theme=theme,
            keywords=keywords,
            n=n_variations
        )

        # Save prompts
        self.prompt_generator.save_prompts(prompts, theme, output_dir=output_dir)

        logger.info(f"\n✅ 成功生成 {len(prompts)} 個 prompts")
        logger.info(f"💾 Prompts saved to: {output_dir}")
        logger.info(f"{'='*80}\n")

        return prompts

    def run_full_pipeline(
        self,
        theme: str,
        timeframe: str,
        theme_filter: str = None,
        n_prompts: int = 4,
        output_dir_trends: str = 'data/trends_seasonal',
        output_dir_prompts: str = 'data/prompts_enhanced'
    ) -> Dict:
        """
        執行完整 Pipeline：趨勢提取 → 優化 → Prompt 生成。

        Args:
            theme: 主題名稱 (e.g., 'Christmas')
            timeframe: 時間範圍 (e.g., '2024-11-01 2024-12-31')
            theme_filter: 主題過濾 regex (optional)
            n_prompts: 生成 Prompt 數量
            output_dir_trends: 趨勢輸出目錄
            output_dir_prompts: Prompts 輸出目錄

        Returns:
            Dictionary with all pipeline results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Enhanced Trends Pipeline - Full Execution")
        logger.info(f"{'='*80}")
        logger.info(f"Theme: {theme}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Theme Filter: {theme_filter or 'None'}")
        logger.info(f"{'='*80}\n")

        # Step 1+2: Extract and optimize trends
        trends_result = self.extract_and_optimize_trends(
            timeframe=timeframe,
            theme_filter=theme_filter,
            output_dir=output_dir_trends
        )

        if trends_result['optimized_trends'].empty:
            logger.error("❌ Pipeline 失敗：未獲得優化後關鍵字")
            return trends_result

        # Step 3: Generate prompts
        prompts = self.generate_prompts_from_trends(
            formatted_keywords=trends_result['formatted_keywords'],
            theme=theme,
            n_variations=n_prompts,
            output_dir=output_dir_prompts
        )

        # Complete result
        result = {
            **trends_result,
            'prompts': prompts,
            'pipeline_summary': {
                'theme': theme,
                'timeframe': timeframe,
                'raw_keywords_count': len(trends_result['raw_trends']),
                'optimized_keywords_count': len(trends_result['optimized_trends']),
                'prompts_generated': len(prompts)
            }
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"Pipeline 執行完成")
        logger.info(f"{'='*80}")
        logger.info(f"✅ 原始關鍵字: {result['pipeline_summary']['raw_keywords_count']}")
        logger.info(f"✅ 優化關鍵字: {result['pipeline_summary']['optimized_keywords_count']}")
        logger.info(f"✅ 生成 Prompts: {result['pipeline_summary']['prompts_generated']}")
        logger.info(f"{'='*80}\n")

        return result


def main():
    """
    測試完整 Enhanced Trends Pipeline.
    """
    # Initialize pipeline
    pipeline = EnhancedTrendsPipeline()

    # Run full pipeline for Christmas
    result = pipeline.run_full_pipeline(
        theme='Christmas',
        timeframe='2024-11-01 2024-12-31',
        theme_filter=None,  # 不過濾，使用所有社交媒體趨勢
        n_prompts=4,
        output_dir_trends='data/trends_seasonal',
        output_dir_prompts='data/prompts_enhanced'
    )

    # Display results summary
    print("\n" + "="*80)
    print("Pipeline 執行總結")
    print("="*80)
    print(f"主題: {result['pipeline_summary']['theme']}")
    print(f"時段: {result['pipeline_summary']['timeframe']}")
    print(f"原始關鍵字: {result['pipeline_summary']['raw_keywords_count']}")
    print(f"優化關鍵字: {result['pipeline_summary']['optimized_keywords_count']}")
    print(f"生成 Prompts: {result['pipeline_summary']['prompts_generated']}")
    print("="*80)

    # Display prompts
    print("\n生成的 Prompts:")
    for i, prompt_data in enumerate(result['prompts'], 1):
        print(f"\nVariation {i}:")
        print(f"  Length: {prompt_data['length']} words")
        print(f"  Prompt: {prompt_data['prompt'][:150]}...")


if __name__ == '__main__':
    main()
