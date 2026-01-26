"""
Category-Based Trends Extractor (B+C 混合方案)

結合 Google Trends Categories + 泛用種子詞自動發現熱門趨勢。

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0 - Enhancement v1.2
"""

from pytrends.request import TrendReq
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryTrendsExtractor:
    """
    使用 Google Trends Categories + 泛用種子詞自動發現熱門趨勢。

    方案 B+C 混合:
    - 方案 B: 使用 category 參數過濾相關類別
    - 方案 C: 使用極少泛用種子詞 + related queries 擴展

    優點:
    - 無需預設節日種子詞
    - 自動發現時段內真正熱門趨勢
    - Category 過濾確保相關性
    """

    # Google Trends Category IDs
    # Reference: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    CATEGORIES = {
        'all': 0,                           # All categories (最廣)
        'shopping': 18,                     # Shopping
        'arts_entertainment': 3,            # Arts & Entertainment
        'hobbies_leisure': 8,               # Hobbies & Leisure
        'games': 8,                         # Games
        'holidays': 19,                     # Holidays & Seasonal Events ★
        'gifts': 251,                       # Gifts & Special Event Items ★
        'toys': 237,                        # Toys ★
    }

    def __init__(
        self,
        region: str = 'HK',
        lang: str = 'zh-TW'
    ):
        """
        Initialize category-based trends extractor.

        Args:
            region: Google Trends region code (default: HK)
            lang: Language code (default: zh-TW)
        """
        self.region = region
        self.lang = lang
        self.pytrend = TrendReq(hl=lang, tz=480)

        logger.info(f"CategoryTrendsExtractor initialized: region={region}")

    def extract_category_trends(
        self,
        timeframe: str,
        categories: List[str] = ['holidays', 'gifts', 'toys'],
        broad_seeds: List[str] = None,
        top_n: int = 50,
        theme_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        使用 Category + 泛用種子詞提取熱門趨勢。

        Args:
            timeframe: 時間範圍 (e.g., '2024-11-01 2024-12-31')
            categories: Category 列表 (default: ['holidays', 'gifts', 'toys'])
            broad_seeds: 泛用種子詞 (default: 社交媒體 + 娛樂新聞相關)
            top_n: 返回前 N 個關鍵字
            theme_filter: 可選主題過濾詞 (e.g., '聖誕' for Christmas)

        Returns:
            DataFrame with columns: keyword, trend_score, category, timeframe

        Workflow:
            Step 1: 對每個 category 使用泛用種子詞查詢
            Step 2: 獲取 related_queries 擴展關鍵字池
            Step 3: 合併所有 related queries
            Step 4: 按 trend_score 排序
            Step 5: 可選主題過濾
            Step 6: 返回 Top N

        Example:
            >>> extractor = CategoryTrendsExtractor()
            >>> trends = extractor.extract_category_trends(
            ...     timeframe='2024-11-01 2024-12-31',
            ...     categories=['holidays', 'gifts'],
            ...     theme_filter='聖誕'
            ... )
            >>> print(trends.head())
        """
        if broad_seeds is None:
            # 社交媒體 + 娛樂新聞種子詞（中英混合）
            broad_seeds = [
                # 社交媒體平台
                'instagram', '小紅書',
                # 流行趨勢
                'trending', '熱門', '爆紅',
                # 娛樂內容
                '明星', 'viral'
            ]

        logger.info(f"\n{'='*60}")
        logger.info(f"Category-Based Trends Extraction")
        logger.info(f"{'='*60}\n")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Categories: {categories}")
        logger.info(f"Broad seeds: {broad_seeds}")
        logger.info(f"Theme filter: {theme_filter or 'None (all trends)'}\n")

        all_keywords = []

        # Step 1-2: 對每個 category 查詢
        # Google Trends API 限制: 最多 5 個關鍵詞/次
        # 所以需要分批處理
        batch_size = 5
        seed_batches = [broad_seeds[i:i+batch_size] for i in range(0, len(broad_seeds), batch_size)]

        logger.info(f"Total seed keywords: {len(broad_seeds)}")
        logger.info(f"Processing in {len(seed_batches)} batches (batch_size={batch_size})\n")

        for category_name in categories:
            category_id = self.CATEGORIES.get(category_name, 0)
            logger.info(f"Processing category: {category_name} (ID: {category_id})")

            for batch_idx, seed_batch in enumerate(seed_batches):
                logger.info(f"  Batch {batch_idx+1}/{len(seed_batches)}: {seed_batch}")

                try:
                    # Build payload with category
                    self.pytrend.build_payload(
                        seed_batch,
                        cat=category_id,
                        timeframe=timeframe,
                        geo=self.region
                    )

                    # Get related queries
                    related_queries = self.pytrend.related_queries()

                    # Extract top and rising queries from this batch
                    batch_count = 0
                    for seed in seed_batch:
                        if seed in related_queries:
                            # Top queries
                            top_df = related_queries[seed].get('top')
                            if top_df is not None and not top_df.empty:
                                for _, row in top_df.iterrows():
                                    all_keywords.append({
                                        'keyword': row['query'],
                                        'trend_score': row['value'],
                                        'category': category_name,
                                        'type': 'top'
                                    })
                                    batch_count += 1

                            # Rising queries
                            rising_df = related_queries[seed].get('rising')
                            if rising_df is not None and not rising_df.empty:
                                for _, row in rising_df.iterrows():
                                    # Rising 可能是百分比字串 (e.g., "Breakout", "200%")
                                    value = row['value']
                                    if isinstance(value, str):
                                        if value == 'Breakout':
                                            value = 1000  # 給 Breakout 高分
                                        else:
                                            # 嘗試解析百分比
                                            try:
                                                value = float(value.replace('%', '').replace('+', ''))
                                            except:
                                                value = 100  # 默認值

                                    all_keywords.append({
                                        'keyword': row['query'],
                                        'trend_score': value,
                                        'category': category_name,
                                        'type': 'rising'
                                    })
                                    batch_count += 1

                    logger.info(f"    → Found {batch_count} keywords in this batch")
                    time.sleep(2)  # Rate limiting between batches

                except Exception as e:
                    logger.error(f"    ❌ Error processing batch: {e}")
                    continue

            logger.info(f"  ✅ Category {category_name} completed\n")
            time.sleep(3)  # Rate limiting between categories

        # Step 3: 合併並去重
        if not all_keywords:
            logger.warning("No keywords found!")
            return pd.DataFrame(columns=['keyword', 'trend_score', 'category', 'timeframe'])

        df = pd.DataFrame(all_keywords)

        # 去重 (保留最高分數)
        df = df.sort_values('trend_score', ascending=False).drop_duplicates('keyword')

        logger.info(f"\n{'='*60}")
        logger.info(f"Total unique keywords: {len(df)}")

        # Step 5: 可選主題過濾
        if theme_filter:
            logger.info(f"Applying theme filter: {theme_filter}")
            df = df[df['keyword'].str.contains(theme_filter, case=False, na=False)]
            logger.info(f"After filtering: {len(df)} keywords")

        # Step 4: 按分數排序
        df = df.sort_values('trend_score', ascending=False).head(top_n)

        # 添加 timeframe
        df['timeframe'] = timeframe

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"\n✅ Extracted Top {len(df)} keywords")
        logger.info(f"{'='*60}\n")

        # 顯示 Top 10
        logger.info("Top 10 keywords:")
        for idx, row in df.head(10).iterrows():
            logger.info(f"  {idx+1:2d}. {row['keyword']:30s} (分數: {row['trend_score']:6.1f}, {row['category']})")

        return df

    def extract_timeframe_trends(
        self,
        timeframe: str,
        discovery_mode: str = 'category',
        categories: List[str] = ['holidays', 'gifts', 'toys'],
        theme_filter: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        高層 API: 提取指定時間段的熱門趨勢。

        Args:
            timeframe: 時間範圍 (e.g., '2024-11-01 2024-12-31')
            discovery_mode: 發現模式 ('category' or 'broad')
                - 'category': 使用 B+C 混合方案 (推薦)
                - 'broad': 只使用泛用種子詞 (無 category 過濾)
            categories: Category 列表
            theme_filter: 可選主題過濾
            top_n: 返回前 N 個關鍵字

        Returns:
            DataFrame with keywords and trend scores

        Example:
            >>> extractor = CategoryTrendsExtractor()
            >>>
            >>> # 聖誕節趨勢 (11-12月)
            >>> xmas_trends = extractor.extract_timeframe_trends(
            ...     timeframe='2024-11-01 2024-12-31',
            ...     theme_filter='聖誕'
            ... )
            >>>
            >>> # 萬聖節趨勢 (9-10月) - 無過濾，獲取所有熱門
            >>> halloween_trends = extractor.extract_timeframe_trends(
            ...     timeframe='2024-09-01 2024-10-31'
            ... )
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
    測試 Category-Based Trends Extraction with 社交媒體種子詞.
    """
    extractor = CategoryTrendsExtractor()

    # 擴展社交媒體 + 娛樂種子詞庫
    social_media_seeds = [
        # 社交媒體平台
        'instagram', 'tiktok', '小紅書', 'facebook',
        # 流行趨勢關鍵詞
        'trending', 'viral', '熱門', '爆紅', '爆款',
        # 娛樂內容
        '明星', 'idol', '電影', '動漫', 'ip',
        # 社交媒體特徵
        'hashtag', '挑戰', 'challenge', 'meme'
    ]

    # Test 1: 社交媒體熱門趨勢 (11-12月) - 無過濾
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Social Media Trends (11-12月) - All Topics")
    logger.info("="*80)

    social_trends = extractor.extract_category_trends(
        timeframe='2024-11-01 2024-12-31',
        categories=['arts_entertainment', 'shopping', 'toys'],
        broad_seeds=social_media_seeds,
        theme_filter=None,  # 先看所有熱門
        top_n=30
    )

    print("\n" + "="*80)
    print("Social Media Trends (All Topics):")
    print("="*80)
    print(social_trends.head(20).to_string(index=False))

    # Save
    social_trends.to_csv('data/trends_seasonal/nov_dec_social_media_all.csv', index=False)
    logger.info(f"\n💾 Saved to: data/trends_seasonal/nov_dec_social_media_all.csv")

    # Test 2: 過濾聖誕相關
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Christmas-related from Social Media Trends")
    logger.info("="*80)

    xmas_trends = extractor.extract_category_trends(
        timeframe='2024-11-01 2024-12-31',
        categories=['arts_entertainment', 'shopping', 'toys'],
        broad_seeds=social_media_seeds,
        theme_filter='christmas|聖誕|xmas|santa|聖誕老人',
        top_n=20
    )

    print("\n" + "="*80)
    print("Christmas Trends (Filtered):")
    print("="*80)
    print(xmas_trends.to_string(index=False))

    # Save
    xmas_trends.to_csv('data/trends_seasonal/christmas_social_media.csv', index=False)
    logger.info(f"\n💾 Saved to: data/trends_seasonal/christmas_social_media.csv")

    # Test 3: 對比舊種子詞方法
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Comparison with Old Seed Keywords Method")
    logger.info("="*80)

    old_seeds = ['Christmas', '聖誕節', '聖誕老人', 'Xmas', 'christmas tree']

    old_method_trends = extractor.extract_category_trends(
        timeframe='2024-11-01 2024-12-31',
        categories=['holidays', 'gifts', 'toys'],
        broad_seeds=old_seeds,
        theme_filter=None,
        top_n=20
    )

    print("\n" + "="*80)
    print("Old Method (Direct Christmas Keywords):")
    print("="*80)
    print(old_method_trends.to_string(index=False))

    # Save
    old_method_trends.to_csv('data/trends_seasonal/christmas_old_method.csv', index=False)
    logger.info(f"\n💾 Saved to: data/trends_seasonal/christmas_old_method.csv")

    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"Social Media Method (all): {len(social_trends)} keywords")
    logger.info(f"Social Media Method (filtered): {len(xmas_trends)} keywords")
    logger.info(f"Old Method (direct seeds): {len(old_method_trends)} keywords")


if __name__ == '__main__':
    main()
