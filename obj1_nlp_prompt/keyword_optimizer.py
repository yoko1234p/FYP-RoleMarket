"""
關鍵字優化器 - 提交 GPT 前過濾和優化關鍵字

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0 - Enhancement v1.2

Purpose:
    從社交媒體趨勢提取的關鍵字中，過濾出適合 Character IP 設計的視覺化關鍵字。

Filters:
    1. 移除技術性/平台名稱（instagram, facebook, download）
    2. 移除過於具體的電影/明星名稱
    3. 優先選擇視覺化、情感化的關鍵字
    4. 提取可設計的元素（meme 風格、節日元素、情感表達）
"""

import pandas as pd
from typing import List, Dict, Set
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordOptimizer:
    """
    優化和過濾關鍵字，確保適合 Character IP 設計。
    """

    # 黑名單：技術性/平台詞彙
    BLACKLIST_PATTERNS = [
        # 平台名稱
        r'instagram', r'facebook', r'tiktok', r'小紅書',
        # 技術操作
        r'download', r'login', r'線上看', r'app',
        # 過於具體的地點
        r'票房', r'紀錄', r'上映',
        # 搜尋行為
        r'看', r'查', r'搜尋'
    ]

    # 視覺化關鍵字特徵
    VISUAL_INDICATORS = [
        'meme', 'cute', 'kawaii', '可愛',
        'style', '風格', 'character', '角色',
        'design', '設計', 'art', '藝術',
        'cozy', '溫馨', 'festive', '節日',
        'decoration', '裝飾', 'celebration', '慶祝'
    ]

    # 情感/氛圍關鍵字（高價值）
    EMOTIONAL_KEYWORDS = [
        'chill', 'happy', 'cozy', 'warm', 'festive',
        'cheerful', 'cute', 'adorable', 'lovely',
        '溫馨', '可愛', '歡樂', '溫暖', '節日'
    ]

    # 節日/主題關鍵字（高價值）
    THEME_KEYWORDS = [
        'christmas', '聖誕', 'xmas', 'santa',
        'halloween', '萬聖節', 'pumpkin',
        'spring', '春天', 'lunar new year',
        'valentine', '情人節'
    ]

    def __init__(self):
        """Initialize keyword optimizer."""
        logger.info("KeywordOptimizer initialized")

    def is_blacklisted(self, keyword: str) -> bool:
        """
        檢查關鍵字是否在黑名單中。

        Args:
            keyword: 關鍵字

        Returns:
            True if blacklisted, False otherwise
        """
        keyword_lower = keyword.lower()
        for pattern in self.BLACKLIST_PATTERNS:
            if re.search(pattern, keyword_lower):
                return True
        return False

    def calculate_visual_score(self, keyword: str) -> float:
        """
        計算關鍵字的視覺化分數。

        分數越高 = 越適合視覺設計

        Scoring:
            - 包含視覺化指標詞: +2.0
            - 包含情感/氛圍詞: +3.0 (高價值)
            - 包含節日/主題詞: +4.0 (最高價值)
            - 包含 "meme": +1.5 (流行文化)
            - 長度適中 (2-4 words): +0.5
            - 包含中文: +0.3 (本地化)

        Args:
            keyword: 關鍵字

        Returns:
            視覺化分數 (0-10)
        """
        keyword_lower = keyword.lower()
        score = 0.0

        # 節日/主題關鍵字（最高價值）
        for theme_word in self.THEME_KEYWORDS:
            if theme_word in keyword_lower:
                score += 4.0
                break

        # 情感/氛圍關鍵字（高價值）
        for emotion_word in self.EMOTIONAL_KEYWORDS:
            if emotion_word in keyword_lower:
                score += 3.0
                break

        # 視覺化指標
        for visual_word in self.VISUAL_INDICATORS:
            if visual_word in keyword_lower:
                score += 2.0
                break

        # Meme 文化（流行）
        if 'meme' in keyword_lower:
            score += 1.5

        # 長度適中
        word_count = len(keyword.split())
        if 2 <= word_count <= 4:
            score += 0.5

        # 本地化（包含中文）
        if re.search(r'[\u4e00-\u9fff]', keyword):
            score += 0.3

        return min(score, 10.0)  # Cap at 10

    def extract_visual_concepts(self, keyword: str) -> List[str]:
        """
        從關鍵字中提取視覺化概念。

        Examples:
            "christmas meme" → ["christmas", "meme style"]
            "chill guy" → ["chill", "relaxed mood"]
            "聖誕電影" → ["christmas", "cinematic style"]

        Args:
            keyword: 關鍵字

        Returns:
            視覺化概念列表
        """
        concepts = []
        keyword_lower = keyword.lower()

        # 節日概念
        if any(x in keyword_lower for x in ['christmas', '聖誕', 'xmas']):
            concepts.append('christmas theme')
        if any(x in keyword_lower for x in ['halloween', '萬聖節']):
            concepts.append('halloween theme')

        # 風格概念
        if 'meme' in keyword_lower:
            concepts.append('meme style')
        if any(x in keyword_lower for x in ['電影', 'movie', 'film', 'cinema']):
            concepts.append('cinematic style')
        if any(x in keyword_lower for x in ['cute', '可愛', 'kawaii']):
            concepts.append('cute aesthetic')

        # 情緒概念
        if 'chill' in keyword_lower:
            concepts.append('relaxed mood')
        if 'happy' in keyword_lower:
            concepts.append('cheerful mood')
        if any(x in keyword_lower for x in ['cozy', '溫馨']):
            concepts.append('cozy atmosphere')

        return concepts

    def optimize_keywords(
        self,
        keywords_df: pd.DataFrame,
        top_n: int = 15,
        min_visual_score: float = 2.0,
        preserve_high_trend: bool = True
    ) -> pd.DataFrame:
        """
        優化關鍵字列表。

        Workflow:
            1. 移除黑名單關鍵字
            2. 計算視覺化分數
            3. 提取視覺化概念
            4. 按分數排序
            5. 返回 Top N

        Args:
            keywords_df: 關鍵字 DataFrame (must have 'keyword', 'trend_score')
            top_n: 返回前 N 個關鍵字
            min_visual_score: 最低視覺化分數閾值
            preserve_high_trend: 保留高趨勢分數關鍵字（即使視覺分數低）

        Returns:
            優化後的 DataFrame with additional columns:
                - visual_score: 視覺化分數
                - visual_concepts: 視覺化概念
                - combined_score: 綜合分數
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"關鍵字優化流程")
        logger.info(f"{'='*60}\n")
        logger.info(f"Input: {len(keywords_df)} keywords")

        # Step 1: 移除黑名單
        df = keywords_df.copy()
        df['is_blacklisted'] = df['keyword'].apply(self.is_blacklisted)
        df_filtered = df[~df['is_blacklisted']].copy()

        logger.info(f"Step 1: 移除黑名單 → {len(df_filtered)} keywords remaining")

        # Step 2: 計算視覺化分數
        df_filtered['visual_score'] = df_filtered['keyword'].apply(
            self.calculate_visual_score
        )

        # Step 3: 提取視覺化概念
        df_filtered['visual_concepts'] = df_filtered['keyword'].apply(
            self.extract_visual_concepts
        )

        # Step 4: 計算綜合分數
        # Normalize trend_score to 0-10
        max_trend = df_filtered['trend_score'].max()
        if max_trend > 0:
            df_filtered['normalized_trend'] = (
                df_filtered['trend_score'] / max_trend * 10
            )
        else:
            df_filtered['normalized_trend'] = 0

        # Combined score: 70% visual + 30% trend
        df_filtered['combined_score'] = (
            0.7 * df_filtered['visual_score'] +
            0.3 * df_filtered['normalized_trend']
        )

        logger.info(f"Step 2-4: 計算分數完成")

        # Step 5: 過濾低分關鍵字
        df_filtered = df_filtered[
            df_filtered['visual_score'] >= min_visual_score
        ]

        logger.info(f"Step 5: 過濾低分 (>={min_visual_score}) → {len(df_filtered)} keywords")

        # Step 6: 排序並選擇 Top N
        df_sorted = df_filtered.sort_values('combined_score', ascending=False)
        df_top = df_sorted.head(top_n)

        logger.info(f"Step 6: 選擇 Top {top_n} → {len(df_top)} keywords")
        logger.info(f"\n{'='*60}")
        logger.info(f"優化完成")
        logger.info(f"{'='*60}\n")

        # 顯示 Top 10
        logger.info("Top 10 優化後關鍵字:")
        for idx, row in df_top.head(10).iterrows():
            concepts_str = ', '.join(row['visual_concepts']) if row['visual_concepts'] else 'N/A'
            logger.info(
                f"  {idx+1:2d}. {row['keyword']:30s} "
                f"(視覺: {row['visual_score']:.1f}, "
                f"趨勢: {row['normalized_trend']:.1f}, "
                f"總分: {row['combined_score']:.1f})"
            )
            if row['visual_concepts']:
                logger.info(f"      → Concepts: {concepts_str}")

        return df_top

    def format_for_prompt(
        self,
        optimized_df: pd.DataFrame,
        include_concepts: bool = True
    ) -> Dict[str, any]:
        """
        格式化優化後的關鍵字，準備提交給 GPT。

        Args:
            optimized_df: 優化後的 DataFrame
            include_concepts: 是否包含視覺化概念

        Returns:
            Dictionary with:
                - keywords: List of keyword strings
                - concepts: List of visual concepts
                - summary: 摘要信息
        """
        keywords = optimized_df['keyword'].tolist()

        # 提取所有視覺化概念（去重）
        all_concepts = []
        for concepts_list in optimized_df['visual_concepts']:
            all_concepts.extend(concepts_list)
        unique_concepts = list(set(all_concepts))

        summary = {
            'total_keywords': len(keywords),
            'avg_visual_score': optimized_df['visual_score'].mean(),
            'avg_trend_score': optimized_df['normalized_trend'].mean(),
            'top_concept': max(set(all_concepts), key=all_concepts.count) if all_concepts else None
        }

        result = {
            'keywords': keywords,
            'concepts': unique_concepts if include_concepts else [],
            'summary': summary
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"格式化完成 - 準備提交 GPT")
        logger.info(f"{'='*60}")
        logger.info(f"關鍵字數量: {len(keywords)}")
        logger.info(f"視覺化概念: {', '.join(unique_concepts[:5])}..." if unique_concepts else "視覺化概念: N/A")
        logger.info(f"平均視覺分數: {summary['avg_visual_score']:.2f}")
        logger.info(f"主要概念: {summary['top_concept']}")
        logger.info(f"{'='*60}\n")

        return result


def main():
    """
    測試關鍵字優化器。
    """
    # Load social media trends
    logger.info("載入社交媒體趨勢數據...")
    df = pd.read_csv('data/trends_seasonal/nov_dec_social_media_all.csv')

    logger.info(f"原始數據: {len(df)} keywords\n")

    # Initialize optimizer
    optimizer = KeywordOptimizer()

    # Optimize keywords
    optimized = optimizer.optimize_keywords(
        df,
        top_n=15,
        min_visual_score=2.0
    )

    # Save optimized keywords
    optimized.to_csv(
        'data/trends_seasonal/nov_dec_optimized_keywords.csv',
        index=False
    )
    logger.info(f"💾 Saved optimized keywords to: data/trends_seasonal/nov_dec_optimized_keywords.csv")

    # Format for GPT
    formatted = optimizer.format_for_prompt(optimized)

    logger.info("\n" + "="*80)
    logger.info("準備提交給 GPT 的關鍵字:")
    logger.info("="*80)
    for idx, kw in enumerate(formatted['keywords'], 1):
        logger.info(f"  {idx:2d}. {kw}")

    logger.info("\n視覺化概念:")
    for concept in formatted['concepts']:
        logger.info(f"  - {concept}")


if __name__ == '__main__':
    main()
