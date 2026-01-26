"""
Meme Trend Analyzer - 深度分析時段內具體 Meme 趨勢

從社交媒體關鍵字中提取具體 meme 名稱、視覺特徵、情感表達，
轉化為可用的設計指引。

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0 - Enhancement v1.2
"""

import pandas as pd
from typing import Dict, List, Tuple
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemeAnalyzer:
    """
    分析 Meme 趨勢，提取具體 meme 類型和視覺特徵。

    Features:
        - 識別具體 meme 名稱（chill guy, pepe, wojak, etc.）
        - 提取 meme 的視覺特徵（表情、姿勢、風格）
        - 分析 meme 的情感和文化含義
        - 生成設計指引
    """

    # Meme 數據庫 - 常見 meme 的視覺特徵
    MEME_DATABASE = {
        'chill guy': {
            'character': 'Anthropomorphic dog/humanoid',
            'pose': 'Standing relaxed, hands in pockets or sides',
            'expression': 'Neutral, unbothered, slight smile, half-closed eyes',
            'mood': 'Calm, nonchalant, "my life is going ok" vibe',
            'color_palette': 'Brown/tan fur, red sweater, casual clothing',
            'style': 'Simple cartoon, clean lines, minimalist',
            'cultural_meaning': '表達淡定、隨遇而安的態度',
            'visual_keywords': ['relaxed posture', 'neutral face', 'casual stance', 'unbothered'],
            'peak_period': '2024-11 (爆紅)',
            'origin': 'Twitter/TikTok viral character',
            'adaptability': 'HIGH - 適合 mascot 設計'
        },
        'pepe': {
            'character': 'Green frog',
            'pose': 'Various (crying, smiling, shocked)',
            'expression': 'Exaggerated emotions',
            'mood': 'Varies (sad, happy, angry)',
            'color_palette': 'Bright green, simple colors',
            'style': 'MS Paint-style, rough lines',
            'cultural_meaning': '表達各種情緒和反應',
            'visual_keywords': ['exaggerated expression', 'emotional', 'reactive'],
            'adaptability': 'MEDIUM - 版權爭議'
        },
        'wojak': {
            'character': 'Bald humanoid with simple face',
            'pose': 'Usually stationary, head-focused',
            'expression': 'Sad, crying, anxious',
            'mood': 'Melancholic, relatable struggles',
            'color_palette': 'Pink/beige skin, minimal colors',
            'style': 'MS Paint-style, very simple',
            'cultural_meaning': '表達生活困境和焦慮',
            'visual_keywords': ['simple face', 'emotional', 'relatable'],
            'adaptability': 'LOW - 情緒負面'
        },
        'happy cat': {
            'character': 'Smiling white cat',
            'pose': 'Sitting, looking at camera',
            'expression': 'Wide smile, squinting eyes',
            'mood': 'Joyful, wholesome, content',
            'color_palette': 'White/cream fur, simple',
            'style': 'Photo-based meme',
            'cultural_meaning': '表達純粹的快樂和滿足',
            'visual_keywords': ['big smile', 'squinting eyes', 'wholesome'],
            'adaptability': 'HIGH - 正面情緒'
        },
        'first time': {
            'character': 'Various (format-based)',
            'pose': 'Usually showing surprised/confused reaction',
            'expression': 'Shocked, confused, awkward',
            'mood': 'Relatable first-time experience',
            'style': 'Image macro format',
            'cultural_meaning': '表達第一次經歷某事的感受',
            'visual_keywords': ['surprised look', 'awkward pose'],
            'adaptability': 'MEDIUM - 需要情境'
        },
        'duolingo': {
            'character': 'Green owl mascot (Duo)',
            'pose': 'Menacing stare, aggressive',
            'expression': 'Intense eyes, threatening',
            'mood': 'Humorous threat, persistence',
            'color_palette': 'Bright green, large eyes',
            'style': 'Official mascot design',
            'cultural_meaning': 'Duolingo 提醒學習的幽默威脅',
            'visual_keywords': ['intense gaze', 'threatening', 'persistent'],
            'adaptability': 'LOW - 特定品牌'
        },
        'spongebob': {
            'character': 'SpongeBob SquarePants',
            'pose': 'Various iconic poses',
            'expression': 'Mocking, sarcastic (alternating caps)',
            'mood': 'Sarcastic, mocking',
            'style': 'Cartoon screenshot',
            'cultural_meaning': '嘲諷或模仿他人',
            'visual_keywords': ['exaggerated', 'cartoon style'],
            'adaptability': 'LOW - 版權限制'
        }
    }

    # Meme 情感分類
    MEME_EMOTIONS = {
        'chill': ['chill guy', 'happy cat'],
        'wholesome': ['happy cat'],
        'sarcastic': ['spongebob'],
        'anxious': ['wojak'],
        'neutral': ['chill guy'],
        'joyful': ['happy cat'],
        'threatening': ['duolingo']
    }

    def __init__(self):
        """Initialize meme analyzer."""
        logger.info("MemeAnalyzer initialized")

    def extract_meme_names(self, keywords_df: pd.DataFrame) -> List[Dict]:
        """
        從關鍵字中提取具體 meme 名稱。

        Args:
            keywords_df: 關鍵字 DataFrame (must have 'keyword', 'trend_score')

        Returns:
            List of meme dictionaries with details
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Meme 名稱提取")
        logger.info(f"{'='*60}\n")

        detected_memes = []

        for _, row in keywords_df.iterrows():
            keyword = row['keyword'].lower()
            trend_score = row['trend_score']

            # Check each known meme
            for meme_name, meme_data in self.MEME_DATABASE.items():
                if meme_name in keyword:
                    meme_info = {
                        'meme_name': meme_name,
                        'original_keyword': row['keyword'],
                        'trend_score': trend_score,
                        'visual_features': meme_data,
                        'detected_in': keyword
                    }
                    detected_memes.append(meme_info)
                    logger.info(f"✅ 發現 Meme: {meme_name}")
                    logger.info(f"   關鍵字: {row['keyword']}")
                    logger.info(f"   趨勢分數: {trend_score:,.0f}")
                    logger.info(f"   適應性: {meme_data['adaptability']}\n")

        logger.info(f"{'='*60}")
        logger.info(f"總共發現 {len(detected_memes)} 個 Meme")
        logger.info(f"{'='*60}\n")

        return detected_memes

    def analyze_meme_characteristics(
        self,
        detected_memes: List[Dict],
        top_n: int = 3
    ) -> Dict:
        """
        分析 Meme 的共同特徵。

        Args:
            detected_memes: 檢測到的 meme 列表
            top_n: 分析前 N 個最熱門的 memes

        Returns:
            Dictionary with:
                - dominant_memes: 主要 meme 列表
                - common_emotions: 共同情緒
                - visual_guidelines: 視覺設計指引
                - adaptability_score: 適應性評分
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Meme 特徵分析 (Top {top_n})")
        logger.info(f"{'='*60}\n")

        # Sort by trend score
        sorted_memes = sorted(
            detected_memes,
            key=lambda x: x['trend_score'],
            reverse=True
        )[:top_n]

        # Extract characteristics
        dominant_memes = []
        all_visual_keywords = []
        all_moods = []
        adaptability_scores = []

        for meme in sorted_memes:
            meme_name = meme['meme_name']
            features = meme['visual_features']

            dominant_memes.append({
                'name': meme_name,
                'trend_score': meme['trend_score'],
                'character': features['character'],
                'mood': features['mood'],
                'adaptability': features['adaptability']
            })

            all_visual_keywords.extend(features['visual_keywords'])
            all_moods.append(features['mood'])

            # Parse adaptability
            adapt_score = {
                'HIGH': 3,
                'MEDIUM': 2,
                'LOW': 1
            }[features['adaptability'].split(' -')[0]]
            adaptability_scores.append(adapt_score)

        # Find common emotions
        common_emotions = self._find_common_emotions([m['meme_name'] for m in sorted_memes])

        # Generate visual guidelines
        visual_guidelines = self._generate_visual_guidelines(sorted_memes)

        # Calculate average adaptability
        avg_adaptability = sum(adaptability_scores) / len(adaptability_scores)

        result = {
            'dominant_memes': dominant_memes,
            'common_emotions': common_emotions,
            'visual_guidelines': visual_guidelines,
            'adaptability_score': avg_adaptability,
            'summary': {
                'total_memes': len(detected_memes),
                'analyzed_memes': len(sorted_memes),
                'avg_adaptability': avg_adaptability
            }
        }

        # Display results
        logger.info("主要 Memes:")
        for idx, meme in enumerate(dominant_memes, 1):
            logger.info(f"  {idx}. {meme['name'].upper()}")
            logger.info(f"     趨勢分數: {meme['trend_score']:,.0f}")
            logger.info(f"     角色: {meme['character']}")
            logger.info(f"     情緒: {meme['mood']}")
            logger.info(f"     適應性: {meme['adaptability']}\n")

        logger.info(f"共同情緒: {', '.join(common_emotions)}")
        logger.info(f"平均適應性: {avg_adaptability:.1f}/3.0")
        logger.info(f"\n{'='*60}\n")

        return result

    def _find_common_emotions(self, meme_names: List[str]) -> List[str]:
        """找出這些 memes 的共同情緒。"""
        emotion_counts = {}

        for emotion, memes in self.MEME_EMOTIONS.items():
            count = sum(1 for name in meme_names if name in memes)
            if count > 0:
                emotion_counts[emotion] = count

        # Return emotions sorted by frequency
        return sorted(emotion_counts.keys(), key=lambda x: emotion_counts[x], reverse=True)

    def _generate_visual_guidelines(self, memes: List[Dict]) -> Dict:
        """
        生成視覺設計指引。

        基於檢測到的 memes，生成可應用到角色設計的具體指引。
        """
        guidelines = {
            'expressions': [],
            'poses': [],
            'moods': [],
            'style_notes': [],
            'color_suggestions': [],
            'design_dos': [],
            'design_donts': []
        }

        for meme in memes:
            features = meme['visual_features']

            # Collect expressions
            if 'expression' in features:
                guidelines['expressions'].append(features['expression'])

            # Collect poses
            if 'pose' in features:
                guidelines['poses'].append(features['pose'])

            # Collect moods
            if 'mood' in features:
                guidelines['moods'].append(features['mood'])

            # Collect style notes
            if 'style' in features:
                guidelines['style_notes'].append(features['style'])

        # Generate design dos/don'ts
        top_meme = memes[0]['visual_features']

        if top_meme['adaptability'] == 'HIGH':
            guidelines['design_dos'].extend([
                f"採用 {memes[0]['meme_name']} 的放鬆姿態",
                f"表達「{top_meme['mood']}」的情緒",
                "保持簡潔的線條和造型",
                "使用中性或正面的表情"
            ])

        guidelines['design_donts'].extend([
            "避免版權爭議的 meme 角色",
            "避免過度負面的情緒",
            "避免過於複雜的設計"
        ])

        # Remove duplicates
        for key in ['expressions', 'poses', 'moods', 'style_notes']:
            guidelines[key] = list(set(guidelines[key]))

        return guidelines

    def generate_enhanced_prompt_guidance(
        self,
        analysis_result: Dict,
        original_prompt: str
    ) -> Dict:
        """
        基於 meme 分析，生成增強的 prompt 指引。

        Args:
            analysis_result: Meme 分析結果
            original_prompt: 原始 prompt

        Returns:
            Dictionary with:
                - enhanced_prompt: 增強的 prompt
                - specific_meme_features: 具體 meme 特徵描述
                - design_notes: 設計備註
        """
        dominant_meme = analysis_result['dominant_memes'][0]
        meme_name = dominant_meme['name']
        visual_guidelines = analysis_result['visual_guidelines']

        # Extract specific features
        meme_data = self.MEME_DATABASE[meme_name]

        specific_features = {
            'expression': meme_data['expression'],
            'pose': meme_data['pose'],
            'mood': meme_data['mood'],
            'visual_keywords': ', '.join(meme_data['visual_keywords'][:3])
        }

        # Generate enhanced prompt snippet
        enhancement_snippet = (
            f"Adopting {meme_name} aesthetic: {meme_data['expression']}, "
            f"{meme_data['pose']}, exuding {meme_data['mood']}"
        )

        # Design notes
        design_notes = [
            f"主要參考: {meme_name.upper()} meme",
            f"情緒: {dominant_meme['mood']}",
            f"關鍵視覺特徵: {specific_features['visual_keywords']}",
            f"適應性評分: {analysis_result['adaptability_score']:.1f}/3.0"
        ]

        if analysis_result['adaptability_score'] >= 2.5:
            design_notes.append("✅ 高度適合 mascot 設計")
        elif analysis_result['adaptability_score'] >= 2.0:
            design_notes.append("⚠️ 需要調整以適應品牌")
        else:
            design_notes.append("❌ 不建議直接使用，考慮其他方向")

        result = {
            'enhancement_snippet': enhancement_snippet,
            'specific_meme_features': specific_features,
            'design_notes': design_notes,
            'visual_guidelines': visual_guidelines
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"增強 Prompt 指引")
        logger.info(f"{'='*60}")
        logger.info(f"\n增強片段:")
        logger.info(f"  {enhancement_snippet}\n")
        logger.info(f"設計備註:")
        for note in design_notes:
            logger.info(f"  - {note}")
        logger.info(f"{'='*60}\n")

        return result


def main():
    """測試 Meme Analyzer."""

    # Load social media trends
    logger.info("載入社交媒體趨勢數據...")
    trends_df = pd.read_csv('data/trends_seasonal/nov_dec_social_media_all.csv')

    # Initialize analyzer
    analyzer = MemeAnalyzer()

    # Step 1: Extract meme names
    detected_memes = analyzer.extract_meme_names(trends_df)

    if not detected_memes:
        logger.warning("❌ 未檢測到任何已知 meme")
        return

    # Step 2: Analyze characteristics
    analysis = analyzer.analyze_meme_characteristics(detected_memes, top_n=3)

    # Step 3: Generate enhanced guidance
    sample_prompt = "Lulu豬, chubby pastel piglet mascot..."
    enhanced_guidance = analyzer.generate_enhanced_prompt_guidance(
        analysis,
        sample_prompt
    )

    # Save results
    output = {
        'detected_memes': detected_memes,
        'analysis': analysis,
        'enhanced_guidance': enhanced_guidance
    }

    import json
    with open('data/trends_seasonal/meme_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("💾 分析結果已儲存: data/trends_seasonal/meme_analysis_results.json")


if __name__ == '__main__':
    main()
