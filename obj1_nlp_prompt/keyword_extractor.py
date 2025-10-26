"""
TF-IDF Keyword Filtering Module

Filters raw trending keywords using TF-IDF to select high-quality, distinctive keywords.
Supports Chinese tokenization (jieba) for Hong Kong Traditional Chinese content.

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
Story: 2.2 - TF-IDF Keyword Filtering
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Filter keywords using TF-IDF with Chinese tokenization.

    Features:
    - Chinese tokenization (jieba)
    - TF-IDF scoring
    - Top-K selection per theme
    - Quality thresholding

    Usage:
        >>> extractor = KeywordExtractor(top_k=5, min_tfidf=0.3)
        >>> keywords = extractor.filter_keywords(trend_data)
        >>> extractor.save_keywords(keywords, 'Halloween')
    """

    def __init__(self, top_k: int = 5, min_tfidf: float = 0.3):
        """
        Initialize keyword extractor.

        Args:
            top_k: Number of top keywords to select per theme
            min_tfidf: Minimum TF-IDF score threshold
        """
        self.top_k = top_k
        self.min_tfidf = min_tfidf

        # Initialize TF-IDF vectorizer with jieba tokenizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._jieba_tokenize,
            lowercase=False,  # Preserve Chinese characters
            max_features=None,
            min_df=1
        )

        logger.info(f"KeywordExtractor initialized: top_k={top_k}, min_tfidf={min_tfidf}")

    def _jieba_tokenize(self, text: str) -> List[str]:
        """
        Chinese tokenization using jieba.

        Args:
            text: Input text (Chinese + English)

        Returns:
            List of tokens
        """
        # Use jieba for Chinese segmentation
        tokens = list(jieba.cut(text, cut_all=False))
        # Filter out single characters and whitespace
        tokens = [t.strip() for t in tokens if len(t.strip()) > 1]
        return tokens

    def filter_keywords(
        self,
        trends_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter keywords using TF-IDF across all themes.

        Args:
            trends_dict: Dictionary mapping theme to trends DataFrame
                         Each DataFrame must have 'keyword' and 'theme' columns

        Returns:
            Dictionary mapping theme to filtered keywords DataFrame
            Each DataFrame has columns: keyword, tfidf_score, trend_score, theme

        Example:
            >>> trends = {
            ...     'Halloween': pd.DataFrame({'keyword': ['萬聖節', '南瓜'], 'theme': 'Halloween'}),
            ...     'Christmas': pd.DataFrame({'keyword': ['聖誕節', '禮物'], 'theme': 'Christmas'})
            ... }
            >>> filtered = extractor.filter_keywords(trends)
        """
        logger.info(f"Filtering keywords from {len(trends_dict)} themes")

        # Combine all keywords into corpus
        corpus = []
        theme_mapping = []

        for theme, df in trends_dict.items():
            for keyword in df['keyword']:
                corpus.append(str(keyword))
                theme_mapping.append(theme)

        if not corpus:
            logger.warning("No keywords found in trends data")
            return {}

        # Compute TF-IDF
        logger.info(f"Computing TF-IDF for {len(corpus)} keywords...")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        feature_names = self.vectorizer.get_feature_names_out()

        # Get TF-IDF scores for each keyword
        keyword_scores = []
        for idx, keyword in enumerate(corpus):
            # Get TF-IDF vector for this keyword
            tfidf_vector = tfidf_matrix[idx].toarray().flatten()
            # Use max TF-IDF score as keyword score
            max_score = np.max(tfidf_vector) if len(tfidf_vector) > 0 else 0.0

            keyword_scores.append({
                'keyword': keyword,
                'tfidf_score': max_score,
                'theme': theme_mapping[idx]
            })

        # Convert to DataFrame
        scores_df = pd.DataFrame(keyword_scores)

        # Filter by theme and select top-K
        filtered_dict = {}
        for theme in trends_dict.keys():
            theme_df = scores_df[scores_df['theme'] == theme].copy()

            # Merge with original trend scores
            if not trends_dict[theme].empty:
                theme_df = theme_df.merge(
                    trends_dict[theme][['keyword', 'trend_score']],
                    on='keyword',
                    how='left'
                )

            # Apply TF-IDF threshold
            theme_df = theme_df[theme_df['tfidf_score'] >= self.min_tfidf]

            # Sort by TF-IDF score and select top-K
            theme_df = theme_df.sort_values('tfidf_score', ascending=False)
            theme_df = theme_df.head(self.top_k)

            filtered_dict[theme] = theme_df

            logger.info(f"  {theme}: {len(trends_dict[theme])} → {len(theme_df)} keywords "
                       f"(avg TF-IDF: {theme_df['tfidf_score'].mean():.3f})")

        total_filtered = sum(len(df) for df in filtered_dict.values())
        logger.info(f"Total filtered keywords: {total_filtered} (target: {len(trends_dict) * self.top_k})")

        return filtered_dict

    def save_keywords(
        self,
        keywords_dict: Dict[str, pd.DataFrame],
        output_dir: str = 'data/keywords'
    ):
        """
        Save filtered keywords to CSV files.

        Args:
            keywords_dict: Dictionary mapping theme to keywords DataFrame
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for theme, df in keywords_dict.items():
            filename = output_path / f"{theme.lower().replace(' ', '_')}_keywords.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"Saved keywords to: {filename}")

    def load_trends(self, trends_dir: str = 'data/trends') -> Dict[str, pd.DataFrame]:
        """
        Load trends data from CSV files.

        Args:
            trends_dir: Directory containing trends CSV files

        Returns:
            Dictionary mapping theme to trends DataFrame
        """
        trends_path = Path(trends_dir)
        trends_dict = {}

        for csv_file in trends_path.glob('*_trends.csv'):
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            # Extract theme from filename
            theme_name = csv_file.stem.replace('_trends', '').replace('_', ' ').title()
            trends_dict[theme_name] = df
            logger.info(f"Loaded {len(df)} trends from: {csv_file}")

        return trends_dict


def main():
    """
    Main execution for Story 2.2.

    Load trends from Story 2.1 and filter using TF-IDF.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Story 2.2: TF-IDF Keyword Filtering")
    logger.info(f"{'='*60}\n")

    # Initialize extractor
    extractor = KeywordExtractor(top_k=5, min_tfidf=0.3)

    # Load trends from Story 2.1
    trends_dict = extractor.load_trends('data/trends')

    if not trends_dict:
        logger.error("No trends data found. Run Story 2.1 first!")
        return

    # Filter keywords
    filtered_dict = extractor.filter_keywords(trends_dict)

    # Save filtered keywords
    extractor.save_keywords(filtered_dict, 'data/keywords')

    # Summary
    total_keywords = sum(len(df) for df in filtered_dict.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"Story 2.2 Completion Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Themes processed: {len(filtered_dict)}")
    logger.info(f"Keywords per theme: {extractor.top_k}")
    logger.info(f"Total filtered keywords: {total_keywords}")
    logger.info(f"Target: {len(filtered_dict)} themes × 5 keywords = {len(filtered_dict) * 5}")
    logger.info(f"Output directory: data/keywords/")
    logger.info(f"{'='*60}\n")

    return filtered_dict


if __name__ == '__main__':
    main()
