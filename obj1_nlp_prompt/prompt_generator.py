"""
LLM-based Prompt Generator Module

Uses GPT_API_free (Llama 3.1) to generate creative Midjourney prompts
for character IP seasonal designs.

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
Story: 2.4 - LLM-based Prompt Generator
"""

from openai import OpenAI
import pandas as pd
from pathlib import Path
from typing import List, Dict
import time
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    Generate Midjourney prompts using LLM (GPT_API_free).

    Features:
    - Template-based generation
    - Multiple variations per theme
    - Quality validation (length, content)
    - Automatic retry on failure

    Usage:
        >>> generator = PromptGenerator()
        >>> prompts = generator.generate_variations('Halloween', keywords, n=4)
        >>> generator.save_prompts(prompts, 'Halloween')
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        template_path: str = 'obj1_nlp_prompt/templates/prompt_template.txt',
        character_desc_path: str = 'data/character_descriptions/lulu_pig.txt'
    ):
        """
        Initialize prompt generator.

        Args:
            api_key: GPT_API_free API key (defaults to env variable)
            base_url: API base URL (defaults to env variable)
            template_path: Path to prompt template file
            character_desc_path: Path to character description file
        """
        load_dotenv()

        self.api_key = api_key or os.getenv('GPT_API_FREE_KEY')
        self.base_url = base_url or os.getenv('GPT_API_FREE_BASE_URL')

        if not self.api_key:
            raise ValueError("GPT_API_FREE_KEY not found in environment")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Load template
        self.template = self._load_template(template_path)
        # Load character description
        self.character_desc = self._load_character_description(character_desc_path)

        logger.info("PromptGenerator initialized successfully")

    def _load_template(self, template_path: str) -> str:
        """Load prompt template from file."""
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        logger.info(f"Loaded template from: {template_path}")
        return template

    def _load_character_description(self, desc_path: str) -> str:
        """Load character description from file."""
        with open(desc_path, 'r', encoding='utf-8') as f:
            description = f.read()
        logger.info(f"Loaded character description from: {desc_path}")
        return description

    def generate_prompt(
        self,
        theme: str,
        keywords: List[str],
        variation_hint: str = ""
    ) -> str:
        """
        Generate a single Midjourney prompt.

        Args:
            theme: Seasonal theme (e.g., 'Halloween')
            keywords: List of trending keywords
            variation_hint: Optional hint for variation (e.g., "outdoor scene", "indoor cozy")

        Returns:
            Generated Midjourney prompt string

        Example:
            >>> generator = PromptGenerator()
            >>> prompt = generator.generate_prompt('Halloween', ['pumpkin', 'costume'])
            >>> print(prompt)
        """
        # Format keywords
        keywords_str = ', '.join(keywords[:5])  # Use top 5 keywords

        # Fill template
        filled_template = self.template.format(
            character_description=self.character_desc,
            theme=theme,
            keywords=keywords_str,
            character_name="Lulu Pig"
        )

        # Add variation hint if provided
        if variation_hint:
            filled_template += f"\n\nVariation Focus: {variation_hint}"

        # Call GPT API
        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a creative prompt engineer specializing in Midjourney prompts for character IP designs. Generate concise, vivid prompts that maintain character consistency while incorporating seasonal elements.'
                    },
                    {
                        'role': 'user',
                        'content': filled_template
                    }
                ],
                max_tokens=200,
                temperature=0.8  # Higher temperature for creativity
            )

            prompt = response.choices[0].message.content.strip()
            elapsed = time.time() - start_time

            logger.info(f"Generated prompt for {theme} in {elapsed:.2f}s")
            return prompt

        except Exception as e:
            logger.error(f"Error generating prompt for {theme}: {e}")
            return self._fallback_prompt(theme, keywords)

    def generate_variations(
        self,
        theme: str,
        keywords: List[str],
        n: int = 4
    ) -> List[Dict[str, str]]:
        """
        Generate multiple prompt variations for a theme.

        Args:
            theme: Seasonal theme
            keywords: List of trending keywords
            n: Number of variations to generate (default: 4)

        Returns:
            List of prompt dictionaries with keys: theme, variation, prompt, length

        Example:
            >>> variations = generator.generate_variations('Christmas', keywords, n=4)
            >>> print(f"Generated {len(variations)} variations")
        """
        logger.info(f"Generating {n} variations for theme: {theme}")

        # Variation hints for diversity
        variation_hints = [
            "outdoor festive scene with decorations",
            "indoor cozy atmosphere with warm lighting",
            "action pose celebrating the season",
            "portrait style with seasonal accessories"
        ]

        variations = []
        for i in range(n):
            hint = variation_hints[i] if i < len(variation_hints) else ""

            prompt = self.generate_prompt(theme, keywords, variation_hint=hint)

            # Validate prompt
            is_valid, validation_msg = self._validate_prompt(prompt, theme)
            if not is_valid:
                logger.warning(f"Validation failed for {theme} variation {i+1}: {validation_msg}")
                # Retry once
                prompt = self.generate_prompt(theme, keywords, variation_hint=hint)

            variations.append({
                'theme': theme,
                'variation': i + 1,
                'prompt': prompt,
                'length': len(prompt.split()),
                'keywords_used': ', '.join(keywords[:5])
            })

            # Rate limiting
            time.sleep(1)

        logger.info(f"Generated {len(variations)} variations for {theme}")
        return variations

    def generate_all_themes(
        self,
        keywords_dict: Dict[str, pd.DataFrame],
        variations_per_theme: int = 4
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate prompts for all themes.

        Args:
            keywords_dict: Dictionary mapping theme to keywords DataFrame
            variations_per_theme: Number of variations per theme

        Returns:
            Dictionary mapping theme to list of prompt variations
        """
        all_prompts = {}

        for i, (theme, keywords_df) in enumerate(keywords_dict.items(), 1):
            logger.info(f"Processing theme {i}/{len(keywords_dict)}: {theme}")

            keywords = keywords_df['keyword'].tolist()
            variations = self.generate_variations(theme, keywords, n=variations_per_theme)

            all_prompts[theme] = variations

            # Save immediately
            self.save_prompts(variations, theme)

        return all_prompts

    def save_prompts(
        self,
        prompts: List[Dict[str, str]],
        theme: str,
        output_dir: str = 'data/prompts'
    ):
        """
        Save generated prompts to files.

        Args:
            prompts: List of prompt dictionaries
            theme: Theme name
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual prompt files
        for prompt_dict in prompts:
            filename = output_path / f"{theme.lower().replace(' ', '_')}_variation_{prompt_dict['variation']}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(prompt_dict['prompt'])

        # Save metadata CSV
        df = pd.DataFrame(prompts)
        csv_filename = output_path / f"{theme.lower().replace(' ', '_')}_prompts.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

        logger.info(f"Saved {len(prompts)} prompts for {theme} to: {output_dir}")

    def _validate_prompt(self, prompt: str, theme: str) -> tuple[bool, str]:
        """
        Validate generated prompt quality.

        Args:
            prompt: Generated prompt string
            theme: Theme name

        Returns:
            (is_valid, validation_message)
        """
        word_count = len(prompt.split())

        # Check length
        if word_count < 50:
            return False, f"Too short: {word_count} words (min: 50)"
        if word_count > 150:
            return False, f"Too long: {word_count} words (max: 150)"

        # Check for character mention (Lulu or pig)
        if 'lulu' not in prompt.lower() and 'pig' not in prompt.lower():
            return False, "Missing character reference"

        # Check for theme mention
        if theme.lower() not in prompt.lower():
            # Theme might be implicit, check for season keywords
            season_keywords = {
                'Halloween': ['halloween', 'pumpkin', 'spooky', 'costume'],
                'Christmas': ['christmas', 'xmas', 'santa', 'snow', 'gift'],
                'Spring Festival': ['spring', 'lunar', 'new year', '春節'],
                'Summer': ['summer', 'beach', 'sun', 'tropical'],
                "Valentine's Day": ['valentine', 'love', 'heart', 'romantic'],
                'Mid-Autumn Festival': ['mid-autumn', 'moon', 'mooncake', '中秋'],
                'New Year': ['new year', 'countdown', 'celebration']
            }

            theme_found = False
            for keyword in season_keywords.get(theme, []):
                if keyword in prompt.lower():
                    theme_found = True
                    break

            if not theme_found:
                return False, f"Missing theme reference: {theme}"

        return True, "Valid"

    def _fallback_prompt(self, theme: str, keywords: List[str]) -> str:
        """Generate fallback prompt if API fails."""
        keywords_str = ', '.join(keywords[:3])
        return (f"Lulu Pig, adorable pink pig character, celebrating {theme} "
                f"with {keywords_str}, cute kawaii style, vibrant colors, "
                f"cheerful mood, merchandise-ready design, character illustration")

    def load_keywords(self, keywords_dir: str = 'data/keywords') -> Dict[str, pd.DataFrame]:
        """
        Load filtered keywords from CSV files.

        Args:
            keywords_dir: Directory containing keywords CSV files

        Returns:
            Dictionary mapping theme to keywords DataFrame
        """
        keywords_path = Path(keywords_dir)
        keywords_dict = {}

        for csv_file in keywords_path.glob('*_keywords.csv'):
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            theme_name = csv_file.stem.replace('_keywords', '').replace('_', ' ').title()
            keywords_dict[theme_name] = df
            logger.info(f"Loaded {len(df)} keywords from: {csv_file}")

        return keywords_dict


def main():
    """
    Main execution for Story 2.4.

    Generate Midjourney prompts for all themes using GPT_API_free.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Story 2.4: LLM-based Prompt Generator")
    logger.info(f"{'='*60}\n")

    # Initialize generator
    generator = PromptGenerator()

    # Load keywords from Story 2.2
    keywords_dict = generator.load_keywords('data/keywords')

    if not keywords_dict:
        logger.error("No keywords data found. Run Story 2.2 first!")
        return

    # Generate prompts for all themes
    all_prompts = generator.generate_all_themes(keywords_dict, variations_per_theme=4)

    # Summary
    total_prompts = sum(len(prompts) for prompts in all_prompts.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"Story 2.4 Completion Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Themes processed: {len(all_prompts)}")
    logger.info(f"Variations per theme: 4")
    logger.info(f"Total prompts generated: {total_prompts}")
    logger.info(f"Target: {len(all_prompts)} themes × 4 variations = {len(all_prompts) * 4}")
    logger.info(f"Output directory: data/prompts/")
    logger.info(f"{'='*60}\n")

    return all_prompts


if __name__ == '__main__':
    main()
