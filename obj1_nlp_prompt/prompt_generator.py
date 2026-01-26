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
        model: str = None,
        template_path: str = 'obj1_nlp_prompt/templates/prompt_template.txt',
        character_desc_path: str = 'data/character_descriptions/lulu_pig.txt'
    ):
        """
        Initialize prompt generator.

        Args:
            api_key: GPT_API_free API key (defaults to env variable)
            base_url: API base URL (defaults to env variable)
            model: Model name (defaults to env variable GPT_API_FREE_MODEL)
            template_path: Path to prompt template file
            character_desc_path: Path to character description file
        """
        load_dotenv()

        self.api_key = api_key or os.getenv('GPT_API_FREE_KEY')
        self.base_url = base_url or os.getenv('GPT_API_FREE_BASE_URL')
        self.model = model or os.getenv('GPT_API_FREE_MODEL', 'gpt-3.5-turbo')

        if not self.api_key:
            raise ValueError("GPT_API_FREE_KEY not found in environment")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Load template
        self.template = self._load_template(template_path)
        # Load character description
        self.character_desc = self._load_character_description(character_desc_path)

        logger.info(f"PromptGenerator initialized successfully (model: {self.model})")

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

        logger.info(f"  → Using keywords: {keywords_str}")

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

            # Enhanced system prompt with keyword enforcement
            system_prompt = (
                "You are a creative prompt engineer specializing in Midjourney prompts for character IP designs. "
                "⚠️ CRITICAL: You MUST incorporate the provided trending keywords into the scene/costume/props. "
                "The trending keywords are the CORE requirement - they represent current market trends that MUST be visible in the design. "
                "Blend them naturally with the seasonal theme, but keywords take PRIORITY over generic seasonal elements. "
                "Generate concise, vivid prompts that maintain character consistency while incorporating both trending keywords and seasonal elements."
            )

            # Add keyword emphasis at the beginning of user content
            keyword_reminder = (
                f"🔥 MANDATORY KEYWORDS TO INTEGRATE: {keywords_str}\n"
                f"⚠️ These keywords MUST appear in your prompt through costume/props/scene elements.\n\n"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': keyword_reminder + filled_template
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
            # Parse and display API error details
            error_msg = self._parse_api_error(e)
            logger.error(f"❌ Error generating prompt for {theme}: {error_msg}")

            # Check if it's a quota/limit error
            if self._is_quota_error(e):
                logger.error("⚠️  API quota/usage limit reached!")
                logger.error("   Please check your API credits or wait for quota reset.")
                raise  # Re-raise to stop execution

            # For other errors, use fallback
            logger.warning(f"Using fallback prompt for {theme}")
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

            # Validate prompt (including keyword usage)
            is_valid, validation_msg = self._validate_prompt(prompt, theme, keywords)
            if not is_valid:
                logger.warning(f"Validation failed for {theme} variation {i+1}: {validation_msg}")
                # Retry once
                prompt = self.generate_prompt(theme, keywords, variation_hint=hint)
                # Re-validate after retry
                is_valid, validation_msg = self._validate_prompt(prompt, theme, keywords)
                if not is_valid:
                    logger.warning(f"Retry also failed validation: {validation_msg}")

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

    def _validate_prompt(self, prompt: str, theme: str, keywords: List[str] = None) -> tuple[bool, str]:
        """
        Validate generated prompt quality.

        Args:
            prompt: Generated prompt string
            theme: Theme name
            keywords: List of trending keywords that should be integrated

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

        # Check for keyword usage (CRITICAL for trend-driven designs)
        if keywords and len(keywords) > 0:
            prompt_lower = prompt.lower()
            keywords_found = []
            keywords_missing = []

            for kw in keywords[:5]:  # Check top 5 keywords
                # Simple keyword matching (check for partial matches)
                kw_parts = kw.lower().split()
                found = False

                for part in kw_parts:
                    if len(part) >= 2 and part in prompt_lower:
                        found = True
                        break

                if found:
                    keywords_found.append(kw)
                else:
                    keywords_missing.append(kw)

            # Require at least 40% of keywords to be present (2 out of 5, or 1 out of 2)
            if len(keywords) <= 2:
                min_required = 1  # At least 1 keyword for 1-2 keywords
            else:
                min_required = max(2, int(len(keywords[:5]) * 0.4))  # At least 40%

            if len(keywords_found) < min_required:
                return False, (
                    f"Missing trending keywords: Found {len(keywords_found)}/{len(keywords[:5])} keywords. "
                    f"Found: {', '.join(keywords_found) if keywords_found else 'None'}. "
                    f"Missing: {', '.join(keywords_missing[:3])}"
                )

            logger.info(f"  ✅ Keyword validation passed: {len(keywords_found)}/{len(keywords[:5])} keywords found")

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

    def _parse_api_error(self, exception: Exception) -> str:
        """
        Parse API error and extract useful information.

        Args:
            exception: Exception from API call

        Returns:
            Formatted error message string

        Example error response body from GPT_API_free:
        {
            "error": {
                "message": "Error details...",
                "type": "chatanywhere_error",
                "param": null,
                "code": "400 BAD_REQUEST"
            }
        }
        """
        # Try to extract error details from OpenAI exception
        try:
            # Check if it's an OpenAI API error
            if hasattr(exception, 'response'):
                response_json = exception.response.json()
                if 'error' in response_json:
                    error_obj = response_json['error']
                    message = error_obj.get('message', str(exception))
                    error_type = error_obj.get('type', 'unknown')
                    error_code = error_obj.get('code', 'unknown')

                    return f"[{error_type}] {error_code}: {message}"

            # Check if it's a standard exception with body attribute
            if hasattr(exception, 'body') and isinstance(exception.body, dict):
                error_obj = exception.body.get('error', {})
                message = error_obj.get('message', str(exception))
                error_type = error_obj.get('type', 'unknown')
                error_code = error_obj.get('code', 'unknown')

                return f"[{error_type}] {error_code}: {message}"

            # Fallback: return exception string
            return str(exception)

        except Exception:
            # If parsing fails, return original exception string
            return str(exception)

    def _is_quota_error(self, exception: Exception) -> bool:
        """
        Check if error is related to API quota/usage limit.

        Args:
            exception: Exception from API call

        Returns:
            True if quota/limit error, False otherwise
        """
        error_str = str(exception).lower()

        # Common quota/limit error keywords
        quota_keywords = [
            'quota',
            'rate limit',
            'usage limit',
            'insufficient',
            'exceeded',
            'too many requests',
            '429',
            'billing',
            'credit'
        ]

        # Check exception message
        for keyword in quota_keywords:
            if keyword in error_str:
                return True

        # Check error code from response
        try:
            if hasattr(exception, 'response'):
                response_json = exception.response.json()
                error_code = response_json.get('error', {}).get('code', '')
                if '429' in str(error_code) or 'quota' in str(error_code).lower():
                    return True

            if hasattr(exception, 'body') and isinstance(exception.body, dict):
                error_code = exception.body.get('error', {}).get('code', '')
                if '429' in str(error_code) or 'quota' in str(error_code).lower():
                    return True

        except Exception:
            pass

        return False

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

    def load_seasonal_keywords(
        self,
        themes: List[str],
        use_seasonal: bool = True,
        top_n: int = 20,
        output_dir: str = 'data/trends_seasonal'
    ) -> Dict[str, pd.DataFrame]:
        """
        使用 SeasonalTrendsExtractor 提取季節性關鍵字.

        Args:
            themes: 主題列表
            use_seasonal: 是否使用季節性時段（True=seasonal, False=12-month average）
            top_n: 每個主題的關鍵字數量
            output_dir: 輸出目錄

        Returns:
            Dictionary mapping theme to keywords DataFrame

        Example:
            >>> generator = PromptGenerator()
            >>> seasonal_keywords = generator.load_seasonal_keywords(
            ...     themes=['Christmas', 'Halloween'],
            ...     use_seasonal=True
            ... )
        """
        if use_seasonal:
            logger.info("使用 SeasonalTrendsExtractor 提取季節性關鍵字")
            from obj1_nlp_prompt.seasonal_trends_extractor import SeasonalTrendsExtractor

            extractor = SeasonalTrendsExtractor()
            keywords_dict = extractor.extract_all_themes_seasonal(
                themes=themes,
                top_n=top_n,
                output_dir=output_dir
            )
        else:
            logger.info("使用標準 TrendsExtractor 提取 12 個月平均關鍵字")
            from obj1_nlp_prompt.trends_extractor import TrendsExtractor

            extractor = TrendsExtractor()
            keywords_dict = extractor.extract_all_themes(
                themes=themes,
                timeframe='today 12-m',
                top_n=top_n
            )

        logger.info(f"已提取 {len(keywords_dict)} 個主題的關鍵字")
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
