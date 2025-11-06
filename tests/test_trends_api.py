"""
Unit Tests for TrendsAPIWrapper

測試 Obj 1 API Wrapper 功能。

Author: Developer (James)
Date: 2025-11-06
"""

import sys
from pathlib import Path
import unittest

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.trends_api import (
    TrendsAPIWrapper,
    TrendsAPIError,
    PromptGenerationError
)


class TestTrendsAPIWrapper(unittest.TestCase):
    """Test cases for TrendsAPIWrapper."""

    @classmethod
    def setUpClass(cls):
        """Initialize TrendsAPIWrapper once for all tests."""
        cls.wrapper = TrendsAPIWrapper(region='HK', lang='zh-TW')

    def test_initialization(self):
        """Test TrendsAPIWrapper initialization."""
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.region, 'HK')
        self.assertEqual(self.wrapper.lang, 'zh-TW')
        self.assertIsNotNone(self.wrapper.prompt_generator)

    def test_extract_keywords_simple_valid_input(self):
        """Test extract_keywords_simple with valid input."""
        keywords_str = "春節, 紅色, 喜慶, 燈籠"
        result = self.wrapper.extract_keywords_simple(keywords_str)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertIn("春節", result)
        self.assertIn("紅色", result)

    def test_extract_keywords_simple_with_spaces(self):
        """Test extract_keywords_simple handles extra spaces."""
        keywords_str = "  春節  ,   紅色,喜慶  "
        result = self.wrapper.extract_keywords_simple(keywords_str)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, ["春節", "紅色", "喜慶"])

    def test_extract_keywords_simple_empty_string(self):
        """Test extract_keywords_simple with empty string."""
        result = self.wrapper.extract_keywords_simple("")
        self.assertEqual(result, [])

    def test_extract_keywords_simple_only_commas(self):
        """Test extract_keywords_simple with only commas."""
        result = self.wrapper.extract_keywords_simple(", , ,")
        self.assertEqual(result, [])

    def test_generate_prompt_with_valid_input(self):
        """Test generate_prompt with valid input."""
        character_name = "Lulu Pig"
        character_desc = "可愛粉紅豬，大眼睛"
        trend_keywords = ["春節", "紅色"]

        try:
            prompt = self.wrapper.generate_prompt(
                character_name=character_name,
                character_desc=character_desc,
                trend_keywords=trend_keywords,
                max_retries=1  # Reduce retries for testing
            )

            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
            print(f"\n✅ Generated prompt (first 200 chars):\n{prompt[:200]}...")

        except PromptGenerationError as e:
            # If LLM API fails, that's acceptable in test environment
            print(f"\n⚠️ Prompt generation skipped (LLM API unavailable): {e}")
            self.skipTest("LLM API unavailable")

    def test_generate_prompt_empty_keywords(self):
        """Test generate_prompt raises error with empty keywords."""
        with self.assertRaises(ValueError):
            self.wrapper.generate_prompt(
                character_name="Lulu Pig",
                character_desc="可愛粉紅豬",
                trend_keywords=[]
            )

    def test_generate_prompt_none_keywords(self):
        """Test generate_prompt handles None keywords."""
        with self.assertRaises((ValueError, AttributeError)):
            self.wrapper.generate_prompt(
                character_name="Lulu Pig",
                character_desc="可愛粉紅豬",
                trend_keywords=None
            )


class TestTrendsAPIWrapperEdgeCases(unittest.TestCase):
    """Edge case tests for TrendsAPIWrapper."""

    def test_initialization_with_different_regions(self):
        """Test initialization with different regions."""
        regions = ['HK', 'TW', 'US']
        for region in regions:
            wrapper = TrendsAPIWrapper(region=region, lang='zh-TW')
            self.assertEqual(wrapper.region, region)

    def test_extract_keywords_with_special_characters(self):
        """Test extract_keywords_simple with special characters."""
        wrapper = TrendsAPIWrapper()
        keywords_str = "春節🎉, 紅色❤️, emoji😊"
        result = wrapper.extract_keywords_simple(keywords_str)

        self.assertEqual(len(result), 3)
        self.assertIn("春節🎉", result)

    def test_extract_keywords_with_long_input(self):
        """Test extract_keywords_simple with many keywords."""
        wrapper = TrendsAPIWrapper()
        keywords_str = ", ".join([f"keyword{i}" for i in range(50)])
        result = wrapper.extract_keywords_simple(keywords_str)

        self.assertEqual(len(result), 50)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
