"""
Category Prompt Builder Module

Dynamically combines base prompts with category modifiers.
Supports both simple modifiers and complex modifiers requiring user input.

Author: Product Manager (John)
Usage: Objective 4 (Streamlit Web App) - Category selection feature
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryPromptBuilder:
    """
    Build final prompts by combining base prompts with category modifiers.

    Features:
    - Simple modifiers (direct suffix)
    - Complex modifiers (user input required)
    - Category validation
    - Examples and placeholders for user guidance

    Usage:
        >>> builder = CategoryPromptBuilder()
        >>>
        >>> # Simple modifier (no input needed)
        >>> final_prompt = builder.apply_category(
        ...     base_prompt="Lulu Pig celebrating Halloween...",
        ...     category="2D Animation"
        ... )
        >>>
        >>> # Complex modifier (user input required)
        >>> final_prompt = builder.apply_category(
        ...     base_prompt="Lulu Pig celebrating Halloween...",
        ...     category="Product",
        ...     user_input="plush toy"
        ... )
    """

    def __init__(self, config_path: str = 'config/character_config.json'):
        """
        Initialize category prompt builder.

        Args:
            config_path: Path to character config JSON file
        """
        self.config_path = config_path
        self.config = self._load_config()

        self.simple_modifiers = self.config['categories']['simple_modifiers']
        self.complex_modifiers = self.config['categories']['complex_modifiers']

        logger.info(f"CategoryPromptBuilder initialized")
        logger.info(f"  Simple modifiers: {len(self.simple_modifiers)}")
        logger.info(f"  Complex modifiers: {len(self.complex_modifiers)}")

    def _load_config(self) -> Dict:
        """Load character configuration from JSON."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def get_all_categories(self) -> List[str]:
        """
        Get list of all available categories.

        Returns:
            List of category names (simple + complex)
        """
        simple_cats = list(self.simple_modifiers.keys())
        complex_cats = list(self.complex_modifiers.keys())
        return simple_cats + complex_cats

    def is_simple_modifier(self, category: str) -> bool:
        """Check if category is a simple modifier (no input required)."""
        return category in self.simple_modifiers

    def is_complex_modifier(self, category: str) -> bool:
        """Check if category is a complex modifier (input required)."""
        return category in self.complex_modifiers

    def requires_user_input(self, category: str) -> bool:
        """Check if category requires user input."""
        return self.is_complex_modifier(category)

    def get_input_prompt(self, category: str) -> Optional[str]:
        """
        Get input prompt text for complex modifiers.

        Args:
            category: Category name

        Returns:
            Input prompt text (Chinese), or None if simple modifier
        """
        if self.is_complex_modifier(category):
            return self.complex_modifiers[category]['input_prompt']
        return None

    def get_placeholder(self, category: str) -> Optional[str]:
        """Get placeholder text for complex modifier input field."""
        if self.is_complex_modifier(category):
            return self.complex_modifiers[category]['placeholder']
        return None

    def get_examples(self, category: str) -> Optional[List[str]]:
        """Get example inputs for complex modifiers."""
        if self.is_complex_modifier(category):
            return self.complex_modifiers[category]['examples']
        return None

    def apply_category(
        self,
        base_prompt: str,
        category: str,
        user_input: Optional[str] = None
    ) -> str:
        """
        Apply category modifier to base prompt.

        Args:
            base_prompt: Original prompt from Epic 2 (theme-based)
            category: Category name (e.g., "2D Animation", "Product")
            user_input: User input for complex modifiers (required if category needs it)

        Returns:
            Final prompt with category modifier applied

        Raises:
            ValueError: If category is invalid or user_input missing for complex modifier

        Example:
            >>> builder = CategoryPromptBuilder()
            >>>
            >>> # Simple modifier
            >>> base = "Lulu Pig celebrating Halloween with pumpkins, cute kawaii style"
            >>> final = builder.apply_category(base, "2D Animation")
            >>> print(final)
            "Lulu Pig celebrating Halloween with pumpkins, cute kawaii style, 2D animation style, cartoon aesthetic, vibrant colors, animation-ready, cel-shaded"
            >>>
            >>> # Complex modifier
            >>> final = builder.apply_category(base, "Product", user_input="plush toy")
            >>> print(final)
            "Lulu Pig celebrating Halloween with pumpkins, cute kawaii style, plush toy product design, merchandise-ready, 3D product render, commercial quality, realistic materials"
        """
        # Validate category
        if category not in self.get_all_categories():
            raise ValueError(f"Invalid category: {category}. Available: {self.get_all_categories()}")

        # Simple modifier
        if self.is_simple_modifier(category):
            suffix = self.simple_modifiers[category]['suffix']
            return base_prompt + suffix

        # Complex modifier
        if self.is_complex_modifier(category):
            if not user_input:
                raise ValueError(f"Category '{category}' requires user input. "
                               f"Prompt: {self.get_input_prompt(category)}")

            template = self.complex_modifiers[category]['template']

            # Replace placeholder in template
            if category == "Product":
                modifier = template.format(product_type=user_input)
            elif category == "Collaboration":
                modifier = template.format(brand_name=user_input)
            elif category == "LuLu World":
                modifier = template.format(scene_description=user_input)
            elif category == "PR/Seeding":
                modifier = template.format(campaign_focus=user_input)
            elif category == "Campaign":
                modifier = template.format(campaign_theme=user_input)
            else:
                # Generic fallback
                modifier = template.format(user_input=user_input)

            return base_prompt + modifier

    def get_category_info(self, category: str) -> Dict:
        """
        Get complete information about a category.

        Args:
            category: Category name

        Returns:
            Dictionary with category metadata

        Example:
            >>> info = builder.get_category_info("Product")
            >>> print(info)
            {
                'name': 'Product',
                'type': 'complex',
                'requires_input': True,
                'input_prompt': '請輸入產品類型...',
                'placeholder': 'plush toy',
                'examples': ['plush toy 玩具公仔', 'T-shirt T恤', ...]
            }
        """
        if self.is_simple_modifier(category):
            return {
                'name': category,
                'type': 'simple',
                'requires_input': False,
                'suffix': self.simple_modifiers[category]['suffix']
            }
        elif self.is_complex_modifier(category):
            modifier = self.complex_modifiers[category]
            return {
                'name': category,
                'type': 'complex',
                'requires_input': True,
                'input_prompt': modifier['input_prompt'],
                'placeholder': modifier['placeholder'],
                'examples': modifier['examples'],
                'template': modifier['template']
            }
        else:
            raise ValueError(f"Unknown category: {category}")

    def batch_apply(
        self,
        base_prompts: List[str],
        category: str,
        user_input: Optional[str] = None
    ) -> List[str]:
        """
        Apply category to multiple base prompts at once.

        Args:
            base_prompts: List of base prompts
            category: Category to apply
            user_input: User input for complex modifiers

        Returns:
            List of final prompts with category applied

        Example:
            >>> base_prompts = [
            ...     "Lulu Pig celebrating Halloween...",
            ...     "Lulu Pig celebrating Christmas...",
            ... ]
            >>> finals = builder.batch_apply(base_prompts, "Sticker")
            >>> print(len(finals))
            2
        """
        return [
            self.apply_category(base, category, user_input)
            for base in base_prompts
        ]


def demo():
    """
    Demo usage of CategoryPromptBuilder.
    """
    print("\n" + "="*80)
    print("Category Prompt Builder Demo")
    print("="*80 + "\n")

    builder = CategoryPromptBuilder()

    # Sample base prompt
    base_prompt = "Lulu Pig celebrating Halloween with pumpkins, cute kawaii style, vibrant colors"

    print(f"Base Prompt:\n  {base_prompt}\n")
    print("="*80 + "\n")

    # Demo 1: Simple modifiers
    print("【Simple Modifiers】(No user input required)\n")

    for category in ["2D Animation", "3D Animation", "Comic", "Sticker"]:
        final = builder.apply_category(base_prompt, category)
        print(f"Category: {category}")
        print(f"  → {final}\n")

    print("="*80 + "\n")

    # Demo 2: Complex modifiers
    print("【Complex Modifiers】(User input required)\n")

    test_inputs = {
        "Product": "plush toy",
        "Collaboration": "Sanrio",
        "LuLu World": "entrance gate",
        "PR/Seeding": "new product launch",
        "Campaign": "summer sale"
    }

    for category, user_input in test_inputs.items():
        info = builder.get_category_info(category)
        print(f"Category: {category}")
        print(f"  Input Prompt: {info['input_prompt']}")
        print(f"  User Input: \"{user_input}\"")

        final = builder.apply_category(base_prompt, category, user_input)
        print(f"  → {final}\n")

    print("="*80 + "\n")

    # Demo 3: Category info
    print("【Category Information】\n")

    product_info = builder.get_category_info("Product")
    print(f"Category: Product")
    print(f"  Type: {product_info['type']}")
    print(f"  Requires Input: {product_info['requires_input']}")
    print(f"  Placeholder: {product_info['placeholder']}")
    print(f"  Examples: {', '.join(product_info['examples'][:3])}...")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    demo()
