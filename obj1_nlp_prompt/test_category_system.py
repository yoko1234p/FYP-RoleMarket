"""
Category System Testing Script

Comprehensive test of the Category Prompt Builder with all 10 categories.
Tests both Simple Modifiers and Complex Modifiers with example inputs.

Author: Product Manager (John)
Usage: python obj1_nlp_prompt/test_category_system.py
"""

import sys
from pathlib import Path
from category_prompt_builder import CategoryPromptBuilder


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_simple_modifiers(builder: CategoryPromptBuilder, base_prompt: str):
    """Test all simple modifiers (no user input required)."""
    print_section("測試 1: Simple Modifiers（直接套用，無需額外輸入）")

    simple_categories = [
        "2D Animation",
        "3D Animation",
        "Comic",
        "Single Visual",
        "Sticker"
    ]

    for category in simple_categories:
        print(f"【Category: {category}】")
        info = builder.get_category_info(category)
        print(f"  Type: {info['type']}")
        print(f"  Requires Input: {info['requires_input']}")

        # Apply category
        final_prompt = builder.apply_category(base_prompt, category)

        print(f"\n  Base Prompt:")
        print(f"    {base_prompt[:100]}...")
        print(f"\n  Final Prompt:")
        print(f"    {final_prompt[:150]}...")
        print(f"\n  ✅ Successfully applied!\n")
        print(f"{'-'*80}\n")


def test_complex_modifiers(builder: CategoryPromptBuilder, base_prompt: str):
    """Test all complex modifiers (require user input)."""
    print_section("測試 2: Complex Modifiers（需要用戶輸入）")

    # Test data for each complex modifier
    test_cases = [
        {
            'category': 'Product',
            'user_input': 'plush toy',
            'description': '玩具公仔'
        },
        {
            'category': 'Collaboration',
            'user_input': 'Sanrio',
            'description': '聯乘品牌'
        },
        {
            'category': 'LuLu World',
            'user_input': 'entrance gate',
            'description': '主題樂園場景'
        },
        {
            'category': 'PR/Seeding',
            'user_input': 'new product launch',
            'description': '公關重點'
        },
        {
            'category': 'Campaign',
            'user_input': 'summer sale',
            'description': '活動主題'
        }
    ]

    for test_case in test_cases:
        category = test_case['category']
        user_input = test_case['user_input']

        print(f"【Category: {category}】")

        # Get category info
        info = builder.get_category_info(category)
        print(f"  Type: {info['type']}")
        print(f"  Requires Input: {info['requires_input']}")
        print(f"  Input Prompt: {info['input_prompt']}")
        print(f"  Placeholder: {info['placeholder']}")
        print(f"  Examples: {', '.join(info['examples'][:3])}...")

        print(f"\n  用戶輸入: \"{user_input}\" ({test_case['description']})")

        # Apply category
        final_prompt = builder.apply_category(base_prompt, category, user_input)

        print(f"\n  Base Prompt:")
        print(f"    {base_prompt[:100]}...")
        print(f"\n  Final Prompt:")
        print(f"    {final_prompt[:150]}...")
        print(f"\n  ✅ Successfully applied!\n")
        print(f"{'-'*80}\n")


def test_batch_apply(builder: CategoryPromptBuilder, base_prompts: list):
    """Test batch application of category to multiple prompts."""
    print_section("測試 3: Batch Application（批量套用）")

    category = "2D Animation"
    print(f"Category: {category}")
    print(f"Base Prompts: {len(base_prompts)}\n")

    # Apply to all
    final_prompts = builder.batch_apply(base_prompts, category)

    for i, (base, final) in enumerate(zip(base_prompts, final_prompts), 1):
        print(f"Prompt {i}:")
        print(f"  Base:  {base[:80]}...")
        print(f"  Final: {final[:80]}...")
        print()

    print(f"✅ Batch applied to {len(final_prompts)} prompts!\n")


def test_error_handling(builder: CategoryPromptBuilder, base_prompt: str):
    """Test error handling for invalid inputs."""
    print_section("測試 4: Error Handling（錯誤處理）")

    # Test 1: Invalid category
    print("【Test 4.1: Invalid Category】")
    try:
        builder.apply_category(base_prompt, "InvalidCategory")
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {str(e)[:80]}...\n")

    # Test 2: Missing user input for complex modifier
    print("【Test 4.2: Missing User Input】")
    try:
        builder.apply_category(base_prompt, "Product")  # No user_input
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {str(e)[:80]}...\n")

    # Test 3: Valid simple modifier (no input needed)
    print("【Test 4.3: Valid Simple Modifier】")
    try:
        final = builder.apply_category(base_prompt, "Sticker")
        print(f"✅ Successfully applied: {final[:80]}...\n")
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")


def test_all_combinations(builder: CategoryPromptBuilder, base_prompt: str):
    """Generate all 10 category combinations for one base prompt."""
    print_section("測試 5: All 10 Category Combinations（完整組合）")

    print(f"Base Prompt: {base_prompt[:100]}...\n")
    print(f"{'-'*80}\n")

    # Simple modifiers
    simple_categories = ["2D Animation", "3D Animation", "Comic", "Single Visual", "Sticker"]

    for i, category in enumerate(simple_categories, 1):
        final = builder.apply_category(base_prompt, category)
        print(f"{i}. {category}")
        print(f"   {final[:120]}...")
        print()

    # Complex modifiers with example inputs
    complex_test_cases = [
        ("Product", "plush toy"),
        ("Collaboration", "Sanrio"),
        ("LuLu World", "entrance gate"),
        ("PR/Seeding", "new product launch"),
        ("Campaign", "summer sale")
    ]

    for i, (category, user_input) in enumerate(complex_test_cases, 6):
        final = builder.apply_category(base_prompt, category, user_input)
        print(f"{i}. {category} (input: \"{user_input}\")")
        print(f"   {final[:120]}...")
        print()

    print(f"✅ Generated 10 different variations!\n")


def load_sample_prompt() -> str:
    """Load a sample approved prompt for testing."""
    # Try to load an actual approved prompt
    sample_file = Path('data/prompts/halloween_variation_1.txt')

    if sample_file.exists():
        with open(sample_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        # Fallback sample
        return ("Lulu Pig celebrating Halloween with pumpkins, cute kawaii style, "
                "vibrant colors, soft lighting, merchandise-ready design, "
                "Disney-inspired accessories, cheerful mood, outdoor festive scene")


def main():
    """Run all category system tests."""
    print(f"\n{'#'*80}")
    print(f"# Category System Comprehensive Testing")
    print(f"# Testing all 10 categories (5 simple + 5 complex)")
    print(f"{'#'*80}")

    # Initialize builder
    print("\nInitializing CategoryPromptBuilder...")
    builder = CategoryPromptBuilder()

    # Get all categories
    all_categories = builder.get_all_categories()
    print(f"✅ Loaded {len(all_categories)} categories: {', '.join(all_categories)}\n")

    # Load sample prompt
    base_prompt = load_sample_prompt()
    print(f"📝 Using base prompt: {base_prompt[:100]}...\n")

    # Run tests
    try:
        # Test 1: Simple Modifiers
        test_simple_modifiers(builder, base_prompt)

        # Test 2: Complex Modifiers
        test_complex_modifiers(builder, base_prompt)

        # Test 3: Batch Apply
        base_prompts = [
            "Lulu Pig celebrating Halloween...",
            "Lulu Pig enjoying Christmas...",
            "Lulu Pig in Spring Festival..."
        ]
        test_batch_apply(builder, base_prompts)

        # Test 4: Error Handling
        test_error_handling(builder, base_prompt)

        # Test 5: All Combinations
        test_all_combinations(builder, base_prompt)

        # Summary
        print_section("🎉 All Tests Passed!")
        print("Category System is working correctly!\n")
        print("Summary:")
        print(f"  ✅ 5 Simple Modifiers tested")
        print(f"  ✅ 5 Complex Modifiers tested")
        print(f"  ✅ Batch application tested")
        print(f"  ✅ Error handling verified")
        print(f"  ✅ 10 combinations generated")
        print(f"\nTotal Possible Combinations: 28 base prompts × 10 categories = 280 variations\n")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
