"""
Interactive Category Demo

Generate custom prompt combinations by selecting theme, variation, and category.
Great for testing before integrating into Streamlit Web App.

Author: Product Manager (John)
Usage: python obj1_nlp_prompt/demo_category_interactive.py <theme> <variation> <category> [user_input]

Examples:
  python obj1_nlp_prompt/demo_category_interactive.py halloween 1 "2D Animation"
  python obj1_nlp_prompt/demo_category_interactive.py christmas 2 Product "plush toy"
  python obj1_nlp_prompt/demo_category_interactive.py summer 3 Collaboration "Sanrio"
"""

import sys
from pathlib import Path
from category_prompt_builder import CategoryPromptBuilder


def load_prompt(theme: str, variation: int) -> str:
    """Load a specific prompt by theme and variation."""
    # Convert theme to filename format
    theme_lower = theme.lower().replace(' ', '_')

    # Find matching file
    prompt_file = Path(f'data/prompts/{theme_lower}_variation_{variation}.txt')

    if not prompt_file.exists():
        # Try alternative formats
        for file in Path('data/prompts').glob(f'*variation_{variation}.txt'):
            if theme_lower in file.stem.lower():
                prompt_file = file
                break

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {theme} variation {variation}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def list_available_prompts():
    """List all available theme/variation combinations."""
    print("\n📋 Available Prompts:\n")

    prompt_files = sorted(Path('data/prompts').glob('*_variation_*.txt'))

    themes_dict = {}
    for file in prompt_files:
        # Parse filename
        parts = file.stem.rsplit('_', 2)  # e.g., ["halloween", "variation", "1"]
        if len(parts) == 3:
            theme = parts[0].replace('_', ' ').title()
            variation = parts[2]

            if theme not in themes_dict:
                themes_dict[theme] = []
            themes_dict[theme].append(variation)

    for theme, variations in sorted(themes_dict.items()):
        print(f"  {theme}: variations {', '.join(sorted(variations))}")

    print()


def list_categories(builder: CategoryPromptBuilder):
    """List all available categories with their requirements."""
    print("\n🎨 Available Categories:\n")

    print("【Simple Modifiers (no input needed)】")
    all_cats = builder.get_all_categories()
    for cat in all_cats:
        if builder.is_simple_modifier(cat):
            print(f"  • {cat}")

    print("\n【Complex Modifiers (user input required)】")
    for cat in all_cats:
        if builder.is_complex_modifier(cat):
            info = builder.get_category_info(cat)
            print(f"  • {cat}")
            print(f"    Input: {info['input_prompt']}")
            print(f"    Example: {info['placeholder']}")

    print()


def main():
    """Interactive demo."""
    builder = CategoryPromptBuilder()

    # Check arguments
    if len(sys.argv) < 4:
        print("\n" + "="*80)
        print("Interactive Category Demo")
        print("="*80)
        print("\nUsage:")
        print("  python obj1_nlp_prompt/demo_category_interactive.py <theme> <variation> <category> [user_input]")
        print("\nExamples:")
        print('  python obj1_nlp_prompt/demo_category_interactive.py halloween 1 "2D Animation"')
        print('  python obj1_nlp_prompt/demo_category_interactive.py christmas 2 Product "plush toy"')
        print('  python obj1_nlp_prompt/demo_category_interactive.py summer 3 Collaboration "Sanrio"')
        print()

        list_available_prompts()
        list_categories(builder)
        sys.exit(1)

    # Parse arguments
    theme = sys.argv[1]
    variation = int(sys.argv[2])
    category = sys.argv[3]
    user_input = sys.argv[4] if len(sys.argv) > 4 else None

    # Load base prompt
    try:
        base_prompt = load_prompt(theme, variation)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        list_available_prompts()
        sys.exit(1)

    # Display selection
    print("\n" + "="*80)
    print(f"Selected Combination")
    print("="*80)
    print(f"Theme: {theme.title()}")
    print(f"Variation: {variation}")
    print(f"Category: {category}")
    if user_input:
        print(f"User Input: {user_input}")
    print()

    # Display base prompt
    print("="*80)
    print("Base Prompt")
    print("="*80)
    print(base_prompt)
    print()

    # Check if category requires input
    if builder.requires_user_input(category) and not user_input:
        info = builder.get_category_info(category)
        print("="*80)
        print("❌ This category requires user input!")
        print("="*80)
        print(f"Input Prompt: {info['input_prompt']}")
        print(f"Placeholder: {info['placeholder']}")
        print(f"Examples: {', '.join(info['examples'])}")
        print()
        print(f"Usage: python obj1_nlp_prompt/demo_category_interactive.py {theme} {variation} \"{category}\" \"<your_input>\"")
        print()
        sys.exit(1)

    # Apply category
    try:
        final_prompt = builder.apply_category(base_prompt, category, user_input)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Display final prompt
    print("="*80)
    print("Final Prompt (with Category Applied)")
    print("="*80)
    print(final_prompt)
    print()

    # Calculate stats
    base_words = len(base_prompt.split())
    final_words = len(final_prompt.split())
    added_words = final_words - base_words

    print("="*80)
    print("Statistics")
    print("="*80)
    print(f"Base Prompt:  {base_words} words")
    print(f"Final Prompt: {final_words} words")
    print(f"Added:        +{added_words} words from category modifier")
    print()

    print("✅ Ready to send to Midjourney API!")
    print()


if __name__ == '__main__':
    main()
