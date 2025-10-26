"""
Quick Prompt Review Tool (Batch Mode)

Simplified review script for fast approval of generated prompts.
Can run in auto-approve mode or display prompts for quick review.

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
Story: 2.5 - Human Review & Validation Pipeline
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_prompts(prompts_dir: str = 'data/prompts') -> list:
    """Load all generated prompt files."""
    prompts_path = Path(prompts_dir)
    prompts = []

    # Load all .txt files
    for txt_file in sorted(prompts_path.glob('*_variation_*.txt')):
        # Parse filename: theme_variation_N.txt
        filename = txt_file.stem  # e.g., "halloween_variation_1"
        parts = filename.rsplit('_', 2)  # Split from right: ["halloween", "variation", "1"]

        if len(parts) == 3:
            theme = parts[0].replace('_', ' ').title()
            variation = parts[2]

            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()

            prompts.append({
                'theme': theme,
                'variation': variation,
                'prompt': prompt_text,
                'file': txt_file.name,
                'word_count': len(prompt_text.split())
            })

    return prompts


def display_prompt(prompt_dict: dict, index: int, total: int):
    """Display a single prompt with formatting."""
    print(f"\n{'='*80}")
    print(f"Prompt {index}/{total}: {prompt_dict['theme']} - Variation {prompt_dict['variation']}")
    print(f"{'='*80}")
    print(f"File: {prompt_dict['file']}")
    print(f"Word Count: {prompt_dict['word_count']} words")
    print(f"\n{'-'*80}")
    print(prompt_dict['prompt'])
    print(f"{'-'*80}\n")


def quick_validation(prompt: str) -> tuple[bool, list[str]]:
    """Quick automated validation checks."""
    issues = []

    # Check 1: Length
    word_count = len(prompt.split())
    if word_count < 50:
        issues.append(f"Too short: {word_count} words (min: 50)")
    elif word_count > 150:
        issues.append(f"Too long: {word_count} words (max: 150)")

    # Check 2: Character reference
    if 'lulu' not in prompt.lower() and 'pig' not in prompt.lower():
        issues.append("Missing character reference (Lulu Pig)")

    # Check 3: Basic quality
    if len(prompt) < 100:
        issues.append("Prompt too brief")

    is_valid = len(issues) == 0
    return is_valid, issues


def auto_approve_all(prompts: list, output_log: str = 'data/prompts/review_log.csv'):
    """Automatically approve all prompts that pass validation."""
    logger.info(f"\n{'='*80}")
    logger.info("Auto-Approve Mode: Validating all prompts")
    logger.info(f"{'='*80}\n")

    review_results = []
    approved = 0
    flagged = 0

    for i, prompt_dict in enumerate(prompts, 1):
        is_valid, issues = quick_validation(prompt_dict['prompt'])

        status = 'approved' if is_valid else 'flagged'
        feedback = '; '.join(issues) if issues else 'Passed all checks'

        review_results.append({
            'timestamp': datetime.now().isoformat(),
            'theme': prompt_dict['theme'],
            'variation': prompt_dict['variation'],
            'file': prompt_dict['file'],
            'prompt': prompt_dict['prompt'],
            'word_count': prompt_dict['word_count'],
            'status': status,
            'feedback': feedback,
            'reviewer': 'auto'
        })

        if is_valid:
            approved += 1
            logger.info(f"✅ {i}/{len(prompts)}: {prompt_dict['theme']} V{prompt_dict['variation']} - APPROVED")
        else:
            flagged += 1
            logger.warning(f"⚠️  {i}/{len(prompts)}: {prompt_dict['theme']} V{prompt_dict['variation']} - FLAGGED")
            logger.warning(f"   Issues: {'; '.join(issues)}")

    # Save review log
    df = pd.DataFrame(review_results)
    Path(output_log).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_log, index=False, encoding='utf-8-sig')

    # Summary
    print(f"\n{'='*80}")
    print("Auto-Review Complete!")
    print(f"{'='*80}")
    print(f"Total Prompts: {len(prompts)}")
    print(f"Approved: {approved} ({approved/len(prompts)*100:.1f}%)")
    print(f"Flagged: {flagged} ({flagged/len(prompts)*100:.1f}%)")
    print(f"\nReview log saved: {output_log}")
    print(f"{'='*80}\n")

    return approved, flagged


def display_all_prompts(prompts: list):
    """Display all prompts for manual review (no interaction needed)."""
    logger.info(f"\n{'='*80}")
    logger.info("Display Mode: Showing all 28 prompts")
    logger.info(f"{'='*80}\n")

    for i, prompt_dict in enumerate(prompts, 1):
        display_prompt(prompt_dict, i, len(prompts))

        # Validation feedback
        is_valid, issues = quick_validation(prompt_dict['prompt'])
        if is_valid:
            print("✅ Auto-validation: PASSED\n")
        else:
            print("⚠️  Auto-validation: FLAGGED")
            for issue in issues:
                print(f"   - {issue}")
            print()


def main():
    """
    Main execution for quick review.

    Modes:
    - auto: Automatically approve all valid prompts
    - display: Display all prompts without interaction
    """
    import sys

    print(f"\n{'='*80}")
    print("Quick Prompt Review Tool")
    print(f"{'='*80}\n")

    mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'

    # Load prompts
    prompts = load_all_prompts('data/prompts')
    logger.info(f"Loaded {len(prompts)} prompts from data/prompts/\n")

    if mode == 'auto':
        # Auto-approve mode
        approved, flagged = auto_approve_all(prompts)

        if flagged > 0:
            print(f"⚠️  {flagged} prompts flagged for review.")
            print(f"Run 'python obj1_nlp_prompt/quick_review.py display' to see details.\n")
        else:
            print("✅ All prompts approved! Ready for Objective 2 (Midjourney API).\n")

    elif mode == 'display':
        # Display mode
        display_all_prompts(prompts)
        print(f"{'='*80}")
        print("Review Complete - All prompts displayed")
        print(f"{'='*80}\n")

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python obj1_nlp_prompt/quick_review.py [auto|display]")


if __name__ == '__main__':
    main()
