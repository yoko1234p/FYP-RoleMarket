"""
Prompt Review and Validation Script

Manual review workflow for generated Midjourney prompts.
Allows human validation before using prompts for image generation.

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
Story: 2.5 - Human Review & Validation Pipeline
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptReviewer:
    """
    Human-in-the-loop prompt review system.

    Features:
    - Interactive CLI review
    - Approval/rejection tracking
    - Revision flagging
    - Review log export

    Usage:
        >>> reviewer = PromptReviewer()
        >>> reviewer.review_all_prompts('data/prompts')
    """

    def __init__(self, output_log: str = 'data/prompts/review_log.csv'):
        """
        Initialize prompt reviewer.

        Args:
            output_log: Path to review log CSV file
        """
        self.output_log = output_log
        self.review_results = []
        logger.info(f"PromptReviewer initialized, log: {output_log}")

    def review_prompt(
        self,
        theme: str,
        variation: int,
        prompt: str,
        keywords: str = ""
    ) -> Dict[str, any]:
        """
        Review a single prompt with human validation.

        Args:
            theme: Seasonal theme
            variation: Variation number (1-4)
            prompt: Generated prompt text
            keywords: Keywords used

        Returns:
            Review result dictionary
        """
        print(f"\n{'='*80}")
        print(f"Theme: {theme} | Variation: {variation}")
        print(f"{'='*80}")
        print(f"\nKeywords: {keywords}")
        print(f"\nPrompt ({len(prompt.split())} words):")
        print(f"{'-'*80}")
        print(prompt)
        print(f"{'-'*80}\n")

        # Checklist
        print("Review Checklist:")
        print("  1. Contains Lulu Pig character description?")
        print("  2. Incorporates seasonal theme naturally?")
        print("  3. Uses trending keywords appropriately?")
        print("  4. Prompt length 50-150 words?")
        print("  5. Suitable for merchandise design?")
        print("  6. No inappropriate content?")
        print("  7. Clear and actionable for Midjourney?")

        # Get approval
        while True:
            approval = input("\nApprove this prompt? (y/n/s): ").lower()
            if approval in ['y', 'n', 's']:
                break
            print("Invalid input. Enter 'y' (yes), 'n' (no), or 's' (skip)")

        # Get feedback if rejected
        feedback = ""
        if approval == 'n':
            print("\nWhat needs improvement?")
            print("  1. Missing character description")
            print("  2. Irrelevant keywords")
            print("  3. Poor readability")
            print("  4. Inappropriate content")
            print("  5. Length issue")
            print("  6. Other (specify)")

            feedback = input("Enter issue numbers (comma-separated) or custom feedback: ")

        result = {
            'timestamp': datetime.now().isoformat(),
            'theme': theme,
            'variation': variation,
            'prompt': prompt,
            'keywords': keywords,
            'word_count': len(prompt.split()),
            'status': 'approved' if approval == 'y' else ('skipped' if approval == 's' else 'rejected'),
            'feedback': feedback,
            'reviewer': 'human'
        }

        self.review_results.append(result)
        return result

    def review_all_prompts(self, prompts_dir: str = 'data/prompts'):
        """
        Review all generated prompts interactively.

        Args:
            prompts_dir: Directory containing prompt files
        """
        prompts_path = Path(prompts_dir)

        # Load all prompt CSVs
        all_prompts = []
        for csv_file in sorted(prompts_path.glob('*_prompts.csv')):
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            all_prompts.extend(df.to_dict('records'))

        if not all_prompts:
            logger.error(f"No prompts found in {prompts_dir}")
            return

        logger.info(f"Found {len(all_prompts)} prompts to review\n")

        # Review each prompt
        approved_count = 0
        rejected_count = 0
        skipped_count = 0

        for i, prompt_dict in enumerate(all_prompts, 1):
            print(f"\n\nPrompt {i}/{len(all_prompts)}")

            result = self.review_prompt(
                theme=prompt_dict['theme'],
                variation=prompt_dict['variation'],
                prompt=prompt_dict['prompt'],
                keywords=prompt_dict.get('keywords_used', '')
            )

            if result['status'] == 'approved':
                approved_count += 1
            elif result['status'] == 'rejected':
                rejected_count += 1
            else:
                skipped_count += 1

            # Show progress
            print(f"\nProgress: {approved_count} approved, {rejected_count} rejected, {skipped_count} skipped")

        # Save review log
        self.save_review_log()

        # Summary
        print(f"\n{'='*80}")
        print(f"Review Complete!")
        print(f"{'='*80}")
        print(f"Total prompts: {len(all_prompts)}")
        print(f"Approved: {approved_count} ({approved_count/len(all_prompts)*100:.1f}%)")
        print(f"Rejected: {rejected_count} ({rejected_count/len(all_prompts)*100:.1f}%)")
        print(f"Skipped: {skipped_count} ({skipped_count/len(all_prompts)*100:.1f}%)")
        print(f"\nReview log saved to: {self.output_log}")
        print(f"{'='*80}\n")

        if rejected_count > 0:
            print(f"⚠️  {rejected_count} prompts need revision!")
            print(f"Review feedback in: {self.output_log}")
            print(f"Regenerate rejected prompts using prompt_generator.py\n")

    def save_review_log(self):
        """Save review results to CSV."""
        df = pd.DataFrame(self.review_results)

        # Ensure output directory exists
        Path(self.output_log).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.output_log, index=False, encoding='utf-8-sig')
        logger.info(f"Saved review log with {len(df)} entries")

    def get_rejected_prompts(self) -> List[Dict]:
        """Get list of rejected prompts for regeneration."""
        return [r for r in self.review_results if r['status'] == 'rejected']

    def get_approval_rate(self) -> float:
        """Calculate approval rate percentage."""
        if not self.review_results:
            return 0.0

        approved = sum(1 for r in self.review_results if r['status'] == 'approved')
        total = len(self.review_results)
        return (approved / total) * 100.0


def main():
    """
    Main execution for Story 2.5.

    Interactive prompt review workflow.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Story 2.5: Human Review & Validation Pipeline")
    logger.info(f"{'='*80}\n")

    print("Welcome to Prompt Review System!")
    print("\nInstructions:")
    print("  - Review each generated prompt carefully")
    print("  - Check against quality checklist")
    print("  - Approve (y), Reject (n), or Skip (s) each prompt")
    print("  - Provide feedback for rejected prompts")
    print("\nTarget: 100% approval rate (0 rejected prompts)\n")

    input("Press Enter to start review...")

    # Initialize reviewer
    reviewer = PromptReviewer()

    # Review all prompts
    reviewer.review_all_prompts('data/prompts')

    # Check if target met
    approval_rate = reviewer.get_approval_rate()
    if approval_rate == 100.0:
        print("✅ Target achieved: 100% approval rate!")
    else:
        print(f"⚠️  Approval rate: {approval_rate:.1f}% (target: 100%)")
        rejected = reviewer.get_rejected_prompts()
        print(f"   {len(rejected)} prompts need regeneration")


if __name__ == '__main__':
    main()
