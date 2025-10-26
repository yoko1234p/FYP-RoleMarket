"""
Batch Image Generation Pipeline

Automated pipeline for generating character IP design images
using approved prompts with character reference (--cref).

Features:
- Load all approved prompts from review log
- Batch generate images with progress tracking
- Automatic retry on failures
- Cost tracking and budget monitoring
- Resume capability (skip already generated images)
- Detailed logging and error reporting

Author: Product Manager (John)
Epic: 3 - Objective 2: Midjourney API Integration
Story: 3.3 - Batch Image Generation Pipeline

Usage:
    # Generate all prompts
    python obj2_midjourney_api/batch_generate.py

    # Generate specific range
    python obj2_midjourney_api/batch_generate.py --start 0 --end 5

    # Dry run (no actual generation)
    python obj2_midjourney_api/batch_generate.py --dry-run

    # Set budget limit
    python obj2_midjourney_api/batch_generate.py --max-cost 10.0
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from typing import List, Dict, Optional
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from obj2_midjourney_api.ttapi_client import TTAPIClient


class BatchImageGenerator:
    """
    Batch image generation pipeline for character IP designs.

    Manages the entire process of generating images from approved prompts,
    including progress tracking, error handling, and cost management.
    """

    def __init__(
        self,
        cref_urls: Optional[List[str]] = None,
        cref_weight: int = 100,
        mode: str = 'relax',
        max_cost: Optional[float] = None,
        output_log: str = 'reports/batch_generation_log.csv'
    ):
        """
        Initialize batch generator.

        Args:
            cref_urls: List of character reference image URLs
            cref_weight: Character reference weight (0-100)
            mode: TTAPI mode ('fast', 'relax', 'turbo'), default 'relax' for cost efficiency
            max_cost: Maximum budget in USD (None = no limit)
            output_log: Path to save generation log
        """
        self.client = TTAPIClient()
        self.cref_urls = cref_urls
        self.cref_weight = cref_weight
        self.mode = mode
        self.max_cost = max_cost
        self.output_log = Path(output_log)
        self.output_log.parent.mkdir(parents=True, exist_ok=True)

        # Load existing log if available
        self.generation_log = self._load_existing_log()

        print(f"\n{'='*80}")
        print("Batch Image Generator Initialized")
        print(f"{'='*80}")
        print(f"Character Reference URLs: {len(cref_urls) if cref_urls else 0}")
        print(f"Reference Weight: {cref_weight}")
        print(f"Generation Mode: {mode} (fast=90s, relax=~10min, turbo=60s)")
        print(f"Max Budget: ${max_cost if max_cost else 'No limit'}")
        print(f"Output Log: {output_log}")
        print(f"{'='*80}\n")

    def _load_existing_log(self) -> pd.DataFrame:
        """
        Load existing generation log if available.

        Returns:
            DataFrame with previous generation results
        """
        if self.output_log.exists():
            df = pd.read_csv(self.output_log, encoding='utf-8-sig')
            print(f"✅ Loaded existing log: {len(df)} previous generations")
            return df
        else:
            print("📝 No existing log found. Starting fresh.")
            return pd.DataFrame(columns=[
                'index', 'theme', 'variation', 'prompt', 'task_id',
                'status', 'image_url', 'local_path', 'duration',
                'cost', 'timestamp', 'error'
            ])

    def load_approved_prompts(self, review_log_path: str = 'data/prompts/review_log.csv') -> pd.DataFrame:
        """
        Load approved prompts from review log.

        Args:
            review_log_path: Path to review log CSV

        Returns:
            DataFrame of approved prompts

        Raises:
            FileNotFoundError: If review log doesn't exist
            ValueError: If no approved prompts found
        """
        print(f"\n{'='*80}")
        print("Loading Approved Prompts")
        print(f"{'='*80}\n")

        review_log = Path(review_log_path)
        if not review_log.exists():
            raise FileNotFoundError(f"Review log not found: {review_log_path}")

        df = pd.read_csv(review_log, encoding='utf-8-sig')

        # Filter approved prompts (case-insensitive)
        approved = df[df['status'].str.lower() == 'approved'].copy()

        if len(approved) == 0:
            raise ValueError("No approved prompts found in review log")

        print(f"✅ Total prompts in log: {len(df)}")
        print(f"✅ Approved prompts: {len(approved)}")
        print(f"✅ Approval rate: {len(approved)/len(df)*100:.1f}%\n")

        # Display breakdown by theme
        theme_counts = approved['theme'].value_counts()
        print("Breakdown by theme:")
        for theme, count in theme_counts.items():
            print(f"  {theme:25} {count} prompts")

        print(f"\n{'='*80}\n")

        return approved

    def _is_already_generated(self, index: int) -> bool:
        """
        Check if a prompt has already been generated successfully.

        Args:
            index: Prompt index

        Returns:
            True if already generated, False otherwise
        """
        if len(self.generation_log) == 0:
            return False

        matches = self.generation_log[self.generation_log['index'] == index]
        if len(matches) == 0:
            return False

        # Check if last attempt was successful
        last_attempt = matches.iloc[-1]
        return last_attempt['status'] == 'completed'

    def generate_batch(
        self,
        prompts_df: pd.DataFrame,
        start_index: int = 0,
        end_index: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict:
        """
        Generate images for a batch of prompts.

        Args:
            prompts_df: DataFrame of approved prompts
            start_index: Starting index (inclusive)
            end_index: Ending index (exclusive), None = all remaining
            dry_run: If True, only simulate generation (no API calls)

        Returns:
            Dictionary with generation summary
        """
        if end_index is None:
            end_index = len(prompts_df)

        # Validate indices
        if start_index < 0 or start_index >= len(prompts_df):
            raise ValueError(f"Invalid start_index: {start_index}")
        if end_index <= start_index or end_index > len(prompts_df):
            raise ValueError(f"Invalid end_index: {end_index}")

        batch = prompts_df.iloc[start_index:end_index].copy()

        print(f"\n{'='*80}")
        print(f"Batch Generation: Prompts {start_index} to {end_index-1}")
        print(f"{'='*80}\n")

        print(f"Total prompts in batch: {len(batch)}")
        print(f"Estimated cost: ${len(batch) * 0.40:.2f}")

        if self.max_cost:
            current_cost = self.client.total_cost
            remaining_budget = self.max_cost - current_cost
            print(f"Budget: ${current_cost:.2f} / ${self.max_cost:.2f}")
            print(f"Remaining: ${remaining_budget:.2f}")

            if remaining_budget < 0.40:
                print(f"\n❌ Insufficient budget! Need at least $0.40")
                return {'status': 'budget_exceeded', 'generated': 0}

        if dry_run:
            print(f"\n🔍 DRY RUN MODE - No actual generation\n")

        print(f"\n{'='*80}\n")

        # Generation loop
        results = []
        skipped = 0
        failed = 0
        succeeded = 0

        for idx, row in batch.iterrows():
            prompt_index = idx
            theme = row['theme']
            variation = row['variation']
            prompt = row['prompt']

            print(f"\n{'-'*80}")
            print(f"Prompt {prompt_index}/{len(prompts_df)-1}: {theme} (Variation {variation})")
            print(f"{'-'*80}")

            # Check if already generated
            if self._is_already_generated(prompt_index):
                print(f"⏭️  Skipping (already generated)")
                skipped += 1
                continue

            # Check budget before each generation
            if self.max_cost:
                if self.client.total_cost >= self.max_cost:
                    print(f"🛑 Budget limit reached (${self.max_cost})")
                    print(f"   Stopping generation")
                    break

            # Generate filename
            filename = f"{theme.lower().replace(' ', '_')}_var{variation}.png"

            if dry_run:
                print(f"📄 Prompt: {prompt[:80]}...")
                print(f"📁 Filename: {filename}")
                print(f"💰 Cost: $0.40 (simulated)")
                results.append({
                    'index': prompt_index,
                    'theme': theme,
                    'variation': variation,
                    'status': 'dry_run',
                    'cost': 0.40
                })
                succeeded += 1
                time.sleep(0.1)  # Simulate processing time
                continue

            # Actual generation
            try:
                print(f"🚀 Generating image...")
                print(f"   Prompt: {prompt[:80]}...")

                result = self.client.imagine(
                    prompt=prompt,
                    cref_urls=self.cref_urls,
                    cref_weight=self.cref_weight,
                    wait_for_completion=True,
                    save_image=True,
                    image_filename=filename
                )

                print(f"✅ Generation completed")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Duration: {result['duration']:.2f}s")
                print(f"   Image: {result.get('local_path', 'N/A')}")
                print(f"   Cost: ${result['cost']:.2f}")

                # Log result
                log_entry = {
                    'index': prompt_index,
                    'theme': theme,
                    'variation': variation,
                    'prompt': prompt,
                    'task_id': result['task_id'],
                    'status': 'completed',
                    'image_url': result.get('image_url', ''),
                    'local_path': result.get('local_path', ''),
                    'duration': result.get('duration', 0),
                    'cost': result.get('cost', 0),
                    'timestamp': datetime.now().isoformat(),
                    'error': ''
                }

                results.append(log_entry)
                succeeded += 1

                # Update log immediately
                self.generation_log = pd.concat([
                    self.generation_log,
                    pd.DataFrame([log_entry])
                ], ignore_index=True)
                self._save_log()

            except Exception as e:
                print(f"❌ Generation failed: {e}")

                # Log error
                error_entry = {
                    'index': prompt_index,
                    'theme': theme,
                    'variation': variation,
                    'prompt': prompt,
                    'task_id': '',
                    'status': 'failed',
                    'image_url': '',
                    'local_path': '',
                    'duration': 0,
                    'cost': 0,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }

                results.append(error_entry)
                failed += 1

                # Update log
                self.generation_log = pd.concat([
                    self.generation_log,
                    pd.DataFrame([error_entry])
                ], ignore_index=True)
                self._save_log()

        # Summary
        print(f"\n{'='*80}")
        print("Batch Generation Summary")
        print(f"{'='*80}\n")

        print(f"Total processed: {len(batch)}")
        print(f"Succeeded: {succeeded}")
        print(f"Failed: {failed}")
        print(f"Skipped (already generated): {skipped}")

        if not dry_run:
            cost_summary = self.client.get_cost_summary()
            print(f"\nCost:")
            print(f"  This batch: ${sum(r.get('cost', 0) for r in results):.2f}")
            print(f"  Total session: ${cost_summary['total_cost']:.2f}")
            print(f"  Images generated: {cost_summary['images_generated']}")

        print(f"\n{'='*80}\n")

        return {
            'status': 'completed',
            'total': len(batch),
            'succeeded': succeeded,
            'failed': failed,
            'skipped': skipped,
            'results': results
        }

    def _save_log(self):
        """Save generation log to CSV."""
        self.generation_log.to_csv(self.output_log, index=False, encoding='utf-8-sig')

    def get_generation_status(self) -> Dict:
        """
        Get overall generation progress status.

        Returns:
            Dictionary with status information
        """
        if len(self.generation_log) == 0:
            return {
                'total_attempts': 0,
                'completed': 0,
                'failed': 0,
                'completion_rate': 0.0
            }

        completed = len(self.generation_log[self.generation_log['status'] == 'completed'])
        failed = len(self.generation_log[self.generation_log['status'] == 'failed'])

        return {
            'total_attempts': len(self.generation_log),
            'completed': completed,
            'failed': failed,
            'completion_rate': completed / len(self.generation_log) * 100 if len(self.generation_log) > 0 else 0
        }


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Batch generate images from approved prompts'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index (default: 0)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending index (default: all remaining)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (no actual generation)'
    )
    parser.add_argument(
        '--max-cost',
        type=float,
        default=None,
        help='Maximum budget in USD (default: no limit)'
    )
    parser.add_argument(
        '--cref-urls',
        nargs='+',
        default=None,
        help='Character reference image URLs'
    )
    parser.add_argument(
        '--cref-weight',
        type=int,
        default=100,
        help='Character reference weight 0-100 (default: 100)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='relax',
        choices=['fast', 'relax', 'turbo'],
        help='TTAPI generation mode (default: relax for cost efficiency)'
    )

    args = parser.parse_args()

    print("\n" + "#"*80)
    print("# Batch Image Generation Pipeline")
    print("#"*80)

    try:
        # Initialize generator
        generator = BatchImageGenerator(
            cref_urls=args.cref_urls,
            cref_weight=args.cref_weight,
            mode=args.mode,
            max_cost=args.max_cost
        )

        # Load approved prompts
        prompts = generator.load_approved_prompts()

        # Check current status
        status = generator.get_generation_status()
        if status['total_attempts'] > 0:
            print(f"\n📊 Previous Generation Status:")
            print(f"   Total attempts: {status['total_attempts']}")
            print(f"   Completed: {status['completed']}")
            print(f"   Failed: {status['failed']}")
            print(f"   Completion rate: {status['completion_rate']:.1f}%\n")

        # Generate batch
        summary = generator.generate_batch(
            prompts,
            start_index=args.start,
            end_index=args.end,
            dry_run=args.dry_run
        )

        # Final summary
        print("\n" + "#"*80)
        print("# Generation Complete")
        print("#"*80 + "\n")

        print(f"Status: {summary['status']}")
        if summary['status'] != 'budget_exceeded':
            print(f"Processed: {summary['total']}")
            print(f"Succeeded: {summary['succeeded']}")
            print(f"Failed: {summary['failed']}")
            print(f"Skipped: {summary['skipped']}")

        print(f"\nLog saved to: {generator.output_log}")
        print("\n" + "#"*80 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
