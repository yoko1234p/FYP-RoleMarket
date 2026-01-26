"""
Async Batch Image Generation Pipeline with Concurrent Task Monitoring

Handles TTAPI relax mode (~10 min per image) with concurrent task submission
and real-time progress monitoring.

Key Features:
- Submit up to 10 tasks concurrently (TTAPI limit)
- Monitor all tasks in parallel with threading
- Real-time progress display
- Automatic retry on failures
- Resume capability

Author: Product Manager (John)
Epic: 3 - Objective 2: Midjourney API Integration

Usage:
    # Generate 5 images with 3 concurrent tasks
    python obj2_midjourney_api/batch_generate_async.py --max-images 5 --max-concurrent 3

    # Generate all 28 images with 10 concurrent tasks
    python obj2_midjourney_api/batch_generate_async.py --max-concurrent 10
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from typing import List, Dict, Optional
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from obj2_midjourney_api.ttapi_client import TTAPIClient


class AsyncBatchGenerator:
    """
    Asynchronous batch image generator with concurrent task monitoring.

    Handles relax mode (~10 min) by submitting multiple tasks concurrently
    and monitoring them in parallel threads.
    """

    def __init__(
        self,
        cref_urls: List[str],
        cref_weight: int = 100,
        mode: str = 'relax',
        max_concurrent: int = 10,
        output_log: str = 'reports/batch_generation_log.csv'
    ):
        """
        Initialize async batch generator.

        Args:
            cref_urls: Character reference image URLs
            cref_weight: Character reference weight (0-100)
            mode: TTAPI mode ('fast', 'relax', 'turbo')
            max_concurrent: Maximum concurrent tasks (max 10 for TTAPI)
            output_log: Path to save generation log
        """
        self.client = TTAPIClient()
        self.cref_urls = cref_urls
        self.cref_weight = cref_weight
        self.mode = mode
        self.max_concurrent = min(max_concurrent, 10)  # TTAPI limit
        self.output_log = Path(output_log)
        self.output_log.parent.mkdir(parents=True, exist_ok=True)

        # Thread-safe progress tracking
        self.lock = Lock()
        self.completed = 0
        self.failed = 0
        self.total = 0

        # Load existing log
        self.generation_log = self._load_existing_log()

        print(f"\n{'='*80}")
        print("Async Batch Image Generator Initialized")
        print(f"{'='*80}")
        print(f"Mode: {mode} (~10 min per image)")
        print(f"Max Concurrent Tasks: {self.max_concurrent}")
        print(f"Character Reference URLs: {len(cref_urls)}")
        print(f"Reference Weight: {cref_weight}")
        print(f"{'='*80}\n")

    def _load_existing_log(self) -> pd.DataFrame:
        """Load existing generation log."""
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

    def _is_already_generated(self, index: int) -> bool:
        """Check if prompt already generated successfully."""
        if len(self.generation_log) == 0:
            return False

        matches = self.generation_log[self.generation_log['index'] == index]
        if len(matches) == 0:
            return False

        last_attempt = matches.iloc[-1]
        return last_attempt['status'] == 'completed'

    def _save_log(self):
        """Save generation log to CSV (thread-safe)."""
        with self.lock:
            self.generation_log.to_csv(self.output_log, index=False, encoding='utf-8-sig')

    def _update_progress(self, success: bool = True):
        """Update progress counters (thread-safe)."""
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

            # Print progress
            progress = (self.completed + self.failed) / self.total * 100 if self.total > 0 else 0
            print(f"\n{'='*80}")
            print(f"Progress: {self.completed + self.failed}/{self.total} ({progress:.1f}%)")
            print(f"✅ Completed: {self.completed} | ❌ Failed: {self.failed}")
            print(f"{'='*80}\n")

    def generate_single_image(
        self,
        prompt_index: int,
        theme: str,
        variation: int,
        prompt: str
    ) -> Dict:
        """
        Generate a single image (runs in thread).

        Args:
            prompt_index: Prompt index
            theme: Theme name
            variation: Variation number
            prompt: Prompt text

        Returns:
            Generation result dictionary
        """
        filename = f"{theme.lower().replace(' ', '_')}_var{variation}.png"

        try:
            print(f"\n[Thread {prompt_index}] 🚀 Starting: {theme} (Variation {variation})")
            print(f"[Thread {prompt_index}] Prompt: {prompt[:80]}...")

            result = self.client.imagine(
                prompt=prompt,
                cref_urls=self.cref_urls,
                cref_weight=self.cref_weight,
                wait_for_completion=True,
                save_image=True,
                image_filename=filename
            )

            print(f"[Thread {prompt_index}] ✅ Completed: {theme}")
            print(f"[Thread {prompt_index}]    Duration: {result['duration']:.2f}s")
            print(f"[Thread {prompt_index}]    Cost: ${result['cost']}")

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

            # Update log (thread-safe)
            with self.lock:
                self.generation_log = pd.concat([
                    self.generation_log,
                    pd.DataFrame([log_entry])
                ], ignore_index=True)
                self._save_log()

            self._update_progress(success=True)
            return log_entry

        except Exception as e:
            print(f"[Thread {prompt_index}] ❌ Failed: {theme}")
            print(f"[Thread {prompt_index}]    Error: {e}")

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

            # Update log (thread-safe)
            with self.lock:
                self.generation_log = pd.concat([
                    self.generation_log,
                    pd.DataFrame([error_entry])
                ], ignore_index=True)
                self._save_log()

            self._update_progress(success=False)
            return error_entry

    def generate_batch_async(
        self,
        prompts_df: pd.DataFrame,
        max_images: Optional[int] = None
    ) -> Dict:
        """
        Generate images concurrently using thread pool.

        Args:
            prompts_df: DataFrame of approved prompts
            max_images: Maximum number of images to generate (None = all)

        Returns:
            Summary dictionary
        """
        # Filter already generated
        to_generate = []
        skipped = 0

        for idx, row in prompts_df.iterrows():
            if self._is_already_generated(idx):
                skipped += 1
                continue

            to_generate.append({
                'index': idx,
                'theme': row['theme'],
                'variation': row['variation'],
                'prompt': row['prompt']
            })

            if max_images and len(to_generate) >= max_images:
                break

        self.total = len(to_generate)

        print(f"\n{'='*80}")
        print("Batch Generation Plan")
        print(f"{'='*80}\n")
        print(f"Total prompts: {len(prompts_df)}")
        print(f"Already generated: {skipped}")
        print(f"To generate: {self.total}")
        print(f"Estimated cost: ${self.total * 0.40:.2f}")
        print(f"Estimated time (sequential): {self.total * 10:.0f} min")
        print(f"Estimated time (concurrent x{self.max_concurrent}): {self.total / self.max_concurrent * 10:.0f} min")
        print(f"\n{'='*80}\n")

        if self.total == 0:
            print("✅ All images already generated!")
            return {
                'status': 'completed',
                'total': 0,
                'completed': 0,
                'failed': 0,
                'skipped': skipped
            }

        # Start time
        start_time = time.time()

        # Submit tasks to thread pool
        print(f"🚀 Starting concurrent generation with {self.max_concurrent} workers...\n")

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            futures = []
            for item in to_generate:
                future = executor.submit(
                    self.generate_single_image,
                    item['index'],
                    item['theme'],
                    item['variation'],
                    item['prompt']
                )
                futures.append(future)

            # Wait for completion
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"❌ Thread exception: {e}")

        # Summary
        total_time = time.time() - start_time

        print(f"\n{'='*80}")
        print("Batch Generation Summary")
        print(f"{'='*80}\n")
        print(f"Total processed: {self.total}")
        print(f"✅ Completed: {self.completed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"\nTotal time: {total_time/60:.2f} minutes")
        print(f"Average time per image: {total_time/self.total:.2f}s")

        # Cost summary
        cost_summary = self.client.get_cost_summary()
        print(f"\n💰 Cost:")
        print(f"   This batch: ${self.completed * 0.40:.2f}")
        print(f"   Total session: ${cost_summary['total_cost']:.2f}")
        print(f"   Images generated: {cost_summary['images_generated']}")

        print(f"\n{'='*80}\n")

        return {
            'status': 'completed',
            'total': self.total,
            'completed': self.completed,
            'failed': self.failed,
            'skipped': skipped,
            'duration': total_time
        }


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Async batch generate images with concurrent monitoring'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to generate (default: all)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Maximum concurrent tasks (default: 3, max: 10)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='relax',
        choices=['fast', 'relax', 'turbo'],
        help='TTAPI generation mode (default: relax)'
    )
    parser.add_argument(
        '--cref-weight',
        type=int,
        default=100,
        help='Character reference weight 0-100 (default: 100)'
    )

    args = parser.parse_args()

    print("\n" + "#"*80)
    print("# Async Batch Image Generation Pipeline")
    print("#"*80)

    try:
        # Load Imgur URLs from config
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.reference_images import CREF_URLS

        # Initialize generator
        generator = AsyncBatchGenerator(
            cref_urls=CREF_URLS,
            cref_weight=args.cref_weight,
            mode=args.mode,
            max_concurrent=args.max_concurrent
        )

        # Load approved prompts
        review_log = pd.read_csv('data/prompts/review_log.csv', encoding='utf-8-sig')
        approved = review_log[review_log['status'].str.lower() == 'approved'].copy()

        print(f"✅ Loaded {len(approved)} approved prompts\n")

        # Generate batch
        summary = generator.generate_batch_async(
            approved,
            max_images=args.max_images
        )

        print("\n" + "#"*80)
        print("# Generation Complete")
        print("#"*80 + "\n")

        print(f"Status: {summary['status']}")
        print(f"Total: {summary['total']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")

        if summary.get('duration'):
            print(f"Duration: {summary['duration']/60:.2f} minutes")

        print(f"\nLog saved to: reports/batch_generation_log.csv")
        print("\n" + "#"*80 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
