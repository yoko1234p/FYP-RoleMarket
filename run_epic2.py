#!/usr/bin/env python3
"""
Epic 2 Master Execution Script

Runs all 6 stories of Objective 1 (Trend Intelligence & Prompt Generation) in sequence.

Author: Product Manager (John)
Epic: 2 - Objective 1: Trend Intelligence & Prompt Generation
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: str, story_name: str) -> bool:
    """
    Run a Python script and capture output.

    Args:
        script_path: Path to Python script
        story_name: Story identifier for logging

    Returns:
        True if script succeeded, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {story_name}: {script_path}")
    logger.info(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(result.stdout)
        if result.stderr:
            logger.warning(f"stderr: {result.stderr}")

        logger.info(f"\n✅ {story_name} completed successfully\n")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ {story_name} failed!")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def check_prerequisites():
    """Check if all required files and dependencies exist."""
    logger.info("Checking prerequisites...")

    # Check environment variables
    if not os.getenv('GPT_API_FREE_KEY'):
        logger.error("❌ GPT_API_FREE_KEY not found in environment")
        logger.error("   Please set it in .env file")
        return False

    # Check character config
    config_path = Path('config/character_config.json')
    if not config_path.exists():
        logger.error(f"❌ Character config not found: {config_path}")
        return False

    # Check character description
    desc_path = Path('data/character_descriptions/lulu_pig.txt')
    if not desc_path.exists():
        logger.error(f"❌ Character description not found: {desc_path}")
        return False

    # Check prompt template
    template_path = Path('obj1_nlp_prompt/templates/prompt_template.txt')
    if not template_path.exists():
        logger.error(f"❌ Prompt template not found: {template_path}")
        return False

    logger.info("✅ All prerequisites satisfied\n")
    return True


def main():
    """
    Main execution flow for Epic 2.

    Executes Stories 2.1-2.6 in sequence.
    """
    start_time = datetime.now()

    logger.info(f"\n{'#'*80}")
    logger.info(f"# Epic 2: Objective 1 - Trend Intelligence & Prompt Generation")
    logger.info(f"# Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*80}\n")

    # Check prerequisites
    if not check_prerequisites():
        logger.error("\n❌ Prerequisites check failed. Exiting.")
        sys.exit(1)

    # Story execution plan
    stories = [
        {
            'script': 'obj1_nlp_prompt/trends_extractor.py',
            'name': 'Story 2.1: Google Trends Data Extraction',
            'auto': True
        },
        {
            'script': 'obj1_nlp_prompt/keyword_extractor.py',
            'name': 'Story 2.2: TF-IDF Keyword Filtering',
            'auto': True
        },
        {
            'script': 'obj1_nlp_prompt/prompt_generator.py',
            'name': 'Story 2.4: LLM-based Prompt Generator',
            'auto': True
        },
        {
            'script': 'obj1_nlp_prompt/review_prompts.py',
            'name': 'Story 2.5: Human Review & Validation',
            'auto': False  # Requires human interaction
        }
    ]

    # Execute stories
    for i, story in enumerate(stories, 1):
        if not story['auto']:
            logger.info(f"\n{'='*80}")
            logger.info(f"{story['name']} requires manual execution")
            logger.info(f"{'='*80}")
            logger.info(f"\nPlease run manually:")
            logger.info(f"  python {story['script']}")
            logger.info(f"\nThis is an interactive review process.")
            logger.info(f"Press Enter to continue after completing the review...")
            input()
            continue

        success = run_script(story['script'], story['name'])

        if not success:
            logger.error(f"\n❌ Epic 2 execution failed at {story['name']}")
            logger.error(f"   Please fix errors and re-run.")
            sys.exit(1)

        # Brief pause between stories
        import time
        time.sleep(2)

    # Completion summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    logger.info(f"\n{'#'*80}")
    logger.info(f"# Epic 2 Execution Complete!")
    logger.info(f"# End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"# Total Duration: {elapsed}")
    logger.info(f"{'#'*80}\n")

    logger.info("✅ All automated stories completed successfully!")
    logger.info("\n📋 Next Steps:")
    logger.info("  1. Complete Story 2.5 manual review: python obj1_nlp_prompt/review_prompts.py")
    logger.info("  2. Verify all prompts approved (target: 100% approval rate)")
    logger.info("  3. Fill in metrics in: docs/contextual_reports/objective_1_completion.md")
    logger.info("  4. Prepare for Objective 2 (Midjourney API Integration)")
    logger.info("\n📁 Generated Data:")
    logger.info("  - data/trends/ - Raw trending keywords")
    logger.info("  - data/keywords/ - Filtered keywords (TF-IDF)")
    logger.info("  - data/prompts/ - Generated Midjourney prompts")
    logger.info(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
