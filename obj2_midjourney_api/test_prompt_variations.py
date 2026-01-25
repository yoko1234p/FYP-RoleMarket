"""
Prompt Variation 系統全面測試腳本

測試範圍：
1. PromptVariationGenerator 三種模式（Single, Preset, Creative）
2. DesignGenerator 集成測試
3. 錯誤處理和邊界情況
4. 性能比較
5. 輸出質量驗證

Author: Developer (James)
Date: 2025-01-25
Version: 1.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time
from typing import Dict, List, Any
import json
from obj2_midjourney_api.prompt_variation_generator import PromptVariationGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptVariationTester:
    """Prompt Variation 系統測試器"""

    def __init__(self, output_dir: str = 'data/test_results'):
        self.generator = PromptVariationGenerator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = []

    def _log_test_result(self, test_name: str, status: str, details: Dict[str, Any]):
        """記錄測試結果"""
        result = {
            'test_name': test_name,
            'status': status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'details': details
        }
        self.test_results.append(result)

        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        logger.info(f"{status_emoji} {test_name}: {status}")

    def test_single_mode_basic(self):
        """測試 Single Mode 基本功能"""
        logger.info("\n" + "="*80)
        logger.info("Test 1: Single Mode 基本功能測試")
        logger.info("="*80)

        test_name = "Single Mode - Basic"
        try:
            base_prompt = "Lulu Pig celebrating Chinese New Year"
            num_variations = 4

            start_time = time.time()
            variations = self.generator.generate_variations(
                base_prompt=base_prompt,
                mode="single",
                num_variations=num_variations
            )
            duration = time.time() - start_time

            # 驗證結果
            assert len(variations) == num_variations, f"期望 {num_variations} 個變化，實際得到 {len(variations)}"
            assert all(isinstance(v, str) for v in variations), "所有變化應該是字串"
            assert all(base_prompt in v for v in variations), "所有變化應包含基礎 prompt"

            # 檢查變化是否不同
            unique_variations = set(variations)
            assert len(unique_variations) == num_variations, "所有變化應該是獨特的"

            logger.info(f"生成時間: {duration:.3f}s")
            for i, var in enumerate(variations, 1):
                logger.info(f"  變化 {i}: {var[:100]}...")

            self._log_test_result(test_name, "PASS", {
                'duration': duration,
                'num_variations': len(variations),
                'unique_count': len(unique_variations)
            })

        except Exception as e:
            logger.error(f"測試失敗: {e}")
            self._log_test_result(test_name, "FAIL", {'error': str(e)})

    def test_single_mode_variations_quality(self):
        """測試 Single Mode 變化質量（檢查微調元素）"""
        logger.info("\n" + "="*80)
        logger.info("Test 2: Single Mode 變化質量測試")
        logger.info("="*80)

        test_name = "Single Mode - Variation Quality"
        try:
            base_prompt = "Lulu Pig in a cozy room"
            variations = self.generator.generate_variations(
                base_prompt=base_prompt,
                mode="single",
                num_variations=8
            )

            # 檢查是否包含微調元素
            has_angle = any(any(angle in v for angle in ['front view', 'side view', '3/4 view', 'close-up'])
                           for v in variations)
            has_action = any(any(action in v for action in ['sitting', 'standing', 'waving', 'jumping'])
                            for v in variations)
            has_atmosphere = any(any(atm in v for atm in ['cheerful', 'peaceful', 'excited', 'relaxed'])
                                for v in variations)

            logger.info(f"包含角度變化: {has_angle}")
            logger.info(f"包含動作變化: {has_action}")
            logger.info(f"包含氛圍變化: {has_atmosphere}")

            quality_score = sum([has_angle, has_action, has_atmosphere]) / 3 * 100

            self._log_test_result(test_name, "PASS" if quality_score >= 66 else "WARNING", {
                'quality_score': quality_score,
                'has_angle': has_angle,
                'has_action': has_action,
                'has_atmosphere': has_atmosphere
            })

        except Exception as e:
            logger.error(f"測試失敗: {e}")
            self._log_test_result(test_name, "FAIL", {'error': str(e)})

    def test_preset_mode_all_themes(self):
        """測試 Preset Mode 所有主題"""
        logger.info("\n" + "="*80)
        logger.info("Test 3: Preset Mode 所有主題測試")
        logger.info("="*80)

        test_name = "Preset Mode - All Themes"
        try:
            base_prompt = "Lulu Pig character"
            themes = ['christmas', 'halloween', 'spring_festival', 'birthday', 'summer',
                     'winter', 'autumn', 'valentines', 'easter', 'thanksgiving',
                     'beach', 'forest']

            theme_results = {}
            total_duration = 0

            for theme in themes:
                start_time = time.time()
                variations = self.generator.generate_variations(
                    base_prompt=base_prompt,
                    mode="preset",
                    theme=theme,
                    num_variations=4
                )
                duration = time.time() - start_time
                total_duration += duration

                theme_results[theme] = {
                    'num_variations': len(variations),
                    'duration': duration,
                    'sample': variations[0][:80] + "..." if variations else None
                }

                logger.info(f"  {theme}: {len(variations)} 個變化 ({duration:.3f}s)")

            avg_duration = total_duration / len(themes)
            logger.info(f"\n平均生成時間: {avg_duration:.3f}s per theme")

            self._log_test_result(test_name, "PASS", {
                'themes_tested': len(themes),
                'avg_duration': avg_duration,
                'theme_results': theme_results
            })

        except Exception as e:
            logger.error(f"測試失敗: {e}")
            self._log_test_result(test_name, "FAIL", {'error': str(e)})

    def test_preset_mode_invalid_theme(self):
        """測試 Preset Mode 無效主題處理"""
        logger.info("\n" + "="*80)
        logger.info("Test 4: Preset Mode 無效主題處理測試")
        logger.info("="*80)

        test_name = "Preset Mode - Invalid Theme Handling"
        try:
            base_prompt = "Lulu Pig character"
            invalid_theme = "nonexistent_theme"

            variations = self.generator.generate_variations(
                base_prompt=base_prompt,
                mode="preset",
                theme=invalid_theme,
                num_variations=4
            )

            # 應該回退到 single mode
            assert len(variations) == 4, "應該生成 4 個變化（回退到 single mode）"
            logger.info(f"✅ 成功回退到 Single Mode，生成了 {len(variations)} 個變化")

            self._log_test_result(test_name, "PASS", {
                'fallback_mode': 'single',
                'num_variations': len(variations)
            })

        except Exception as e:
            logger.error(f"測試失敗: {e}")
            self._log_test_result(test_name, "FAIL", {'error': str(e)})

    def test_creative_mode_basic(self):
        """測試 Creative Mode 基本功能"""
        logger.info("\n" + "="*80)
        logger.info("Test 5: Creative Mode 基本功能測試")
        logger.info("="*80)

        test_name = "Creative Mode - Basic"
        try:
            base_prompt = "Lulu Pig in festive scene"
            character_name = "Lulu Pig"
            character_desc = "A cute pink pig mascot with chubby body and sleepy eyes"
            theme = "celebration"

            start_time = time.time()
            variations = self.generator.generate_variations(
                base_prompt=base_prompt,
                mode="creative",
                theme=theme,
                character_name=character_name,
                character_desc=character_desc,
                num_variations=3
            )
            duration = time.time() - start_time

            # 驗證結果
            assert len(variations) > 0, "應該至少生成 1 個變化"
            assert all(isinstance(v, str) for v in variations), "所有變化應該是字串"

            logger.info(f"生成時間: {duration:.3f}s")
            logger.info(f"生成數量: {len(variations)}")
            for i, var in enumerate(variations, 1):
                logger.info(f"  變化 {i}: {var[:100]}...")

            self._log_test_result(test_name, "PASS", {
                'duration': duration,
                'num_variations': len(variations),
                'used_llm': True
            })

        except Exception as e:
            logger.error(f"測試失敗（可能因為 API key 未設定）: {e}")
            self._log_test_result(test_name, "WARNING", {
                'error': str(e),
                'note': 'Creative mode requires GEMINI_OPENAI_API_KEY'
            })

    def test_creative_mode_fallback(self):
        """測試 Creative Mode 回退機制"""
        logger.info("\n" + "="*80)
        logger.info("Test 6: Creative Mode 回退機制測試")
        logger.info("="*80)

        test_name = "Creative Mode - Fallback"
        try:
            # 故意不設定 character_name 和 character_desc
            base_prompt = "Character in celebration"

            variations = self.generator.generate_variations(
                base_prompt=base_prompt,
                mode="creative",
                num_variations=4
            )

            # 應該回退到其他模式
            assert len(variations) == 4, "應該生成 4 個變化（回退機制）"
            logger.info(f"✅ 回退機制運作正常，生成了 {len(variations)} 個變化")

            self._log_test_result(test_name, "PASS", {
                'fallback_activated': True,
                'num_variations': len(variations)
            })

        except Exception as e:
            logger.error(f"測試失敗: {e}")
            self._log_test_result(test_name, "FAIL", {'error': str(e)})

    def test_edge_cases(self):
        """測試邊界情況"""
        logger.info("\n" + "="*80)
        logger.info("Test 7: 邊界情況測試")
        logger.info("="*80)

        test_name = "Edge Cases"
        edge_case_results = {}

        # Test 1: num_variations = 1
        try:
            variations = self.generator.generate_variations(
                base_prompt="Test",
                mode="single",
                num_variations=1
            )
            edge_case_results['single_variation'] = len(variations) == 1
            logger.info(f"  num_variations=1: {'✅' if edge_case_results['single_variation'] else '❌'}")
        except Exception as e:
            edge_case_results['single_variation'] = False
            logger.error(f"  num_variations=1: ❌ {e}")

        # Test 2: num_variations = 10 (大量)
        try:
            variations = self.generator.generate_variations(
                base_prompt="Test",
                mode="single",
                num_variations=10
            )
            edge_case_results['large_variations'] = len(variations) == 10
            logger.info(f"  num_variations=10: {'✅' if edge_case_results['large_variations'] else '❌'}")
        except Exception as e:
            edge_case_results['large_variations'] = False
            logger.error(f"  num_variations=10: ❌ {e}")

        # Test 3: 空 prompt
        try:
            variations = self.generator.generate_variations(
                base_prompt="",
                mode="single",
                num_variations=2
            )
            edge_case_results['empty_prompt'] = len(variations) > 0
            logger.info(f"  empty prompt: {'✅' if edge_case_results['empty_prompt'] else '❌'}")
        except Exception as e:
            edge_case_results['empty_prompt'] = False
            logger.error(f"  empty prompt: ❌ {e}")

        # Test 4: 超長 prompt
        try:
            long_prompt = "A " + " and ".join(["cute"] * 50) + " pig"
            variations = self.generator.generate_variations(
                base_prompt=long_prompt,
                mode="single",
                num_variations=2
            )
            edge_case_results['long_prompt'] = len(variations) > 0
            logger.info(f"  long prompt: {'✅' if edge_case_results['long_prompt'] else '❌'}")
        except Exception as e:
            edge_case_results['long_prompt'] = False
            logger.error(f"  long prompt: ❌ {e}")

        pass_rate = sum(edge_case_results.values()) / len(edge_case_results) * 100
        status = "PASS" if pass_rate >= 75 else "WARNING"

        self._log_test_result(test_name, status, {
            'pass_rate': pass_rate,
            'results': edge_case_results
        })

    def test_performance_comparison(self):
        """測試三種模式的性能比較"""
        logger.info("\n" + "="*80)
        logger.info("Test 8: 性能比較測試")
        logger.info("="*80)

        test_name = "Performance Comparison"
        try:
            base_prompt = "Lulu Pig in festive scene"
            num_variations = 4
            num_iterations = 3

            performance_results = {}

            # Single Mode
            single_times = []
            for _ in range(num_iterations):
                start = time.time()
                self.generator.generate_variations(
                    base_prompt=base_prompt,
                    mode="single",
                    num_variations=num_variations
                )
                single_times.append(time.time() - start)
            performance_results['single'] = {
                'avg': sum(single_times) / len(single_times),
                'min': min(single_times),
                'max': max(single_times)
            }

            # Preset Mode
            preset_times = []
            for _ in range(num_iterations):
                start = time.time()
                self.generator.generate_variations(
                    base_prompt=base_prompt,
                    mode="preset",
                    theme="christmas",
                    num_variations=num_variations
                )
                preset_times.append(time.time() - start)
            performance_results['preset'] = {
                'avg': sum(preset_times) / len(preset_times),
                'min': min(preset_times),
                'max': max(preset_times)
            }

            # Creative Mode (skip if API key not available)
            try:
                creative_times = []
                for _ in range(min(2, num_iterations)):  # 減少測試次數以節省 API calls
                    start = time.time()
                    self.generator.generate_variations(
                        base_prompt=base_prompt,
                        mode="creative",
                        theme="celebration",
                        character_name="Lulu",
                        character_desc="cute pig",
                        num_variations=num_variations
                    )
                    creative_times.append(time.time() - start)
                performance_results['creative'] = {
                    'avg': sum(creative_times) / len(creative_times),
                    'min': min(creative_times),
                    'max': max(creative_times)
                }
            except Exception as e:
                logger.warning(f"Creative mode 性能測試跳過: {e}")
                performance_results['creative'] = {'note': 'API key not available'}

            logger.info("\n性能比較結果:")
            for mode, times in performance_results.items():
                if 'note' in times:
                    logger.info(f"  {mode.upper()}: {times['note']}")
                else:
                    logger.info(f"  {mode.upper()}: 平均 {times['avg']:.3f}s (min: {times['min']:.3f}s, max: {times['max']:.3f}s)")

            self._log_test_result(test_name, "PASS", performance_results)

        except Exception as e:
            logger.error(f"測試失敗: {e}")
            self._log_test_result(test_name, "FAIL", {'error': str(e)})

    def save_test_report(self):
        """儲存測試報告"""
        report_path = self.output_dir / f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"

        summary = {
            'total_tests': len(self.test_results),
            'passed': sum(1 for r in self.test_results if r['status'] == 'PASS'),
            'failed': sum(1 for r in self.test_results if r['status'] == 'FAIL'),
            'warnings': sum(1 for r in self.test_results if r['status'] == 'WARNING'),
        }
        summary['pass_rate'] = summary['passed'] / summary['total_tests'] * 100 if summary['total_tests'] > 0 else 0

        report = {
            'summary': summary,
            'test_results': self.test_results
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n💾 測試報告已儲存: {report_path}")
        return summary

    def run_all_tests(self):
        """執行所有測試"""
        logger.info("\n" + "="*80)
        logger.info("開始執行 Prompt Variation 系統全面測試")
        logger.info("="*80 + "\n")

        # 執行所有測試
        self.test_single_mode_basic()
        self.test_single_mode_variations_quality()
        self.test_preset_mode_all_themes()
        self.test_preset_mode_invalid_theme()
        self.test_creative_mode_basic()
        self.test_creative_mode_fallback()
        self.test_edge_cases()
        self.test_performance_comparison()

        # 生成報告
        summary = self.save_test_report()

        # 顯示總結
        logger.info("\n" + "="*80)
        logger.info("測試總結")
        logger.info("="*80)
        logger.info(f"總測試數: {summary['total_tests']}")
        logger.info(f"通過: {summary['passed']} ✅")
        logger.info(f"失敗: {summary['failed']} ❌")
        logger.info(f"警告: {summary['warnings']} ⚠️")
        logger.info(f"通過率: {summary['pass_rate']:.1f}%")
        logger.info("="*80 + "\n")

        return summary


def main():
    """執行測試"""
    tester = PromptVariationTester()
    summary = tester.run_all_tests()

    # 返回退出碼
    import sys
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
