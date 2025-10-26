"""
Error Handling Test for PromptGenerator

Tests various error scenarios including API errors, quota limits, and malformed responses.

Author: Product Manager (John)
Usage: python obj1_nlp_prompt/test_error_handling.py
"""

from prompt_generator import PromptGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_parse_api_error():
    """Test API error parsing with various error formats."""
    print("\n" + "="*80)
    print("Test 1: API Error Parsing")
    print("="*80 + "\n")

    generator = PromptGenerator()

    # Simulate different error types
    test_cases = [
        {
            'name': 'Mock ChatAnywhere Error',
            'exception': type('MockException', (), {
                'response': type('MockResponse', (), {
                    'json': lambda: {
                        "error": {
                            "message": "Unexpected character ('}' (code 125)): was expecting double-quote to start field name【如果您遇到问题，欢迎加入QQ群咨询：1048463714】",
                            "type": "chatanywhere_error",
                            "param": None,
                            "code": "400 BAD_REQUEST"
                        }
                    }
                })()
            })(),
            'expected_contains': ['chatanywhere_error', '400 BAD_REQUEST']
        },
        {
            'name': 'Mock Quota Error',
            'exception': type('MockException', (), {
                'response': type('MockResponse', (), {
                    'json': lambda: {
                        "error": {
                            "message": "You exceeded your current quota, please check your plan and billing details.",
                            "type": "insufficient_quota",
                            "param": None,
                            "code": "429"
                        }
                    }
                })()
            })(),
            'expected_contains': ['insufficient_quota', '429']
        },
        {
            'name': 'Simple Exception',
            'exception': Exception("Connection timeout"),
            'expected_contains': ['Connection timeout']
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")

        error_msg = generator._parse_api_error(test_case['exception'])
        print(f"  Parsed Error: {error_msg}")

        # Verify expected content
        all_found = all(keyword in error_msg for keyword in test_case['expected_contains'])
        if all_found:
            print(f"  ✅ All expected keywords found\n")
        else:
            print(f"  ❌ Missing keywords: {test_case['expected_contains']}\n")


def test_is_quota_error():
    """Test quota error detection."""
    print("\n" + "="*80)
    print("Test 2: Quota Error Detection")
    print("="*80 + "\n")

    generator = PromptGenerator()

    test_cases = [
        {
            'name': 'Quota Exceeded Error',
            'exception': Exception("You exceeded your current quota"),
            'expected': True
        },
        {
            'name': 'Rate Limit Error',
            'exception': Exception("Rate limit exceeded, please try again later"),
            'expected': True
        },
        {
            'name': '429 Error Code',
            'exception': type('MockException', (), {
                'response': type('MockResponse', (), {
                    'json': lambda: {
                        "error": {
                            "message": "Too many requests",
                            "type": "rate_limit_error",
                            "code": "429"
                        }
                    }
                })()
            })(),
            'expected': True
        },
        {
            'name': 'Normal Error (Not Quota)',
            'exception': Exception("Invalid request format"),
            'expected': False
        },
        {
            'name': 'Network Error (Not Quota)',
            'exception': Exception("Connection refused"),
            'expected': False
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")

        is_quota = generator._is_quota_error(test_case['exception'])
        print(f"  Is Quota Error: {is_quota}")
        print(f"  Expected: {test_case['expected']}")

        if is_quota == test_case['expected']:
            print(f"  ✅ Correct detection\n")
        else:
            print(f"  ❌ Incorrect detection\n")


def test_error_message_display():
    """Test actual error message formatting."""
    print("\n" + "="*80)
    print("Test 3: Error Message Display")
    print("="*80 + "\n")

    # Example error messages from GPT_API_free
    error_messages = [
        {
            'type': 'chatanywhere_error',
            'code': '400 BAD_REQUEST',
            'message': 'Unexpected character encountered【如果您遇到问题，欢迎加入QQ群咨询：1048463714】'
        },
        {
            'type': 'insufficient_quota',
            'code': '429',
            'message': 'You exceeded your current quota, please check your plan and billing details.'
        },
        {
            'type': 'invalid_request_error',
            'code': '400',
            'message': 'Invalid model specified. Please use gpt-3.5-turbo or gpt-4.'
        }
    ]

    for i, error in enumerate(error_messages, 1):
        print(f"Error {i}:")
        formatted = f"[{error['type']}] {error['code']}: {error['message']}"
        print(f"  {formatted}")
        print()


def test_fallback_behavior():
    """Test fallback prompt generation when API fails."""
    print("\n" + "="*80)
    print("Test 4: Fallback Prompt Generation")
    print("="*80 + "\n")

    generator = PromptGenerator()

    theme = "Halloween"
    keywords = ["pumpkin", "costume", "spooky", "candy", "trick-or-treat"]

    print(f"Theme: {theme}")
    print(f"Keywords: {', '.join(keywords)}\n")

    fallback = generator._fallback_prompt(theme, keywords)

    print("Fallback Prompt:")
    print(f"  {fallback}\n")

    # Validate fallback
    checks = [
        ("Contains 'Lulu Pig'", 'lulu pig' in fallback.lower()),
        ("Contains theme", theme.lower() in fallback.lower()),
        ("Contains keywords", any(kw in fallback.lower() for kw in keywords[:3])),
        ("Length > 30 words", len(fallback.split()) > 30)
    ]

    for check_name, result in checks:
        print(f"  {check_name}: {'✅' if result else '❌'}")

    print()


def main():
    """Run all error handling tests."""
    print("\n" + "#"*80)
    print("# PromptGenerator Error Handling Test Suite")
    print("#"*80)

    try:
        test_parse_api_error()
        test_is_quota_error()
        test_error_message_display()
        test_fallback_behavior()

        print("\n" + "="*80)
        print("✅ All Error Handling Tests Completed")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
