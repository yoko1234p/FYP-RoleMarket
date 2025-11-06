"""
Configuration for Streamlit Web Application

管理 API keys、常數和應用配置。

Author: Developer (James)
Date: 2025-11-06
Version: 1.0
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Support both GPT_API_TOKEN and GPT_API_FREE_KEY
GPT_API_TOKEN = os.getenv("GPT_API_TOKEN") or os.getenv("GPT_API_FREE_KEY")

# Validate required API keys
if not GPT_API_TOKEN:
    raise ValueError(
        "GPT_API_TOKEN or GPT_API_FREE_KEY not found in environment. "
        "Please set it in .env file or environment variables."
    )

# Optional API keys (warnings only)
if not GOOGLE_API_KEY:
    import warnings
    warnings.warn(
        "GOOGLE_API_KEY not found. Image generation features will be unavailable."
    )

# Application constants
APP_TITLE = "AI 角色設計與需求預測系統"
APP_VERSION = "1.0"

# Obj 1 - Trend Analysis
DEFAULT_REGION = "HK"
DEFAULT_LANG = "zh-TW"
MAX_KEYWORDS = 10
TRENDS_CACHE_TTL = 3600  # 1 hour in seconds

# Obj 2 - Image Generation
CLIP_SIMILARITY_THRESHOLD = 0.80
REFERENCE_IMAGES_DIR = PROJECT_ROOT / "data" / "reference_images"

# Obj 3 - Forecasting
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "models" / "transformer_lulu" / "best_model.pth"
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_NHEAD = 8

# Streamlit cache settings
CACHE_TTL_TRENDS = 3600  # 1 hour
CACHE_TTL_MODEL = None  # Never expire (resource cache)

# Error messages (Traditional Chinese)
ERROR_MESSAGES = {
    "empty_keywords": "請輸入至少一個關鍵字",
    "api_timeout": "API 請求超時，請稍後重試",
    "api_error": "API 調用失敗：{error}",
    "invalid_input": "輸入格式無效：{details}",
    "model_load_error": "模型載入失敗，請檢查模型檔案",
}

# Success messages
SUCCESS_MESSAGES = {
    "trends_fetched": "✅ 趨勢數據提取成功！",
    "prompt_generated": "✅ Prompt 生成成功！",
    "images_generated": "✅ 圖片生成成功！",
    "prediction_complete": "✅ 銷量預測完成！",
}
