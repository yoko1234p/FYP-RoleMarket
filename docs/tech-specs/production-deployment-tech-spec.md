# Technical Specification: Production Deployment for Streamlit Web Application

**Project:** FYP-RoleMarket - AI-Driven Character IP Design & Demand Forecasting System
**Feature:** Deploy Streamlit web application to production with full Obj 1-3 integration
**Level:** 1 (Deployment + Configuration + Bug Fixes)
**Field Type:** Brownfield (Modifying existing codebase)
**Author:** PM Agent (John)
**Date:** 2025-11-07
**Status:** Draft

---

## 📋 Context

### Current State

**Completed Work:**
- ✅ **Objective 1** (NLP Prompt Generation): Fully functional standalone
- ✅ **Objective 2** (Image Generation): Functional with Google Gemini 2.5 Flash Image API
- ✅ **Objective 3** (Sales Forecasting): Hybrid Transformer model trained (R² = 0.6788)
- ✅ **Objective 4** (Web Integration): Streamlit app completed (Stories 4.1-4.3)
- ✅ **Manual Testing**: Comprehensive testing completed on 2025-11-07

**Testing Results Summary:**
- ✅ Character information input works
- ✅ Manual keyword input works
- ✅ Prompt generation works (Issue #2 fixed during testing)
- ⚠️ Google Trends auto-extraction fails (Issue #1 - Medium priority)
- ❌ Image generation blocked by API configuration (Issue #3 - High priority)
- ⏸️ Sales forecasting UI untested (depends on image generation)

**Discovered Issues:**

1. **Issue #1: Google Trends Auto-Extraction Failure**
   - **Severity:** Medium
   - **Impact:** Auto-extraction returns "⚠️ 未找到相關趨勢數據"
   - **Workaround:** Manual input works perfectly
   - **Status:** Requires investigation

2. **Issue #2: Prompt Generation Parameter Mismatch**
   - **Severity:** High (was blocking)
   - **Status:** ✅ FIXED in `obj4_web_app/utils/trends_api.py:115-119`

3. **Issue #3: Gemini API Regional Restriction** ⭐ **NEW DISCOVERY**
   - **Root Cause:** Gemini 2.5 Flash Image API not available in Hong Kong region
   - **Impact:** Image generation feature unavailable for HK users
   - **Requires:** VPN or alternative API
   - **Status:** Documented, needs UI warning and future API replacement plan

**Existing Infrastructure:**
- Dockerfile present (`/Dockerfile`)
- HF Spaces deployment documentation prepared (`/hf-spaces-deploy/README.md`)
- Model artifacts ready (`/models/transformer_lulu/best_transformer_model.pth`)
- Reference images available (`/data/reference_images/`)

---

## 🎯 The Change

### What We're Building

Deploy the complete AI Character IP Design & Demand Forecasting system to production with two deployment targets:

1. **Streamlit Cloud**: Full web application (Obj 1-3 integration)
2. **Hugging Face Spaces**: Sales forecasting model hosting

### Key Requirements

**Must-Have (Blocking Deployment):**
1. ✅ Configure Streamlit Secrets for API keys (GOOGLE_API_KEY, GPT_API_FREE_KEY)
2. ✅ Add regional restriction detection for Gemini API
3. ✅ Display user-friendly warning for HK users about image generation
4. ✅ Document API replacement strategy for future (non-Gemini alternatives)
5. ✅ Debug and fix Google Trends auto-extraction (Issue #1)
6. ✅ Deploy Transformer model to Hugging Face Spaces
7. ✅ Complete end-to-end testing with real API credentials

**Nice-to-Have (Post-Launch):**
- Add retry logic with exponential backoff for Google Trends
- Implement alternative image generation API (e.g., Replicate, RunPod)
- Add caching for Google Trends results
- Optimize CLIP model loading time

---

## 🛠️ Implementation Details

### Part 1: Fix Google Trends Auto-Extraction (Issue #1)

**File to Modify:** `obj4_web_app/utils/trends_extractor_wrapper.py`

**Root Cause Analysis:**
Based on testing, the issue is likely:
- Google Trends API rate limiting
- Region/timeframe configuration mismatch
- pytrends library connection issues

**Implementation Steps:**

1. **Add Debug Logging:**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def extract_keywords(self, theme: str, region: str = 'HK') -> Dict[str, List[str]]:
       logger.info(f"Extracting keywords for theme={theme}, region={region}")
       try:
           # Existing code
           logger.debug(f"pytrends query: {query_term}")
           logger.debug(f"Response: {trends_data}")
       except Exception as e:
           logger.error(f"Google Trends extraction failed: {str(e)}", exc_info=True)
   ```

2. **Add Retry Logic with Exponential Backoff:**
   ```python
   import time
   from functools import wraps

   def retry_with_backoff(max_retries=3, base_delay=2):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if attempt < max_retries - 1:
                           delay = base_delay ** attempt
                           logger.warning(f"Attempt {attempt+1} failed, retrying in {delay}s: {str(e)}")
                           time.sleep(delay)
                       else:
                           raise
               return None
           return wrapper
       return decorator

   @retry_with_backoff(max_retries=3, base_delay=2)
   def _fetch_trends_data(self, query_term: str, timeframe: str):
       # Existing pytrends API call
       pass
   ```

3. **Improve Error Messages:**
   ```python
   if not trending_keywords:
       error_msg = f"⚠️ 未找到相關趨勢數據。\n\n可能原因：\n"
       error_msg += "1. Google Trends API 限流（請稍後重試）\n"
       error_msg += "2. 主題關鍵字未找到相關數據\n"
       error_msg += "3. 網絡連接問題\n\n"
       error_msg += "💡 建議：請使用「✍️ 手動輸入」標籤頁手動輸入關鍵字。"
       return {"error": error_msg}
   ```

4. **Add Regional Configuration:**
   ```python
   REGION_CONFIGS = {
       'HK': {'geo': 'HK', 'hl': 'zh-TW', 'tz': 480},
       'TW': {'geo': 'TW', 'hl': 'zh-TW', 'tz': 480},
       'US': {'geo': 'US', 'hl': 'en-US', 'tz': 360},
   }

   def __init__(self, region: str = 'HK'):
       config = REGION_CONFIGS.get(region, REGION_CONFIGS['HK'])
       self.pytrends = TrendReq(hl=config['hl'], tz=config['tz'], geo=config['geo'])
   ```

**Testing Strategy:**
- Test with multiple themes (🎊 新年, 🎉 節日/慶典, 🎃 Halloween)
- Test with different regions (HK, TW, US)
- Test retry logic by simulating API failures
- Verify error messages are user-friendly

---

### Part 2: Add Gemini API Regional Restriction Handling

**Background:**
User discovered that Gemini 2.5 Flash Image API is not available in Hong Kong region. VPN is required, or alternative API needed.

**Files to Modify:**
1. `obj4_web_app/utils/design_generator.py`
2. `obj4_web_app/pages/1_🎨_設計生成.py`
3. `obj4_web_app/config.py`

**Implementation Steps:**

1. **Add Region Detection in `config.py`:**
   ```python
   import requests

   def detect_user_region() -> str:
       """Detect user's region using IP geolocation."""
       try:
           response = requests.get('https://ipapi.co/json/', timeout=3)
           if response.ok:
               data = response.json()
               return data.get('country_code', 'UNKNOWN')
       except:
           pass
       return 'UNKNOWN'

   # Gemini API Regional Restrictions
   GEMINI_RESTRICTED_REGIONS = ['HK', 'CN']
   USER_REGION = detect_user_region()
   IS_GEMINI_RESTRICTED = USER_REGION in GEMINI_RESTRICTED_REGIONS
   ```

2. **Add Warning UI in Design Generation Page:**
   ```python
   # In obj4_web_app/pages/1_🎨_設計生成.py

   from obj4_web_app.config import IS_GEMINI_RESTRICTED, USER_REGION

   # After character input section, before image generation
   if IS_GEMINI_RESTRICTED:
       st.warning(f"""
       ⚠️ **地區限制通知**

       偵測到您的地區為 **{USER_REGION}**，Google Gemini 2.5 Flash Image API
       目前不支援此地區。

       **解決方案：**
       1. 🌐 使用 VPN 連接到支援地區（如：美國、台灣）
       2. 📝 先生成 Prompt，稍後使用其他工具生成圖片
       3. ⏳ 等待我們整合替代 API（計劃中）

       **替代 API 選項（未來實作）：**
       - ⭐ **Hugging Face FLUX** (推薦！免費 tier + 冇地區限制)
       - Midjourney API (TTAPI) (已有 API key)
       - Replicate (Stable Diffusion, Flux)
       - RunPod (自託管模型)

       詳情請參閱：`docs/api-alternatives.md`
       """)

       # Add expandable section for technical details
       with st.expander("🔧 技術細節與未來計劃"):
           st.markdown("""
           ### 當前狀態
           - Google Gemini API 在香港/中國大陸不可用
           - 需要 VPN 才能使用

           ### 替代方案評估（更新於 2025-11-07）

           | API | 免費 Tier | 成本 | 速度 | 品質 | 地區限制 | 狀態 |
           |-----|-----------|------|------|------|---------|------|
           | **HF FLUX.1-dev** ⭐ | ✅ 每月 credits | ~$0.0012/圖 | ~10s | 高 | ✅ 冇 | 🚀 **推薦** |
           | HF Stable Diffusion XL | ✅ 每月 credits | ~$0.0006/圖 | ~5s | 中高 | ✅ 冇 | 📋 備選 |
           | Midjourney (TTAPI) | ❌ | ~$0.02/圖 | ~15s | 極高 | ✅ 冇 | ✅ 已有 API Key |
           | Replicate (Flux) | ❌ | ~$0.003/圖 | ~5s | 高 | ✅ 冇 | 📋 計劃中 |
           | RunPod (SDXL) | ❌ | ~$0.0005/圖 | ~3s | 中 | ✅ 冇 | 📋 長期 |

           ### 建議優先順序（已更新）
           1. **短期（1週內）**: 整合 **Hugging Face FLUX.1-dev**（免費 tier + 冇地區限制）⭐
           2. **備選方案**: Midjourney TTAPI（已有 API key，品質最高）
           3. **中期（2-4週）**: 測試 Replicate Flux（穩定商業方案）
           4. **長期優化**: 自託管 RunPod（最低成本，需要維護）
           """)
   ```

3. **Update Design Generator to Handle Restrictions:**
   ```python
   # In obj4_web_app/utils/design_generator.py

   from obj4_web_app.config import IS_GEMINI_RESTRICTED, GOOGLE_API_KEY

   class DesignGenerator:
       def __init__(self):
           self.api_available = GOOGLE_API_KEY is not None and not IS_GEMINI_RESTRICTED

       def generate_images(self, prompt: str, ...):
           if not self.api_available:
               if IS_GEMINI_RESTRICTED:
                   raise RegionalRestrictionError(
                       "Gemini API 在您的地區不可用。請使用 VPN 或等待替代 API 整合。"
                   )
               else:
                   raise APIKeyMissingError("GOOGLE_API_KEY 未配置")
   ```

4. **Create API Alternatives Documentation:**
   Create new file: `docs/api-alternatives.md`
   ```markdown
   # Image Generation API Alternatives

   ## Context
   Google Gemini 2.5 Flash Image API has regional restrictions (unavailable in HK/CN).
   This document tracks alternative APIs for future implementation.

   ## Recommended Alternatives

   ### 1. Midjourney via TTAPI (Immediate Solution)
   - **Status**: ✅ API Key already available
   - **Cost**: ~$0.02/image
   - **Speed**: ~15s/image
   - **Quality**: Excellent (industry-leading)
   - **Implementation Effort**: Low (already used in Obj 2)
   - **Existing Code**: `obj2_midjourney_api/ttapi_client.py`

   ### 2. Replicate (Flux Model)
   - **Status**: 📋 Planned
   - **Cost**: ~$0.003/image
   - **Speed**: ~5s/image
   - **Quality**: High
   - **API Docs**: https://replicate.com/docs

   ### 3. RunPod (Self-hosted SDXL)
   - **Status**: 📋 Long-term consideration
   - **Cost**: ~$0.0005/image (cheapest)
   - **Speed**: ~3s/image
   - **Quality**: Medium-High (customizable)
   - **Complexity**: Requires infrastructure management

   ## Implementation Priority
   1. Week 1: Integrate Midjourney TTAPI
   2. Week 2-3: Test Replicate Flux
   3. Month 2+: Evaluate RunPod for cost optimization
   ```

**Testing:**
- Test region detection from different IPs
- Verify warning displays correctly for HK region
- Test VPN workaround
- Ensure graceful degradation when API unavailable

---

### Part 3: Streamlit Cloud Deployment Configuration

**Files to Create/Modify:**

1. **Create `.streamlit/config.toml` (if not exists):**
   ```toml
   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   font = "sans serif"

   [server]
   headless = true
   enableCORS = false
   enableXsrfProtection = true
   maxUploadSize = 200

   [browser]
   gatherUsageStats = false
   ```

2. **Create `requirements-streamlit.txt` (Optimized for Streamlit Cloud):**
   ```txt
   # Core AI/ML Libraries (Streamlit Cloud optimized)
   torch>=2.0.0,<2.2.0
   transformers>=4.30.0,<4.40.0

   # NLP & Trend Analysis
   pytrends>=4.9.0
   jieba>=0.42.1
   scikit-learn>=1.3.0

   # LLM Integration
   openai>=1.0.0

   # Data Processing
   pandas>=2.0.0,<2.2.0
   numpy>=1.24.0,<1.27.0
   Pillow>=10.0.0

   # Web Application
   streamlit>=1.28.0
   plotly>=5.17.0
   matplotlib>=3.7.0

   # Utilities
   python-dotenv>=1.0.0
   requests>=2.31.0
   tqdm>=4.66.0

   # Google Generative AI
   google-generativeai>=0.3.0

   # Geolocation
   requests>=2.31.0
   ```

3. **Update `.gitignore` to Exclude Secrets:**
   ```
   # Environment variables
   .env
   .env.local

   # Streamlit secrets
   .streamlit/secrets.toml

   # API keys
   **/secrets.json
   **/*_api_key.txt
   ```

4. **Create Streamlit Secrets Template (`docs/streamlit-secrets-template.toml`):**
   ```toml
   # Streamlit Secrets Configuration Template
   # Copy this to Streamlit Cloud Dashboard → App Settings → Secrets

   # GPT_API_free (Llama 3.1)
   GPT_API_FREE_KEY = "sk-YOUR-KEY-HERE"
   GPT_API_FREE_BASE_URL = "https://api.chatanywhere.org/v1"
   GPT_API_FREE_MODEL = "gpt-3.5-turbo"

   # Google Gemini API (Optional - for image generation)
   # Note: Not available in HK region without VPN
   GOOGLE_API_KEY = "YOUR-GOOGLE-API-KEY-HERE"

   # TTAPI Midjourney API (Backup for Gemini)
   TTAPI_API_KEY = "c14155db-6ea4-74cc-dffa-fb55416a8fa0"

   # Google Trends Configuration
   TRENDS_REGION = "HK"
   TRENDS_LANGUAGE = "zh-TW"

   # CLIP Thresholds
   CLIP_THRESHOLD_CORE = 0.75
   CLIP_THRESHOLD_STYLE = 0.60
   ```

5. **Update `config.py` to Support Streamlit Secrets:**
   ```python
   import os
   from pathlib import Path
   from dotenv import load_dotenv
   import streamlit as st

   # Load environment variables from .env file (local dev)
   load_dotenv()

   def get_secret(key: str, default=None):
       """Get secret from Streamlit secrets or environment variable."""
       # Try Streamlit secrets first (production)
       if hasattr(st, 'secrets') and key in st.secrets:
           return st.secrets[key]
       # Fallback to environment variable (local dev)
       return os.getenv(key, default)

   # API Keys (supports both .env and Streamlit Secrets)
   GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
   GPT_API_TOKEN = get_secret("GPT_API_TOKEN") or get_secret("GPT_API_FREE_KEY")
   TTAPI_API_KEY = get_secret("TTAPI_API_KEY")

   # Validate required API keys
   if not GPT_API_TOKEN:
       raise ValueError(
           "GPT_API_TOKEN or GPT_API_FREE_KEY not found. "
           "Please configure in .env file or Streamlit Secrets."
       )
   ```

---

### Part 4: Hugging Face Spaces Model Deployment

**Goal:** Deploy Transformer model for sales forecasting to HF Spaces.

**Files to Create in `hf-spaces-deploy/`:**

1. **`app.py` (HF Spaces Entry Point):**
   ```python
   """
   Hugging Face Spaces - Sales Forecasting Model Demo

   This Space hosts the Hybrid Transformer model for demand forecasting.
   """

   import streamlit as st
   import torch
   import numpy as np
   from pathlib import Path
   from huggingface_hub import hf_hub_download

   # Import model architecture
   import sys
   sys.path.append('.')
   from obj3_lstm_forecast.hybrid_transformer_model import HybridTransformer

   st.set_page_config(page_title="RoleMarket Sales Forecasting", page_icon="📊")

   @st.cache_resource
   def load_model():
       """Load model from HF Hub (cached)."""
       try:
           model_path = hf_hub_download(
               repo_id="your-username/rolemarket-transformer",
               filename="best_transformer_model.pth"
           )

           model = HybridTransformer(
               d_model=64,
               num_layers=2,
               nhead=8,
               dim_feedforward=256,
               dropout=0.1
           )

           model.load_state_dict(torch.load(model_path, map_location='cpu'))
           model.eval()

           return model
       except Exception as e:
           st.error(f"模型載入失敗: {str(e)}")
           return None

   st.title("📊 RoleMarket Sales Forecasting Model")
   st.markdown("---")

   model = load_model()

   if model:
       st.success("✅ Model loaded successfully!")
       st.info("""
       **Model Specs:**
       - Architecture: Hybrid Transformer
       - R² Score: 0.6788
       - MAE: 327.26 units
       - Training: 1,075 samples (Lulu Pig character)
       """)

       # Add prediction interface
       st.subheader("Sales Prediction")

       col1, col2 = st.columns(2)
       with col1:
           season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
           clip_score = st.slider("CLIP Similarity Score", 0.0, 1.0, 0.80)

       with col2:
           trend_keyword = st.text_input("Trend Keyword", "春節")
           historical_sales = st.number_input("Historical Sales (4 weeks avg)", 0, 5000, 1500)

       if st.button("Predict Sales"):
           # Placeholder prediction logic
           st.success(f"Predicted Sales: ~{int(historical_sales * 1.15)} units")
   ```

2. **`requirements.txt` for HF Spaces:**
   ```txt
   streamlit==1.31.0
   torch==2.1.0
   numpy==1.24.3
   pandas==2.0.3
   scikit-learn==1.3.0
   huggingface_hub==0.20.0
   transformers==4.36.0
   ```

3. **Upload Model to HF Hub:**
   Create script: `scripts/upload_model_to_hf.py`
   ```python
   """
   Upload trained Transformer model to Hugging Face Hub.
   """

   from huggingface_hub import HfApi, create_repo
   from pathlib import Path
   import os

   def upload_model():
       api = HfApi()

       # Create repository
       repo_id = "your-username/rolemarket-transformer"
       try:
           create_repo(repo_id, repo_type="model", exist_ok=True)
           print(f"✅ Repository created: {repo_id}")
       except Exception as e:
           print(f"Repository may already exist: {e}")

       # Upload model file
       model_path = Path("models/transformer_lulu/best_transformer_model.pth")
       if model_path.exists():
           api.upload_file(
               path_or_fileobj=str(model_path),
               path_in_repo="best_transformer_model.pth",
               repo_id=repo_id,
               repo_type="model"
           )
           print(f"✅ Model uploaded: {model_path}")
       else:
           print(f"❌ Model file not found: {model_path}")

       # Upload model card
       readme_content = """
       ---
       license: apache-2.0
       tags:
       - transformer
       - time-series
       - demand-forecasting
       - character-ip
       ---

       # RoleMarket Demand Forecasting Model

       Hybrid Transformer model for character IP sales prediction.

       ## Model Details
       - **Architecture**: Hybrid Transformer (D_MODEL=64, NUM_LAYERS=2)
       - **R² Score**: 0.6788
       - **MAE**: 327.26 units
       - **RMSE**: 456.40 units
       - **Dataset**: Lulu Pig (1,075 samples)

       ## Usage
       ```python
       from huggingface_hub import hf_hub_download
       import torch

       model_path = hf_hub_download(
           repo_id="your-username/rolemarket-transformer",
           filename="best_transformer_model.pth"
       )

       model = HybridTransformer(d_model=64, num_layers=2, nhead=8)
       model.load_state_dict(torch.load(model_path, map_location='cpu'))
       model.eval()
       ```

       ## Training
       - Experiment: Exp #11v2
       - Framework: PyTorch 2.0+
       - Training Environment: Kaggle GPU T4

       ## Citation
       ```bibtex
       @misc{rolemarket2025,
         title={AI-Driven Market-Informed Character IP Design & Demand Forecasting},
         author={FYP Project Team},
         year={2025}
       }
       ```
       """

       api.upload_file(
           path_or_fileobj=readme_content.encode(),
           path_in_repo="README.md",
           repo_id=repo_id,
           repo_type="model"
       )
       print("✅ Model card uploaded")

   if __name__ == "__main__":
       # Ensure HF_TOKEN is set
       if not os.getenv("HF_TOKEN"):
           print("❌ HF_TOKEN not found. Run: huggingface-cli login")
       else:
           upload_model()
   ```

---

### Part 5: End-to-End Testing Plan

**Test Scenarios (Based on Manual Testing Report):**

1. **Scenario A: Spring Festival Theme (Retry)**
   - ✅ Character input
   - ✅ Manual keyword input
   - ✅ Prompt generation
   - 🔄 Google Trends auto-extraction (retest after fix)
   - 🔄 Image generation (with regional warning)
   - 🔄 Sales forecasting (full flow)

2. **Scenario B: Halloween Theme (New)**
   - All steps from Scenario A
   - Test with different seasonal data

3. **Scenario C: Christmas Theme (New)**
   - All steps from Scenario A
   - Test with winter season

**Test Checklist:**

**Pre-Deployment Tests (Local):**
- [ ] Google Trends auto-extraction works for all 3 themes
- [ ] Regional restriction warning displays correctly
- [ ] Streamlit Secrets loading works (test with `st.secrets`)
- [ ] Image generation works with VPN
- [ ] Sales forecasting model loads correctly
- [ ] All pages render without errors
- [ ] Error messages are user-friendly

**Post-Deployment Tests (Streamlit Cloud):**
- [ ] App starts successfully
- [ ] Secrets configured correctly in dashboard
- [ ] Google Trends API works
- [ ] Regional warning shows for HK users
- [ ] Image generation works (if GOOGLE_API_KEY configured + VPN)
- [ ] Sales forecasting works
- [ ] Model downloads from HF Hub
- [ ] All UI elements render correctly
- [ ] Performance acceptable (< 3s page load)

**HF Spaces Tests:**
- [ ] Model Space deploys successfully
- [ ] Model loads from HF Hub
- [ ] Prediction interface works
- [ ] README displays correctly

---

## 🏗️ Development Context

### Tech Stack

**Frontend:**
- Streamlit 1.28.0+
- Plotly 5.17.0+ (data visualization)
- Matplotlib 3.7.0+ (charts)

**Backend/ML:**
- PyTorch 2.0+ (Transformer model)
- Transformers 4.30.0+ (CLIP model)
- scikit-learn 1.3.0+ (preprocessing)
- pytrends 4.9.0+ (Google Trends API)

**APIs:**
- Google Gemini 2.5 Flash Image (regional restrictions)
- GPT_API_free (GPT-3.5-turbo via chatanywhere.org)
- Google Trends API (unofficial via pytrends)
- TTAPI (Midjourney backup)

**Deployment:**
- Streamlit Cloud (web app)
- Hugging Face Spaces (model hosting)
- Hugging Face Hub (model storage)

### Constraints & Limitations

**API Limitations:**
1. **Google Gemini API**
   - ❌ Not available in HK/CN regions
   - Requires VPN for HK users
   - Cost: Free tier (60 requests/min)

2. **Google Trends API**
   - Rate limiting (unknown exact limits)
   - Unofficial API (pytrends library)
   - Data freshness: ~15 minutes delay

3. **GPT_API_free**
   - Limited usage quota
   - Educational/research purposes only
   - May have downtime

**Performance Constraints:**
- Streamlit Cloud: 1GB RAM (free tier)
- Model size: ~1.5MB (Transformer model)
- CLIP model: ~1.7GB (ViT-Large/14)
- Cold start time: ~10-15s (model loading)

**Regional Constraints:**
- Image generation blocked in HK without VPN
- Google Trends may have different data for different regions

### Existing Code Patterns

**Error Handling Pattern:**
```python
try:
    result = api_call()
except SpecificError as e:
    logger.error(f"Error: {str(e)}", exc_info=True)
    st.error(f"❌ {user_friendly_message}")
    return None
```

**Streamlit Cache Pattern:**
```python
@st.cache_resource
def load_model():
    """Expensive resource loading (model, API clients)."""
    pass

@st.cache_data(ttl=3600)
def fetch_trends(theme: str):
    """Data fetching with TTL."""
    pass
```

**Configuration Pattern:**
```python
# config.py
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY required")
```

---

## 📝 Acceptance Criteria

### Must-Have

1. ✅ **Google Trends Auto-Extraction Fixed**
   - Auto-extraction works for at least 80% of themes
   - Clear error messages when fails
   - Manual input workaround always available

2. ✅ **Regional Restriction Handling**
   - Warning displays for HK users
   - Documentation created for API alternatives
   - Graceful degradation (app still usable)

3. ✅ **Streamlit Cloud Deployment**
   - App deployed and accessible via public URL
   - Secrets configured correctly
   - All pages load without errors
   - End-to-end flow works (at least 2/3 scenarios)

4. ✅ **HF Spaces Model Deployment**
   - Model uploaded to HF Hub
   - HF Space deployed with prediction interface
   - Model loads successfully from Hub

5. ✅ **Testing Complete**
   - All 3 test scenarios executed
   - Test report updated with results
   - Known issues documented

### Nice-to-Have

- Retry logic with exponential backoff for Trends API
- Caching for Trends results
- Alternative API integration (Midjourney TTAPI)
- Performance optimization (< 2s page load)
- Mobile responsive design

---

## 🧪 Testing Strategy

### Unit Tests

**Google Trends Retry Logic:**
```python
def test_retry_with_backoff():
    """Test retry mechanism works correctly."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=1)
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Simulated failure")
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 3
```

**Regional Detection:**
```python
def test_regional_detection():
    """Test region detection works."""
    region = detect_user_region()
    assert region in ['HK', 'TW', 'US', 'UNKNOWN']
```

### Integration Tests

**Full Prompt Generation Flow:**
```python
def test_full_prompt_generation_flow():
    """Test Obj 1 end-to-end."""
    # Setup
    trends_api = TrendsAPIWrapper(region='HK')

    # Act
    prompt = trends_api.generate_prompt(
        character_name="Lulu Pig",
        character_desc="可愛粉紅豬，大眼睛",
        trend_keywords=["春節", "紅色", "喜慶"]
    )

    # Assert
    assert prompt is not None
    assert len(prompt) > 50
    assert "Lulu Pig" in prompt
    assert any(keyword in prompt for keyword in ["春節", "紅色", "喜慶"])
```

### Manual Testing Checklist

See **Part 5: End-to-End Testing Plan** above.

---

## 🚀 Deployment Strategy

### Phase 1: Pre-Deployment (Day 1-2)

**Day 1: Bug Fixes & Configuration**
1. Fix Google Trends auto-extraction (Issue #1)
   - Add retry logic
   - Improve error messages
   - Test with multiple themes
2. Add regional restriction handling
   - Region detection
   - Warning UI
   - Documentation

**Day 2: Testing & Validation**
3. Complete end-to-end testing locally
   - Test all 3 scenarios
   - Verify fixes work
   - Update test report
4. Prepare deployment files
   - `.streamlit/config.toml`
   - `requirements-streamlit.txt`
   - Secrets template

### Phase 2: Streamlit Cloud Deployment (Day 3)

**Steps:**
1. Create Streamlit Cloud account (if needed)
2. Connect GitHub repository
3. Configure deployment settings:
   - App path: `obj4_web_app/app.py`
   - Python version: 3.10
   - Requirements file: `requirements.txt`
4. Add Secrets via Dashboard:
   - Copy from `docs/streamlit-secrets-template.toml`
   - Paste into Secrets section
5. Deploy and monitor build logs
6. Test deployed app
7. Update DNS/domain (optional)

**Rollback Plan:**
- Keep local environment working
- Git commit before deployment
- Can revert deployment via Streamlit dashboard
- Local testing always available

### Phase 3: HF Spaces Deployment (Day 3-4)

**Steps:**
1. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

2. Upload model to HF Hub:
   ```bash
   python scripts/upload_model_to_hf.py
   ```

3. Create HF Space:
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Select "Streamlit" SDK
   - Name: `rolemarket-forecasting`

4. Push code to Space:
   ```bash
   cd hf-spaces-deploy/
   git init
   git add .
   git commit -m "feat: 初始化銷量預測 Demo"
   git remote add space https://huggingface.co/spaces/your-username/rolemarket-forecasting
   git push space main
   ```

5. Wait for build (~5-10 minutes)

6. Test Space functionality

### Phase 4: Post-Deployment (Day 4-5)

**Monitoring:**
1. Check Streamlit Cloud logs for errors
2. Monitor HF Space logs
3. Test from different devices/networks
4. Collect user feedback (if any)

**Documentation:**
1. Update README.md with deployment URLs
2. Update manual testing report with final results
3. Create deployment guide for future reference
4. Document known issues and workarounds

**Known Issues to Monitor:**
- Google Trends API rate limiting
- Gemini API regional blocks
- Model loading time on cold start
- Memory usage on Streamlit Cloud

---

## 📚 Related Documentation

### Existing Docs (Reference)
- [`docs/testing/manual-testing-report.md`](../testing/manual-testing-report.md) - Manual testing results (2025-11-07)
- [`docs/implementation-roadmap.md`](../implementation-roadmap.md) - Overall project roadmap
- [`docs/experiment-log-lulu-transformer.md`](../experiment-log-lulu-transformer.md) - Obj 3 training details
- [`docs/strategy-improvements-v1.2.md`](../strategy-improvements-v1.2.md) - Obj 1&2 improvements
- [`hf-spaces-deploy/README.md`](../../hf-spaces-deploy/README.md) - HF deployment guide
- [`README.md`](../../README.md) - Project overview

### New Docs to Create
- `docs/api-alternatives.md` - Image generation API alternatives (Gemini replacement plan)
- `docs/streamlit-secrets-template.toml` - Secrets configuration template
- `docs/deployment-guide.md` - Step-by-step deployment guide (post-deployment)
- `scripts/upload_model_to_hf.py` - Model upload script

### Files to Modify
- `obj4_web_app/utils/trends_extractor_wrapper.py` - Fix Google Trends issue
- `obj4_web_app/utils/design_generator.py` - Add regional restriction handling
- `obj4_web_app/pages/1_🎨_設計生成.py` - Add warning UI
- `obj4_web_app/config.py` - Add Streamlit Secrets support + region detection
- `.gitignore` - Exclude secrets
- `requirements.txt` - Verify all dependencies

---

## ✅ Definition of Done

This technical specification is complete when:

1. ✅ All implementation details are clear and actionable
2. ✅ Acceptance criteria defined and testable
3. ✅ Testing strategy documented
4. ✅ Deployment plan step-by-step
5. ✅ Known issues and workarounds documented
6. ✅ Rollback plan exists
7. ✅ Related documentation linked

This feature/change is complete when:

1. ✅ All acceptance criteria met
2. ✅ All tests pass (unit + integration + manual)
3. ✅ Deployed to Streamlit Cloud successfully
4. ✅ Model deployed to HF Spaces
5. ✅ End-to-end testing completed (3 scenarios)
6. ✅ Documentation updated
7. ✅ Known issues documented in testing report
8. ✅ No critical bugs blocking usage

---

## 🔄 Next Steps

After this Tech-Spec is approved:

1. **Immediate (Today):**
   - Review and approve this Tech-Spec
   - Create GitHub issues/tasks for implementation
   - Set up Streamlit Cloud account

2. **Day 1-2: Implementation**
   - Fix Google Trends issue
   - Add regional restriction handling
   - Create API alternatives documentation
   - Prepare deployment files

3. **Day 3: Deployment**
   - Deploy to Streamlit Cloud
   - Configure Secrets
   - Upload model to HF Hub
   - Deploy HF Space

4. **Day 4-5: Testing & Documentation**
   - Complete end-to-end testing
   - Update testing report
   - Create deployment guide
   - Monitor for issues

---

**Tech-Spec Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** Draft → Ready for Review
