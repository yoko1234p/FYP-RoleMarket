# Source Tree & Module Organization

**Project:** AI-Driven Market-Informed Character IP Design & Demand Forecasting
**Version:** 1.0
**Last Updated:** 2025-11-06
**Status:** Obj 1-3 Complete, Obj 4 Pending

---

## Project Structure Overview

```
FYP-RoleMarket/
├── obj1_nlp_prompt/           # Objective 1: Trend Analysis & Prompt Generation
├── obj2_midjourney_api/       # Objective 2: Image Generation (Google Gemini)
├── obj3_lstm_forecast/        # Objective 3: Sales Forecasting (Transformer)
├── obj4_web_app/              # Objective 4: Streamlit Web Application (⏳ Pending)
├── data/                      # Data Storage (cache, images, datasets)
├── models/                    # Trained Model Weights
├── config/                    # Configuration Files
├── docs/                      # Documentation
├── tests/                     # Test Scripts
├── scripts/                   # Utility Scripts
├── reports/                   # Analysis Reports
├── hf-spaces-deploy/          # Hugging Face Spaces Deployment (Optional)
├── .bmad-core/                # BMAD Framework Files (PM/Dev/Architect agents)
├── requirements.txt           # Python Dependencies
├── docker-compose.yml         # Docker Setup
└── README.md                  # Project Overview
```

---

## Module Breakdown

### 1. Objective 1: Trend Analysis & Prompt Generation (`obj1_nlp_prompt/`)

**Purpose:** 提取 Google Trends 數據，分析文化趨勢，生成 AI 圖片 Prompts

**Directory Structure:**
```
obj1_nlp_prompt/
├── __init__.py
├── enhanced_trends_pipeline.py      # ⭐ 主流程 - 完整 pipeline 入口
├── category_trends_extractor.py     # Google Trends 提取
├── cultural_trend_adapter.py        # 文化趨勢轉化（Meme, Holiday, Design Style）
├── prompt_generator.py              # LLM-based Prompt 生成
├── keyword_extractor.py             # TF-IDF 關鍵字提取
├── keyword_optimizer.py             # 關鍵字過濾優化
├── meme_analyzer.py                 # Meme 趨勢分析
├── seasonal_trends_extractor.py     # 季節趨勢提取
├── demo_category_interactive.py     # Interactive Demo
└── templates/
    └── prompt_template.txt          # Prompt 模板
```

**Key Files:**

| File | Purpose | Status | Dependencies |
|------|---------|--------|--------------|
| `enhanced_trends_pipeline.py` | **主要入口點** - 整合所有模組 | ✅ 完成 | 所有其他模組 |
| `category_trends_extractor.py` | 社交媒體種子詞提取趨勢 | ✅ 完成 | pytrends |
| `cultural_trend_adapter.py` | 將趨勢轉化為角色設計元素 | ✅ 完成 | - |
| `prompt_generator.py` | GPT API 調用生成 Prompt | ✅ 完成 | OpenAI API |

**Integration Points:**
- **Input:** 趨勢關鍵字（如 "春節, 紅色, 喜慶"）
- **Output:** 完整 AI 圖片 Prompt（供 Obj 2 使用）
- **External APIs:** Google Trends (pytrends), GPT_API_free

**Usage Example:**
```python
from obj1_nlp_prompt.enhanced_trends_pipeline import EnhancedTrendsPipeline

pipeline = EnhancedTrendsPipeline(region='HK', lang='zh-TW')
prompt = pipeline.generate_prompt(
    character_name="Lulu Pig",
    character_desc="可愛粉紅豬",
    trend_keywords=["春節", "紅色", "喜慶"]
)
```

---

### 2. Objective 2: Image Generation (`obj2_midjourney_api/`)

**Purpose:** 使用 Google Gemini API 生成角色設計圖，並用 CLIP 驗證一致性

**Directory Structure:**
```
obj2_midjourney_api/
├── __init__.py
├── google_gemini_client.py          # ⭐ Google Gemini API Client
├── gemini_image_client.py           # Alternative Gemini Client
├── character_focused_validator.py   # ⭐ CLIP 相似度驗證
├── clip_validator.py                # CLIP Validator (Deprecated)
├── batch_generate_async.py          # 批量生成（非同步）
├── generate_scene_variations.py     # 場景變化生成
├── analyze_reference_images.py      # 參考圖分析
├── ttapi_client.py                  # ⚠️ Legacy - 原 TTAPI Midjourney Client
├── test_*.py                        # 測試腳本（多個）
└── validators/
    └── ...                          # 驗證器模組
```

**Key Files:**

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `google_gemini_client.py` | **主要生成 Client** | ✅ 完成 | 取代 TTAPI Midjourney |
| `character_focused_validator.py` | **CLIP 驗證** | ✅ 完成 | 相似度 threshold ≥ 0.80 |
| `ttapi_client.py` | TTAPI Midjourney Client | ⚠️ Deprecated | 已棄用，保留供參考 |

**Integration Points:**
- **Input:** Prompt（來自 Obj 1），Reference Image Path
- **Output:** 生成圖片 + CLIP 相似度分數
- **External APIs:** Google Gemini 2.5 Flash Image
- **Models:** CLIP ViT-Large/14 (local inference)

**Usage Example:**
```python
from obj2_midjourney_api.google_gemini_client import GoogleGeminiImageClient
from obj2_midjourney_api.character_focused_validator import CharacterValidator

# 生成圖片
client = GoogleGeminiImageClient()
result = client.generate(
    prompt="Lulu Pig celebrating Christmas",
    reference_image_path="data/reference_images/lulu_pig_ref_1.jpg"
)

# 驗證相似度
validator = CharacterValidator()
similarity = validator.compute_clip_similarity(
    result['image'],
    "data/reference_images/lulu_pig_ref_1.jpg"
)
```

**Technical Debt:**
- ⚠️ 多個測試腳本散落（`test_*.py`），缺少統一測試框架
- ⚠️ `ttapi_client.py` 已棄用但未移除（保留供歷史參考）

---

### 3. Objective 3: Sales Forecasting (`obj3_lstm_forecast/`)

**Purpose:** 基於趨勢數據和 CLIP embeddings 預測銷量

**Directory Structure:**
```
obj3_lstm_forecast/
├── __init__.py
├── hybrid_transformer_model.py      # ⭐ 最終 Transformer 架構
├── kaggle_train_lulu_exp11v2.py     # ⭐ 最終訓練腳本（R² = 0.6788）
├── hybrid_lstm_model.py             # Legacy LSTM 版本（已淘汰）
├── kaggle_train_lulu_*.py           # 實驗腳本（Exp 10-14）
├── generate_lulu_production_data_v*.py  # 數據生成腳本
├── train.py                         # 本地訓練腳本
├── test_local_*.py                  # 本地測試腳本
└── data/
    └── ...                          # 訓練數據（CSV）
```

**Key Files:**

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `hybrid_transformer_model.py` | **最終模型架構** | ✅ 生產 | D_MODEL=64, NUM_LAYERS=2 |
| `kaggle_train_lulu_exp11v2.py` | **最終訓練腳本** | ✅ 生產 | R²=0.6788, MAE=327.26 |
| `hybrid_lstm_model.py` | Legacy LSTM 版本 | ⚠️ Deprecated | 實驗結果較差，已淘汰 |
| `kaggle_train_lulu_exp12*.py` | Ensemble 實驗 | ⚠️ Overfitting | 數據洩漏問題，不採用 |

**Model Architecture (Exp #11v2):**
```python
class HybridTransformer(nn.Module):
    """
    Input:
        - Time-series: (batch, 4, 1) - 過去 4 季度 Google Trends
        - Static: (batch, 772) - CLIP 768-dim + Product Type 4-dim

    Architecture:
        1. Time-series → Embedding → Positional Encoding → Transformer Encoder
        2. Static → FC Layers
        3. Fusion → Output (預測銷量)

    Hyperparameters:
        - D_MODEL = 64
        - NUM_LAYERS = 2
        - NHEAD = 8
        - DROPOUT = 0.1
        - LR = 0.0001
        - EPOCHS = 400 (early stop at 155)
    """
```

**Integration Points:**
- **Input:**
  - Time-series: Google Trends 歷史（4 季度）
  - Static: CLIP embeddings (768-dim) + Season encoding (4-dim)
- **Output:** 預測銷量（數值）
- **Model Weights:** `models/transformer_lulu/best_model.pth`

**Usage Example:**
```python
from obj3_lstm_forecast.hybrid_transformer_model import HybridTransformer
import torch

# 載入模型
model = HybridTransformer(d_model=64, num_layers=2, nhead=8)
model.load_state_dict(torch.load("models/transformer_lulu/best_model.pth"))
model.eval()

# 預測
ts = torch.FloatTensor(trends_history).unsqueeze(0)  # (1, 4, 1)
static = torch.FloatTensor(clip_embedding + season_encoding).unsqueeze(0)  # (1, 772)

with torch.no_grad():
    prediction = model(ts, static)
```

**Technical Debt:**
- ⚠️ 14+ 個實驗腳本（`kaggle_train_lulu_exp*.py`）未清理
- ⚠️ 數據生成腳本版本過多（v1, v2, v2.5, v3）
- ⚠️ 模型權重僅儲存於 local，未上傳至 Hugging Face Hub

---

### 4. Objective 4: Web Application (`obj4_web_app/`) ⏳ Pending

**Purpose:** Streamlit Web UI 整合 Obj 1-3

**Planned Structure:**
```
obj4_web_app/
├── app.py                          # Streamlit 主入口
├── pages/
│   ├── 1_🎨_設計生成.py            # Page 1: Obj 1 + Obj 2
│   └── 2_📊_銷量預測.py            # Page 2: Obj 3
├── utils/
│   ├── __init__.py
│   ├── trends_api.py               # Obj 1 Wrapper
│   ├── design_generator.py         # Obj 2 Wrapper
│   ├── forecast_predictor.py       # Obj 3 Wrapper
│   └── ui_helpers.py               # 共用 UI 函數
├── config.py                       # App 配置
└── README.md                       # 使用說明
```

**Status:** ⏳ 待開發（已規劃於 Epic 4）

---

### 5. Data Directory (`data/`)

**Purpose:** 所有數據儲存（快取、圖片、數據集）

**Structure:**
```
data/
├── cache/                          # API 快取
├── reference_images/               # 參考圖片（角色一致性）
│   ├── lulu_pig_ref_1.jpg
│   ├── lulu_pig_ref_2.jpg
│   └── lulu_pig_ref_3.jpg
├── generated_images/               # 生成的設計圖
├── clip_embeddings/                # CLIP embeddings 快取
├── prompts/                        # 生成的 Prompts
├── prompts_enhanced/               # 優化後的 Prompts
├── trends/                         # Google Trends 數據
├── trends_seasonal/                # 季節趨勢數據
├── production_sales/               # 生產數據集（最終版本）
├── lulu_production_sales*/         # 數據集版本（v1, v2, v2.5, v3, augmented, enhanced）
├── simulated_sales/                # 模擬銷售數據
└── results/                        # 實驗結果
```

**Key Directories:**

| Directory | Purpose | Size | Notes |
|-----------|---------|------|-------|
| `reference_images/` | Reference Images | ~10MB | 3 張 Lulu Pig 參考圖 |
| `generated_images/` | 生成設計圖 | ~100MB+ | 測試階段生成圖片 |
| `clip_embeddings/` | CLIP Embeddings | ~50MB | 768-dim vectors (*.npy) |
| `lulu_production_sales*/` | 訓練數據集 | ~5MB | 多個版本（最終: production_sales） |

**Technical Debt:**
- ⚠️ 數據集版本過多（7+ 個版本），缺少版本管理策略
- ⚠️ 生成圖片未分類整理（建議按 theme/date 分類）

---

### 6. Models Directory (`models/`)

**Purpose:** 訓練好的模型權重

**Structure:**
```
models/
├── transformer_lulu/               # ⭐ 最終生產模型（Exp #11v2）
│   ├── best_model.pth
│   ├── training_results.json
│   └── training_curve.png
├── transformer_production/         # 生產級模型（備份）
├── ensemble_lulu/                  # Ensemble 實驗（未採用）
├── lstm/                           # Legacy LSTM 模型（已淘汰）
├── exp7_deeper_model/              # 實驗 7: 更深模型
├── exp8_hyperparam_search/         # 實驗 8: 超參數搜索
├── test_original/                  # 本地測試模型
└── test_v2_5/                      # 本地測試模型 v2.5
```

**Production Model:**
- **Location:** `models/transformer_lulu/best_model.pth`
- **Architecture:** HybridTransformer (D_MODEL=64, NUM_LAYERS=2)
- **Performance:** R²=0.6788, MAE=327.26
- **Size:** ~10MB

**Technical Debt:**
- ⚠️ 實驗模型未清理（7+ 個目錄）
- ⚠️ 模型權重未版本控制（Git LFS or Hugging Face Hub）

---

### 7. Configuration (`config/`)

**Purpose:** API keys 和配置文件

**Structure:**
```
config/
├── reference_images.py             # Reference Image 路徑配置
└── seasonal_timeframes.json        # 季節時間框架配置
```

**Environment Variables (`.env`):**
```bash
GOOGLE_API_KEY=<Google AI Studio API Key>
GPT_API_TOKEN=<GPT_API_free Token>
```

**Security Note:**
- ⚠️ `.env` 檔案應加入 `.gitignore`（已處理）
- ⚠️ 提供 `.env.example` 供參考

---

### 8. Documentation (`docs/`)

**Purpose:** 專案文檔

**Structure:**
```
docs/
├── prd.md                          # Product Requirements Document
├── implementation-roadmap.md       # 實施路線圖
├── experiment-log-lulu-transformer.md  # Obj 3 實驗記錄
├── phase-a-completion-report.md    # Phase A 完成報告
├── strategy-improvements-v1.2.md   # 策略改進記錄
├── epic-4-web-integration.md       # Obj 4 Epic
├── architecture/                   # 架構文檔（本目錄）
│   ├── tech-stack.md
│   ├── source-tree.md
│   └── coding-standards.md
└── stories/                        # User Stories
    ├── story-4.1-*.md
    ├── story-4.2-*.md
    └── story-4.3-*.md
```

---

### 9. Testing (`tests/`)

**Purpose:** 測試腳本

**Structure:**
```
tests/
├── test_character_focused_validation.py
├── test_complete_e2e_detailed.py
├── test_full_pipeline.py
└── ...
```

**Status:** ⚠️ 測試覆蓋率低（< 20%），待改進

---

### 10. Scripts (`scripts/`)

**Purpose:** 工具腳本

**Status:** ⚠️ 目前為空，待補充部署/清理腳本

---

## Key Integration Points

### Cross-Module Data Flow

```
[User Input: 趨勢關鍵字]
        ↓
[Obj 1: enhanced_trends_pipeline.py]
        ↓ (Prompt)
[Obj 2: google_gemini_client.py]
        ↓ (Generated Image)
[Obj 2: character_focused_validator.py]
        ↓ (CLIP Embedding)
[Obj 3: hybrid_transformer_model.py]
        ↓ (Sales Prediction)
[Output: 預測銷量]
```

### File Dependencies

**Obj 1 → Obj 2:**
- Output: Prompt (string)
- Format: 詳細設計描述（150-200 words）

**Obj 2 → Obj 3:**
- Output: CLIP Embedding (768-dim numpy array)
- Format: `.npy` file or in-memory array

**Obj 3 → Web UI:**
- Output: Prediction (float)
- Format: JSON or dict `{'predicted_sales': float, 'confidence': float}`

---

## Navigation Guide for AI Agents

### Quick Reference: "Where do I find...?"

| Task | Location |
|------|----------|
| **趨勢分析入口** | `obj1_nlp_prompt/enhanced_trends_pipeline.py` |
| **圖片生成入口** | `obj2_midjourney_api/google_gemini_client.py` |
| **CLIP 驗證** | `obj2_midjourney_api/character_focused_validator.py` |
| **預測模型** | `obj3_lstm_forecast/hybrid_transformer_model.py` |
| **訓練腳本（最終）** | `obj3_lstm_forecast/kaggle_train_lulu_exp11v2.py` |
| **模型權重** | `models/transformer_lulu/best_model.pth` |
| **Reference Images** | `data/reference_images/lulu_pig_ref_*.jpg` |
| **API Keys 配置** | `.env` (root directory) |
| **文檔** | `docs/` |
| **PRD** | `docs/prd.md` |

---

## Cleanup Recommendations

### High Priority
1. ⚠️ 清理 Obj 3 實驗腳本（保留 Exp #11v2 + 1-2 個關鍵實驗）
2. ⚠️ 整理數據集版本（保留 production_sales + 1 個備份）
3. ⚠️ 移除或封存 `obj2_midjourney_api/ttapi_client.py`

### Medium Priority
4. ⚠️ 統一測試腳本至 `tests/` 目錄
5. ⚠️ 建立 `scripts/` 工具腳本（部署、清理、數據生成）

### Low Priority
6. ⚠️ 上傳模型權重至 Hugging Face Hub
7. ⚠️ 使用 Git LFS 管理大型檔案

---

**Document Owner:** Architect (Winston)
**Last Review:** 2025-11-06
**Next Review:** After Obj 4 completion
