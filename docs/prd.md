# Product Requirements Document (PRD)

**Project:** AI-Driven Market-Informed Character IP Design Extension and Demand Forecasting System

**Version:** 1.0
**Date:** 2025-01-26
**Author:** Product Manager
**Status:** Draft - In Progress

---

## 1. Goals and Background Context

### Goals - 商業價值導向

- **縮短產品上市時間 70%：** 將季節性設計迭代從 3-4 週壓縮至 3-5 小時
- **降低庫存風險 20-30%：** 通過數據驅動嘅需求預測（LSTM R² > 0.7）
- **提升設計產能 4 倍：** 喺相同時間內生成 28 張設計變體
- **實現即刻商業化部署：** 系統總成本 $10-30（TTAPI Midjourney quota）
- **建立數據驅動決策文化：** 提供量化嘅趨勢分析同銷量預測
- **證明 AI-first 競爭優勢：** 定位 ToyzeroPlus 為行業 AI 創新者
- **創造可複製商業模式：** 建立標準化工作流程（Trends → Design → Forecast）

### Background - 技術架構與商業理由

#### 系統概述

本系統為 **商業級別 AI pipeline**，整合四個核心階段：市場趨勢分析 → 商業級設計生成 → 需求預測 → 統一 Web 介面，專為 character IP 設計公司（如 ToyzeroPlus）提供即刻可部署嘅生產工具。

#### 四階段技術架構

**Stage 1: Trend Intelligence (Objective 1)**
- **數據源：** Google Trends API (pytrends)，針對香港市場（繁體中文）
- **關鍵詞提取：** TF-IDF 演算法，從 3 個季節主題（Halloween, Christmas, Spring）提取高頻趨勢詞
- **Prompt 生成：** GPT_API_free (Llama 3.1) 將關鍵詞轉換為詳細設計 prompts
- **輸出：** 28 個 Midjourney-ready prompts (7 themes × 4 variations)

**Stage 2: Commercial-Grade Design Generation (Objective 2)**
- **核心技術：** TTAPI Midjourney API（PPU pay-per-use mode）
- **Character Consistency：** Midjourney v6+ --cref (character reference) parameter
  - 輸入：1-2 張高質量 Pikachu 參考圖像（公開可用）
  - 機制：Midjourney 內建 character reference 系統自動維持角色一致性
  - 無需訓練：直接使用 Midjourney 商業服務，0 訓練時間
- **質量驗證：** CLIP ViT-Large/14 similarity scoring
  - Core feature threshold: >0.75
  - Style threshold: >0.60
- **成本：** ~$10-30 for 28-40 image generations (TTAPI quota)
- **選擇理由：**
  1. **Industry Validation：** ToyzeroPlus actively uses Midjourney as primary design tool and is satisfied with its production-quality output
  2. **Development Efficiency：** API integration saves 2 days vs LoRA training (Day 4-5 instead of Day 4-7)
  3. **Commercial Viability：** Immediate deployment readiness with industry-standard tooling
  4. **Zero Training Overhead：** No model training, no GPU compute, no weight hosting

**Stage 3: Hybrid LSTM Demand Forecasting (Objective 3)**
- **Architecture：** 2-layer LSTM (hidden_dim=128) + dropout (0.2)
- **Input Features：**
  - Temporal: Google Trends time-series (past 3-4 seasons)
  - Static: CLIP embeddings (768-dim) extracted from generated designs
- **Training Data：** 60 simulated data points using rule-based sales simulation
  - Simulation factors: Trends correlation (30%), CLIP similarity (25%), seasonal patterns (20%), production constraints (15%), random noise (10%)
- **Target Metrics：** R² > 0.7, MAE < 200 units, RMSE < 250 units
- **訓練環境：** Kaggle/Google Colab Free Tier

**Stage 4: Streamlit Web Application (Objective 4)**
- **Frontend：** Streamlit (Python full-stack)
- **Page 1 - Design Generation：**
  - Input: Seasonal keywords (e.g., "春節, 紅色, 喜慶")
  - Process: Trend extraction → Prompt generation → Midjourney API call (cached)
  - Output: 4 mood board variations with CLIP scores
- **Page 2 - Forecast Dashboard：**
  - Input: Select season (Q1/Q2/Q3/Q4)
  - Process: Load CLIP embeddings → LSTM inference
  - Output: Sales predictions + trend visualizations
- **Deployment：** Streamlit Cloud (free tier for demo)

#### Why Midjourney API vs Custom Model Training

| 考慮因素 | Midjourney API (TTAPI) | LoRA Fine-Tuning |
|---------|------------------------|------------------|
| **Image Quality** | Production-grade (ToyzeroPlus validated) | Experimental, requires tuning |
| **Development Time** | 2 days (Day 4-5) | 4 days (Day 4-7) |
| **Character Consistency** | Built-in --cref parameter | Requires 10-15 training images + tuning |
| **Total Cost** | $10-30 (TTAPI quota) | Free but requires GPU compute time |
| **Commercial Readiness** | Industry-standard tool | Academic prototype |
| **Maintenance** | Zero (managed service) | Model versioning, weight hosting |
| **Business Case** | Immediate deployment | Proof-of-concept only |

**Decision：** Midjourney API via TTAPI 因為 ToyzeroPlus 已經喺生產環境使用 Midjourney 並對輸出質量滿意，證明商業可行性。系統設計目標係即刻可部署嘅商業工具，唔係學術研究原型。

#### Timeline & Resources

- **Total Timeline：** 15 天 (17-18 days with buffer)
  - Day 1-3: Objective 1 (NLP Pipeline)
  - Day 4-5: Objective 2 (Midjourney API Integration) ← **Saves 2 days vs LoRA**
  - Day 6-9: Objective 3 (LSTM Forecasting)
  - Day 10-12: Objective 4 (Web Integration)
  - Day 13-15: Testing + Documentation
  - Day 16-18: Buffer + Polish

- **Budget：** $10-30 (TTAPI Midjourney quota only)
- **Compute：** Kaggle/Colab Free Tier (LSTM training only, no image generation compute needed)

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR1: Objective 1 - Trend Intelligence & Prompt Generation

**FR1.1: Google Trends Data Extraction**
- **Description:** 使用 pytrends 庫從 Google Trends 提取季節性關鍵詞數據
- **Input:**
  - 季節主題清單（7 themes: Halloween, Christmas, Spring Festival, Summer, Valentine's Day, Mid-Autumn Festival, New Year）
  - 時間範圍：past 12 months
  - 地區：Hong Kong (繁體中文)
- **Output:** CSV 文件包含 top 20 trending keywords per theme
- **Acceptance Criteria:**
  - 成功提取 7 themes × 20 keywords = 140 raw keywords
  - Data freshness < 24 hours
  - 處理時間 < 30 seconds

**FR1.2: Keyword Extraction (TF-IDF)**
- **Description:** 使用 TF-IDF 從 Google Trends 結果篩選高質量關鍵詞
- **Input:** 140 raw keywords from FR1.1
- **Processing:**
  - TF-IDF vectorization with Chinese tokenization
  - Top-K selection (K=5 per theme)
- **Output:** 35 high-quality keywords (7 themes × 5 keywords)
- **Acceptance Criteria:**
  - Keywords 唔重複
  - 每個 keyword TF-IDF score > 0.3
  - 處理時間 < 10 seconds

**FR1.3: LLM-based Prompt Generation**
- **Description:** 使用 GPT_API_free (Llama 3.1) 將關鍵詞轉換為 Midjourney-ready prompts
- **Input:**
  - Selected keywords from FR1.2
  - Character base description (e.g., "Pikachu, yellow electric mouse, red cheeks, brown stripes on back")
  - Prompt template with style guidance
- **Processing:**
  - Generate 4 prompt variations per theme
  - Each prompt 包含: character description + seasonal elements + style keywords
- **Output:** 28 Midjourney prompts (7 themes × 4 variations)
- **Acceptance Criteria:**
  - Prompt length: 50-150 words
  - 包含 character consistency keywords
  - 100% prompts 通過 human review (no inappropriate content)
  - Generation time < 10 seconds per prompt (total < 5 minutes)

---

#### FR2: Objective 2 - Midjourney API Design Generation

**FR2.1: Reference Image Selection**
- **Description:** 選擇 1-2 張高質量 Pikachu 參考圖像用於 Midjourney --cref parameter
- **Requirements:**
  - High resolution (>1024px shortest side)
  - Clear character features (frontal view preferred)
  - Publicly accessible URL (for --cref parameter)
  - No copyright restrictions
- **Output:**
  - Selected reference image(s) stored in `data/reference_images/`
  - Image URL(s) documented
- **Acceptance Criteria:**
  - Reference image meets quality requirements
  - Successfully tested with TTAPI Midjourney API

**FR2.2: TTAPI Midjourney Integration**
- **Description:** 整合 TTAPI Midjourney API 進行商業級圖像生成
- **API Configuration:**
  - Endpoint: TTAPI Midjourney API (PPU mode)
  - Model: Midjourney v6+
  - Parameters:
    - `--cref <reference_image_url>` for character consistency
    - `--ar 1:1` for square format
    - Mode: fast/turbo (based on cost optimization)
- **Input:**
  - 28 prompts from FR1.3
  - Reference image URL(s) from FR2.1
  - TTAPI API key (from .env)
- **Processing:**
  - Batch generation: 4 images per theme
  - Rate limiting: respect TTAPI quota limits
  - Retry logic: see FR2.4
- **Output:**
  - 28 generated images stored in `data/generated_images/{theme}/{variation}.png`
  - Metadata JSON: prompt, timestamp, TTAPI cost, generation time
- **Acceptance Criteria:**
  - 100% images generated successfully (or fallback to cached)
  - Total cost < $30
  - Average generation time: 60-90 seconds per image
  - Images resolution: 1024×1024 or higher

**FR2.3: Character Consistency Validation (CLIP)**
- **Description:** 使用 CLIP ViT-Large/14 驗證生成圖像嘅角色一致性
- **Input:**
  - Generated images from FR2.2
  - Reference image(s) from FR2.1
  - Character description text
- **Processing:**
  - Extract CLIP embeddings (768-dim) for all images
  - Calculate cosine similarity:
    - Core features: Generated image vs Reference image
    - Style: Generated image vs Character description text
- **Output:**
  - Similarity scores stored in `data/clip_scores.json`
  - CLIP embeddings stored in `data/clip_embeddings/*.npy`
- **Acceptance Criteria:**
  - Core feature similarity > 0.75 for 100% images
  - Style similarity > 0.60 for 100% images
  - Processing time < 5 seconds per image

**FR2.4: Error Handling & Retry Logic**
- **Description:** 處理 TTAPI Midjourney API 可能嘅錯誤情況
- **Error Scenarios:**
  1. **Rate Limit (429):**
     - Wait 60 seconds, retry max 3 times
     - Log warning and continue with cached images if retry fails
  2. **API Downtime (500/503):**
     - Exponential backoff: 5s, 15s, 45s
     - After 3 failures, fallback to cached images
  3. **Quota Exceeded:**
     - Log error with cost summary
     - Use cached images for remaining generations
  4. **Invalid Image URL (--cref error):**
     - Fallback to alternative reference image
     - If all references fail, generate without --cref (log warning)
- **Output:**
  - Error logs in `logs/midjourney_errors.log`
  - Retry statistics in generation metadata
- **Acceptance Criteria:**
  - System remains operational even with API failures
  - Clear error messages for debugging
  - No crashes due to API errors

**FR2.5: Image Caching Strategy**
- **Description:** 實現本地緩存機制避免重複 API calls
- **Cache Structure:**
  ```
  data/cache/
  ├── images/
  │   └── {prompt_hash}.png
  └── metadata/
      └── {prompt_hash}.json
  ```
- **Cache Logic:**
  - Before API call: Check if prompt hash exists in cache
  - If exists: Load from cache, skip API call
  - If not exists: Call API, save to cache
  - Cache invalidation: Manual deletion only (no TTL for FYP demo)
- **Acceptance Criteria:**
  - 100% cache hit for repeated prompts
  - Cache lookup time < 0.1 seconds
  - Reduces total API cost for testing/demo iterations

**FR2.6: Cost Tracking**
- **Description:** 記錄 TTAPI Midjourney quota 使用情況
- **Tracked Metrics:**
  - Total images generated
  - Total TTAPI cost (estimated based on quota pricing)
  - Cost per theme
  - Cache hit rate (saved cost)
- **Output:**
  - Cost report in `reports/ttapi_cost_analysis.md`
  - Real-time cost tracking in Streamlit app (optional)
- **Acceptance Criteria:**
  - Accurate cost estimation (±10% of actual)
  - Total cost < $30 for 28 images

---

#### FR3: Objective 3 - LSTM Demand Forecasting

**FR3.1: Sales Data Simulation**
- **Description:** 生成 60 個模擬歷史銷售數據點用於 LSTM 訓練
- **Simulation Formula:**
  ```
  Sales = (Trend_Score × 0.30) + (CLIP_Similarity × 0.25) +
          (Seasonal_Factor × 0.20) + (Production_Constraint × 0.15) +
          (Random_Noise × 0.10)
  ```
- **Input:**
  - Google Trends scores (past 3-4 seasons)
  - CLIP similarity scores (simulated for historical designs)
  - Seasonal patterns (Q1=1.2, Q2=0.8, Q3=1.1, Q4=1.3)
  - Production limits (max 5000 units per design)
- **Output:**
  - `data/simulated_sales.csv` with 60 rows (5 years × 4 seasons × 3 designs)
  - Columns: season, design_id, trend_score, clip_score, seasonal_factor, sales_volume
- **Acceptance Criteria:**
  - Sales range: 500-4500 units
  - Correlation with trends: R > 0.6
  - No unrealistic outliers (MAD test)

**FR3.2: Hybrid LSTM Architecture**
- **Description:** 設計並實現結合時序與靜態特徵嘅 LSTM 模型
- **Architecture:**
  ```
  Input:
    - Temporal: Trend time-series (sequence_length=4 seasons)
    - Static: CLIP embeddings (768-dim)

  Model:
    - LSTM Layer 1: input_dim=1, hidden_dim=128, dropout=0.2
    - LSTM Layer 2: hidden_dim=128, dropout=0.2
    - Concatenate: LSTM output + CLIP embeddings
    - Dense Layer 1: (128 + 768) → 256, ReLU
    - Dense Layer 2: 256 → 128, ReLU
    - Output Layer: 128 → 1 (sales prediction)
  ```
- **Training Configuration:**
  - Loss: MSE (Mean Squared Error)
  - Optimizer: Adam (lr=0.001)
  - Batch size: 8
  - Epochs: 100 (early stopping patience=15)
  - Train/Val/Test split: 60%/20%/20%
- **Acceptance Criteria:**
  - Model converges (validation loss decreases)
  - Training time < 30 minutes on Kaggle/Colab Free Tier
  - Model weights saved in `models/lstm/best_model.pth`

**FR3.3: CLIP Embeddings Extraction**
- **Description:** 從 Objective 2 生成嘅圖像提取 CLIP embeddings 作為 LSTM 靜態特徵
- **Input:** Generated images from FR2.2
- **Processing:**
  - Load CLIP ViT-Large/14 model
  - Extract 768-dim embeddings for each image
  - Normalize embeddings (L2 norm)
- **Output:** `data/clip_embeddings/{theme}_{variation}.npy`
- **Acceptance Criteria:**
  - 28 embedding files (one per image)
  - Embedding shape: (768,)
  - Extraction time < 5 seconds per image

**FR3.4: Model Training & Validation**
- **Description:** 訓練 LSTM 模型並評估預測準確度
- **Training Process:**
  1. Load simulated data (FR3.1) and CLIP embeddings (FR3.3)
  2. Prepare train/val/test splits
  3. Train model using FR3.2 architecture
  4. Track metrics: MSE, MAE, RMSE, R²
  5. Save best model based on validation R²
- **Validation Metrics:**
  - **R² (Coefficient of Determination):** > 0.7
  - **MAE (Mean Absolute Error):** < 200 units
  - **RMSE (Root Mean Squared Error):** < 250 units
- **Output:**
  - Trained model: `models/lstm/best_model.pth`
  - Training history: `models/lstm/training_history.json`
  - Evaluation report: `reports/lstm_evaluation.md`
- **Acceptance Criteria:**
  - All validation metrics meet thresholds
  - Model generalizes well (test R² within 5% of validation R²)
  - No overfitting (train R² - val R² < 0.15)

**FR3.5: GRU Baseline Comparison**
- **Description:** 訓練 GRU baseline model 用於比較 LSTM 性能（學術要求）
- **GRU Architecture:** Same as FR3.2 but replace LSTM layers with GRU layers
- **Training:** Same configuration as FR3.2
- **Comparison Metrics:**
  - R², MAE, RMSE on test set
  - Training time
  - Model size
- **Output:**
  - GRU model: `models/gru/best_model.pth`
  - Comparison report: `reports/lstm_vs_gru_comparison.md`
- **Acceptance Criteria:**
  - GRU model trained successfully
  - Documented performance comparison (LSTM 應該表現更好或相當)
  - 分析結果包含喺 Final Report

---

#### FR4: Objective 4 - Streamlit Web Application

**FR4.1: Design Generation Page**
- **Description:** Streamlit page 用於輸入關鍵詞並展示生成嘅設計變體
- **UI Components:**
  - **Input Section:**
    - Text area: User-provided keywords (e.g., "春節, 紅色, 喜慶")
    - Dropdown: Select theme (7 options)
    - Button: "Generate Designs"
  - **Output Section:**
    - 4×1 grid display: 4 mood board variations
    - For each image:
      - Display image
      - Show CLIP core similarity score
      - Show CLIP style similarity score
      - Show generation metadata (timestamp, cost)
- **Functionality:**
  - On button click:
    1. Extract keywords using TF-IDF (FR1.2)
    2. Generate prompt using GPT_API_free (FR1.3)
    3. Call TTAPI Midjourney API (FR2.2) or load from cache (FR2.5)
    4. Validate with CLIP (FR2.3)
    5. Display results
  - Progress bar showing pipeline stages
  - Error handling with user-friendly messages
- **Acceptance Criteria:**
  - Page loads in < 3 seconds
  - End-to-end generation completes in < 7 minutes (acceptable for demo)
  - All 4 images displayed correctly
  - CLIP scores visible

**FR4.2: Forecast Dashboard Page**
- **Description:** Streamlit page 用於選擇季節並展示銷量預測
- **UI Components:**
  - **Input Section:**
    - Dropdown: Select season (Q1/Q2/Q3/Q4)
    - Dropdown: Select design variation (28 options from generated images)
    - Button: "Predict Sales"
  - **Output Section:**
    - **Sales Prediction:**
      - Predicted sales volume (units)
      - Confidence interval (±10%)
    - **Trend Visualization:**
      - Line chart: Google Trends over past 4 seasons
      - Bar chart: CLIP similarity scores
    - **Feature Importance:**
      - Breakdown: Trend contribution, CLIP contribution, Seasonal factor
- **Functionality:**
  - On button click:
    1. Load CLIP embeddings for selected design
    2. Load trend data for selected season
    3. Run LSTM inference (FR3.4)
    4. Display prediction + visualizations
  - Real-time updates (no page reload)
- **Acceptance Criteria:**
  - Page loads in < 2 seconds
  - Prediction completes in < 5 seconds
  - Charts render correctly
  - Prediction accuracy matches test R² > 0.7

**FR4.3: End-to-End Integration**
- **Description:** 整合所有 4 個 objectives 成為統一嘅 Streamlit 應用
- **Navigation:**
  - Sidebar: Page selector (Design Generation / Forecast Dashboard)
  - Header: Project title, logo, navigation links
  - Footer: Cost summary, API status
- **Data Flow:**
  - Page 1 (Design) → generates images → saves CLIP embeddings
  - Page 2 (Forecast) → loads CLIP embeddings → runs LSTM
  - Shared cache/data across pages
- **Deployment:**
  - Local development: `streamlit run app.py`
  - Cloud deployment: Streamlit Cloud (free tier)
  - Demo mode: All API calls pre-cached
- **Acceptance Criteria:**
  - Smooth navigation between pages
  - No data loss when switching pages
  - Demo mode works without API keys
  - Successfully deployed to Streamlit Cloud

---

### 2.2 Non-Functional Requirements

#### NFR1: Performance

**NFR1.1: Response Time**
- Trend extraction (FR1.1): < 30 seconds
- Keyword extraction (FR1.2): < 10 seconds
- Prompt generation (FR1.3): < 10 seconds per prompt
- Midjourney image generation (FR2.2): 60-90 seconds per image
- CLIP validation (FR2.3): < 5 seconds per image
- LSTM inference (FR3.4): < 5 seconds
- **Total end-to-end time:** < 7 minutes (acceptable for demo video)

**NFR1.2: Resource Usage**
- Memory: < 8GB RAM (compatible with Kaggle/Colab Free Tier)
- Storage: < 5GB total (images + models + cache)
- TTAPI quota: < $30 total cost

**NFR1.3: Scalability**
- System 可以處理 7 themes (current) → 20+ themes (future) without architecture changes
- Cache strategy 支援 100+ cached images

---

#### NFR2: Testing

**NFR2.1: Integration Testing**
- **Scope:** End-to-end testing of all 4 objectives
- **Test Scenarios:**
  1. **Scenario A: Spring Festival Designs**
     - Input: "春節, 紅色, 喜慶"
     - Expected: 4 Pikachu designs with CNY elements
     - Validation: CLIP core > 0.75, style > 0.60
     - Forecast: Q1 2026 sales prediction
  2. **Scenario B: Halloween Designs**
     - Input: "Halloween, orange, spooky"
     - Expected: 4 Pikachu designs with Halloween elements
     - Forecast: Q4 2025 sales prediction
  3. **Scenario C: Christmas Designs**
     - Input: "Christmas, green, festive"
     - Expected: 4 Pikachu designs with Christmas elements
     - Forecast: Q4 2025 sales prediction
- **Acceptance Criteria:**
  - 100% test scenarios pass
  - No critical errors during execution
  - All outputs match expected format

**NFR2.2: Error Recovery Testing**
- Test TTAPI API failures (FR2.4)
- Test cache recovery (FR2.5)
- Test invalid inputs (empty keywords, wrong season)
- **Acceptance Criteria:** System gracefully handles all error scenarios

**NFR2.3: Unit Testing (Optional - Time Permitting)**
- pytest for core functions:
  - `extract_keywords()` (FR1.2)
  - `generate_prompt()` (FR1.3)
  - `calculate_clip_similarity()` (FR2.3)
  - `simulate_sales()` (FR3.1)
- Coverage target: > 70% (if time permits)

---

#### NFR3: Documentation

**NFR3.1: Contextual Report**
- **Purpose:** 週期性報告記錄開發進度、技術決策、遇到嘅問題
- **Format:** Markdown files in `docs/contextual_reports/`
- **Frequency:**
  - Day 3: Objective 1 completion report
  - Day 5: Objective 2 completion report
  - Day 9: Objective 3 completion report
  - Day 12: Objective 4 completion report
  - Day 15: Integration & testing report
- **Content:**
  - What was completed
  - Technical challenges encountered
  - Solutions implemented
  - Deviation from original plan (if any)
  - Next steps
- **Word Count:** 未知（根據實際進度決定，每個報告約 500-1000 字）

**NFR3.2: Final Report (12,000-15,000 words)**
- **Structure & Word Count Breakdown:**

  **1. Introduction (1,200 words)**
  - Background and motivation (400 words)
  - Problem statement (300 words)
  - Objectives and scope (300 words)
  - Report structure (200 words)

  **2. Literature Review (2,000 words)**
  - Market trend analysis in AI design (500 words)
  - Character IP design automation (500 words)
  - Demand forecasting methods (500 words)
  - Midjourney API vs custom models comparison (500 words)

  **3. Methodology (3,500 words)**
  - **Objective 1: Trend Intelligence (800 words)**
    - Google Trends API usage
    - TF-IDF keyword extraction
    - GPT_API_free prompt generation
  - **Objective 2: Midjourney API Integration (1,200 words)**
    - TTAPI platform overview
    - Character reference (--cref) mechanism
    - CLIP validation methodology
    - Error handling & caching strategies
  - **Objective 3: LSTM Forecasting (1,000 words)**
    - Hybrid LSTM architecture design
    - Sales data simulation
    - CLIP embeddings as static features
    - Training process and hyperparameters
  - **Objective 4: Web Application (500 words)**
    - Streamlit architecture
    - UI/UX design decisions
    - Integration strategy

  **4. Implementation (2,500 words)**
  - System architecture (600 words)
  - Technology stack justification (400 words)
  - Module-by-module implementation details (1,000 words)
  - Challenges and solutions (500 words)

  **5. Experiments and Results (3,000 words)**
  - **Objective 1 Results (500 words)**
    - Keyword extraction quality
    - Prompt generation examples
  - **Objective 2 Results (800 words)**
    - Generated images showcase (4-6 examples)
    - CLIP similarity analysis
    - Cost analysis
    - Character consistency evaluation
  - **Objective 3 Results (1,200 words)**
    - LSTM training history
    - Model performance metrics (R², MAE, RMSE)
    - LSTM vs GRU comparison
    - Feature importance analysis
  - **Objective 4 Results (500 words)**
    - Web app screenshots
    - User flow demonstration
    - Performance metrics

  **6. Discussion (1,500 words)**
  - Commercial viability analysis (500 words)
  - Midjourney API advantages vs limitations (400 words)
  - LSTM forecasting insights (300 words)
  - Future improvements (300 words)

  **7. Conclusion (800 words)**
  - Summary of achievements (300 words)
  - Contributions to industry (200 words)
  - Limitations and future work (300 words)

  **8. References (500 words)**
  - Academic papers (10-15 citations)
  - Technical documentation (TTAPI, Midjourney, CLIP, LSTM)
  - Industry reports

- **Format:** PDF (generated from LaTeX or Markdown)
- **Location:** `docs/final_report.pdf`

**NFR3.3: Demo Video Script**
- **Purpose:** 5-6 分鐘 demo video 腳本用於 FYP 展示
- **Structure:**

  **1. Introduction (0:30)**
  - Project title and objectives
  - Problem statement (ToyzeroPlus pain points)
  - Solution overview (4-stage pipeline)

  **2. System Architecture (0:45)**
  - High-level architecture diagram
  - Technology stack overview
  - Why Midjourney API (commercial focus)

  **3. Live Demo - Design Generation (1:30)**
  - Open Streamlit app (Design Generation page)
  - Input: "春節, 紅色, 喜慶"
  - Show progress bar (Trends → Prompts → Midjourney API)
  - Display 4 generated Pikachu designs
  - Highlight CLIP similarity scores (>0.75 core, >0.60 style)
  - Show cost tracking ($X spent)

  **4. Live Demo - Demand Forecasting (1:30)**
  - Switch to Forecast Dashboard page
  - Select: Q1 2026, Spring Festival design #2
  - Show LSTM prediction (e.g., 2,450 units ±245)
  - Display trend charts and feature importance
  - Explain hybrid LSTM (temporal + CLIP embeddings)

  **5. Results & Impact (1:00)**
  - Key metrics achieved:
    - 28 designs generated in 5 hours (vs 3-4 weeks manual)
    - LSTM R² = 0.78 (>0.7 target)
    - Total cost $24 (within $30 budget)
  - Commercial value:
    - ToyzeroPlus can deploy immediately
    - 70% time reduction
    - Data-driven decision making

  **6. Conclusion (0:45)**
  - Summary of 4 objectives completed
  - Commercial readiness (industry-standard tools)
  - Future enhancements (multi-character, voting integration)
  - Thank you + Q&A invitation

- **Format:** Script in `docs/demo_video_script.md` + recorded video in `docs/demo_video.mp4`
- **Acceptance Criteria:**
  - Script covers all 4 objectives
  - Demo runs smoothly without API calls (use cached data)
  - Video length: 5-6 minutes
  - Professional narration and editing

**NFR3.4: Code Documentation (Excluded)**
- **README.md:** NOT REQUIRED (as per user request)
- **API Documentation:** NOT REQUIRED (as per user request)
- **Inline Comments:** Minimal (only for complex logic)

---

### 2.3 Out of Scope

以下功能明確排除喺 MVP 之外：

- ❌ Scheduled/automated execution (cron jobs, GitHub Actions)
- ❌ User voting/feedback features
- ❌ Multi-character IP support (focus on single character - Pikachu)
- ❌ Real-time trend monitoring (batch processing only)
- ❌ Advanced hyperparameter tuning (use baseline configurations)
- ❌ Production deployment infrastructure (local/cloud demo only)
- ❌ Extensive NLP model comparison (TextRank, RAKE benchmarks)
- ❌ Mobile app / responsive design optimization
- ❌ User authentication / multi-user support
- ❌ Export to production-ready formats (PSD, AI files)
- ❌ Detailed README or API documentation

---

## 3. User Interface Design Goals

### 3.1 Interactive Exploration

**Goal:** 提供互動式參數調整，鼓勵用戶實驗同探索不同設計可能性

**Design Principles:**
- **Parameter Sliders:**
  - CLIP similarity threshold adjustment (0.60-0.90)
  - LSTM confidence interval display (±5% to ±20%)
  - Trend weight visualization (adjust trend importance in forecast)
- **Real-time Preview:**
  - Instant CLIP score updates when hovering over images
  - Dynamic forecast chart updates when changing season selection
  - Live cost estimation as user selects number of variations
- **Experimentation Features:**
  - "Compare Designs" mode: side-by-side view of 2-4 variations
  - "What-if Analysis": adjust seasonal factors and see forecast changes
  - Prompt editing: modify LLM-generated prompts before Midjourney API call

**Implementation Details:**
- Streamlit widgets: `st.slider()`, `st.number_input()`, `st.selectbox()`
- Session state management for preserving user selections
- Debouncing for API calls (prevent excessive requests during slider adjustment)

**Acceptance Criteria:**
- Users can adjust ≥3 interactive parameters
- Real-time updates complete in <1 second for cached data
- No page reload required for parameter changes

---

### 3.2 Dashboard Data-Focused

**Goal:** 強調數據可視化同關鍵指標，支援數據驅動決策

**Key Metrics Display:**
- **Top-Level KPIs (Header):**
  - Total designs generated: 28/28
  - Average CLIP core score: 0.81
  - Total TTAPI cost: $24.50 / $30 budget
  - LSTM model R²: 0.78

- **Design Generation Page Metrics:**
  - Per-image CLIP scores (core + style) with color coding:
    - Green: >0.80 (excellent)
    - Yellow: 0.70-0.80 (good)
    - Red: <0.70 (needs review)
  - Generation time per image
  - Cost breakdown by theme

- **Forecast Dashboard Metrics:**
  - **Primary Metric:** Predicted sales volume (large display, e.g., "2,450 units")
  - **Confidence Interval:** ±245 units (±10%)
  - **Trend Chart:** Line graph showing Google Trends over past 4 seasons
  - **Feature Importance Bar Chart:**
    - Trend contribution: 35%
    - CLIP similarity: 28%
    - Seasonal factor: 22%
    - Production constraint: 15%
  - **Historical Comparison:** Compare predicted vs simulated historical sales

**Chart Types:**
- Line charts: Google Trends time-series (plotly)
- Bar charts: CLIP scores comparison, feature importance (matplotlib/seaborn)
- Scatter plots: Sales vs Trend correlation (optional)
- Gauge charts: Cost usage, CLIP score ranges

**Implementation Details:**
- Streamlit charting: `st.plotly_chart()`, `st.bar_chart()`, `st.line_chart()`
- Custom CSS for KPI cards with color coding
- Responsive grid layout: `st.columns()` for metrics display

**Acceptance Criteria:**
- ≥6 KPIs displayed prominently
- ≥3 interactive charts (zoomable, hoverable)
- Color coding for quick visual assessment (green/yellow/red)
- Charts load in <2 seconds

---

### 3.3 Business Report Style

**Goal:** 專業商業風格設計，適合向 ToyzeroPlus 管理層展示，可直接用於商業報告

**Visual Design:**
- **Color Scheme:**
  - Primary: Corporate blue (#1E3A8A) for headers and key actions
  - Secondary: Grey (#6B7280) for text and borders
  - Accent: Green (#10B981) for positive metrics, Red (#EF4444) for warnings
  - Background: Light grey (#F9FAFB) for page background, White (#FFFFFF) for cards
- **Typography:**
  - Headers: Sans-serif (e.g., Inter, Roboto) bold
  - Body text: Sans-serif regular, 14-16px
  - Metrics: Monospace for numbers (e.g., "2,450 units")
- **Layout:**
  - Card-based design with subtle shadows for sections
  - Consistent spacing (16px/24px grid)
  - Professional margins (wider for readability)

**Branding Elements:**
- **ToyzeroPlus Logo Placeholder:**
  - Header left: Company logo (placeholder or actual if provided)
  - Footer: "Powered by AI-Driven Design Intelligence"
- **Project Branding:**
  - Title: "Market-Informed Character IP Design System"
  - Subtitle: "AI-Powered Trend Analysis & Demand Forecasting"
  - Version: "v1.0 FYP Demo"

**Export-Ready Features:**
- **Screenshot-Friendly Layout:**
  - Clean, printable design (no unnecessary decorations)
  - High-contrast text for readability in reports
  - Chart legends and labels clearly visible
- **Report Generation (Optional - Future):**
  - "Export to PDF" button for saving dashboard state
  - CSV download for metrics and predictions
  - Image gallery export for generated designs

**Professional Elements:**
- Status indicators: "✓ Completed", "⏳ Processing", "⚠ Warning"
- Timestamp display: "Last updated: 2025-01-26 14:30 HKT"
- Data source attribution: "Source: Google Trends (Hong Kong)"
- Confidence intervals displayed with error bars

**Implementation Details:**
- Custom CSS injection: `st.markdown()` with `unsafe_allow_html=True`
- Color variables defined in `styles.css`
- Consistent component styling using Streamlit theming
- Logo image in `assets/logo.png`

**Acceptance Criteria:**
- Professional appearance suitable for business presentation
- Consistent color scheme throughout app
- Screenshots can be directly inserted into PowerPoint/Keynote
- Clear branding with ToyzeroPlus context
- Print-friendly layout (high contrast, readable fonts)

---

### 3.4 UI/UX Constraints

**Technical Constraints:**
- Streamlit limitations: No custom JavaScript (use built-in widgets only)
- Responsive design: Optimize for desktop (1920×1080), tablet support optional
- Browser compatibility: Chrome, Safari, Edge (modern versions)

**Performance Constraints:**
- Page load time: <3 seconds
- Chart rendering: <2 seconds
- Real-time updates: <1 second for cached data

**Accessibility (Basic):**
- Color contrast ratio: ≥4.5:1 (WCAG AA)
- Font size: ≥14px for body text
- Alt text for images (generated designs)

**User Experience Priorities:**
1. **Clarity:** Users understand what each metric means without extensive training
2. **Speed:** Fast interactions encourage exploration
3. **Trust:** Professional design builds confidence in AI predictions
4. **Guidance:** Clear labels and tooltips prevent confusion

---

## 4. Technical Assumptions

### 4.1 External API Dependencies

**Assumption 1: TTAPI Midjourney API Stability**
- **Assumption:** TTAPI Midjourney API remains accessible and stable throughout 15-day development period
- **Impact if False:** Critical blocker for Objective 2
- **Mitigation:**
  - Pre-purchase TTAPI quota on Day 0 to lock in pricing
  - Test API on Day 4 before committing to full batch generation
  - Backup plan: Use DALL-E 3 or Flux API (also available on TTAPI)
  - Cache strategy: Generate and store all 28 images early; web app reads from cache
- **Validation:** Test API call on Day 0-1 during environment setup
- **Risk Level:** MEDIUM

**Assumption 2: TTAPI Quota Pricing Remains Stable**
- **Assumption:** TTAPI Midjourney PPU mode pricing remains at estimated $10-30 range for 28-40 images
- **Impact if False:** Budget overrun; project cost analysis invalid
- **Mitigation:**
  - Lock in pricing by purchasing quota upfront
  - Document exact pricing on Day 0 for Final Report
  - If price increases, reduce image count (e.g., 4 themes × 3 variations = 12 images instead of 28)
- **Validation:** Check TTAPI pricing page and purchase quota on Day 0
- **Risk Level:** LOW

**Assumption 3: GPT_API_free Service Availability**
- **Assumption:** GPT_API_free (free ChatGPT API) remains accessible without rate limits during development
- **Impact if False:** Blocks Objective 1 prompt generation
- **Mitigation:**
  - Backup option 1: Hugging Face Mistral-7B (free inference API)
  - Backup option 2: Manual prompt writing (28 prompts = 2-3 hours work)
  - Pre-generate and cache all 28 prompts on Day 3; web app uses cached prompts
- **Validation:** Test API call on Day 0-1
- **Risk Level:** LOW-MEDIUM

---

### 4.2 Data and Resources

**Assumption 4: Pikachu Reference Images Available**
- **Assumption:** 1-2 high-quality Pikachu reference images are publicly available and copyright-free
- **Impact if False:** Cannot use Midjourney --cref parameter effectively
- **Mitigation:**
  - Search multiple sources: Official Pokémon press kit, DeviantArt, Pinterest, Wikimedia Commons
  - Test multiple reference images to find optimal one
  - If copyright issues arise, use generic "yellow electric mouse" description without --cref
- **Validation:** Identify and download reference images on Day 0
- **Risk Level:** LOW

**Assumption 5: Google Trends Data Validity**
- **Assumption:** Google Trends data in Traditional Chinese (Hong Kong region) accurately represents toy industry target market trends
- **Impact if False:** Trend extraction (Objective 1) produces irrelevant keywords; LSTM predictions less accurate
- **Mitigation:**
  - Validate keywords through ToyzeroPlus informal review (if feasible)
  - Cross-reference with Instagram hashtag trends (manual spot-check)
  - Acknowledge limitation in Final Report if data quality concerns arise
- **Validation:** Manual review of extracted keywords on Day 3
- **Risk Level:** LOW

---

### 4.3 Technical Feasibility

**Assumption 6: Midjourney --cref Character Consistency**
- **Assumption:** Midjourney v6+ character reference (--cref) parameter provides sufficient character consistency (CLIP core >0.75) for FYP demonstration
- **Impact if False:** Core value proposition (brand consistency) fails; generated images don't maintain character identity
- **Mitigation:**
  - Early validation: Generate 4-5 test images on Day 4 to verify --cref quality
  - Test multiple reference images to find optimal one
  - Prompt engineering: Emphasize character features in text prompts ("yellow electric mouse, red cheeks, brown stripes")
  - Lower CLIP threshold to >0.70 if --cref proves less consistent
  - Fallback positioning: Acknowledge as limitation in Final Report; focus on commercial workflow automation value
- **Validation:** Generate test images on Day 4 morning before full batch
- **Risk Level:** MEDIUM
- **Industry Validation:** ToyzeroPlus actively uses Midjourney and is satisfied with output quality, providing real-world evidence of feasibility

**Assumption 7: LSTM Training with 60 Data Points**
- **Assumption:** 60 simulated historical data points provide sufficient signal for LSTM training to achieve R² > 0.7
- **Impact if False:** Model underfits; forecast accuracy too low for demonstration
- **Mitigation:**
  - Academic consensus: 50-100 points is minimum viable for LSTM
  - GRU backup: Simpler architecture may perform better with limited data
  - Extend simulation: Generate 120 points (10 years) if time permits during Day 6-9
  - Adjust success criteria: R² > 0.6 acceptable if dataset limitation acknowledged in Final Report
  - Fallback positioning: Emphasize proof-of-concept value rather than production accuracy
- **Validation:** Early LSTM training experiments on Day 6-7
- **Risk Level:** MEDIUM

**Assumption 8: Rule-Based Sales Simulation Validity**
- **Assumption:** Self-generated simulation using Trends + CLIP + seasonal factors + noise produces realistic sales patterns without real historical data
- **Impact if False:** LSTM model trains on unrealistic data; predictions not credible
- **Mitigation:**
  - Validate simulation parameters with ToyzeroPlus logic (informal discussion)
  - Ensure correlation structure: Sales should correlate with trends (R > 0.6)
  - Include realistic constraints (max 5000 units, seasonal variations)
  - Acknowledge as limitation in Final Report
  - Emphasize that simulation demonstrates methodology; real deployment would use actual sales data
- **Validation:** Visual inspection of simulated sales distribution on Day 6
- **Risk Level:** LOW-MEDIUM
- **Justification:** No access to real ToyzeroPlus sales data; simulation is standard academic approach for proof-of-concept

---

### 4.4 Development Environment

**Assumption 9: Kaggle/Colab Free Tier Sufficiency**
- **Assumption:** Kaggle/Google Colab Free Tier provides sufficient compute (RAM, GPU) for LSTM training
- **Impact if False:** Cannot train LSTM model; Objective 3 blocked
- **Mitigation:**
  - Reduce model complexity: Use 1-layer LSTM instead of 2-layer
  - Reduce batch size: 4 instead of 8
  - Use Kaggle Notebooks (more stable than Colab for long-running jobs)
  - Local training as backup (if laptop has sufficient RAM)
- **Validation:** Test LSTM training on sample data on Day 6
- **Risk Level:** LOW

**Assumption 10: Streamlit Cloud Free Tier Deployment**
- **Assumption:** Streamlit Cloud free tier supports deployment of app with pre-cached data (no live API calls during demo)
- **Impact if False:** Cannot publicly deploy app; demo limited to local execution
- **Mitigation:**
  - Local demo fallback: Run `streamlit run app.py` locally, record screen for demo video
  - Streamlit Community Cloud has generous limits for public repos
  - Cache all data locally in Git repo (images, embeddings, predictions)
- **Validation:** Test deployment on Day 10-11
- **Risk Level:** LOW

---

### 4.5 Project Scope and Acceptance

**Assumption 11: Commercial Focus Acceptable for FYP**
- **Assumption:** FYP evaluation accepts commercial-focused projects demonstrating real business value rather than purely academic research
- **Impact if False:** Project may be criticized for lack of novel research contribution
- **Mitigation:**
  - Hybrid approach: Include academic components (LSTM vs GRU comparison, ablation studies)
  - Emphasize innovation: Hybrid LSTM architecture combining temporal + static features is novel
  - Industry validation: ToyzeroPlus collaboration demonstrates real-world applicability
  - Documentation: Final Report includes Literature Review and academic rigor
- **Validation:** Confirmed by FYP supervisor guidance (from Project Brief)
- **Risk Level:** LOW
- **Confirmed:** User stated FYP supervisor encouraged commercial viability and industry-standard tools

**Assumption 12: Video Demo Format Acceptable**
- **Assumption:** 5-6 minute demo video is acceptable for FYP evaluation; live system not required during presentation
- **Impact if False:** Must prepare for live demo with potential API failures
- **Mitigation:**
  - Prepare both: Pre-recorded video + live demo capability
  - Video advantages: Professional editing, no API dependency, repeatable
  - Live demo backup: Demo mode with all cached data (no API calls)
- **Validation:** Confirmed by FYP supervisor (from Project Brief)
- **Risk Level:** VERY LOW

**Assumption 13: Design Mood Board Quality Threshold**
- **Assumption:** Mood board quality acceptable at 60% designer incorporation rate (vs 100% polished finals)
- **Impact if False:** Generated designs deemed too low quality for FYP demonstration
- **Mitigation:**
  - Midjourney provides production-grade quality (ToyzeroPlus validated)
  - Focus on process automation value, not final design polish
  - Select best 12-16 images (out of 28) for demo showcase
  - Acknowledge as "concept exploration" not "final production artwork"
- **Validation:** ToyzeroPlus informal review of generated samples (if feasible)
- **Risk Level:** LOW
- **Industry Evidence:** ToyzeroPlus actively uses Midjourney for design, proving quality acceptability

---

### 4.6 Timeline and Resource Assumptions

**Assumption 14: 15-Day Timeline Feasibility**
- **Assumption:** All 4 objectives can be completed within 15 days (17-18 days with buffer)
- **Impact if False:** Project incomplete at submission deadline
- **Mitigation:**
  - Strict time-boxing: End-of-day checkpoint; cut scope immediately if behind
  - Pre-defined scope cuts:
    - Drop LSTM vs GRU comparison
    - Reduce themes from 7 to 4
    - Simplify Streamlit UI (remove interactive sliders)
  - Buffer days: Day 16-18 absorb 2-3 day overrun
  - Midjourney API saves 2 days vs LoRA training (validated estimate)
- **Validation:** Daily progress tracking against timeline
- **Risk Level:** MEDIUM
- **Justification:** Midjourney API approach significantly reduces timeline risk vs custom model training

**Assumption 15: Solo Development Feasibility**
- **Assumption:** Single developer (FYP student) can complete all technical components without team support
- **Impact if False:** Cannot complete all 4 objectives; must reduce scope
- **Mitigation:**
  - Modular architecture: Each objective can function independently
  - Reusable code: LLM prompts, API wrappers, data pipelines can be templated
  - Community resources: Stack Overflow, GitHub examples, TTAPI documentation
  - Supervisor support: Technical guidance when blocked
- **Validation:** Ongoing; adjust scope if development velocity too slow
- **Risk Level:** MEDIUM

---

### 4.7 Assumptions Summary Table

| # | Assumption | Validation Method | Risk Level | Mitigation Strategy |
|---|------------|-------------------|------------|---------------------|
| 1 | TTAPI Midjourney API stable | Test on Day 0 | MEDIUM | DALL-E/Flux backup, caching |
| 2 | TTAPI pricing stable (~$10-30) | Lock pricing Day 0 | LOW | Pre-purchase quota |
| 3 | GPT_API_free available | Test on Day 0 | LOW-MED | Mistral-7B backup, manual prompts |
| 4 | Pikachu reference images available | Source on Day 0 | LOW | Multiple sources, generic fallback |
| 5 | Google Trends data valid | Manual review Day 3 | LOW | Cross-reference Instagram |
| 6 | Midjourney --cref effective | Test on Day 4 | MEDIUM | Multiple refs, lower threshold |
| 7 | 60 data points sufficient for LSTM | Early training Day 6-7 | MEDIUM | GRU backup, extend to 120 points |
| 8 | Simulation produces realistic data | Visual inspection Day 6 | LOW-MED | ToyzeroPlus validation |
| 9 | Kaggle/Colab Free Tier sufficient | Test training Day 6 | LOW | Reduce model complexity |
| 10 | Streamlit Cloud deployment works | Test deploy Day 10-11 | LOW | Local demo fallback |
| 11 | Commercial focus acceptable FYP | Supervisor confirmed | LOW | Academic components included |
| 12 | Video demo format acceptable | Supervisor confirmed | VERY LOW | Prepare both video + live |
| 13 | Mood board quality threshold | ToyzeroPlus review | LOW | Showcase best 12-16 images |
| 14 | 15-day timeline feasible | Daily tracking | MEDIUM | Scope cuts, buffer days |
| 15 | Solo development feasible | Ongoing velocity check | MEDIUM | Modular architecture |

---

## 5. Epic List

### Epic Breakdown Strategy

本 PRD 將 4 個 Objectives 對應到 5 個 Epics，確保每個 Epic 可以獨立執行並交付可演示嘅價值。

### Epic Sequencing Rationale

```
Epic 1 (Foundation)
    ↓ provides project structure
Epic 2 (Obj 1: NLP Pipeline)
    ↓ generates prompts
Epic 3 (Obj 2: Midjourney API)
    ↓ generates images + CLIP embeddings
Epic 4 (Obj 3: LSTM Forecasting)
    ↓ provides prediction capability
Epic 5 (Obj 4: Web Integration)
    ↓ delivers end-to-end demo
Epic 6 (FYP Delivery)
    ↓ completes academic requirements
```

---

### Epic 1: Foundation & Environment Setup
**Timeline:** Day 0-1 (2 days)
**Owner:** Developer (solo)
**Dependencies:** None

**Goal:** 建立完整開發環境、驗證所有外部依賴、設置項目結構

**Key Deliverables:**
- Project repository structure (`src/`, `data/`, `models/`, `docs/`)
- Python environment with all dependencies installed
- TTAPI account + quota purchased ($10-30)
- GPT_API_free API key validated
- Pikachu reference images identified and downloaded
- All critical API dependencies tested (TTAPI, GPT_API_free, pytrends)

**Success Criteria:**
- All API keys configured in `.env`
- Test API calls successful (Midjourney, GPT, Google Trends)
- Reference images meet quality requirements (>1024px, clear features)
- Git repository initialized with `.gitignore`

**Stories:** 4-5 stories (environment setup, API registration, reference images, testing)

---

### Epic 2: Objective 1 - Trend Intelligence & Prompt Generation
**Timeline:** Day 2-3 (2 days)
**Owner:** Developer (solo)
**Dependencies:** Epic 1 (environment setup complete)

**Goal:** 實現從 Google Trends 到 Midjourney-ready prompts 嘅自動化 pipeline

**Key Deliverables:**
- `obj1_nlp_prompt/trends_extractor.py`: Google Trends data extraction
- `obj1_nlp_prompt/keyword_extractor.py`: TF-IDF keyword filtering
- `obj1_nlp_prompt/prompt_generator.py`: GPT_API_free prompt generation
- 28 generated prompts stored in `data/prompts/`
- Contextual Report: Objective 1 completion (500-1000 words)

**Success Criteria:**
- Extract 140 raw keywords from Google Trends (7 themes × 20 keywords)
- Filter to 35 high-quality keywords (TF-IDF score >0.3)
- Generate 28 Midjourney prompts (7 themes × 4 variations)
- 100% prompts pass human review (no inappropriate content)
- Total pipeline execution time < 5 minutes

**Stories:** 6-7 stories (Trends API, TF-IDF, prompt templates, LLM integration, validation)

---

### Epic 3: Objective 2 - Midjourney API Integration & Validation
**Timeline:** Day 4-5 (2 days)
**Owner:** Developer (solo)
**Dependencies:** Epic 2 (prompts generated)

**Goal:** 使用 TTAPI Midjourney API 生成 28 張角色一致嘅設計變體，並使用 CLIP 驗證質量

**Key Deliverables:**
- `obj2_midjourney_api/ttapi_client.py`: TTAPI Midjourney API wrapper
- `obj2_midjourney_api/image_generator.py`: Batch generation with --cref
- `obj2_midjourney_api/clip_validator.py`: CLIP similarity validation
- `obj2_midjourney_api/cache_manager.py`: Local image caching
- 28 generated images in `data/generated_images/{theme}/{variation}.png`
- CLIP embeddings (768-dim) stored in `data/clip_embeddings/*.npy`
- Cost analysis report: `reports/ttapi_cost_analysis.md`
- Contextual Report: Objective 2 completion (500-1000 words)

**Success Criteria:**
- 100% images generated successfully (28/28)
- CLIP core similarity > 0.75 for all images
- CLIP style similarity > 0.60 for all images
- Total TTAPI cost < $30
- Cache hit rate > 80% for repeated generations (testing)
- Error handling tested (rate limits, API downtime scenarios)

**Stories:** 8-10 stories (TTAPI integration, --cref testing, batch generation, CLIP validation, caching, error handling, cost tracking)

---

### Epic 4: Objective 3 - LSTM Demand Forecasting
**Timeline:** Day 6-9 (4 days)
**Owner:** Developer (solo)
**Dependencies:** Epic 3 (CLIP embeddings available)

**Goal:** 訓練 hybrid LSTM 模型預測設計變體嘅銷量，並進行 GRU baseline 比較

**Key Deliverables:**
- `obj3_lstm_forecast/sales_simulator.py`: Rule-based sales data generation
- `obj3_lstm_forecast/hybrid_lstm.py`: 2-layer LSTM architecture
- `obj3_lstm_forecast/gru_baseline.py`: GRU comparison model
- `obj3_lstm_forecast/train.py`: Training pipeline with early stopping
- Simulated sales data: `data/simulated_sales.csv` (60 rows)
- Trained LSTM model: `models/lstm/best_model.pth`
- Trained GRU model: `models/gru/best_model.pth`
- Training history: `models/lstm/training_history.json`
- Evaluation report: `reports/lstm_evaluation.md`
- LSTM vs GRU comparison: `reports/lstm_vs_gru_comparison.md`
- Contextual Report: Objective 3 completion (500-1000 words)

**Success Criteria:**
- LSTM R² > 0.7, MAE < 200 units, RMSE < 250 units
- GRU model trained for baseline comparison
- No overfitting (train R² - val R² < 0.15)
- Training time < 30 minutes on Kaggle/Colab
- Simulated sales data realistic (correlation with trends R > 0.6)
- CLIP embeddings successfully integrated as static features

**Stories:** 10-12 stories (simulation logic, LSTM architecture, data preprocessing, training loop, evaluation metrics, GRU baseline, hyperparameter tuning, ablation studies)

---

### Epic 5: Objective 4 - Streamlit Web Application
**Timeline:** Day 10-12 (3 days)
**Owner:** Developer (solo)
**Dependencies:** Epic 2, 3, 4 (all data pipelines complete)

**Goal:** 整合所有 4 個 objectives 成為統一嘅 Streamlit web app，並部署到 Streamlit Cloud

**Key Deliverables:**
- `obj4_web_app/app.py`: Main Streamlit application
- `obj4_web_app/pages/1_design_generation.py`: Design generation page
- `obj4_web_app/pages/2_forecast_dashboard.py`: Forecast dashboard page
- `obj4_web_app/components/`: Reusable UI components
- `obj4_web_app/styles.css`: Custom CSS for business report style
- `assets/logo.png`: ToyzeroPlus logo placeholder
- Deployed app URL (Streamlit Cloud)
- Contextual Report: Objective 4 completion (500-1000 words)

**Success Criteria:**
- Design Generation Page:
  - Input keywords → display 4 mood boards
  - CLIP scores visible for each image
  - Progress bar shows pipeline stages
  - Total execution time < 7 minutes (acceptable for demo)
- Forecast Dashboard Page:
  - Select season/design → display sales prediction
  - Interactive charts (Trends, CLIP, Feature importance)
  - Prediction completes in < 5 seconds
- Navigation:
  - Smooth page switching without data loss
  - Sidebar navigation functional
  - Demo mode works without API keys (cached data)
- Deployment:
  - Successfully deployed to Streamlit Cloud
  - Public URL accessible
  - All cached data included in deployment

**Stories:** 8-10 stories (page layouts, data integration, charts, UI styling, navigation, caching, deployment, demo mode)

---

### Epic 6: Testing, Documentation & FYP Delivery
**Timeline:** Day 13-18 (6 days: 3 testing + 3 documentation + buffer)
**Owner:** Developer (solo)
**Dependencies:** Epic 5 (web app complete)

**Goal:** 完成所有學術要求：Integration testing, Final Report (12,000-15,000 words), Demo Video, submission

**Key Deliverables:**
- **Integration Testing (Day 13-15):**
  - Test scenarios A, B, C executed and documented
  - Error recovery testing completed
  - Bug fixes for critical issues
  - Contextual Report: Integration & Testing (500-1000 words)
- **Final Report (Day 13-18):**
  - `docs/final_report.pdf` (12,000-15,000 words)
  - Sections: Introduction, Literature Review, Methodology, Implementation, Results, Discussion, Conclusion, References
  - Word count breakdown per section (NFR3.2)
- **Demo Video (Day 15-18):**
  - `docs/demo_video_script.md` (5-6 minute script)
  - `docs/demo_video.mp4` (recorded and edited video)
  - Covers all 4 objectives + results + commercial value
- **Code Repository:**
  - Clean up code (remove debug prints, commented code)
  - Ensure reproducibility (requirements.txt, .env.example)
  - Organize file structure
- **FYP Submission Package:**
  - Final Report PDF
  - Demo Video MP4
  - Code repository ZIP (or GitHub link)
  - All required forms/documentation

**Success Criteria:**
- All 3 integration test scenarios pass (100% success rate)
- Final Report meets word count (12,000-15,000 words) and structure requirements
- Demo video runs smoothly, length 5-6 minutes, professional quality
- Repository is clean and reproducible
- All FYP deliverables submitted on time

**Stories:** 12-15 stories (test scenarios, bug fixes, report sections, video recording, editing, submission prep)

---

### Epic Summary Table

| Epic # | Epic Name | Timeline | Dependencies | Story Count | Key Deliverable |
|--------|-----------|----------|--------------|-------------|-----------------|
| 1 | Foundation & Setup | Day 0-1 | None | 4-5 | Environment + APIs validated |
| 2 | Objective 1: NLP Pipeline | Day 2-3 | Epic 1 | 6-7 | 28 Midjourney prompts |
| 3 | Objective 2: Midjourney API | Day 4-5 | Epic 2 | 8-10 | 28 images + CLIP embeddings |
| 4 | Objective 3: LSTM Forecasting | Day 6-9 | Epic 3 | 10-12 | Trained LSTM (R² > 0.7) |
| 5 | Objective 4: Web Integration | Day 10-12 | Epic 2,3,4 | 8-10 | Deployed Streamlit app |
| 6 | Testing & FYP Delivery | Day 13-18 | Epic 5 | 12-15 | Final Report + Demo Video |
| **Total** | - | **18 days** | - | **48-59 stories** | **FYP Submission** |

---

### Critical Path

```
Epic 1 → Epic 2 → Epic 3 → Epic 4 → Epic 5 → Epic 6
(2 days) (2 days) (2 days) (4 days) (3 days) (6 days) → 18 days total with buffer
```

**Parallel Work Opportunities:**
- Day 13-18: Final Report writing can overlap with bug fixing and video recording
- Epic 4 (LSTM) training can run overnight while working on other tasks

**Risk Mitigation:**
- Buffer built into Day 16-18 for Epic 6
- Epic 2 saves 2 days vs LoRA training (original plan Day 4-7)
- Each Epic delivers standalone value (can demo partial completion if time runs out)

---

## 6. Epic Details

### Epic 1: Foundation & Environment Setup (Day 0-1)

#### Story 1.1: Initialize Project Repository
**As a** developer
**I want to** create project repository structure
**So that** all code and data are organized systematically

**Acceptance Criteria:**
- Create folder structure:
  ```
  FYP-RoleMarket/
  ├── src/
  │   ├── obj1_nlp_prompt/
  │   ├── obj2_midjourney_api/
  │   ├── obj3_lstm_forecast/
  │   └── obj4_web_app/
  ├── data/
  │   ├── reference_images/
  │   ├── prompts/
  │   ├── generated_images/
  │   ├── clip_embeddings/
  │   └── cache/
  ├── models/
  │   ├── lstm/
  │   └── gru/
  ├── docs/
  │   ├── contextual_reports/
  │   └── reports/
  ├── .env.example
  ├── .gitignore
  └── requirements.txt
  ```
- Initialize Git repository with `.gitignore` (Python template)
- Create `.env.example` with placeholder API keys

**Effort:** 2 hours

---

#### Story 1.2: Setup Python Environment & Dependencies
**As a** developer
**I want to** install all required Python libraries
**So that** I can develop all 4 objectives without dependency issues

**Acceptance Criteria:**
- Create `requirements.txt` with dependencies:
  ```
  pytrends>=4.9.0
  openai>=1.0.0  # for GPT_API_free
  requests>=2.31.0  # for TTAPI
  torch>=2.0.0
  transformers>=4.30.0  # for CLIP
  scikit-learn>=1.3.0  # for TF-IDF
  pandas>=2.0.0
  numpy>=1.24.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
  plotly>=5.14.0
  streamlit>=1.25.0
  python-dotenv>=1.0.0
  ```
- Create virtual environment: `python3 -m venv venv`
- Install all dependencies: `pip install -r requirements.txt`
- Verify installation (import test for each library)

**Effort:** 1 hour

---

#### Story 1.3: Register & Validate TTAPI Midjourney Account
**As a** developer
**I want to** register TTAPI account and purchase Midjourney quota
**So that** I can use Midjourney API for image generation

**Acceptance Criteria:**
- Register account at https://ttapi.io
- Navigate to Midjourney API section
- Purchase PPU mode quota (~$10-30 budget)
- Document exact pricing in `docs/ttapi_pricing.md`
- Generate API key
- Test API call with sample prompt:
  ```python
  import requests

  response = requests.post(
      "https://api.ttapi.io/midjourney/imagine",
      headers={"Authorization": f"Bearer {TTAPI_KEY}"},
      json={"prompt": "test image"}
  )
  assert response.status_code == 200
  ```
- Store API key in `.env` file
- Add cost tracking log: `docs/cost_log.md`

**Effort:** 2 hours

---

#### Story 1.4: Validate GPT_API_free Access
**As a** developer
**I want to** obtain and test GPT_API_free API key
**So that** I can use LLM for prompt generation

**Acceptance Criteria:**
- Follow GPT_API_free GitHub instructions: https://github.com/chatanywhere/GPT_API_free
- Obtain API key (free tier)
- Test API call with sample prompt:
  ```python
  from openai import OpenAI

  client = OpenAI(
      api_key=GPT_API_KEY,
      base_url="https://api.chatanywhere.com.cn"
  )

  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": "Hello"}]
  )
  assert response.choices[0].message.content
  ```
- Store API key in `.env` file
- Document rate limits (if any) in `docs/gpt_api_limits.md`

**Effort:** 1 hour

---

#### Story 1.5: Source & Validate Pikachu Reference Images
**As a** developer
**I want to** find and download 1-2 high-quality Pikachu reference images
**So that** I can use Midjourney --cref parameter for character consistency

**Acceptance Criteria:**
- Search sources:
  - Official Pokémon press kit
  - Wikimedia Commons
  - DeviantArt (Creative Commons licensed)
  - Pinterest (verify copyright-free)
- Select 1-2 images meeting requirements:
  - Resolution: >1024px shortest side
  - Frontal view of Pikachu
  - Clear features (yellow body, red cheeks, brown stripes)
  - No background clutter
  - Copyright-free or fair use
- Download and save to `data/reference_images/pikachu_ref_1.png`, `pikachu_ref_2.png`
- Upload to publicly accessible URL (for --cref parameter):
  - Option 1: GitHub repository assets
  - Option 2: Imgur/Cloudinary free tier
- Document image sources in `docs/reference_image_sources.md`
- Test TTAPI Midjourney API with --cref:
  ```python
  response = requests.post(
      "https://api.ttapi.io/midjourney/imagine",
      headers={"Authorization": f"Bearer {TTAPI_KEY}"},
      json={
          "prompt": "Pikachu wearing a hat",
          "cref": "<reference_image_url>"
      }
  )
  ```

**Effort:** 2 hours

---

### Epic 2: Objective 1 - Trend Intelligence & Prompt Generation (Day 2-3)

#### Story 2.1: Implement Google Trends Data Extraction
**As a** system
**I want to** extract trending keywords from Google Trends
**So that** designs reflect current market trends

**Acceptance Criteria:**
- Create `src/obj1_nlp_prompt/trends_extractor.py`
- Implement `TrendsExtractor` class:
  ```python
  class TrendsExtractor:
      def __init__(self, region='HK', lang='zh-TW'):
          self.pytrend = TrendReq(hl=lang, tz=480)  # HK timezone
          self.region = region

      def extract_keywords(self, theme, timeframe='today 12-m'):
          # Extract top 20 trending keywords for theme
          # Return DataFrame with keyword, trend_score
          pass
  ```
- Support 7 themes: Halloween, Christmas, Spring Festival, Summer, Valentine's Day, Mid-Autumn Festival, New Year
- Extract 20 keywords per theme (140 total)
- Save results to `data/trends/{theme}_trends.csv`
- Log execution time (target: <30 seconds)

**Effort:** 4 hours

---

#### Story 2.2: Implement TF-IDF Keyword Filtering
**As a** system
**I want to** filter raw keywords using TF-IDF
**So that** only high-quality, distinctive keywords are used

**Acceptance Criteria:**
- Create `src/obj1_nlp_prompt/keyword_extractor.py`
- Implement `KeywordExtractor` class:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer

  class KeywordExtractor:
      def __init__(self, top_k=5):
          self.vectorizer = TfidfVectorizer(tokenizer=jieba.cut)  # Chinese tokenizer
          self.top_k = top_k

      def filter_keywords(self, trend_data):
          # Apply TF-IDF to 140 keywords
          # Select top 5 per theme (35 total)
          # Return keywords with TF-IDF score > 0.3
          pass
  ```
- Support Chinese tokenization (jieba library)
- Filter to 35 keywords (7 themes × 5 keywords)
- Save results to `data/keywords/{theme}_keywords.csv`
- Log TF-IDF scores for analysis

**Effort:** 3 hours

---

#### Story 2.3: Design Prompt Generation Template
**As a** system
**I want to** create reusable prompt templates
**So that** prompts maintain consistent structure and quality

**Acceptance Criteria:**
- Create `src/obj1_nlp_prompt/templates/prompt_template.txt`:
  ```
  Generate a detailed Midjourney prompt for a character IP design variation.

  Character Base Description:
  {character_description}

  Seasonal Theme:
  {theme}

  Trending Keywords:
  {keywords}

  Requirements:
  - Maintain character identity (appearance, colors, features)
  - Incorporate seasonal elements from keywords
  - Use Midjourney-compatible style keywords
  - Keep prompt length 50-150 words
  - Include artistic style (e.g., "cute illustration style", "vibrant colors")

  Output only the Midjourney prompt, no explanations.
  ```
- Create character description file `data/character_descriptions/pikachu.txt`:
  ```
  Pikachu, a yellow electric mouse Pokémon with red circular cheeks,
  brown stripes on its back, pointed ears with black tips, and a lightning bolt-shaped tail.
  Cute and friendly appearance, large expressive eyes.
  ```
- Validate template with sample generation

**Effort:** 2 hours

---

#### Story 2.4: Implement LLM-based Prompt Generator
**As a** system
**I want to** use GPT_API_free to generate Midjourney prompts
**So that** prompts are creative and contextually relevant

**Acceptance Criteria:**
- Create `src/obj1_nlp_prompt/prompt_generator.py`
- Implement `PromptGenerator` class:
  ```python
  from openai import OpenAI

  class PromptGenerator:
      def __init__(self, api_key, template_path):
          self.client = OpenAI(api_key=api_key, base_url="...")
          self.template = load_template(template_path)

      def generate_prompt(self, character_desc, theme, keywords):
          # Fill template with character_desc, theme, keywords
          # Call GPT API
          # Return generated Midjourney prompt
          pass

      def generate_variations(self, character_desc, theme, keywords, n=4):
          # Generate n variations for same theme
          # Return list of prompts
          pass
  ```
- Generate 4 variations per theme (28 total prompts)
- Validate prompt quality:
  - Length: 50-150 words
  - Contains character description
  - Contains seasonal keywords
  - No inappropriate content (manual review)
- Save prompts to `data/prompts/{theme}_variation_{1-4}.txt`
- Log generation time per prompt (target: <10 seconds)

**Effort:** 4 hours

---

#### Story 2.5: Human Review & Validation Pipeline
**As a** developer
**I want to** manually review all generated prompts
**So that** no inappropriate or low-quality prompts are used

**Acceptance Criteria:**
- Create review checklist script `src/obj1_nlp_prompt/review_prompts.py`:
  ```python
  def review_prompts(prompt_dir):
      prompts = load_all_prompts(prompt_dir)
      for prompt in prompts:
          print(f"Theme: {prompt.theme}, Variation: {prompt.variation}")
          print(f"Prompt: {prompt.text}")
          approve = input("Approve? (y/n): ")
          if approve == 'y':
              mark_as_approved(prompt)
          else:
              mark_for_revision(prompt)
  ```
- Review all 28 prompts manually
- Flag any prompts with:
  - Missing character description
  - Irrelevant keywords
  - Inappropriate content
  - Poor readability
- Regenerate flagged prompts
- Document approval status in `data/prompts/review_log.csv`
- **Target:** 100% approval rate (0 flagged prompts)

**Effort:** 2 hours

---

#### Story 2.6: Write Contextual Report - Objective 1
**As a** developer
**I want to** document Objective 1 completion
**So that** progress and challenges are recorded for Final Report

**Acceptance Criteria:**
- Create `docs/contextual_reports/objective_1_completion.md`
- Include sections:
  - **What was completed:** List all deliverables
  - **Technical challenges:** Google Trends rate limits, Chinese tokenization issues, prompt quality control
  - **Solutions implemented:** Retry logic, jieba library, manual review process
  - **Deviation from plan:** Any timeline changes, scope adjustments
  - **Metrics achieved:**
    - Keywords extracted: 140 raw → 35 filtered
    - Prompts generated: 28 (100% approval)
    - Total time: Day 2-3 (on schedule)
  - **Next steps:** Begin Objective 2 (Midjourney API integration)
- Word count: 500-1000 words
- Include code snippets, screenshots if helpful

**Effort:** 2 hours

---

### Epic 3: Objective 2 - Midjourney API Integration & Validation (Day 4-5)

**Stories Summary (detailed breakdown available on request):**
- Story 3.1: Implement TTAPI Midjourney API Client (4h)
- Story 3.2: Test --cref Parameter with Reference Images (3h)
- Story 3.3: Implement Batch Image Generation Pipeline (5h)
- Story 3.4: Implement CLIP Similarity Validation (4h)
- Story 3.5: Implement Image Caching Strategy (3h)
- Story 3.6: Implement Error Handling & Retry Logic (4h)
- Story 3.7: Implement Cost Tracking & Logging (2h)
- Story 3.8: Write Cost Analysis Report (2h)
- Story 3.9: Write Contextual Report - Objective 2 (2h)

**Total Effort:** ~29 hours (Day 4-5)

---

### Epic 4: Objective 3 - LSTM Demand Forecasting (Day 6-9)

**Stories Summary (detailed breakdown available on request):**
- Story 4.1: Design Rule-Based Sales Simulation Logic (4h)
- Story 4.2: Generate 60-Point Simulated Sales Dataset (3h)
- Story 4.3: Implement Hybrid LSTM Architecture (5h)
- Story 4.4: Implement Data Preprocessing Pipeline (3h)
- Story 4.5: Implement Training Loop with Early Stopping (4h)
- Story 4.6: Extract CLIP Embeddings from Generated Images (3h)
- Story 4.7: Train LSTM Model on Kaggle/Colab (4h)
- Story 4.8: Evaluate Model Performance (R², MAE, RMSE) (3h)
- Story 4.9: Implement GRU Baseline Model (4h)
- Story 4.10: Write LSTM vs GRU Comparison Report (3h)
- Story 4.11: Write LSTM Evaluation Report (3h)
- Story 4.12: Write Contextual Report - Objective 3 (2h)

**Total Effort:** ~41 hours (Day 6-9)

---

### Epic 5: Objective 4 - Streamlit Web Application (Day 10-12)

**Stories Summary (detailed breakdown available on request):**
- Story 5.1: Setup Streamlit Multi-Page App Structure (3h)
- Story 5.2: Implement Design Generation Page Layout (5h)
- Story 5.3: Integrate Objective 1 & 2 Pipelines (Design Page) (4h)
- Story 5.4: Implement Forecast Dashboard Page Layout (5h)
- Story 5.5: Integrate Objective 3 Pipeline (Forecast Page) (4h)
- Story 5.6: Implement Interactive Charts (Plotly/Matplotlib) (5h)
- Story 5.7: Apply Business Report Style CSS (4h)
- Story 5.8: Implement Demo Mode (Cached Data) (3h)
- Story 5.9: Deploy to Streamlit Cloud (3h)
- Story 5.10: Write Contextual Report - Objective 4 (2h)

**Total Effort:** ~38 hours (Day 10-12)

---

### Epic 6: Testing, Documentation & FYP Delivery (Day 13-18)

**Stories Summary (detailed breakdown available on request):**
- Story 6.1: Execute Integration Test Scenario A (Spring Festival) (2h)
- Story 6.2: Execute Integration Test Scenario B (Halloween) (2h)
- Story 6.3: Execute Integration Test Scenario C (Christmas) (2h)
- Story 6.4: Execute Error Recovery Testing (3h)
- Story 6.5: Fix Critical Bugs (8h buffer)
- Story 6.6: Write Introduction Section (Final Report) (4h)
- Story 6.7: Write Literature Review Section (Final Report) (6h)
- Story 6.8: Write Methodology Section (Final Report) (8h)
- Story 6.9: Write Implementation Section (Final Report) (6h)
- Story 6.10: Write Experiments & Results Section (Final Report) (8h)
- Story 6.11: Write Discussion Section (Final Report) (4h)
- Story 6.12: Write Conclusion Section (Final Report) (3h)
- Story 6.13: Compile References Section (Final Report) (2h)
- Story 6.14: Write Demo Video Script (3h)
- Story 6.15: Record Demo Video (4h)
- Story 6.16: Edit Demo Video (Professional Quality) (5h)
- Story 6.17: Clean Up Code Repository (3h)
- Story 6.18: Prepare FYP Submission Package (2h)
- Story 6.19: Write Contextual Report - Integration & Testing (2h)

**Total Effort:** ~77 hours (Day 13-18, includes buffer and parallel work)

---

## 7. Appendix

### A. Abbreviations & Glossary

- **TTAPI:** Third-party API platform providing access to Midjourney API
- **--cref:** Midjourney character reference parameter for consistency
- **CLIP:** Contrastive Language-Image Pre-training (OpenAI model)
- **LSTM:** Long Short-Term Memory (recurrent neural network)
- **GRU:** Gated Recurrent Unit (simpler alternative to LSTM)
- **TF-IDF:** Term Frequency-Inverse Document Frequency
- **R²:** Coefficient of Determination (model accuracy metric)
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **FYP:** Final Year Project
- **PPU:** Pay-Per-Use (TTAPI pricing model)

---

### B. References

**Technical Documentation:**
- Midjourney Documentation: https://docs.midjourney.com/
- TTAPI Midjourney API: https://ttapi.io/docs/apiReference/midjourney
- CLIP Paper: https://arxiv.org/abs/2103.00020
- LSTM Tutorial: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Key Libraries:**
- pytrends: https://pypi.org/project/pytrends/
- TTAPI Python SDK: https://ttapi.io/docs
- PyTorch: https://pytorch.org/
- Streamlit: https://docs.streamlit.io/
- Transformers (CLIP): https://huggingface.co/docs/transformers/

**Related Projects:**
- GPT_API_free: https://github.com/chatanywhere/GPT_API_free

---

**PRD Document Status:** ✅ Complete - Ready for Development

**Total Sections:** 7 (Goals, Requirements, UI Goals, Technical Assumptions, Epic List, Epic Details, Appendix)

**Total Word Count:** ~8,000 words (PRD specification document)

**Next Steps:** Begin Epic 1 - Foundation & Environment Setup (Day 0)
