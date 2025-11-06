# AI-Driven Market-Informed Character IP Design & Demand Forecasting

**FYP Project - ToyzeroPlus Commercial AI Pipeline**

## 專案概述

商業級 AI 系統，整合市場趨勢分析、Midjourney 設計生成、LSTM 需求預測，專為 character IP 設計公司提供即刻可部署嘅生產工具。

## 核心功能

- **Objective 1:** Google Trends 趨勢分析 + LLM Prompt 生成 - ✅ **完成**
  - ✅ **Cultural Trend Adapter** - 智能轉化所有文化趨勢（Meme, Holiday, Design Style, Social Media, Mood）
  - ✅ **Reference Image 優先策略** - 保持角色一致性（CLIP >= 0.8）
- **Objective 2:** Google Gemini Image 商業級設計生成（Reference Image Consistency） - ✅ **完成**
  - ✅ **角色一致性驗證** - CLIP 相似度達標（0.8157）
  - ✅ **快速生成** - 11.18s/圖（Google Gemini 2.5 Flash Image）
- **Objective 3:** Hybrid Transformer 銷量預測（結合 Trends + CLIP embeddings） - ✅ **完成**
  - ✅ **企業級預測** - R² = 0.6788（超越 0.65 目標）
  - ✅ **MAE = 327.26** - 11.5% 誤差率
  - ✅ **生產就緒** - Exp #11v2 最終方案
- **Objective 4:** Streamlit 統一 Web 介面 - ✅ **完成**
  - ✅ **Story 4.1 完成** - Streamlit 基礎架構 + Obj 1 整合
  - ✅ **Story 4.2 完成** - Obj 2 圖片生成與 CLIP 驗證整合
  - ✅ **Story 4.3 完成** - Obj 3 銷量預測儀表板（2025-11-06）

### 最新改進（v1.3 - 2025-10-29）

**Phase A 完成：Objective 3 Transformer 預測模型達到企業級標準**

| 實驗 | R² | MAE | RMSE | 狀態 |
|------|-----|-----|------|------|
| Exp #10 (Baseline) | 0.5127 | 419.26 | 589.42 | ❌ 訓練不足 |
| Exp #11v2 (最終) | **0.6788** | **327.26** | **456.40** | ✅ 採用 |
| Exp #12v3 (Ensemble) | 0.9525 | 138.06 | - | ❌ Data Leakage |
| Exp #14 (數據增強) | 0.9737 | - | - | ❌ Data Leakage |

**核心成果：**
1. ✅ **企業級標準達成** - R² = 0.6788（超越 0.65 目標）
2. ✅ **14+ 次實驗迭代** - 完整優化路徑記錄
3. ✅ **生產就緒模型** - Exp #11v2（Hybrid Transformer）
4. ✅ **數據洩漏診斷** - Ensemble 和數據增強方案驗證

詳細說明：[`docs/experiment-log-lulu-transformer.md`](docs/experiment-log-lulu-transformer.md)

---

### Objective 1 & 2 改進（v1.2 - 2025-10-27）

**策略轉變：從詳細描述策略 → Reference Image 優先策略**

| 指標 | v1.1 詳細描述 | v1.2 Reference Image | 改進 |
|-----|-------------|---------------------|-----|
| **CLIP 相似度** | ~0.78 | **0.8157** | +4.5% ✅ |
| **Prompt 長度** | 150-200 words | **79 words** | -52% ✅ |
| **生成速度** | ~15s | **11.18s** | +25% ✅ |

**核心改進：**
1. ✅ **Cultural Trend Adapter** - 支援 5 大類文化趨勢（非僅 Meme）
2. ✅ **Reference Image 策略** - 不描述角色，只添加場景元素
3. ✅ **簡化 Prompt** - 從 ~680 字元角色描述縮減至簡潔指示

詳細說明：[`docs/strategy-improvements-v1.2.md`](docs/strategy-improvements-v1.2.md)

## 專案結構

```
FYP-RoleMarket/
├── obj1_nlp_prompt/       # Trend Intelligence & Prompt Generation
├── obj2_midjourney_api/   # Midjourney API Design Generation
├── obj3_lstm_forecast/    # LSTM Demand Forecasting
├── obj4_web_app/          # Streamlit Web Application
├── data/                  # Data storage (cache, images, trends)
├── tests/                 # Integration & unit tests
├── docs/                  # PRD, reports, documentation
└── config/                # API keys & configuration
```

## 快速開始

### 方法一：使用 Streamlit Web 介面（推薦）

1. **安裝依賴：**
   ```bash
   # 使用 virtual environment (推薦)
   python3 -m venv .venv
   source .venv/bin/activate

   # 安裝所有依賴
   pip install -r requirements.txt
   ```

2. **設置 API Keys：**

   建立 `.env` 檔案並填入以下內容：
   ```bash
   # GPT_API_free (Llama 3.1)
   GPT_API_FREE_KEY=your_gpt_api_key_here

   # Google Gemini API (Optional - for image generation)
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **啟動 Streamlit 應用：**
   ```bash
   streamlit run obj4_web_app/app.py
   ```

   應用會自動在瀏覽器打開 `http://localhost:8501`

4. **使用流程：**
   - 📊 **頁面 1: 設計生成** - 輸入趨勢關鍵字，生成 AI Prompt
   - 🎨 **頁面 2: 圖片生成** (Coming Soon) - 生成設計圖並驗證角色一致性
   - 📈 **頁面 3: 銷量預測** (Coming Soon) - 上傳設計圖，預測銷量

### 方法二：命令列執行（進階用戶）

**Objective 1 - 趨勢分析與 Prompt 生成：**
```bash
python obj1_nlp_prompt/enhanced_trends_pipeline.py
```

**Objective 2 - Google Gemini 圖片生成：**
```bash
python obj2_midjourney_api/google_gemini_client.py
```

**Objective 3 - Transformer 銷量預測：**
```bash
python obj3_lstm_forecast/kaggle_train_lulu_transformer.py
```

## 技術棧

- **AI Models:**
  - CLIP ViT-Large/14 (角色一致性驗證)
  - Hybrid Transformer (需求預測，D_MODEL=64, NUM_LAYERS=2)
  - GPT-3.5-turbo (Prompt 生成)
- **APIs:** Google Gemini 2.5 Flash Image, Google Trends, GPT_API_free
- **Framework:** PyTorch 2.0+, Streamlit, Transformers, Google GenerativeAI SDK
- **成本:** Free (Google Gemini Flash Image)

## 文檔

- **PRD:** [`docs/prd.md`](docs/prd.md)
- **完整實驗記錄（Obj 3）:** [`docs/experiment-log-lulu-transformer.md`](docs/experiment-log-lulu-transformer.md) ⭐ **最新**
- **Phase A 完成報告:** [`docs/phase-a-completion-report.md`](docs/phase-a-completion-report.md)
- **實施路線圖:** [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md)
- **最新策略改進（v1.2 - Obj 1&2）:** [`docs/strategy-improvements-v1.2.md`](docs/strategy-improvements-v1.2.md)
- **PRD Enhancement v1.1:** [`docs/prd-enhancement-v1.1.md`](docs/prd-enhancement-v1.1.md) (已淘汰)

## 測試報告

- **Objective 3 完整實驗記錄:** [`docs/experiment-log-lulu-transformer.md`](docs/experiment-log-lulu-transformer.md) ⭐ **最新**
- **Phase A 完成總結:** [`docs/phase-a-completion-report.md`](docs/phase-a-completion-report.md)
- **完整端到端測試（Obj 1&2）:** [`data/generated_images/e2e_test/e2e_20251027_170132_report.md`](data/generated_images/e2e_test/e2e_20251027_170132_report.md)
- **Epic 3 完成總結:** [`docs/epic_3_completion_summary.md`](docs/epic_3_completion_summary.md)

---

**Version:** 1.3
**Author:** Product Manager
**Last Updated:** 2025-10-29
**Status:** Phase A 完成（Obj 1-3 ✅），Phase B（Obj 4）待進行
