# Brainstorming Session Results

**Session Date:** 2025-01-25
**Facilitator:** Business Analyst Mary
**Participant:** Project Developer

---

## Executive Summary

**Topic:** AI-Driven Market-Informed Character IP Design Extension and Demand Forecasting System - 技術實作

**Session Goals:**
- 將 4 個 objectives 拆解到最基本的技術需求
- 在嚴格限制下(免費資源、17-18日、Python、可交付)找到最可行方案
- 系統性優化每個 objective 的實作策略

**Techniques Used:**
1. First Principles Thinking (階段 1, 45分鐘)
2. Resource Constraints (階段 2, 30分鐘)
3. SCAMPER Method (階段 3, 35分鐘)

**Total Ideas Generated:** 26 個決策點 + 15 個優化方案

### Key Themes Identified:

- **專案定位轉變**: 從學術 demo 轉變為 ToyzeroPlus 真實商業應用案例
- **實用主義**: 優先選擇簡單有效的方案而非完美但複雜的解決方案
- **時間優化**: 透過簡化實驗和刪除非必要功能,從 14-18 日優化至 12-17 日
- **完整循環系統**: 發現專案本身就是一個 self-validating pipeline (生成 → 評估 → 預測)
- **混合模型架構**: LSTM 處理時間序列趨勢 + 靜態設計特徵的創新架構
- **免費資源最大化**: 善用 GPT_API_free、Kaggle、HF Inference API 等免費工具

---

## Technique Sessions

### Session 1: First Principles Thinking - 45分鐘

**Description:** 將每個 Objective 拆解到最基本的技術挑戰,從底層思考「真正需要做什麼」而非「用什麼現成工具」

#### Ideas Generated:

1. **Obj 1 核心需求識別**
   - 需要「流行趨勢 + 熱門話題」可以與角色融合
   - 視覺元素 + 情感/氛圍都重要
   - 數字指標 = 某段時間的出現次數
   - 只追蹤上升趨勢

2. **數據源簡化決策**
   - 原計劃: Instagram + Reddit + Google Trends
   - 簡化為: 只用 Google Trends (pytrends)
   - 備案: 自己模擬數據

3. **LLM 處理策略**
   - 視覺元素 + 情感/氛圍交給開放 LLM 處理
   - 用 GPT_API_free (無限制訪問)
   - 需要角色描述 template

4. **LoRA 一致性基本原理**
   - 必須保留: 角色核心特徵(大眼睛、尖耳、身體比例)
   - 可以變化: 顏色、配飾、服裝、背景
   - 訓練圖分配: 10張表情 + 2張側面 + 3張配飾 (方案 B 調整版)
   - 定位: 生成 mood board 靈感而非完美最終產品

5. **LSTM 預測架構突破**
   - 發現: 每年都是全新設計,不是追蹤同一設計
   - 關鍵洞察: LSTM 處理時間依賴(趨勢變化) + 靜態設計特徵
   - 混合架構: LSTM(趨勢時間序列) + Dense(當前設計 CLIP features)
   - 數據結構: 60 個訓練點,每點包含過去 3-4 季歷史

6. **模擬銷量規則設計**
   - 場景 B: 基於規則的模擬(而非隨機)
   - 規則: Google Trends 高 + CLIP 相似度高 + 生產數量限制 + 季節因素 + 隨機噪音
   - 5年歷史數據 (2021-2025) 跨年份追蹤

7. **Web App 用戶流程**
   - 系統自動運行(定時 or 手動觸發)
   - 創意總監打開 app 查看結果
   - 顯示: mood board + 預測圖表 + 趨勢關鍵字
   - 決策交給公司開會

#### Insights Discovered:

- **ToyzeroPlus 真實素材可用**: 專案從學術 demo 升級為真實商業案例
- **完整循環系統**: Google Trends → LLM → SDXL → CLIP → LSTM → 預測,形成 self-validating pipeline
- **時間序列混合模型**: LSTM 不是純時間序列也不是純特徵預測,而是混合兩者
- **Mood board 定位**: 系統目標是提供靈感選擇,而非完全自動化決策

#### Notable Connections:

- CLIP embeddings 同時用於: 一致性驗證(Obj 2) + LSTM 預測特徵(Obj 3)
- Google Trends 數據除了生成 prompt,還可以做趨勢儀表板
- LoRA 訓練完可以多重用途: 產品包裝、社交媒體宣傳圖

---

### Session 2: Resource Constraints - 30分鐘

**Description:** 在嚴格限制($0 預算、17-18日、Google Colab、Python)下激發最實用、最可行的創意解決方案

#### Ideas Generated:

1. **Obj 1 免費資源方案**
   - pytrends (免費 Google Trends API)
   - GPT_API_free (無限制 ChatGPT API)
   - 簡化比較: 只用 TF-IDF + LLM (省 1-2 日)
   - 合併步驟: pytrends → 直接餵 LLM

2. **Obj 2 訓練環境選擇**
   - Kaggle 取代 Colab (30小時/週 GPU,無 90 分鐘限制)
   - Image inpainting 處理水印問題
   - 訓練完即刻提取 CLIP features (for Obj 3)
   - 最小可行實驗: rank 8 vs 16 比較(+1 日,符合學術要求)

3. **Obj 3 效率優化**
   - 並行處理生成 60 時間點數據
   - GRU 作為備案模型(比 LSTM 簡單快速)
   - 特徵 concatenate (617-dim) 而非降維
   - 模擬規則加入生產數量限制

4. **Obj 4 架構簡化**
   - Streamlit 主方案 (Python 全棧,2-3日)
   - Next.js + AG Chat 備案
   - HF Inference API 解決 SDXL RAM 限制
   - 影片 demo 解決等待時間問題

#### Insights Discovered:

- **免費資源充足**: GPT_API_free、Kaggle、HF API 完全可以支撐整個專案
- **學術與效率平衡**: 最小可行實驗(2次訓練)既符合 FYP 要求又節省時間
- **架構分離優勢**: Streamlit(前端) + HF API(SDXL) 分離,RAM 不再是問題

#### Notable Connections:

- Kaggle 解決了 Colab 的時間限制,同時提供更長的 GPU 時間
- 並行處理策略可以應用到多個 objectives
- 備案模型(GRU)的存在降低了技術風險

---

### Session 3: SCAMPER Method - 35分鐘

**Description:** 系統性優化每個 objective 的方案,確保最優化且可交付

#### Ideas Generated (按 SCAMPER 類別):

**Substitute (替代):**
- Image inpainting 處理水印
- GRU 備案模型
- Streamlit/Next.js 雙平台方案

**Combine (結合):**
- pytrends → LLM 一次過處理
- 訓練完即刻提取 CLIP features
- 2 頁結構 (生成 + 分析) 合併多功能

**Adapt (調整):**
- 訓練圖比例調整: 10 張表情主導(符合 mood board 定位)
- CLIP 一致性分級: core >0.75, style >0.60
- 混合模式簡化為純手動(務實選擇)
- 模擬規則加入生產數量限制

**Modify (修改):**
- CLIP 閾值彈性化(支持創意變化)
- 60 時間點數據結構確認

**Eliminate (刪除):**
- 刪除 TextRank + RAKE 比較
- 刪除定時自動執行功能(-1 日)
- 不加用戶反饋功能
- 簡化 hyperparameter tuning

**Put to other use (另作他用):**
- Google Trends → 趨勢儀表板
- LoRA → 產品包裝 + 社交媒體宣傳
- LSTM → 特徵重要性分析 + 市場趨勢報告
- Web app → 客戶參與 + 投資者演示(Future Work)

**Reverse (反轉):**
- 完整循環: 生成 → 評估(已內建)
- 反推訓練圖需求: 必須包含服裝 + 姿勢 + 角度
- 反向優化: 不適用(已評估)
- 流程反轉: Future Work

#### Insights Discovered:

- **時間優化潛力**: 透過刪除非必要功能,Obj 4 從 3-4 日降至 2-3 日
- **Future Work 清單**: 定時自動、客戶參與、投資者演示、流程反轉
- **多重商業價值**: LoRA 和 LSTM 都有超越 FYP 的長期應用價值

#### Notable Connections:

- 刪除定時功能與簡化比較實驗,共節省 2-3 日時間
- 多個 objectives 的輸出都可以作為其他 objectives 的輸入(系統整合性高)

---

## Idea Categorization

### Immediate Opportunities
*Ideas ready to implement now*

1. **使用免費 API 生態系統**
   - Description: pytrends + GPT_API_free + Kaggle + HF Inference API
   - Why immediate: 完全免費,無需申請,立即可用
   - Resources needed: 網絡連接、API 文檔

2. **ToyzeroPlus 真實角色素材**
   - Description: 從公司獲取 12-15 張產品照/設計稿
   - Why immediate: 已有工作關係,可立即獲取
   - Resources needed: 與公司溝通、圖片收集

3. **Streamlit 快速原型**
   - Description: 用 Streamlit 搭建 2 頁 web app
   - Why immediate: Python 全棧,2-3 日完成
   - Resources needed: Streamlit Cloud 免費帳號

4. **最小可行實驗設計**
   - Description: 每個 objective 只做必要的對比實驗
   - Why immediate: 平衡學術要求與時間限制
   - Resources needed: 實驗設計文檔

5. **混合 LSTM 架構**
   - Description: 時間序列(趨勢) + 靜態特徵(設計)混合預測
   - Why immediate: 創新且符合實際需求
   - Resources needed: PyTorch, 訓練數據

### Future Innovations
*Ideas requiring development/research*

1. **定時自動執行系統**
   - Description: GitHub Actions 或 APScheduler 定期生成設計
   - Development needed: 排程系統、錯誤處理、通知機制
   - Timeline estimate: 額外 1-2 日開發時間

2. **客戶參與投票功能**
   - Description: 讓粉絲在 web app 投票喜歡的設計
   - Development needed: 投票系統、數據收集、分析儀表板
   - Timeline estimate: 額外 2-3 日

3. **反向流程優化**
   - Description: 從目標銷量反推需要的設計特徵
   - Development needed: 反向 LSTM、特徵生成算法
   - Timeline estimate: 需要進一步研究(1-2 週)

4. **多角色支持**
   - Description: 擴展到 ToyzeroPlus 其他角色 IP
   - Development needed: 多 LoRA 管理、角色切換 UI
   - Timeline estimate: 每個新角色 +2-3 日

### Moonshots
*Ambitious, transformative concepts*

1. **完全自動化設計流水線**
   - Description: 從趨勢監測到生產決策的全自動系統
   - Transformative potential: 徹底改變 ToyzeroPlus 的產品開發流程
   - Challenges to overcome: 模型可靠性、商業決策信任度、成本控制

2. **行業級設計預測平台**
   - Description: 將系統擴展為 SaaS 平台,服務整個設計玩具行業
   - Transformative potential: 創造新商業模式,賦能中小型設計公司
   - Challenges to overcome: 數據隱私、多客戶管理、定價策略

3. **AI 驅動的完整 IP 開發生態**
   - Description: 整合設計、製造、營銷、銷售預測的端到端系統
   - Transformative potential: 重新定義角色 IP 產業的運作方式
   - Challenges to overcome: 跨部門整合、供應鏈協調、大規模數據需求

### Insights & Learnings
*Key realizations from the session*

- **專案定位的重要性**: 從「學術 demo」轉變為「真實商業應用」大大提升了 FYP 的價值和說服力
- **時間序列的本質**: LSTM 不是處理「同一產品的歷史」,而是「市場趨勢 + 設計特徵」的混合預測
- **限制激發創意**: 嚴格的時間和預算限制反而幫助聚焦最實用的解決方案
- **系統性思考價值**: First Principles → Resource Constraints → SCAMPER 三階段方法系統性地優化了整個專案
- **免費資源生態豐富**: 2025 年的 AI 開源生態已經足夠支撐完整的商業級應用
- **模塊化設計優勢**: 每個 objective 的輸出都可以被其他部分重用,形成緊密整合的系統

---

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: Objective 1 - NLP 生成設計提示

- **Rationale**:
  - 是整個 pipeline 的起點
  - 技術風險最低(已有成熟方案)
  - 可以快速驗證整體可行性

- **Next steps**:
  1. 設定 pytrends 收集 3 個季節的歷史數據
  2. 測試 GPT_API_free 生成 prompt 的質量
  3. 建立角色描述 template
  4. 實作 TF-IDF keyword extraction
  5. 完成 4 個季節 × 4 個設計的 prompt 生成

- **Resources needed**:
  - pytrends library
  - GPT_API_free API key
  - Google Trends 歷史數據
  - 開發時間: 2-3 日

- **Timeline**: Day 1-3

---

#### #2 Priority: Objective 2 - LoRA 訓練角色一致性

- **Rationale**:
  - 從 ToyzeroPlus 獲取真實素材是專案的核心優勢
  - LoRA 訓練是生成高質量設計的關鍵
  - 需要較長訓練時間,應盡早開始

- **Next steps**:
  1. 從 ToyzeroPlus 收集 12-15 張角色圖(10表情+2側面+3配飾)
  2. 使用 image inpainting 處理水印(如需要)
  3. 在 Kaggle 設定訓練環境
  4. 訓練 2 個 LoRA 版本(rank 8 vs 16)
  5. 用 CLIP 驗證一致性(>0.75 core, >0.60 style)
  6. 生成測試設計並提取 CLIP features

- **Resources needed**:
  - ToyzeroPlus 角色圖片
  - Kaggle GPU (30 hrs/week)
  - Kohya_ss 或 diffusers
  - CLIP model
  - 開發時間: 3-4 日

- **Timeline**: Day 4-7

---

#### #3 Priority: Objective 3 - LSTM 需求預測模型

- **Rationale**:
  - 是專案的創新核心(混合時間序列 + 特徵預測)
  - 需要依賴 Obj 1 和 Obj 2 的輸出
  - 包含模擬數據生成和模型訓練,工作量大

- **Next steps**:
  1. 設計 5 年歷史數據模擬規則(Trends + CLIP + 生產限制 + 噪音)
  2. 並行生成 60 個歷史數據點
  3. 構建 LSTM 混合架構(時間序列 + 靜態特徵)
  4. 訓練 LSTM 和 GRU(對比實驗)
  5. 評估模型性能(MAE, RMSE, R²)
  6. 創建預測 API

- **Resources needed**:
  - PyTorch / TensorFlow
  - 60 個完整數據點(Obj 1 + Obj 2 輸出)
  - Kaggle/Colab CPU
  - 開發時間: 3-4 日

- **Timeline**: Day 8-11

---

## Reflection & Follow-up

### What Worked Well

- **結構化框架**: 三階段 brainstorming 系統性地覆蓋了所有面向
- **深度追問**: 透過引導性問題幫助挖掘真實需求和限制
- **實用主義**: 始終聚焦於「17-18 日內可交付」的現實目標
- **彈性調整**: 當發現 ToyzeroPlus 資源可用時,立即調整專案定位
- **視覺化思考**: 用具體場景和數據範例幫助理解抽象概念

### Areas for Further Exploration

- **LSTM 架構細節**: 具體的 layer 結構、activation functions、regularization 策略需要在實作時決定
- **模擬數據真實性驗證**: 可能需要與 ToyzeroPlus 驗證模擬規則是否合理
- **LoRA 訓練超參數**: rank 8 vs 16 之外,learning rate、steps 等參數可能需要微調
- **Web app UX 設計**: 具體的介面布局、互動流程需要 wireframe 設計
- **錯誤處理策略**: API 失敗、模型預測異常等邊界情況的處理

### Recommended Follow-up Techniques

- **Prototyping**: 快速搭建 Streamlit prototype 驗證技術可行性
- **Risk Assessment**: 用 QA agent 的 `*risk-profile` 評估高風險部分
- **Time Boxing**: 為每個 objective 設定嚴格時間限制,避免過度優化

### Questions That Emerged

- ToyzeroPlus 是否願意分享一些真實銷售數據來驗證模擬規則?
- 是否需要考慮不同地區(香港 vs 海外)的趨勢差異?
- LSTM 預測結果的準確度需要達到什麼水平才被認為「有用」?
- 如果某個 objective 遇到無法解決的技術問題,是否有 pivot 方案?
- Final Report 需要多詳細的實驗記錄和代碼文檔?

### Next Session Planning

- **Suggested topics**:
  1. 建立詳細的 17-18 日 implementation roadmap
  2. 設計實驗記錄和文檔結構(符合 FYP 要求)
  3. 準備 risk mitigation strategies
  4. 規劃 demo 影片腳本

- **Recommended timeframe**: 1-2 日內完成規劃,然後立即開始實作

- **Preparation needed**:
  - 確認 ToyzeroPlus 圖片獲取流程
  - 註冊所有需要的免費服務帳號
  - 準備開發環境(Python、Git、Kaggle)
  - 複習相關技術文檔(SDXL, LoRA, LSTM)

---

*Session facilitated using the BMAD-METHOD™ brainstorming framework*

---

## 附錄: 完整技術架構總結

### 資料流動圖

```
階段 1: 歷史數據準備 (離線)
────────────────────────────────
Google Trends (2021-2025)
    ↓ pytrends
TF-IDF 提取關鍵字
    ↓
GPT_API_free 生成 prompts
    ↓
SDXL + LoRA 生成設計圖
    ↓
CLIP 提取 512-dim features
    ↓
基於規則模擬銷量
    ↓
[60 個歷史數據點]
    ↓
LSTM + GRU 訓練
    ↓
[訓練好的預測模型]

階段 2: 實際使用 (2026)
────────────────────────────────
Google Trends (2026 實時)
    ↓ pytrends
TF-IDF 提取當前關鍵字
    ↓
GPT_API_free 生成 4 個 prompts
    ↓
HF Inference API (SDXL + LoRA)
    ↓
CLIP 提取新設計 features
    ↓
LSTM 預測 (過去 3 季歷史 + 新設計)
    ↓
Streamlit 顯示:
  - 4 張 mood board
  - 預測圖表
  - 趨勢儀表板
```

### 最終時間分配

```
Day 1-3:   Obj 1 (Google Trends + LLM prompt)
Day 4-7:   Obj 2 (準備圖 + LoRA 訓練 + 實驗)
Day 8-11:  Obj 3 (模擬數據 + LSTM 訓練)
Day 12-14: Obj 4 (Streamlit 開發)
Day 15-17: 整合測試 + 錄影 demo + 文檔
Day 18:    緩衝時間

總計: 17-18 日 ✅
```

### 技術棧清單

**數據收集:**
- pytrends (Google Trends API)

**NLP & Prompt:**
- GPT_API_free (ChatGPT)
- scikit-learn (TF-IDF)

**圖像生成:**
- Hugging Face Inference API (SDXL)
- LoRA weights (Kaggle 訓練)

**特徵提取:**
- CLIP (OpenAI)
- transformers library

**預測模型:**
- PyTorch (LSTM + GRU)
- numpy, pandas

**Web 應用:**
- Streamlit
- Streamlit Cloud (部署)

**輔助工具:**
- Image inpainting (LaMa)
- matplotlib/plotly (圖表)

### Future Work 清單

1. **功能擴展**
   - 定時自動執行系統
   - 客戶參與投票功能
   - 投資者演示模式
   - 反轉流程(趨勢先行)

2. **技術優化**
   - 從目標銷量反推設計特徵
   - 多角色 IP 支持
   - 更精細的模擬銷量規則
   - 實時趨勢監測

3. **商業應用**
   - SaaS 平台化
   - 行業級設計預測服務
   - 完整 IP 開發生態整合
