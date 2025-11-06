# Story 4.3: Obj 3 銷量預測與市場洞察儀表板

**Story ID:** STORY-4.3
**Epic:** EPIC-004 - Streamlit Web Application Integration
**Status:** ✅ Done
**Priority:** High
**Points:** 10
**Created:** 2025-11-06
**Completed:** 2025-11-06
**Assigned To:** Developer (James)
**Depends On:** STORY-4.2

---

## User Story

**As a** ToyzeroPlus 產品經理，
**I want** 基於生成的設計圖和市場趨勢預測未來季度銷量，並查看市場洞察建議，
**So that** 我可以數據驅動地決定哪些設計應該投入生產，以及最佳上市時機。

---

## Story Context

### Existing System Integration

**整合對象：** Objective 3 - Hybrid Transformer 銷量預測

**核心模組：**
- `obj3_lstm_forecast/hybrid_transformer_model.py` - Transformer 模型架構
- `obj3_lstm_forecast/kaggle_train_lulu_exp11v2.py` - 最終訓練腳本（R² = 0.6788）
- `models/transformer_lulu/` - 訓練好的模型權重

**技術棧：**
- PyTorch 2.0+ (Transformer Model)
- Transformers 4.30+ (CLIP embeddings)
- NumPy 1.24+ (數據處理)
- Plotly 5.17+ (互動圖表)
- Streamlit 1.28+ (儀表板)

**整合模式：**
- 從 Story 4.2 的 session state 讀取生成圖片和 CLIP embeddings
- 透過 `utils/forecast_predictor.py` wrapper 調用 Obj 3 預測模型
- 建立 Page 2: 預測儀表板，顯示銷量預測和市場洞察
- 使用 Plotly 視覺化歷史趨勢和預測結果

**現有 Touch Points：**
- `HybridTransformerModel.predict(time_series, static_features)` - 預測 API
- CLIP embeddings（768-dim）作為 static features
- Google Trends 歷史數據作為 time-series features
- 模型輸出：預測銷量（數值）

---

## Acceptance Criteria

### Functional Requirements

**FR1: Page 2 儀表板基礎結構**
- [ ] 建立 `pages/2_📊_銷量預測.py`
- [ ] 頁面標題：「銷量預測與市場洞察」
- [ ] 分為 3 個區塊：
  1. 輸入區（季節、設計選擇）
  2. 預測結果區（銷量數字、信心區間）
  3. 市場洞察區（歷史趨勢、Feature Importance）

**FR2: 預測輸入介面**
- [ ] 季節選擇器（Spring/Summer/Fall/Winter）
- [ ] 設計選擇器（從 Story 4.2 生成的 4 張圖中選擇）
  - 顯示縮圖 + CLIP 相似度
  - 支援選擇多個設計進行對比
- [ ] "預測銷量" 按鈕

**FR3: 銷量預測結果顯示**
- [ ] 顯示預測銷量（大字體，使用 `st.metric()`）
- [ ] 顯示誤差範圍（基於 MAE = 327.26）
  - 如：預測 1,500 件 ± 327 件
- [ ] 顯示預測信心度（基於模型 R² = 0.6788）
  - 如：信心度 68%
- [ ] 對比顯示（如選擇多個設計）

**FR4: 歷史趨勢視覺化**
- [ ] Plotly line chart 顯示過去 4 季度銷量趨勢
- [ ] 當前預測點標註在圖表上
- [ ] 互動功能（hover 顯示詳細數據）
- [ ] 季節性標註（Spring/Summer/Fall/Winter 顏色區分）

**FR5: 市場洞察摘要**
- [ ] Feature Importance 分析（基於 Obj 3 實驗結果）
  - Google Trends 影響權重
  - CLIP Similarity 影響權重
  - 季節因素影響權重
- [ ] 自動生成建議（基於預測結果）
  - 最佳上市時機
  - 競爭程度評估
  - 生產數量建議
- [ ] 風險提示（如：預測誤差較大時）

**FR6: 錯誤處理**
- [ ] 如 Story 4.2 未完成（無生成圖片），顯示引導訊息
- [ ] 模型載入失敗時顯示錯誤
- [ ] 預測失敗時提供重試選項

### Integration Requirements

**IR1: Obj 3 API 封裝**
- [ ] 建立 `utils/forecast_predictor.py` wrapper
- [ ] 實作 `load_model()` 函數（載入訓練好的模型）
- [ ] 實作 `predict_sales(season, clip_embedding, trends_history)` 函數
- [ ] 實作 `get_feature_importance()` 函數

**IR2: Session State 管理**
- [ ] 從 `st.session_state['generated_images']` 讀取圖片（Story 4.2）
- [ ] 從 `st.session_state['clip_embeddings']` 讀取 CLIP embeddings（Story 4.2）
- [ ] 從 `st.session_state['trends_data']` 讀取趨勢歷史（Story 4.1）
- [ ] 儲存預測結果至 `st.session_state['predictions']`

**IR3: 現有功能保留**
- [ ] Obj 3 CLI 腳本仍可獨立運行
- [ ] `hybrid_transformer_model.py` 的 API 不被修改
- [ ] 訓練好的模型權重不被修改

### Quality Requirements

**QR1: 性能優化**
- [ ] 使用 `@st.cache_resource` 快取模型載入（首次 < 5 秒）
- [ ] 使用 `@st.cache_data` 快取歷史趨勢數據
- [ ] 預測時間 < 3 秒

**QR2: 用戶體驗**
- [ ] 預測過程顯示 loading spinner
- [ ] 預測完成顯示成功通知
- [ ] 數字顯示格式化（千位分隔符）
- [ ] 圖表互動流暢

**QR3: 準確性驗證**
- [ ] 預測結果與 Obj 3 原始模型輸出一致（誤差 < 1%）
- [ ] MAE 和 R² 指標顯示正確
- [ ] Feature Importance 權重總和 = 100%

**QR4: 測試覆蓋**
- [ ] 為 `utils/forecast_predictor.py` 編寫單元測試
- [ ] 測試不同季節的預測結果
- [ ] 執行 Obj 3 regression test

---

## Technical Notes

### Integration Approach

**Wrapper 設計模式：**
```python
# utils/forecast_predictor.py
import torch
from obj3_lstm_forecast.hybrid_transformer_model import HybridTransformerModel

class ForecastPredictorWrapper:
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self.mae = 327.26  # From Exp #11v2
        self.r2 = 0.6788

    @st.cache_resource
    def load_model(_self):
        """載入訓練好的 Transformer 模型"""
        model = HybridTransformerModel(
            d_model=64,
            num_layers=2,
            nhead=8,
            clip_dim=768,
            product_type_dim=4
        )
        model.load_state_dict(torch.load(_self.model_path))
        model.eval()
        return model

    def predict_sales(
        self,
        season: str,
        clip_embedding: np.ndarray,
        trends_history: List[float]
    ) -> Dict[str, float]:
        """
        預測銷量

        Args:
            season: "Spring", "Summer", "Fall", "Winter"
            clip_embedding: (768,) CLIP embedding
            trends_history: [Q-3, Q-2, Q-1, Q0] Google Trends scores

        Returns:
            {
                'predicted_sales': float,
                'lower_bound': float,  # predicted - MAE
                'upper_bound': float,  # predicted + MAE
                'confidence': float    # R²
            }
        """
        if self.model is None:
            self.model = self.load_model()

        # 準備輸入數據
        season_encoding = self._encode_season(season)  # One-hot (4,)
        static_features = np.concatenate([clip_embedding, season_encoding])  # (772,)
        time_series = np.array(trends_history).reshape(-1, 1)  # (4, 1)

        # 轉換為 tensor
        ts_tensor = torch.FloatTensor(time_series).unsqueeze(0)  # (1, 4, 1)
        static_tensor = torch.FloatTensor(static_features).unsqueeze(0)  # (1, 772)

        # 預測
        with torch.no_grad():
            prediction = self.model(ts_tensor, static_tensor)

        predicted_sales = prediction.item()

        return {
            'predicted_sales': predicted_sales,
            'lower_bound': predicted_sales - self.mae,
            'upper_bound': predicted_sales + self.mae,
            'confidence': self.r2
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """返回 Feature Importance（基於 Obj 3 實驗分析）"""
        return {
            'Google Trends': 0.35,
            'CLIP Similarity': 0.30,
            'Season': 0.20,
            'Product Type': 0.15
        }

    def _encode_season(self, season: str) -> np.ndarray:
        """季節 one-hot encoding"""
        season_map = {
            'Spring': [1, 0, 0, 0],
            'Summer': [0, 1, 0, 0],
            'Fall': [0, 0, 1, 0],
            'Winter': [0, 0, 0, 1]
        }
        return np.array(season_map[season])
```

### Existing Pattern Reference

**Plotly 互動圖表：**
```python
import plotly.graph_objects as go

def create_trend_chart(historical_sales, predicted_sales, season):
    """建立歷史趨勢 + 預測圖表"""
    fig = go.Figure()

    # 歷史數據
    fig.add_trace(go.Scatter(
        x=['Q-3', 'Q-2', 'Q-1', 'Q0'],
        y=historical_sales,
        mode='lines+markers',
        name='歷史銷量',
        line=dict(color='blue', width=2)
    ))

    # 預測點
    fig.add_trace(go.Scatter(
        x=['Q+1'],
        y=[predicted_sales],
        mode='markers',
        name='預測銷量',
        marker=dict(color='red', size=12, symbol='star')
    ))

    fig.update_layout(
        title=f'{season} 季度銷量預測',
        xaxis_title='季度',
        yaxis_title='銷量（件）',
        hovermode='x unified',
        height=400
    )

    return fig

# 在 Streamlit 中顯示
st.plotly_chart(fig, use_container_width=True)
```

**Streamlit Metric Display：**
```python
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="預測銷量",
        value=f"{int(predicted_sales):,} 件",
        delta=f"±{int(mae):,} 件"
    )

with col2:
    st.metric(
        label="預測信心度",
        value=f"{int(r2 * 100)}%"
    )

with col3:
    st.metric(
        label="預計營收",
        value=f"${int(predicted_sales * unit_price):,}"
    )
```

### Key Constraints

- **模型載入時間：**
  - Transformer model size: ~10MB
  - 首次載入約 3-5 秒
  - 使用 `@st.cache_resource` 只載入一次

- **預測準確性：**
  - MAE = 327.26（11.5% 誤差率）
  - R² = 0.6788（68.8% 變異解釋力）
  - 信心區間 = [predicted - MAE, predicted + MAE]

- **歷史數據需求：**
  - 需要過去 4 季度的 Google Trends 數據
  - 如無歷史數據，使用模擬數據或顯示警告

---

## Tasks

### Task 1: 實作 Obj 3 API Wrapper (2.5 hrs)
- [ ] 建立 `utils/forecast_predictor.py`
- [ ] 實作 `ForecastPredictorWrapper` 類別
- [ ] 實作模型載入函數（帶 cache）
- [ ] 實作預測函數
- [ ] 實作 Feature Importance 函數
- [ ] 編寫單元測試

### Task 2: 實作 Page 2 基礎結構 (1.5 hrs)
- [ ] 建立 `pages/2_📊_銷量預測.py`
- [ ] 實作頁面佈局（3 個區塊）
- [ ] 實作季節選擇器
- [ ] 實作設計選擇器（讀取 Story 4.2 結果）

### Task 3: 實作預測結果顯示 (2 hrs)
- [ ] 實作 "預測銷量" 按鈕邏輯
- [ ] 整合預測 API 調用
- [ ] 顯示預測結果（metric cards）
- [ ] 顯示誤差範圍和信心度

### Task 4: 實作歷史趨勢圖表 (2 hrs)
- [ ] 實作 Plotly 互動圖表
- [ ] 顯示歷史 4 季度銷量
- [ ] 標註當前預測點
- [ ] 季節性顏色標記
- [ ] 測試圖表互動功能

### Task 5: 實作市場洞察摘要 (2 hrs)
- [ ] 顯示 Feature Importance 長條圖
- [ ] 實作自動建議生成邏輯
  - 最佳上市時機（基於季節因素）
  - 生產數量建議（基於預測值）
  - 競爭程度評估（基於 Trends 分數）
- [ ] 實作風險提示（預測誤差大時）

### Task 6: 測試與優化 (2 hrs)
- [ ] 端到端測試（Story 4.1 → 4.2 → 4.3）
- [ ] 驗證預測結果準確性
- [ ] 執行 Obj 3 regression test
- [ ] 性能優化（cache 驗證）
- [ ] 更新文檔

---

## Definition of Done

### Functionality
- [ ] Page 2 可正常顯示並運作
- [ ] 可基於 Story 4.2 生成的設計預測銷量
- [ ] 歷史趨勢圖表正確顯示
- [ ] 市場洞察建議合理且清晰
- [ ] 測試 3 個季節（Spring/Fall/Winter），均能成功預測

### Integration
- [ ] 完整流程打通（Story 4.1 → 4.2 → 4.3）
- [ ] Session state 正確傳遞
- [ ] Obj 3 原有 CLI 腳本仍可運行（regression test）

### Quality
- [ ] 單元測試通過（`pytest tests/test_forecast_predictor.py`）
- [ ] 預測結果與 Obj 3 原始模型一致（誤差 < 1%）
- [ ] 模型載入使用 cache（驗證不重複載入）
- [ ] 預測時間 < 3 秒

### Documentation
- [ ] `utils/forecast_predictor.py` 函數有完整註解
- [ ] `obj4_web_app/README.md` 更新完整使用流程
- [ ] 主 `README.md` 更新 Objective 4 完成狀態

---

## Testing Scenarios

### Scenario 1: 春節主題預測（高峰季）
**前置條件：**
- Story 4.1: 春節主題 Prompt 已生成
- Story 4.2: 4 張春節設計圖已生成（CLIP ≥ 0.80）

**操作：**
1. 導航至 Page 2
2. 選擇季節: Winter
3. 選擇設計: 最高 CLIP 相似度的圖片
4. 點擊 "預測銷量"

**預期結果：**
- 預測銷量: 1,600-1,900 件（冬季高峰）
- 信心度: 68%
- 誤差範圍: ±327 件
- 市場洞察: "冬季是最佳上市時機"
- 歷史趨勢圖顯示季節性波動

### Scenario 2: 多設計對比預測
**操作：**
1. 選擇季節: Spring
2. 選擇設計: 圖片 1 和圖片 3（CLIP 相似度不同）
3. 點擊 "預測銷量"

**預期結果：**
- 顯示 2 個預測結果（side-by-side）
- CLIP 相似度高的設計預測銷量較高
- 對比差異清晰顯示
- 建議選擇較高預測值的設計

### Scenario 3: 無歷史數據處理
**操作：** 清除 session state 中的 `trends_data`

**預期結果：**
- 顯示警告訊息："無歷史趨勢數據，使用模擬數據"
- 預測仍可執行（使用平均值）
- 信心度顯示為較低（如 50%）

### Scenario 4: 模型載入失敗
**操作：** 模擬模型權重檔案不存在

**預期結果：**
- 顯示錯誤訊息："模型載入失敗，請檢查模型檔案"
- 提供診斷資訊（檔案路徑）
- Streamlit app 不 crash

---

## Dev Notes

### 模型載入路徑
```python
# config.py
MODEL_PATH = "models/transformer_lulu/best_model.pth"

# 檢查模型存在
import os
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型檔案不存在: {MODEL_PATH}")
```

### 模擬歷史數據（如無真實數據）
```python
def get_mock_historical_sales(season: str) -> List[float]:
    """生成模擬歷史數據（用於 Demo）"""
    base = 1000
    seasonal_multiplier = {
        'Spring': 1.1,
        'Summer': 0.9,
        'Fall': 1.0,
        'Winter': 1.3
    }
    multiplier = seasonal_multiplier.get(season, 1.0)

    # 過去 4 季度
    return [
        base * 0.9 * multiplier,
        base * 1.0 * multiplier,
        base * 1.1 * multiplier,
        base * 1.05 * multiplier
    ]
```

### Feature Importance 視覺化
```python
import plotly.express as px

def plot_feature_importance(importance_dict):
    """繪製 Feature Importance 長條圖"""
    fig = px.bar(
        x=list(importance_dict.keys()),
        y=list(importance_dict.values()),
        labels={'x': '特徵', 'y': '重要性（%）'},
        title='銷量影響因素分析',
        color=list(importance_dict.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=300)
    return fig
```

### 測試指令
```bash
# 單元測試
pytest tests/test_forecast_predictor.py -v

# Obj 3 Regression Test
python obj3_lstm_forecast/test_local_original.py

# 完整端到端測試
streamlit run obj4_web_app/app.py
# 手動測試 Story 4.1 → 4.2 → 4.3 完整流程
```

---

## Agent Model Used
*將由 Developer Agent 填寫*

---

## Dev Agent Record

### Debug Log References
*將由 Developer Agent 記錄*

### Completion Notes
*將由 Developer Agent 填寫*

### File List
*將由 Developer Agent 維護*

### Change Log
*將由 Developer Agent 記錄*

---

**Story Status:** Draft
**Next Action:** 等待 Story 4.2 完成後開始實作
