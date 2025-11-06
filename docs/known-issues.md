# 已知問題清單 (Known Issues)

**專案：** FYP-RoleMarket - AI 角色設計與需求預測系統
**最後更新：** 2025-11-06
**狀態：** 待收集手動測試問題

---

## 📋 問題收集流程

### 如何報告問題
1. 在本文件中新增問題
2. 標明優先級（P0/P1/P2）
3. 提供詳細步驟重現
4. 附上錯誤訊息或截圖（如有）

### 優先級定義
- **P0 (Critical):** 系統崩潰或核心功能完全無法使用
- **P1 (High):** 主要功能有問題，影響用戶體驗
- **P2 (Medium):** 次要功能問題或 UI/UX 改進
- **P3 (Low):** 優化建議或美化

---

## 🔴 P0 - Critical Issues

### 模板（待填入）
```markdown
### [P0-001] 問題標題

**發現日期：** YYYY-MM-DD
**發現者：** XXX
**狀態：** Open / In Progress / Fixed

**問題描述：**
簡短描述問題

**重現步驟：**
1. 步驟 1
2. 步驟 2
3. 步驟 3

**預期行為：**
應該發生什麼

**實際行為：**
實際發生什麼

**錯誤訊息：**
```
貼上錯誤訊息
```

**影響範圍：**
- 影響的功能
- 影響的用戶數

**暫時解決方案：**
（如有）

**修復計劃：**
預計修復時間和方案
```

---

## 🟠 P1 - High Priority Issues

*（待填入手動測試發現的問題）*

---

## 🟡 P2 - Medium Priority Issues

### [P2-001] CLIP Embedding 未實際提取

**發現日期：** 2025-11-06
**發現者：** Developer (James)
**狀態：** Known Limitation

**問題描述：**
Story 4.3 銷量預測中使用的 CLIP embedding 是模擬數據，而非從實際圖片提取。

**影響範圍：**
- Page 2: 銷量預測功能
- 預測準確度可能受影響（使用隨機向量 × similarity）

**當前實作：**
```python
# obj4_web_app/pages/2_📊_銷量預測.py:181
clip_embedding = np.random.rand(768) * clip_similarity
```

**建議修復方案：**
1. 在 Story 4.2 的 `design_generator.py` 中使用 `validator.model.encode_image()` 提取實際 embedding
2. 儲存至 `st.session_state['clip_embeddings']`
3. Story 4.3 直接讀取實際 embedding

**優先級理由：**
雖然不影響系統運行，但影響預測準確度，應在部署前修復。

---

### [P2-002] Google Trends Rate Limiting 可能影響用戶體驗

**發現日期：** 2025-11-06
**發現者：** Developer (James)
**狀態：** Known Limitation

**問題描述：**
連續提取多個主題的 Google Trends 數據時，每次需要 sleep 2 秒以避免 429 錯誤。

**影響範圍：**
- Page 1: Google Trends 自動提取功能
- 用戶需等待 ~2 秒每次提取

**當前實作：**
```python
# obj1_nlp_prompt/trends_extractor.py:134
time.sleep(2)  # Rate limiting
```

**建議改進方案：**
1. 加入 Loading spinner 提示用戶等待
2. 使用 cache 機制（同一主題 1 小時內不重複提取）
3. 加入 rate limit 提示訊息

**優先級理由：**
不影響功能，但可改善用戶體驗。

---

## 🟢 P3 - Low Priority Issues / Enhancements

### [P3-001] Session State 依賴鏈可優化

**發現日期：** 2025-11-06
**發現者：** Developer (James)
**狀態：** Enhancement

**問題描述：**
Page 2 依賴 Page 1 的 session state，用戶直接訪問 Page 2 時會顯示警告。

**當前實作：**
- 已有前置檢查（Line 77-86）
- 顯示警告訊息引導用戶

**建議改進方案：**
1. 加入「快速開始」按鈕直接跳轉至 Page 1
2. 加入進度條顯示當前步驟（Step 1/2/3）
3. 加入「載入範例數據」功能供快速測試

**優先級理由：**
已有基本檢查，屬於 UX 優化。

---

## 📊 問題統計

| 優先級 | 待修復 | 進行中 | 已修復 | 總計 |
|--------|--------|--------|--------|------|
| P0 (Critical) | 0 | 0 | 0 | 0 |
| P1 (High) | 0 | 0 | 0 | 0 |
| P2 (Medium) | 2 | 0 | 0 | 2 |
| P3 (Low) | 1 | 0 | 0 | 1 |
| **總計** | **3** | **0** | **0** | **3** |

---

## 📅 修復計劃

### Sprint 1: Bug Fixes（預計 2-3 天）
- [ ] 收集完整手動測試問題清單
- [ ] 修復所有 P0 issues
- [ ] 修復主要 P1 issues

### Sprint 2: Improvements（預計 1-2 天）
- [ ] 修復 P2-001: CLIP Embedding 實際提取
- [ ] 修復 P2-002: Rate Limiting 改進
- [ ] 補充 edge case 測試

### Sprint 3: Enhancements（低優先）
- [ ] P3 issues 根據時間決定是否修復
- [ ] UI/UX 優化
- [ ] 性能優化

---

## 🔍 測試建議

### 手動測試 Checklist

#### Page 1: 設計生成
- [ ] **Obj 1: Trend Analysis**
  - [ ] 手動輸入關鍵字 → 生成 Prompt
  - [ ] Google Trends 自動提取 → 選擇關鍵字 → 生成 Prompt
  - [ ] 空關鍵字錯誤處理
  - [ ] 特殊字符處理
  - [ ] 長關鍵字列表（>10 個）

- [ ] **Obj 2: Image Generation**
  - [ ] 生成 1 張圖片
  - [ ] 生成 4 張圖片
  - [ ] CLIP 相似度顯示正確
  - [ ] 下載按鈕功能正常
  - [ ] Reference Image 選擇器正常
  - [ ] API 錯誤處理（無效 API key）

#### Page 2: 銷量預測
- [ ] **前置檢查**
  - [ ] 直接訪問 Page 2（未生成圖片）→ 應顯示警告
  - [ ] 完成 Page 1 後訪問 Page 2 → 正常顯示

- [ ] **Obj 3: Forecast**
  - [ ] 選擇季節 → 輸入 Trends → 選擇設計 → 預測
  - [ ] 預測結果顯示正確（銷量、信心度、誤差）
  - [ ] Plotly 圖表正常顯示
  - [ ] Feature Importance 顯示正確
  - [ ] 市場洞察建議合理

#### Cross-Page Flow
- [ ] Page 1 生成 Prompt → Page 1 生成圖片 → Page 2 預測銷量
- [ ] 返回 Page 1 重新生成 → Page 2 數據更新
- [ ] Session state 正確維護

---

## 📝 報告新問題

**請按照以下格式報告新問題：**

```markdown
### [P?-XXX] 問題標題

**發現日期：** 2025-11-XX
**發現者：** [你的名字]
**狀態：** Open

**問題描述：**
[詳細描述]

**重現步驟：**
1. [步驟 1]
2. [步驟 2]

**預期行為：**
[應該發生什麼]

**實際行為：**
[實際發生什麼]

**錯誤訊息：**
```
[貼上錯誤訊息]
```

**影響範圍：**
[哪些功能受影響]
```

---

**下次更新：** 收集手動測試問題後
