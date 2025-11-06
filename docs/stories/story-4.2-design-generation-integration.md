# Story 4.2: Obj 2 設計生成與 CLIP 驗證整合

**Story ID:** STORY-4.2
**Epic:** EPIC-004 - Streamlit Web Application Integration
**Status:** ✅ Done
**Priority:** High
**Points:** 8
**Created:** 2025-11-06
**Completed:** 2025-11-06
**Assigned To:** Developer (James)
**Depends On:** STORY-4.1

---

## User Story

**As a** ToyzeroPlus 設計團隊成員，
**I want** 基於生成的 Prompt 自動生成 4 張角色設計圖並查看角色一致性分數，
**So that** 我可以快速獲得多個設計變化並選擇最符合品牌形象的設計。

---

## Story Context

### Existing System Integration

**整合對象：** Objective 2 - Google Gemini Design Generation

**核心模組：**
- `obj2_midjourney_api/google_gemini_client.py` - Google Gemini API 整合
- `obj2_midjourney_api/character_focused_validator.py` - CLIP 相似度驗證
- `obj2_midjourney_api/gemini_image_client.py` - 圖片生成 client

**技術棧：**
- Google Gemini 2.5 Flash Image API
- Transformers 4.30+ (CLIP Model: `openai/clip-vit-large-patch14`)
- PyTorch 2.0+ (CLIP inference)
- Pillow 10.0+ (圖片處理)

**整合模式：**
- 從 Story 4.1 的 session state 讀取生成的 Prompt
- 透過 `utils/design_generator.py` wrapper 調用 Obj 2 API
- 顯示生成圖片 + CLIP 相似度分數
- 儲存圖片至 session state 供 Story 4.3 使用

**現有 Touch Points：**
- `google_gemini_client.generate_image(prompt, reference_image_url)` - 圖片生成
- `character_focused_validator.compute_clip_similarity(img1, img2)` - CLIP 驗證
- Reference Image 儲存於 `data/reference_images/lulu_pig_ref_*.jpg`

---

## Acceptance Criteria

### Functional Requirements

**FR1: 圖片生成介面設計**
- [ ] 在 Page 1 新增 "生成設計圖" 區塊（位於 Prompt 顯示區下方）
- [ ] "生成 4 張設計圖" 按鈕（僅在 Prompt 生成後啟用）
- [ ] Reference Image 選擇器（下拉選單，可選擇 `lulu_pig_ref_1/2/3.jpg`）
- [ ] 生成參數設定（可摺疊區塊）：
  - 圖片數量（預設 4，可調整 1-4）
  - 生成模式（預設 "fast"）

**FR2: 圖片生成流程**
- [ ] 點擊 "生成" 按鈕後顯示 progress bar（0-100%）
- [ ] 顯示預計時間（約 11.18s × 張數）
- [ ] 逐張顯示生成結果（不等全部完成）
- [ ] 每張圖片下方顯示：
  - 圖片編號（變化 1/2/3/4）
  - CLIP 相似度分數（vs Reference Image）
  - "下載" 按鈕

**FR3: CLIP 相似度驗證**
- [ ] 自動計算每張生成圖 vs Reference Image 的 CLIP 相似度
- [ ] 相似度 ≥ 0.80 顯示為綠色 ✅
- [ ] 相似度 < 0.80 顯示為黃色 ⚠️
- [ ] 顯示平均相似度（4 張圖片平均）

**FR4: 圖片下載功能**
- [ ] 每張圖片有獨立 "下載" 按鈕
- [ ] 檔案命名格式：`{character_name}_{theme}_{timestamp}_var{n}.png`
- [ ] "下載全部" 按鈕（ZIP 壓縮包）

**FR5: 錯誤處理**
- [ ] Google Gemini API 失敗顯示錯誤訊息
- [ ] API Quota 超過時提示並中止
- [ ] 網路超時（60 秒）自動 retry（最多 3 次）
- [ ] 部分圖片失敗時顯示成功數量（如：3/4 張成功）

### Integration Requirements

**IR1: Obj 2 API 封裝**
- [ ] 建立 `utils/design_generator.py` wrapper
- [ ] 實作 `generate_designs(prompt, reference_image_path, num_images)` 函數
- [ ] 實作 `compute_similarity(generated_image, reference_image)` 函數
- [ ] 實作 `save_image(image, filename)` 函數

**IR2: Session State 管理**
- [ ] 從 `st.session_state['generated_prompt']` 讀取 Prompt（Story 4.1）
- [ ] 儲存生成圖片至 `st.session_state['generated_images']`（供 Story 4.3）
- [ ] 儲存 CLIP embeddings 至 `st.session_state['clip_embeddings']`（供 Story 4.3）

**IR3: 現有功能保留**
- [ ] Obj 2 CLI 腳本仍可獨立運行
- [ ] `google_gemini_client.py` 的 API 不被修改
- [ ] 現有測試腳本（如 `test_comparison_google_gemini.py`）仍能執行

### Quality Requirements

**QR1: 性能優化**
- [ ] 使用 `@st.cache_resource` 快取 CLIP model 載入
- [ ] 圖片生成使用 async/await（如可行）
- [ ] 避免重複生成相同 Prompt（檢查 session state）

**QR2: 用戶體驗**
- [ ] Progress bar 即時更新（每張圖完成後 +25%）
- [ ] 生成過程中禁用其他操作按鈕
- [ ] 完成後顯示成功通知（`st.success()`）

**QR3: 錯誤處理**
- [ ] 所有 Google Gemini API 錯誤有 try-except
- [ ] Retry 機制帶 exponential backoff
- [ ] 錯誤訊息使用繁體中文

**QR4: 測試覆蓋**
- [ ] 為 `utils/design_generator.py` 編寫單元測試
- [ ] Mock Google Gemini API 測試錯誤處理
- [ ] 執行 Obj 2 regression test

---

## Technical Notes

### Integration Approach

**Wrapper 設計模式：**
```python
# utils/design_generator.py
from obj2_midjourney_api.google_gemini_client import GeminiImageClient
from obj2_midjourney_api.character_focused_validator import CharacterValidator

class DesignGeneratorWrapper:
    def __init__(self):
        self.client = GeminiImageClient()
        self.validator = CharacterValidator()
        self._clip_model = None

    @property
    def clip_model(self):
        """Lazy load CLIP model (cached)"""
        if self._clip_model is None:
            self._clip_model = self.validator.load_clip_model()
        return self._clip_model

    def generate_designs(
        self,
        prompt: str,
        reference_image_path: str,
        num_images: int = 4,
        progress_callback: callable = None
    ) -> List[Dict[str, Any]]:
        """
        生成設計圖並計算 CLIP 相似度

        Returns:
            List[Dict]: [
                {
                    'image': PIL.Image,
                    'clip_similarity': float,
                    'generation_time': float
                },
                ...
            ]
        """
        results = []
        ref_image = Image.open(reference_image_path)

        for i in range(num_images):
            try:
                # 生成圖片
                generated_image = self.client.generate_image(
                    prompt=prompt,
                    reference_image=reference_image_path
                )

                # 計算 CLIP 相似度
                similarity = self.validator.compute_clip_similarity(
                    generated_image, ref_image
                )

                results.append({
                    'image': generated_image,
                    'clip_similarity': similarity,
                    'generation_time': 11.18  # 平均時間
                })

                # 進度回調
                if progress_callback:
                    progress_callback((i + 1) / num_images)

            except Exception as e:
                # 記錄失敗但繼續
                results.append({'error': str(e)})

        return results
```

### Existing Pattern Reference

**Streamlit Image Display：**
```python
# 顯示 4 張圖片（2x2 grid）
cols = st.columns(2)
for i, result in enumerate(generated_images):
    col = cols[i % 2]
    with col:
        st.image(result['image'], caption=f"變化 {i+1}")

        # CLIP 相似度顯示
        similarity = result['clip_similarity']
        color = "green" if similarity >= 0.80 else "orange"
        st.markdown(f"<span style='color:{color}'>CLIP: {similarity:.4f}</span>", unsafe_allow_html=True)

        # 下載按鈕
        img_bytes = image_to_bytes(result['image'])
        st.download_button(
            label="下載",
            data=img_bytes,
            file_name=f"design_{i+1}.png",
            mime="image/png"
        )
```

**Progress Bar：**
```python
progress_bar = st.progress(0)
status_text = st.empty()

def update_progress(progress):
    progress_bar.progress(progress)
    status_text.text(f"生成中... {int(progress * 100)}%")

# 在生成函數中調用
generate_designs(prompt, ref_img, progress_callback=update_progress)
```

### Key Constraints

- **API Quota 限制：**
  - Google Gemini Flash Image: 免費 tier 有每分鐘 15 次限制
  - 需實作 rate limiting（如超過則延遲）

- **圖片大小限制：**
  - 生成圖片約 1024x1024 px
  - Streamlit 顯示時自動縮放
  - 下載原始解析度

- **CLIP Model 載入：**
  - Model size: ~1.7GB（CLIP-ViT-Large）
  - 使用 `@st.cache_resource` 避免重複載入
  - 首次載入約 5-10 秒

---

## Tasks

### Task 1: 實作 Obj 2 API Wrapper (2 hrs)
- [ ] 建立 `utils/design_generator.py`
- [ ] 實作 `DesignGeneratorWrapper` 類別
- [ ] 實作圖片生成函數（帶 progress callback）
- [ ] 實作 CLIP 相似度計算函數
- [ ] 編寫單元測試

### Task 2: 實作圖片生成 UI (2 hrs)
- [ ] 在 Page 1 新增 "生成設計圖" 區塊
- [ ] 實作 Reference Image 選擇器
- [ ] 實作 "生成" 按鈕邏輯
- [ ] 實作 Progress Bar 和狀態顯示

### Task 3: 實作圖片顯示與 CLIP 驗證 (2 hrs)
- [ ] 實作 2x2 grid 圖片顯示
- [ ] 顯示 CLIP 相似度分數（顏色標示）
- [ ] 顯示平均相似度
- [ ] 實作圖片快取（避免重複生成）

### Task 4: 實作下載功能 (1 hr)
- [ ] 實作單張圖片下載按鈕
- [ ] 實作 "下載全部" ZIP 功能
- [ ] 檔案命名規則實作
- [ ] 測試下載功能

### Task 5: 錯誤處理與優化 (1.5 hrs)
- [ ] 實作 Google Gemini API 錯誤處理
- [ ] 實作 retry 機制（exponential backoff）
- [ ] 實作 rate limiting（避免超過 quota）
- [ ] 優化 CLIP model 載入（cache）

### Task 6: 測試與整合 (1.5 hrs)
- [ ] 端到端測試（Story 4.1 → Story 4.2）
- [ ] 執行 Obj 2 regression test
- [ ] 測試錯誤處理路徑
- [ ] 更新文檔

---

## Definition of Done

### Functionality
- [ ] 可基於 Story 4.1 生成的 Prompt 生成 4 張設計圖
- [ ] CLIP 相似度正確計算並顯示（目標 ≥ 0.80）
- [ ] 圖片可下載（單張 + 全部）
- [ ] 測試 3 組不同 Prompt，均能成功生成

### Integration
- [ ] Obj 2 API wrapper 測試通過
- [ ] Session state 正確傳遞（Story 4.1 → 4.2 → 4.3）
- [ ] Obj 2 原有 CLI 腳本仍可運行（regression test）

### Quality
- [ ] 單元測試通過（`pytest tests/test_design_generator.py`）
- [ ] CLIP model 載入使用 cache（驗證不重複載入）
- [ ] 錯誤處理覆蓋所有 API 調用

### Documentation
- [ ] `utils/design_generator.py` 函數有完整註解
- [ ] `obj4_web_app/README.md` 更新使用說明
- [ ] 錄製 Demo 影片（選填）

---

## Testing Scenarios

### Scenario 1: 正常生成流程
**前置條件：** Story 4.1 已生成春節主題 Prompt

**操作：**
1. 選擇 Reference Image: `lulu_pig_ref_1.jpg`
2. 點擊 "生成 4 張設計圖"
3. 等待生成完成

**預期結果：**
- Progress bar 從 0% → 100%
- 顯示 4 張設計圖（2x2 grid）
- 每張圖 CLIP 相似度 ≥ 0.80（綠色 ✅）
- 平均相似度 ≥ 0.80
- 可下載每張圖片

### Scenario 2: 部分失敗處理
**操作：** 模擬第 2 張圖片生成失敗（網路超時）

**預期結果：**
- 顯示 3/4 張成功訊息
- 失敗的圖片位置顯示錯誤訊息
- 其他 3 張圖片正常顯示
- 提供 "重試失敗圖片" 按鈕

### Scenario 3: API Quota 超過
**操作：** 連續生成 2 組（共 8 張圖），超過 Gemini 免費 quota

**預期結果：**
- 第 2 組生成時顯示 quota 錯誤
- 錯誤訊息包含預計恢復時間
- Streamlit app 不 crash

### Scenario 4: CLIP 相似度低於門檻
**操作：** 使用不匹配的 Reference Image（如貓的圖片）

**預期結果：**
- 生成成功但 CLIP 相似度 < 0.80
- 分數顯示為黃色 ⚠️
- 提示用戶選擇其他 Reference Image

---

## Dev Notes

### Google Gemini API 設定
```python
# config.py
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 生成配置
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,
}
```

### CLIP Model Cache
```python
# Streamlit cache for CLIP model
@st.cache_resource
def load_clip_model():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor
```

### 測試指令
```bash
# 單元測試
pytest tests/test_design_generator.py -v

# Obj 2 Regression Test
python obj2_midjourney_api/test_comparison_google_gemini.py
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
**Next Action:** 等待 Story 4.1 完成後開始實作
