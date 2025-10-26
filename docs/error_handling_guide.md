# PromptGenerator Error Handling 錯誤處理機制

**版本：** 1.1
**作者：** Product Manager (John)
**更新日期：** 2025-10-26

---

## 📋 功能概述

`PromptGenerator` 現在具備完善的錯誤處理機制，能夠：

✅ **捕獲並解析 API 錯誤訊息**
✅ **檢測 API quota/usage limit 錯誤**
✅ **顯示詳細錯誤資訊**
✅ **自動 fallback 至備用 prompt**
✅ **支援 chatanywhere API 錯誤格式**

---

## 🔧 錯誤類型處理

### 1. API Quota/Usage Limit 錯誤

**觸發條件：**
- API 使用次數超過限制
- API 餘額不足
- Rate limit 超過

**錯誤訊息範例：**
```
❌ Error generating prompt for Halloween: [insufficient_quota] 429: You exceeded your current quota
⚠️  API quota/usage limit reached!
   Please check your API credits or wait for quota reset.
```

**行為：**
- 停止執行（raise exception）
- 不使用 fallback prompt
- 提示用戶檢查 API credits

### 2. ChatAnywhere 格式錯誤

**錯誤 Response Body：**
```json
{
    "error": {
        "message": "Unexpected character ('}' (code 125)): was expecting double-quote to start field name【如果您遇到问题，欢迎加入QQ群咨询：1048463714】",
        "type": "chatanywhere_error",
        "param": null,
        "code": "400 BAD_REQUEST"
    }
}
```

**錯誤訊息範例：**
```
❌ Error generating prompt for Halloween: [chatanywhere_error] 400 BAD_REQUEST: Unexpected character...
⚠️  Using fallback prompt for Halloween
```

**行為：**
- 顯示詳細錯誤訊息
- 自動使用 fallback prompt
- 繼續執行（不中斷）

### 3. 一般 API 錯誤

**常見錯誤：**
- Invalid model specified
- Connection timeout
- Network error
- Invalid request format

**行為：**
- 顯示錯誤訊息
- 自動使用 fallback prompt
- 繼續執行

---

## 📊 錯誤解析機制

### `_parse_api_error(exception)` 方法

**功能：** 從 exception 中提取錯誤詳情

**支援格式：**

1. **OpenAI API Error (有 response 屬性)**
```python
exception.response.json() = {
    "error": {
        "message": "...",
        "type": "...",
        "code": "..."
    }
}
```

2. **Standard Exception (有 body 屬性)**
```python
exception.body = {
    "error": {
        "message": "...",
        "type": "...",
        "code": "..."
    }
}
```

3. **Simple Exception**
```python
str(exception)
```

**輸出格式：**
```
[{type}] {code}: {message}
```

### `_is_quota_error(exception)` 方法

**功能：** 檢測是否為 quota/limit 錯誤

**檢測關鍵字：**
- `quota`
- `rate limit`
- `usage limit`
- `insufficient`
- `exceeded`
- `too many requests`
- `429`
- `billing`
- `credit`

**返回值：**
- `True`: 確認為 quota error
- `False`: 非 quota error

---

## 🛠️ 使用範例

### 正常使用（有錯誤處理）

```python
from obj1_nlp_prompt.prompt_generator import PromptGenerator

generator = PromptGenerator()

try:
    # 生成 prompt
    prompt = generator.generate_prompt(
        theme="Halloween",
        keywords=["pumpkin", "costume", "spooky"]
    )
    print(f"Generated: {prompt}")

except Exception as e:
    # Quota error 會在這裡被捕獲
    print(f"Failed: {e}")
    print("Please check your API credits!")
```

### Fallback Prompt

當 API 錯誤（非 quota error）時，自動生成 fallback prompt：

```python
# API 失敗時
# 自動返回：
"Lulu Pig, adorable pink pig character, celebrating Halloween with pumpkin,
costume, spooky, cute kawaii style, vibrant colors, cheerful mood,
merchandise-ready design, character illustration"
```

---

## 📝 Log 輸出範例

### 成功情況
```
INFO - Generated prompt for Halloween in 1.85s
```

### API 錯誤（非 Quota）
```
ERROR - ❌ Error generating prompt for Halloween: [chatanywhere_error] 400 BAD_REQUEST: Invalid format
WARNING - Using fallback prompt for Halloween
```

### Quota 錯誤
```
ERROR - ❌ Error generating prompt for Halloween: [insufficient_quota] 429: Quota exceeded
ERROR - ⚠️  API quota/usage limit reached!
ERROR -    Please check your API credits or wait for quota reset.
Traceback (most recent call last):
  ...
Exception: [insufficient_quota] 429: Quota exceeded
```

---

## 📊 API Usage Limits (Free Tier)

**GPT_API_free 免費版使用限制：**

### High-Performance Models (5 requests/day)
- `gpt-5`
- `gpt-4o`
- `gpt-4.1`

### DeepSeek Models (30 requests/day)
- `deepseek-r1`
- `deepseek-v3`
- `deepseek-v3-2-exp`

### Standard Models (200 requests/day) ⭐ Recommended
- `gpt-4o-mini`
- `gpt-3.5-turbo`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-5-mini` ✅ Currently configured in `.env`
- `gpt-5-nano`

**建議配置：**
```bash
# .env
GPT_API_FREE_MODEL=gpt-5-mini  # 200次/天，適合開發測試
```

**生產環境建議：**
- 使用 `gpt-3.5-turbo` 或 `gpt-5-mini`（200次/天額度）
- 28 個 prompts 生成約需 28 次 API calls
- 每日可重新生成 ~7 次（28 × 7 = 196 < 200）

**高質量需求：**
- 使用 `gpt-4o` 或 `gpt-5`（5次/天額度）
- 需分批生成或升級付費版

---

## 🔍 Debug 模式

如需查看詳細錯誤訊息，設置 logging level：

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

---

## ⚙️ 環境變數配置

**`.env` 檔案：**
```bash
# GPT API Configuration
GPT_API_FREE_KEY=your_api_key_here
GPT_API_FREE_BASE_URL=https://api.chatanywhere.org/v1
GPT_API_FREE_MODEL=gpt-3.5-turbo  # 或 gpt-4, gpt-5-mini

# Retry Configuration (可選)
GPT_API_MAX_RETRIES=3
GPT_API_RETRY_DELAY=2
```

---

## 🐛 常見問題

### Q1: 收到 "chatanywhere_error" 錯誤怎麼辦？

**A:** 這通常是請求格式問題。檢查：
1. Model 名稱是否正確（`.env` 中的 `GPT_API_FREE_MODEL`）
2. API key 是否有效
3. 是否需要更新 openai library

```bash
pip install --upgrade openai
```

### Q2: 如何處理 Quota 錯誤？

**A:** Quota 錯誤會立即停止執行。需要：
1. 檢查 API 餘額
2. 等待 quota 重置（通常每日/每月）
3. 升級 API plan

### Q3: Fallback prompt 質量如何？

**A:** Fallback prompt 是預設模板，質量較低。建議：
1. 確保 API 正常運作
2. 檢查 API credits
3. 使用穩定的 model (gpt-3.5-turbo)

### Q4: 如何自定義錯誤處理？

**A:** 繼承 `PromptGenerator` 並覆寫方法：

```python
class CustomPromptGenerator(PromptGenerator):
    def _parse_api_error(self, exception):
        # 自定義錯誤解析
        return f"Custom error: {exception}"

    def _is_quota_error(self, exception):
        # 自定義 quota 檢測
        return "my_quota_keyword" in str(exception).lower()
```

---

## 📊 測試

執行錯誤處理測試：

```bash
source .venv/bin/activate
python obj1_nlp_prompt/test_error_handling.py
```

**測試涵蓋：**
- ✅ API 錯誤解析
- ✅ Quota 錯誤檢測
- ✅ 錯誤訊息格式化
- ✅ Fallback prompt 生成

---

## 🔄 版本歷史

**v1.1** (2025-10-26)
- ✅ 添加完整錯誤處理機制
- ✅ 支援 chatanywhere API 錯誤格式
- ✅ Quota error 自動檢測
- ✅ 詳細錯誤訊息顯示

**v1.0** (2025-10-26)
- ✅ 初始版本
- ✅ 基本 prompt 生成功能

---

**維護者：** Product Manager (John)
**支援：** FYP-RoleMarket Project
