# Story 3.2: --cref Parameter Testing Guide

**版本：** 1.0
**作者：** Product Manager (John)
**更新日期：** 2025-10-26

---

## 📋 測試目標

驗證 TTAPI Midjourney API 的 `--cref`（Character Reference）參數能夠：
- 保持角色 IP 一致性
- 支援不同權重值（0-100）
- 支援多個參考圖片
- 與其他 Midjourney 參數兼容

---

## 🎯 測試計劃

### 測試案例

| # | 測試名稱 | --cref | --cw | 預期結果 | 成本 |
|---|---------|--------|------|---------|------|
| 1 | Baseline (No --cref) | None | N/A | Lulu Pig with Halloween theme (may vary) | $0.40 |
| 2 | Low Weight | 1 ref | 50 | Partial character consistency | $0.40 |
| 3 | Medium Weight | 1 ref | 75 | Good character consistency | $0.40 |
| 4 | High Weight | 1 ref | 100 | Strong character consistency | $0.40 |
| 5 | Multiple References | 2 refs | 100 | Averaged character features | $0.40 |

**總成本：** $2.00 (5 images)

---

## 📝 測試前準備

### 1. 上傳 Reference Images 到 Discord

**為什麼要用 Discord？**
- Midjourney 官方推薦
- 穩定的 CDN（https://cdn.discordapp.com/）
- 永久 URL（不會過期）
- 免費

**步驟：**

1. **加入任何 Discord 服務器** 或創建私人頻道

2. **上傳 Reference Images：**
   - `data/reference_images/lulu_pig_ref_1.png`
   - `data/reference_images/lulu_pig_ref_2.png`

3. **獲取 CDN URLs：**
   - 上傳後，在圖片上右鍵 → "Copy Image Address"
   - URL 格式：`https://cdn.discordapp.com/attachments/[channel_id]/[file_id]/[filename]`
   - 範例：`https://cdn.discordapp.com/attachments/123456789/987654321/lulu_pig_ref_1.png`

4. **驗證 URLs：**
   ```bash
   # 測試 URL 是否可訪問
   curl -I "YOUR_DISCORD_CDN_URL"
   # 應該返回 200 OK
   ```

### 2. 更新測試腳本

編輯 `obj2_midjourney_api/test_cref.py`：

```python
# Line ~60: Update CREF_URLS
CREF_URLS = [
    "https://cdn.discordapp.com/attachments/.../lulu_pig_ref_1.png",  # 替換為實際 URL
    "https://cdn.discordapp.com/attachments/.../lulu_pig_ref_2.png"   # 替換為實際 URL
]
```

### 3. 驗證 API Key

確認 `.env` 中的 TTAPI_API_KEY 有效：

```bash
echo $TTAPI_API_KEY  # 或直接查看 .env
# 應該顯示: c14155db-6ea4-74cc-dffa-fb55416a8fa0
```

---

## 🚀 執行測試

### Dry Run（推薦先執行）

```bash
source .venv/bin/activate
python obj2_midjourney_api/test_cref.py
```

**檢查輸出：**
- ✅ 2 個 reference images 已找到
- ✅ 5 個測試案例已列出
- ✅ 總成本估算正確（$2.00）

### 實際執行（消耗 API credits）

```bash
source .venv/bin/activate
python obj2_midjourney_api/test_cref.py --actual-run
```

**執行時間：** 約 5-10 分鐘（每張圖 1-2 分鐘）

---

## 📊 預期結果

### 測試輸出

```
################################################################################
# TTAPI --cref Parameter Test Suite
################################################################################

================================================================================
Checking Reference Images
================================================================================

✅ Found: lulu_pig_ref_1.png (241.1 KB)
✅ Found: lulu_pig_ref_2.png (195.7 KB)

Total: 2 reference images

🚀 EXECUTING ACTUAL API CALLS

⚠️  This will cost approximately $0.40 per image

--------------------------------------------------------------------------------
Test 1/5: Baseline (No --cref)
--------------------------------------------------------------------------------

INFO:obj2_midjourney_api.ttapi_client:Submitting imagine task...
INFO:obj2_midjourney_api.ttapi_client:Task submitted: task_abc123
INFO:obj2_midjourney_api.ttapi_client:Task task_abc123 status: processing (10s)
INFO:obj2_midjourney_api.ttapi_client:Task task_abc123 completed
INFO:obj2_midjourney_api.ttapi_client:Downloading image to: data/generated_images/test_baseline_no_cref.png
INFO:obj2_midjourney_api.ttapi_client:Image saved: data/generated_images/test_baseline_no_cref.png
INFO:obj2_midjourney_api.ttapi_client:Image generated in 45.23s (Cost: $0.4)

✅ Test 1 completed successfully
   Task ID: task_abc123
   Duration: 45.23s
   Image: data/generated_images/test_baseline_no_cref.png
   Cost: $0.4

[... 測試 2-5 類似 ...]

================================================================================
Test Summary
================================================================================

✅ PASS  Test 1: Baseline (No --cref)
✅ PASS  Test 2: Low Weight (--cw 50)
✅ PASS  Test 3: Medium Weight (--cw 75)
✅ PASS  Test 4: High Weight (--cw 100)
✅ PASS  Test 5: Multiple References (--cw 100)

Passed: 5/5
Failed: 0/5

Total Cost: $2.0
Images Generated: 5
```

### 生成的圖片

測試完成後，檢查 `data/generated_images/` 目錄：

```bash
ls -lh data/generated_images/test_*.png
```

應該包含：
- `test_baseline_no_cref.png` - 無參考圖（基準）
- `test_cref_weight_50.png` - 低權重
- `test_cref_weight_75.png` - 中權重
- `test_cref_weight_100.png` - 高權重
- `test_cref_multiple_refs.png` - 多參考圖

---

## 🔍 結果分析

### 視覺比較

手動檢查生成的圖片，比較：

1. **Baseline vs. --cref 100**
   - Baseline 應該有更多變化
   - --cref 100 應該保持 Lulu Pig 的核心特徵

2. **不同權重 (50 vs. 75 vs. 100)**
   - Weight 越高，角色一致性越強
   - Weight 越低，創意自由度越高

3. **單參考 vs. 多參考**
   - 單參考：保持單一風格
   - 多參考：融合多個特徵

### 成功標準

✅ **測試通過條件：**
- 所有 5 個測試案例成功完成
- 圖片已下載到 `data/generated_images/`
- --cref 100 的圖片明顯比 baseline 更一致
- 沒有 API 錯誤或超時

---

## 🐛 常見問題

### Q1: Reference image URL 無法訪問

**問題：**
```
❌ Test failed: Failed to submit task: Invalid reference image URL
```

**解決方法：**
1. 驗證 URL 可公開訪問：`curl -I "YOUR_URL"`
2. 確保使用 HTTPS
3. 確認 Discord CDN URL 格式正確
4. 檢查圖片檔案沒有損壞

### Q2: Task 超時

**問題：**
```
TimeoutError: Task task_abc123 timed out after 300s
```

**解決方法：**
1. Midjourney 伺服器可能繁忙，稍後重試
2. 增加 timeout 時間：
   ```python
   client = TTAPIClient(timeout=600)  # 10 分鐘
   ```

### Q3: API Quota 錯誤

**問題：**
```
❌ Error: [insufficient_quota] 429: API quota exceeded
```

**解決方法：**
1. 檢查 TTAPI 帳戶餘額
2. 等待 quota 重置
3. 升級 API plan

---

## 📄 文檔更新

測試完成後，更新：

1. **測試結果記錄** (`docs/test_results/story_3.2_cref_test.md`)
   - 記錄所有 5 個測試的結果
   - 截圖比較不同權重的效果
   - 記錄實際成本

2. **最佳實踐建議**
   - 推薦的 --cw 權重值
   - 單參考 vs. 多參考使用時機
   - Character consistency 評估標準

---

## ✅ Story 3.2 完成標準

- [x] 測試腳本已創建（`test_cref.py`）
- [x] Dry run 驗證通過
- [ ] Reference images 已上傳到 Discord CDN
- [ ] CREF_URLS 已更新
- [ ] 5 個測試案例全部執行成功
- [ ] 結果分析文檔已完成
- [ ] Git commit 已提交

---

## 🔗 相關資源

- **TTAPI Documentation:** https://docs.ttapi.io/
- **Midjourney --cref Guide:** https://docs.midjourney.com/docs/character-reference
- **Discord CDN Info:** https://discord.com/developers/docs/reference

---

**維護者：** Product Manager (John)
**支援：** FYP-RoleMarket Project
