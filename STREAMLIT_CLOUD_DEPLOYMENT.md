# Streamlit Cloud 部署指南

## 📋 部署前檢查清單

### 1. **環境變數設置** (必須！)

在 Streamlit Cloud 的 **Settings > Secrets** 中添加：

```toml
# .streamlit/secrets.toml 格式
GEMINI_API_KEY = "AIzaSyCAAv-UdJeeOHSipvHpyjwHQEvf-CP006g"
GEMINI_OPENAI_API_KEY = "sk-35KnyRmNEgN8GnpOAjrpOSoaPCinUKm0WMOzCUuAc3dah6eC"
OPENAI_API_KEY = "your-openai-key"  # 如果使用 GPT prompt generation
```

⚠️ **不要** 將 API keys 提交到 Git！

---

## 🚀 部署步驟

### 選項 1: 使用優化的 requirements (推薦)

1. 在 Streamlit Cloud 設置中，將 **Python version** 設為 `3.11` 或更高
2. 將 **Requirements file** 路徑改為 `requirements-cloud.txt`
3. 確保 `packages.txt` 已包含系統依賴

### 選項 2: 使用完整 requirements

如果需要 GPU 支持（Cloud 可能不支援），使用 `requirements.txt`

---

## ⚠️ 已知限制

### 1. **多線程警告** (可忽略)
```
Thread 'ThreadPoolExecutor-1_0': missing ScriptRunContext!
```
- **原因**: Streamlit Cloud 的多線程限制
- **影響**: 無，功能正常
- **解決**: 代碼已添加異常處理

### 2. **記憶體限制**
- Streamlit Cloud 免費版: ~1GB RAM
- PyTorch + Transformers: ~800MB
- **解決**: 使用 `requirements-cloud.txt` (CPU-only)

### 3. **文件寫入**
- Cloud 環境可能無法寫入某些目錄
- 已配置 `data/generated_images/` 為輸出目錄
- 如果失敗，圖像會保存在臨時目錄

---

## 🐛 常見錯誤排查

### Error: "No module named 'requests'"
**解決**: 檢查 requirements 是否正確安裝
```bash
# 在 Cloud 日誌中應該看到:
Successfully installed requests-2.31.0
```

### Error: "GEMINI_OPENAI_API_KEY not found"
**解決**:
1. 檢查 Streamlit Cloud Secrets 設置
2. 確保變數名稱完全匹配（大小寫敏感）

### Error: "Memory limit exceeded"
**解決**:
1. 使用 `requirements-cloud.txt`
2. 減少同時生成的圖像數量 (2 張而非 4 張)
3. 考慮升級到 Streamlit Cloud Pro

### Error: "Deployment timeout"
**解決**:
1. 確保使用 `requirements-cloud.txt`
2. 檢查 PyTorch 是否使用 CPU-only 版本
3. 重新部署

---

## 📊 性能優化

### 建議設置 (Streamlit Cloud):
- **生成圖像數量**: 2 張 (而非 4 張)
- **多線程**: 保持啟用 (已優化錯誤處理)
- **CLIP 驗證**: 保持啟用

### App 配置:
```python
# 在 config.py 中可添加 Cloud 環境檢測
import os

IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"

if IS_CLOUD:
    # Cloud 優化設置
    DEFAULT_NUM_IMAGES = 2
    ENABLE_MULTITHREADING = True
else:
    # 本地設置
    DEFAULT_NUM_IMAGES = 4
    ENABLE_MULTITHREADING = True
```

---

## 📝 部署後驗證

1. ✅ App 成功啟動
2. ✅ 可以選擇 API (OpenAI-Compatible / Official Google)
3. ✅ Manual Input 功能正常
4. ✅ 圖像生成功能正常 (可能較慢)
5. ✅ CLIP 驗證計算正常

---

## 🆘 需要幫助？

如果部署失敗，請檢查：
1. **Logs**: Streamlit Cloud > Manage app > Logs
2. **Secrets**: 確保 API keys 正確設置
3. **Requirements**: 使用 `requirements-cloud.txt`
4. **Python Version**: 3.11+

---

## 📚 相關文件

- `requirements-cloud.txt`: Cloud 優化的依賴
- `packages.txt`: 系統依賴
- `.streamlit/config.toml`: Streamlit 配置
- `obj4_web_app/config.py`: App 配置

---

## 🔄 更新部署

推送代碼到 GitHub 後，Streamlit Cloud 會自動重新部署：

```bash
git add .
git commit -m "update: Streamlit Cloud 優化"
git push origin main
```

等待 2-5 分鐘部署完成。
