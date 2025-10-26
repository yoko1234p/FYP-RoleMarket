# Epic 1: Foundation Setup - 剩餘任務指南

## Epic 1.3: 註冊並驗證 TTAPI Midjourney 帳號

### 步驟

1. **註冊 TTAPI 帳號**
   - 訪問：https://ttapi.io
   - 創建帳號並登入

2. **獲取 API Key**
   - Dashboard → API Keys
   - 創建新的 API key
   - 複製 API key（只會顯示一次！）

3. **充值 TTAPI Quota（PPU Mode）**
   - 根據 PRD，預算為 ~$10-30
   - 建議初始充值 $30（約可生成 28-40 張圖）
   - 選擇 PPU (Pay-Per-Use) 模式

4. **設置環境變數**
   ```bash
   # 編輯 .env 檔案
   nano .env

   # 添加：
   TTAPI_API_KEY=你嘅_API_KEY_喺度
   ```

5. **驗證 API 連接**
   ```bash
   # 在 Docker 容器內執行測試
   docker exec -it fyp-rolemarket python -c "
   from obj2_midjourney_api.ttapi_client import TTAPIClient
   client = TTAPIClient()
   print('✓ TTAPI Client initialized successfully')
   "
   ```

### 驗收標準
- [ ] TTAPI 帳號已註冊
- [ ] API key 已獲取並設置到 .env
- [ ] 已充值至少 $10 quota
- [ ] Python TTAPIClient 可成功初始化（不報錯）

---

## Epic 1.4: 驗證 GPT_API_free 存取

### 步驟

1. **獲取 GPT_API_free Access Token**
   - 項目：https://github.com/chatanywhere/GPT_API_free
   - 關注公眾號或訪問指定網站獲取免費 token
   - 或購買付費 token（更穩定）

2. **設置環境變數**
   ```bash
   # 編輯 .env 檔案
   nano .env

   # 添加：
   GPT_API_FREE_KEY=你嘅_ACCESS_TOKEN
   GPT_API_FREE_BASE_URL=https://api.chatanywhere.org/v1
   ```

3. **驗證 API 連接**
   ```bash
   # 測試 OpenAI client
   docker exec -it fyp-rolemarket python -c "
   import os
   from openai import OpenAI
   from dotenv import load_dotenv

   load_dotenv()
   client = OpenAI(
       api_key=os.getenv('GPT_API_FREE_KEY'),
       base_url=os.getenv('GPT_API_FREE_BASE_URL')
   )

   # 簡單測試
   response = client.chat.completions.create(
       model='gpt-3.5-turbo',
       messages=[{'role': 'user', 'content': 'Hello'}],
       max_tokens=10
   )
   print('✓ GPT_API_free connected:', response.choices[0].message.content)
   "
   ```

### 驗收標準
- [ ] GPT_API_free token 已獲取
- [ ] Token 已設置到 .env
- [ ] OpenAI client 可成功調用 API（不報錯）
- [ ] 測試 API 返回正確響應

---

## Epic 1.5: 準備 Pikachu 參考圖像

### 步驟

1. **選擇高質量 Pikachu 參考圖**
   - 要求：
     - 高解析度（至少 1024×1024）
     - 清晰展示角色特徵（顏色、造型、表情）
     - 正面或 3/4 側面視角
     - 乾淨背景（純色或簡單背景）
   - 建議來源：
     - 官方 Pokémon 素材
     - 高質量 fan art（需注意版權）
     - Stable Diffusion/Midjourney 生成的 Pikachu

2. **準備 1-2 張圖像**
   - 根據 PRD：「1-2 張高質量 Pikachu 參考圖像」
   - 命名規範：
     - `pikachu_ref_1.jpg` (主要參考)
     - `pikachu_ref_2.jpg` (次要參考，可選)

3. **放置到項目目錄**
   ```bash
   # 複製圖像到項目
   cp /path/to/your/pikachu_ref_1.jpg data/reference_images/
   cp /path/to/your/pikachu_ref_2.jpg data/reference_images/
   ```

4. **（可選）上傳到公開 URL**
   - TTAPI --cref 參數需要 HTTP URL
   - 選項：
     - 上傳到 Imgur: https://imgur.com
     - 上傳到 Google Drive（設置公開分享）
     - 上傳到 GitHub（raw URL）
   - 記錄 URL 以便後續使用

5. **驗證圖像質量**
   ```bash
   # 使用 CLIP 驗證圖像特徵
   docker exec -it fyp-rolemarket python -c "
   from PIL import Image
   from transformers import CLIPProcessor, CLIPModel
   import torch

   model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
   processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

   # 載入圖像
   image = Image.open('data/reference_images/pikachu_ref_1.jpg')

   # 提取特徵
   inputs = processor(images=image, return_tensors='pt')
   outputs = model.get_image_features(**inputs)

   print(f'✓ Image loaded, CLIP embedding shape: {outputs.shape}')
   print(f'  Expected: torch.Size([1, 768])')
   "
   ```

### 驗收標準
- [ ] 1-2 張高質量 Pikachu 圖像已準備
- [ ] 圖像已放置到 `data/reference_images/`
- [ ] 圖像解析度 >= 1024×1024
- [ ] （可選）圖像已上傳到公開 URL
- [ ] CLIP 可成功提取圖像特徵（768-dim embeddings）

---

## 完成檢查清單

完成以上三個 Epic 後，執行以下檢查：

```bash
# 1. 檢查環境變數
cat .env | grep -E "(TTAPI_API_KEY|GPT_API_FREE_KEY)"

# 2. 檢查參考圖像
ls -lh data/reference_images/

# 3. 全面測試（Docker 容器內）
docker exec -it fyp-rolemarket python -c "
print('=== Epic 1.3-1.5 驗證 ===')

# Test 1: TTAPI Client
try:
    from obj2_midjourney_api.ttapi_client import TTAPIClient
    client = TTAPIClient()
    print('✓ Epic 1.3: TTAPI Client OK')
except Exception as e:
    print(f'✗ Epic 1.3: TTAPI Client FAILED - {e}')

# Test 2: GPT_API_free
try:
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv('GPT_API_FREE_KEY'),
        base_url=os.getenv('GPT_API_FREE_BASE_URL')
    )
    print('✓ Epic 1.4: GPT_API_free OK')
except Exception as e:
    print(f'✗ Epic 1.4: GPT_API_free FAILED - {e}')

# Test 3: Reference Images
try:
    from PIL import Image
    import os
    ref_dir = 'data/reference_images'
    images = [f for f in os.listdir(ref_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) >= 1:
        img = Image.open(os.path.join(ref_dir, images[0]))
        print(f'✓ Epic 1.5: Found {len(images)} reference image(s), size: {img.size}')
    else:
        print('✗ Epic 1.5: No reference images found')
except Exception as e:
    print(f'✗ Epic 1.5: Reference Images FAILED - {e}')

print('\\n=== Epic 1 完成度 ===')
print('Epic 1.1: ✅ 項目結構初始化')
print('Epic 1.2: ✅ Python 環境與 Docker 設置')
print('Epic 1.3-1.5: 請根據以上測試結果確認')
"
```

## 時間估算

- Epic 1.3: ~2 小時（註冊、充值、驗證）
- Epic 1.4: ~1 小時（獲取 token、測試）
- Epic 1.5: ~2 小時（搜尋/生成圖像、驗證質量）

**Total: ~5 小時**（可並行處理）

---

**完成後**，你就可以開始 Epic 2（Objective 1: Trend Intelligence & Prompt Generation）！
