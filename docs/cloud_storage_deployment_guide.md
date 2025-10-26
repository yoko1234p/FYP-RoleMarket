# Cloud Storage Deployment Guide for Reference Images

**版本：** 1.0
**作者：** Product Manager (John)
**更新日期：** 2025-10-26

---

## 📋 目標

將 reference images 部署到雲端存儲，確保：
1. ✅ **HTTP 公開訪問**：TTAPI 可以通過 HTTPS URL 訪問
2. ✅ **永久可用**：不依賴 Discord CDN（有效期限制）
3. ✅ **高可用性**：99.9%+ uptime
4. ✅ **全球 CDN**：快速訪問（重要性較低）

---

## 🌐 雲端存儲方案選擇

### Option 1: AWS S3 + CloudFront（推薦）

**優勢：**
- ✅ 高可靠性（99.999999999% durability）
- ✅ 全球 CDN（CloudFront）
- ✅ 靈活的訪問控制
- ✅ 低成本（$0.023/GB 存儲 + $0.085/GB 流量）

**成本估算：**
- 存儲：2 images × 0.2MB = 0.4MB ≈ **$0.00001/月**
- 流量：假設 1000 次訪問/月 × 0.2MB = 200MB ≈ **$0.017/月**
- **總計：~$0.02/月**

**設置步驟：**

1. **創建 S3 Bucket**
   ```bash
   aws s3 mb s3://fyp-rolemarket-references --region us-east-1
   ```

2. **上傳 Reference Images**
   ```bash
   aws s3 cp data/reference_images/lulu_pig_ref_1.png \
     s3://fyp-rolemarket-references/lulu_pig_ref_1.png \
     --acl public-read

   aws s3 cp data/reference_images/lulu_pig_ref_2.png \
     s3://fyp-rolemarket-references/lulu_pig_ref_2.png \
     --acl public-read
   ```

3. **配置 Bucket Policy（公開訪問）**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "PublicReadGetObject",
         "Effect": "Allow",
         "Principal": "*",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::fyp-rolemarket-references/*"
       }
     ]
   }
   ```

4. **獲取公開 URLs**
   ```
   https://fyp-rolemarket-references.s3.amazonaws.com/lulu_pig_ref_1.png
   https://fyp-rolemarket-references.s3.amazonaws.com/lulu_pig_ref_2.png
   ```

5. **（可選）設置 CloudFront CDN**
   - 創建 CloudFront Distribution
   - Origin: S3 bucket
   - 獲得 CDN URLs：
   ```
   https://d111111abcdef8.cloudfront.net/lulu_pig_ref_1.png
   https://d111111abcdef8.cloudfront.net/lulu_pig_ref_2.png
   ```

---

### Option 2: Google Cloud Storage（GCS）

**優勢：**
- ✅ 整合 Google Cloud 生態系統
- ✅ 全球 CDN
- ✅ 簡單的權限管理

**成本估算：**
- 存儲：$0.020/GB/月
- 流量：$0.12/GB（北美/歐洲）
- **總計：~$0.02/月**

**設置步驟：**

1. **創建 Bucket**
   ```bash
   gsutil mb -l us-central1 gs://fyp-rolemarket-references
   ```

2. **上傳檔案**
   ```bash
   gsutil cp data/reference_images/lulu_pig_ref_1.png \
     gs://fyp-rolemarket-references/

   gsutil cp data/reference_images/lulu_pig_ref_2.png \
     gs://fyp-rolemarket-references/
   ```

3. **設置公開訪問**
   ```bash
   gsutil iam ch allUsers:objectViewer \
     gs://fyp-rolemarket-references
   ```

4. **獲取公開 URLs**
   ```
   https://storage.googleapis.com/fyp-rolemarket-references/lulu_pig_ref_1.png
   https://storage.googleapis.com/fyp-rolemarket-references/lulu_pig_ref_2.png
   ```

---

### Option 3: Azure Blob Storage

**優勢：**
- ✅ 整合 Microsoft Azure 生態系統
- ✅ 全球 CDN（Azure CDN）

**成本估算：**
- 存儲：$0.0184/GB/月
- 流量：$0.087/GB
- **總計：~$0.02/月**

**設置步驟：**

1. **創建 Storage Account**
   ```bash
   az storage account create \
     --name fyprolemarketrefs \
     --resource-group fyp-rolemarket \
     --location eastus \
     --sku Standard_LRS
   ```

2. **創建 Container**
   ```bash
   az storage container create \
     --name references \
     --account-name fyprolemarketrefs \
     --public-access blob
   ```

3. **上傳檔案**
   ```bash
   az storage blob upload \
     --account-name fyprolemarketrefs \
     --container-name references \
     --name lulu_pig_ref_1.png \
     --file data/reference_images/lulu_pig_ref_1.png

   az storage blob upload \
     --account-name fyprolemarketrefs \
     --container-name references \
     --name lulu_pig_ref_2.png \
     --file data/reference_images/lulu_pig_ref_2.png
   ```

4. **獲取公開 URLs**
   ```
   https://fyprolemarketrefs.blob.core.windows.net/references/lulu_pig_ref_1.png
   https://fyprolemarketrefs.blob.core.windows.net/references/lulu_pig_ref_2.png
   ```

---

### Option 4: Imgur（免費，簡單）

**優勢：**
- ✅ 完全免費
- ✅ 無需帳號設置
- ✅ 永久存儲

**限制：**
- ❌ 沒有版本控制
- ❌ 有廣告（對 API 無影響）
- ❌ 上傳限制（10MB/image）

**設置步驟：**

1. **手動上傳到 Imgur**
   - 訪問：https://imgur.com/upload
   - 上傳 `lulu_pig_ref_1.png` 和 `lulu_pig_ref_2.png`

2. **獲取直接連結**
   - 右鍵點擊圖片 → "Copy image address"
   - 範例：
   ```
   https://i.imgur.com/abc123.png
   https://i.imgur.com/def456.png
   ```

---

## 🛠️ 更新配置

### 1. 更新 `config/reference_images.py`

```python
# Cloud Storage URLs (after deployment)
CREF_URLS_CLOUD = [
    "https://your-actual-cloud-url.com/lulu_pig_ref_1.png",
    "https://your-actual-cloud-url.com/lulu_pig_ref_2.png"
]

# Switch to cloud URLs
CREF_URLS = CREF_URLS_CLOUD
```

### 2. 驗證 URLs

```bash
# Test URL accessibility
curl -I "https://your-cloud-url.com/lulu_pig_ref_1.png"

# Should return: HTTP/2 200
```

### 3. 更新 `.env`（可選）

```bash
# Add cloud storage URLs to environment
CREF_URL_1=https://your-cloud-url.com/lulu_pig_ref_1.png
CREF_URL_2=https://your-cloud-url.com/lulu_pig_ref_2.png
```

---

## 📊 成本比較

| 方案 | 存儲成本 | 流量成本 | CDN | 總計/月 | 推薦指數 |
|------|---------|---------|-----|---------|---------|
| AWS S3 | $0.00001 | $0.017 | ✅ | **$0.02** | ⭐⭐⭐⭐⭐ |
| GCS | $0.00001 | $0.024 | ✅ | **$0.02** | ⭐⭐⭐⭐ |
| Azure | $0.00001 | $0.017 | ✅ | **$0.02** | ⭐⭐⭐⭐ |
| Imgur | $0 | $0 | ✅ | **FREE** | ⭐⭐⭐ |
| Discord CDN | $0 | $0 | ✅ | **FREE** | ⭐⭐ (臨時) |

---

## 🔒 安全建議

### 1. 公開訪問設置
- ✅ **僅 Reference Images 公開**
- ❌ **Generated Images 不應公開**（包含客戶 IP）

### 2. CORS 設置（如需前端訪問）
```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["https://your-app-domain.com"],
      "AllowedMethods": ["GET"],
      "AllowedHeaders": ["*"]
    }
  ]
}
```

### 3. 版本控制
- 保留本地備份：`data/reference_images/`
- 雲端版本控制（S3 Versioning）
- Git LFS 追蹤（可選）

---

## 📝 部署檢查清單

### Development (當前)
- [x] Local files: `data/reference_images/`
- [x] Discord CDN URLs 已配置
- [x] `config/reference_images.py` 已創建

### Production (雲端部署)
- [ ] 選擇雲端存儲方案
- [ ] 創建 bucket/container
- [ ] 上傳 reference images
- [ ] 設置公開訪問
- [ ] 獲取公開 URLs
- [ ] 更新 `CREF_URLS_CLOUD` 配置
- [ ] 驗證 URLs 可訪問
- [ ] 測試 TTAPI --cref 功能
- [ ] 更新文檔

---

## 🔄 遷移腳本

創建自動化上傳腳本：

```python
# scripts/upload_references_to_cloud.py
import boto3
from pathlib import Path

def upload_to_s3(bucket_name='fyp-rolemarket-references'):
    """Upload reference images to AWS S3."""
    s3 = boto3.client('s3')

    ref_dir = Path('data/reference_images')
    for img_file in ref_dir.glob('lulu_pig_ref_*.png'):
        print(f"Uploading {img_file.name}...")

        s3.upload_file(
            str(img_file),
            bucket_name,
            img_file.name,
            ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'}
        )

        url = f"https://{bucket_name}.s3.amazonaws.com/{img_file.name}"
        print(f"✅ Uploaded: {url}")

if __name__ == '__main__':
    upload_to_s3()
```

---

## 🚀 推薦方案

**For FYP/Development:**
- **當前：** Discord CDN（已完成）
- **原因：** 免費、快速、無需設置

**For Production/Deployment:**
- **推薦：** AWS S3 + CloudFront
- **原因：**
  - 最可靠（99.999999999% durability）
  - 完整的基礎設施支持
  - 易於整合 CI/CD
  - 教育帳號可能有免費額度

**For Quick Prototype:**
- **推薦：** Imgur
- **原因：** 完全免費、零設置

---

## 📞 支援資源

- **AWS S3 文檔：** https://docs.aws.amazon.com/s3/
- **GCS 文檔：** https://cloud.google.com/storage/docs
- **Azure Blob 文檔：** https://docs.microsoft.com/azure/storage/blobs/
- **Imgur API：** https://apidocs.imgur.com/

---

**維護者：** Product Manager (John)
**支援：** FYP-RoleMarket Project
