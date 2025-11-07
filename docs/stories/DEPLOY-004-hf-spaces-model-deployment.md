# User Story: Hugging Face Spaces Model Deployment

**Story ID:** DEPLOY-004
**Epic:** Production Deployment
**Priority:** Medium
**Estimated Effort:** 6-8 hours
**Status:** Ready for Development

---

## 📖 Story

**As a** researcher or stakeholder,
**I want** the Transformer forecasting model hosted on Hugging Face Spaces,
**So that** I can access the model independently for sales predictions.

---

## 🎯 Acceptance Criteria

1. ✅ **Model Upload to HF Hub**
   - Model file uploaded to Hugging Face Hub
   - Model card (README.md) created with specs
   - Model accessible via public URL
   - License specified (apache-2.0)

2. ✅ **HF Space Created**
   - Streamlit Space created for model demo
   - Space deployed successfully
   - Public URL accessible
   - Space renders prediction interface

3. ✅ **Prediction Interface**
   - Input fields for prediction features
   - Prediction button functional
   - Results displayed clearly
   - Model specs shown (R², MAE, training info)

4. ✅ **Model Loading**
   - Model loads from HF Hub successfully
   - Cold start time < 15 seconds
   - Model inference works correctly
   - Error handling for load failures

5. ✅ **Documentation**
   - Model card includes usage instructions
   - Code examples for loading model
   - Citation information
   - Training details documented

---

## 🔧 Technical Details

**Files to Create:**
- `hf-spaces-deploy/app.py` (Space entry point)
- `hf-spaces-deploy/requirements.txt` (HF Space dependencies)
- `scripts/upload_model_to_hf.py` (Model upload script)

**Files to Use:**
- `models/transformer_lulu/best_transformer_model.pth` (trained model)
- `obj3_lstm_forecast/hybrid_transformer_model.py` (model architecture)

**Implementation Approach:**

1. Create model upload script:
   ```python
   from huggingface_hub import HfApi, create_repo

   api = HfApi()
   repo_id = "your-username/rolemarket-transformer"

   # Create repo
   create_repo(repo_id, repo_type="model", exist_ok=True)

   # Upload model
   api.upload_file(
       path_or_fileobj="models/transformer_lulu/best_transformer_model.pth",
       path_in_repo="best_transformer_model.pth",
       repo_id=repo_id
   )
   ```

2. Create HF Space app:
   ```python
   import streamlit as st
   from huggingface_hub import hf_hub_download

   @st.cache_resource
   def load_model():
       model_path = hf_hub_download(
           repo_id="your-username/rolemarket-transformer",
           filename="best_transformer_model.pth"
       )
       # Load model...
       return model

   st.title("📊 RoleMarket Sales Forecasting")
   model = load_model()
   # Prediction interface...
   ```

3. Deploy to HF Spaces:
   ```bash
   cd hf-spaces-deploy/
   git init
   git remote add space https://huggingface.co/spaces/your-username/rolemarket-forecasting
   git add .
   git commit -m "feat: 初始化銷量預測 Demo"
   git push space main
   ```

**Reference:** Tech-Spec Section "Part 4: Hugging Face Spaces Model Deployment"

---

## 🧪 Test Cases

### Test Case 1: Model Upload Success
**Given:** Trained model at `models/transformer_lulu/best_transformer_model.pth`
**When:** Running upload script
**Then:**
- Model uploaded to HF Hub successfully
- Model accessible at repo URL
- Model card displays correctly

### Test Case 2: HF Space Deployment
**Given:** Space repository pushed to HF
**When:** HF builds the Space
**Then:**
- Build completes successfully (< 10 min)
- Space accessible via public URL
- App loads without errors

### Test Case 3: Model Loading in Space
**Given:** Space is running
**When:** User visits Space URL
**Then:**
- Model loads from HF Hub
- Cold start completes in < 15s
- Success message displayed

### Test Case 4: Prediction Interface
**Given:** Model loaded successfully
**When:** User inputs prediction features
**And:** User clicks "Predict Sales"
**Then:**
- Prediction computed
- Result displayed clearly
- Result is reasonable (> 0, < 10000)

### Test Case 5: Model Card Documentation
**Given:** Model uploaded to HF Hub
**When:** User views model card
**Then:**
- Model specs clearly listed
- Usage code examples provided
- Training details documented
- Citation information included

---

## 📚 Related Documentation

- Tech-Spec: `docs/tech-specs/production-deployment-tech-spec.md` (Part 4)
- Training Log: `docs/experiment-log-lulu-transformer.md`
- HF Deployment Guide: `hf-spaces-deploy/README.md`
- HF Hub Docs: https://huggingface.co/docs/hub

---

## ✅ Definition of Done

- [ ] Model upload script created and tested
- [ ] Model uploaded to HF Hub successfully
- [ ] Model card created with complete documentation
- [ ] HF Space created and deployed
- [ ] Space app loads and renders correctly
- [ ] Model loads from HF Hub in Space
- [ ] Prediction interface functional
- [ ] All test cases pass
- [ ] Public URLs documented in README

---

## 🔗 Dependencies

**Blocks:**
- None (independent deployment)

**Depends On:**
- None (model already trained)

**Nice-to-Have:**
- DEPLOY-003 (can link main app to HF Space)

---

## 📋 Subtasks

### Model Upload (3-4h)
1. [ ] Create model upload script (1h)
2. [ ] Login to Hugging Face CLI (`huggingface-cli login`) (15min)
3. [ ] Run upload script and verify (30min)
4. [ ] Create comprehensive model card (1h)
5. [ ] Test model download from Hub (15min)

### HF Space Deployment (3-4h)
6. [ ] Create Space app (`hf-spaces-deploy/app.py`) (2h)
7. [ ] Create requirements.txt for Space (15min)
8. [ ] Test app locally (30min)
9. [ ] Create HF Space on website (15min)
10. [ ] Push code to Space repository (15min)
11. [ ] Wait for build and test (30min)
12. [ ] Document Space URL (15min)

---

## 🌐 Expected URLs

After completion, these URLs should be accessible:

1. **Model Repository:**
   - `https://huggingface.co/your-username/rolemarket-transformer`
   - Contains model file and documentation

2. **HF Space:**
   - `https://huggingface.co/spaces/your-username/rolemarket-forecasting`
   - Interactive prediction interface

3. **Direct App URL:**
   - `https://your-username-rolemarket-forecasting.hf.space`
   - Embeddable demo

---

**Story Created:** 2025-11-07
**Last Updated:** 2025-11-07
