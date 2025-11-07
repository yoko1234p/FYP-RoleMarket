# User Story: Streamlit Cloud Deployment Configuration

**Story ID:** DEPLOY-003
**Epic:** Production Deployment
**Priority:** High
**Estimated Effort:** 4-6 hours
**Status:** Ready for Development

---

## 📖 Story

**As a** project owner,
**I want** to deploy the Streamlit web application to Streamlit Cloud,
**So that** users can access the system via a public URL without local setup.

---

## 🎯 Acceptance Criteria

1. ✅ **Streamlit Cloud Configuration Files**
   - `.streamlit/config.toml` created with theme and server settings
   - `requirements.txt` verified for Streamlit Cloud compatibility
   - `.gitignore` updated to exclude secrets

2. ✅ **Secrets Management**
   - Secrets template created (`docs/streamlit-secrets-template.toml`)
   - `config.py` updated to support both `.env` and Streamlit Secrets
   - All required secrets documented (GPT_API_FREE_KEY, GOOGLE_API_KEY, etc.)

3. ✅ **Deployment Successful**
   - App deployed to Streamlit Cloud
   - Public URL accessible
   - All pages load without errors
   - Secrets configured correctly in dashboard

4. ✅ **Testing Post-Deployment**
   - Character input works
   - Manual keyword input works
   - Prompt generation works
   - Google Trends extraction tested (after DEPLOY-001)
   - Regional warning displays (after DEPLOY-002)

5. ✅ **Documentation**
   - Deployment guide created (post-deployment)
   - Public URL documented in README
   - Known issues documented

---

## 🔧 Technical Details

**Files to Create:**
- `.streamlit/config.toml`
- `docs/streamlit-secrets-template.toml`
- `docs/deployment-guide.md` (post-deployment)

**Files to Modify:**
- `obj4_web_app/config.py` (Streamlit Secrets support)
- `.gitignore` (exclude secrets)
- `README.md` (add deployment URL)

**Implementation Approach:**

1. Create `.streamlit/config.toml`:
   ```toml
   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   font = "sans serif"

   [server]
   headless = true
   enableCORS = false
   enableXsrfProtection = true
   maxUploadSize = 200

   [browser]
   gatherUsageStats = false
   ```

2. Update `config.py` to support Streamlit Secrets:
   ```python
   def get_secret(key: str, default=None):
       """Get secret from Streamlit secrets or environment variable."""
       if hasattr(st, 'secrets') and key in st.secrets:
           return st.secrets[key]
       return os.getenv(key, default)

   GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
   GPT_API_TOKEN = get_secret("GPT_API_TOKEN") or get_secret("GPT_API_FREE_KEY")
   ```

3. Deploy to Streamlit Cloud:
   - Connect GitHub repository
   - Set app path: `obj4_web_app/app.py`
   - Add secrets via dashboard
   - Monitor build logs
   - Test deployed app

**Reference:** Tech-Spec Section "Part 3: Streamlit Cloud Deployment Configuration"

---

## 🧪 Test Cases

### Test Case 1: Local Development with .env
**Given:** `.env` file with secrets
**When:** Running `streamlit run obj4_web_app/app.py` locally
**Then:**
- App starts successfully
- Secrets loaded from `.env`
- All features work

### Test Case 2: Streamlit Cloud with Secrets
**Given:** Secrets configured in Streamlit Cloud dashboard
**When:** App deployed to Streamlit Cloud
**Then:**
- App starts successfully
- Secrets loaded from `st.secrets`
- All features work

### Test Case 3: Missing Secrets Handling
**Given:** Required secret (GPT_API_TOKEN) not configured
**When:** App starts
**Then:**
- Clear error message displayed
- Instructions to configure secrets
- App does not crash silently

### Test Case 4: Public Access
**Given:** Deployed app URL
**When:** User visits URL from any device
**Then:**
- App loads within 5 seconds
- All pages accessible
- No authentication required
- UI renders correctly

---

## 📚 Related Documentation

- Tech-Spec: `docs/tech-specs/production-deployment-tech-spec.md` (Part 3)
- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud

---

## ✅ Definition of Done

- [ ] `.streamlit/config.toml` created
- [ ] Secrets template created and documented
- [ ] `config.py` updated to support Streamlit Secrets
- [ ] `.gitignore` updated to exclude secrets
- [ ] App deployed to Streamlit Cloud successfully
- [ ] Public URL accessible and working
- [ ] All secrets configured correctly
- [ ] Post-deployment testing completed
- [ ] Deployment guide created
- [ ] README updated with deployment URL

---

## 🔗 Dependencies

**Blocks:**
- DEPLOY-005 (End-to-end testing requires deployment)

**Depends On:**
- DEPLOY-001 (Google Trends fix should be deployed)
- DEPLOY-002 (Regional restriction handling should be deployed)

**Optional:**
- Can deploy without DEPLOY-001/002, but functionality limited

---

## 📋 Deployment Steps

### Pre-Deployment
1. [ ] Create/verify `.streamlit/config.toml`
2. [ ] Create secrets template
3. [ ] Update `config.py` for Streamlit Secrets
4. [ ] Update `.gitignore`
5. [ ] Test locally with `st.secrets` simulation
6. [ ] Commit and push to GitHub

### Deployment
7. [ ] Create Streamlit Cloud account
8. [ ] Connect GitHub repository
9. [ ] Configure deployment (app path, Python version)
10. [ ] Add secrets via dashboard
11. [ ] Deploy and monitor build

### Post-Deployment
12. [ ] Test deployed app
13. [ ] Verify all secrets working
14. [ ] Test from different devices/networks
15. [ ] Create deployment guide
16. [ ] Update README with URL

---

**Story Created:** 2025-11-07
**Last Updated:** 2025-11-07
