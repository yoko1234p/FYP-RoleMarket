# User Story: Add Gemini API Regional Restriction Handling

**Story ID:** DEPLOY-002
**Epic:** Production Deployment
**Priority:** High
**Estimated Effort:** 6-8 hours
**Status:** Ready for Development

---

## 📖 Story

**As a** Hong Kong user of the web application,
**I want** to see a clear warning about Gemini API regional restrictions,
**So that** I understand why image generation is unavailable and know my options.

---

## 🎯 Acceptance Criteria

1. ✅ **Region Detection**
   - Detect user's region using IP geolocation
   - Cache detection result in session
   - Support fallback if detection fails

2. ✅ **Warning UI Display**
   - Show clear warning for HK/CN users
   - Explain regional restriction
   - Provide 3 solution options (VPN, manual tools, wait for alternatives)
   - Display list of planned alternative APIs

3. ✅ **API Alternatives Documentation**
   - Create `docs/api-alternatives.md`
   - Document Midjourney TTAPI (immediate solution)
   - Document Replicate Flux (planned)
   - Document RunPod SDXL (long-term)
   - Include cost/speed/quality comparison

4. ✅ **Graceful Degradation**
   - App still usable without image generation
   - Prompt generation still works
   - Clear error message if API called despite restriction

5. ✅ **Testing**
   - Test region detection from different IPs
   - Test warning display for HK users
   - Test app works without image generation
   - Verify documentation is complete

---

## 🔧 Technical Details

**Files to Create:**
- `docs/api-alternatives.md`

**Files to Modify:**
- `obj4_web_app/config.py` (region detection)
- `obj4_web_app/pages/1_🎨_設計生成.py` (warning UI)
- `obj4_web_app/utils/design_generator.py` (error handling)

**Implementation Approach:**

1. Add region detection in `config.py`:
   ```python
   import requests

   def detect_user_region() -> str:
       """Detect user's region using IP geolocation."""
       try:
           response = requests.get('https://ipapi.co/json/', timeout=3)
           if response.ok:
               return response.json().get('country_code', 'UNKNOWN')
       except:
           pass
       return 'UNKNOWN'

   GEMINI_RESTRICTED_REGIONS = ['HK', 'CN']
   USER_REGION = detect_user_region()
   IS_GEMINI_RESTRICTED = USER_REGION in GEMINI_RESTRICTED_REGIONS
   ```

2. Add warning UI in design generation page:
   ```python
   if IS_GEMINI_RESTRICTED:
       st.warning(f"""
       ⚠️ **地區限制通知**

       偵測到您的地區為 **{USER_REGION}**，Google Gemini 2.5 Flash Image API
       目前不支援此地區。

       **解決方案：**
       1. 🌐 使用 VPN 連接到支援地區（如：美國、台灣）
       2. 📝 先生成 Prompt，稍後使用其他工具生成圖片
       3. ⏳ 等待我們整合替代 API（計劃中）

       詳情請參閱：`docs/api-alternatives.md`
       """)
   ```

3. Create comprehensive API alternatives documentation

**Reference:** Tech-Spec Section "Part 2: Add Gemini API Regional Restriction Handling"

---

## 🧪 Test Cases

### Test Case 1: HK User Sees Warning
**Given:** User IP resolves to Hong Kong
**When:** User opens Design Generation page
**Then:**
- Warning banner displays at top
- Warning explains regional restriction
- 3 solution options clearly listed
- Link to API alternatives documentation

### Test Case 2: Non-HK User No Warning
**Given:** User IP resolves to Taiwan
**When:** User opens Design Generation page
**Then:**
- No regional warning shown
- Image generation UI available
- Normal functionality

### Test Case 3: Region Detection Fails
**Given:** IP geolocation API timeout
**When:** App loads
**Then:**
- Defaults to 'UNKNOWN' region
- No restriction applied (conservative approach)
- App functions normally

### Test Case 4: HK User Attempts Image Generation
**Given:** HK user clicks "Generate Images"
**When:** API call is made
**Then:**
- Clear error message displayed
- Suggests using VPN or alternative
- Does not crash app

### Test Case 5: Documentation Complete
**Given:** User reads `docs/api-alternatives.md`
**When:** Reviewing documentation
**Then:**
- 3 API alternatives documented
- Cost/speed/quality comparison table
- Implementation priority listed
- Code examples provided

---

## 📚 Related Documentation

- Tech-Spec: `docs/tech-specs/production-deployment-tech-spec.md` (Part 2)
- Testing Report: `docs/testing/manual-testing-report.md` (Issue #3)
- New Doc: `docs/api-alternatives.md` (to be created)

---

## ✅ Definition of Done

- [ ] Region detection implemented in `config.py`
- [ ] Warning UI added to design generation page
- [ ] API alternatives documentation created
- [ ] Error handling updated in design generator
- [ ] Region detection tested from multiple IPs
- [ ] Warning display tested for HK/CN users
- [ ] App tested without image generation
- [ ] Documentation reviewed for completeness
- [ ] Code reviewed and merged

---

## 🔗 Dependencies

**Blocks:**
- DEPLOY-003 (Deployment needs regional handling)

**Depends On:**
- None (can start immediately)

---

## 📋 Subtasks

1. [ ] Implement region detection (2h)
2. [ ] Add warning UI with solution options (2h)
3. [ ] Create API alternatives documentation (2h)
4. [ ] Update error handling for restrictions (1h)
5. [ ] Test from different regions (1h)
6. [ ] Review and refine documentation (1h)

---

**Story Created:** 2025-11-07
**Last Updated:** 2025-11-07
