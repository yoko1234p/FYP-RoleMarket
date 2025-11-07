# User Story: End-to-End Testing & Validation

**Story ID:** DEPLOY-005
**Epic:** Production Deployment
**Priority:** High
**Estimated Effort:** 8-10 hours
**Status:** Blocked (Requires DEPLOY-001, DEPLOY-002, DEPLOY-003)

---

## 📖 Story

**As a** QA engineer / Product Manager,
**I want** to perform comprehensive end-to-end testing of the deployed application,
**So that** we can verify all features work correctly before announcing production readiness.

---

## 🎯 Acceptance Criteria

1. ✅ **All 3 Test Scenarios Executed**
   - Scenario A: Spring Festival Theme (retry after fixes)
   - Scenario B: Halloween Theme (new)
   - Scenario C: Christmas Theme (new)
   - Each scenario tests full flow: input → trends → prompt → images → forecast

2. ✅ **Google Trends Testing**
   - Auto-extraction tested for all 3 themes
   - Success rate >= 80% measured
   - Retry logic verified
   - Error messages validated

3. ✅ **Regional Restriction Testing**
   - Warning displays correctly for HK users
   - Non-HK users see no warning
   - VPN workaround tested
   - Graceful degradation verified

4. ✅ **Prompt Generation Testing**
   - All 3 scenarios generate valid prompts
   - Prompts include character + keywords
   - Prompt quality acceptable (50-85 words)
   - Download feature works

5. ✅ **Image Generation Testing** (if API available)
   - Images generated successfully
   - CLIP similarity >= 0.75
   - Character consistency maintained
   - Generation time acceptable (< 15s/image)

6. ✅ **Sales Forecasting Testing**
   - Model loads correctly
   - Predictions computed successfully
   - Results reasonable (> 0, < 10000)
   - Visualization renders correctly

7. ✅ **Cross-Browser/Device Testing**
   - Desktop Chrome/Firefox/Safari
   - Mobile iOS/Android (basic check)
   - Tablet (optional)

8. ✅ **Performance Testing**
   - Page load time < 5s
   - Cold start time < 15s (model loading)
   - UI responsive during API calls
   - No memory leaks observed

9. ✅ **Testing Report Updated**
   - All test results documented
   - Screenshots captured
   - Known issues documented
   - Deployment readiness assessed

---

## 🔧 Technical Details

**Files to Update:**
- `docs/testing/manual-testing-report.md` (update with final results)

**Test Environment:**
- **Local:** For initial validation
- **Streamlit Cloud:** For deployment testing
- **HF Spaces:** For model testing

**Tools Needed:**
- Multiple browsers (Chrome, Firefox, Safari)
- VPN (for testing regional restrictions)
- Mobile device or emulator
- Screenshot tool
- Network monitor (DevTools)

**Reference:** Tech-Spec Section "Part 5: End-to-End Testing Plan"

---

## 🧪 Test Cases

### Scenario A: Spring Festival Theme (Retry)

**Pre-conditions:**
- All fixes deployed (DEPLOY-001, DEPLOY-002)
- API keys configured

**Test Steps:**
1. Navigate to Design Generation page
2. Enter character info:
   - Name: "Lulu Pig"
   - Description: "可愛粉紅豬，大眼睛，圓滾滾身材"
3. Select theme: "🎊 新年"
4. Click "提取關鍵字" (auto-extraction)
5. Verify keywords extracted successfully
6. Select 3-5 keywords
7. Click "生成 Prompt"
8. Verify prompt generated successfully
9. Select reference image
10. Click "生成圖片" (if API available)
11. Verify images generated
12. Navigate to Sales Forecasting page
13. Upload/select generated image
14. Verify prediction computed

**Expected Results:**
- ✅ All steps complete without critical errors
- ✅ Google Trends extraction works (or clear error)
- ✅ Prompt quality acceptable
- ✅ Images maintain character consistency
- ✅ Sales prediction reasonable

---

### Scenario B: Halloween Theme (New)

**Test Steps:**
1. Enter character info (same as Scenario A)
2. Select theme: "🎃 Halloween"
3. Extract keywords
4. Generate prompt
5. Generate images (if API available)
6. Test forecasting
7. Compare results to expected seasonal patterns

**Expected Results:**
- ✅ Halloween-specific keywords extracted
- ✅ Prompt includes Halloween elements
- ✅ Images show Halloween theme
- ✅ Sales prediction reflects fall season

---

### Scenario C: Christmas Theme (New)

**Test Steps:**
1. Enter character info (same as Scenario A)
2. Select theme: "🎄 Christmas"
3. Extract keywords
4. Generate prompt
5. Generate images (if API available)
6. Test forecasting
7. Compare results to expected seasonal patterns

**Expected Results:**
- ✅ Christmas-specific keywords extracted
- ✅ Prompt includes Christmas elements
- ✅ Images show Christmas theme
- ✅ Sales prediction reflects winter season

---

### Regional Restriction Testing

**Test Case 1: HK User**
1. Access app from HK IP (or use VPN to HK)
2. Navigate to Design Generation
3. Verify warning displays
4. Verify solution options listed
5. Test app functionality (without image generation)

**Test Case 2: Non-HK User**
1. Access app from US/TW IP
2. Navigate to Design Generation
3. Verify no warning displays
4. Test full functionality including image generation

---

### Performance Testing

**Test Case 1: Page Load Time**
1. Clear browser cache
2. Navigate to app URL
3. Measure time to interactive (< 5s target)
4. Repeat 3 times, calculate average

**Test Case 2: Model Cold Start**
1. Access app first time (cold start)
2. Navigate to Sales Forecasting page
3. Measure model loading time (< 15s target)

**Test Case 3: API Response Time**
1. Click "提取關鍵字"
2. Measure response time (< 10s target)
3. Click "生成 Prompt"
4. Measure response time (< 5s target)

---

### Cross-Browser Testing

Test on:
- [ ] Chrome (Desktop)
- [ ] Firefox (Desktop)
- [ ] Safari (Desktop/iOS)
- [ ] Edge (Desktop)
- [ ] Mobile Safari (iOS)
- [ ] Mobile Chrome (Android)

Verify:
- UI renders correctly
- All buttons clickable
- Forms submittable
- Charts render correctly
- No JavaScript errors in console

---

## 📚 Related Documentation

- Tech-Spec: `docs/tech-specs/production-deployment-tech-spec.md` (Part 5)
- Previous Testing: `docs/testing/manual-testing-report.md` (2025-11-07)
- Test Scenarios: Same report, Scenario A details

---

## ✅ Definition of Done

- [ ] All 3 test scenarios executed successfully
- [ ] Google Trends tested across multiple themes
- [ ] Regional restriction handling verified
- [ ] Prompt generation tested for all scenarios
- [ ] Image generation tested (if API available)
- [ ] Sales forecasting tested for all scenarios
- [ ] Cross-browser testing completed (4+ browsers)
- [ ] Performance testing completed (load time, cold start)
- [ ] Testing report updated with final results
- [ ] Screenshots captured for all scenarios
- [ ] Known issues documented with severity
- [ ] Deployment readiness assessment completed

---

## 🔗 Dependencies

**Blocks:**
- Production announcement (cannot announce until testing complete)

**Depends On:**
- DEPLOY-001 (Google Trends fix)
- DEPLOY-002 (Regional restriction handling)
- DEPLOY-003 (Streamlit Cloud deployment)

**Optional:**
- DEPLOY-004 (HF Spaces can be tested independently)

---

## 📋 Test Execution Plan

### Day 1: Scenario Testing (4-5h)
1. [ ] Setup test environment
2. [ ] Execute Scenario A (Spring Festival) - 1.5h
3. [ ] Execute Scenario B (Halloween) - 1.5h
4. [ ] Execute Scenario C (Christmas) - 1.5h
5. [ ] Document results and screenshots

### Day 2: Specialized Testing (3-4h)
6. [ ] Regional restriction testing (1h)
7. [ ] Performance testing (1h)
8. [ ] Cross-browser testing (2h)

### Day 3: Documentation & Validation (1-2h)
9. [ ] Update testing report with all results
10. [ ] Create test summary with metrics
11. [ ] Document known issues
12. [ ] Assess deployment readiness
13. [ ] Review with stakeholders

---

## 📊 Test Metrics to Track

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Google Trends Success Rate | >= 80% | TBD | ⏳ |
| Prompt Generation Success | 100% | TBD | ⏳ |
| Image Generation Success | >= 90% | TBD | ⏳ |
| Forecasting Success | 100% | TBD | ⏳ |
| Page Load Time | < 5s | TBD | ⏳ |
| Model Cold Start | < 15s | TBD | ⏳ |
| Cross-Browser Support | 4+ browsers | TBD | ⏳ |
| Critical Bugs | 0 | TBD | ⏳ |

---

## 🚨 Known Issues to Retest

From previous testing (2025-11-07):

1. **Issue #1: Google Trends Auto-Extraction**
   - **Previous Status:** Failed
   - **Expected After Fix:** >= 80% success rate
   - **Retest:** REQUIRED

2. **Issue #2: Prompt Generation**
   - **Previous Status:** Fixed
   - **Expected After Fix:** 100% success
   - **Retest:** Verify fix still works

3. **Issue #3: Gemini API Regional Restriction**
   - **Previous Status:** Documented
   - **Expected After Fix:** Clear warning displayed
   - **Retest:** Verify warning + graceful degradation

---

## 📸 Screenshots to Capture

1. [ ] Scenario A - Spring Festival (all steps)
2. [ ] Scenario B - Halloween (all steps)
3. [ ] Scenario C - Christmas (all steps)
4. [ ] Regional restriction warning
5. [ ] Google Trends success/failure
6. [ ] Prompt generation output
7. [ ] Image generation results
8. [ ] Sales forecasting output
9. [ ] Mobile view (iOS/Android)
10. [ ] Error states (if any)

---

**Story Created:** 2025-11-07
**Last Updated:** 2025-11-07
