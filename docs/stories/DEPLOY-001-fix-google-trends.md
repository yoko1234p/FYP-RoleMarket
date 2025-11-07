# User Story: Fix Google Trends Auto-Extraction

**Story ID:** DEPLOY-001
**Epic:** Production Deployment
**Priority:** High
**Estimated Effort:** 4-6 hours
**Status:** Ready for Development

---

## 📖 Story

**As a** designer using the web application,
**I want** the Google Trends auto-extraction to work reliably,
**So that** I can quickly discover trending keywords without manual research.

---

## 🎯 Acceptance Criteria

1. ✅ **Retry Logic Implemented**
   - Auto-extraction retries up to 3 times on failure
   - Exponential backoff used (2^n seconds delay)
   - Logging added for each retry attempt

2. ✅ **Error Messages Improved**
   - Clear error message when all retries fail
   - Error message explains possible causes
   - Workaround instructions provided (use manual input)

3. ✅ **Regional Configuration**
   - Support for HK, TW, US regions
   - Proper geo/hl/tz parameters for each region
   - Region detection from user IP (optional)

4. ✅ **Debug Logging**
   - Log all API requests with parameters
   - Log API responses for debugging
   - Error logs include full stack trace

5. ✅ **Testing**
   - Test with 3+ themes (新年, Halloween, Christmas)
   - Test with different regions
   - Test retry logic with simulated failures
   - Success rate >= 80% for valid themes

---

## 🔧 Technical Details

**Files to Modify:**
- `obj4_web_app/utils/trends_extractor_wrapper.py`

**Implementation Approach:**

1. Add `retry_with_backoff` decorator:
   ```python
   def retry_with_backoff(max_retries=3, base_delay=2):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if attempt < max_retries - 1:
                           delay = base_delay ** attempt
                           logger.warning(f"Attempt {attempt+1} failed, retrying in {delay}s")
                           time.sleep(delay)
                       else:
                           raise
           return wrapper
       return decorator
   ```

2. Add regional configuration:
   ```python
   REGION_CONFIGS = {
       'HK': {'geo': 'HK', 'hl': 'zh-TW', 'tz': 480},
       'TW': {'geo': 'TW', 'hl': 'zh-TW', 'tz': 480},
       'US': {'geo': 'US', 'hl': 'en-US', 'tz': 360},
   }
   ```

3. Improve error messages:
   ```python
   error_msg = """
   ⚠️ 未找到相關趨勢數據。

   可能原因：
   1. Google Trends API 限流（請稍後重試）
   2. 主題關鍵字未找到相關數據
   3. 網絡連接問題

   💡 建議：請使用「✍️ 手動輸入」標籤頁手動輸入關鍵字。
   """
   ```

**Reference:** Tech-Spec Section "Part 1: Fix Google Trends Auto-Extraction"

---

## 🧪 Test Cases

### Test Case 1: Successful Extraction
**Given:** Valid theme "🎊 新年"
**When:** User clicks "提取關鍵字"
**Then:**
- Keywords extracted successfully
- Top 5-10 keywords displayed
- Trend scores shown in chart

### Test Case 2: Retry on Failure
**Given:** Temporary API failure (simulated)
**When:** User clicks "提取關鍵字"
**Then:**
- First attempt fails
- Retries automatically (logged)
- Succeeds on retry 2 or 3

### Test Case 3: All Retries Fail
**Given:** Persistent API failure
**When:** User clicks "提取關鍵字"
**Then:**
- Error message displayed with possible causes
- Workaround instructions shown
- Manual input tab still accessible

### Test Case 4: Regional Configuration
**Given:** User in Taiwan region
**When:** Auto-extraction runs
**Then:**
- Uses TW region configuration
- Returns Taiwan-specific trends
- Keywords in Traditional Chinese

---

## 📚 Related Documentation

- Tech-Spec: `docs/tech-specs/production-deployment-tech-spec.md` (Part 1)
- Testing Report: `docs/testing/manual-testing-report.md` (Issue #1)
- Existing Code: `obj4_web_app/utils/trends_extractor_wrapper.py`

---

## ✅ Definition of Done

- [ ] Retry logic implemented with exponential backoff
- [ ] Regional configuration added for HK/TW/US
- [ ] Error messages improved with helpful context
- [ ] Debug logging added (requests/responses/errors)
- [ ] Unit tests written for retry logic
- [ ] Integration tests pass for 3+ themes
- [ ] Success rate >= 80% in testing
- [ ] Code reviewed and merged
- [ ] Testing report updated with results

---

## 🔗 Dependencies

**Blocks:**
- DEPLOY-003 (Streamlit deployment needs this fixed)

**Depends On:**
- None (can start immediately)

---

**Story Created:** 2025-11-07
**Last Updated:** 2025-11-07
