# Coding Standards & Best Practices

**Project:** AI-Driven Market-Informed Character IP Design & Demand Forecasting
**Version:** 1.0
**Last Updated:** 2025-11-06
**Applies To:** All Python code in this project

---

## Executive Summary

本文檔定義 FYP-RoleMarket 專案嘅程式碼標準。目標係確保程式碼可讀性、可維護性和團隊協作效率。

**核心原則：**
- ✅ **Pragmatic over Perfect** - 務實優先，不過度工程
- ✅ **Consistency First** - 保持一致性比選擇「最佳」方案更重要
- ✅ **AI-Agent Friendly** - 程式碼應易於 AI Agent 理解和修改

---

## Python Style Guide

### Base Standard

**遵循 PEP 8** with the following adjustments:

- **Line Length:** 100 characters (not 79)
  - Reason: 現代編輯器支援更寬顯示
  - Exception: Docstrings 仍保持 79 chars

- **Indentation:** 4 spaces (no tabs)

- **Quotes:**
  - Prefer double quotes `"` for strings
  - Use single quotes `'` for dict keys (if needed)
  ```python
  # Good
  message = "Hello World"
  config = {"key": "value"}

  # Avoid
  message = 'Hello World'
  ```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| **Modules** | lowercase_with_underscores | `enhanced_trends_pipeline.py` |
| **Classes** | PascalCase | `HybridTransformer`, `GoogleGeminiClient` |
| **Functions** | lowercase_with_underscores | `generate_prompt()`, `compute_similarity()` |
| **Variables** | lowercase_with_underscores | `clip_embedding`, `trends_data` |
| **Constants** | UPPERCASE_WITH_UNDERSCORES | `MAX_RETRIES`, `API_TIMEOUT` |
| **Private** | _leading_underscore | `_internal_helper()`, `_cache` |

**Examples:**
```python
# Module: enhanced_trends_pipeline.py

MAX_KEYWORDS = 10  # Constant

class EnhancedTrendsPipeline:  # Class (PascalCase)
    def __init__(self):
        self._cache = {}  # Private attribute

    def generate_prompt(self, keywords):  # Public method
        return self._build_prompt(keywords)  # Private method

    def _build_prompt(self, keywords):
        prompt_text = "..."  # Variable
        return prompt_text
```

---

## Code Organization

### File Structure

**Standard Python Module Structure:**
```python
"""
Module docstring - 簡短說明模組用途

Author: [Name]
Date: [YYYY-MM-DD]
Version: [X.Y]
"""

# 1. Standard library imports
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party imports
import torch
import numpy as np
import pandas as pd

# 3. Local imports
from obj1_nlp_prompt.prompt_generator import PromptGenerator

# 4. Constants
MAX_RETRIES = 3
API_TIMEOUT = 60

# 5. Logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 6. Classes and functions
class MyClass:
    ...

def my_function():
    ...

# 7. Main execution (if applicable)
if __name__ == "__main__":
    main()
```

### Import Order

**Follow isort standard:**
1. Standard library
2. Third-party packages
3. Local modules

**Use absolute imports** for local modules:
```python
# Good
from obj1_nlp_prompt.prompt_generator import PromptGenerator

# Avoid
from .prompt_generator import PromptGenerator  # Relative import
```

---

## Documentation Standards

### Docstrings

**Use Google Style Docstrings:**

```python
def predict_sales(
    season: str,
    clip_embedding: np.ndarray,
    trends_history: List[float]
) -> Dict[str, float]:
    """
    預測指定季節的銷量。

    Args:
        season: 季節名稱 (Spring/Summer/Fall/Winter)
        clip_embedding: CLIP embedding vector (768-dim)
        trends_history: 過去 4 季度的 Google Trends 分數

    Returns:
        包含預測結果的字典:
        {
            'predicted_sales': 預測銷量,
            'confidence': 信心度 (R²),
            'lower_bound': 下限 (predicted - MAE),
            'upper_bound': 上限 (predicted + MAE)
        }

    Raises:
        ValueError: 如果 season 不在允許值中
        RuntimeError: 如果模型未載入

    Example:
        >>> model = ForecastPredictor()
        >>> result = model.predict_sales(
        ...     season="Spring",
        ...     clip_embedding=np.random.rand(768),
        ...     trends_history=[45, 52, 48, 50]
        ... )
        >>> print(result['predicted_sales'])
        1523.45
    """
    ...
```

**Docstring Requirements:**
- **All public functions** must have docstrings
- **All classes** must have class-level docstrings
- **Private functions** (`_function`) - optional but recommended
- **Modules** - required docstring at top

### Comments

**Use comments for "why", not "what":**

```python
# Good - Explains WHY
# Use exponential backoff to avoid hitting API rate limit
time.sleep(2 ** retry_count)

# Avoid - States the obvious
# Sleep for 2 seconds
time.sleep(2)
```

**Chinese Comments Allowed:**
- 繁體中文註解係允許嘅（本項目主要使用繁體中文）
- 但建議 docstrings 使用中英混合（function signature 用英文，說明可用中文）

---

## Type Hints

### Type Annotation Standards

**Required for:**
- All public function signatures
- Function parameters and return types
- Complex data structures

**Optional for:**
- Local variables (if obvious from context)
- Private functions (but recommended)

**Examples:**
```python
from typing import Dict, List, Optional, Tuple

# Function with type hints
def generate_prompt(
    character_name: str,
    character_desc: str,
    trend_keywords: List[str],
    max_length: int = 200
) -> str:
    """Generate AI image prompt."""
    ...

# Complex return type
def fetch_trends(keywords: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch Google Trends data."""
    ...

# Optional parameter
def load_model(model_path: Optional[str] = None) -> torch.nn.Module:
    """Load model from path or default location."""
    ...

# Multiple return values
def compute_metrics(y_true, y_pred) -> Tuple[float, float, float]:
    """Compute MAE, RMSE, R²."""
    ...
```

---

## Error Handling

### Exception Handling Pattern

**Use specific exceptions:**
```python
# Good
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.exceptions.Timeout:
    logger.error(f"API timeout: {url}")
    raise APITimeoutError(f"Request to {url} timed out")
except requests.exceptions.HTTPError as e:
    logger.error(f"HTTP error: {e}")
    raise APIError(f"API returned error: {e}")

# Avoid - Too broad
try:
    response = requests.get(url)
except Exception as e:
    print(f"Error: {e}")
```

### Custom Exceptions

**Define custom exceptions for domain-specific errors:**
```python
class TrendsAPIError(Exception):
    """Raised when Google Trends API fails."""
    pass

class PromptGenerationError(Exception):
    """Raised when LLM prompt generation fails."""
    pass

class ModelLoadError(Exception):
    """Raised when model weights fail to load."""
    pass
```

### Logging Standards

**Use Python logging module:**
```python
import logging

logger = logging.getLogger(__name__)

# Log levels usage
logger.debug("Detailed info for debugging")          # Development only
logger.info("General informational message")         # Normal operation
logger.warning("Something unexpected happened")      # Potential issue
logger.error("Error occurred, operation failed")     # Error but recoverable
logger.critical("Critical failure, system unstable") # Rare, severe errors
```

**Logging Format:**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## Streamlit-Specific Standards (Obj 4)

### Session State Management

**Naming Convention:**
```python
# Good - Clear, descriptive keys
st.session_state['generated_prompt']
st.session_state['clip_embeddings']
st.session_state['prediction_results']

# Avoid - Ambiguous keys
st.session_state['data']
st.session_state['temp']
```

**Initialization Pattern:**
```python
# Initialize session state
if 'generated_prompt' not in st.session_state:
    st.session_state['generated_prompt'] = None
```

### Caching

**Use Streamlit cache decorators:**
```python
# Cache expensive computations
@st.cache_data(ttl=3600)  # 1 hour TTL
def fetch_google_trends(keywords: List[str]) -> pd.DataFrame:
    """Fetch trends data (cached)."""
    ...

# Cache resources (models, connections)
@st.cache_resource
def load_transformer_model(model_path: str) -> torch.nn.Module:
    """Load model (cached across sessions)."""
    ...
```

---

## Testing Standards

### Test File Naming

```
tests/
├── test_trends_api.py           # Unit tests for trends_api.py
├── test_design_generator.py     # Unit tests for design_generator.py
└── test_e2e_pipeline.py         # End-to-end integration test
```

### Test Function Naming

```python
# Pattern: test_<function_name>_<scenario>
def test_generate_prompt_with_valid_input():
    """Test prompt generation with valid keywords."""
    ...

def test_generate_prompt_with_empty_keywords():
    """Test prompt generation handles empty keywords."""
    ...

def test_clip_similarity_above_threshold():
    """Test CLIP similarity returns expected value."""
    ...
```

### Test Structure (AAA Pattern)

```python
def test_predict_sales_returns_dict():
    """Test sales prediction returns correct dict structure."""
    # Arrange - Setup test data
    model = ForecastPredictor()
    season = "Spring"
    embedding = np.random.rand(768)
    trends = [45, 52, 48, 50]

    # Act - Execute function
    result = model.predict_sales(season, embedding, trends)

    # Assert - Verify expectations
    assert isinstance(result, dict)
    assert 'predicted_sales' in result
    assert 'confidence' in result
    assert result['predicted_sales'] > 0
```

---

## Git Commit Standards

### Conventional Commits

**Format:** `<type>(<scope>): <subject>`

**Types:**
- `feat`: 新功能
- `fix`: Bug 修復
- `docs`: 文檔更新
- `style`: 程式碼格式（不影響功能）
- `refactor`: 重構
- `test`: 測試相關
- `chore`: 雜項（依賴更新等）

**Examples:**
```bash
feat(obj4): 新增 Streamlit 基礎架構
fix(obj2): 修復 CLIP 相似度計算錯誤
docs(architecture): 更新 tech-stack.md
refactor(obj3): 重構 Transformer forward 方法
test(obj1): 新增 trends_api 單元測試
```

**Commit Message in Chinese:**
- 中文 commit message 係允許嘅
- 但建議使用 Conventional Commits 格式

---

## Code Review Checklist

### Before Submitting Code

- [ ] Code follows PEP 8 (line length 100)
- [ ] All public functions have docstrings
- [ ] Type hints added to function signatures
- [ ] No hardcoded API keys or secrets
- [ ] Logging used instead of print statements
- [ ] Error handling covers edge cases
- [ ] Tests added for new functionality
- [ ] No commented-out code blocks (remove or explain)
- [ ] Import order is clean (standard → third-party → local)

### Reviewer Focus Areas

1. **Correctness** - Does it work as intended?
2. **Readability** - Can others understand it?
3. **Error Handling** - Are edge cases covered?
4. **Performance** - Are there obvious bottlenecks?
5. **Security** - Any API keys exposed?

---

## Common Anti-Patterns to Avoid

### ❌ Avoid These

**1. Bare except:**
```python
# Bad
try:
    result = api_call()
except:  # Too broad
    pass
```

**2. Mutable default arguments:**
```python
# Bad
def add_item(item, items=[]):  # Dangerous!
    items.append(item)
    return items

# Good
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

**3. Global variables:**
```python
# Bad
global_cache = {}  # Avoid global state

def fetch_data(key):
    global global_cache  # Dangerous
    ...

# Good - Use class attributes or pass as parameter
class DataFetcher:
    def __init__(self):
        self.cache = {}

    def fetch_data(self, key):
        ...
```

**4. String concatenation in loops:**
```python
# Bad
result = ""
for item in items:
    result += str(item)  # Slow for large lists

# Good
result = "".join(str(item) for item in items)
```

---

## Performance Guidelines

### Optimization Priorities

1. **Correctness First** - 先確保功能正確
2. **Readability Second** - 保持程式碼可讀
3. **Performance Last** - 只在必要時優化

### Common Optimizations

**Cache expensive operations:**
```python
@functools.lru_cache(maxsize=128)
def expensive_computation(param):
    ...
```

**Use generators for large datasets:**
```python
# Good - Memory efficient
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield process(line)

# Avoid - Loads entire file
def read_large_file(path):
    with open(path) as f:
        return [process(line) for line in f]  # Memory hungry
```

---

## Security Best Practices

### API Keys & Secrets

**Never commit secrets:**
```python
# Bad
API_KEY = "sk-1234567890abcdef"  # NEVER do this

# Good
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")
```

### Input Validation

**Always validate user input:**
```python
def predict_sales(season: str, ...):
    # Validate input
    allowed_seasons = ["Spring", "Summer", "Fall", "Winter"]
    if season not in allowed_seasons:
        raise ValueError(f"Invalid season: {season}. Must be one of {allowed_seasons}")
    ...
```

---

## Tools & Automation (Recommended)

### Linting

```bash
# Install Ruff (modern, fast linter)
pip install ruff

# Run linter
ruff check obj1_nlp_prompt/
```

### Formatting

```bash
# Install Black
pip install black

# Format code
black obj1_nlp_prompt/ --line-length 100
```

### Type Checking

```bash
# Install mypy
pip install mypy

# Check types
mypy obj1_nlp_prompt/
```

---

## Project-Specific Conventions

### Obj 1-3 Specific

**Obj 1 (NLP):**
- Prompt 長度限制: 150-200 words
- 關鍵字數量: Top 10
- Region: 'HK', Language: 'zh-TW'

**Obj 2 (Image Gen):**
- CLIP 相似度門檻: ≥ 0.80
- Reference Image: 固定使用 `lulu_pig_ref_1/2/3.jpg`

**Obj 3 (Forecasting):**
- Model input shape: (batch, 4, 1) for time-series, (batch, 772) for static
- Prediction unit: 銷量（件數）

### File Naming

**Data Files:**
- Prompts: `prompt_{theme}_{timestamp}.txt`
- Images: `{character}_{theme}_{timestamp}_var{n}.png`
- Models: `best_model.pth` (固定名稱)

---

**Document Owner:** Architect (Winston)
**Approved By:** Development Team
**Effective Date:** 2025-11-06
**Review Cycle:** Quarterly or after major updates
