# 🛤️ Streamlit Path Handling Guide

## 📂 **Repository Structure Overview**

```
D:\Oxford\Extra\Finance_NLP\alpha-agents\
├── Home.py                           # Main landing page
├── pages/
│   ├── 1_Fundamental_Screener.py     # Uses main repo utils
│   └── 2_QualAgent_Analysis.py       # Uses QualAgent subdirectory
├── utils/                            # Main repo utilities
│   ├── stock_screener.py            # For Fundamental Screener
│   └── db_util.py                   # Database utilities
└── agents/QualAgent/                 # QualAgent subdirectory
    ├── utils/                       # QualAgent-specific utilities
    │   ├── test_llm_api.py          # API testing
    │   └── weight_manager.py        # Weight management
    └── engines/                     # QualAgent engines
        └── enhanced_scoring_system.py
```

## 🔧 **Path Handling Solutions**

### **Page 1: Fundamental Screener**
**File**: `pages/1_Fundamental_Screener.py`
**Imports**: Main repo utils (`utils/stock_screener.py`, `utils/db_util.py`)

```python
# Add main repo utils to path for Fundamental Screener
main_repo_path = Path(__file__).parent.parent
sys.path.insert(0, str(main_repo_path))

from utils.stock_screener import StockScreener
from utils.db_util import save_fundamental_screen
```

### **Page 2: QualAgent Analysis**
**File**: `pages/2_QualAgent_Analysis.py`
**Imports**: QualAgent subdirectory (`agents/QualAgent/utils/`, `agents/QualAgent/engines/`)

```python
# Add QualAgent to path
qual_agent_path = Path(__file__).parent.parent / "agents" / "QualAgent"
sys.path.insert(0, str(qual_agent_path))

from utils.test_llm_api import LLMAPITester
from utils.weight_manager import InteractiveWeightManager
from engines.enhanced_scoring_system import WeightingScheme
```

## 📋 **Import Resolution Strategy**

### **For Fundamental Screener (Page 1)**:
1. **Target**: Main repo utilities
2. **Path**: `D:\Oxford\Extra\Finance_NLP\alpha-agents\utils\`
3. **Method**: Add parent directory to sys.path
4. **Result**: Can import `utils.stock_screener`, `utils.db_util`

### **For QualAgent Analysis (Page 2)**:
1. **Target**: QualAgent subdirectory utilities
2. **Path**: `D:\Oxford\Extra\Finance_NLP\alpha-agents\agents\QualAgent\`
3. **Method**: Add QualAgent directory to sys.path
4. **Fallback**: Direct module imports if package imports fail
5. **Result**: Can import `utils.test_llm_api`, `engines.enhanced_scoring_system`

## ✅ **Verification Steps**

### **Test Fundamental Screener**:
```bash
streamlit run Home.py
# Navigate to "📊 Open Fundamental Screener"
# Should load without import errors
```

### **Test QualAgent Analysis**:
```bash
streamlit run Home.py
# Navigate to "🧠 Open QualAgent Analysis"
# Should show "✅ Standard package imports successful!" in sidebar
```

## 🚨 **Troubleshooting**

### **If Fundamental Screener fails**:
- Check `D:\Oxford\Extra\Finance_NLP\alpha-agents\utils\stock_screener.py` exists
- Verify `utils/__init__.py` exists in main repo
- Check error message for specific missing module

### **If QualAgent Analysis fails**:
- Check `D:\Oxford\Extra\Finance_NLP\alpha-agents\agents\QualAgent\utils\test_llm_api.py` exists
- Verify all `__init__.py` files exist in QualAgent subdirectories
- Check sidebar debug information for path details

## 🎯 **Why Different Path Handling?**

**Fundamental Screener** was built as part of the main repository and expects to import from the main `utils/` directory.

**QualAgent Analysis** is a specialized module with its own subdirectory structure, requiring imports from `agents/QualAgent/utils/` and `agents/QualAgent/engines/`.

**Both approaches are correct** - they just target different parts of the codebase based on where their dependencies are located.

## 🔄 **Path Resolution Flow**

```
pages/1_Fundamental_Screener.py
├── Adds: D:\Oxford\Extra\Finance_NLP\alpha-agents\
├── Imports: utils.stock_screener (from main repo)
└── Result: ✅ Fundamental screening functionality

pages/2_QualAgent_Analysis.py
├── Adds: D:\Oxford\Extra\Finance_NLP\alpha-agents\agents\QualAgent\
├── Imports: utils.test_llm_api (from QualAgent subdirectory)
└── Result: ✅ Enhanced qualitative analysis functionality
```

This dual-path approach ensures both pages can access their respective dependencies without conflicts! 🎯✅