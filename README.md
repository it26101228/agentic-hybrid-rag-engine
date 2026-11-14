# ⚡ Agentic Hybrid RAG Engine — Tech Vault Sri Lanka

![Python 3.12.10](https://img.shields.io/badge/Python-3.12.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Smolagents](https://img.shields.io/badge/HuggingFace-smolagents-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastEmbed ONNX](https://img.shields.io/badge/FastEmbed-bge--small--en--v1.5-8A2BE2?style=for-the-badge)
![Groq Cloud](https://img.shields.io/badge/Groq_Cloud-gpt--oss--20b-F55036?style=for-the-badge)
![Evals Passing](https://img.shields.io/badge/Automated_Evals-4%2F4_Passing_(100%25)-00C853?style=for-the-badge)
![SLIIT Architect](https://img.shields.io/badge/SLIIT_Architect-it26101228-00897B?style=for-the-badge)

> [!IMPORTANT]
> **Executive Summary:** An industrial, modular **Agentic Hybrid Retrieval-Augmented Generation (RAG)** system built in Python. It combines **deterministic financial & tax tools** (18% Sri Lankan VAT & manager approval bouncers) with a **50/50 Hybrid Search Engine** (Exact Keyword Stemming + 384-d Neural Vector Embeddings via ONNX Runtime).

---

## 🏗️ 1. System Architecture & Modular Blueprint

Unlike single-notebook prototypes, this repository separates **Ingestion**, **Hybrid Retrieval**, **Agentic Tool Execution**, and **Automated Evaluations** into isolated Python packages:

```python
# Project Directory Architecture (it26101228/agentic-hybrid-rag-engine)
agentic_hybrid_rag_engine = {
    "data/": {
        "shop_policies.txt": "Official store warranty, dead-pixel & Sri Lanka courier manual"
    },
    "src/ingestion/": {
        "chunker.py": "Sliding-window word slicer with 3 strict input bouncers",
        "loader.py": "Path/folder guards & metadata card factory (chunk_id, source, text)"
    },
    "src/retrieval/": {
        "retriever.py": "Detective 1: Keyword search with stopword removal & plural stemming",
        "vector_search.py": "Detective 2: 384-d neural semantic search (BAAI/bge-small-en-v1.5)",
        "hybrid_retriever.py": "Hybrid Boss: 50/50 normalized score fusion & chunk_id deduplication"
    },
    "src/agent/": {
        "tools.py": "4 @tool functions: pricing, 18% VAT math, approvals & Hybrid RAG",
        "agent_runner.py": "ToolCallingAgent + GroqSafeModel (tool_choice='auto' & final_answer guard)"
    },
    "src/evals/": {
        "test_agent_bouncers.py": "4-test automated evaluation suite with strict assert checks"
    }
}
```

---

## 🧠 2. Core Engineering Capabilities

### 🛠️ A. Deterministic Pricing, Tax & Approval Pipeline (`src/agent/tools.py`)
LLMs should never guess arithmetic or business rules from memory. The agent delegates all calculations to deterministic Python tools:
* 💵 **`get_item_price(item_name)`:** Looks up verified base prices in LKR (`laptop`: **250,000 LKR**, `monitor`: **75,000 LKR**, `mouse`: **3,500 LKR**) and blocks missing inventory items (`playstation`) with an explicit anti-hallucination warning.
* 🧮 **`calculate_final_price(original_price)`:** Applies exact **18% Sri Lankan VAT** (`original_price * 1.18`).
* 🛡️ **`check_approval_status(final_price)`:** Enforces corporate purchasing rules (`AUTO_APPROVED` for orders up to 100,000 LKR; `MANAGER_APPROVAL_REQUIRED` above 100,000 LKR).

### 🔍 B. Hybrid RAG Engine: Keyword + Neural Meaning Search (`src/retrieval/`)
To solve both the **exact-code matching problem** and the **vocabulary mismatch problem**, `hybrid_retriever.py` combines two detectives on a normalized `0.0` to `1.0` scale:

1. **Keyword Detective (`retriever.py`):** Strips punctuation, removes English stopwords, normalizes plurals (`costs` -> `cost`, `pixels` -> `pixel`), and computes:
   $$\text{kw\_norm} = \min(\text{keyword\_score} \times 0.25,\ 1.0)$$
2. **Neural Vector Detective (`vector_search.py`):** Embeds queries and chunks into **384-dimensional meaning vectors** using `BAAI/bge-small-en-v1.5` on CPU, filtering out background noise below `min_score = 0.58`.
3. **50/50 Hybrid Score Fusion (`hybrid_retriever.py`):** Deduplicates cards by `chunk_id` and ranks results by:
   $$\text{hybrid\_score} = (0.5 \times \text{kw\_norm}) + (0.5 \times \text{semantic\_score})$$

> [!TIP]
> **Real-World Hybrid Rescue Example:** When a customer asks *"My screen has tiny black dots that won't light up"*, **0 exact keywords** match the policy manual (`keyword_score = 0`). The Neural Vector Detective recognizes that *"tiny black dots on a screen"* means **`"monitor dead pixels"`** (`semantic_score = 0.6002`) and automatically rescues **Card 3** (`Hybrid Score: 0.3001`)!

---

## ✅ 3. Automated End-to-End Evaluation Suite (`src/evals/test_agent_bouncers.py`)

Every architectural milestone is verified by automated `assert` crash-tests:

| Test ID | Evaluation Scenario | Customer Query / Test Input | Verified System Outcome | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Test 1** | **LKR VAT & Auto-Approval** | Final price of a `mouse` with 18% VAT + approval status | Computes **`4,130 LKR`** & returns **`AUTO_APPROVED`** | 🟢 **PASSED** |
| **Test 2** | **Missing-Inventory Bouncer** | Request price with VAT for a `playstation` | Blocks missing item; **0% fake price hallucination** | 🟢 **PASSED** |
| **Test 3** | **RAG Policy & Pizza Guard** | Laptop warranty above 200,000 LKR + `pepperoni pizza` policy | Pulls **3-year** warranty (`Card 1`) & rejects pizza query | 🟢 **PASSED** |
| **Test 4** | **Hybrid Meaning Rescue** | `"My screen has tiny black dots that won't light up"` | Rescues **`Card 3`** (`dead pixels` policy, `Score: 0.3001`) | 🟢 **PASSED** |

---

## 🚀 4. Quickstart & Local Reproduction

> [!WARNING]
> **API Key & Disk Cache Protection:** Store your free Groq API key inside a local `.env` file (`GROQ_API_KEY=your_key_here` — protected by `.gitignore`). Neural weights (~67 MB) are automatically routed to `D:/AI_Vault/huggingface_cache` via `HF_HOME` to keep the primary `C:` SSD clean.

```powershell
# 1. Clone the repository
git clone [https://github.com/it26101228/agentic-hybrid-rag-engine.git](https://github.com/it26101228/agentic-hybrid-rag-engine.git)
cd agentic-hybrid-rag-engine

# 2. Create and activate your virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the 4-tool Agentic RAG customer mission
python src/agent/agent_runner.py

# 5. Run the 4-test automated evaluation suite
python src/evals/test_agent_bouncers.py
```