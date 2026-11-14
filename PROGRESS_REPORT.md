# 📊 Master Sprint Progress Report: Applied AI Systems Architect

![Sprint Status](https://img.shields.io/badge/Sprint_Status-Days_1--5_Completed-00C853?style=for-the-badge)
![SLIIT Student ID](https://img.shields.io/badge/SLIIT_Architect-it26101228-00897B?style=for-the-badge)
![Evaluations](https://img.shields.io/badge/End--to--End_Evals-4%2F4_Passing_(100%25)-3776AB?style=for-the-badge)

* **Student ID:** `it26101228` (SLIIT IT — Year 1, Semester 2)
* **Repository:** `it26101228/agentic-hybrid-rag-engine`
* **Hardware & Vault Config:** Intel Core i5 (11th Gen), 16 GB RAM, `HF_HOME` isolated on `D:\AI_Vault\huggingface_cache`

---

## 🧭 Executive Summary

This ledger tracks the day-by-day engineering milestones of the **Tech Vault Sri Lanka Agentic Hybrid RAG Engine**. Over 5 days, the system evolved from deterministic LKR tax/approval functions into a full **Hybrid Retrieval-Augmented Generation (RAG)** architecture combining exact keyword stemming with 384-dimensional neural embeddings (`BAAI/bge-small-en-v1.5`) and automated end-to-end assertions.

---

## 🗂️ Master Module Ledger (Days 1–5)

| Module / Path | Component Name | Core Architect Rules & Features | Status |
| :--- | :--- | :--- | :--- |
| **`data/shop_policies.txt`** | Official Store Manual | Warranty rules (3-yr laptop >200k LKR, 2-yr monitor), 30-day dead-pixel replacement, & Sri Lanka courier fees. | 🟢 **Verified** |
| **`src/ingestion/chunker.py`** | Sliding-Window Chunker | Configurable `chunk_size` & `overlap` word slicer protected by 3 strict `ValueError` bouncers. | 🟢 **Verified** |
| **`src/ingestion/loader.py`** | Metadata Card Loader | Validates file paths (`exists()`, `is_file()`) and packages chunks into dicts (`chunk_id`, `source`, `text`). | 🟢 **Verified** |
| **`src/retrieval/retriever.py`** | Keyword Detective | Regex punctuation cleaning, stopword filtering, plural stemming (`costs` -> `cost`), and `top_k` ranking. | 🟢 **Verified** |
| **`src/retrieval/vector_search.py`** | Neural Vector Detective | 384-d ONNX embeddings (`BAAI/bge-small-en-v1.5` via `fastembed`), cosine similarity, & `min_score=0.58` bouncer. | 🟢 **Verified** |
| **`src/retrieval/hybrid_retriever.py`** | 50/50 Hybrid Boss | Combines normalized keyword scores (`kw_norm`) & neural scores (`semantic_score`), deduplicated by `chunk_id`. | 🟢 **Verified** |
| **`src/agent/tools.py`** | 4-Tool Agent Suite | `get_item_price`, `calculate_final_price` (18% VAT), `check_approval_status` (200k LKR rule), & `search_shop_policies`. | 🟢 **Verified** |
| **`src/agent/agent_runner.py`** | Production Agent Runner | `ToolCallingAgent` + custom `GroqSafeModel` (`openai/gpt-oss-20b`, `tool_choice="auto"`, `final_answer` wrapper). | 🟢 **Verified** |
| **`src/evals/test_agent_bouncers.py`** | 4-Test Evaluation Suite | Automated `assert` suite verifying VAT math, PlayStation bouncer, RAG policy + pizza guard, and Hybrid rescue. | 🟢 **Verified** |

---

## 📅 Day-by-Day Architectural Breakdown

### 🛠️ Days 1–3: Deterministic Financial Pipeline & Agent Bouncers
* **Environment & Vault Isolation:** Configured Python `3.12.10` in `.venv`, locked `.env` secrets behind `.gitignore`, and redirected Hugging Face cache files to `D:\AI_Vault\huggingface_cache`.
* **Deterministic Pricing & Sri Lankan VAT (`src/agent/tools.py`):**
  * Built `get_item_price` for verified inventory (`laptop`: 250,000 LKR, `monitor`: 75,000 LKR, `mouse`: 3,500 LKR) with an explicit anti-hallucination return string for missing items (`playstation`).
  * Built `calculate_final_price` to compute exact **18% Sri Lankan VAT** (`original_price * 1.18`).
  * Built `check_approval_status` enforcing `AUTO_APPROVED` up to **200,000 LKR** and `MANAGER_APPROVAL_REQUIRED` above **200,000 LKR**.
* **Cloud Agent Wrapper (`src/agent/agent_runner.py`):** Engineered `GroqSafeModel` to enforce `tool_choice="auto"` and wrap plain-text responses into `smolagents` `final_answer` calls.

### 📄 Day 4: Document Ingestion & Keyword RAG Detective
* **Sliding-Window Chunker (`src/ingestion/chunker.py`):** Built `chunk_text()` to slice policy documents into 25-word overlapping windows (`overlap=5`) so context is never cut in half at card boundaries.
* **Metadata Factory (`src/ingestion/loader.py`):** Built `load_and_chunk_file()` to attach `chunk_id` and `source` tracking to every text slice.
* **Keyword Retriever (`src/retrieval/retriever.py`):** Built `retrieve_top_chunks()` with custom stopword removal and plural normalization so customer queries accurately score matching policy cards.

### 🧠 Day 5: Neural Vector Search & 50/50 Hybrid RAG Engine
* **ONNX Neural Embeddings (`src/retrieval/vector_search.py`):** Integrated `fastembed` (`BAAI/bge-small-en-v1.5`, ~67.2 MB cached on `D:`) to convert queries and policy chunks into 384-dimensional vectors scored via Cosine Similarity (`min_score=0.58`).
* **50/50 Hybrid Fusion (`src/retrieval/hybrid_retriever.py`):** Merged Keyword and Neural detectives onto a unified `0.0` to `1.0` scale deduplicated by `chunk_id`:
  ```python
  kw_norm = min(card["keyword_score"] * 0.25, 1.0)
  hybrid_score = round((0.5 * kw_norm) + (0.5 * card["semantic_score"]), 4)
  ```
* **Zero-Keyword Semantic Rescue (`Test 4`):** Verified that querying `"My screen has tiny black dots that won't light up"` (`keyword_score = 0`) successfully retrieves **Card 3** (`dead pixels` replacement policy, `Hybrid Score: 0.3001`).