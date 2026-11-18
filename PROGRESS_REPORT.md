# 📊 Applied AI Systems Architect Sprint — Living Master Report (Days 0–6)

![Sprint Status](https://img.shields.io/badge/Sprint_Status-Week_1_Complete_(Days_0--6)-00C853?style=for-the-badge)
![SLIIT Architect](https://img.shields.io/badge/SLIIT_Architect-it26101228-00897B?style=for-the-badge)
![Automated Evals](https://img.shields.io/badge/Automated_Evals-4%2F4_Passing_(100%25)-3776AB?style=for-the-badge)
![Hybrid Hit Rate](https://img.shields.io/badge/Hybrid_RAG_Hit_Rate-100.0%25_(11%2F11)-2962FF?style=for-the-badge)
![Workdays Logged](https://img.shields.io/badge/Sprint_Ledger-7%2F54_Workdays_Done-6A1B9A?style=for-the-badge)

> [!IMPORTANT]
> **System Handover & Mentorship Protocol:**
> - **Architect:** Diman Randinu Mudunkothge (`dimanrandinu` | SLIIT Y1S2 IT — Student ID: `it26101228`)
> - **Repository:** `https://github.com/it26101228/agentic-hybrid-rag-engine`
> - **Local Workstation:** 11th Gen Intel Core i5 | 16 GB RAM | 256 GB SSD (`C:`) + 1 TB HDD (`D:\AI_Vault`) | Windows 11 | Python `3.12.10` (`.venv`)
> - **Compute & Inference Stack:** Local CPU ONNX Runtime (`fastembed` / `BAAI/bge-small-en-v1.5`, ~67.2 MB cached at `D:\AI_Vault\huggingface_cache`) + Kaggle Dual NVIDIA T4 GPUs (`2 x 16 GB = 32 GB VRAM`, 30 hrs/wk) + Groq LPU Cloud Inference (`openai/gpt-oss-20b` via `GroqSafeModel`)
> - **4 Non-Negotiable Mentorship Rules:** (1) 360-Degree Impact Evaluation (Skill, Codebase, Lifestyle/Health, Career); (2) Simple English + Plain-Language Definitions for Every New Term; (3) Strict Fact-Checking (Zero Guessing); (4) Always Explain **"WHY?"** and **"FOR WHAT?"** for Every Step.
> - **Last Updated:** Wednesday, November 18, 2026 (End of Week 1 / Day 6 Verified Closeout — 7 of 54 Engineering Workdays Complete; 47 Workdays Remaining)

---

## 🧭 Executive Summary

This Living Master Report is the single source of truth tracking the engineering architecture, debugging evolutions, benchmark artifacts, and daily milestones of the **Tech Vault Sri Lanka Agentic Hybrid RAG Engine**. Across **Week 1 (Pre-Flight Day 0 + Days 1–6)**, the system evolved from cloud GPU experiments into a modular, 5-package Python production architecture combining **deterministic LKR pricing/VAT/approval tools** with a **50/50 Hybrid Retrieval-Augmented Generation (RAG) Engine** (`Keyword Stemming + 384-d ONNX Cosine Similarity`) verified by a **100% (4/4) automated evaluation suite** and an **11-query CPU retrieval benchmark (`100.0%` Hybrid Hit Rate vs. `81.8%` Keyword-Only)**.

---

## 🗂️ Section 1: Master Module & Verification Ledger (Days 0–6)

| Module / Path | Component Name | Core Architect Rules & Verified Features | Sprint Day | Status |
| :--- | :--- | :--- | :---: | :---: |
| **`Notebooks/day1_agent_benchmark.ipynb`** | GPU Agent Lab | First `CodeAgent` on Kaggle Dual T4 GPUs benchmarking open LLMs (`day1_gpu_benchmarks.csv`). | Day 1 | 🟢 **Verified** |
| **`Notebooks/day2-tokenizers-and-detective-agent.ipynb`** | Tokenizer & Detective Lab | Subword tokenization (`input_ids`, `attention_mask`) & multi-step Detective Agent workflow. | Day 2 | 🟢 **Verified** |
| **`data/shop_policies.txt`** | Official Store Manual | 182-word manual: 3-yr laptop warranty (>200k LKR), 2-yr monitor + 30-day dead-pixel rule, & SL courier fees. | Day 4 | 🟢 **Verified** |
| **`src/ingestion/chunker.py`** | Sliding-Window Chunker | Configurable `chunk_size=25` & `overlap=5` (`step=20`) slicer protected by 3 strict `ValueError` bouncers. | Day 4 | 🟢 **Verified** |
| **`src/ingestion/loader.py`** | Metadata Card Factory | Validates file paths (`exists()`, `is_file()`) and outputs 9 structured cards (`chunk_id`, `source`, `text`). | Day 4 | 🟢 **Verified** |
| **`src/retrieval/retriever.py`** | Keyword Detective | Regex punctuation stripping, stopword filtering, plural stemming (`costs` -> `cost`), & `top_k` ranking. | Day 4 | 🟢 **Verified** |
| **`src/retrieval/vector_search.py`** | Neural Vector Detective | 384-d ONNX embeddings (`BAAI/bge-small-en-v1.5` via `fastembed`), Cosine Similarity, & `min_score=0.58` bouncer. | Day 5 | 🟢 **Verified** |
| **`src/retrieval/hybrid_retriever.py`** | 50/50 Hybrid Boss | Normalized fusion (`0.5 * kw_norm + 0.5 * semantic_score`), `chunk_id` deduplication, & `>= 0.20` noise gate. | Day 5 | 🟢 **Verified** |
| **`src/agent/tools.py`** | 4-Tool Agent Suite | `get_item_price` (LKR), `calculate_final_price` (18% VAT), `check_approval_status` (>200k LKR), & `search_shop_policies`. | Days 3–5 | 🟢 **Verified** |
| **`src/agent/agent_runner.py`** | Production Agent Runner | `build_shop_agent()` factory using `ToolCallingAgent` + custom `GroqSafeModel` (`tool_choice="auto"`). | Days 3–4 | 🟢 **Verified** |
| **`src/evals/test_agent_bouncers.py`** | 4-Test Evaluation Suite | Automated `assert` suite verifying VAT math (`4,130 LKR`), PlayStation bouncer, pizza guard, & Hybrid rescue. | Days 3–5 | 🟢 **Verified** |
| **`benchmarks/benchmark_retrievers.py`** | 11-Query Benchmark Harness | Measures Keyword vs. Neural vs. Hybrid Hit Rate % and CPU latency (ms), exporting `day6_retriever_benchmarks.csv`. | Day 6 | 🟢 **Verified** |

---

## 📅 Section 2: Day-by-Day Engineering Log (Days 0–6)

### ⚙️ Pre-Flight (`Sunday, November 8, 2026`) — Hardware Protection & Python Rust-Buster
- **What Happened:**
  - Verified Kaggle phone number to unlock **Dual NVIDIA T4 GPUs (`2 x 16 GB = 32 GB VRAM`, 30 hrs/week)** so local integrated Intel Iris Xe graphics never overload on heavy LLM weights.
  - Created Hugging Face account (`dimanrandinu`), stored `HF_TOKEN` inside Kaggle Secrets, and completed the 90-minute Python Rust-Buster (type-hinted `@tool` functions, dictionaries/JSON, classes, and `pipeline()`).
  - Configured `D:\AI_Vault` and routed `HF_HOME` to `D:\AI_Vault\huggingface_cache` to keep the 256 GB `C:` boot SSD clean.
- **Core Lessons (`WHY` & `FOR WHAT`):**
  - *Hardware Division of Labor:* Local CPU (`D:\AI_Vault` + VS Code) handles clean modular `.py` engineering and Git; cloud GPUs/LPUs handle heavy matrix math.
  - *Type Hints Are Functional in Agentic AI:* `smolagents` reads `item_name: str -> str` and docstrings to generate strict JSON tool schemas for the LLM.

### 🚀 Day 1 (`Monday, November 9, 2026`) — Transformer Architecture, First GPU Agent & Repo Scaffolding
- **What Happened:**
  - Completed Hugging Face LLM Course Chapter 1 (`1.1–1.6`) and Agents Course Unit 1, mastering the **ReAct loop (`Think -> Act -> Observe`)**.
  - Built and executed `Notebooks/day1_agent_benchmark.ipynb` on Kaggle T4 GPUs, benchmarking open LLMs across VRAM usage, `tokens/sec`, and output validity (`benchmarks/day1_gpu_benchmarks.csv`).
  - Scaffolded the public GitHub repository (`it26101228/agentic-hybrid-rag-engine`) with separated folders (`src/ingestion/`, `src/retrieval/`, `src/agent/`, `src/evals/`, `benchmarks/`, `Notebooks/`).
- **Core Lessons (`WHY` & `FOR WHAT`):**
  - *Encoders vs. Decoders:* Encoders (`bge-small-en-v1.5`) read text bidirectionally to build search embeddings; Decoders (`gpt-oss-20b`) predict tokens forward to reason and call tools.

### 🔬 Day 2 (`Tuesday, November 10, 2026`) — Subword Tokenizers & The 3 Unbreakable Habits
- **What Happened:**
  - Built `Notebooks/day2-tokenizers-and-detective-agent.ipynb` inspecting how `AutoTokenizer` splits human text into `input_ids` and `attention_mask`, and built a multi-step Detective Agent.
  - Locked in the **3 Unbreakable Sprint Habits:** (1) *Paper Blueprint First Rule*, (2) *Saturday Solo Build*, and (3) *8:30 PM Hard Shutdown*.
- **Core Lessons (`WHY` & `FOR WHAT`):**
  - *Why Notebooks Don't Belong in Production:* `.ipynb` files wrap code inside heavy JSON cell metadata; production systems belong in modular `.py` packages under `src/`.

### 🛠️ Day 3 (`Wednesday, November 11, 2026`) — Local `.venv` Migration, 3-Tool LKR Pipeline & `GroqSafeModel`
- **What Happened:**
  - Migrated to local Windows 11 VS Code + Python `3.12.10` (`.venv`) on `D:\AI_Vault`.
  - Built `src/agent/tools.py` with 3 deterministic `@tool` functions (`get_item_price`, `calculate_final_price` with 18% VAT, `check_approval_status` with `> 200,000 LKR` cutoff) and an `if __name__ == "__main__":` import gate.
  - Engineered `GroqSafeModel` in `src/agent/agent_runner.py` to bypass the 2026 Hugging Face Router credit-card gate using Groq's free LPU endpoint (`openai/gpt-oss-20b`) and resolve the `tool_choice` conflict (`kw["tool_choice"] = "auto"`).
  - Created `.env` + `.gitignore` secret protection, pinned `requirements.txt`, and built `src/evals/test_agent_bouncers.py` (catching and eliminating a `$499.99 USD` hallucination on missing item `'playstation'`).

### 📄 Day 4 (`Thursday–Friday, November 12–13, 2026`) — Document Ingestion & Keyword RAG Detective
- **What Happened:**
  - Created `data/shop_policies.txt` (182 words covering Tech Vault Sri Lanka warranties, dead-pixel rules, Sri Lankan courier fees, and return windows).
  - Built `src/ingestion/chunker.py` (`chunk_size=25`, `overlap=5`, `step=20`, 3 `ValueError` bouncers) and `src/ingestion/loader.py` (`exists()`/`is_file()` guards + metadata card packaging into **9 policy cards**).
  - Built `src/retrieval/retriever.py` (regex tokenization, stopword removal, plural stemming) and added `search_shop_policies` (`@tool #4`) to `src/agent/tools.py`.
- **Core Lessons (`WHY` & `FOR WHAT`):**
  - *Return Instruction Strings Instead of Crashing Tools:* Returning `"Error: 'playstation' is not in the shop inventory. Do NOT guess or invent a price."` allows the agent to report unavailability cleanly in 2 steps without crashing.

### 🧠 Day 5 (`Saturday, November 14, 2026`) — ONNX Neural Search, 50/50 Hybrid Fusion & 4/4 Evals
- **What Happened:**
  - Built `src/retrieval/vector_search.py` using `fastembed` (`BAAI/bge-small-en-v1.5`, 384-d embeddings, Cosine Similarity dot product, `min_score=0.58` noise bouncer).
  - Built `src/retrieval/hybrid_retriever.py` fusing normalized keyword scores (`kw_norm = min(keyword_score * 0.25, 1.0)`) and neural scores (`0.5 * kw_norm + 0.5 * semantic_score`) with `chunk_id` deduplication and a `>= 0.20` noise filter.
  - Upgraded `search_shop_policies` in `src/agent/tools.py` to call `retrieve_hybrid_chunks` with **zero changes** to `agent_runner.py` (clean encapsulation), updated inventory monitor price to `75,000 LKR`, and passed **4/4 tests** in `src/evals/test_agent_bouncers.py` (rescuing **Card 3** on *"My screen has tiny black dots that won't light up"*, `Hybrid Score: 0.3001`).

### 📊 Day 6 (`Monday, November 16, 2026`) — Quantitative CPU Retriever Benchmark & Portfolio Audit
- **Schedule Note:** Swapped Sunday, Nov 15 rest day with Tuesday, Jan 12, 2027 buffer day; logged Nov 17–18 as non-coding buffer days (`7/54` workdays complete).
- **What Happened:**
  - Built and executed `benchmarks/benchmark_retrievers.py` across **11 ground-truth queries** (`5 Keyword`, `4 Semantic`, `2 Out-of-Scope`) on local Intel Core i5 CPU with ONNX model warm-up (`top_k=2`).
  - Exported verified scorecard to `benchmarks/day6_retriever_benchmarks.csv`, proving **100.0% Hybrid Hit Rate (`11/11`)** vs. **81.8% Keyword-Only (`9/11`)**.
  - Audited and updated `README.md` (added `Hybrid RAG Hit Rate: 100.0% (11/11)` badge, `benchmarks/` directory tree, Section 4 benchmark table, fixed `200,000 LKR` threshold and `git clone` syntax) and pushed to `origin/main` (commit `e45d428`).

---

## 💻 Section 3: Production Architecture & Core Formulas

### A. Verified Repository Directory Map (`D:\AI_Vault\agentic-hybrid-rag-engine\`)
```text
agentic-hybrid-rag-engine/
├── .env                              # [LOCAL ONLY - .gitignore] GROQ_API_KEY
├── .venv/                            # [LOCAL ONLY - .gitignore] Python 3.12.10 virtual env
├── .vscode/settings.json             # [ON GITHUB] Pylance extraPaths for src/ subfolders
├── .gitignore                        # [ON GITHUB] Shields .env, .venv/, and __pycache__/
├── README.md                         # [ON GITHUB] Flagship portfolio documentation & benchmark tables
├── PROGRESS_REPORT.md                # [ON GITHUB] Living Master Report (Days 0–6)
├── requirements.txt                  # [ON GITHUB] smolagents, huggingface_hub, openai, fastembed, python-dotenv
├── data/
│   └── shop_policies.txt             # [ON GITHUB] 182-word official store policy & warranty manual (9 cards)
├── benchmarks/
│   ├── benchmark_retrievers.py       # [ON GITHUB] 11-query CPU Hit-Rate & latency benchmark harness
│   ├── day1_gpu_benchmarks.csv       # [ON GITHUB] Day 1 Kaggle T4 GPU model comparison
│   └── day6_retriever_benchmarks.csv # [ON GITHUB] Day 6 Lexical vs. Neural vs. Hybrid CPU scorecard
├── Notebooks/
│   ├── day1_agent_benchmark.ipynb    # [ON GITHUB] Day 1 Kaggle T4 GPU CodeAgent benchmark
│   └── day2-tokenizers-and-detective-agent.ipynb # [ON GITHUB] Day 2 Tokenizer & Detective Agent lab
└── src/
    ├── ingestion/
    │   ├── chunker.py                # [ON GITHUB] Sliding-window word slicer (chunk_size=25, overlap=5)
    │   └── loader.py                 # [ON GITHUB] Path validator & metadata card packager
    ├── retrieval/
    │   ├── retriever.py              # [ON GITHUB] Lexical keyword search + stopword/plural stemming
    │   ├── vector_search.py          # [ON GITHUB] FastEmbed ONNX 384-d Cosine Similarity search
    │   └── hybrid_retriever.py       # [ON GITHUB] 50/50 normalized score fusion & 0.20 noise filter
    ├── agent/
    │   ├── tools.py                  # [ON GITHUB] 4 @tool functions (LKR pricing, 18% VAT, approval, Hybrid RAG)
    │   └── agent_runner.py           # [ON GITHUB] build_shop_agent() + GroqSafeModel wrapper
    └── evals/
        └── test_agent_bouncers.py    # [ON GITHUB] 4/4 automated end-to-end assert evaluation suite
```

### B. Core Mathematical & Fusion Formulas
1. **Sliding-Window Chunker Step:**
   $$\text{step} = \text{chunk\_size} - \text{overlap} = 25 - 5 = 20\text{ words/step} \quad (182\text{ words} \rightarrow 9\text{ cards})$$
2. **Sri Lankan VAT & Corporate Compliance Gate (`src/agent/tools.py`):**
   $$\text{final\_price} = \text{round}(\text{original\_price} \times 1.18,\ 2)$$
   $$\text{Approval Status} = \begin{cases} \texttt{MANAGER\_APPROVAL\_REQUIRED} & \text{if } \text{final\_price} > 200,000\text{ LKR} \\ \texttt{AUTO\_APPROVED} & \text{if } \text{final\_price} \le 200,000\text{ LKR} \end{cases}$$
3. **50/50 Hybrid Score Fusion (`src/retrieval/hybrid_retriever.py`):**
   $$\text{kw\_norm} = \min(\text{keyword\_score} \times 0.25,\ 1.0)$$
   $$\text{hybrid\_score} = \text{round}\big((0.5 \times \text{kw\_norm}) + (0.5 \times \text{semantic\_score}),\ 4\big) \ge 0.20$$

---

## 📈 Section 4: Quantitative Benchmark & Evaluation Scorecards

### A. Day 6 Retrieval Benchmark Results (`benchmarks/day6_retriever_benchmarks.csv`)
Executed across **11 ground-truth queries** (`5 Keyword`, `4 Semantic`, `2 Out-of-Scope`) on local **11th Gen Intel Core i5 CPU** (warmed ONNX model, `top_k=2`):

| Retriever Engine | Overall Hit Rate | Keyword Hits (`5`) | Semantic Hits (`4`) | Out-of-Scope Guards (`2`) | Avg CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Keyword-Only (Lexical)** | `81.8%` (`9/11`) | `5/5` | `2/4` | `2/2` | `0.23 ms` |
| **2. Neural-Only (FastEmbed ONNX)** | `100.0%` (`11/11`) | `5/5` | `4/4` | `2/2` | `110.07 ms` |
| **3. 50/50 Hybrid RAG (Boss)** | **`100.0%` (`11/11`)** | **`5/5`** | **`4/4`** | **`2/2`** | **`116.51 ms`** |

- **Root-Cause Findings (`WHY` & `FOR WHAT`):**
  1. *Why Keyword-Only Scored `81.8%`:* Missed `Q5` (*"tiny black dots"* vs. *"dead pixels"*) due to zero shared vocabulary, and missed `Q7` (*"water or coffee gets spilled"* vs. *"liquid spills"*) because plural stemming (`-s` -> `""`) does not normalize past-tense `-ed` verbs.
  2. *Why Neural/Hybrid Averaged `110–116 ms`:* `vector_search.py` currently embeds all **9 policy cards + 1 query = 10 texts** on-the-fly per query (`~11 ms/text` on CPU). Persisting card vectors in a vector database (Week 2 target) will cut CPU search latency by ~90% (down to `~11–15 ms`).

### B. Automated End-to-End Agent Evaluation Matrix (`src/evals/test_agent_bouncers.py`)

| Test ID | Evaluation Scenario | Customer Query / Input | Verified Outcome | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Test 1** | **LKR VAT & Auto-Approval** | Final price of `mouse` (`3,500 LKR`) + 18% VAT + status | Computes **`4,130 LKR`** & **`AUTO_APPROVED`** | 🟢 **PASSED** |
| **Test 2** | **Missing-Inventory Guard** | Final price with VAT for unavailable `playstation` | Blocks missing item; **0% price hallucination** | 🟢 **PASSED** |
| **Test 3** | **RAG Policy & Noise Guard** | Laptop warranty (>200k LKR) + `pepperoni pizza` catering | Retrieves **3-year** warranty (`Card 1`) & rejects pizza | 🟢 **PASSED** |
| **Test 4** | **Zero-Keyword Hybrid Rescue** | `"My screen has tiny black dots that won't light up"` | Rescues **`Card 3`** (`dead pixels`, `Score: 0.3001`) | 🟢 **PASSED** |

---

## 🌐 Section 5: Cloud Infrastructure & GitHub Milestones (Days 0–6)

- **GitHub Repository (`it26101228/agentic-hybrid-rag-engine`):** 10 verified commits on `main` through commit `e45d428` (`feat: add day 6 retriever benchmarks and update README`), plus Day 6 `PROGRESS_REPORT.md` sync.
- **Hugging Face Ecosystem (`dimanrandinu`):** Completed LLM Course Ch. 1 & Agents Unit 1; resolved 2026 HF Router billing gate via custom Groq provider routing; isolated `BAAI/bge-small-en-v1.5` ONNX cache to `D:/AI_Vault/huggingface_cache` with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **Kaggle Dual-T4 GPU Cluster (`30 hrs/week` quota):** Used on Days 0–2 (`day1_agent_benchmark.ipynb`, `day2-tokenizers-and-detective-agent.ipynb`), then intentionally conserved 100% during Days 3–6 via local CPU ONNX + Groq LPU architecture.

---

## 🧠 Section 6: Master Glossary of 29 Tools, Concepts & Habits (Days 0–6)

1. **Hardware Cache Isolation (`HF_HOME` on `D:\AI_Vault`):** Redirecting model weights off the `C:` boot SSD.
2. **Encoder vs. Decoder Models:** Bidirectional search encoders (`bge-small-en-v1.5`) vs. autoregressive reasoning decoders (`gpt-oss-20b`).
3. **ReAct Loop (`Think -> Act -> Observe`):** Pausing LLM text generation to execute deterministic Python tools.
4. **`@tool` Decorator, Type Hints & Docstrings:** Converting Python functions into LLM-readable JSON schemas.
5. **Subword Tokenization (`input_ids` & `attention_mask`):** Splitting strings into integer token IDs before Transformer inference.
6. **Paper Blueprint First Rule:** Designing inputs, logic/bouncers, and outputs on physical paper before coding.
7. **Organic Input Bouncers (`raise ValueError` & `.get()`):** Blocking invalid/negative/empty inputs before corrupting math.
8. **Python Virtual Environments (`.venv`):** Isolating project packages on `D:` without polluting global Windows Python.
9. **`CodeAgent` vs. `ToolCallingAgent`:** Code-block execution (`tool_choice="none"`) vs. structured JSON tool calls (`tool_choice="auto"`).
10. **OOP Subclassing (`class GroqSafeModel`):** Overriding `_prepare_completion_kwargs` and `generate` to prevent Groq `tool_choice` crashes.
11. **`if __name__ == "__main__":` Import Gate:** Preventing local module test blocks from firing upon import.
12. **Secret Management (`.env` + `.gitignore`):** Keeping `GROQ_API_KEY` out of Git commits.
13. **Cross-Folder Module Resolution (`pathlib.Path` & `sys.path.insert`):** Connecting sibling folders under `src/` cleanly.
14. **Automated Agent Evals (`assert` Crash-Tests):** Catching hallucinations and regressions automatically.
15. **Line Ending Normalization (`LF` vs. `CRLF`):** Managing Linux vs. Windows newline markers in Git.
16. **Sliding-Window Chunking & Overlap (`chunk_size=25`, `overlap=5`):** Preserving multi-word rules across card boundaries.
17. **Metadata Card Packaging (`chunk_id`, `source`, `text`):** Attaching provenance metadata and path bouncers in `loader.py`.
18. **Lexical Keyword Retrieval & Plural Stemming:** Regex tokenization (`\b\w+\b`), stopword removal, and plural trimming.
19. **Deterministic Tool Error Strings:** Returning clear anti-hallucination instructions instead of uncaught exceptions.
20. **384-d Neural Embeddings & Cosine Similarity:** Dot-product semantic scoring with a `min_score=0.58` noise gate.
21. **Score Normalization & 50/50 Hybrid Fusion:** Scaling integer keyword counts (`min(kw * 0.25, 1.0)`) to fuse equally with cosine scores.
22. **Modular Encapsulation:** Upgrading `tools.py` to Hybrid RAG with zero edits to `agent_runner.py`.
23. **IDE Static Analysis Config (`.vscode/settings.json`):** Configuring `python.analysis.extraPaths` for zero Pylance warnings.
24. **Windows PowerShell Execution Policy:** Using `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` for `.venv` activation.
25. **Ground-Truth Benchmark Design:** Curating labeled Keyword, Semantic, and Out-of-Scope test suites (`BENCHMARK_SUITE`).
26. **False Positive Noise Bouncing:** Verifying `len(found_ids) == 0` on unrelated queries (`"pepperoni pizza"`, `"Tokyo flights"`).
27. **Model Warm-Up (Cold Start vs. Warm Execution):** Running dummy inference before `time.perf_counter()` so disk I/O does not skew CPU latency.
28. **Public Interface vs. Internal Helper:** Calling exposed module APIs (`retrieve_semantic_chunks`) rather than private helpers.
29. **Documentation Parity & SSOT Auditing:** Keeping `README.md`, `PROGRESS_REPORT.md`, CSV artifacts, and Master PDFs 100% synchronized.

---

## 🎯 Section 7: Honest Architect Assessment & Week 2 Handover Blueprint

### A. Where You Stand at the End of Week 1 (No-Sugarcoating Evaluation)
- **Strengths Proven:** You built a 5-package Python AI system (`ingestion`, `retrieval`, `agent`, `evals`, `benchmarks`), caught documentation/threshold regressions (`100,000` vs. `200,000 LKR`) during code audits, refused black-box code, and proved **100.0% Hybrid Hit Rate** on local CPU.
- **Current Gaps Targeted for Week 2:**
  1. *Cold-Start Syntax Muscle Memory:* Continue 3-line cold solo warmups (`import`, `assert`, SQL/Pydantic syntax) before each new module so typing speed matches architectural understanding.
  2. *Persistent Storage & Latency Optimization:* Upgrade the hardcoded 3-item inventory dictionary in `tools.py` to a persistent **SQLite relational database (`data/tech_vault.db`)** with **Pydantic V2 validation schemas**, and eliminate on-the-fly card re-embedding (`116.51 ms` -> `~15 ms`).

### B. Immediate Next Step — Week 2 Kickoff (Day 7: SQLite + Pydantic V2)
1. **Paper Blueprint:** Design `products` table (`sku`, `name`, `category`, `price_lkr`, `stock_qty`) and Pydantic V2 input/output models.
2. **Build & Verify:** Create `src/database/db_manager.py` and `src/database/schemas.py`, seed `data/tech_vault.db`, and connect `get_item_price()` in `src/agent/tools.py` to live SQLite queries.