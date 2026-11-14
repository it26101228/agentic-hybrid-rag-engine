# Agentic Hybrid RAG Engine (Tech Vault Sri Lanka)

An industrial, modular **Agentic Hybrid Retrieval-Augmented Generation (RAG)** system built in Python using `smolagents`, **Groq Cloud LLM inference** (`openai/gpt-oss-20b`), and local **ONNX neural embeddings** (`fastembed` / `BAAI/bge-small-en-v1.5`).

Designed and engineered by **Student ID: `it26101228`** as part of the Applied AI Systems Architect Sprint.

---

## 1. System Architecture Overview

Unlike single-notebook prototypes, this repository separates **Ingestion**, **Hybrid Retrieval**, **Agentic Tool Execution**, and **Automated End-to-End Evaluations** into isolated packages:

```text
agentic-hybrid-rag-engine/
├── data/
│   └── shop_policies.txt          # Official store warranty, dead-pixel, and Sri Lanka courier manual
├── src/
│   ├── ingestion/
│   │   ├── chunker.py             # Sliding-window word chunker with 3 strict input bouncers
│   │   └── loader.py              # File/directory guards & metadata card factory (chunk_id, source, text)
│   ├── retrieval/
│   │   ├── retriever.py           # Detective 1: Keyword search with stopword removal & plural stemming
│   │   ├── vector_search.py       # Detective 2: 384-d neural semantic search (BAAI/bge-small-en-v1.5)
│   │   └── hybrid_retriever.py    # Hybrid Boss: 50/50 normalized score fusion & chunk_id deduplication
│   ├── agent/
│   │   ├── tools.py               # 4 @tool functions for pricing, 18% VAT math, approvals & Hybrid RAG
│   │   └── agent_runner.py        # ToolCallingAgent + GroqSafeModel (tool_choice='auto' & final_answer guard)
│   └── evals/
│       └── test_agent_bouncers.py # 4-test automated evaluation suite with strict assert verifications
├── requirements.txt               # Pinned core dependencies (smolagents, huggingface_hub, openai, fastembed)
├── PROGRESS_REPORT.md             # Day-by-day engineering sprint ledger
└── README.md                      # System documentation and reproduction guide