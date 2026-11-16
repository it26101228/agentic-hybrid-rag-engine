# ⚡ Agentic Hybrid RAG Engine — Tech Vault Sri Lanka

![Python 3.12.10](https://img.shields.io/badge/Python-3.12.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Smolagents](https://img.shields.io/badge/HuggingFace-smolagents-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastEmbed ONNX](https://img.shields.io/badge/FastEmbed-bge--small--en--v1.5-8A2BE2?style=for-the-badge)
![Groq Cloud](https://img.shields.io/badge/Groq_Cloud-gpt--oss--20b-F55036?style=for-the-badge)
![Evals Passing](https://img.shields.io/badge/Automated_Evals-4%2F4_Passing_(100%25)-00C853?style=for-the-badge)
![Hybrid Hit Rate](https://img.shields.io/badge/Hybrid_RAG_Hit_Rate-100.0%25_(11%2F11)-2962FF?style=for-the-badge)
![SLIIT Architect](https://img.shields.io/badge/SLIIT_Architect-it26101228-00897B?style=for-the-badge)

> [!IMPORTANT]
> **Executive Summary:** An industrial, modular **Agentic Hybrid Retrieval-Augmented Generation (RAG)** system built in Python. It combines **deterministic financial & tax tools** (18% Sri Lankan VAT & manager approval bouncers) with a **50/50 Hybrid Search Engine** (Exact Keyword Stemming + 384-d Neural Vector Embeddings via ONNX Runtime).

---

## 🏗️ 1. System Architecture & Modular Blueprint

Unlike single-notebook prototypes, this repository separates **Ingestion**, **Hybrid Retrieval**, **Agentic Tool Execution**, **Automated Evaluations**, and **Quantitative Benchmarks** into isolated Python packages:

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
    },
    "benchmarks/": {
        "benchmark_retrievers.py": "11-query CPU latency & Hit-Rate benchmark harness",
        "day6_retriever_benchmarks.csv": "Reproducible Keyword vs. Neural vs. Hybrid scorecard"
    }
}