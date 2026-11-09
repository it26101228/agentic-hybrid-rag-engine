# Agentic Hybrid RAG Engine 🚀
Production-grade Applied AI Agent & Hybrid RAG Architecture built during a 9-week intensive sprint.

## 📊 Day 1 Benchmark Results (NVIDIA Tesla T4 GPU)
Tested two open-source instruction-tuned models on structured JSON data extraction across 5 messy invoice emails:

| Model Name | Parameters | Peak VRAM (GB)* | Throughput (tok/s) | JSON Validity (%) |
| :--- | :---: | :---: | :---: | :---: |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 8.64 | 26.1 | 100% |
| Qwen/Qwen2.5-3B-Instruct | 3.0B | 11.52 | 20.6 | 100% |

*\*Note on VRAM: Measured in a shared notebook session where the Cell 1 `smolagents` `CodeAgent` instance retained ~5.6 GB of baseline VRAM on `cuda:0`. Standalone model weights account for ~3.0 GB (1.5B) and ~5.9 GB (3.0B) in `float16`.*

* **Key Takeaway:** Both models passed the strict Pydantic border guard (`InvoiceTicket`) with **100% validity (5/5)** using greedy decoding (`do_sample=False`), proving small open-weight models on a single T4 GPU are reliable for structured extraction when paired with schema validation.

## 🛠️ Tech Stack
* **Orchestration:** `smolagents` (`CodeAgent` with custom `@tool`)
* **Inference & Benchmarking:** Hugging Face `transformers` (`float16`), PyTorch, Kaggle T4 GPUs
* **Validation:** Pydantic v2 (`BaseModel`)