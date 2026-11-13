# Master Sprint Progress Report: Applied AI Systems Architect
**Student ID:** \it26101228\ (SLIIT IT - Year 1, Semester 2)  
**Project Repository:** \it26101228/agentic-hybrid-rag-engine\  
**Current Status:** Days 1–4 Completed & Pushed to GitHub  

---

## Executive Summary

This report updates our Master Sprint Ledger with all architectural milestones achieved through **Day 4**. We have successfully transitioned our system from a standalone math calculator into an **Agentic RAG Engine** capable of combining live inventory pricing, 18% Sri Lankan VAT calculations, automated approval bouncers, and keyword-based store policy retrieval.

---

## Comprehensive Sprint Ledger (Days 1–4)

| Module / Path | Component Name | Core Architect Rules & Features | Status |
| :--- | :--- | :--- | :--- |
| **\data/\** | Official Store Policies | Created \shop_policies.txt\ containing warranty rules, dead-pixel replacement criteria, and outstation courier fees (Anuradhapura, Kandy, Galle). | **Completed & Pushed**[cite: 1] |
| **\src/ingestion/\** | Document Chunker (\chunker.py\) | Sliding-word window slicer with configurable \chunk_size\ and \overlap\. Protected by 3 strict input bouncers preventing infinite loops and invalid sizes. | **Completed & Pushed**[cite: 1] |
| **\src/ingestion/\** | Metadata Loader (\loader.py\) | Implemented file/folder bouncers (\exists()\, \is_file()\) and packaged each raw chunk into a dictionary card (\chunk_id\, \source\, \	ext\). | **Completed & Pushed**[cite: 1] |
| **\src/retrieval/\** | Keyword Retriever (\etriever.py\) | Highlighter detective featuring regex cleaning, stopword filtering, plural normalization/stemming (\costs\ $\rightarrow$ \cost\), and unique keyword scoring (\	op_k\). | **Completed & Pushed**[cite: 1] |
| **\src/agent/\** | Tool Suite (\	ools.py\) | Built 4 industrial tools: \get_item_price\, \calculate_final_price\, \check_approval_status\, and \search_shop_policies\ (with strict anti-hallucination guards). | **Completed & Pushed**[cite: 1] |
| **\src/agent/\** | Agent Runner (\gent_runner.py\) | Integrated the 4-tool \ToolCallingAgent\ with an upgraded \GroqSafeModel\ supporting \	ool_choice="auto"\, 400-error fallbacks, and automatic \inal_answer\ wrapping. | **Completed & Pushed**[cite: 1] |
| **\src/evals/\** | Evaluation Suite (\	est_agent_bouncers.py\) | Automated 3-test end-to-end test runner validating mouse VAT math, PlayStation missing-inventory bouncers, and RAG policy retrieval + missing pizza policy rejection. | **Completed & Pushed**[cite: 1] |

---

## Key Technical Achievements Completed Today (Day 4)

1. **The Ingestion & Token-Card Pipeline:** Successfully sliced complex store documents into overlapping index cards with strict metadata labeling so the AI never loses track of source files.
2. **The Keyword Highlighter Detective:** Built custom stopword filtering and plural stemming so variations like \"costs"\ cleanly match \"cost"\ and \"pixels"\ match \"pixel"\.
3. **Anti-Hallucination Policy Guards:** Configured \search_shop_policies\ and item lookup tools to return explicit warning strings (\"No matching policies found. Do NOT guess..."\) instead of raising unhandled exceptions, preventing LLM panic loops and fake data generation.
4. **End-to-End Evaluation Success:** Verified that all 3 automated tests in \	est_agent_bouncers.py\ execute successfully and pass 100% on the Groq cloud model.
