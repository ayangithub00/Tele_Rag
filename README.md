# 📡 TeleRAG — 3GPP Telecom Standards RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from 3GPP telecom standards documentation, built with a focus on **minimizing hallucinations**.

---

## Overview

TeleRAG lets users ask natural-language questions about 3GPP specifications and returns answers grounded entirely in the source documents — not the LLM's general knowledge. If an answer isn't present in the retrieved context, the system explicitly says so instead of guessing.

**Knowledge base (3GPP specs used):**
- TS 23.501 — System architecture for the 5G System
- TS 23.502 — Procedures for the 5G System
- TS 23.503 — Policy and charging control framework
- TS 38.300 — NR and NG-RAN overall description

---

## Architecture

```
PDF Documents (3GPP specs)
        │
        ▼
  PyPDFLoader (per-page extraction)
        │
        ▼
  RecursiveCharacterTextSplitter
  (chunk_size=1000, overlap=200)
        │
        ▼
  HuggingFace Embeddings
  (BAAI/bge-small-en-v1.5)
        │
        ▼
  FAISS Vector Store (local, persisted)
        │
        ▼
  User Query → similarity_search(k=20)
        │
        ▼
  Retrieved Chunks → Context
        │
        ▼
  ChatMistralAI (mistral-small-latest)
  + Strict context-only prompt
        │
        ▼
  Answer + Source Citations (Streamlit UI)
```

**Pipeline stages:**
1. **Ingestion** (`ingestion.py`) — loads PDFs, splits into chunks, embeds them, and persists a FAISS index to disk.
2. **Retrieval + Generation** (`app.py`) — embeds the user query, retrieves the top-k most relevant chunks, and passes them as context to the LLM.

---

## Hallucination Minimization Strategy

Reducing hallucination was a core design goal, achieved through several layered techniques:

1. **Context-only prompting** — the LLM is explicitly instructed to answer *only* from the retrieved context and never use external/general knowledge.

2. **Explicit fallback response** — if the answer isn't present in the retrieved chunks, the model is instructed to reply with a fixed, unambiguous message (`Information not available in provided documents.`) instead of guessing or extrapolating.

3. **Source transparency** — every answer is accompanied by the retrieved source chunks in an expandable panel, so the user can independently verify the answer against the original text.

4. **Grounded retrieval over generation** — the system relies on semantic similarity search (FAISS + BGE embeddings) to fetch relevant passages before any generation happens, rather than letting the LLM answer freely.

5. **Tested against out-of-scope and complex queries** — the system was validated on:
   - **Out-of-scope questions** (not covered by the loaded specs) → correctly returns the fallback "not available" message rather than fabricating an answer.
   - **Precise, single-topic questions** (e.g., "What is the role of NRF in AMF selection?") → returns accurate, source-grounded answers matching the exact spec language.
   - **Complex, multi-part questions** spanning multiple specs/topics (e.g., handover + AMF relocation + SMF + UPF in one query) → the system appropriately returns "not available" when a single retrieval pass can't surface all relevant context across topics, rather than blending partial information into an incorrect answer. This was a deliberate trade-off: correctness over completeness.

**Known limitation & future improvement:** very broad, multi-concept queries can dilute the semantic search vector, reducing retrieval precision. This can be addressed with **multi-query retrieval** — decomposing a complex question into sub-questions, retrieving context for each independently, and merging results before generation.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDFLoader (langchain_community) |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Vector Store | FAISS (local, on-disk) |
| LLM | Mistral (mistral-small-latest via ChatMistralAI) |

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r req.txt
```

### 2. Set environment variables
Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_api_key_here
```

### 3. Add source documents
Place 3GPP PDF specs inside `knowledge_base/`.

### 4. Run ingestion (builds the vector store)
```bash
python ingestion.py
```
This creates a local `vector_store/` folder containing the FAISS index. Re-run this whenever the source documents change.

### 5. Launch the app
```bash
streamlit run app.py --server.fileWatcherType none
```

> **Note (macOS):** if you hit a segmentation fault when loading embeddings/FAISS together, set these environment variables at the top of `app.py` / `ingestion.py` before any imports — this resolves a known OpenMP conflict between PyTorch and FAISS:
> ```python
> import os
> os.environ["OMP_NUM_THREADS"] = "1"
> os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
> os.environ["TOKENIZERS_PARALLELISM"] = "false"
> ```

---

## Project Structure

```
Tele_Rag/
├── app.py              # Streamlit chatbot UI + retrieval/generation logic
├── ingestion.py         # PDF ingestion → chunking → embedding → FAISS index
├── req.txt              # Python dependencies
├── knowledge_base/      # Source 3GPP PDF specs
├── vector_store/        # Persisted FAISS index (generated, gitignored)
├── .env                 # API keys (gitignored)
└── README.md
```

---

## Example Queries

| Query | Result |
|---|---|
| "What is the role of NRF in AMF selection?" | Accurate, source-grounded answer citing discovery parameters (GUAMI, TAI, AMF Set ID, etc.) |
| "SMF's role in updating PDU session during handover" | Detailed, spec-accurate answer covering N4 Session Modification, Nsmf_PDUSession_Update, H-SMF/V-SMF interaction |
| Question unrelated to loaded specs | `Information not available in provided documents.` |

---

## Possible Future Enhancements

- Multi-query decomposition for complex, multi-topic questions
- Source filename (not just page number) shown alongside citations
- Re-ranking retrieved chunks before passing to the LLM
- Incremental ingestion instead of full re-indexing on every run
- Evaluation harness to systematically measure hallucination rate across a test question set