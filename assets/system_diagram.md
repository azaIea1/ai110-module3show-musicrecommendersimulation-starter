# VibeFinder 2.0 — System Architecture Diagram

## Full RAG Pipeline

```mermaid
flowchart TD
    A([User: natural language query]) --> B{Guardrail\nKeyword check}
    B -- Off-topic --> Z([Reject: friendly redirect message])
    B -- Music-related --> C[Embed query\nall-MiniLM-L6-v2\n384-dim vector]

    subgraph Startup ["🔧 Startup — runs once"]
        K[(songs.csv\n30 songs)] --> L[song_to_text\nrich descriptions]
        L --> M[Embed all songs\nsentence-transformers]
        M --> N[(FAISS Index\nIndexFlatIP\ncosine similarity)]
    end

    C --> D[FAISS cosine search\ntop-k nearest neighbors]
    N --> D
    D --> E[Retrieved songs\n+ confidence scores\n0.0 – 1.0]

    E --> CF{Confidence\ncheck}
    CF -- score < 0.25 --> W([⚠️ Low confidence warning\nshown to user])
    CF -- score ≥ 0.25 --> F

    W --> F[Build augmented prompt\nRAG injection:\nquery + retrieved songs]
    F --> G[Groq LLaMA 3 8B\nfree tier API]
    G --> H([Natural language\nrecommendation response])

    H --> I[Display to user\nwith retrieval metadata table]
    I --> J[(vibefinder.log\nTimestamp, query,\nscores, errors)]
    I --> K2{Continue\nchat?}
    K2 -- Yes --> A
    K2 -- No --> END([Session ends])
```

---

## Component Summary

| Component | Technology | Purpose |
|---|---|---|
| **Guardrail** | Keyword set + regex | Block off-topic queries before LLM call |
| **Query Embedder** | sentence-transformers `all-MiniLM-L6-v2` | Map user query to 384-dim semantic vector |
| **Song Catalog** | `data/songs.csv` (30 songs) | Ground truth database for recommendations |
| **song_to_text** | Custom Python function | Converts song metadata to descriptive English text |
| **FAISS Index** | `faiss-cpu IndexFlatIP` | Fast cosine similarity search over song embeddings |
| **Confidence Score** | Cosine similarity (0–1) | Retrieval quality signal; warns user on low scores |
| **Prompt Builder** | String templating | Injects retrieved songs into LLM prompt (RAG) |
| **LLM Generator** | Groq LLaMA 3 8B (free) | Generates grounded natural-language recommendations |
| **Logger** | Python `logging` module | Records all queries, scores, and errors to file |
| **Test Harness** | Custom Python script | Automated pass/fail evaluation of 12 test cases |

---

## Data Flow Summary

```
User input (string)
  → Guardrail (keyword filter)
  → Embedding (float32 vector, shape [1, 384])
  → FAISS search (cosine similarity over [30, 384] index)
  → Retrieved songs (list of dicts) + scores (list of floats)
  → Augmented prompt (string with song context injected)
  → Groq API (chat completion, max 600 tokens)
  → Response text (string)
  → Display + logging
```

---

## How RAG Prevents Hallucination

Without RAG, the LLM would generate recommendations from its training data. This
leads to:
- Made-up song titles that sound plausible but don't exist
- Real songs that exist but aren't in the catalog
- Inconsistent quality depending on what the model "remembers"

With RAG, the prompt explicitly contains the songs the LLM is allowed to reference.
The LLM acts as a *writer* that explains songs it has been *given*, not a *memory*
that recalls songs it *learned about*. This is the core value proposition of RAG.
