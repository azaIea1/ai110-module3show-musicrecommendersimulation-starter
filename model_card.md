# 🎧 Model Card: VibeFinder 2.0 — RAG Music Recommender

## 1. Model Name

**VibeFinder 2.0**

---

## 2. Intended Use

VibeFinder 2.0 is a conversational music recommendation assistant designed to
suggest songs from a curated catalog based on natural-language user requests.
Users can describe their mood, activity, or desired vibe in plain English and
receive grounded, personalized recommendations.

It is built for classroom exploration of Retrieval-Augmented Generation (RAG)
architecture. It is not intended for production streaming platforms, commercial
music promotion, or real listener profiling.

---

## 3. How the Model Works

VibeFinder 2.0 uses a three-stage RAG pipeline:

**Stage 1 — Retrieval.** Every song in the catalog is converted to a rich
natural-language description using the `song_to_text` function, then embedded
into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`.
These embeddings are stored in a FAISS flat inner-product index (equivalent to
cosine similarity after L2 normalization). When a user submits a query, the same
embedding model encodes it, and FAISS returns the top-k most semantically similar
songs with cosine similarity scores serving as retrieval confidence.

**Stage 2 — Augmentation.** The retrieved songs and their metadata are injected
directly into the LLM prompt. This ensures the LLM cannot hallucinate song titles
or artists — it can only recommend songs that were actually retrieved from the catalog.

**Stage 3 — Generation.** Groq's LLaMA 3 8B model (free tier) receives the augmented
prompt and generates a conversational recommendation explaining why each song fits
the user's request. A guardrail checks for music-related keywords before any LLM call.

**How this differs from VibeFinder 1.0:**
The original system required structured numeric inputs (genre string, energy float,
valence float, acousticness boolean). VibeFinder 2.0 accepts any natural-language
description and maps it to the catalog semantically rather than arithmetically.

---

## 4. Data

The catalog contains 30 songs across 15 genres and 14 distinct moods. Songs were
expanded from the original 18 to improve genre diversity, particularly adding more
songs for ambient/focused, melancholic, and aggressive categories. Each song has
10 attributes: id, title, artist, genre, mood, energy, tempo_bpm, valence,
danceability, and acousticness. All numeric values are 0–1 except tempo_bpm. All
songs and artist names are fictional and designed for educational purposes.

---

## 5. Strengths

The RAG approach eliminates hallucination — every recommended song exists in the
catalog. Retrieval confidence scores give users and evaluators a transparent quality
signal. The system handles vague, emotional, and context-rich queries well (e.g.,
"music for a rainy afternoon coding session") that would require explicit parameter
tuning in the original system. The guardrail prevents off-topic use without relying
on an expensive LLM classifier.

---

## 6. Limitations and Bias

**Catalog sparsity:** With only 30 songs, retrieval confidence is limited for niche
genres. Metal, reggae, classical, and country each have 1–2 songs, creating a ceiling
on retrieval quality for those genres.

**English-centric embeddings:** The sentence-transformers model was trained on English
text. Queries in other languages or using culturally specific music vocabulary may
not embed correctly into the same semantic space as the English song descriptions.

**Western genre bias:** The catalog covers primarily Western popular and acoustic
genres. No songs represent K-pop, Afrobeats, Bollywood, cumbia, or other global
music traditions.

**Guardrail brittleness:** The keyword-based guardrail can be bypassed with phrasing
that avoids all music vocabulary. A classifier-based guardrail would be more robust.

**Filter bubble preservation:** The RAG system retrieves songs most similar to the
query, which means users who always ask for "chill lofi" will always get lofi results.
There is no diversity injection or serendipity mechanism.

---

## 7. Evaluation

**Unit tests (pytest):** 20 RAG-specific tests + 2 legacy tests = 22 total. All 22 pass.
Tests cover text conversion correctness, retrieval count, score ordering, metadata
filtering, confidence labeling, guardrail acceptance/rejection, and prompt injection.

**Automated harness:** 12 predefined query→expectation pairs. 10/12 passed (83%).
Failures: "aggressive heavy metal" retrieved edm (1 metal song vs. multiple edm songs —
catalog imbalance issue), and "romantic r&b" retrieved pop (similar embedding space).
Average retrieval confidence: 0.48 across all 12 cases.

**Manual review:** 5 sample conversations reviewed for response quality. LLM
responses consistently referenced only retrieved songs, correctly explained mood/energy
matches, and maintained a conversational tone. No hallucinated titles observed.

**Summary:** *10/12 harness tests passed. RAG retrieval confidence averaged 0.48 (medium-high).
Guardrail correctly blocked all 3 off-topic test queries. No hallucinations in 5 manual reviews.*

---

## 8. Testing and Reliability Features

- **Confidence scoring:** FAISS cosine similarity scores (0–1) serve as retrieval
  confidence. Scores below 0.25 trigger a user-visible warning.
- **Logging:** All queries, retrieval results, and API responses are logged to
  `vibefinder.log` with timestamps and log levels.
- **Error handling:** API failures fall back to showing top retrieved songs without
  LLM text rather than crashing.
- **Guardrail:** Keyword-based filter rejects off-topic queries before the LLM call.
- **Automated test harness:** `tests/test_harness.py` provides a repeatable
  evaluation loop with pass/fail scoring.

---

## 9. Reflection and Ethics

**What are the limitations or biases in your system?**
See Section 6. The most significant are catalog sparsity for minority genres, the
English-centric embedding model, and the absence of any diversity mechanism that
would surface cross-genre recommendations.

**Could your AI be misused, and how would you prevent that?**
The most likely misuse is using VibeFinder as a general-purpose chatbot. The keyword
guardrail reduces this but doesn't eliminate it. A secondary risk is that catalog
expansion (adding songs from streaming APIs) could inadvertently introduce songs with
harmful themes. Content filtering on catalog entries would be needed at that stage.

**What surprised you while testing your AI's reliability?**
The most surprising finding was that the semantic embedding space represents energy
level and acoustic texture more strongly than genre labels. "Heavy metal" and "intense
EDM" frequently retrieve the same songs because both are high-energy and electronic,
even though they sound very different to a human listener. This suggests that
genre-label-based vocabulary ("metal", "edm") in queries is processed semantically
rather than as distinct category tokens — the model treats them as synonyms for
"loud, high energy, electronic." Expanding the song descriptions with sub-genre
vocabulary or lyrical theme descriptions would likely improve discrimination.

**Collaboration with AI during this project:**

*One instance where AI gave a helpful suggestion:* When building the FAISS index, I
initially used raw dot products. The AI suggested adding L2 normalization of both the
stored embeddings and query vectors before using `IndexFlatIP`, which converts the
inner product to proper cosine similarity. This was a subtle but important correctness
fix — without it, songs with longer embedding vectors (a numerical artifact) would score
higher regardless of semantic relevance.

*One instance where AI's suggestion was flawed:* The AI recommended using
`faiss.IndexIVFFlat` for "better scalability." While this is good advice for large
datasets, IVFFlat requires a training step with at minimum `nlist × 39` vectors.
With only 30 songs, this would produce a runtime error. The simpler `IndexFlatIP`
(exhaustive search) is not only correct but actually faster for a 30-song catalog.
This was a case where the AI applied a general best practice without accounting for
the specific constraints of the problem.

---

## 10. Future Work

1. **Expand the catalog** to 500+ songs using a public music metadata API (MusicBrainz
   or Last.fm) to improve retrieval quality and genre coverage.
2. **Hybrid retrieval:** Combine semantic RAG with the original rule-based scorer as a
   re-ranking step, giving the best of both approaches.
3. **User feedback loop:** Allow users to thumbs-up/thumbs-down recommendations and
   adjust retrieval weights accordingly — moving toward learned preferences.
4. **Streamlit UI:** The `streamlit` dependency is already in `requirements.txt`.
   A web UI would make the demo more accessible than a CLI.
5. **Classifier-based guardrail:** Replace keyword matching with a small classifier
   for more robust off-topic detection.
