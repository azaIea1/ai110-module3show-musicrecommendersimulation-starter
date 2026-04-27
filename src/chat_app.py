"""
VibeFinder 2.0 — RAG-powered music recommendation chatbot.

This is the main entry point for the interactive CLI experience.
Users type natural-language requests and the system:

  1. Checks the guardrail (is this music-related?)
  2. Embeds the query and retrieves the top-5 most relevant songs
     from the FAISS index (RAG retrieval step)
  3. Injects those songs into the prompt and calls Groq LLaMA 3
     (RAG generation step)
  4. Streams the response back to the user with retrieval metadata

Run from the project root:
    python -m src.chat_app

Environment:
    GROQ_API_KEY  — required (free at https://console.groq.com)
    LOG_LEVEL     — optional, default INFO (set to DEBUG for verbose logs)
"""

from __future__ import annotations

import logging
import os

# Load .env file if present (keeps API key out of source code)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; key can still be set manually
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup — writes to both console and a log file
# ---------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = "vibefinder.log"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATALOG_PATH = "data/songs.csv"
TOP_K = 5

BANNER = """
╔══════════════════════════════════════════════════════╗
║        🎵  VibeFinder 2.0  —  RAG Edition  🎵        ║
║                                                      ║
║  Ask me anything about what you want to listen to.  ║
║  Type  quit  or  exit  to leave.                    ║
║  Type  history  to see this session's exchanges.    ║
╚══════════════════════════════════════════════════════╝
"""

EXAMPLE_QUERIES = [
    "  • I want something chill to study to",
    "  • Give me high-energy songs for a workout",
    "  • I feel sad and nostalgic — what should I listen to?",
    "  • Something happy and upbeat for a Sunday morning",
    "  • I need intense music for deep focus coding",
]


def print_retrieval_info(
    retrieved_songs: List,
    rag_engine,
    verbose: bool = False,
) -> None:
    """Print a compact table of retrieved songs with confidence scores."""
    print("\n  ┌─ Retrieved from catalog ──────────────────────────────────")
    for i, (song, score) in enumerate(retrieved_songs, start=1):
        label = rag_engine.confidence_label(score)
        print(
            f"  │ {i}. {song['title']:<22} {song['artist']:<18} "
            f"[{label:<6} {score:.3f}]"
        )
    print("  └───────────────────────────────────────────────────────────\n")


def run_chat(catalog_path: str = CATALOG_PATH, top_k: int = TOP_K) -> None:
    """
    Main interactive loop for the VibeFinder chatbot.

    Initialises the RAG engine once (embeddings are cached in memory),
    then loops over user input until the user exits.
    """
    from src.rag_recommender import MusicRAG
    from src.groq_chat import generate_recommendation, is_music_related

    print(BANNER)
    print("  Try asking:")
    for ex in EXAMPLE_QUERIES:
        print(ex)
    print()

    # --- Initialise RAG (loads model + builds FAISS index) ---
    print("  Initialising RAG engine (first run downloads ~80 MB model)…")
    try:
        rag = MusicRAG(catalog_path)
    except FileNotFoundError as exc:
        print(f"\n  ❌  Error: {exc}")
        print(f"  Make sure you run this from the project root directory.")
        sys.exit(1)
    except ImportError as exc:
        print(f"\n  ❌  Missing dependency: {exc}")
        sys.exit(1)

    print(f"  ✅  Catalog loaded: {len(rag.songs)} songs indexed.\n")

    conversation_history: List[Dict] = []
    session_log: List[Dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! Keep listening. 🎵\n")
            break

        if not user_input:
            continue

        # --- Special commands ---
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye! Keep listening. 🎵\n")
            break

        if user_input.lower() == "history":
            if not session_log:
                print("  (No exchanges yet.)\n")
            else:
                for i, entry in enumerate(session_log, start=1):
                    print(f"\n  [{i}] You: {entry['query']}")
                    print(f"       Bot: {entry['response'][:120]}…")
            print()
            continue

        # --- Parse optional explicit filters from the query ---
        # Stretch feature: if user mentions a specific genre or mood we
        # narrow the retrieval pool (RAG Enhancement).
        genre_filter = _extract_filter(user_input, "genre")
        mood_filter = _extract_filter(user_input, "mood")

        # --- RAG: Retrieve ---
        logger.info("User query: %r", user_input)
        try:
            retrieved = rag.retrieve(
                user_input,
                k=top_k,
                genre_filter=genre_filter,
                mood_filter=mood_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval error: %s", exc)
            print(f"\n  ⚠️  Retrieval error: {exc}\n")
            continue

        # Show retrieval metadata (transparency / rubric requirement)
        print_retrieval_info(retrieved, rag)

        # --- Warn on low confidence ---
        top_score = retrieved[0][1] if retrieved else 0.0
        if top_score < 0.12:
            print(
                "  ⚠️  Low retrieval confidence — no songs closely match your request.\n"
                "      Showing best available options.\n"
            )

        # --- RAG: Generate ---
        try:
            response, success = generate_recommendation(
                query=user_input,
                retrieved_songs=retrieved,
                conversation_history=conversation_history,
                rag_engine=rag,
            )
        except EnvironmentError as exc:
            print(f"\n  ❌  {exc}\n")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            logger.error("Generation error: %s", exc)
            print(f"\n  ⚠️  Generation error: {exc}\n")
            continue

        # --- Display response ---
        print(f"VibeFinder: {response}\n")

        # --- Update multi-turn conversation history ---
        if success:
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            # Keep history bounded to last 6 turns (3 exchanges)
            if len(conversation_history) > 12:
                conversation_history = conversation_history[-12:]

        # --- Session log ---
        session_log.append({"query": user_input, "response": response, "top_score": top_score})
        logger.info("Exchange logged. Session total: %d", len(session_log))


def _extract_filter(query: str, filter_type: str) -> Optional[str]:
    """
    Parse an explicit genre or mood keyword from the query string.

    Examples:
        "genre: lofi, something to study to" → "lofi"
        "mood: chill music please"           → "chill"

    Returns None if no explicit filter is found.
    """
    import re
    pattern = rf"\b{filter_type}:\s*([a-z\-&]+)"
    match = re.search(pattern, query.lower())
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point called by `python -m src.chat_app`."""
    run_chat()


if __name__ == "__main__":
    main()
