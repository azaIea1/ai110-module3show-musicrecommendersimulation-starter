"""
Groq LLaMA 3 integration for VibeFinder 2.0.

This module handles the "Generation" step of RAG: it takes the retrieved
songs (from rag_recommender.py) and the user's query, builds an augmented
prompt, and calls the Groq API to produce a natural-language recommendation
response grounded in real catalog data.

Key features:
  - Off-topic guardrail: queries unrelated to music are rejected before
    hitting the LLM, saving API calls and preventing misuse.
  - Structured logging: every request/response cycle is logged with
    timestamp, query, top confidence score, and any errors.
  - Graceful error handling: API errors return a friendly fallback message
    instead of crashing.
  - Conversation history: supports multi-turn chat via a messages list.

Setup:
    export GROQ_API_KEY="your_key_here"
    # Free key from https://console.groq.com
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast, free-tier Groq model
MAX_TOKENS = 600
TEMPERATURE = 0.7

# Keywords that suggest a music-related query — used by the guardrail.
MUSIC_KEYWORDS = {
    "song", "music", "listen", "playlist", "vibe", "mood", "genre",
    "artist", "band", "track", "album", "beat", "chill", "relax",
    "workout", "study", "focus", "energy", "dance", "play", "recommend",
    "suggest", "something", "feel", "feeling", "sad", "happy", "intense",
    "calm", "upbeat", "slow", "fast", "acoustic", "electronic", "lofi",
    "pop", "rock", "jazz", "hip-hop", "edm", "ambient", "classical",
    "folk", "r&b", "reggae", "metal", "synthwave", "indie", "gym", "run",
    "driving", "sleep", "party", "morning", "night", "quiet", "loud",
    "soft", "heavy", "mellow", "groovy", "pump", "work", "coding",
}

SYSTEM_PROMPT = """\
You are VibeFinder, a warm and knowledgeable music recommendation assistant.
You help users discover songs from a curated catalog based on how they feel,
what they're doing, or what kind of vibe they're looking for.

When recommending songs:
- Reference ONLY songs from the retrieved catalog provided in the user message.
- Explain briefly why each song fits the user's request (mood, energy, genre, feel).
- Be conversational, concise, and enthusiastic about music.
- If the retrieved songs don't perfectly match, acknowledge that and pick the closest fits.
- Format your response as a short paragraph followed by a numbered list of recommendations.
- Keep the total response under 300 words.
"""


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

def is_music_related(query: str) -> bool:
    """
    Lightweight keyword-based guardrail to detect off-topic queries.

    This runs BEFORE the LLM call to save API quota and prevent the assistant
    from being used as a general-purpose chatbot.

    Args:
        query: The raw user input string.

    Returns:
        True if the query appears to be music-related, False otherwise.
    """
    tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
    overlap = tokens & MUSIC_KEYWORDS
    is_related = len(overlap) >= 1 or len(query.split()) <= 4
    logger.debug(
        "Guardrail check: query=%r overlap=%s is_music=%s",
        query[:60],
        overlap,
        is_related,
    )
    return is_related


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_augmented_prompt(
    query: str,
    retrieved_songs: List[Tuple[Dict, float]],
    rag_engine=None,
) -> str:
    """
    Build the RAG-augmented user message by injecting retrieved songs into
    the prompt.

    This is the core of RAG: instead of asking the LLM to recall songs from
    its training data (which may be hallucinated), we hand it the actual
    catalog entries so every recommendation is grounded in real data.

    Args:
        query: The user's natural-language request.
        retrieved_songs: List of (song_dict, score) tuples from MusicRAG.retrieve().
        rag_engine: Optional MusicRAG instance to get full song text descriptions.

    Returns:
        A formatted string to send as the user turn to the LLM.
    """
    catalog_block = []
    for rank, (song, score) in enumerate(retrieved_songs, start=1):
        if rag_engine:
            desc = rag_engine.get_song_text(song)
        else:
            desc = (
                f'"{song["title"]}" by {song["artist"]} — '
                f'{song["genre"]}, {song["mood"]} mood, energy {song["energy"]:.2f}'
            )
        confidence = rag_engine.confidence_label(score) if rag_engine else "?"
        catalog_block.append(
            f"{rank}. [Retrieval confidence: {confidence} ({score:.3f})]\n   {desc}"
        )

    catalog_str = "\n".join(catalog_block)

    return (
        f"The user asked: \"{query}\"\n\n"
        f"Here are the most relevant songs retrieved from the catalog "
        f"(ranked by semantic similarity to the user's request):\n\n"
        f"{catalog_str}\n\n"
        f"Based ONLY on these retrieved songs, please give the user "
        f"personalised music recommendations with brief explanations."
    )


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def generate_recommendation(
    query: str,
    retrieved_songs: List[Tuple[Dict, float]],
    conversation_history: Optional[List[Dict]] = None,
    rag_engine=None,
) -> Tuple[str, bool]:
    """
    Call the Groq LLaMA 3 API to generate a recommendation response.

    The retrieved songs are injected into the prompt so the LLM's answer is
    always grounded in actual catalog data (RAG pattern).

    Args:
        query: The user's natural-language request.
        retrieved_songs: List of (song_dict, score) from MusicRAG.retrieve().
        conversation_history: Optional list of prior {"role", "content"} dicts
                              for multi-turn support.
        rag_engine: Optional MusicRAG instance for richer song descriptions.

    Returns:
        (response_text, success_bool) tuple.
        On API error, returns a friendly fallback message with success=False.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
    """
    # --- Guardrail check ---
    if not is_music_related(query):
        logger.warning("Off-topic query blocked: %r", query[:80])
        return (
            "I'm VibeFinder — I can only help you discover music! "
            "Try asking me something like: \"What's good to listen to while studying?\" "
            "or \"Give me high-energy songs for a workout.\"",
            False,
        )

    # --- API key check ---
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY not set.")
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get a free key at https://console.groq.com and run:\n"
            "  export GROQ_API_KEY='your_key_here'"
        )

    # --- Import Groq SDK ---
    try:
        from groq import Groq  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "groq package is required. Run: pip install groq"
        ) from exc

    # --- Build messages ---
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)

    augmented_user_message = build_augmented_prompt(query, retrieved_songs, rag_engine)
    messages.append({"role": "user", "content": augmented_user_message})

    # --- Call API ---
    client = Groq(api_key=api_key)
    top_score = retrieved_songs[0][1] if retrieved_songs else 0.0
    logger.info(
        "Calling Groq API | query=%r | retrieved=%d | top_score=%.3f",
        query[:60],
        len(retrieved_songs),
        top_score,
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        reply = response.choices[0].message.content.strip()
        logger.info("Groq response received (%d chars)", len(reply))
        return reply, True

    except Exception as exc:  # noqa: BLE001 — broad catch for API errors
        logger.error("Groq API error: %s", exc)
        # Graceful fallback: show the top retrieved songs without LLM
        fallback_lines = [
            "⚠️  Could not reach the AI service right now. "
            "Here are the top catalog matches based on your request:\n"
        ]
        for i, (song, score) in enumerate(retrieved_songs[:5], start=1):
            fallback_lines.append(
                f"  {i}. {song['title']} — {song['artist']} "
                f"({song['genre']}, {song['mood']}, energy {song['energy']:.2f})"
            )
        return "\n".join(fallback_lines), False
