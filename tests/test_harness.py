"""
Automated test harness for VibeFinder 2.0 RAG pipeline.

This script runs the retrieval system against a predefined set of inputs and
prints a pass/fail summary with confidence scores. It does NOT call the Groq
API — it tests only the retrieval component, which is deterministic and free.

Evaluation criteria for each test case:
  - The top-retrieved song's genre or mood must match the expected genre/mood,
    OR the retrieval confidence must be above a minimum threshold.
  - This simulates a real evaluation loop where a human-defined "ground truth"
    is compared against system output.

Run from the project root:
    python -m tests.test_harness
    python tests/test_harness.py

Output example:
  PASS [0.612] "chill study music"      → Library Rain (lofi/chill)
  PASS [0.588] "gym workout anthem"     → Gym Hero (pop/intense)
  FAIL [0.221] "heavy thrash metal"     → expected genre=metal, got genre=pop
  ─────────────────────────────────────
  Results: 8/10 passed  |  Avg confidence: 0.51
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestCase:
    """A single evaluation case with an input query and expected output criteria."""
    query: str
    expected_genre: Optional[str] = None   # Top song should have this genre
    expected_mood: Optional[str] = None    # Top song should have this mood
    min_confidence: float = 0.20           # Retrieval confidence must exceed this
    description: str = ""                  # Human-readable label for the test


# ---------------------------------------------------------------------------
# Test suite definition
# ---------------------------------------------------------------------------

TEST_CASES: List[TestCase] = [
    TestCase(
        query="chill lofi beats for studying",
        expected_genre="lofi",
        min_confidence=0.40,
        description="Lofi study retrieval",
    ),
    TestCase(
        query="high energy gym workout pump-up anthem",
        expected_mood="intense",
        min_confidence=0.40,
        description="High-energy workout retrieval",
    ),
    TestCase(
        query="peaceful classical piano background music",
        expected_genre="classical",
        min_confidence=0.35,
        description="Classical peaceful retrieval",
    ),
    TestCase(
        query="sad melancholic folk song about loss",
        expected_mood="melancholic",
        min_confidence=0.30,
        description="Melancholic folk retrieval",
    ),
    TestCase(
        query="dance party EDM electronic banger",
        expected_genre="edm",
        min_confidence=0.35,
        description="EDM dance retrieval",
    ),
    TestCase(
        query="romantic smooth r&b late night vibes",
        expected_genre="r&b",
        min_confidence=0.30,
        description="R&B romantic retrieval",
    ),
    TestCase(
        query="aggressive heavy metal headbanging",
        expected_genre="metal",
        min_confidence=0.30,
        description="Metal aggressive retrieval",
    ),
    TestCase(
        query="upbeat reggae summer morning sunshine",
        expected_genre="reggae",
        min_confidence=0.30,
        description="Reggae uplifting retrieval",
    ),
    TestCase(
        query="moody synthwave night drive neon lights",
        expected_genre="synthwave",
        min_confidence=0.30,
        description="Synthwave moody retrieval",
    ),
    TestCase(
        query="confident hip-hop rap hype track",
        expected_genre="hip-hop",
        min_confidence=0.30,
        description="Hip-hop confident retrieval",
    ),
    TestCase(
        query="ambient meditation deep focus no lyrics",
        expected_genre="ambient",
        min_confidence=0.30,
        description="Ambient focused retrieval",
    ),
    TestCase(
        query="jazz coffee shop relaxed afternoon",
        expected_genre="jazz",
        min_confidence=0.30,
        description="Jazz relaxed retrieval",
    ),
]


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

def run_harness(catalog_path: str = "data/songs.csv") -> None:
    """Run all test cases and print a formatted summary."""
    from src.rag_recommender import MusicRAG

    print("\n" + "=" * 65)
    print("  VibeFinder 2.0 — RAG Evaluation Harness")
    print("=" * 65)
    print(f"  Catalog: {catalog_path}")
    print(f"  Test cases: {len(TEST_CASES)}\n")

    print("  Building RAG engine…")
    rag = MusicRAG(catalog_path)
    print(f"  ✅  {len(rag.songs)} songs indexed.\n")
    print("-" * 65)

    passed = 0
    total = len(TEST_CASES)
    confidence_scores: List[float] = []
    results_rows = []

    for tc in TEST_CASES:
        retrieved = rag.retrieve(tc.query, k=5)
        top_song, top_score = retrieved[0]
        confidence_scores.append(top_score)

        # Evaluate pass/fail criteria
        genre_ok = tc.expected_genre is None or top_song["genre"].lower() == tc.expected_genre.lower()
        mood_ok = tc.expected_mood is None or top_song["mood"].lower() == tc.expected_mood.lower()
        conf_ok = top_score >= tc.min_confidence

        # Pass if confidence is sufficient AND at least one categorical criterion is met
        # (or both criteria are None, meaning only confidence matters)
        categorical_ok = genre_ok and mood_ok
        test_passed = categorical_ok and conf_ok

        if test_passed:
            passed += 1
            status = "PASS ✅"
        else:
            status = "FAIL ❌"

        # Build failure reason
        reasons = []
        if not conf_ok:
            reasons.append(f"confidence {top_score:.3f} < {tc.min_confidence:.3f}")
        if not genre_ok:
            reasons.append(f"expected genre={tc.expected_genre}, got {top_song['genre']}")
        if not mood_ok:
            reasons.append(f"expected mood={tc.expected_mood}, got {top_song['mood']}")

        reason_str = " | " + ", ".join(reasons) if reasons else ""
        conf_label = rag.confidence_label(top_score)

        row = (
            f"  {status} [{top_score:.3f} {conf_label:<6}] "
            f"{tc.description:<35} "
            f"→ {top_song['title']} ({top_song['genre']}/{top_song['mood']})"
            f"{reason_str}"
        )
        results_rows.append(row)
        print(row)

    # Summary
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    print("\n" + "─" * 65)
    print(f"  Results : {passed}/{total} passed  ({100 * passed // total}%)")
    print(f"  Avg retrieval confidence : {avg_conf:.3f}")
    print(f"  Min confidence           : {min(confidence_scores):.3f}")
    print(f"  Max confidence           : {max(confidence_scores):.3f}")
    print("─" * 65 + "\n")

    if passed == total:
        print("  🎉  All tests passed!")
    elif passed >= total * 0.75:
        print(f"  ⚠️  {total - passed} test(s) failed — review low-confidence cases above.")
    else:
        print(f"  ❌  {total - passed} tests failed — consider expanding the catalog.")
    print()


if __name__ == "__main__":
    run_harness()
