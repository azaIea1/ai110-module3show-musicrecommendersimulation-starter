"""
Unit tests for the RAG pipeline components.

Tests cover:
  - song_to_text: output format and content
  - MusicRAG.retrieve: returns expected number of results
  - MusicRAG.retrieve: genre/mood metadata filtering
  - MusicRAG.confidence_label: correct label bucketing
  - Guardrail: off-topic queries are rejected
  - Guardrail: music queries are accepted
  - build_augmented_prompt: injects song data into the prompt

Run with:
    pytest tests/test_rag.py -v
"""

import os
import sys

import pytest

# Allow imports from src/ when running pytest from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_recommender import MusicRAG, song_to_text
from src.groq_chat import build_augmented_prompt, is_music_related

CATALOG_PATH = "data/songs.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rag():
    """Build the RAG engine once for all tests in this module."""
    return MusicRAG(CATALOG_PATH)


# ---------------------------------------------------------------------------
# song_to_text tests
# ---------------------------------------------------------------------------

class TestSongToText:
    def test_includes_title_and_artist(self):
        song = {
            "id": 1, "title": "Test Song", "artist": "Test Artist",
            "genre": "pop", "mood": "happy",
            "energy": 0.8, "tempo_bpm": 120, "valence": 0.9,
            "danceability": 0.8, "acousticness": 0.2,
        }
        text = song_to_text(song)
        assert "Test Song" in text
        assert "Test Artist" in text

    def test_includes_genre_and_mood(self):
        song = {
            "id": 2, "title": "X", "artist": "Y",
            "genre": "lofi", "mood": "chill",
            "energy": 0.3, "tempo_bpm": 75, "valence": 0.6,
            "danceability": 0.5, "acousticness": 0.8,
        }
        text = song_to_text(song)
        assert "lofi" in text
        assert "chill" in text

    def test_energy_labels_are_descriptive(self):
        high_energy_song = {
            "id": 3, "title": "A", "artist": "B",
            "genre": "edm", "mood": "euphoric",
            "energy": 0.95, "tempo_bpm": 145, "valence": 0.85,
            "danceability": 0.95, "acousticness": 0.02,
        }
        text = song_to_text(high_energy_song)
        assert "high energy" in text.lower() or "intense" in text.lower()

    def test_returns_non_empty_string(self):
        song = {
            "id": 4, "title": "Quiet", "artist": "Pianist",
            "genre": "classical", "mood": "peaceful",
            "energy": 0.2, "tempo_bpm": 60, "valence": 0.75,
            "danceability": 0.25, "acousticness": 0.98,
        }
        text = song_to_text(song)
        assert isinstance(text, str)
        assert len(text) > 50


# ---------------------------------------------------------------------------
# MusicRAG retrieval tests
# ---------------------------------------------------------------------------

class TestMusicRAGRetrieve:
    def test_retrieve_returns_correct_count(self, rag):
        results = rag.retrieve("chill music for studying", k=5)
        assert len(results) == 5

    def test_retrieve_returns_tuple_structure(self, rag):
        results = rag.retrieve("upbeat gym music", k=3)
        for song, score in results:
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_retrieve_sorted_by_score_descending(self, rag):
        results = rag.retrieve("high energy workout", k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted highest-first"

    def test_retrieve_with_genre_filter(self, rag):
        results = rag.retrieve("chill vibes", k=5, genre_filter="lofi")
        for song, _ in results:
            assert song["genre"].lower() == "lofi", (
                f"Expected genre 'lofi', got '{song['genre']}'"
            )

    def test_retrieve_with_mood_filter(self, rag):
        results = rag.retrieve("something focused", k=3, mood_filter="focused")
        for song, _ in results:
            assert song["mood"].lower() == "focused"

    def test_retrieve_k_larger_than_catalog_returns_all(self, rag):
        results = rag.retrieve("music", k=1000)
        assert len(results) == len(rag.songs)

    def test_retrieve_different_queries_produce_different_results(self, rag):
        results_chill = rag.retrieve("soft quiet study music", k=3)
        results_intense = rag.retrieve("aggressive loud metal headbanging", k=3)
        top_chill = results_chill[0][0]["title"]
        top_intense = results_intense[0][0]["title"]
        assert top_chill != top_intense, (
            "Very different queries should produce different top results"
        )

    def test_high_energy_query_retrieves_high_energy_songs(self, rag):
        results = rag.retrieve("maximum energy intense powerful music", k=5)
        top_song, _ = results[0]
        assert float(top_song["energy"]) >= 0.7, (
            f"Expected high energy song at top, got energy={top_song['energy']}"
        )

    def test_chill_query_retrieves_low_energy_songs(self, rag):
        results = rag.retrieve("soft gentle calm peaceful quiet relaxing", k=3)
        top_song, _ = results[0]
        assert float(top_song["energy"]) <= 0.65, (
            f"Expected low energy song at top, got energy={top_song['energy']}"
        )


# ---------------------------------------------------------------------------
# Confidence label tests
# ---------------------------------------------------------------------------

class TestConfidenceLabel:
    def test_high_label(self, rag):
        assert rag.confidence_label(0.60) == "High"

    def test_medium_label(self, rag):
        assert rag.confidence_label(0.45) == "Medium"

    def test_low_label(self, rag):
        assert rag.confidence_label(0.30) == "Low"

    def test_very_low_label(self, rag):
        assert rag.confidence_label(0.10) == "Very Low"


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------

class TestGuardrail:
    def test_music_query_passes(self):
        assert is_music_related("give me chill music to study") is True

    def test_workout_query_passes(self):
        assert is_music_related("high energy workout playlist") is True

    def test_song_keyword_passes(self):
        assert is_music_related("recommend a good song") is True

    def test_off_topic_blocked(self):
        assert is_music_related("write me a Python function to sort a list") is False

    def test_off_topic_homework_blocked(self):
        assert is_music_related("what is the capital of France") is False

    def test_short_query_passes(self):
        # Short queries (<=4 words) are allowed through to avoid false positives
        assert is_music_related("something sad please") is True


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestBuildAugmentedPrompt:
    def test_prompt_contains_query(self):
        song = {
            "id": 1, "title": "Test Track", "artist": "Test Artist",
            "genre": "pop", "mood": "happy",
            "energy": 0.8, "tempo_bpm": 120, "valence": 0.9,
            "danceability": 0.8, "acousticness": 0.2,
        }
        prompt = build_augmented_prompt("morning coffee vibes", [(song, 0.72)])
        assert "morning coffee vibes" in prompt

    def test_prompt_contains_song_title(self):
        song = {
            "id": 1, "title": "Unique Song Name XYZ", "artist": "Artist",
            "genre": "lofi", "mood": "chill",
            "energy": 0.3, "tempo_bpm": 75, "valence": 0.6,
            "danceability": 0.5, "acousticness": 0.8,
        }
        prompt = build_augmented_prompt("relaxing music", [(song, 0.65)])
        assert "Unique Song Name XYZ" in prompt

    def test_prompt_contains_multiple_songs(self):
        songs = [
            {
                "id": i, "title": f"Song {i}", "artist": "Artist",
                "genre": "pop", "mood": "happy",
                "energy": 0.8, "tempo_bpm": 120, "valence": 0.9,
                "danceability": 0.8, "acousticness": 0.2,
            }
            for i in range(1, 4)
        ]
        retrieved = [(s, 0.7 - i * 0.1) for i, s in enumerate(songs)]
        prompt = build_augmented_prompt("happy music", retrieved)
        assert "Song 1" in prompt
        assert "Song 2" in prompt
        assert "Song 3" in prompt
