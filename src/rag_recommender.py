"""
RAG (Retrieval-Augmented Generation) core for VibeFinder 2.0.

This module converts the song catalog into dense vector embeddings using
sentence-transformers, stores them in a FAISS index, and retrieves the most
semantically relevant songs for any natural-language query.

The retrieval step is the "R" in RAG — it finds candidate songs BEFORE the
LLM generates a response, so the LLM's answer is always grounded in real
catalog data rather than hallucinated titles.

Usage:
    from src.rag_recommender import MusicRAG
    rag = MusicRAG("data/songs.csv")
    results = rag.retrieve("something chill to study to", k=5)
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Song text representation
# ---------------------------------------------------------------------------

def song_to_text(song: Dict) -> str:
    """
    Convert a song dictionary into a rich natural-language description
    suitable for embedding.

    The description is designed so that a semantic search for phrases like
    "relaxing background music for studying" will surface songs with low
    energy, focused/chill mood, and high acousticness — even if none of
    those exact words appear in the CSV.

    Args:
        song: Dictionary with keys matching songs.csv columns.

    Returns:
        A multi-sentence string describing the song's character.
    """
    energy = float(song.get("energy", 0.5))
    valence = float(song.get("valence", 0.5))
    acousticness = float(song.get("acousticness", 0.5))
    tempo = float(song.get("tempo_bpm", 100))
    danceability = float(song.get("danceability", 0.5))

    # Human-readable labels for numeric ranges
    energy_label = (
        "very low energy, calm and quiet"
        if energy < 0.3
        else "low energy, gentle"
        if energy < 0.5
        else "moderate energy"
        if energy < 0.7
        else "high energy, driving"
        if energy < 0.9
        else "very high energy, intense and powerful"
    )

    valence_label = (
        "dark and melancholic"
        if valence < 0.35
        else "slightly somber"
        if valence < 0.5
        else "neutral"
        if valence < 0.65
        else "positive and uplifting"
        if valence < 0.8
        else "very happy and bright"
    )

    acoustic_label = (
        "fully electronic and produced"
        if acousticness < 0.2
        else "mostly electronic"
        if acousticness < 0.45
        else "balanced acoustic and electronic"
        if acousticness < 0.65
        else "mostly acoustic"
        if acousticness < 0.85
        else "fully acoustic and organic"
    )

    tempo_label = (
        "very slow tempo"
        if tempo < 70
        else "slow tempo"
        if tempo < 90
        else "moderate tempo"
        if tempo < 115
        else "upbeat tempo"
        if tempo < 140
        else "fast tempo"
    )

    dance_label = "highly danceable" if danceability > 0.75 else "moderately danceable" if danceability > 0.5 else "not particularly danceable"

    return (
        f'"{song["title"]}" by {song["artist"]} is a {song["genre"]} song with a {song["mood"]} mood. '
        f"It has {energy_label} (energy: {energy:.2f}), sounds {valence_label} (valence: {valence:.2f}), "
        f"and has {acoustic_label} instrumentation (acousticness: {acousticness:.2f}). "
        f"The track has a {tempo_label} ({tempo:.0f} BPM) and is {dance_label} (danceability: {danceability:.2f})."
    )


# ---------------------------------------------------------------------------
# MusicRAG class
# ---------------------------------------------------------------------------

class MusicRAG:
    """
    Retrieval-Augmented Generation engine for music recommendations.

    Embeds all songs in the catalog using sentence-transformers and builds
    a FAISS index for fast approximate nearest-neighbor search.

    Attributes:
        songs: List of song dicts loaded from CSV.
        song_texts: List of natural-language song descriptions.
        index: FAISS index (flat cosine similarity via inner product on
               L2-normalised vectors).
        embedder: SentenceTransformer model used for encoding.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # ~80 MB, free, runs locally

    def __init__(self, csv_path: str) -> None:
        """
        Load songs and build the FAISS index.

        Args:
            csv_path: Path to the songs CSV file.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            ImportError: If sentence-transformers or faiss-cpu are not installed.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Song catalog not found: {csv_path}")

        logger.info("Loading song catalog from %s", csv_path)
        self.songs = self._load_songs(csv_path)
        self.song_texts = [song_to_text(s) for s in self.songs]

        logger.info("Loading embedding model: %s", self.MODEL_NAME)
        self._load_embedder()

        logger.info("Building FAISS index for %d songs", len(self.songs))
        self._build_index()
        logger.info("RAG engine ready.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_songs(self, csv_path: str) -> List[Dict]:
        """Load and type-cast songs from CSV."""
        songs = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["id"] = int(row["id"])
                row["energy"] = float(row["energy"])
                row["tempo_bpm"] = float(row["tempo_bpm"])
                row["valence"] = float(row["valence"])
                row["danceability"] = float(row["danceability"])
                row["acousticness"] = float(row["acousticness"])
                songs.append(row)
        return songs

    def _load_embedder(self) -> None:
        """Import and initialise the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            ) from exc
        self.embedder = SentenceTransformer(self.MODEL_NAME)

    def _build_index(self) -> None:
        """Embed all song descriptions and store them in a FAISS flat-IP index."""
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Run: pip install faiss-cpu"
            ) from exc

        embeddings = self.embedder.encode(
            self.song_texts, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(embeddings)
        self.embeddings = embeddings

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        genre_filter: Optional[str] = None,
        mood_filter: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve the top-k most semantically similar songs for a query.

        This is the core RAG retrieval step. The query is embedded with the
        same model used to embed songs, and cosine similarity is used to
        rank candidates. Optional metadata filters (genre, mood) allow the
        RAG Enhancement stretch feature: when a user explicitly names a genre
        or mood, we narrow the candidate pool BEFORE scoring so every
        retrieved song satisfies the hard constraint.

        Args:
            query: Natural-language user request (e.g. "chill music to study").
            k: Number of songs to retrieve.
            genre_filter: If provided, only retrieve songs of this genre.
            mood_filter: If provided, only retrieve songs of this mood.

        Returns:
            List of (song_dict, cosine_similarity_score) tuples, highest first.
            The score serves as a retrieval confidence metric (0–1 range).
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss-cpu is required.") from exc

        # Determine which songs are eligible after metadata filtering
        if genre_filter or mood_filter:
            eligible_indices = [
                i
                for i, s in enumerate(self.songs)
                if (genre_filter is None or s["genre"].lower() == genre_filter.lower())
                and (mood_filter is None or s["mood"].lower() == mood_filter.lower())
            ]
            if not eligible_indices:
                logger.warning(
                    "No songs match genre=%s mood=%s — falling back to full catalog",
                    genre_filter,
                    mood_filter,
                )
                eligible_indices = list(range(len(self.songs)))
        else:
            eligible_indices = list(range(len(self.songs)))

        # Embed the query
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        import faiss as _faiss
        _faiss.normalize_L2(query_vec)

        # If filtering, build a temporary index over eligible songs only
        if len(eligible_indices) < len(self.songs):
            sub_embeddings = self.embeddings[eligible_indices].copy()
            dim = sub_embeddings.shape[1]
            sub_index = _faiss.IndexFlatIP(dim)
            sub_index.add(sub_embeddings)
            search_k = min(k, len(eligible_indices))
            scores, local_positions = sub_index.search(query_vec, search_k)
            positions = [eligible_indices[p] for p in local_positions[0]]
            scores = scores[0]
        else:
            search_k = min(k, len(self.songs))
            scores, positions = self.index.search(query_vec, search_k)
            positions = positions[0]
            scores = scores[0]

        results = []
        for pos, score in zip(positions, scores):
            if pos == -1:
                continue
            results.append((self.songs[pos], float(score)))

        logger.debug(
            "Retrieved %d songs for query=%r (top score=%.3f)",
            len(results),
            query[:60],
            results[0][1] if results else 0.0,
        )
        return results

    def get_song_text(self, song: Dict) -> str:
        """Return the pre-computed natural-language description for a song."""
        idx = next(
            (i for i, s in enumerate(self.songs) if s["id"] == song["id"]), None
        )
        if idx is not None:
            return self.song_texts[idx]
        return song_to_text(song)

    def confidence_label(self, score: float) -> str:
        """
        Convert a cosine similarity score into a human-readable confidence label.

        Args:
            score: Cosine similarity in [0, 1].

        Returns:
            One of: "High", "Medium", "Low", "Very Low".
        """
        if score >= 0.30:
            return "High"
        if score >= 0.22:
            return "Medium"
        if score >= 0.15:
            return "Low"
        return "Very Low"
