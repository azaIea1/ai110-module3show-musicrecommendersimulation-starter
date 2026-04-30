import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """Represents a song and its audio/genre attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences for music recommendation."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return them as a list of dicts with numeric types."""
    songs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['id'] = int(row['id'])
            row['energy'] = float(row['energy'])
            row['tempo_bpm'] = float(row['tempo_bpm'])
            row['valence'] = float(row['valence'])
            row['danceability'] = float(row['danceability'])
            row['acousticness'] = float(row['acousticness'])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """
    Calculate a relevance score for a single song given user preferences.

    Scoring recipe:
      +3.0 for a genre match
      +2.0 for a mood match
      up to +1.5 for energy proximity  (1.5 * (1 - |target - song_energy|))
      up to +1.0 for valence proximity (if 'valence' key is in user_prefs)
      +0.5 for acousticness preference match (if 'likes_acoustic' key is in user_prefs)

    Returns a (score, explanation_string) tuple.
    """
    score = 0.0
    reasons = []

    # Genre match: +3.0
    if song.get('genre', '').lower() == user_prefs.get('genre', '').lower():
        score += 3.0
        reasons.append("genre match (+3.0)")

    # Mood match: +2.0
    if song.get('mood', '').lower() == user_prefs.get('mood', '').lower():
        score += 2.0
        reasons.append("mood match (+2.0)")

    # Energy proximity: up to +1.5 (rewards songs close to the user's target energy)
    target_energy = float(user_prefs.get('energy', 0.5))
    energy_diff = abs(target_energy - float(song.get('energy', 0.5)))
    energy_points = round(1.5 * (1.0 - energy_diff), 2)
    score += energy_points
    reasons.append(f"energy proximity (+{energy_points:.2f})")

    # Valence proximity: up to +1.0 (optional feature)
    if 'valence' in user_prefs:
        valence_diff = abs(float(user_prefs['valence']) - float(song.get('valence', 0.5)))
        valence_points = round(1.0 * (1.0 - valence_diff), 2)
        score += valence_points
        reasons.append(f"valence proximity (+{valence_points:.2f})")

    # Acousticness preference: +0.5
    if 'likes_acoustic' in user_prefs:
        acousticness = float(song.get('acousticness', 0.5))
        if user_prefs['likes_acoustic'] and acousticness > 0.5:
            score += 0.5
            reasons.append("acoustic preference match (+0.50)")
        elif not user_prefs['likes_acoustic'] and acousticness <= 0.5:
            score += 0.5
            reasons.append("electronic preference match (+0.50)")

    return round(score, 2), "; ".join(reasons)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    # TODO: Implement scoring logic using your Algorithm Recipe from Phase 2.
    # Expected return format: (score, reasons)
    return []

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Score every song in the catalog, rank them, and return the top-k recommendations.

    Uses sorted() (non-destructive) rather than .sort() so the original songs list
    is never modified. Returns a list of (song_dict, score, explanation) tuples.
    """
    scored = []
    for song in songs:
        song_score, explanation = score_song(user_prefs, song)
        scored.append((song, song_score, explanation))

    # sorted() creates a new sorted list; .sort() would modify the original list in place
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# OOP implementation – required by tests/test_recommender.py
# ---------------------------------------------------------------------------

class Recommender:
    """OOP wrapper around the scoring logic, operating on Song dataclass instances."""

    def __init__(self, songs: List[Song]):
        """Initialise the recommender with a catalog of Song objects."""
        self.songs = songs

    def _score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Score a single Song against a UserProfile and return (score, reasons list)."""
        score = 0.0
        reasons = []

        # Genre match: +3.0
        if song.genre.lower() == user.favorite_genre.lower():
            score += 3.0
            reasons.append("genre match (+3.0)")

        # Mood match: +2.0
        if song.mood.lower() == user.favorite_mood.lower():
            score += 2.0
            reasons.append("mood match (+2.0)")

        # Energy proximity: up to +1.5
        energy_diff = abs(user.target_energy - song.energy)
        energy_points = round(1.5 * (1.0 - energy_diff), 2)
        score += energy_points
        reasons.append(f"energy proximity (+{energy_points:.2f})")

        # Acousticness preference: +0.5
        if user.likes_acoustic and song.acousticness > 0.5:
            score += 0.5
            reasons.append("acoustic preference match (+0.50)")
        elif not user.likes_acoustic and song.acousticness <= 0.5:
            score += 0.5
            reasons.append("electronic preference match (+0.50)")

        return round(score, 2), reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects ranked from highest to lowest score."""
        scored = [(song, self._score_song(user, song)[0]) for song in self.songs]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [song for song, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a plain-language explanation string for why a song was recommended."""
        _, reasons = self._score_song(user, song)
        return "; ".join(reasons) if reasons else "No matching features found"
