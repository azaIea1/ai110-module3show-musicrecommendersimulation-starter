"""
Command line runner for the Music Recommender Simulation.

Loads the song catalog, runs the recommender against several distinct user
profiles, and prints a ranked list of top-5 results for each profile.

Run from the project root with:
    python -m src.main
"""

from .recommender import load_songs, recommend_songs


# ---------------------------------------------------------------------------
# User profiles for testing
# ---------------------------------------------------------------------------

PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "valence": 0.85,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "valence": 0.58,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "valence": 0.45,
        "likes_acoustic": False,
    },
    "Gym Workout": {
        "genre": "pop",
        "mood": "intense",
        "energy": 0.95,
        "valence": 0.75,
        "likes_acoustic": False,
    },
    "Late Night Study": {
        "genre": "ambient",
        "mood": "focused",
        "energy": 0.30,
        "valence": 0.60,
        "likes_acoustic": True,
    },
}

# ---------------------------------------------------------------------------
# Experimental weight-shifted profile (double energy weight, halved genre)
# used in Phase 4 Step 3 – a separate scoring pass to show sensitivity
# ---------------------------------------------------------------------------

EXPERIMENTAL_PREFS = {
    "genre": "pop",
    "mood": "happy",
    "energy": 0.85,
    "valence": 0.85,
    "likes_acoustic": False,
    "_energy_weight": 3.0,   # doubled from 1.5
    "_genre_weight": 1.5,    # halved from 3.0
}


def print_recommendations(profile_name: str, user_prefs: dict, songs: list) -> None:
    """Print formatted top-5 recommendations for a single user profile."""
    print(f"\n{'=' * 58}")
    print(f"  Profile : {profile_name}")
    print(f"  Genre   : {user_prefs.get('genre')} | "
          f"Mood: {user_prefs.get('mood')} | "
          f"Energy: {user_prefs.get('energy')}")
    print(f"{'=' * 58}")

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print(f"\n  Top 5 Recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  {rank}. {song['title']}  —  {song['artist']}")
        print(f"     Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        print(f"     Score : {score:.2f}")
        print(f"     Why   : {explanation}")
        print()


def main() -> None:
    """Load songs and run the recommender for every defined user profile."""
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Standard profile runs
    for profile_name, user_prefs in PROFILES.items():
        print_recommendations(profile_name, user_prefs, songs)

    # Phase 4 Step 3 – Experiment: double energy weight, halve genre weight
    print(f"\n{'#' * 58}")
    print("  EXPERIMENT: Double energy weight / Halve genre weight")
    print(f"  (Comparing to 'High-Energy Pop' baseline above)")
    print(f"{'#' * 58}")
    print_recommendations("High-Energy Pop (Experimental)", EXPERIMENTAL_PREFS, songs)


if __name__ == "__main__":
    main()
