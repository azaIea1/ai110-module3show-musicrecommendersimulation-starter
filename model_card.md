# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

VibeFinder 1.0 is designed to suggest songs from a small catalog based on a user's stated preferences for genre, mood, energy, and acousticness. It is built for classroom exploration of how content-based recommendation systems work. It is not intended for production use, real streaming platforms, or users with more than a handful of catalog entries. It should not be used to make decisions about what music to promote commercially or to profile real listeners.

---

## 3. How the Model Works

VibeFinder compares each song in the catalog to what the user says they like. Every song gets a "score" based on how well it matches the user's taste profile. Songs that match on multiple dimensions — like genre and mood at the same time — score much higher than songs that only match one thing.

For categorical features like genre and mood, the system awards full bonus points for an exact match and zero otherwise. For numerical features like energy (how loud and active a song feels, on a 0–1 scale), the system uses a proximity formula: a song at exactly the user's preferred energy level scores the maximum number of points, and that number shrinks the further away the song's energy is from the target. The same proximity approach applies to valence (how positive or upbeat the music sounds). Acousticness (whether the song is acoustic or electronic) earns a small bonus if the song fits the user's preference.

Once every song has a score, the system sorts the entire catalog from highest to lowest and returns the top five suggestions. The scoring and ranking steps are kept separate: the scoring function judges one song at a time, and the ranking step just sorts all the scores afterward. This separation makes it easy to experiment with the weights without touching the sorting logic.

---

## 4. Data

The catalog contains 18 songs across 11 genres (pop, lofi, rock, ambient, jazz, synthwave, indie pop, edm, classical, hip-hop, r&b, country, metal, folk, reggae) and 12 distinct moods. Songs were taken from a 10-song starter set and expanded with 8 additional entries to improve genre and mood diversity. Each song has 9 attributes: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, and acousticness. All numerical values are on a 0–1 scale except tempo_bpm. The dataset was designed for a classroom exercise, so the catalog is tiny compared to real platforms like Spotify (which indexes over 80 million tracks). The song names and artist names are fictional.

---

## 5. Strengths

The system works best when a user has a clearly defined taste with a specific genre and mood in mind. For profiles like "pop/happy" or "lofi/chill," the top results reliably feel right: the songs with matching genre and mood jump to the top with large score gaps over everything else, making the ranking very interpretable. The system is also completely transparent — every recommendation comes with a plain-language reason explaining exactly which features drove the score, which is a significant advantage over black-box neural recommenders. Its simplicity makes it easy to debug and to understand where it might fail.

---

## 6. Limitations and Bias

The most significant bias is toward genre dominance. Because a genre match is worth 3.0 points — twice as much as a mood match and far more than any numerical feature — the catalog is effectively divided into genre silos before any other comparison happens. A perfectly mood-matched, energy-matched song from a slightly different genre (e.g., "indie pop" instead of "pop") will almost always rank below a genre-matched song that disagrees on everything else. This creates a "filter bubble" effect: users who prefer pop will almost never see jazz or folk recommendations even if those songs would genuinely match their energy and mood preferences.

A second limitation is dataset imbalance. Pop, lofi, and rock each have multiple entries while many genres appear only once. Users with preferences for underrepresented genres (e.g., reggae or classical) receive genre matches less often, so their scores cluster closer together and the system essentially falls back on energy proximity to differentiate results, which feels arbitrary.

Finally, the system has no memory and no collaborative signal. It cannot learn from what a user skips or replays, and it cannot use information about what similar listeners enjoyed. Two users with identical preference profiles will always receive identical recommendations regardless of what they have heard before.

---

## 7. Evaluation

Five distinct user profiles were tested: High-Energy Pop, Chill Lofi, Deep Intense Rock, Gym Workout, and Late Night Study. For each profile, the top 5 results were inspected to see whether the ranking matched musical intuition.

The results for High-Energy Pop and Chill Lofi were the most satisfying — the top two results matched on both genre and mood, and the score gap between them and the third-place song was large enough to feel convincing. The Deep Intense Rock profile correctly surfaced Storm Runner at the top (rock/intense), but the second-place result was Gym Hero (pop/intense), which shares the mood but not the genre — this exposed the mood weight competing with the genre weight in an interesting way.

For the Late Night Study profile (ambient/focused), the system correctly placed Spacewalk Thoughts first (ambient/chill, close energy), but could not find a song with both "ambient" genre and "focused" mood in the catalog, so the mood match went unfulfilled. This highlights the catalog diversity problem directly.

A weight-shift experiment was also run: doubling the energy weight from 1.5 to 3.0 and halving the genre weight from 3.0 to 1.5 for the High-Energy Pop profile. The top result stayed the same (Sunrise City), but Rooftop Lights moved from #3 to #2 because its strong energy proximity and mood match became worth more relative to the weaker genre bonus. Gym Hero dropped from #2 to #3. This confirmed that genre weight is the dominant sorting force in the default configuration.

---

## 8. Future Work

Three improvements would make the biggest difference. First, adding a diversity penalty that prevents more than two songs from the same genre from appearing in a single top-5 list would reduce the filter bubble effect and surface more surprising but relevant recommendations. Second, replacing the binary genre-match rule with a genre-similarity score (for example, treating "indie pop" as 80% similar to "pop" rather than 0% similar) would make the system far less brittle. Third, incorporating a short interaction history — even just the last three songs a user played — would allow the system to adapt its weights in real time rather than relying entirely on a static preference profile.

---

## 9. Personal Reflection

The biggest learning moment was realizing how much damage a single weight choice can do. Setting genre at 3.0 effectively sorted the entire catalog into buckets before any of the musical texture features — energy, valence, acousticness — had a chance to contribute meaningfully. Real systems like Spotify avoid this by learning feature weights from billions of user interactions rather than setting them by hand, but even their systems can create filter bubbles when learned weights over-reward signals that are easy to measure (genre, explicit skips) over signals that are harder to capture (emotional state, context).

Using AI tools helped with structuring the scoring formula and thinking through edge cases like conflicting preferences, but I had to check every suggested weight value manually against the catalog to see whether it actually produced sensible rankings. The most surprising moment was discovering that a profile asking for "ambient/focused" music — which feels like a completely reasonable request — got a top result that was ambient but chill rather than focused, simply because no song in the catalog combined both features. It was a concrete demonstration of how a recommender is only as good as the data it can draw from, and how gaps in that data translate directly into gaps in the experience.

