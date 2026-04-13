# Reflection: Profile Comparisons

This file documents what changed between different user profiles, why those changes make sense, and what they reveal about the recommender's logic.

---

## High-Energy Pop vs. Chill Lofi

**High-Energy Pop** (genre: pop, mood: happy, energy: 0.85) → Top result: *Sunrise City* (pop/happy, energy 0.82)  
**Chill Lofi** (genre: lofi, mood: chill, energy: 0.38) → Top result: *Library Rain* (lofi/chill, energy 0.35)

These two profiles are essentially opposites, and the recommendations reflect that cleanly. Pop/happy songs like Sunrise City score near 8.0 for the first profile because they hit all three scoring criteria: genre match, mood match, and close energy. For the Chill Lofi profile, those same songs drop to the bottom of the list since their energy (0.82–0.93) is as far from 0.38 as possible.

The key insight: the energy proximity formula penalizes songs far from the target, so the same high-energy catalog entries that help one profile actively hurt the other. The system correctly separates the two audiences without needing any extra logic.

---

## Deep Intense Rock vs. Gym Workout

**Deep Intense Rock** (genre: rock, mood: intense, energy: 0.92) → Top result: *Storm Runner* (rock/intense, energy 0.91)  
**Gym Workout** (genre: pop, mood: intense, energy: 0.95) → Top result: *Gym Hero* (pop/intense, energy 0.93)

Both profiles want intense, high-energy music, so their second and third results overlap. *Storm Runner* appears at #1 for Rock and #3 for Gym Workout (because the Gym Workout profile doesn't match its genre), and *Gym Hero* appears at #1 for Gym Workout and #2 for Deep Intense Rock (because it shares the intense mood but not the rock genre).

This comparison shows how mood and energy can pull songs toward each other across genre boundaries. A non-programmer explanation: imagine two playlists — "Intense Rock Workout" and "Pop Gym Session." They'd share a few tracks in the overlap zone (high-energy, intense-mood), but the headliners of each list would be different. That's exactly what the recommender produces.

---

## High-Energy Pop vs. Late Night Study

**High-Energy Pop** (genre: pop, mood: happy, energy: 0.85) → #1 Sunrise City (score 7.95)  
**Late Night Study** (genre: ambient, mood: focused, energy: 0.30) → #1 Spacewalk Thoughts (score 5.92)

The score gap tells the story. The High-Energy Pop profile gets a comfortable #1 at 7.95 because the catalog has a pop/happy/high-energy song (Sunrise City) that matches almost perfectly. The Late Night Study profile's top score is only 5.92 because no song in the catalog is both ambient AND focused — *Spacewalk Thoughts* is ambient/chill, so the mood match is missed. The system still surfaces the most relevant song available, but the gap between what the user wants and what the catalog has is visible in the lower score.

This is a real limitation: the recommender can only be as good as the catalog behind it. If a user wants something the catalog doesn't contain, the system compensates with partial matches, but there's no way to warn the user that their exact preference isn't represented.

---

## Chill Lofi vs. Late Night Study

**Chill Lofi** (genre: lofi, mood: chill, energy: 0.38) → #1 Library Rain (7.94), #2 Midnight Coding (7.92)  
**Late Night Study** (genre: ambient, mood: focused, energy: 0.30) → #1 Spacewalk Thoughts (5.92), #2 Focus Flow (4.84)

Both profiles prefer quiet, low-energy music, so the same pool of songs (lofi, ambient, folk) tends to float to the top. However, the Chill Lofi profile gets much tighter top scores (both top two songs above 7.9) because the catalog has two perfect lofi/chill matches. The Late Night Study profile gets more spread-out scores because its genre (ambient) only has one entry in the catalog and no song satisfies both the ambient genre AND the focused mood simultaneously.

From a user-experience standpoint: the Chill Lofi listener would be very happy with these recommendations. The Late Night Study listener would probably find the top result passable but not ideal — the vibe is right (quiet, ambient) but the mood is chill rather than focused. This gap would push a real user to skip the first result or adjust their preferences.

---

## Experiment: Double Energy Weight / Halve Genre Weight

**Baseline** (genre weight: 3.0, energy weight: 1.5) for High-Energy Pop:  
1. Sunrise City (7.95)  2. Gym Hero (5.80)  3. Rooftop Lights (4.82)

**Modified** (genre weight: 1.5, energy weight: 3.0) for same profile:  
1. Sunrise City (7.90)  2. Rooftop Lights (6.19)  3. Gym Hero (5.68)

Rooftop Lights moved from #3 to #2, while Gym Hero dropped from #2 to #3. Why? Rooftop Lights has both a mood match (happy) and a decent energy proximity (energy 0.76, target 0.85), and it benefits when energy counts for more. Gym Hero has a genre match but a worse energy proximity (energy 0.93, target 0.85) and its mood is "intense" not "happy." When genre weight is cut in half, Gym Hero loses more points than Rooftop Lights gains, which flips their positions.

The takeaway: even a small weight adjustment reshuffles the middle of the ranking. The top spot stays stable because Sunrise City satisfies all criteria. But songs that partially satisfy preferences are very sensitive to how the weights are set, which is why real systems use machine learning to tune these numbers from data rather than setting them by hand.

