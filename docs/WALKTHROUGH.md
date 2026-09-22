# Worked Example: ACT-R Style Memory Plus Our Graph, With Real Numbers

A walkthrough of the whole system using plain arithmetic. One simplification for readability: real ACT-R wraps the history sum in a logarithm to compress it, and we skip that here. The behavior is the same, the numbers are just easier to follow. All constants below are illustrative and get tuned in milestone 3 (Honda's published values anchor the tuning).

## The five rules

1. **History.** Every use of a memory leaves a trace. A trace is worth `1 / sqrt(days since that use)`. A memory's history score is the sum of its traces. Fresh trace = 1.00. A week old = 0.38. A month = 0.18. Forty days = 0.16. Fast fade, long faint tail, never zero.
2. **Relevance, two routes.** Route one: 2 x similarity between the current moment and the memory (0 to 1 from the embedding model). Route two: if a person, place, or thing present right now has an edge to the memory, add that edge's weight. Edges start at 0.5.
3. **Spread.** Any memory that gets recalled passes along its memory-to-memory edges: the neighbor receives `0.5 x (recalled memory's total score) x (edge weight)`.
4. **The bar.** Total score = history + relevance routes + received spread, plus a random wobble of about ±0.3. At or above 1.0, the memory comes to mind. Below, retrieval fails. Faded but connected memories sit just under the bar, which is exactly where spread can save them.
5. **Updates after recall.** Each recalled memory gets a fresh trace (worth 1.00 today, fading per rule 1). Each edge that carried the recall gets +0.2 weight. A neighbor that was pulled above 0.5 but not recalled gets a quarter-trace (0.25). No training anywhere. These are counters.

## The cast

Agent Ada, over 90 simulated days.

- **M1** (day 1): "Met Ben at the well. He showed me the berry patch." Edges: Ben 0.5, well 0.5, berry patch 0.5.
- **M3** (day 20): "Ate the odd mushrooms by the cave." Edge: mushrooms 0.5.
- **M4** (day 20, that night): "Sick all night. Awful." No shared words with M3. The engine adds a happened-right-after edge M3 to M4, weight 0.8 (close in time, LLM affirms a cause candidate).
- Plus mundane berry-trip memories most days, which fade in days, as they should.

## Scene 1, day 25: the entity route keeps Ben's memory alive

Ada is at the berry patch. The moment: "picking berries at the patch." Score M1:

- History: creation trace (24 days old) = 0.20, plus one recall from a trip on day 10 (15 days old) = 0.26. Total 0.46.
- Similarity: "picking berries" barely resembles "met Ben at the well." Say 0.25. Route one = 0.50.
- Entity route: the berry patch is present, and it has an edge to M1. +0.50.

Total: 0.46 + 0.50 + 0.50 = **1.46. Over the bar. M1 comes to mind.** Ada thinks of Ben while picking berries, which is exactly what a person does.

Now the same scene with the entity route turned off (that's the decay-only ablation): 0.46 + 0.50 = 0.96. **Under the bar.** Ben quietly fades from Ada's life even though she visits his berry patch every day. Over weeks, this difference compounds: each of our recalls adds a trace and bumps the patch edge and the Ben edge (+0.2 each time), so by day 90, "do you know Ben?" is an instant yes for our Ada and a coin flip for the ablated one. And note the wobble: at 1.46 with ±0.3 noise, some days M1 doesn't surface. Ada is human. She doesn't think of Ben every single time.

## Scene 2, day 60: the rescue

Ada finds odd mushrooms near the rocks. The moment: "found odd mushrooms."

Score M3 ("ate the odd mushrooms by the cave"):
- History: one trace, 40 days old = 0.16.
- Similarity: mushrooms match mushrooms, 0.8. Route one = 1.60.
- Entity route: mushrooms present, edge 0.5. +0.50.
- Total: **2.26. Recalled easily.** So far any vector store does this.

Score M4 ("sick all night"):
- History: 40 days old = 0.16.
- Similarity: "found odd mushrooms" vs "sick all night" share nothing. 0.05. Route one = 0.10.
- Entity route: nothing present connects. 0.
- Direct total: 0.26. Dead. A flat store and a decay-only store both stop here, and **Ada eats the mushrooms again.**
- Spread: M3 was just recalled at 2.26, and it has a 0.8 edge to M4. Received = 0.5 x 2.26 x 0.8 = **0.90.**
- New total: 0.26 + 0.90 = **1.16. Over the bar. The sickness comes to mind. Ada doesn't eat the mushrooms.**

That number, 0.90 arriving through an edge from a memory that shares zero words with the moment, is the entire project in one line of arithmetic.

## Scene 3, the updates, and why the aversion wears in

After scene 2: M3 and M4 each get a fresh trace. The M3-M4 edge goes 0.8 + 0.2 = 1.0. The mushrooms-M3 edge goes 0.5 + 0.2 = 0.7.

Day 90, mushrooms again. M3 scores about 0.30 history + 1.60 + 0.70 = 2.60. Spread to M4: 0.5 x 2.60 x 1.0 = 1.30, before M4's own improved history even counts. The rescue that barely cleared the bar at day 60 is now automatic. Repetition carved the association in. That is conditioning, produced by counters.

## Side note: the reminder bump

Day 40, a traveler mentions the well, and M1 gets recalled once. The new trace is worth 1.00 the next day, so for a few days Ben keeps popping into Ada's head. A week later that trace is worth 0.38, a month later 0.18, and M1 is back near its old baseline. One reminder = one fading trace. Under the comparison kernel (Hou), that same mention resets M1's whole clock and permanently raises its strength, so Ada dwells on Ben for weeks. The reminiscence probe measures exactly this difference.

## What to notice

- **Three ways in.** A memory surfaces because the moment resembles it, because something present is wired to it, or because a recalled neighbor pulled it. Only the first exists in a vector store.
- **The bar makes forgetting real.** M4 at 0.26 is functionally forgotten. It still exists, which is why an edge can save it. Deletion could never be rescued.
- **Survival is social.** M1 stays alive because Ada's routine keeps touching things connected to it, not because she rehearses meeting Ben.
- **The noise is a feature.** Borderline memories surface some days and not others. Blank now, remember later.
- **Everything is a counter.** Traces, weights, bumps. No training, fully deterministic given a seed, and every recall can be explained after the fact: "M4 scored 1.16, of which 0.90 arrived through the edge from M3."
