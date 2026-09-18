# human-like-agent-simulation

Give LLM agents a memory that works like ours: memories fade with time, get stronger each time they're recalled, and pull each other up by association. Then drop those agents into a living simulated world and see what changes.

The goal is human-likeness. Not faster retrieval, not fewer tokens — if those happen, fine. We want agents that remember, forget, and free-associate the way people do.

## The idea

### The starting point: memory as a fading, strengthening curve

The paper we build on ([Hou et al., CHI 2024](https://arxiv.org/abs/2404.00573)) gives every memory a *probability of coming to mind* right now:

```
p = [1 − exp(−r · e^(−t/g))] / [1 − e^(−1)]
```

Strip the math and it's three claims about how remembering works:

- **Relevance (`r`)** — a memory only surfaces when the present resembles it. Nothing around you reminds you of it, it stays buried. (Measured as similarity between the current context and the memory.)
- **Time (`t`)** — the longer since you last touched a memory, the deeper it sinks. `e^(−t/g)` is the classic forgetting curve: steep at first, then flattening.
- **Strength (`g`)** — how well-worn the memory is. Every recall bumps `g` up, and a bigger `g` slows the decay. Remembering is rehearsal: each recall re-carves the groove.

The subtle part is *how* strength grows: a recall after a long gap adds more strength than an immediate re-recall. That's the human spacing effect — cramming fades, spaced review sticks — falling out of one small formula. (The denominator just normalizes so a perfectly relevant, just-recalled memory scores 1.)

So an agent running this forgets naturally and keeps what keeps mattering. That's already far more human than the standard "vector store that remembers everything forever."

### What's missing: every memory lives and dies alone

In that model, each memory is an isolated jar on a shelf, fading on its own schedule. A memory can only be reached if the *current moment* directly resembles *it*.

Humans don't work that way. Recall is chained: one memory tugs on the next through shared people, places, causes, and moments. A weak cue can reach a faded memory *through* a strong neighboring one — you can't remember the restaurant until you remember who you were with, and then it all comes back. Embedding similarity can't do that chain, because the target memory may share nothing, textually or semantically, with the cue.

### The improvement: a memory graph

We keep the paper's per-memory dynamics and connect the jars:

- **Memories become nodes** carrying the `r`/`t`/`g` dynamics above.
- **Typed edges link them**: same person or place, happened-right-after, part of the same conversation, evidence-for-an-insight.
- **Recall spreads.** Retrieval starts from what's relevant now, then activation flows along edges — recalling a memory partially re-activates its neighbors, weighted by their strength. (This is spreading activation, straight out of the ACT-R cognitive architecture.)
- **Recall reinforces structure.** A recall strengthens the node *and* the links it traveled — associations themselves get worn in or fade.
- **Forgetting becomes structural.** An isolated memory fades fastest. A well-connected one keeps getting rescued by its network. That's the human pattern: trivia goes, the woven-in stuff stays.

Concrete example: ask an agent about the market. Similarity search finds market memories — fine. But the edge from a market memory to *the person you met there*, and from that person to *their friend*, surfaces a memory that shares not one word with "market." That class of recall is the whole point.

As far as we can find, **nobody has published this combination.** Graph memory systems (HippoRAG, Zep, A-Mem, Mem0-g) have no decay or strength; decay systems (Hou et al., MemoryBank, Generative Agents) have no relations. The one adjacent attempt used forgetting only to *prune* a graph — and lost to flat vector search — so the question is genuinely open and we have a named baseline to beat. Details in [docs/RESEARCH.md](docs/RESEARCH.md).

## The papers

- **Source**: [Hou et al., CHI 2024](https://arxiv.org/abs/2404.00573) — the formula above. Sibling: an [ACT-R-inspired architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (HAI 2025) with the same three factors plus native spreading activation and decades of human-fitted parameters. Simpler cousin: [MemoryBank](https://arxiv.org/abs/2305.10250) (Ebbinghaus curve).
- **The OG**: [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al. 2023, "Smallville") — the original recency × importance × relevance memory stream plus reflection. We inherit its scoring skeleton and its reflection-as-consolidation trick; we fix what it lacks (nothing ever fades, importance never changes, memory is a flat list). Full breakdown in [docs/RESEARCH.md](docs/RESEARCH.md).

## Why bother

- **Believable agents.** Characters that forget acquaintances but remember friends, need reminding, and free-associate — because their memory actually works that way, not because a prompt says "act forgetful."
- **Forgetting is a feature.** Human forgetting is adaptive: it clears the stale and trivial and keeps what recurs. An agent that remembers everything forever gets *less* human over time and drowns in its own history.
- **Association is how remembering happens.** People retrieve by connection, not nearest-neighbor search over their whole life.
- **A real scientific hole**, per above — with an honest chance the answer is "the graph doesn't help." We'd publish that too.
- Long-lived companions and assistants benefit downstream — consequence, not goal.

## What we take from Maya

[Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) is our own Generative Agents implementation — the OG paper plus custom engineering — and it feeds this project three ways:

- **Architecture worth copying**: the deterministic-engine + validated-LLM pattern (decay math and graph traversal are seeded and deterministic; the LLM only rates, links, and reflects, with every output validated), citation-checked reflections (already the evidence edges our graph needs), every feature behind an ablation flag, and decision-inspector explainability that can make memory visible — "this memory lost by 0.1 because it faded; this one got rescued by an edge."
- **Mistakes its own gap analysis warns about**: eval suites that never touch the LLM layer, no retrieval-precision metric, single-seed science, no LLM record/replay, sim time too fast to watch. We build the fixes in from day one.
- **A testbed**: Maya's memory stream is a swappable module in our own codebase. Likely play — develop the engine against Maya (fast, instrumented, ours), then deploy into TerraLingua. The engine also back-fills Maya's missing loops (trait evolution, relationship decay).

Details in [docs/GAPS.md](docs/GAPS.md).

## The world: TerraLingua

[TerraLingua](https://github.com/cognizant-ai-lab/terralingua) (Cognizant AI Lab, Apache 2.0) is a persistent 2D world where LLM agents forage, reproduce, die — and write the environment themselves: persistent text artifacts that outlive their authors, so later generations inherit an LLM-made culture.

It fits unusually well: real simulated time with birth and death (decay matters), recurring agents and artifacts (the graph matters), and open-ended LLM-generated content (memory shapes culture, culture shapes memory). Caveat: no pluggable memory interface — we'd wire into the agent loop ourselves. Fallbacks if that spikes badly: [AI Town](https://github.com/a16z-infra/ai-town) / [Smallville](https://github.com/joonspk-research/generative_agents) (drop-in memory slot) and DeepMind's [Concordia](https://github.com/google-deepmind/concordia) (swappable memory components, LLM game master). Open decision, tracked in [docs/GAPS.md](docs/GAPS.md).

## How we'll know it worked

1. **Watch the agents.** Do they behave more like people — forming relationships, forgetting selectively, acting on associations? Generative Agents' believability interviews and TerraLingua's "AI Anthropologist" log analysis are the templates.
2. **Ablate everything.** Decay off, reinforcement off, edges off — one at a time. Each mechanism visibly changes behavior or it's cut.
3. **Benchmarks as a sanity check** — [LongMemEval](https://arxiv.org/abs/2410.10813) (primary), [LoCoMo](https://www.emergentmind.com/topics/locomo), [MemBench](https://arxiv.org/abs/2506.21605) — human-like memory must not wreck task utility, against Mem0, A-Mem, Zep, and flat-vector baselines.
4. **New probes for what no benchmark measures**: did the *right* things fade, does interference behave sanely, and can association rescue faded memories that similarity alone loses.

## Repo map

Docs only — this is the planning and research home. Code lives elsewhere once we build.

- [docs/RESEARCH.md](docs/RESEARCH.md) — full paper survey: source papers, OG mechanics, graph systems, benchmarks, environments.
- [docs/GAPS.md](docs/GAPS.md) — what the papers don't cover, what Maya fills, open decisions.

## Roadmap

1. **Ground truth** — finish the source papers (incl. getting the paywalled ACT-R one); lock the exact formulation.
2. **Core engine** — standalone memory library reproducing the human-like paper's decay/strength/relevance behavior.
3. **Graph layer** — typed edges, LLM-assisted linking at write time, spreading-activation recall.
4. **Benchmark harness** — LongMemEval + LoCoMo with ablations, baselines, and our forgetting-quality probes.
5. **Simulation** — wire into Maya, then TerraLingua; run the human-likeness studies.
6. **Write-up** — findings, honest negatives included.
