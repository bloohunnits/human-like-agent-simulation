# human-like-agent-simulation

**Goal: give LLM agents a memory that works the way human memory does — memories fade with time, get stronger every time they're recalled, and are *connected to each other* — then put those agents in a living simulated world and watch what changes.**

The point is human-likeness, not compute or retrieval efficiency. Faster lookup and fewer tokens are nice side effects if they happen; the thing we're actually chasing is agents that remember, forget, and associate the way people do.

---

## Where this starts

Two papers anchor the project:

1. **The human-like memory paper** — ["My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents](https://arxiv.org/abs/2404.00573) (Hou, Tamoto, Miyashita, CHI 2024). One closed-form recall probability that combines exactly the three factors we care about:
   - **Relevance** — how related the memory is to what's happening now (cosine similarity `r`)
   - **Time decay** — how long since the memory was last touched (`t`)
   - **Memory strength** — a consolidation gradient `g` that grows each time the memory is recalled, so recalled memories flatten their own forgetting curve. Spaced recalls strengthen more than crammed ones, just like in humans.

   Closely related and worth mining: [Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (HAI 2025), which does the same three factors via the ACT-R cognitive architecture, and [MemoryBank](https://arxiv.org/abs/2305.10250) (AAAI 2024), which uses the Ebbinghaus forgetting curve.

2. **The OG paper** — [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (Park et al., 2023, "Smallville"). The original recency × importance × relevance memory stream, plus **reflection**: agents periodically distill recent memories into higher-level insights. See [docs/RESEARCH.md](docs/RESEARCH.md) for exactly what we're taking from it and what it got wrong (nothing is ever forgotten, importance never changes, and memory is a flat list).

## What we add: relations between memories

Every system above treats memories as **isolated items ranked by a score**. But human memory is associative — one memory pulls up another. Pure embedding similarity misses connections that run through shared people, places, causes, and chains of events ("the friend of the person I met at the market").

So we add a **memory graph**:

- Memories are nodes carrying content, timestamps, and a strength value that decays and is reinforced per the human-like model.
- Edges link memories that share entities, follow each other in time, cause each other, or serve as evidence for a reflection/insight.
- Recall **spreads along edges** (spreading activation): recalling a memory partially re-activates and strengthens its neighbors, so an association can rescue a faded memory that similarity search alone would never surface.
- Reflection (from the OG paper) becomes a graph operation: an insight node with typed evidence edges down to the memories it came from.

As far as we can find, **nobody has published this combination** — graph memory systems (HippoRAG, Zep, A-Mem, Mem0-g) have no decay or strength dynamics, and decay/strength systems (Hou et al., MemoryBank, Generative Agents) have no relational structure. Details and near-misses in [docs/RESEARCH.md](docs/RESEARCH.md).

## Why this matters

- **Believable agents.** Characters in simulations and games that forget acquaintances but remember friends, that need reminding, that free-associate — because their memory actually works that way, not because a prompt tells them to act forgetful.
- **Forgetting is a feature, not a bug.** Human forgetting is adaptive: it clears out the stale and the trivial and keeps what keeps mattering. An agent that remembers everything forever gets *less* human over time, and drowning in its own history is its failure mode.
- **Association is how remembering actually happens.** People retrieve by connection, not by nearest-neighbor search over everything they've ever experienced. If we want human-like agents, retrieval has to follow relationships.
- **A real scientific hole.** Decay + reinforcement fused *into* graph retrieval is unexplored, and the one adjacent attempt (forgetting used only to prune a graph) produced a negative result — so the question is genuinely open, and we have a baseline to beat.
- **Long-lived companions and assistants** stand to benefit downstream — but that's a consequence, not the goal.

## What we take from Maya

Our own [Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) project is the OG paper plus custom engineering, and it feeds this project on three fronts:

- **Architecture to reuse** — the deterministic-engine + validated-LLM-meaning-layer pattern (decay math and graph traversal deterministic and seeded; the LLM only rates, links, and reflects, with validation), citation-checked reflections (which are literally the evidence edges our graph needs, already prototyped), every cognition feature behind an ablation config flag, and the audit-trail/decision-inspector approach that can make decay and association *visible* ("this memory lost by 0.1 because it faded; this one was rescued by an edge").
- **Mistakes its own gap analysis warns about** — eval suites that never touch the LLM layer, no memory-retrieval precision metric, single-seed science, no LLM record/replay, and simulated time running too fast to watch. We build the fixes in from day one.
- **Maya as a testbed** — its memory stream is a swappable module in our own codebase. Likely play: develop the memory engine against Maya (fast, instrumented, ours), then deploy into TerraLingua for the open-ended experiments. The engine also back-fills Maya's missing loops (trait evolution, relationship decay).

Details in [docs/GAPS.md](docs/GAPS.md).

## The simulation: TerraLingua

[TerraLingua](https://github.com/cognizant-ai-lab/terralingua) (Cognizant AI Lab, Apache 2.0) is a persistent 2D world where LLM agents forage, reproduce, die, and — the interesting part — **generate the environment's content themselves**: they write, modify, and exchange persistent text artifacts that outlive their creators, so later generations inherit an LLM-authored culture.

Why it fits this project unusually well:

- **Real simulated time with birth and death** — decay and strength actually matter over an agent's lifetime.
- **Recurring agents, places, and artifacts** — exactly the recurring entities a memory graph is about.
- **LLM-generated environment** — the world's content is open-ended, so memory shapes culture and culture shapes memory.

Caveat: TerraLingua has **no pluggable memory interface** — memory is implicit in context plus the shared artifact layer. We'd wire our memory system into the agent loop ourselves. That's feasible (small young Python codebase) but it's real work, so we're keeping two fallbacks on the table: **Generative Agents / [AI Town](https://github.com/a16z-infra/ai-town)** (its memory stream is literally a drop-in slot for our model) and **[Concordia](https://github.com/google-deepmind/concordia)** (DeepMind; component-based agents with swappable memory, and an LLM game master generates the world — closest in spirit to TerraLingua with far better extension points). The environment decision is an open question, tracked in [docs/GAPS.md](docs/GAPS.md).

## How we'll measure it

Human-likeness first, benchmarks as a sanity check:

1. **In-simulation observation** — do agents with this memory behave more like people? Relationship formation, selective forgetting, memory-driven behavior differences vs. baseline agents. Generative Agents' believability interviews and ablation design are a reusable template; TerraLingua's "AI Anthropologist" analysis layer helps here.
2. **Ablations** — turn off decay, turn off reinforcement, turn off edges, one at a time. Each mechanism should visibly change behavior, or it isn't earning its place.
3. **Existing memory benchmarks** — [LongMemEval](https://arxiv.org/abs/2410.10813) (primary), [LoCoMo](https://www.emergentmind.com/topics/locomo), [MemBench](https://arxiv.org/abs/2506.21605) — to verify human-like memory doesn't wreck task utility, and to compare against Mem0, A-Mem, Zep, and flat vector baselines.
4. **New probes for what no benchmark measures** — forgetting *quality* (did the right things fade?), interference between similar memories, and multi-hop recall across faded memories that only association can rescue.

## Repo map

This repo is docs only — the planning and research home for the project. Code lives elsewhere when we start building.

- [docs/RESEARCH.md](docs/RESEARCH.md) — the full paper survey: source papers, the OG paper's mechanics, graph memory systems, benchmarks, with links.
- [docs/GAPS.md](docs/GAPS.md) — what the papers don't cover, what we take from Maya, and the open decisions we still have to make.

## Roadmap

1. **Ground truth** — finish reading the source papers; lock the exact decay/strength/relevance formulation we're improving.
2. **Core memory engine** — standalone library: memory nodes with decay + recall reinforcement + relevance, reproducing the human-like paper's behavior.
3. **Graph layer** — typed edges, LLM-assisted linking at write time, spreading-activation recall.
4. **Benchmark harness** — LongMemEval + LoCoMo runs with ablations and baselines, plus our forgetting-quality probes.
5. **Simulation** — wire the memory into TerraLingua (or the chosen fallback) and run the human-likeness studies.
6. **Write-up** — findings, honest negatives included.
