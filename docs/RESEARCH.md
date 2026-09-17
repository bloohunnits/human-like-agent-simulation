# Research Notes

The paper survey behind the project. Three threads: the human-like memory papers we're improving, the graph-memory systems we're borrowing relation mechanics from, and the benchmarks/environments we'll evaluate in. Compiled 2026-09-17.

## 1. The human-like memory papers (decay + strength + relevance)

### Primary: Hou, Tamoto & Miyashita (CHI 2024)

["My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents](https://arxiv.org/abs/2404.00573)

One closed-form recall probability combining all three factors:

```
p_n(t) = [1 − exp(−r · e^(−t / g_n))] / [1 − e^(−1)]
```

- `r` — **relevance**: cosine similarity between current context and the memory
- `t` — **time decay**: seconds since the memory was last retrieved
- `g_n` — **strength** (consolidation gradient): `g_n = g_{n−1} + S(t)` with `g_0 = 1` and `S(t) = (1−e^(−t)) / (1+e^(−t))` — every recall flattens the decay curve, and *spaced* recalls strengthen more than massed ones (the human spacing effect).

This is the architecture we start from. What it doesn't have: any structure *between* memories — each memory decays and strengthens alone.

### Sibling: ACT-R-inspired architecture (HAI 2025)

[Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (Honda, Fujita, Zempo, Fukushima; ACM full text paywalled, no arXiv preprint found)

Same three factors via the ACT-R cognitive architecture: base-level activation `B_i = ln(Σ_j t_j^(−d))` (power-law decay over every past retrieval, `d ≈ 0.5`), retrieval adds terms to the sum (strength = frequency + recency), context relevance as a spreading-activation analog, plus noise and a retrieval threshold below which memories are functionally forgotten. Valuable to us because ACT-R's **spreading activation** is the cognitive-science ancestor of the graph mechanism we want, and its parameters are already fit to decades of human data.

### Also relevant

- [MemoryBank](https://arxiv.org/abs/2305.10250) (AAAI 2024) — Ebbinghaus curve `R = e^(−t/S)`; strength `S` starts at 1, each recall increments `S` and resets `t`. Simple, widely cited; decay can delete memories outright. Relevance is retrieval-side (FAISS cosine), not fused into one score.
- [Memoria](https://arxiv.org/abs/2310.03052) (ICML 2024) — Hebbian engram strength between memory units; the rare prior work with *connections* between memories, but no clock-time decay and operates inside the model rather than as an agent memory store.
- [FadeMem](https://arxiv.org/pdf/2601.18642) (2026) — differential decay modulated by relevance and access frequency; recent, worth a read.

## 2. The OG paper: Generative Agents (Park et al., 2023)

[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — the Smallville paper; also the basis of our own [Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) project (see [GAPS.md](GAPS.md) § Maya).

**Its memory model:**
- Append-only **memory stream** of natural-language records, each with creation and last-accessed timestamps.
- Retrieval score = recency + importance + relevance (equal weights, each min-max normalized to [0,1]):
  - *Recency*: exponential decay over sandbox time since last retrieval, factor 0.995
  - *Importance*: LLM rates 1–10 at creation ("1 mundane … 10 poignant") — **static forever after**
  - *Relevance*: cosine similarity to the query
- **Reflection**: when summed importance of recent events exceeds 150, the agent generates "salient questions," retrieves against them, and writes insight records that **cite their evidence** ("because of 1, 8, 15"); reflections can recurse into trees of abstraction.

**What it lacks (our checklist):**
- No true forgetting — nothing is ever removed or suppressed; recency only reranks
- No strength/reinforcement — importance never updates; retrieval doesn't consolidate
- Flat linear stream — no relations between memories except implicit reflection citations
- Hand-set equal weights; decay tied to the clock, not to usage patterns

**What we take from it:**
- The normalized weighted-sum scoring skeleton (swap recency → real decay, importance → dynamic strength initialized by the LLM's write-time rating)
- Reflection as consolidation — and its evidence citations are *literally edges*: in a graph memory they become first-class typed edges from insight to evidence, the relational structure the paper never materialized
- Recursive reflection → graph hierarchy levels; strength can propagate along edges (recalling a node partially strengthens neighbors — spreading activation, per ACT-R)
- Its evaluation framing: believability interviews + ablations (full stack vs. no-reflection vs. no-planning) — reusable as ablate-decay / ablate-strength / ablate-edges

## 3. Graph memory systems (the relation mechanics)

| System | Graph mechanism | Decay/strength? |
|---|---|---|
| [HippoRAG / HippoRAG 2](https://github.com/osu-nlp-group/hipporag) (NeurIPS'24, [ICML'25](https://arxiv.org/abs/2502.14802)) | LLM extracts entities/triples into an open KG; retrieval = **Personalized PageRank** seeded from query entities, so activation spreads across relations embeddings miss. ~20% over SOTA RAG on multi-hop QA. | None — append-only, monotonic |
| [Zep / Graphiti](https://arxiv.org/abs/2501.13956) ([code](https://github.com/getzep/graphiti)) | Hierarchical temporal KG (episode → entity → community); **bi-temporal edges** with validity intervals — contradicted facts get *invalidated*, not deleted. +18.5% on LongMemEval vs baselines. | Discrete truth-maintenance only — no graded decay or use-based strength |
| [A-Mem](https://arxiv.org/abs/2502.12110) | Zettelkasten notes; LLM links new notes to old ones and *evolves* old notes when related ones arrive. Big multi-hop gains on LoCoMo at ~1–2.5k context tokens. | None |
| [Mem0 / Mem0-g](https://arxiv.org/pdf/2504.19413) | Fact extraction with ADD/UPDATE/DELETE; graph variant adds entity/relation store with conflict detection. LoCoMo LLM-judge: 66.9 → 68.4 with graph. | None — deletion is contradiction-driven, not time-driven |
| [GraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) → LazyGraphRAG, LightRAG | Corpus KG + community summaries for "global" questions. | Static-corpus RAG, not agent memory |

### Prior work combining decay + reinforcement + graph? Essentially none. Three near-misses:

1. [Selective Forgetting](https://arxiv.org/html/2608.28978) — KG memory with a forgetting score (recency, access frequency, centrality, age) used **only to prune** nodes; and its graph *underperformed* flat vector retrieval on LongMemEval (0.417 vs 0.468 F1). Both prior art and a cautionary negative result — the open question "does strength-weighted graph retrieval beat flat retrieval?" is genuinely unanswered, and this is the baseline to beat.
2. [Not All Memories Age the Same](https://arxiv.org/html/2604.26970v1) — learns per-predicate/entity decay rates in KGs via survival analysis; headline finding that **uniform decay was 18× worse than no decay** — decay rates must be heterogeneous. A KG-freshness paper, not an agent memory; no recall dynamics.
3. [PowerMem](https://github.com/oceanbase/powermem) (OceanBase) — engineering product with Ebbinghaus decay + hybrid vector/graph retrieval, but decay is a re-ranking layer bolted *beside* the graph — no decay-aware traversal, no research paper.

**The open slot: decay and reinforcement fused *into* graph retrieval itself** — strength-weighted PageRank / spreading activation that decays — evaluated for human-likeness and on standard benchmarks.

## 4. Benchmarks

Benchmarks are our sanity check (human-like memory must not wreck utility), not the goal.

- **[LongMemEval](https://arxiv.org/abs/2410.10813)** (ICLR 2025, [code](https://github.com/xiaowu0162/longmemeval)) — primary. 500 questions over long chat histories testing extraction, multi-session reasoning, **temporal reasoning, knowledge updates, abstention**. Commercial assistants drop ~30%.
- **[LoCoMo](https://www.emergentmind.com/topics/locomo)** (ACL 2024) — ~1,540 QA over very-long multi-session conversations (6–12 simulated months); 20.8% temporal questions. Known cross-paper scoring inconsistency (the Mem0-vs-Zep dispute) — report both F1 and LLM-judge with a pinned judge model.
- **[MemBench](https://arxiv.org/abs/2506.21605)** (ACL 2025 Findings) — factual + reflective memory; its **capacity axis** (degradation as the store grows) is the only mainstream metric that touches forgetting behavior.
- Secondary: [DMR](https://arxiv.org/abs/2501.13956) (near-saturated; cite, don't build on), [MultiChallenge](https://scale.com/research/multichallenge) (in-context multi-turn), [Test of Time](https://arxiv.org/abs/2406.09170) (synthetic temporal logic — could seed our own decay/interference probes).
- Memory-specific 2025–26: [AMA-Bench](https://arxiv.org/html/2602.22769), [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (ICLR 2026), [WorldLines](https://arxiv.org/pdf/2606.18847), [Vending-Bench](https://arxiv.org/abs/2502.15840) (long-horizon coherence).

## 5. Simulation environments

- **[TerraLingua](https://github.com/cognizant-ai-lab/terralingua)** (Cognizant AI Lab, 2026, Apache 2.0 · [project page](https://www.cognizant.com/us/en/ai-lab/terralingua) · [blog](https://www.cognizant.com/us/en/ai-lab/blog/when-ai-agents-build-societies-terralingua)) — persistent 2D grid world; LLM agents forage, reproduce, die, and author persistent **text artifacts** that outlive them (LLM-generated environment/culture). A separate "AI Anthropologist" LLM analyzes logs. **No pluggable memory interface** — we'd modify the agent loop (small, young Python codebase: ~72 stars). Generational turnover + recurring artifacts make decay and relations genuinely matter.
- **[Generative Agents / Smallville](https://github.com/joonspk-research/generative_agents)** — its memory stream is a direct drop-in slot for our model; aging codebase, costly to run. **[AI Town](https://github.com/a16z-infra/ai-town)** (TypeScript/Convex re-implementation, maintenance mode, MIT) — easiest hack target, shallow world.
- **[Concordia](https://github.com/google-deepmind/concordia)** (DeepMind) — component-based agents with **swappable associative memory**, LLM game master generates the world. Best-maintained framework option; text-only, narrative time.
- **Maya** (ours) — Generative Agents-faithful sim with deterministic engine, swappable memory stream, decision inspector, audit trail, ablation flags. See [GAPS.md](GAPS.md).
- Considered and set aside: Project Sid (code unreleased), Voyager/MineDojo (single-agent, procedural skills not social memory), AgentSociety/OASIS (scale over per-agent depth), Sotopia (episodes too short for decay), Crafter/Craftax (rule-generated content, no recurring named entities).
