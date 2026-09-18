# Research Notes

The survey behind the project, compiled 2026-09-17. Three threads: the human-like memory papers we're improving, the graph systems we're borrowing relation mechanics from, and the benchmarks and environments we'll evaluate in. The plain-language version of the core idea is in the [README](../README.md); this file keeps the precise details.

## 1. The human-like memory papers

### Primary: Hou, Tamoto & Miyashita (CHI 2024)

["My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents](https://arxiv.org/abs/2404.00573)

One recall probability per memory:

```
p_n(t) = [1 − exp(−r · e^(−t / g_n))] / [1 − e^(−1)]
```

`r` = cosine similarity between current context and memory (relevance). `t` = seconds since last retrieval (decay). `g_n` = consolidation gradient (strength): `g_n = g_{n−1} + S(t)` with `g_0 = 1` and `S(t) = (1−e^(−t))/(1+e^(−t))` — each recall flattens the memory's own forgetting curve, and spaced recalls add more strength than massed ones (the spacing effect). What it lacks: any structure *between* memories.

### Sibling: ACT-R-inspired architecture (HAI 2025)

[Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (Honda, Fujita, Zempo, Fukushima — ACM paywalled, no arXiv preprint found; we still need access.)

Same three factors via ACT-R: base-level activation `B_i = ln(Σ_j t_j^(−d))` — power-law decay over every past retrieval, `d ≈ 0.5`; each retrieval adds a term (strength = frequency + recency); context relevance as a spreading-activation analog; plus noise and a retrieval threshold below which a memory is functionally forgotten. Matters to us because ACT-R's spreading activation is the cognitive-science ancestor of our graph mechanism, with parameters already fit to decades of human data.

### Also relevant

- [MemoryBank](https://arxiv.org/abs/2305.10250) (AAAI 2024) — Ebbinghaus curve `R = e^(−t/S)`; each recall increments `S` and resets `t`; decay can delete outright. Relevance is retrieval-side (FAISS cosine), not fused into the score.
- [Memoria](https://arxiv.org/abs/2310.03052) (ICML 2024) — Hebbian engram strength *between* memory units; rare prior work with connections, but no clock-time decay and it lives inside the model, not in an agent store.
- [FadeMem](https://arxiv.org/pdf/2601.18642) (2026) — differential decay modulated by relevance and access frequency.

## 2. The OG paper: Generative Agents (Park et al., 2023)

[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — the Smallville paper, and the basis of our own [Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) (see [GAPS.md](GAPS.md)).

**Its memory model.** An append-only stream of natural-language records with creation and last-accessed timestamps. Retrieval score = recency + importance + relevance, equal weights, each min-max normalized: recency is exponential decay over sandbox time (factor 0.995), importance is an LLM rating of 1–10 at creation — static forever — and relevance is cosine similarity. **Reflection**: when summed importance of recent events passes 150, the agent generates "salient questions," retrieves against them, and writes insight records that cite their evidence ("because of 1, 8, 15"); reflections recurse into trees.

**What it lacks** (our checklist): nothing is ever forgotten (recency only reranks); no reinforcement (importance never updates, retrieval doesn't consolidate); a flat stream with no relations beyond implicit citations; hand-set weights; decay tied to the clock rather than usage.

**What we take**: the normalized weighted-sum skeleton (recency → real decay, importance → dynamic strength seeded by the write-time rating); reflection as consolidation, whose citations are literally edges — in a graph they become first-class insight→evidence links, the structure the paper never materialized; recursive reflection → graph hierarchy, with strength propagating along edges (spreading activation, per ACT-R); and its evaluation framing — believability interviews plus ablations, reusable as ablate-decay / ablate-strength / ablate-edges.

## 3. Graph memory systems

| System | Graph mechanism | Decay/strength? |
|---|---|---|
| [HippoRAG / HippoRAG 2](https://github.com/osu-nlp-group/hipporag) (NeurIPS'24, [ICML'25](https://arxiv.org/abs/2502.14802)) | LLM extracts entities/triples into an open KG; retrieval = Personalized PageRank seeded from query entities. ~20% over SOTA RAG on multi-hop QA. | None — append-only |
| [Zep / Graphiti](https://arxiv.org/abs/2501.13956) ([code](https://github.com/getzep/graphiti)) | Hierarchical temporal KG (episode → entity → community); bi-temporal edges with validity intervals — contradicted facts are invalidated, not deleted. +18.5% on LongMemEval vs baselines. | Discrete truth-maintenance only |
| [A-Mem](https://arxiv.org/abs/2502.12110) | Zettelkasten notes; an LLM links new notes to old and evolves old ones as related notes arrive. Big multi-hop LoCoMo gains at ~1–2.5k context tokens. | None |
| [Mem0 / Mem0-g](https://arxiv.org/pdf/2504.19413) | Fact extraction with ADD/UPDATE/DELETE; graph variant adds an entity/relation store with conflict detection. LoCoMo LLM-judge 66.9 → 68.4 with graph. | None — deletion is contradiction-driven |
| [GraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) → LazyGraphRAG, LightRAG | Corpus KG + community summaries for "global" questions. | Static-corpus RAG, not agent memory |

### Has anyone combined decay + reinforcement + graph? Essentially no. Three near-misses:

1. [Selective Forgetting](https://arxiv.org/html/2608.28978) — KG memory with a forgetting score (recency, access frequency, centrality, age) used **only to prune**; its graph *underperformed* flat vector retrieval on LongMemEval (0.417 vs 0.468 F1). Both prior art and our baseline to beat.
2. [Not All Memories Age the Same](https://arxiv.org/html/2604.26970v1) — learns per-predicate/entity decay rates via survival analysis; found **uniform decay 18× worse than no decay**, so rates must be heterogeneous. KG-freshness work, not agent memory; no recall dynamics.
3. [PowerMem](https://github.com/oceanbase/powermem) (OceanBase) — engineering product with Ebbinghaus decay + hybrid vector/graph retrieval, but decay is a re-ranking layer *beside* the graph; no decay-aware traversal, no paper.

The open slot: decay and reinforcement fused *into* graph retrieval itself — strength-weighted PageRank / spreading activation that fades — evaluated for human-likeness and on standard benchmarks.

## 4. Benchmarks

Sanity check, not the goal: human-like memory must not wreck utility.

- **[LongMemEval](https://arxiv.org/abs/2410.10813)** (ICLR 2025, [code](https://github.com/xiaowu0162/longmemeval)) — primary. 500 questions over long chat histories: extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. Commercial assistants drop ~30%.
- **[LoCoMo](https://www.emergentmind.com/topics/locomo)** (ACL 2024) — ~1,540 QA over multi-session conversations spanning 6–12 simulated months; 20.8% temporal. Cross-paper scoring is inconsistent (the Mem0/Zep dispute) — report F1 *and* LLM-judge with a pinned judge.
- **[MemBench](https://arxiv.org/abs/2506.21605)** (ACL 2025 Findings) — factual + reflective memory; its capacity axis (degradation as the store grows) is the only mainstream metric that even touches forgetting.
- Secondary: [DMR](https://arxiv.org/abs/2501.13956) (near-saturated; cite, don't build on), [MultiChallenge](https://scale.com/research/multichallenge) (in-context multi-turn), [Test of Time](https://arxiv.org/abs/2406.09170) (synthetic temporal logic — could seed our own probes).
- Memory-specific 2025–26: [AMA-Bench](https://arxiv.org/html/2602.22769), [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (ICLR 2026), [WorldLines](https://arxiv.org/pdf/2606.18847), [Vending-Bench](https://arxiv.org/abs/2502.15840).

## 5. Simulation environments

- **[TerraLingua](https://github.com/cognizant-ai-lab/terralingua)** (Cognizant AI Lab 2026, Apache 2.0 · [page](https://www.cognizant.com/us/en/ai-lab/terralingua) · [blog](https://www.cognizant.com/us/en/ai-lab/blog/when-ai-agents-build-societies-terralingua)) — persistent 2D grid world; agents forage, reproduce, die, and author persistent text artifacts that outlive them; an "AI Anthropologist" LLM analyzes the logs. No pluggable memory interface — we'd modify the agent loop (small young Python codebase, ~72 stars). Generational turnover + recurring artifacts make decay and relations genuinely matter.
- **[Smallville](https://github.com/joonspk-research/generative_agents)** — memory stream is a direct drop-in slot; aging and costly to run. **[AI Town](https://github.com/a16z-infra/ai-town)** (TypeScript/Convex re-implementation, MIT, maintenance mode) — easiest hack target, shallow world.
- **[Concordia](https://github.com/google-deepmind/concordia)** (DeepMind) — component-based agents with swappable associative memory; an LLM game master generates the world. Best-maintained option; text-only, narrative time.
- **Maya** (ours) — deterministic engine, swappable memory stream, decision inspector, audit trail, ablation flags. See [GAPS.md](GAPS.md).
- Set aside: Project Sid (code unreleased), Voyager/MineDojo (single-agent, procedural skills), AgentSociety/OASIS (scale over per-agent depth), Sotopia (episodes too short for decay), Crafter/Craftax (rule-generated content, no recurring named entities).
