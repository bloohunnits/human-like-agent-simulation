# Research Notes

The survey behind the project, compiled 2026-09-17. Three threads: the human-like memory papers we build on, the graph systems we borrow relation mechanics from, and the benchmarks and environments we evaluate in. The plain-language version of the core idea is in the [README](../README.md). This file keeps the precise details.

## 1. The human-like memory papers

### Primary: Hou, Tamoto & Miyashita (CHI 2024)

["My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents](https://arxiv.org/abs/2404.00573)

One recall probability per memory:

```
p_n(t) = [1 − exp(−r · e^(−t / g_n))] / [1 − e^(−1)]
```

`r` is cosine similarity between current context and memory (relevance). `t` is seconds since last retrieval (decay). `g_n` is the consolidation gradient (strength): `g_n = g_{n−1} + S(t)` with `g_0 = 1` and `S(t) = (1−e^(−t))/(1+e^(−t))`. Each recall flattens the memory's own forgetting curve, and spaced recalls add more strength than massed ones (the spacing effect). What it lacks: any structure between memories.

### Sibling: ACT-R-inspired architecture (HAI 2025)

[Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (Honda, Fujita, Zempo, Fukushima. ACM paywalled, no arXiv preprint found. We still need access.)

Same three factors via ACT-R. Base-level activation `B_i = ln(Σ_j t_j^(−d))` gives power-law decay over every past retrieval, with `d ≈ 0.5`. Each retrieval adds a term to the sum, so strength is frequency plus recency. Context relevance acts as a spreading-activation analog, with noise and a retrieval threshold below which a memory is functionally forgotten. It matters to us because ACT-R's spreading activation is the cognitive-science ancestor of our graph mechanism, with parameters fit to decades of human data.

### Also relevant

- [MemoryBank](https://arxiv.org/abs/2305.10250) (AAAI 2024). Ebbinghaus curve `R = e^(−t/S)`. Each recall increments `S` and resets `t`, and decay can delete outright. Relevance is retrieval-side (FAISS cosine), not fused into the score.
- [Memoria](https://arxiv.org/abs/2310.03052) (ICML 2024). Hebbian engram strength between memory units. Rare prior work with connections between memories, but no clock-time decay, and it lives inside the model rather than in an agent store.
- [FadeMem](https://arxiv.org/pdf/2601.18642) (2026). Differential decay modulated by relevance and access frequency.

## 2. The OG paper: Generative Agents (Park et al., 2023)

[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). The Smallville paper, and the basis of our own [Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) (see [GAPS.md](GAPS.md)).

**Its memory model.** An append-only stream of natural-language records with creation and last-accessed timestamps. Retrieval score is recency plus importance plus relevance, equal weights, each min-max normalized. Recency is exponential decay over sandbox time (factor 0.995). Importance is an LLM rating of 1 to 10 at creation, static forever. Relevance is cosine similarity. **Reflection**: when the summed importance of recent events passes 150, the agent generates "salient questions", retrieves against them, and writes insight records that cite their evidence ("because of 1, 8, 15"). Reflections recurse into trees.

**What it lacks** (our checklist). Nothing is ever forgotten, recency only reranks. No reinforcement: importance never updates and retrieval does not consolidate. A flat stream with no relations beyond the implicit citations. Hand-set weights. Decay tied to the clock rather than usage.

**What we take.** The normalized weighted-sum skeleton, with recency swapped for real decay and importance swapped for dynamic strength seeded by the write-time rating. Reflection as consolidation, whose citations are literally edges: in a graph they become first-class insight-to-evidence links, the structure the paper never materialized. Recursive reflection maps to graph hierarchy, with strength propagating along edges (spreading activation, per ACT-R). And its evaluation framing of believability interviews plus ablations, reusable as ablate-decay, ablate-strength, ablate-edges.

## 3. Graph memory systems

| System | Graph mechanism | Decay/strength? |
|---|---|---|
| [HippoRAG / HippoRAG 2](https://github.com/osu-nlp-group/hipporag) (NeurIPS'24, [ICML'25](https://arxiv.org/abs/2502.14802), where HippoRAG 2's paper title is "From RAG to Memory") | LLM extracts entities and triples into an open knowledge graph. Retrieval runs Personalized PageRank seeded from query entities. Up to 20% over SOTA retrieval on multi-hop QA (best case, average gains smaller). | None. Append-only. |
| [Zep / Graphiti](https://arxiv.org/abs/2501.13956) ([code](https://github.com/getzep/graphiti)) | Hierarchical temporal knowledge graph (episode, entity, community). Bi-temporal edges with validity intervals, so contradicted facts are invalidated, not deleted. Authors (Zep is a company) report up to +18.5% on LongMemEval with 90% lower latency. | Discrete truth-maintenance only. |
| [A-Mem](https://arxiv.org/abs/2502.12110) (NeurIPS'25) | Zettelkasten notes. An LLM links new notes to old ones and evolves old notes as related ones arrive. Multi-hop LoCoMo gains of about 2x with small models, more modest (roughly 17%) with GPT-4o, at about 1,200 context tokens vs about 16,900 for baselines. | None. |
| [Mem0 / Mem0-g](https://arxiv.org/pdf/2504.19413) | Fact extraction with ADD/UPDATE/DELETE. The graph variant adds an entity-relation store with conflict detection. LoCoMo LLM-judge 66.9 to 68.4 with the graph. | None. Deletion is contradiction-driven. |
| [GraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/), then LazyGraphRAG and LightRAG | Corpus knowledge graph plus community summaries for "global" questions. | Static-corpus RAG, not agent memory. |

### Has anyone combined decay, reinforcement, and graph? Essentially no. Three near-misses:

1. [Selective Forgetting](https://arxiv.org/abs/2608.28978) ("Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents", verified against the paper 2026-09-18). Knowledge-graph memory with a forgetting score (recency, access frequency, degree centrality, age) used only to prune, never in retrieval. Two findings that cut both ways. Pruning 9.8% of a 27,021-node graph left token F1 unchanged (+0.001, CI [-0.015, +0.016]) with judged correctness down at most 3.8 points, so selective deletion was roughly harmless. Meanwhile the graph itself underperformed flat vector retrieval on LongMemEval (0.417 vs 0.468 token F1, a significant -0.050, CI [-0.085, -0.016]), with a large drop on questions about prior assistant turns (0.911 to 0.607) that looks like an extraction artifact rather than a fundamental graph property. Read carefully: forgetting wasn't the problem, the graph retrieval was. Mechanics, for the record: GPT-4o-mini decomposes each turn into a fixed 9-type entity ontology (Person, Location, Event, Preference, Goal, ...) with typed edges, and the graph replaces the raw text. Retrieval is cosine similarity over node descriptors (0.75 threshold), then an unweighted two-hop breadth-first expansion from the top 5 nodes, capped at 15. No decay, strength, or edge weighting anywhere in ranking. Their own error analysis blames extraction lossiness (structured abstraction "discarding information required for precise or verbatim recall") and a conflict policy that kept stale values. Prior art, a warning that graph structure alone can hurt, and the reason our flat baseline stays first-class.
2. [Not All Memories Age the Same: Autodiscovery of Adaptive Decay in Knowledge Graphs](https://arxiv.org/abs/2604.26970) (Karhade 2026). Learns decay rates for predicate clusters, contexts, and entities via survival analysis. Found uniform decay 18x worse than no temporal weighting at all (NDCG@5 0.015 vs 0.274), so decay rates must vary. Knowledge-graph freshness work, not agent memory, and no recall dynamics.
3. [PowerMem](https://github.com/oceanbase/powermem) (OceanBase). Engineering product with Ebbinghaus decay plus hybrid vector and graph retrieval, but decay is a re-ranking layer beside the graph. No decay-aware traversal, no paper.

The open slot: decay and reinforcement fused into graph retrieval itself (strength-weighted PageRank, spreading activation that fades), evaluated for human-likeness and on standard benchmarks.

## 4. Benchmarks

Sanity check, not the goal. Human-like memory must not wreck utility.

- [LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025, [code](https://github.com/xiaowu0162/longmemeval)). Primary. 500 questions over long chat histories: extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. Commercial assistants drop about 30%.
- [LoCoMo](https://www.emergentmind.com/topics/locomo) (ACL 2024). About 1,540 QA over multi-session conversations spanning 6 to 12 simulated months, 20.8% temporal. Cross-paper scoring is inconsistent (the Mem0/Zep dispute), so report F1 and LLM-judge with a pinned judge.
- [MemBench](https://arxiv.org/abs/2506.21605) (ACL 2025 Findings). Factual and reflective memory. Its capacity axis (degradation as the store grows) is the only mainstream metric that even touches forgetting.
- Secondary: [DMR](https://arxiv.org/abs/2501.13956) (near-saturated, cite it, don't build on it), [MultiChallenge](https://scale.com/research/multichallenge) (in-context multi-turn), [Test of Time](https://arxiv.org/abs/2406.09170) (synthetic temporal logic, could seed our own probes).
- Memory-specific 2025-26: [AMA-Bench](https://arxiv.org/html/2602.22769), [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (ICLR 2026), [WorldLines](https://arxiv.org/pdf/2606.18847), [Vending-Bench](https://arxiv.org/abs/2502.15840).

## 5. Simulation environments

- [TerraLingua](https://github.com/cognizant-ai-lab/terralingua) (Cognizant AI Lab 2026, Apache 2.0 · [page](https://www.cognizant.com/us/en/ai-lab/terralingua) · [blog](https://www.cognizant.com/us/en/ai-lab/blog/when-ai-agents-build-societies-terralingua)). Persistent 2D grid world. Agents forage, reproduce, die, and author persistent text artifacts that outlive them. An "AI Anthropologist" LLM analyzes the logs. No pluggable memory interface, so we would modify the agent loop (small young Python codebase, about 72 stars). Generational turnover and recurring artifacts make decay and relations genuinely matter.
- [Smallville](https://github.com/joonspk-research/generative_agents). The memory stream is a direct drop-in slot. Aging and costly to run. [AI Town](https://github.com/a16z-infra/ai-town) (TypeScript/Convex re-implementation, MIT, maintenance mode) is the easiest hack target, with a shallow world.
- [Concordia](https://github.com/google-deepmind/concordia) (DeepMind). Component-based agents with swappable associative memory, and an LLM game master generates the world. Best-maintained option. Text-only, narrative time.
- Maya (ours). Deterministic engine, swappable memory stream, decision inspector, audit trail, ablation flags. See [GAPS.md](GAPS.md).
- Set aside: Project Sid (code unreleased), Voyager/MineDojo (single-agent, procedural skills), AgentSociety and OASIS (scale over per-agent depth), Sotopia (episodes too short for decay), Crafter and Craftax (rule-generated content, no recurring named entities).
