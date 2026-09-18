# Project Proposal: Human-Like Long-Term Memory for LLM Agents

Repo: https://github.com/bloohunnits/human-like-agent-simulation (single team repo, commits tracked here)

Background docs in this repo: [README](README.md) for the idea, [docs/DESIGN.md](docs/DESIGN.md) for mechanics, [docs/RESEARCH.md](docs/RESEARCH.md) for the full survey.

## 1. Problem

LLM agents don't forget well. Most memory systems today do one of two things. They keep everything in a vector store, which gets noisy and expensive as the store grows. Or they add a forgetting curve so memories fade with time unless they get recalled, which is more human but has a flaw: every memory sits on its own clock. A memory only survives if something directly recalls it.

Human memory doesn't work like that. You still remember your childhood best friend's kitchen, not because you rehearse kitchens, but because that kitchen is wired into things you still think about. Memories survive by being connected. Current decay models have nowhere to store a connection, so over a long horizon they forget the wrong things.

Our project: take an existing human-like memory model (decay, strength, relevance) and make connections between memories a factor in what gets recalled and what survives. Memories become nodes in a graph. Recall spreads along edges, and so does reinforcement, so the parts of an agent's life that keep getting touched keep each other alive. Then we put this memory into agents in a long-running simulation and measure whether they remember and forget more like people.

Why it matters: believable characters in simulations and games, and long-lived assistants that don't drown in their own history. Why it's novel: as far as we can find (section 2), nobody has published a system where decay and reinforcement operate inside graph retrieval. The pieces exist separately. The combination doesn't.

This is a research project with hypotheses, not a software project that happens to use papers:

- H1: spreading activation over strength-weighted edges beats flat vector retrieval on associative and temporal recall, where the one prior graph-with-forgetting attempt lost.
- H2: connection-based survival keeps the right memories. Under equal memory budgets, our forgetting matches human patterns (woven-in survives, isolated trivia fades) better than time-only decay.
- H3: agents running this memory read as more believable over long simulated horizons than the same agents with flat or decay-only memory.

Any of these can come out false. H1 in particular has a published negative result to overcome, which is part of what makes this a real question.

## 2. Literature review

The line of work starts with Generative Agents ([Park et al. 2023](https://arxiv.org/abs/2304.03442)): agents in a sandbox town with a memory stream scored by recency, importance, and relevance, plus periodic reflection. Influential, but nothing ever fades, importance never updates, and memory is a flat list.

Human-like decay models came next. MemoryBank ([Zhong et al. 2024](https://arxiv.org/abs/2305.10250)) applies the Ebbinghaus forgetting curve with a strength counter. Hou et al. ([2024](https://arxiv.org/abs/2404.00573)) is our source paper: one closed-form recall probability combining relevance, time since last recall, and a consolidation gradient that grows with each recall, with spaced recalls strengthening more than massed ones. Honda et al. ([HAI 2025](https://dl.acm.org/doi/10.1145/3765766.3765803)) build the same three factors from the ACT-R cognitive architecture, which also gives us spreading activation with parameters fit to human data. In all three, memories are independent. No structure between them.

Graph memory is its own thread. HippoRAG ([Gutiérrez et al. 2024](https://arxiv.org/abs/2405.14831), [HippoRAG 2, 2025](https://arxiv.org/abs/2502.14802)) extracts entities into a knowledge graph and retrieves with Personalized PageRank, beating standard RAG on multi-hop QA by around 20%. Zep/Graphiti ([Rasmussen et al. 2025](https://arxiv.org/abs/2501.13956)) keeps a temporal knowledge graph where contradicted facts get invalidated. A-Mem ([Xu et al. 2025](https://arxiv.org/abs/2502.12110)) links notes Zettelkasten-style. Mem0 ([Chhikara et al. 2025](https://arxiv.org/pdf/2504.19413)) adds a graph variant to fact extraction. None of these has decay or use-based strength. Their graphs only grow.

**The research gap.** No published system fuses decay and recall reinforcement into graph retrieval itself. The three closest attempts each miss it. A 2026 selective-forgetting system ([arXiv:2608.28978](https://arxiv.org/html/2608.28978)) computes a forgetting score over a knowledge graph but only uses it to prune nodes, and its graph actually lost to flat vector retrieval on LongMemEval (0.417 vs 0.468 F1). A knowledge-graph freshness paper ([arXiv:2604.26970](https://arxiv.org/html/2604.26970v1)) learns per-relation decay rates but isn't an agent memory and has no recall dynamics. It also found uniform decay can be 18x worse than no decay, which warns us against naive parameter choices. PowerMem ([OceanBase](https://github.com/oceanbase/powermem)) is an engineering product that bolts Ebbinghaus decay next to a graph channel, with no decay-aware traversal and no paper. On the evaluation side there's a second gap: no benchmark measures forgetting quality, meaning whether the right things faded. LongMemEval's knowledge-update questions and MemBench's capacity axis come closest.

Benchmarks we'll use: LoCoMo ([Maharana et al. 2024](https://arxiv.org/abs/2402.17753)), LongMemEval ([Wu et al. 2025](https://arxiv.org/abs/2410.10813)), MemBench ([2025](https://arxiv.org/abs/2506.21605)). Simulation platforms: TerraLingua ([Cognizant AI Lab 2026](https://github.com/cognizant-ai-lab/terralingua)), Concordia ([DeepMind](https://github.com/google-deepmind/concordia)), and Maya, our own Generative Agents implementation from earlier coursework.

## 3. Goals and scope

**Research scope.** We build and evaluate the memory model. We do not train or fine-tune any base model (LLMs and embedding models stay frozen), and we do not build a simulation from scratch (we integrate into existing ones). Emotion modeling beyond simple valence tags is out of scope. So is any claim about modeling clinical conditions.

**Datasets.**
- LoCoMo: 10 very long multi-session conversations, about 600 turns and up to 26k tokens each, with roughly 1,540 QA pairs (single-hop, multi-hop, temporal, open-domain). Public.
- LongMemEval (S): 500 questions over chat histories of about 115k tokens each, covering extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. Public.
- MemBench for capacity degradation as the store grows. Public.
- Self-generated simulation logs: agent runs of 90+ simulated days in Maya and (stretch) TerraLingua. A 3-agent, 90-day Maya run produces on the order of tens of thousands of memory events. These logs, plus the probe questions and answers, become a dataset we release with the repo.
- Our own forgetting-quality probe set, built from templates over the sim logs (details in section 6). Nothing like it exists, so we have to make it. We expect a few hundred probes.

**Models.**
- Memory engine: ours, described in section 4. Not a neural model. A graph with decay dynamics, which is the point (the contribution is the memory architecture, not a trained model).
- LLM for agents and for write-time tasks (importance rating, link proposal, reflection): a small cheap model, GPT-4o-mini or Claude Haiku class via API, and Llama 3.1 8B via Ollama for local runs. The engine is model-agnostic and we'll report which model each result used.
- Embeddings: sentence-transformers all-MiniLM-L6-v2 (384-dim, local, free) as default, with OpenAI text-embedding-3-small as a check.
- Baselines: full-context (no memory system), flat vector store (no decay, no graph), our reproduction of Hou et al. (decay only, no graph), and published numbers for Mem0, A-Mem, and Zep where comparable. Plus our own ablations: edges without spreading reinforcement, spreading without edge decay, and so on.

**Compute, rough.** No training, so no GPU cluster. Everything runs on our laptops plus API calls, with Ollama on a single consumer GPU / Apple Silicon for local model runs. Token math, roughly: ingesting LoCoMo across all system configs is a few million tokens. LongMemEval-S full-context baseline is the expensive one, about 57M tokens for one pass (500 questions times 115k), which we run once. Memory-system runs are far cheaper since they retrieve small contexts. Across all configs, seeds, and reruns we estimate 200M to 500M tokens over the semester, which at mini-model prices is very roughly $100 to $400 in API spend, less if we push more onto local Llama. Sim runs are local and mostly replayed from logs after the first pass.

## 4. Approach

Algorithms, all detailed in [docs/DESIGN.md](docs/DESIGN.md):

- Per-memory dynamics from Hou et al.: recall probability p(r, t, g) where r is cue similarity, t is time since last recall, and g is a strength that grows with each recall (more for spaced recalls).
- Storage is a property graph, not a vector store. Memory nodes, entity hub nodes (a person is a node, not a string), and typed edges (mentions, co-occurred, followed, cause-candidate, evidence-of) created at write time. Each edge has its own weight and clock.
- Retrieval: seed activation by embedding similarity, spread it 1 to 2 hops along strength-weighted edges with damping, then score each memory with the same formula but a widened cue: r-hat = max(direct similarity, received activation). Strength-weighted Personalized PageRank (HippoRAG-style) is a variant we'll test.
- Updates: recalled nodes get the paper's strength bump, traversed edges get an edge version of it, and neighbors above an activation threshold get a partial refresh. That last rule is the survival mechanism and its threshold is the key parameter to tune.
- Reflection (from Generative Agents, hardened in Maya): insight nodes with validated evidence edges.
- The whole thing reduces exactly to Hou et al. when edges are off, so every ablation is a config flag.

Implementation: Python 3.12 library. NetworkX (or SQLite tables) for the graph, numpy/FAISS for embeddings, pytest for tests. Deterministic and seeded everywhere the LLM isn't, and LLM calls are logged with a replay mode so full experiment runs are reproducible. Sim integration: Maya exposes its memory stream as a swappable port (Java backend, we bridge over its REST layer), TerraLingua is Python and we patch its agent loop directly.

## 5. Repo

Public single project repo: https://github.com/bloohunnits/human-like-agent-simulation. All members commit here. Currently holds the research docs; code lands here too as it's written (the docs-only note in the README predates this proposal and we'll drop it).

## 6. Validation

Unit tests:
- Decay math property tests: strength monotonically increases with recalls, spaced recalls beat massed recalls, recall probability falls with t and rises with g.
- Reduction test: edges off must reproduce Hou et al. behavior exactly.
- Determinism test: same seed and same replay log gives an identical run, byte for byte.
- Graph invariants: no dangling edges, edge caps hold, LLM-proposed links that cite nonexistent nodes get rejected.

Performance metrics:
- Benchmarks: LongMemEval accuracy per question type (temporal and knowledge-update subsets matter most to us), LoCoMo F1 and LLM-judge with the judge model pinned (scores in this literature are inconsistent across papers, so we report both and state the judge), MemBench capacity curves.
- Forgetting quality (ours): probes where the correct behavior is retrieving the current fact over a superseded one, retaining a reinforced memory over a same-age one-off, and answering a question whose evidence is a faded memory reachable only through an edge. Scored automatically against the sim's ground-truth event log.
- Sim believability: Generative Agents-style interview categories (self-knowledge, memory, plans, relationships) with grounding checks against the audit log, run at multiple seeds with confidence intervals. Ablations (decay off, reinforcement off, edges off) on all of the above.

## 7. Deliverables

A spectrum, so partial results still count:

1. Floor: the memory library with the Hou et al. reproduction, unit tests, and baseline numbers on LoCoMo and LongMemEval. Useful on its own since no clean open reproduction of that paper exists.
2. Core: the graph layer plus the ablation study on benchmarks. This answers H1 either way, and a negative answer is still a finding given the prior negative result.
3. Target: forgetting-quality probe set plus Maya integration with believability results. Answers H2 and most of H3.
4. Stretch: TerraLingua integration for generational runs, edge-outlives-node experiments, affective edges.

Each level ships with its writeup section, so the final report exists at whatever level we reach.

## 8. Timeline and milestones

- Sep 22 to Oct 3: lock formulation. Get the paywalled ACT-R paper (library or authors). Repo restructure for code. Milestone: design doc frozen, engine skeleton with failing tests.
- Oct 6 to Oct 17: core engine. Hou reproduction passing property tests, embedding pipeline, record/replay. Milestone: deliverable 1 done, first baseline numbers.
- Oct 20 to Oct 31: graph storage and write pipeline, entity extraction, deterministic edges. Milestone: graphs building from LoCoMo transcripts.
- Nov 3 to Nov 14: spreading retrieval, update rules, ablation harness. Milestone: deliverable 2, H1 answered on benchmarks. Midpoint checkpoint.
- Nov 17 to Nov 28: forgetting-quality probes, Maya integration, first 90-day runs. Milestone: probe set released, H2 numbers.
- Dec 1 to Dec 11: believability interviews, seed sweeps, TerraLingua if time allows. Milestone: deliverable 3.
- Dec 12 to finals: analysis and final report. Milestone: report and demo.

## Why can't a code agent just do this project?

A code agent could write a lot of this code. The decay formula, the graph store, the test harness, the benchmark plumbing. That part is honest to admit. What it can't do is the part that makes this research:

- Decide what "human-like forgetting" means operationally. Turning "you remember the kitchen but not last Tuesday's lunch" into scorable probes is a judgment call about human memory, made by reading cognitive science and arguing about it, not by pattern-matching code.
- Tune the survival parameter by watching behavior. The partial-refresh threshold sits between "nothing ever fades" and "we reproduced the paper we started from." Where it should sit is a call you make by reading agent transcripts and noticing that something feels off, weeks before any metric says so.
- Judge believability. The interview evaluation bottoms out in humans reading agent answers and deciding whether they sound like a person with a past. There's no oracle to delegate that to, and using an LLM as the only judge of LLM human-likeness is circular.
- Interpret results against a messy literature. LoCoMo scores disagree across papers and the one prior graph-plus-forgetting attempt lost to flat retrieval. Deciding whether our H1 result is real, an artifact of scoring, or an artifact of parameters requires skepticism about our own numbers.
- Know when to stop and publish a negative result. An agent optimizes toward making the thing work. A researcher has to notice when the honest finding is that it doesn't, and write that up instead.
- Handle the humans: getting a paywalled paper from its authors, deciding the ethical framing around trauma-like dynamics in simulated characters, and coordinating a team's worth of decisions in one repo.

We expect to use code agents for implementation grunt work and we'll say so in the report. The hypotheses, the probe design, the parameter judgment, and the interpretation stay on us.
