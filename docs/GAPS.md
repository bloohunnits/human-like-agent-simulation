# Gaps and Open Decisions

What the source papers don't cover, what we take from our own Maya project to fill some of it, and the decisions still open. Companion to [RESEARCH.md](RESEARCH.md).

## 1. Gaps the papers leave open (beyond "no sandbox")

The human-like memory paper ([Hou et al.](https://arxiv.org/abs/2404.00573)) gives us the decay/strength/relevance core, but:

1. **No relations between memories.** Every memory decays and strengthens alone. Human recall is associative; this is the project's main addition (the memory graph with spreading activation).
2. **Dialogue-agent scale only.** Evaluated on short companion-chat studies with a handful of users. Nothing about behavior over an agent *lifetime* — thousands of memories, months of simulated time, generational turnover. That's what the simulation is for.
3. **No account of what forgetting should *look like* behaviorally.** The papers measure recall probability, not whether an agent's forgetting reads as human (fading acquaintances, kept core relationships, gist surviving detail). We need behavioral measures, not just retrieval measures.
4. **Parameters are hand-set and uniform.** The KG-freshness literature found uniform decay can be *worse than no decay* (18× on their metric) — decay rates likely need to vary by memory type (episodic vs. semantic vs. emotional, or per-entity). Nobody has tuned this for agents.
5. **No interference or confusion modeling.** Human memory errors are systematic (similar memories blur, gist outlives detail). The papers only model presence/absence of recall. Optional stretch: memory *distortion* as a feature of human-likeness.
6. **Reflection/consolidation is disconnected from decay.** Generative Agents' reflection compresses memories upward but never lets the originals fade; Hou et al. consolidate strength but never abstract. Human memory does both at once: details decay, gist consolidates. Fusing them (insight nodes strengthen as evidence nodes fade) is unexplored.
7. **No forgetting-quality metric exists anywhere.** No benchmark measures whether the *right* things faded. We'll have to build probes: superseded-vs-current fact retrieval, retention of reinforced vs. one-off memories, multi-hop recall across faded memories that only association can rescue.
8. **Evaluation hygiene.** LoCoMo scores are inconsistent across papers (the Mem0/Zep dispute). Report F1 *and* LLM-judge with a pinned judge model; prefer LongMemEval as primary.

## 2. What we take from Maya

Maya (our Generative Agents implementation, `ai-capstone/projects/maya`) is the OG paper plus custom engineering, and both its architecture and its own gap analysis feed this project directly.

### Architecture worth reusing

- **Deterministic floor + LLM meaning layer.** Maya's core pattern: a deterministic engine owns state; the LLM proposes cognition that is *validated* before it touches anything. Apply it here: decay math, strength updates, and graph traversal are deterministic and seeded; the LLM only rates initial importance, proposes links, and writes reflections — all validated. Hallucinations can't corrupt the memory store, and runs are replayable.
- **Citation-validated reflection = our evidence edges, already prototyped.** Maya's reflections must cite real memories and the citations are checked (AI Town skips validation; Maya doesn't). Those citations are exactly the insight→evidence typed edges of our graph. Maya proves the write path works; we make the citations first-class edges and let recall spread across them.
- **Every cognition feature behind a config flag.** Maya's ablation-readiness (flags for reflection, planning, etc.) is the template: decay, reinforcement, spreading activation, and each edge type all get flags from day one, so the ablation study is a config sweep, not a refactor.
- **Audit trail + decision inspector with near-miss memories.** Maya can show *why* an agent acted, including memories that almost got retrieved. For a memory project this is gold: "near-miss" views make decay and association visible ("this memory lost by 0.1 because it faded" / "this one was rescued by an edge"). Build explainability in from the start.
- **Per-pair interaction edges.** Maya's gap analysis flagged AI Town's cheap `participatedTogether` edge table (last-time-we-talked lookup) as a small high-believability win. In our graph that's just an edge type — we get it for free, which is a nice concrete payoff of relations.
- **Untrusted-memory prompt isolation.** Retrieved memories injected into prompts should be marked as untrusted historical content, separated from instructions (prompt-injection hardening AI Town shipped and Maya adopted as a to-do). Cheap, credible safety detail.
- **Embeddings cache keyed by text hash.** Dedupe embedding spend across agents and restarts.

### Mistakes Maya's own gap analysis warns us not to repeat

- **Don't let the eval suite skip the LLM layer.** Maya's 8 metrics all run with the LLM off — the believability claim goes unmeasured. Here, memory *is* the claim: measure retrieval precision, grounding rates, and behavior with the full stack on.
- **No memory-retrieval precision metric** existed in Maya. It's a core metric here from day one.
- **Single-seed science.** Every Maya claim is asserted at seed 42. Seed sweeps, variance, and confidence intervals from the start.
- **LLM record/replay.** Maya logs every LLM call but can't replay from log. A replay-from-log language-model shim makes full runs reproducible — the strongest answer to "LLMs aren't reproducible," and it's much easier to build in early than to retrofit.
- **Time authorship matters.** Maya runs a simulated day per 72 real seconds with a pause button that doesn't pause — memory dynamics would be illegible at that pace. Decay is a function of simulated time, so the sim needs real time controls (pause, speed, ideally scrub) for anyone to *see* memory working.

### Maya as a testbed

Maya is also a candidate environment in its own right: it's our code, the memory stream is a swappable module, simulated months pass quickly, the cast recurs, and the decision inspector can display memory effects no other environment can show. Realistic play: **develop the memory engine against Maya (fast, instrumented, ours), then deploy into TerraLingua for the open-ended-culture experiments.** And the flow goes both ways — this memory engine back-fills Maya's own promised-but-missing loops (trait evolution nudged by reflection, relationship decay).

## 3. Open decisions

1. **Environment: TerraLingua vs. Maya-first vs. Concordia.** TerraLingua is the most interesting (LLM-generated artifacts, birth/death) but has no memory interface; Maya is instrumented and ours; Concordia is the maintained middle. Current lean: Maya first for development, TerraLingua for the headline experiments. Needs a hands-on spike in the TerraLingua codebase to cost the integration.
2. **Decay/strength formulation.** Hou et al.'s single recall-probability formula vs. ACT-R base-level activation vs. MemoryBank's Ebbinghaus. Hou et al. is the paper we set out to improve; ACT-R brings spreading activation natively and human-fitted parameters. Possibly: Hou-style per-node dynamics + ACT-R-style spreading along edges.
3. **Edge types.** Minimum viable set: shared-entity, temporal-succession, evidence-of (reflection citations), same-interaction (per-pair). Causal edges are LLM-judged and noisier — phase 2?
4. **Where strength lives.** Nodes only, or edges too (associations themselves strengthen with co-recall and fade unused — Hebbian, per Memoria)? Edge-strength is the more human story and the bigger novelty; also more parameters to tune.
5. **Heterogeneous decay rates.** Uniform decay is provably risky. By memory type? Per-entity? Learned from recall logs (survival analysis)? Start hand-set-by-type, revisit.
6. **Does the graph pay for itself?** The one prior graph+forgetting system *lost* to flat vector retrieval. Our bet is that strength-weighted spreading changes that — but we hold the flat baseline as a first-class competitor and report honestly if it wins.
7. **Actually obtaining the ACT-R paper.** The HAI 2025 paper is paywalled on ACM with no arXiv preprint found; get access (library / authors) before locking the formulation.
