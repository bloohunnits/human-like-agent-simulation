# Gaps and Open Decisions

What the papers leave open, what our Maya project fills, and what we still have to decide. Companion to [RESEARCH.md](RESEARCH.md).

## 1. What the papers leave open

[Hou et al.](https://arxiv.org/abs/2404.00573) gives us the decay/strength/relevance core, but:

1. **No relations between memories.** Every memory fades alone; human recall is associative. This is the project's main addition.
2. **Chat-scale evaluation only.** Short companion-chat studies with a handful of users — nothing about an agent *lifetime*: thousands of memories, months of simulated time, generational turnover. That's what the simulation is for.
3. **No behavioral picture of forgetting.** The papers measure recall probability, not whether forgetting *reads* as human — fading acquaintances, kept core relationships, gist outliving detail. We need behavioral measures, not just retrieval ones.
4. **Hand-set, uniform parameters.** The KG-freshness literature found uniform decay can be *worse than no decay* (18× on their metric). Decay rates likely need to vary by memory type or entity; nobody has tuned this for agents.
5. **No interference or distortion.** Human memory errors are systematic — similar memories blur, gist survives detail. The papers only model recall vs. no-recall. Stretch goal: distortion as a human-likeness feature.
6. **Consolidation and decay never meet.** Generative Agents' reflection abstracts upward but the originals never fade; Hou et al. strengthen but never abstract. Humans do both at once — details decay while gist consolidates. Insight nodes strengthening as their evidence fades is unexplored.
7. **No forgetting-quality metric exists anywhere.** No benchmark asks whether the *right* things faded. We'll build probes: superseded-vs-current facts, reinforced vs. one-off retention, multi-hop recall across faded memories that only association can rescue.
8. **Evaluation hygiene.** LoCoMo scoring is inconsistent across papers (the Mem0/Zep dispute). Report F1 *and* LLM-judge with a pinned judge; LongMemEval primary.

## 2. What Maya fills

[Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) is the Generative Agents paper plus our own engineering, and both its architecture and its gap analysis feed this project.

### Worth reusing

- **Deterministic floor + validated LLM layer.** A deterministic engine owns state; the LLM proposes cognition that's validated before it touches anything. Here: decay math, strength updates, and traversal are deterministic and seeded; the LLM only rates initial importance, proposes links, and writes reflections. Hallucinations can't corrupt the store, and runs replay.
- **Citation-validated reflection = our evidence edges, already prototyped.** Maya checks that reflections cite real memories (AI Town doesn't bother). Those citations *are* the insight→evidence edges — we promote them to first-class links and let recall spread across them.
- **Everything behind a config flag.** Maya's ablation-readiness is the template: decay, reinforcement, spreading, and each edge type get flags from day one, so the ablation study is a config sweep, not a refactor.
- **Audit trail + near-miss memories.** Maya's decision inspector shows what an agent almost recalled. For a memory project that's gold — "lost by 0.1 because it faded," "rescued by an edge" — build the explainability in from the start.
- **Per-pair interaction edges.** AI Town's cheap `participatedTogether` table (last-time-we-talked) was flagged in Maya's analysis as a high-believability win. In our graph it's just an edge type — free.
- **Untrusted-memory prompt isolation.** Mark retrieved memories as untrusted historical content, separated from instructions (injection hardening AI Town shipped). Cheap, credible.
- **Embeddings cache keyed by text hash** — dedupes embedding spend across agents and restarts.

### Mistakes not to repeat

- **Eval that never touches the LLM.** Maya's 8 metrics run with the LLM off, so the believability claim goes unmeasured. Memory *is* our claim — measure retrieval precision, grounding, and behavior with the full stack on.
- **No retrieval-precision metric** existed in Maya. Core metric here from day one.
- **Single-seed science.** Every Maya claim lives at seed 42. Seed sweeps, variance, confidence intervals.
- **No LLM record/replay.** Maya logs every call but can't replay from log. A replay shim makes full runs reproducible — the strongest answer to "LLMs aren't reproducible" — and is far easier built in early.
- **Time you can't watch.** Maya runs a simulated day per 72 real seconds and its pause button doesn't pause. Decay is a function of simulated time; without real time controls (pause, speed, ideally scrub), nobody can *see* memory working.

### Maya as a testbed

Maya is also a candidate environment: our code, a swappable memory stream, fast simulated months, a recurring cast, and an inspector that can display memory effects nothing else can. Likely play: **develop against Maya, deploy into TerraLingua for the headline experiments.** It flows back too — this engine fills Maya's promised-but-missing loops (trait evolution nudged by reflection, relationship decay).

## 3. Open decisions

1. **Environment.** TerraLingua is the most interesting (LLM-authored artifacts, birth/death) but has no memory interface; Maya is instrumented and ours; Concordia is the maintained middle. Lean: Maya first, TerraLingua for the headline runs. Needs a hands-on spike in the TerraLingua codebase to cost the integration.
2. **Formulation.** Hou et al.'s single formula vs. ACT-R activation vs. MemoryBank's Ebbinghaus. Hou is the paper we set out to improve; ACT-R brings native spreading activation and human-fitted parameters. Likely hybrid: Hou-style per-node dynamics + ACT-R-style spreading along edges.
3. **Edge types.** Minimum set: shared-entity, temporal-succession, evidence-of (reflection citations), same-interaction/co-occurrence. Wanted beyond that: affective edges (spider→fear) and cause/consequence — both LLM-judged and noisier. Phase 2?
4. **Where strength lives.** Nodes only, or edges too (associations strengthen with co-recall and fade unused — Hebbian, per Memoria)? Edge-strength is the more human story and the bigger novelty, and more parameters to tune.
5. **Do edges outlive their nodes?** The "kept the fear, forgot the bite" mechanism: when an episodic node decays below recall, does the association it forged persist (as an edge into a semantic/affective layer)? The most human behavior on the table — an agent that feels something it can no longer explain — and the trickiest to keep from becoming noise.
6. **Does spreading carry state, or just accessibility?** Retrieval-only spreading changes what lands in the context window. Valence-carrying spreading changes the agent's emotional state and behavior (avoidance, comfort-seeking). The second is the human-likeness bet; needs the sim, not a benchmark, to evaluate.
7. **Heterogeneous decay rates.** Uniform decay is provably risky. By memory type? Per-entity? Learned from recall logs? Start hand-set by type, revisit.
8. **Does the graph pay for itself?** The one prior graph+forgetting system lost to flat vectors. Our bet is strength-weighted spreading changes that — but the flat baseline stays a first-class competitor and we report honestly if it wins.
9. **Get the ACT-R paper.** Paywalled on ACM, no arXiv preprint. Library or authors, before locking the formulation.
