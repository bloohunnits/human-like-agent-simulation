# Gaps and Open Decisions

What the papers leave open, what our Maya project fills, and what we still have to decide. Companion to [RESEARCH.md](RESEARCH.md).

## 1. What the papers leave open

[Hou et al.](https://arxiv.org/abs/2404.00573) gives us the decay, strength, and relevance core, but:

1. **No relations between memories.** Every memory fades alone. Human recall is associative. This is the project's main addition.
2. **Chat-scale evaluation only.** Short companion-chat studies with a handful of users. Nothing about an agent lifetime: thousands of memories, months of simulated time, generational turnover. That is what the simulation is for.
3. **No behavioral picture of forgetting.** The papers measure recall probability, not whether forgetting reads as human (fading acquaintances, kept core relationships, gist outliving detail). We need behavioral measures, not just retrieval ones.
4. **Hand-set, uniform parameters.** The knowledge-graph freshness literature found uniform decay can be worse than no decay (18x on their metric). Decay rates likely need to vary by memory type or entity. Nobody has tuned this for agents.
5. **No interference or distortion.** Human memory errors are systematic. Similar memories blur, gist survives detail. The papers only model recall vs no recall. Stretch goal: distortion as a human-likeness feature.
6. **Consolidation and decay never meet.** Generative Agents' reflection abstracts upward but the originals never fade. Hou et al. strengthen but never abstract. Humans do both at once: details decay while gist consolidates. Insight nodes strengthening as their evidence fades is unexplored.
7. **No forgetting-quality metric exists anywhere.** No benchmark asks whether the right things faded. We will build probes: superseded vs current facts, reinforced vs one-off retention, multi-hop recall across faded memories that only association can rescue.
8. **Evaluation hygiene.** The proposal's validation set is Hou et al.'s appendix tests, EmotionBench, Shachi's benchmark list, and TerraLingua's AI Anthropologist, plus our baselines and ablations. Conversational memory benchmarks (LoCoMo, LongMemEval) are optional external checks only. If we use LoCoMo, note its scoring is inconsistent across papers (the Mem0/Zep dispute): report F1 and LLM-judge with a pinned judge.

## 2. What Maya fills

[Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) is the Generative Agents paper plus our own engineering. Both its architecture and its gap analysis feed this project.

### Worth reusing

- **Deterministic floor, validated LLM layer.** A deterministic engine owns state, and the LLM proposes cognition that is validated before it touches anything. Here that means decay math, strength updates, and traversal are deterministic and seeded, while the LLM only rates initial importance, proposes links, and writes reflections. Hallucinations cannot corrupt the store, and runs replay.
- **Citation-validated reflection is our evidence edges, already prototyped.** Maya checks that reflections cite real memories (AI Town does not bother). Those citations are the insight-to-evidence edges. We promote them to first-class links and let recall spread across them.
- **Everything behind a config flag.** Maya's ablation-readiness is the template. Decay, reinforcement, spreading, and each edge type get flags from day one, so the ablation study is a config sweep, not a refactor.
- **Audit trail and near-miss memories.** Maya's decision inspector shows what an agent almost recalled. For a memory project that is gold: "lost by 0.1 because it faded", "rescued by an edge". Build the explainability in from the start.
- **Per-pair interaction edges.** AI Town's cheap participated-together table (last time we talked) was flagged in Maya's analysis as a high-believability win. In our graph it is just an edge type. Free.
- **Untrusted-memory prompt isolation.** Mark retrieved memories as untrusted historical content, separated from instructions (injection hardening AI Town shipped). Cheap and credible.
- **Embeddings cache keyed by text hash.** Dedupes embedding spend across agents and restarts.

### Mistakes not to repeat

- **Eval that never touches the LLM.** Maya's 8 metrics run with the LLM off, so the believability claim goes unmeasured. Memory is our claim. Measure retrieval precision, grounding, and behavior with the full stack on.
- **No retrieval-precision metric** existed in Maya. Core metric here from day one.
- **Single-seed science.** Every Maya claim lives at seed 42. Seed sweeps, variance, confidence intervals.
- **No LLM record and replay.** Maya logs every call but cannot replay from log. A replay shim makes full runs reproducible, which is the strongest answer to "LLMs aren't reproducible", and it is far easier built in early.
- **Time you can't watch.** Maya runs a simulated day per 72 real seconds and its pause button doesn't pause. Decay is a function of simulated time. Without real time controls (pause, speed, ideally scrub), nobody can see memory working.

### Maya as a fallback testbed

The proposal fixes TerraLingua as the sandbox. Maya stays on the bench as a fallback: our code, a swappable memory stream, fast simulated months, a recurring cast, and an inspector that can display memory effects nothing else can. The flow can also go backward later: this engine fills Maya's promised-but-missing loops (trait evolution nudged by reflection, relationship decay).

## 3. Open decisions

1. **Environment: decided by the proposal.** TerraLingua is the sandbox (LLM-authored artifacts, birth and death, real simulated time). Our integration replaces its built-in scratchpad memory in the agent loop. Fallbacks if that spikes badly: Concordia (swappable memory components), AI Town / Smallville (drop-in memory stream), or our own Maya. First task: a hands-on spike in the TerraLingua codebase to cost the integration.
2. **Formulation: decided by the proposal.** ACT-R (Honda et al.) is the base architecture, Hou et al. is the published comparison point, and both kernels run under the same relations graph behind one interface. The reset behavior is why: one mention of an old memory shouldn't rejuvenate it wholesale, and ACT-R's per-retrieval fading traces match the human pattern (brief priming bump, then back near baseline). The reminiscence probe that separates them behaviorally is in [DESIGN.md](DESIGN.md).
3. **Edge types.** Minimum set: shared entity, temporal succession, evidence-of (reflection citations), same interaction or co-occurrence. Wanted beyond that: affective edges (spider wired to fear) and cause-consequence. Both are LLM-judged and noisier. Phase 2?
4. **Where strength lives.** Nodes only, or edges too? Associations strengthening with co-recall and fading unused (Hebbian, per Memoria) is the more human story and the bigger novelty. Also more parameters to tune.
5. **Do edges outlive their nodes?** The keep-the-fear, forget-the-bite mechanism. When an episodic node decays below recall, does the association it forged persist as an edge into a semantic or affective layer? The most human behavior on the table (an agent that feels something it can no longer explain) and the trickiest to keep from becoming noise.
6. **Does spreading carry state, or just accessibility?** Retrieval-only spreading changes what lands in the context window. Valence-carrying spreading changes the agent's emotional state and behavior (avoidance, comfort-seeking). The second is the human-likeness bet, and it needs the sim, not a benchmark, to evaluate.
7. **Heterogeneous decay rates.** Uniform decay is provably risky. By memory type? Per entity? Learned from recall logs? Start hand-set by type, revisit.
8. **Does the graph pay for itself?** The one prior graph-plus-forgetting system lost to flat vectors, and on careful reading the loss came from its graph retrieval, not its forgetting (which was pruning only, and roughly harmless). So the warning is specifically that graph structure alone can hurt. Our bet is that traversal shaped by strength and decay changes the outcome. The flat baseline stays a first-class competitor and we report honestly if it wins.
9. **Read the ACT-R paper: done (2026-09-18).** Full text read via the ACM open-access page. Equations, parameters (d = 0.5, σ = 1.2, optimal w = 11.0), embedding choice (all-MiniLM-L6-v2, same as our default), and the key quote are in [RESEARCH.md](RESEARCH.md). Its own future-work section calls for graph-based relationships among memory chunks, which is this project. Remaining from it: their noise is Gaussian rather than ACT-R's classic logistic, pick one and document it.
