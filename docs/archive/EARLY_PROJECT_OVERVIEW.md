# Historical snapshot before the v12 publication

This is an archived record. Status, uncommitted-work statements and earlier model directions are superseded by [the current handoff](../../HANDOFF.md). Code paths in prose are relative to the repository root.

# human-like-agent-simulation

Relational memory retrieval for agents in multi-agent simulations. Direct access and associations have separate forgetting and rehearsal processes. We test whether this gives agents more appropriate context and makes their remembering and behavior more believable.

Working repo for our CMSC473/673 project (Andre Atkins, Ben Sadorra, Ryan Shechtman). The original course proposal ([PDF](../../proposal/proposal.pdf) · [LaTeX](../../proposal/proposal.tex)) supplies research context. The repository implementation and current design specify the active experiment.

Current status: the independently aging associative pathway is implemented in the live simulation. See [implementation, final audit and run commands](../ASSOCIATIVE_IMPLEMENTATION.md), [equations and walkthroughs](../INDEPENDENT_GRAPH_MEMORY_DESIGN.md), and [session handoff](../../HANDOFF.md).

The [2026-09-28 proposal logic audit](../PROPOSAL_LOGIC_AUDIT.md) distinguishes verified arithmetic from unresolved model choices. Higher graph scores can displace relevant memories, accumulated hub links can block associative access, and the main entity-to-episode condition is equivalent to a weighted lookup index. These limitations must be evaluated; successful mechanism tests do not establish human-like memory.

The default compares already-decayed direct evidence with separately decayed graph evidence. Only selected memories rehearse; only their winning graph routes can learn. Four runnable conditions separate Hou, target-gated graph support, independent support, and connection learning. Actual prompts, scores, paths and updates are visible at the audit UI's `/audit` page. Parameters are uncalibrated and believability remains to be evaluated.

## Earlier architecture exploration (historical)

The following ACT-R and broader graph notes record earlier directions. They do not describe the active implementation or establish the predicted behavioral effects. Partial neighbor refresh, causal edges, reflection and ACT-R integration remain outside the current experiment.


### The starting point: two human-like memory architectures

Both give every memory the same three ingredients: relevance to the current moment, time since last use, and strength built through use. They disagree on the shape of forgetting.

**Hou et al. (CHI 2024)**, the published comparison point. One recall probability per memory:

```
p = [1 − exp(−r · e^(−t/g))] / [1 − e^(−1)]
```

`r` is similarity between the present and the memory. `e^(−t/g)` is the forgetting curve, steep then flattening. `g` grows with each recall, and spaced recalls grow it more. Each memory keeps one clock and one strength number, and recall resets the clock, so a single reminder makes an old memory act brand new for weeks.

**ACT-R (Honda et al., HAI 2025)**, our base architecture per the proposal. No clock ever resets. Every recall leaves its own trace, and each trace fades on a power law:

```
B = ln(Σ_j t_j^(−d)),   d ≈ 0.5
```

A reminder is a quick bump that dies off while steady use builds a floor that lasts. A power law never hits zero, so old memories go faint, not dead. And a hard retrieval threshold (plus noise) means recall below it simply fails: forgotten, but not erased. Old memories end up hovering just under the threshold, where the right cue can still push them over.

That hovering state is what our graph works on.

### What's missing: every memory lives and dies alone

Our current Hou and simplified ACT-R kernels use a memory's own history and its
similarity to the current cue. Native ACT-R also supports direct contextual
associations; our simplified kernel does not implement that mechanism. The
project asks how explicit relations between experiences could affect recall and
long-term accessibility—for example, recalling a study partner through a familiar
subject or a past interaction through its location. The proposed graph adds
experimental mechanisms beyond ordinary direct ACT-R spreading activation.

### The improvement: a memory relations graph

We keep the per-memory dynamics and make relations a factor in recall and in survival.

- **Nodes** are memories with full text, embedding, and the base architecture's activation state.
- **Typed edges** link them at write time: same person, same place, happened together, happened right after, likely caused, cited as evidence. Each edge has its own weight and clock.
- **Recall spreads.** Retrieval starts from what matches the moment, then activation flows along edges, weighted by strength and damped per hop. A memory can surface because it matched, or because a strong neighbor dragged it in. In ACT-R terms: a cue arriving through an edge is extra context activation that can lift a below-threshold memory over the bar.
- **Reinforcement spreads too.** Recall strengthens the memory, the edges it traveled, and (a small fraction) the neighbors it touched. That last rule is the survival mechanism: memories woven into an agent's ongoing life keep each other alive while isolated trivia fades.
- **Everything reduces cleanly.** Graph off, the system reproduces the base architecture exactly, so every mechanism is one config flag away from an ablation.

One sentence: relevance decides what surfaces right now, decay decides what fades, connections decide what survives.

This combination is a research hypothesis, not an established novelty claim.
Language-generated chunks, ACT-R/LLM interfaces, and embedding-derived ACT-R
associations already have published precedents. The
[ACT-R audit and prior-work comparison](../ACTR_MEMORY_ARCHITECTURE.md#9-prior-work-and-the-contribution-claim)
separates those mechanisms from our proposed interfaces and graph extensions.
The earlier survey remains in [docs/RESEARCH.md](../RESEARCH.md).

### A mini walkthrough

Ninety simulated days of one agent, Ada.

1. **Day 1, write.** Ada meets Ben at the well and he shows her the berry patch. Two memory nodes, edges formed at write time: Ben (person), happened-together, patch-to-well (place).
2. **Days 2 to 30, decay and reinforce.** Ada picks berries most days. Each mundane trip memory fades within days, as it should, but each trip touches the patch node and reinforcement leaks along its edges. Ben gets a small refresh even on days Ada never sees him.
3. **Day 45, a non-semantic link.** Ada eats strange mushrooms. Hours later a separate memory is written: sick all night. The records share no words. A temporal-causal edge binds them.
4. **Day 90, what survived.** The trip episodes are gone (correct). Ben is strong (correct: woven into a routine, not rehearsed). Mushrooms surface the sickness through the edge, so Ada avoids them. Berries still bring Ben to mind.
5. **The baselines fail differently.** Store-everything drowns in trivia. Decay-only forgot the trips and Ben with them, because nothing directly recalled him. Ours forgot the trips and kept Ben.

## Earlier behavioral predictions (unvalidated)

Predictions at three scales, checkable in simulation:

- **One agent.** No uncanny perfect recall (quoting a detail from months ago verbatim reads creepy, not attentive). "Oh right" moments when a cue reaches a faded memory through an edge. Quirks with traceable origins, like avoiding the mushroom patch.
- **Two agents.** Relationships that need maintenance: absence weakens bonds, agents drift apart, reunions carry partial memory. Asymmetric memory, where one agent holds a friendship the other lost. Grudges that fade unless refreshed, so reconciliation becomes possible.
- **A society.** Reputation with a half-life (gossip is retelling, retelling is reinforcement). Knowledge lost when nobody retells it, so elders matter. Taboos: a group avoiding something after everyone who remembers why is gone. Flat-memory societies can do none of this.

## The sandbox: TerraLingua

Per the proposal, the sandbox is [TerraLingua](https://github.com/cognizant-ai-lab/terralingua) ([Paolo et al. 2026](https://arxiv.org/abs/2603.16910), Apache 2.0, Python): a persistent 2D world where LLM agents forage, reproduce, die, and write the environment themselves as persistent text artifacts that outlive their authors. Real simulated time with birth and death, so decay matters. Recurring agents and artifacts, so the graph matters. Its built-in memory is a rolling context window plus a 150-token self-rewritten scratchpad, and our integration replaces that in the agent loop. Fallbacks if the integration spikes badly are tracked in [docs/GAPS.md](../GAPS.md).

Planned models: local SBERT all-MiniLM-L6-v2 embeddings, motivated by the supplied proposal's cited similarity-score distribution, and GPT-5 Nano for generation. A locally available Qwen model is a lower API-cost collection option. No model training is planned. Evaluation still requires labeled histories and human judgments.

## Validation

Per the proposal, plus the engineering practices we carry from Maya:

1. **Existing tests for the memory architecture**: Hou et al.'s appendix B (qualitative) and appendix C (quantitative) tests, and [EmotionBench](https://arxiv.org/abs/2308.03656) for responses to real-life situations. [Shachi](https://arxiv.org/abs/2509.21862) lists believability benchmarks for agents.
2. **Baselines and ablations.** Every mechanism sits behind a config flag. Baselines: a flat vector store, and the base architecture with the graph off. Ablate decay, reinforcement, spreading, and each edge type one at a time. Graph off must reproduce the base architecture exactly, which doubles as a correctness test.
3. **Whole-simulation judgment**: TerraLingua's AI Anthropologist, a separate LLM that reads the logs and judges open-endedness. LLM-only judging of believability is circular, so a human stays in the loop.
4. **Unit and integration tests**: property tests on the decay math (both kernels), determinism (same seed and replay log, identical run), graph invariants (no dangling edges, caps hold, links citing nonexistent memories rejected), and an agent completing simulation episodes end to end.

Our own probe ideas beyond the proposal's list (forgetting quality, the reminiscence test, social signatures) live in [docs/GAPS.md](../GAPS.md) as candidates for milestone 5.

## Repo map

- [docs/DESIGN.md](../DESIGN.md): storage and retrieval mechanics. Schema, spreading, update rules, open math questions.
- [docs/RESEARCH.md](../RESEARCH.md): the paper survey with verified numbers and links.
- [docs/GAPS.md](../GAPS.md): open decisions, what we reuse from Maya, probe candidates.

## Milestones (from the proposal)

1. Create a model using the existing memory architecture. (2 weeks)
2. Inject this model into the agent simulation. (1 week)
3. Improve the memory architecture with memory relations: how links are stored, how relatedness is decided, how spreading and reinforcement work. Most of the novel work, most of the time. (4 weeks)
4. Put the new model in the simulation. (1 week)
5. Validate against the benchmarks used for the existing memory architectures and social agent simulations. (2 weeks)
