# human-like-agent-simulation

Give LLM agents a memory that works like ours. Memories fade with time, get stronger each time they are recalled, and pull each other up by association. Then drop those agents into a simulated world and see what changes.

The goal is human-likeness, not faster retrieval or fewer tokens. We want agents that remember, forget, and associate the way people do.

## The idea

### The starting point: memory as a fading, strengthening curve

The paper we build on ([Hou et al., CHI 2024](https://arxiv.org/abs/2404.00573)) gives every memory a probability of coming to mind right now:

```
p = [1 − exp(−r · e^(−t/g))] / [1 − e^(−1)]
```

Strip the math and it makes three claims about remembering:

- **Relevance (`r`).** A memory only surfaces when the present resembles it. If nothing around you reminds you of it, it stays buried. Measured as similarity between the current context and the memory.
- **Time (`t`).** The longer since you last touched a memory, the deeper it sinks. `e^(−t/g)` is the classic forgetting curve. Steep at first, then it flattens.
- **Strength (`g`).** How well-worn the memory is. Every recall bumps `g` up, and a bigger `g` slows the decay. Remembering is rehearsal. Each recall re-carves the groove.

The subtle part is how strength grows. A recall after a long gap adds more strength than an immediate re-recall. That is the human spacing effect: cramming fades, spaced review sticks. It falls out of one small formula. The denominator just normalizes, so a perfectly relevant, just-recalled memory scores 1.

An agent running this forgets naturally and keeps what keeps mattering. Already far more human than a vector store that remembers everything forever.

### What's missing: every memory lives and dies alone

In that model each memory is an isolated jar on a shelf, fading on its own schedule, reachable only when the current moment resembles it. Human memories are not like that. They are built related. A memory's meaning is partly its position in the web: what it is wired to, not just what it says. Embeddings capture what a memory is about. Relations capture what it means to you. Two people can see the same spider on the same wall and it means completely different things, because of what "spider" connects to in each of their histories.

Be precise about the objection here, because it is a fair one: relevance handles the direct match. Mention a spider and similarity will happily retrieve your spider bite memory, since spider matches spider. The failures are in everything around the match. The association outliving the memory that made it. Two records that share no content but are bound by life. The flood from one matched memory to its whole cluster. Old memories changing meaning when something new happens. And a match that should land as behavior, not as a paragraph of retrieved text. Each example below shows one of these:

- **The spider bite.** While the bite memory exists, relevance finds it fine. The failure comes later. Decay eventually drops that childhood episode below recall, and in a flat store the fear vanishes with it. That is backwards. Humans keep the fear long after they lose the episode. In the graph the bite forged a spider-to-fear edge, the episode node fades, the edge survives. "I don't know why I hate spiders." And fear is not a snippet: the edge should fire as state and behavior, not as text the LLM may or may not use.
- **Sarah and calculus.** Every study session memory mentions both, so while those episodes are fresh, similarity connects math to Sarah fine. But study session #14 is mundane, and mundane episodes fade, exactly as they should. What a human keeps is the worn-in association: math means Sarah. In a flat store the link dies with the episodes. In the graph, every session reinforced a math-Sarah edge that outlives them all. And the edge chains: math surfaces Sarah, and Sarah brings context that shares nothing with math (she moved away, she hated exams).
- **The smell of sunscreen.** Similarity gets you, at best, the one memory that happens to mention sunscreen. What humans get is the flood: the dock, the cousins, the water fight, the whole summer. Those memories share nothing with the cue. They connect to it only by having happened together. Co-occurrence edges turn one matched memory into the cluster.
- **The food stall.** Two separate records: "ate at the stall" that evening, "sick all night" hours later at home. The sickness memory never mentions the food. No shared words, no semantic similarity between nausea and noodles. Only a temporal or causal edge binds them. Without it, the cue "that stall" retrieves a perfectly pleasant dinner memory and nothing warns you. With it, the aversion fires. (Humans really do form this link in one trial, hours apart.)
- **The betrayal.** Retrieval is not the problem here either. Mention Dave and similarity returns the camping trip and the betrayal both. The problem is the camping trip is still stored as a warm memory, and it should not read warm anymore. Humans re-color the past. A flat store cannot touch old records. A graph propagates the new valence through the Dave hub and re-weights every connected memory in one pass.
- **The trigger.** A backfire is weakly similar to gunfire at best, so top-k similarity might surface one combat snippet. What happens to a veteran is different in kind: one weak cue detonates a densely connected, over-strong cluster, including the dust, the heat, the friend, memories that share nothing with sound, and it lands as body state, not as retrieved text. That takes spreading activation across strong edges plus valence carried into state. (This models the pattern for simulated characters. It is not a claim to capture the condition.)
- **The lucky socks.** The win memory may well mention the socks, and similarity can retrieve it. But similarity treats the socks as an irrelevant detail in a story about winning. The superstition is a binding: socks cause wins. Causally wrong, psychologically real, strengthened every time the ritual repeats. Only an edge can hold a relation the content itself does not justify.
- **"The summer before Dad got sick."** Ask when something happened and the record itself has no answer. People date events by hopping to landmark events through temporal edges. Similarity cannot order the past. Edges can.

The pattern across all of them: relevance finds the matching text. The edges carry what the match is connected to, what it now means, what survives after it fades, and what it does to the agent.

### The improvement: a memory graph

We keep the paper's per-memory dynamics and wire the memories together.

- **Nodes** carry the `r`/`t`/`g` dynamics above.
- **Typed edges**: same person, same place, happened right after, same moment, cause and consequence, evidence for an insight, and affective ("this means fear").
- **Recall spreads.** Retrieval starts from what is relevant now, then activation flows along edges, weighted by strength. This is spreading activation, straight out of the ACT-R cognitive architecture. It carries more than accessibility: an activated fear edge should change the agent's state and behavior, not just which text lands in its context window.
- **Recall reinforces structure.** A recall strengthens the node and the links it traveled. Associations get worn in or fade.
- **Edges can outlive their nodes.** This is the mechanism behind keeping the fear while forgetting the bite. The episodic memory decays away while the association it forged persists.
- **Forgetting becomes structural.** Isolated trivia fades fastest. Woven-in memories keep getting rescued by their network. And one strong new memory can reorganize the meaning of many old ones through shared hubs.

As far as we can find, nobody has published this combination. Graph memory systems (HippoRAG, Zep, A-Mem, Mem0-g) have no decay or strength. Decay systems (Hou et al., MemoryBank, Generative Agents) have no relations. The one adjacent attempt used forgetting only to prune a graph, and it lost to flat vector search. So the question is genuinely open and we have a named baseline to beat. Details in [docs/RESEARCH.md](docs/RESEARCH.md).

## The papers

- **Source**: [Hou et al., CHI 2024](https://arxiv.org/abs/2404.00573), the formula above. Sibling: an [ACT-R-inspired architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (HAI 2025) with the same three factors plus native spreading activation and decades of human-fitted parameters. Simpler cousin: [MemoryBank](https://arxiv.org/abs/2305.10250) (Ebbinghaus curve).
- **The OG**: [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al. 2023, "Smallville"), the original recency, importance, and relevance memory stream plus reflection. We keep its scoring skeleton and its reflection trick. We fix what it lacks: nothing ever fades, importance never changes, and memory is a flat list. Full breakdown in [docs/RESEARCH.md](docs/RESEARCH.md).

## Why bother

- **Believable agents.** Characters that forget acquaintances but remember friends, need reminding, and free-associate. Their memory actually works that way, rather than a prompt saying "act forgetful."
- **Forgetting is a feature.** Human forgetting is adaptive. It clears the stale and trivial and keeps what recurs. An agent that remembers everything forever gets less human over time and drowns in its own history.
- **Relation is part of meaning.** What a spider means to you depends on what "spider" is wired to in your history. People retrieve by connection, and they feel by connection too.
- **A real scientific hole**, per above. There is an honest chance the answer is "the graph doesn't help." We would publish that too.
- Long-lived companions and assistants benefit downstream. Consequence, not goal.

## What we take from Maya

[Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) is our own Generative Agents implementation, the OG paper plus custom engineering. It feeds this project three ways.

- **Architecture worth copying.** A deterministic engine owns the math (decay, strength, traversal, all seeded). The LLM only rates, links, and reflects, and every output is validated, so hallucinations cannot corrupt the store. Maya's citation-checked reflections are already the evidence edges our graph needs. Every feature sits behind an ablation flag. And its decision inspector shows why an agent recalled or missed a memory ("lost by 0.1 because it faded", "rescued by an edge").
- **Mistakes its own gap analysis warns about.** Eval suites that never touch the LLM layer, no retrieval-precision metric, single-seed science, no LLM record and replay, and sim time too fast to watch. We build the fixes in from day one.
- **A testbed.** Maya's memory stream is a swappable module in our own codebase. Likely play: develop the engine against Maya (fast, instrumented, ours), then deploy into TerraLingua. The engine also back-fills Maya's missing loops (trait evolution, relationship decay).

Details in [docs/GAPS.md](docs/GAPS.md).

## The world: TerraLingua

[TerraLingua](https://github.com/cognizant-ai-lab/terralingua) (Cognizant AI Lab, Apache 2.0) is a persistent 2D world where LLM agents forage, reproduce, die, and write the environment themselves. They leave persistent text artifacts that outlive their authors, so later generations inherit an LLM-made culture.

It fits unusually well. Real simulated time with birth and death, so decay matters. Recurring agents and artifacts, so the graph matters. Open-ended LLM-generated content, so memory shapes culture and culture shapes memory. The caveat: no pluggable memory interface, so we would wire into the agent loop ourselves. Fallbacks if that spikes badly: [AI Town](https://github.com/a16z-infra/ai-town) / [Smallville](https://github.com/joonspk-research/generative_agents) (drop-in memory slot) and DeepMind's [Concordia](https://github.com/google-deepmind/concordia) (swappable memory components, LLM game master). Open decision, tracked in [docs/GAPS.md](docs/GAPS.md).

## How we'll know it worked

1. **Watch the agents.** Do they behave more like people? Forming relationships, forgetting selectively, acting on associations. Generative Agents' believability interviews and TerraLingua's "AI Anthropologist" log analysis are the templates.
2. **Ablate everything.** Decay off, reinforcement off, edges off, one at a time. Each mechanism visibly changes behavior or it gets cut.
3. **Benchmarks as a sanity check.** [LongMemEval](https://arxiv.org/abs/2410.10813) (primary), [LoCoMo](https://www.emergentmind.com/topics/locomo), [MemBench](https://arxiv.org/abs/2506.21605). Human-like memory must not wreck task utility, measured against Mem0, A-Mem, Zep, and flat-vector baselines.
4. **New probes for what no benchmark measures.** Did the right things fade? Does interference behave sanely? Can association rescue faded memories that similarity alone loses?

## Repo map

Docs only. This is the planning and research home. Code lives elsewhere once we build.

- [docs/RESEARCH.md](docs/RESEARCH.md): full paper survey. Source papers, OG mechanics, graph systems, benchmarks, environments.
- [docs/GAPS.md](docs/GAPS.md): what the papers don't cover, what Maya fills, open decisions.

## Roadmap

1. **Ground truth.** Finish the source papers (including getting the paywalled ACT-R one) and lock the exact formulation.
2. **Core engine.** Standalone memory library reproducing the human-like paper's decay, strength, and relevance behavior.
3. **Graph layer.** Typed edges, LLM-assisted linking at write time, spreading-activation recall.
4. **Benchmark harness.** LongMemEval and LoCoMo with ablations, baselines, and our forgetting-quality probes.
5. **Simulation.** Wire into Maya, then TerraLingua. Run the human-likeness studies.
6. **Write-up.** Findings, honest negatives included.
