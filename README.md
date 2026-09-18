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

None of these connections are semantic:

- **The spider bite.** Bitten once as a kid, and "spider" is now wired to fear. Spider and fear share no meaning. The bite made the edge. Deeper still, you can forget the bite and keep the fear. The association outlives the memory that created it. A flat store cannot represent that. A graph can: decayed node, surviving edge, "I don't know why I hate spiders."
- **Sarah and calculus.** Math and calculus are semantically close. Math and Sarah, your study partner, are not. They are related because your life put them in the same room, over and over. Mention math and Sarah surfaces.
- **The smell of sunscreen.** It unlocks one particular summer, the way Proust's madeleine unlocked a childhood. Cue and memory share nothing textual. They share lived co-occurrence.
- **The food stall.** Sick once after eating there, and you avoid the stall, then the dish, then the block. One event, a permanent aversion edge. Humans really do form one-trial taste aversions hours after the meal, with no semantic link between nausea and dinner.
- **The betrayal.** A friend betrays you and every memory involving them changes color. One new memory re-weights hundreds of old ones through a shared person hub. Similarity search cannot re-score your past. A graph does it in one propagation.
- **The trigger.** For a veteran, a car backfire is not "a loud noise". It is wired to a memory complex the sound would never retrieve semantically. PTSD-like dynamics become legible in this model: nodes too strong to decay, edges that spread too wide. (This represents the pattern for simulated characters. It is not a claim to capture the condition.)
- **The lucky socks.** Wore them, won the game. The edge is causally wrong but psychologically real. Human-like includes the irrational associations. Agents get quirks whose origins you can trace.
- **"The summer before Dad got sick."** People date events relative to other events, not calendars. Temporal landmarks are edges.

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
