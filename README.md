# human-like-agent-simulation

Give LLM agents a long-term memory that works like ours. Memories fade with time, strengthen with recall, and survive through their connections. Then drop those agents into a simulated world and see what changes.

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

The subtle part is how strength grows. A recall after a long gap adds more strength than an immediate re-recall. That is the human spacing effect: cramming fades, spaced review sticks. It falls out of one small formula.

An agent running this forgets naturally and keeps what it keeps using. Already far more human than a vector store that remembers everything forever.

### What's missing: decay is blind to connection

In that model every memory sits on its own clock. Use it or lose it, and "use" means one thing: being directly recalled. A memory nothing re-triggers fades, no matter how much it matters.

Now think about what you actually still hold from years ago. You remember your childhood best friend's kitchen. Not because you rehearse kitchens. Because that kitchen is wired into a hundred things you still think about: the friend, the sleepovers, the walk home. What you lost is the stuff nothing else attached to, like what you ate two Tuesdays ago. Human long-term memory is robust against time in exactly this selective way. **Memories survive by being connected.**

The paper's model cannot produce that. Two memories with the same age and recall count get the same survival odds, whether one is woven into everything the agent cares about and the other is noise. Over a long life, that forgets the wrong things. Relevance cannot patch it either, because a connection is not content: "bitten by a spider" and fear share no words, and math still means Sarah years after every study session memory has faded. The relation itself is the fact worth keeping, and a flat store has nowhere to keep it.

### The improvement: memories survive through their connections

We keep the paper's per-memory dynamics and make relations a factor in recall and in survival.

- **Nodes** carry the `r`/`t`/`g` dynamics above.
- **Typed edges** link them at write time: same person, same place, happened together, cause and consequence, evidence for an insight.
- **Recall spreads.** Retrieval starts from what is relevant now and flows along edges, so a related memory can surface without matching the cue. This is spreading activation, from the ACT-R cognitive architecture.
- **Reinforcement spreads too.** This is the long-term key. Recalling a memory partially refreshes its neighbors, so the parts of life an agent keeps engaging with keep each other alive. You never rehearse the kitchen. Thinking about the friend does it for you.
- **Associations can outlive their sources.** The bite memory fades, the spider-to-fear edge persists. The agent keeps the fear without the story.
- **Forgetting becomes structural.** Survival depends on a memory's own strength and its neighborhood. Isolated trivia fades fastest. Woven-in memories resist decay.

The one-sentence version: **relevance decides what surfaces right now, decay decides what fades, and connections decide what survives.**

In practice this changes storage before it changes retrieval. Memories cannot live in a plain vector store, because a connection needs somewhere to exist. We store a graph: memory nodes, entity hubs (Ben is a node, not a word), and typed, weighted edges created at write time. Retrieval keeps the paper's formula untouched and widens one input: a memory's cue strength is its direct match to the moment or the activation it receives from recalled neighbors, whichever is stronger. Recall then strengthens both the memories and the edges it traveled. Remove the edges and the system reduces exactly to the paper, which makes every ablation a config flag. Full mechanics in [docs/DESIGN.md](docs/DESIGN.md).

### A mini walkthrough

Ninety simulated days of one agent, Ada. This is the whole pipeline in miniature.

1. **Day 1, write.** Ada meets Ben at the well and he shows her the berry patch. Two memory nodes are stored, each with an LLM importance rating as its starting strength. Edges form at write time: both involve Ben (person), they happened together (co-occurrence), the patch connects to the well (place).
2. **Days 2 to 30, decay and reinforce.** Ada picks berries most days. Each trip memory is mundane and fades within days, as it should. But each trip touches the patch node, and reinforcement leaks along its edges. Ben gets a small refresh even on days Ada never sees him. The Ben-berries association wears in while the individual episodes disappear.
3. **Day 45, a non-semantic link.** Ada eats strange mushrooms. Hours later a separate memory is written: sick all night. The two records share no words. The engine adds a temporal-causal edge between them.
4. **Day 90, what survived.** The trip episodes are gone (correct). Ben is strong (correct: he was woven into a routine, not rehearsed). Mushrooms now surface the sickness through the edge, so Ada avoids them even though nausea and mushrooms share no content. Berries still bring Ben to mind.
5. **The baselines fail differently.** A store-everything agent drowns: by day 90 retrieval pulls from a pile of trivia. A decay-only agent (the paper as published) forgot the trips and forgot Ben with them, because nothing ever directly recalled him. Ours forgot the trips and kept Ben.

That contrast is what the simulation affords us. Months of time actually pass and people, places, and artifacts recur, so decay and edges have something real to act on. At day 90 we ask every agent the same questions ("who do you remember, what do you avoid, what goes together"), score the answers against the flat and decay-only baselines with each mechanism ablated in turn, and use Maya-style inspection to see why: "Ben survived through 27 edge refreshes", "this memory lost by 0.1 because it faded". No chat benchmark can produce that evidence.

As far as we can find, nobody has published this combination. Graph memory systems (HippoRAG, Zep, A-Mem, Mem0-g) have no decay or strength. Decay systems (Hou et al., MemoryBank, Generative Agents) have no relations. The one adjacent attempt used forgetting only to prune a graph, and it lost to flat vector search. So the question is genuinely open and we have a named baseline to beat. Details in [docs/RESEARCH.md](docs/RESEARCH.md).

## The papers

- **Source**: [Hou et al., CHI 2024](https://arxiv.org/abs/2404.00573), the formula above. Sibling: an [ACT-R-inspired architecture](https://dl.acm.org/doi/10.1145/3765766.3765803) (HAI 2025) with the same three factors plus native spreading activation and decades of human-fitted parameters. Simpler cousin: [MemoryBank](https://arxiv.org/abs/2305.10250) (Ebbinghaus curve).
- **The OG**: [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al. 2023, "Smallville"), the original recency, importance, and relevance memory stream plus reflection. We keep its scoring skeleton and its reflection trick. We fix what it lacks: nothing ever fades, importance never changes, and memory is a flat list. Full breakdown in [docs/RESEARCH.md](docs/RESEARCH.md).

## Why bother

- **Long-term memory that forgets like a human.** Over a long life an agent must shed almost everything. The question is what to keep. Time and direct use alone keep the wrong things. Connection keeps what is woven in.
- **Believable agents.** Characters that forget acquaintances but remember friends, need reminding, and free-associate. Their memory actually works that way, rather than a prompt saying "act forgetful."
- **Forgetting is a feature.** An agent that remembers everything forever gets less human over time and drowns in its own history.
- **A real scientific hole**, per above. There is an honest chance the answer is "the graph doesn't help." We would publish that too.
- Long-lived companions and assistants benefit downstream. Consequence, not goal.

## What we take from Maya

[Maya](https://github.com/bloohunnits/ai-capstone/tree/main/projects/maya) is our own Generative Agents implementation, the OG paper plus custom engineering. It feeds this project three ways.

- **Architecture worth copying.** A deterministic engine owns the math (decay, strength, traversal, all seeded). The LLM only rates, links, and reflects, and every output is validated, so hallucinations cannot corrupt the store. Maya's citation-checked reflections are already the evidence edges our graph needs. Every feature sits behind an ablation flag. And its decision inspector shows why an agent recalled or missed a memory.
- **Mistakes its own gap analysis warns about.** Eval suites that never touch the LLM layer, no retrieval-precision metric, single-seed science, no LLM record and replay, and sim time too fast to watch. We build the fixes in from day one.
- **A testbed.** Maya's memory stream is a swappable module in our own codebase. Likely play: develop the engine against Maya (fast, instrumented, ours), then deploy into TerraLingua. The engine also back-fills Maya's missing loops (trait evolution, relationship decay).

Details in [docs/GAPS.md](docs/GAPS.md).

## The world: TerraLingua

[TerraLingua](https://github.com/cognizant-ai-lab/terralingua) (Cognizant AI Lab, Apache 2.0) is a persistent 2D world where LLM agents forage, reproduce, die, and write the environment themselves. They leave persistent text artifacts that outlive their authors, so later generations inherit an LLM-made culture.

It fits unusually well. Real simulated time with birth and death, so decay matters. Recurring agents and artifacts, so the graph matters. Open-ended LLM-generated content, so memory shapes culture and culture shapes memory. The caveat: no pluggable memory interface, so we would wire into the agent loop ourselves. Fallbacks if that spikes badly: [AI Town](https://github.com/a16z-infra/ai-town) / [Smallville](https://github.com/joonspk-research/generative_agents) (drop-in memory slot) and DeepMind's [Concordia](https://github.com/google-deepmind/concordia) (swappable memory components, LLM game master). Open decision, tracked in [docs/GAPS.md](docs/GAPS.md).

## How we'll know it worked

1. **Watch the agents.** Do they behave more like people? Forming relationships, forgetting selectively, acting on associations. Generative Agents' believability interviews and TerraLingua's "AI Anthropologist" log analysis are the templates.
2. **Ablate everything.** Decay off, reinforcement off, edges off, one at a time. Each mechanism visibly changes behavior or it gets cut. The decay kernel is part of the grid too: the paper's reset-style exponential and ACT-R's power law, each alone and each with the graph.
3. **Benchmarks as a sanity check.** [LongMemEval](https://arxiv.org/abs/2410.10813) (primary), [LoCoMo](https://www.emergentmind.com/topics/locomo), [MemBench](https://arxiv.org/abs/2506.21605). Human-like memory must not wreck task utility, measured against Mem0, A-Mem, Zep, and flat-vector baselines.
4. **New probes for what no benchmark measures.** Did the right things fade? Does interference behave sanely? Can association rescue faded memories that similarity alone loses?

## Repo map

Single project repo for the capstone. Research docs now, code lands here as it's written.

- [docs/DESIGN.md](docs/DESIGN.md): how storage and retrieval actually work. The schema, the widened formula, the update rules, open math questions.
- [docs/RESEARCH.md](docs/RESEARCH.md): full paper survey. Source papers, OG mechanics, graph systems, benchmarks, environments.
- [docs/GAPS.md](docs/GAPS.md): what the papers don't cover, what Maya fills, open decisions.

## Roadmap

1. **Ground truth.** Finish the source papers (including getting the paywalled ACT-R one) and lock the exact formulation.
2. **Core engine.** Standalone memory library reproducing the human-like paper's decay, strength, and relevance behavior.
3. **Graph layer.** Typed edges, LLM-assisted linking at write time, spreading recall and reinforcement.
4. **Benchmark harness.** LongMemEval and LoCoMo with ablations, baselines, and our forgetting-quality probes.
5. **Simulation.** Wire into Maya, then TerraLingua. Run the human-likeness studies.
6. **Write-up.** Findings, honest negatives included.
