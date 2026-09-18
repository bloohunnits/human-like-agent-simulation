# Design: Storage and Retrieval

How association actually enters the formula. Status: proposal, pre-implementation. Companion to the [README](../README.md) and [RESEARCH.md](RESEARCH.md).

## Storage comes first

You cannot bolt edges onto a vector store after the fact. A connection needs somewhere to exist, and the information needed to create it (what else was happening, who was there, what just preceded this) is only cheaply available at write time. So the store is a graph with an embedding index attached, not a vector database with metadata.

### Nodes

- **Memory** (episodic): text, embedding, `t_created`, `t_last_recalled`, strength `g`, importance-at-birth.
- **Entity** (person, place, thing): a hub. Ben is a node, not a word that appears in memory texts. This is the HippoRAG move, and it is what makes "everything about Ben" a one-hop neighborhood instead of a similarity search.
- **Insight**: reflection output. Same fields as a memory, plus evidence edges down to what it came from.

### Edges

Each edge carries a type, its own weight `w` (association strength), and `t_last_traversed`, so associations decay and strengthen on their own clocks, separate from the memories they connect.

Types, in build order: `mentions` (memory to entity), `co-occurred` (same scene or time window), `followed` (temporal succession), `cause-candidate` (close in time plus LLM affirmation), `evidence-of` (reflection citations). Later: `affective` (spider to fear).

### Write pipeline, per new memory

1. Embed and store the node. An LLM importance rating seeds its starting strength.
2. Extract entities and add `mentions` edges, creating entity nodes as needed.
3. Add deterministic edges: `co-occurred` with memories from the same scene window, `followed` to the previous event in the thread.
4. LLM-proposed links (`cause-candidate`, later `affective`) must cite existing node ids and are validated before they land. This is Maya's citation-check pattern applied to linking.

Reflection runs on its own trigger and writes insight nodes with `evidence-of` edges.

## The decay kernel is swappable

The per-memory dynamics have two candidate shapes, and they disagree in a way agents will visibly show, so we implement both behind one interface.

**Hou et al., exponential with reset.** Each memory keeps one clock and one strength number. On recall the clock resets to zero and strength grows, so the whole forgetting curve restarts from the top and fades slower than before. Simple, and it's the paper we reproduce as our baseline. The suspect behavior: a single mention of a decades-old memory rejuvenates it wholesale. Interview an agent once about its childhood and that memory outcompetes recent ones for a long while afterward.

**ACT-R, power law over history.** No clock ever resets. Every retrieval adds one term to a running sum, and each term fades on its own power law: `B = ln(Σ_j t_j^(−d))` with `d ≈ 0.5`. A fresh retrieval adds a big term that itself shrinks fast, so one mention gives a short priming bump and the memory settles back near its baseline. Thirty spaced retrievals build a lasting floor. The spacing effect emerges instead of being bolted on. Two extra properties we want: the power-law tail leaves old memories faint but revivable, which is exactly the state edge rescue acts on, and the retrieval threshold plus noise gives a crisp definition of functionally forgotten for our metrics. Cost is a recall history per node, handled with ACT-R's standard constant-size approximation.

**The experiment this sets up.** A 2x2 grid: each kernel alone, each kernel with the graph factors mixed in. Benchmarks and probes run over all four cells. And one behavioral probe the kernels flatly disagree on, the reminiscence test: mention one old memory to an agent exactly once, then count spontaneous references to it over the following simulated days. Reset predicts a long elevation. Power law predicts a brief bump and a return to baseline. Humans match the second, so this doubles as a believability measurement and turns the kernel choice into a finding instead of a preference.

## Retrieval: association in the formula

The paper scores each memory independently: `p(i) = P(r_i, t_i, g_i)` where `r_i` is similarity between the current context and memory `i`.

We keep that formula untouched and widen one input. Three steps, all deterministic, no LLM in the loop:

**1. Seed.** Compute `r_i` from embedding similarity as usual. Base activation `a_i = P(r_i, t_i, g_i)`.

**2. Spread.** Activation flows along edges for one to two hops with damping `λ < 1`:

```
received_j = Σ_i  a_i · norm(w_ij) · λ
```

Then each memory's effective cue strength is the better of its two routes into mind:

```
r̂_j = max(r_j, received_j)        final score = P(r̂_j, t_j, g_j)
```

That is the whole trick. A memory surfaces either because the moment resembles it (direct cue, `r`) or because a strongly associated neighbor is being recalled (associative cue, `received`). Same formula, richer `r`. Decay and strength still gate everything, so a faded memory needs a strong pull and a strong memory needs only a weak one.

**3. Update.** The recalled set gets the paper's strength update (`g += S(t)`, reset `t`). Every edge traversed gets the edge version of the same update (`w` bumps, spaced traversals bump more). Neighbors whose received activation cleared a threshold get a partial refresh, a fraction of a full recall. That last rule is the survival mechanism: reinforcement leaks one hop, so the parts of life the agent keeps engaging with keep each other alive without ever being directly recalled.

### The reduction property

Remove the edges and set `λ = 0` and this is exactly Hou et al. Every addition is a config flag, so the ablation study is a parameter sweep: paper baseline, plus edges, plus spread, plus leaked reinforcement, each measured alone and together.

### Why this shouldn't repeat the prior failure

The one published graph-with-forgetting system (Selective Forgetting, arXiv:2608.28978, details in [RESEARCH.md](RESEARCH.md)) lost to a flat vector baseline, and its two failure modes map to choices made differently here:

- It replaced conversation text with extracted entity-relation triples, destroying wording needed for precise recall. Our memory nodes keep the full text and embedding. Entity nodes are hubs added alongside, never a substitute.
- Its expansion was an unweighted breadth-first walk with no decay, strength, or edge weighting in ranking. Our spread is scaled by edge strength and damped per hop, and everything it reaches must still pass the decay-and-strength formula to surface.
- Its stale-value bug (a conflict rule keeping old facts) is handled here by the dynamics themselves: a superseded fact stops being used, so it fades while its replacement strengthens. One of the forgetting-quality probes tests exactly this.

None of this guarantees our graph wins. It does mean their negative result doesn't test our mechanism.

### Cost and guardrails

- No LLM calls at retrieval time. LLM work happens once, at write time, and is validated.
- Spread is local (one to two hops from the seed set), so cost scales with neighborhood size, not store size.
- Cap edges per node (keep top-k by weight) so hubs like a best friend don't connect to everything and drown the spread.
- Edges decay too. An association never traversed loosens, which keeps the graph from ossifying into "everything relates to everything".
- Embedding cache keyed by text hash (from Maya's list).

## Open math questions

1. `max` vs `sum` when combining direct and received cue strength. Sum rewards convergent evidence (several weak routes agreeing) but risks feedback loops. Start with max, test sum in sim.
2. Edge weight initialization: uniform, co-occurrence count, or LLM confidence? Start uniform, let traversal differentiate.
3. Bounded 2-hop spread vs full Personalized PageRank (HippoRAG-style). Start bounded, PPR as a variant. PPR handles long chains but obscures why a memory surfaced, and explainability is a design goal.
4. The partial-refresh threshold and fraction. Too generous and nothing ever fades (the store-everything failure returns through the back door). Too stingy and we reproduce the paper. This is the key parameter the simulation has to tune, with sensitivity reported.
5. Does `received` activation update `t_last_recalled`? Probably not (only true recalls reset the clock, partial refresh only bumps `g`), otherwise spreading silently freezes the whole neighborhood's decay.
