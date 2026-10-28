# Team catch-up: relational memory and the class proposal

Updated 2026-09-28. Start here, then use the [PowerPoint](../output/presentations/relational-memory-proposal-v12.pptx), [PDF](../output/presentations/relational-memory-proposal-v12.pdf), [slide text](../output/presentations/slide-text-v12.md), and [presenter guide](../output/presentations/presenter-guide-v12.md). The deck has 14 main slides and 7 backup slides. The PDF is static; the TerraLingua GIF animates in PowerPoint Slide Show.

## What we are trying to learn

Our aim is more believable remembering and behavior in a social simulation. We want an agent's own experiences to supply appropriate context for later decisions. We are not claiming a general improvement in intelligence, efficiency, or task performance.

**Thesis:** Associative memory should give agents more appropriate context for their decisions, making their behavior more believable and consistent with their own experiences.

**H1 — Associative recovery:** Associations will help agents retrieve past experiences relevant to their present situation, including experiences that semantic similarity and time decay alone would miss.

**H2 — Lasting accessibility:** Recalling experiences through their associations will help preserve relevant context across longer gaps.

These are the accepted presentation statements. We evaluate useful recall, distracting reminders and behavior across histories. Raising a score is a property of the chosen formula; appropriate context and believability are the research outcomes to test.

## The example we settled on

Mia gives the agent food beside a blue pinwheel. She says only, "I'm usually here." She does **not** tell it to return for food. Much later, the agent is hungry and notices the same pinwheel; Mia and food are outside its current view. Remembering the encounter gives it a reason to approach and look for Mia, who might help again.

The cue acquires significance through this agent's experience. The graph connects the observed pinwheel to the whole encounter. The episode contains the food transfer and Mia's remark. The action model can infer a possible next step from that episode and its current hunger. The graph does not assign a food reward or an instruction to approach the object. Another agent with a different history could recall something else and act differently.

The relevant contrast is **similarity plus ordinary decay versus the same direct route with an additional, independently aging association**. We should not claim that semantic retrieval cannot recognize this episode: its text contains both pinwheel and food.

### What the numerical example actually shows

The reproducible fixture uses the current episode writer, local MiniLM encoder and retrieval engine. It supplies hypothetical feedback that Mia gave food. It is not an observed simulation event, and no LLM approach behavior has been measured for this scenario.

To make the comparison stronger, the numerical illustration explicitly appends `I am hungry. Current goal: find food.` to the ordinary observation query. Raw cosine similarity is **0.3702016**. That cosine stays constant when the texts stay fixed; the time-dependent contribution changes.

For this episode, both clocks start at zero and there are no further recalls:

```text
M(t) = 0.3702016 × exp(-t / 25)          direct contribution
G(t) = 0.5 × 0.8 × exp(-t / 100)        one active, uncrowded association
F(x) = (1 - exp(-x)) / (1 - exp(-1))    common score transformation

Direct-only score = F(M)
With-association score = F(max(M, G))
```

| Ticks since the encounter | Direct-only final score | With-association final score | Eligibility above .15 |
| ---: | ---: | ---: | --- |
| 0 | .489471 | .521546 | Both |
| 32 | .154733 | .398785 | Both |
| 33 | .148961 | .395361 | Association only |
| 100 | .010690 | .216473 | Association only |
| 139 | .002252 | .150015 | Association only |
| 140 | .002164 | .148595 | Neither |

The direct route reaches the cutoff at **32.8173 ticks**; the associative route reaches it at **139.0108 ticks**. Selection requires a score strictly greater than .15, so **integer ticks 33–139** form the association-only recall window in this fixture. The chart shades that interval between the two cutoff crossings.

There is **no positive-time crossover between the two contributions**. Graph support starts at .4, already above direct support .3702, and decays more slowly. Algebraic equality gives -2.5806 ticks, outside the scenario. This chart illustrates a longer access window, not an initially weaker graph overtaking the direct route.

Our earlier chat compared raw cosine .370 with final score .216. Those are different stages of the calculation. Compare M with G, or F(M) with F(G), as the chart now does.

The curve assumes one fixed eligible route, no new episodes, no intervening rehearsal and no competition for prompt slots. Looking at a point is a read-only probe, not a recall that renews the memory. Actual repeated recalls or a growing graph change the curve. With g=1 and a 25-tick direct time scale, even cosine 1 is below the cutoff at tick 100. This example therefore motivates a slower-forgetting direct baseline as well as the ordinary baseline; it does not by itself establish association-specific superiority.

### Hunger and retrieval are separate in the current simulation

The live query normally embeds observation, incoming messages and optional environment information. Energy and the short plan reach the action prompt separately. The hunger-aware phrase in this fixture is an explicit experimental query variant, not a live implementation change.

The implemented sequence is: notice the pinwheel, retrieve the associated encounter, then interpret it alongside hunger when choosing an action. Hunger alone does not currently activate the unseen pinwheel as a graph source. Adding internal needs or goals as sources would be a new mechanism and should be tested separately.

### The split-record example (added 2026-09-29)

The one-record example has a weakness a reviewer will find: the pinwheel and the food are in the same episode because the writer records everything from one tick together, so similarity already sees the pinwheel in that text. The graph is not linking two memories there; it is a second, slower clock on one memory. To show a relation doing work, the cue and the payoff must be in different records.

`scripts/pinwheel_split_walkthrough.py` builds that case from observation fields the sandbox really provides. Tick 0: Mia and the pinwheel in view, Mia says "I'm usually here," the agent moves. Tick 1: the pinwheel is out of view, Mia says "Here, have some food," and the agent's energy rises from 25 to 45 (the writer now records energy rises; a receiver previously had no trace of a gift at all). The food episode never mentions the pinwheel. Later the pinwheel is in view and the agent is hungry.

| Route to the food episode | Reaches it | Above .15 until tick |
| --- | --- | --- |
| Similarity plus decay (r = .354 with the hunger line, .206 without) | yes | 32.7 (19.1 without the hunger line) |
| One hop from the pinwheel hub (`entity_hops=1`, the old main condition) | never | n/a |
| Two hops: pinwheel → sighting → followed edge → food (`entity_hops=2`) | yes | 58.8 |
| Two hops through Mia: pinwheel → sighting → Mia hub → food (three edges) | yes | 31.8 (54.1 if fresh edges started at weight 1) |

**The gap case and the turned-up link weight.** `scripts/pinwheel_gap_walkthrough.py` puts ten ticks between the sighting and the food, with nobody in view in between. Now the tick chain cannot carry the pinwheel either; the only route is pinwheel → sighting → Mia → food, three links multiplied. At the original fresh link weight of .8 that route fades at tick 34.8 while similarity lasts to 41.7, so the graph adds nothing. At a fresh weight of 1.0 the route lasts to 57.1 and keeps the food line in the prompt for 16 ticks after decay has removed it. The `chained` launcher condition therefore starts fresh links at 1.0. When comparing `chained` against `learned`, give `learned` the same weight through `--memory-config` so the two differ only in the extra hop. Chart: `output/analysis/pinwheel-gap-window.png`.

This is the argument to make: the graph is a positive reminder. Because the score is max(direct, graph), a link can only add. When similarity still works, similarity wins. When decay has removed a memory but a thing in view still links to it, the link brings it back. The claim is that links outlast decay, not that links beat similarity.

So with `entity_hops=2` the food memory is in the prompt at ticks 33 to 58 when no similarity setting could put it there. The sighting episode itself is unchanged (one hop, until tick 139), and the original one-record fixture scores identically under both settings. Two honest limits: the winning route is the followed edge, which is content-blind (whatever was recorded the tick after a sighting comes back with it), and the meaningful chain through Mia multiplies three transmissions, so at the current edge weight it fades about when similarity does. That is a calibration decision for the team, not a code fix. The fixture uses vision radius 1 so one move hides the pinwheel; at the project default of 6 the pinwheel stays in view for several ticks and the payoff usually lands in a record that also mentions it, which is the one-record case. Chart: `output/analysis/pinwheel-split-window.png`; numbers: `output/analysis/pinwheel-split-walkthrough.json`.

## What is implemented

The live entry point is [run_hou_experiment.py](../code/hou_memory/run_hou_experiment.py). [associative_memory.py](../code/hou_memory/associative_memory.py) handles scoring and updates; [memory_agent.py](../code/hou_memory/memory_agent.py) integrates them with TerraLingua.

Each episode has its own **g**, a memory strength, and a last-full-recall clock. Each association has its own **w**, a connection weight, **h**, its retention time, and a reinforcement clock. Uppercase **G** is the associative contribution calculated for a particular episode and cue; it is not a shared memory strength.

Direct support combines cosine similarity and ordinary decay. Graph support comes from currently active structured entity cues and the surviving connections to episodes. The independent mode takes `max(M, G)` after the two routes have aged separately. It does not multiply G by the target memory's ordinary decay again. Without a currently eligible source, that route contributes zero.

The current defaults use observed entity-to-episode routes, maximum 3 recalled episodes, a strict .15 cutoff and a combined 600-token retrieved-text budget. Optional bounded memory-to-memory routes exist but are disabled in the main study. These defaults are starting values, not fitted human-memory parameters.

### What "selected" means and what gets strengthened

1. Score the memories using the pre-retrieval state.
2. Include up to three qualifying episodes whose text fits the budget. This is "selected," "surfaced," or "full recall" in this implementation: the episode text enters the prompt.
3. Every included episode receives ordinary rehearsal: g increases by the spacing rule and its full-recall clock resets.
4. If G beats M for an included episode, its strongest graph route receives credit. A direct-route tie gets no graph credit. Shared edges update once per event.

An edge update starts from its **currently decayed transmission** z and sets `w = z + .2(1-z)`. It increases h by 25, capped at 200, and resets that edge's clock. An old stored weight can therefore become numerically smaller while current transmission improves. Unselected memories and their routes receive no rehearsal. Merely seeing a cue does not renew every connection attached to it.

Rehearsal commits while preparing the memory input, before the LLM responds. It is credit for retrieval, not a reward for a successful action. Retries do not double-count it. Prompt inclusion does not establish that the model used the memory or made an appropriate decision.

Memory and graph state share the world checkpoint. The `/audit` UI records actual retrieval contributions, selected and excluded episodes, routes, updates, the exact prompt and the resulting action. The older `/` page is a historical what-if console, not the live independent model.

## What the logical review found

The [full logic audit](PROPOSAL_LOGIC_AUDIT.md) and [numeric fixtures](../output/analysis/logical-audit-v11.json) retain the details. The formulas are internally coherent, but their design choices have consequences:

| Finding | Meaning for the experiment |
| --- | --- |
| Raising every score can still worsen retrieval | In a controlled three-slot example, three irrelevant linked episodes displace the useful one. Count useful recall, false reminders and displacement. |
| Max can hide semantic distinctions | Once G exceeds M, changing M below G no longer changes that episode's score. Graph links are positive support, not inhibitory evidence; weak routes also do not accumulate. |
| Crowding can defeat persistence | At edge age 100, increasing the hub from one to five episodes lowers support from .216 to .147, below cutoff. The current fan penalty counts all retained links, including faded ones. Evaluate growing and busy cues. |
| A weighted lookup can implement the main graph rule | It stores the same relationships and gives the same one-hop scores. This is representation equivalence, not evidence against associative benefit. Our main study tests the association policy, not the superiority of graph storage. |
| The original bell example depended on the query | An explicit food goal moved the useful offer into semantic top three. We replaced that slide with the subtler Mia–pinwheel encounter and a time-decay comparison. The old fixture remains an archived example. |
| Retrieval and behavior need separate evidence | A memory can enter the prompt and be ignored. Trace actions and use blind human judgments for believability. |

Other boundaries: stored episode text remains intact, so we are not modeling reconstructive memory; links are not evidence of causality; clinical trauma and emotion are outside this implementation. Entity names are normalized strings rather than a complete identity resolver. Episode tags can include intended action targets, which do not prove completed encounters. The short plan may still carry historical information despite its intention-only instruction, so inspect it when evaluating recall effects.

No scoring, selection or reinforcement rule was changed to make this revision's chart look better. Max, crowding and the original defaults remain in place.

## How to run the experiment responsibly

The three main presentation conditions are:

| Launcher condition | Purpose |
| --- | --- |
| `hou` | Semantic similarity and ordinary decay |
| `independent` | Add independently aging associations, without link learning |
| `learned` | Also rehearse the winning association when it helps an included episode |
| `chained` | As `learned`, and an observed entity may continue one step past a tagged episode (`entity_hops=2`), so a cue can reach an episode that never mentions it; fresh links start at weight 1.0 |

`gated` remains an implementation/history control, not a fourth main proposal condition. ACT-R is outside the current presentation and active experiment.

For **H1**, replay matched histories and vary the delay, cue and distractions. At each delay, keep the episode state, text and prompt budget equal across conditions. Label useful episodes from the scenario before examining scores. Include absent/wrong cues, shuffled links and crowded hubs. Check richer, goal-aware queries in both conditions and a slower ordinary-decay baseline.

For **H2**, give both copies the same ordinary recall events and let only one rehearse its connections. Later read-only probes isolate the extra connection effect. The separate H2 chart uses an illustrative cosine .20; it is not a continuation of the pinwheel fixture. At a 90-tick gap after the matched recall, no-link-update score is .135 and learned-link score is .199; ordinary direct scores are identical at .044.

Then run complete simulations, where retrieval can change actions and future histories. These runs measure the total behavioral effect. Turning the graph off only at the final tick does not undo earlier graph-induced rehearsal or information retained in plans. Keep generator settings and prompt budgets consistent. Choose parameters and acceptable false-reminder tradeoffs on development cases, then freeze them for held-out evaluation.

SBERT all-MiniLM-L6-v2 provides cosine scores; the cited source motivates its relatively sparse/higher-contrast similarity-score distribution, not sparse embedding coordinates. GPT-5 Nano is the planned primary generator. A locally available Qwen model is a practical lower API-cost collection option. Human ratings remain necessary for the believability claim.

## Reproduction and current verification

See [setup and run instructions](ASSOCIATIVE_IMPLEMENTATION.md). From the repository root, after installing dependencies:

```bash
code/terralingua/.venv/bin/python scripts/pinwheel_walkthrough.py
code/terralingua/.venv/bin/python scripts/pinwheel_split_walkthrough.py
code/terralingua/.venv/bin/python scripts/pinwheel_gap_walkthrough.py
code/terralingua/.venv/bin/python scripts/logical_memory_audit.py
code/terralingua/.venv/bin/python -m unittest discover -s code/hou_memory -p 'test_*memory*.py' -v
code/terralingua/.venv/bin/python code/hou_memory/test_hou_memory.py
code/terralingua/.venv/bin/python code/hou_memory/test_memory_graph.py
code/terralingua/.venv/bin/python code/hou_memory/smoke_associative_sim.py \
  --output output/validation/my-smoke --sbert
```

After the first MiniLM download, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to reproduce with cached weights. The smoke runner's `--sbert` mode expects that cache. Use a new output directory for each smoke run.

On 2026-09-28, all **51 existing tests** passed: 22 current mechanism/integration tests, 14 kernel checks and 15 historical graph checks. The actual TerraLingua smoke completed 12 ticks for two agents, restored memory and graph state at tick 8, and used real local MiniLM embeddings with scripted generation and **zero generation API calls**. The pinwheel thresholds and audit counterexamples were reproduced. These validate mechanics and integration; the pinwheel action prediction, held-out usefulness and human believability remain untested.

## Presentation changes in v12

- Replaced the bell ranking example with Mia's actual help beside a pinwheel and her understated remark, "I'm usually here."
- Kept the accepted thesis and both hypotheses. Removed the extra disclaimer line from that slide; evaluation remains on its own slide.
- Preserved the existing mathematical explanation and added one editable recall-window chart immediately after it. Both plotted curves are final scores.
- Kept the separate connection-rehearsal chart, models, experimental conditions, audit UI/GIF and sources.
- Updated related backup answers and speaker notes. Main slides use proposal/illustration language, while this document distinguishes current implementation from future evidence.

Historical ACT-R and broader graph designs remain in the repository as context; they are not the work order for this experiment. Read [HANDOFF.md](../HANDOFF.md) for the current entry points and [INDEPENDENT_GRAPH_MEMORY_DESIGN.md](INDEPENDENT_GRAPH_MEMORY_DESIGN.md) for the implemented math and its design history.
