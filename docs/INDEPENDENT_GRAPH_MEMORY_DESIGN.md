# Independent associative retrieval: design and walkthrough

Status: implemented in the live simulation on 2026-09-27. See [the final logic audit, implementation details and run commands](ASSOCIATIVE_IMPLEMENTATION.md). The numerical examples below remain deterministic illustrations, not results from human ratings or agent efficacy trials. Presentation narration may stay in proposal tense; the code now implements these mechanisms.

The [2026-09-28 proposal logic audit](PROPOSAL_LOGIC_AUDIT.md) records additional counterexamples and claim boundaries. In particular, max can remove semantic ranking differences, the all-retained-link fan penalty can block retrieval through a large hub, and the primary one-hop condition is equivalent to a weighted cue-to-episode index. The core formula is unchanged. Presentation v11 now makes its fixed-graph examples and unvalidated selection tradeoffs explicit.

## 1. Purpose and decision

The thesis is to increase the believability of multi-agent simulations by modeling human-like remembering and forgetting. The target behavior is an old experience returning in a specific context, along with selective persistence, ordinary forgetting, and plausible lapses. General task performance and efficiency are not the claimed contribution.

Recommended candidate: give direct retrieval and associative retrieval separate forgetting processes, then choose their larger contribution. Preserve the current Hou implementation as a comparison. Start with associative recovery and connection rehearsal. Leave indirect strengthening of unselected memories off in the primary experiment.

This architecture produces the desired possibility by construction. Whether its resulting selectivity, forgetting curves, and simulation behavior are plausible remains an empirical question. A hand-picked memory crossing a threshold cannot establish human likeness.

## 2. What changes, and why

The historical console prototype uses:

\[
D_i(t)=\exp[-(t-t_i)/(\tau_m g_i)],\qquad
p_i^{old}=F(\max(r_i,a_i)D_i),\qquad
F(x)=\frac{1-e^{-x}}{1-e^{-1}}.
\]

Here r is the cue-to-memory embedding similarity, a is graph support, and D is the target memory's own decay. Both cue routes inherit the same D. With fixed g and bounded cues, an old target eventually becomes very hard to retrieve regardless of the strength of its association.

The candidate uses:

\[
M_i=r_iD_i,\qquad G_i=\text{cue-dependent support through separately aging links},
\]
\[
x_i=\max(M_i,G_i),\qquad p_i=F(x_i).
\]

**Do not multiply x by the target memory's decay again.** That would recreate the problem. Graph support now competes with the already-decayed direct contribution, so G can matter even when G is smaller than raw r.

F is a monotonic score mapping inherited from Hou. It preserves ranking: selecting by x or F(x) produces the same order. Applying this mapping to an independently constructed graph contribution is our extension, not Hou's published equation and not a calibrated human recall probability.

## 3. State and units

Each memory stores a stable ID, original text, embedding, creation time, last full-recall time t_i, and consolidation strength g_i. Use the existing time normalization tau_m. The direct forgetting time scale is tau_m times g_i, in simulation ticks.

Each undirected edge stores one stable ID, endpoints, relation type, creation time, last qualifying reinforcement time t_e, stored transmission weight w_e in [0,1], and retention time scale h_e in ticks. Both adjacency directions must refer to the same state. Persist this state across save/load; rebuilding edges must not silently renew them.

At read time, edge transmission is:

\[
L_e(t)=w_e\exp[-(t-t_e)/h_e].
\]

w is the edge's stored transmission immediately after creation or reinforcement. h controls how quickly that transmission fades. At an age of h ticks the remaining transmission is w/e, not one half. The half-life is h ln(2). All h values are positive and have a finite configured cap.

Independent decay does not guarantee recovery after an arbitrarily long gap. If both pathways have fallen below the selection boundary, a cue alone cannot restore an absent association. Very long retention requires a long-lived or rehearsed link, and special salience rules would need a separate rationale and test.

There is no universal requirement that every connection outlast every memory. The intended example compares a durable association with a neglected memory. Repeated full recalls can make a memory's own time scale larger than an edge's.

## 4. What activates the graph

An edge contributes only when a current source is active. The mere existence of an edge provides no constant bonus.

For the first controlled study, use explicitly observed/tagged entities as sources. Seeing the well gives that entity source c=1. An absent well gives c=0. Use stable IDs or unambiguous tags rather than guessing from substring matches. The live wrapper now accepts configured visible place annotations alongside observed people and artifacts; no labels are inferred from hidden locations.

After the entity-only test, add memory sources in a separately declared condition. A bounded set of direct candidates can supply c_j=M_j from the same frozen pre-retrieval state. Require a seed cutoff and fixed source budget, and apply exactly that policy to both formula variants. Do not recursively use graph-boosted memories as fresh sources in the same call. Merely being an internal seed does not count as full recall.

This planned seed rule differs from the prototype's use of raw direct similarities for all memories. Do not mix the seed-policy change into the first formula comparison. The entity-only fixture isolates the equation placement without that confound.

Allowed initial routes:

- Observed entity to one memory: one mentions edge.
- A direct memory source to a temporal neighbor: one same-tick or adjacent edge.
- A direct memory source through one entity hub to another memory: two mentions edges.

No repeated node within a route, no self-return, and no recursive diffusion. The historical prototype treats a shared hub as one damped transfer using the receiving membership weight. The candidate explicitly multiplies the independently aging physical edges on a two-edge route. That route change must receive its own comparison; the first entity-only test avoids it.

## 5. Graph contribution and combination

For a route P from source j to target i:

\[
A_{P,i}=c_j\left(\prod_{e\in P} L_e(t)\right)f_P,\qquad
G_i=\beta\max_{P\to i} A_{P,i}.
\]

If no eligible route exists, G=0. f is a crowding penalty applied once at a traversed entity hub, otherwise 1. A starting comparison can reuse the existing penalty: 1 for at most three hub members, and 1/[1+ln(n/3)] above three. It is a chosen approximation, not a fitted human interference curve. Count only eligible existing records and keep topology identical across conditions.

Beta is a global graph budget in [0,1]. Apply it once per allowed route. Edge weights and the route length already attenuate the signal; do not also apply the legacy lambda accidentally. Every c, L, and f lies in [0,1], so G <= beta and x <= 1. Consequently 0 <= p <= 1 without a final repair clamp hiding errors.

The two maxima have separate roles: one chooses the strongest route to a target; the other chooses direct versus associative contribution. Max avoids counting correlated routes repeatedly. It ignores convergent evidence from multiple weak routes, which is a deliberate limitation. Adding after decay is another possible design, but additive support before the same target decay would not fix the original problem.

For a bounded additive sensitivity test later, x=M+G-MG is an option. This is a saturating combination, not an assertion of probabilistic independence. It changes the model's behavior, so keep it outside the main candidate initially.

## 6. Selection and forgetting

Require p > theta, then return at most k eligible memories under a fixed retrieved-token cap. Allow fewer than k or an empty set. Break ties by stable ID. A memory receiving graph activation has only become a candidate; it enters the agent's prompt only if it passes these selection rules.

The earlier live wrapper used theta=0 and k=5; the implemented default now uses theta=.15 and k=3, with a 600-token retrieved-text budget. With theta=0, arbitrarily tiny positive values can fill slots when competition is weak. Therefore a nonzero cutoff is necessary for the proposed operational definition of forgetting through threshold failure. Tune the cutoff on development cases, then freeze it for all matched conditions. The example uses theta=.15 and k=3 purely for illustration.

With no cue, no eligible route, beta=0, or all edges disabled, the new score exactly equals bare Hou for the same memory state. This is an instantaneous reduction. To reproduce the entire baseline trajectory, disable graph learning and partial refresh from the start and use the same selection/prompt settings; earlier graph-induced recalls may already have changed memory state.

## 7. Updates happen only after selection

First compute all direct and graph scores from a frozen state. Select and record the winning paths. Only then commit updates, once per event. A read-only preview never changes state.

### Full memory recall

Every memory actually returned receives the existing Hou update:

\[
g_i\leftarrow g_i+\tanh((t-t_i)/(2\tau_m)),\qquad t_i\leftarrow t.
\]

This increases strength and resets the last-recall clock. Returning a memory to the prompt counts as recall in this engineering model; it does not prove the language model attended to it or used it correctly.

### Connection rehearsal

Qualify a winning route only if its target was selected and G_i > M_i using pre-update values. On a tie, credit the direct route. Update each distinct edge on the qualifying winning route once per committed event, even if several selected targets share it. Store alternate-route and counterfactual selection information in the audit trail. Winning the max does not necessarily mean that the edge changed top-k membership.

A simple, bounded candidate rule is:

\[
z_e=L_e(t),\quad
w_e\leftarrow z_e+\eta_w(1-z_e),\quad
h_e\leftarrow\min(h_{max},h_e+\delta_h),\quad
t_e\leftarrow t.
\]

Use the **currently decayed** z, not the old stored w. Otherwise resetting the edge clock would silently restore old transmission in addition to the declared learning increment. eta_w is in [0,1]; delta_h is a nonnegative number of ticks. Weight increases current support, while h increases durability. Compare renewal of w at fixed h against renewal plus h growth to identify which contributes to H2.

Creating a new edge initializes it at the encoding event. If an actual new co-encoding of both endpoints strengthens an existing edge, treat that as an explicit logged learning event. Merely viewing an entity, scanning a route, or previewing a score must not renew all its connections. A long-unused association can therefore become too weak to retrieve its target and require new joint exposure to recover.

### Optional indirect memory refresh

Keep this off in the primary design. It is the older additional rule that raises g for an activated but unselected memory, without resetting t_i. If studied later, specify its activation threshold, strength fraction and once-per-event rule separately. Test its later effect with graph support disabled, since that isolates changed memory strength. A positive result for this optional rule is different from learned connection persistence.

## 8. Walkthrough A: an old experience returns

Stored experience at tick 0: “Met Omar at the well. He showed me the berry patch nearby.” No later full recall has occurred. At tick 100, the current context names the well. Assign illustrative direct similarity r=.20, g=1, tau_m=25, beta=.50, w=.80, h=100, link last reinforced at 0, current entity source c=1, and no crowding penalty.

These are chosen inputs for an explanation, not measured SBERT scores or fitted human parameters. Machine-readable calculations are in `output/analysis/independent-graph-walkthrough.json`.

1. The memory's ordinary contribution is .20 exp(-100/25) = .003663.
2. The surviving connection contributes .50 x 1 x .80 exp(-100/100) = .147152.
3. The proposed max chooses .147152, giving p=.216473.
4. Bare Hou gives p=.005784. The legacy graph rule with its non-decaying raw support .40 gives p=.011548.
5. At theta=.15, only the proposed pathway clears the example cutoff. With three competing eligible memories scoring .32, .24 and .18, it also takes the third slot and displaces .18.

The legacy comparison changes edge-aging semantics as well as placement. For the clean placement experiment, feed the **same aged support G=.147152** to both formulas. The pre-decay version is F(max(.20,.147152) exp(-4))=.005784, while the separate pathway remains .216473. This is the causal comparison of placement. Show the historical prototype only as an additional reference.

What if the context does not activate any eligible route? c=0, G=0, and the score returns to .005784 for a frozen-state probe. What if nobody rehearses the edge for 300 ticks? G=.019915 and p=.031193, below .15. The association lasts longer in this example but still fades.

These alternative probes use copies of the same history and do not commit recall. After a real successful retrieval at tick 100, the memory's own g and clock change. A subsequent cue-absent trial could therefore score higher. That is ordinary rehearsal after recovery, and must not be mislabeled persistent graph activation.

## 9. Walkthrough B: a used connection lasts longer

For H2, start with identical stores and links. At tick 60, both conditions retrieve Omar through the well. Both apply exactly the same full memory recall update, giving g=1+tanh(1.2)=1.833655 and last recall time 60. Only the learning condition updates the winning edge.

Before that update L=.80 exp(-.6)=.439049. With eta_w=.20 and delta_h=25, the learned edge becomes w=.551239, h=125, last reinforcement time 60. The comparison edge retains w=.80, h=100 and time 0. Because those weights refer to different timestamps, compare their current decayed transmission rather than the stored numbers alone.

At tick 150, both conditions use the same well cue and the same node history:

| Probe | Edge learning off | Edge learning on |
| --- | ---: | ---: |
| Direct-memory score in both | .043803 | .043803 |
| Score with associative route | .135077 | .198615 |
| Clears illustrative cutoff .15 | No | Yes |
| Score with graph disabled | .043803 | .043803 |

This isolates connection persistence from the target's ordinary rehearsal. At the H2 final probe, the graph **must be enabled** to expose a learned connection. Graph-off equality is the negative control. In earlier v7 slides, H2 meant partial refresh of g and therefore required a graph-off final probe. The proposed main H2 now concerns edge learning; mixing those probe rules would invalidate the interpretation.

A later end-to-end run may produce different recall histories and g values across conditions. That total system effect is meaningful but does not isolate edge learning. Report it separately from matched-history replay.

## 10. Mechanism predictions and empirical questions

**Thesis.** Associative memory should give simulated agents more appropriate context for their decisions, making their behavior more believable and consistent with their own experiences.

**H1: associative recovery.** Associations will help agents retrieve past experiences relevant to their present situation, including experiences that semantic similarity and time decay alone would miss. Compare the direct-only baseline with associative access at the same prompt budget. Vary the delay and competing memories. At each delay, compare copies of the same memory state so a difference in ordinary rehearsal cannot explain the effect. Time remains a variable across the experiment; it is only equal within each paired comparison. Define an acceptable false-reminder tradeoff before held-out evaluation. Measure relevant retrieval, irrelevant reminders and displacement from the limited prompt. Individual score increases are mechanism checks, not evidence of believability.

**H2: lasting accessibility.** Recalling experiences through their associations will help preserve relevant context across longer gaps. Give both copies the same ordinary full-recall events while only one updates its connections, then vary the gap before a common final cue. Increased transmission is built into the rule; its useful selectivity is not. Separate w renewal from h growth in later sensitivity tests, include a graph-off negative control, and test whether unused links still fade.

With common inputs and decay D in [0,1], max(rD,G) is always at least max(r,G)D. The candidate therefore guarantees a nondecreasing score for each memory under the matched placement comparison. This does **not** guarantee that a particular relevant memory is selected: its distractors can gain more and compete for the same k slots. The empirical question is selective recovery across held-out histories, including false reminders, rather than whether the new formula can increase a score. Connection rehearsal also increases transmission by construction, so H2 must evaluate how long that helps relevant associations and whether it preserves irrelevant ones.

**Believability.** In blinded comparisons of matched simulation scenes, human raters should judge whether remembering, lapses and context-triggered reminders fit the character's experience. Define anchors before collecting judgments. Keep generation model and prompt budgets fixed within a comparison, randomize presentation order, and report uncertainty and rater agreement. Do not treat a numerical retrieval gain as proof of believability or human cognitive validity.

Expected limits: common cues can produce irrelevant reminders; linked memories may crowd out recent experiences; repeated recall can create a feedback loop; exact full-text replay can feel less human-like even when access timing improves. Store and report these failures, not only successful revivals. This model changes accessibility, not loss of detail or reconstructive memory. It is not a clinical model.

## 11. Comparisons and calibration

Main proposal conditions (presentation v10):

1. Hou with no graph.
2. Independent pathway with edge aging and learning disabled (clocks age, no renewals).
3. Independent pathway with winning-route edge learning.

The implemented `gated` preset and the old non-decaying-edge prototype remain architecture-history comparisons. Their equations and walkthroughs above document the design iteration; they are not separate conditions in the class proposal. Use shuffled links preserving relevant topology/degree as far as feasible, wrong/absent cues, isolated targets, crowded hubs, and many distractors. Optional later conditions: weight renewal only, no edge aging, partial memory refresh, alternative source policies, bounded addition. Avoid making every sensitivity condition a separate main-slide “arm.”

Illustrative starting values are tau_m=25 ticks, g0=1, beta=.5, w0=.8, h0=100 ticks, hmax=200 ticks, eta_w=.2, delta_h=25 ticks, theta=.15 and k=3. These are not recommendations inferred from human data. Tune beta, h0 relative to tau_m*g0, cutoff and rehearsal rates on development histories. Freeze them for held-out histories. Changing h changes a timescale, not evidence of a new mechanism; the pre/post placement comparison must reuse the same h values.

Threshold feasibility matters. Since p <= F(beta) when the graph is the only contribution, a cutoff at or above F(beta) prevents graph-only rescue. In this example F(.5)=.62246, above .15. A nonzero threshold plus finite h eventually prevents recall from an unused link. With a crowding penalty or weak multi-edge route, the maximum attainable score can be much lower.

## 12. Implemented work order

1. Introduce a versioned edge schema with stable IDs, w, h, t_e and serialization. Reuse the original stored memory text and embedding. Keep graph state across restart.
2. Implement pure scoring modes for Hou, matched pre-decay graph, and independent graph. Require common sources and topology for the placement test. Return route IDs and score components.
3. Separate score/preview from commit. Update memory state and eligible winning edges once per committed retrieval event. Reject future-dated records and invalid parameters. Use stable ties.
4. Implement fixed-history fixtures and a deterministic trace first. Verify exact graph-off reduction, boundedness, no-cue behavior, eventual edge forgetting, no self-boost, no multi-hop cycles, read-only previews, idempotent commits and persistence equivalence.
5. Add matched memory-history H2 trials. Compare graph-on results and graph-off equality. Keep indirect memory refresh disabled until tested independently.
6. The simulation wrapper now calls this engine. It uses structured observed entity/place cues and still writes one episode per agent turn; same-tick sub-event writing is not enabled.
7. Extend the audit UI with M, G, x, p, cutoff and rank, winning source/route, memory g and clock, each edge's w/h/age, selection outcome, counterfactual graph-off selection, and before/after committed changes. Log the exact prompt and resulting action. The UI supports interpretation and debugging but does not prove causal language-model use.

Read-only views must be read-only even when a cue appears repeatedly. Replaying an event ID must not double-reinforce. Connecting an old memory to a new record must not renew all of the old memory's other edges. Disabling graph scoring only at the last step is not equivalent to never having used the graph.

## 13. Model choices and source boundaries

SBERT all-MiniLM-L6-v2 follows the supplied proposal's rationale: relatively sparse similarity scores, plus local execution. The sparsity claim concerns similarity scores on the cited setting, not sparse embedding coordinates or a universal guarantee on our data. GPT-5 Nano supplies actions/messages. A locally available Qwen model is a lower API-cost collection option, not a required robustness claim. Use one generation model consistently within each treatment comparison.

The new max placement, exponential edge decay, learning rule and illustrative constants are project hypotheses. The supplied proposal and Hou paper motivate dynamic memory; research on associative interference motivates cue specificity and crowding. None establishes this exact new equation as a validated human-memory law.

## 14. Presentation illustration and the similarity objection

The v10 proposal replaces the well/berry example with a constructed bell/food-offer example. The current observation contains a visible named brass-bell artifact. Earlier, Mira offered to save food while that artifact was visible. Three other memories discuss bells without observing one. The actual episode writer includes the visible artifact name in Mira's text and supplies its structured observed-entity tag. It supplies no bell tag for conversations that merely mention an absent bell.

With the current MiniLM encoder and production query formatting, the three conversations score .453966, .450630 and .397777. Mira's offer scores .334174, placing fourth. Thus three slots selected by semantic similarity alone exclude it. This is a ranking argument, not an assertion of zero similarity. At the separate delayed probe at tick 100, Mira's direct score is .009653 and its independently aged graph score is .216473. With the default .15 cutoff, k=3 and 600 retrieved-text tokens, the graph returns the offer and direct-only access returns none. The graph contributes through the explicit observed bell, using the current default entity-only source policy.

The complete strings, creation times, configuration and route snapshots are in `output/analysis/proposal-examples-v10.json`. `scripts/historical_bell_walkthrough.py` recreates them with cached MiniLM and no generation API. It also recreates the connection-learning chart using identical ordinary recalls in both copies.

This example deliberately illustrates a possible failure of the specified baseline. It does not estimate prevalence or establish that every vector-based retriever fails. Different wording, more slots or a different encoder may change the ranking. A richer text system could extract observed relations or represent connection rehearsal history, so it is a possible additional baseline. The graph's use of structured observation metadata is an architectural choice, not hidden proof of semantic incapacity. Held-out experiments must include paraphrases, negative cases and competing relevant memories.

The reproducible query-sensitivity checks make this concrete: appending `Energy: 25` or `I need food.` leaves the offer fourth, while `Current goal: find food.` moves it to third. With that last query, semantic-only top-3 would include it. At the delayed probe ordinary decay still leaves all direct scores below the cutoff. Test goal-aware queries as a stronger baseline rather than presenting this example as a universal limit of semantic retrieval.

Sources:

- Supplied course proposal: `/Users/marshall/Downloads/proposal473_673.pdf`.
- Current kernel: `code/hou_memory/hou_memory.py`.
- Historical graph and broad incident-edge reinforcement: `code/hou_memory/memory_graph.py`.
- Historical graph console scoring and new read-only live audit: `code/hou_memory/webapp.py`.
- Current simulation path and entity writer: `code/hou_memory/memory_agent.py`.
- Hou, Tamoto & Miyashita, CHI EA 2024: https://doi.org/10.1145/3613905.3650839.
- Honda et al., HAI 2025 proceedings: https://doi.org/10.1145/3765766.3765803.
- Anderson & Reder (1999), *The Fan Effect: New Results and New Theories*: https://www.andrew.cmu.edu/user/reder/publications/99_jra_lmr_2.pdf. Supports associative competition as motivation; does not validate our bounded max and edge update rule.
- TerraLingua: https://arxiv.org/abs/2603.16910.
