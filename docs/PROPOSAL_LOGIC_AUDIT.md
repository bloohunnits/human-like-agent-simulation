# Proposal logic audit, 2026-09-28

The v12 presentation uses a new Mia–pinwheel illustration. See [the team catch-up](TEAM_CATCHUP.md) for the verified thresholds and the clarification that weighted lookup equivalence preserves relational information. The findings below remain applicable.

The implemented equations are internally consistent with independent associative access. The v10 presentation nevertheless allowed stronger conclusions than its evidence supports. The corrected v11 presentation keeps the accepted thesis and H1/H2, but separates a possible retrieval mechanism from useful context and believable behavior.

No production scoring, selection or learning rule changed in this audit. The limits below remain properties of the current model. Calling the deck logically coherent does not establish that its proposed model is effective or psychologically valid.

## 1. What the arithmetic actually guarantees

The direct contribution is `M = r exp(-age_memory / (tau_memory * g))`. An eligible connection contributes its own independently aged support, including the hub crowding factor. The final score is `F(max(M, G))`, where `F(x) = (1-exp(-x))/(1-exp(-1))`.

- For an identical frozen memory state, enabling nonnegative graph support cannot reduce an individual memory's score. This follows from `max(M,G) >= M` and monotonic F.
- That does not guarantee inclusion in the prompt. Other memories can gain more and consume the limited slots or tokens.
- A graph route can recover a memory whose direct access has faded. The stored text remains intact. This models accessibility, not recovery of erased content.
- Edges have their own finite decay clocks. They are independent of the target's forgetting factor, not immune to time. An extensively rehearsed target memory can outlast an unused edge.
- If an edge's current transmission is z, renewal sets `w_new = z + eta*(1-z)` and increases or preserves h. With fixed topology and no subsequent events, its transmission cannot be lower at any future delay than the unrenewed edge's transmission. The stored w may still be below its original value because z already includes decay.
- In this implementation p is a deterministic retrieval score. A score of .216 is not a measured 21.6% human recall probability or a Bernoulli sampling probability.

The visible slide equation now explicitly names its uncrowded, single-edge simplification. The general equation retains the crowding factor. The chart now states that it holds the graph fixed and plots separate probes, rather than a continuing simulation that writes new episodes.

## 2. Max can remove useful semantic distinctions

Once G is greater than M, changing direct relevance while M remains below G does not affect the final score. Two episodes with the same G can receive identical scores despite very different semantic relevance. Exact ties use stable memory IDs.

This is the same freedom that permits an indirectly relevant episode with a weak semantic score to return. It can also admit an unrelated reminder. The rule cannot guarantee context-appropriate selection.

The audit reproduced a counterexample using the default three-slot budget, four sequentially created episodes, and controlled embeddings. Scenario relevance is assigned before scoring:

| Episode | Cosine relevance | Direct-only score | Score with graph |
| --- | ---: | ---: | ---: |
| Relevant episode | 1.000 | .289218 | .297374 |
| Irrelevant episode 1 | .100 | .032896 | .300059 |
| Irrelevant episode 2 | .100 | .034224 | .302766 |
| Irrelevant episode 3 | .100 | .035605 | .305494 |

The direct condition returns the relevant episode. Graph access raises every score, but the three irrelevant episodes take all three slots and exclude the relevant one. This is a counterexample to a guarantee, not an estimated frequency of failure on real language.

**Implication:** H1 must measure relevant retrieval together with false reminders and displacement. H2 must measure the persistence of useful and distracting reminders. A higher score or a longer-lived edge alone cannot confirm either hypothesis. Keeping max is coherent as a candidate rule; its selection tradeoff remains unvalidated.

## 3. The current crowding rule can block a durable connection

For a hub with n retained episode links, the code uses `f(n)=1` for n <= 3, otherwise `f(n)=1/(1+ln(n/3))`. It counts all retained links, including faded ones. The simulation normally adds one episode per turn, so repeated visits can grow this count.

At beta=.5, w=.8, h=100 and edge age=100, the same target connection produces:

| Episodes attached to the cue | Graph score | Above .15? |
| ---: | ---: | --- |
| 1 | .216473 | Yes |
| 3 | .216473 | Yes |
| 4 | .170836 | Yes |
| 5 | .146816 | No |
| 10 | .102174 | No |

These fixtures hold edge ages equal to isolate crowding. They are not live simulation trajectories.

The bound is stronger at high counts. The .15 cutoff corresponds to a pre-F contribution of .099619343. With beta=.5, a hub containing **167 or more retained episodes** supplies insufficient graph support even when the target edge has w=1 and zero age. This applies to retrieval through that hub alone under the current defaults. Another cue or the direct route can still retrieve the memory, and changing beta, cutoff or crowding changes the bound.

**Implication:** the isolated persistence chart is correct, but it cannot establish persistence in a growing simulation. The experiment must vary hub size, episode frequency and distractor load. Activity-weighted crowding, pruning or a different fan function would be a new model choice to evaluate; none was silently implemented here.

## 4. The bell example does not establish semantic incapacity

The actual production observation query and MiniLM encoder rank three bell conversations above Mira's offer of food. With three similarity-only slots, the offer is excluded. Its full memory text includes the bell, and its cosine score is .334174 rather than zero.

Adding `Current goal: find food.` to the query moves the offer from fourth to third. At the separate 100-tick delayed probe, ordinary decay still pushes all direct scores below the cutoff. These distinguish a ranking issue from a forgetting issue. The goal-aware query is currently an evaluation variation, not the default simulation query.

The graph also uses structured episode tags. The baseline readable text contains the observed artifact name, but cosine similarity does not explicitly apply the same observed-entity matching rule. Any advantage therefore belongs to the specified association policy over this baseline, not to graphs over every semantic retriever.

**Correction:** v11 makes query dependence visible on the example slide. Goal-aware queries and slower ordinary forgetting are proposed comparison checks. Apply the same expanded query to each treatment when isolating the association mechanism. Do not choose the query merely to obtain the desired ranking.

## 5. Graph structure is not necessary for the primary condition

With `memory_sources=False`, current observed entity cues activate direct entity-to-episode links. A dictionary mapping each cue to episode links, with the same edge weights, clocks, crowding and max rule, gives exactly the same G. The audit independently implemented this lookup representation and reproduced production G across seven controlled fixtures.

**Defensible contribution:** an independently aging and rehearsed cue-to-episode memory policy for simulated agents. The graph represents that policy and supports optional extensions. This main experiment does not establish that multi-hop reasoning, graph topology beyond those associations, or graph storage itself is necessary or superior. Optional memory-source routes remain outside the main conditions.

## 6. Encoding, retrieval and action are different events

- Current graph sources come from structured observation, message senders, inventory and configured visible places. Merely mentioning an absent object in free-form speech does not activate that object's hub.
- Stored episode tags can also name an intended action target. For example, choosing to create a bell can tag the bell even before the action succeeds. That is an association to an intention, not evidence that the bell was observed or successfully created. The episode text explicitly records a chosen action. The bell example itself uses an observed artifact.
- Entity identities are normalized names, not physical object tracking or a learned identity resolver. Controlled examples require unique names; duplicate names are a failure case.
- “Selected,” “returned” and “surfaced” refer to the same engineered event: including memory text in the prompt after the cutoff, count and token rules. The code commits rehearsal for that retrieval before invoking the language model. Retries reuse the event. A generation failure does not automatically roll it back.
- Every included memory receives ordinary rehearsal. Its graph path learns only when its contribution exceeds direct access. That is credit for retrieval, not reward for correct behavior or proof that the LLM attended to the memory.
- The bounded plan can still carry historical information despite its instructions. Its actual content must be audited when comparing recall conditions.

The presentation now distinguishes prompt inclusion, context-appropriate actions and human ratings of believability. None implies the next automatically.

## 7. Experimental logic that remains defensible

Keep the approved hypothesis wording. Define its measurements before testing:

1. **H1:** across prespecified histories and delays, compare relevant recall with the semantic-plus-decay baseline at the same memory/token budget. Label episode relevance before scoring and count false reminders and displacement. Check a goal-aware query and a more slowly forgetting direct baseline before attributing every benefit to associations.
2. **H2:** provide the same ordinary recall events to both copies, update links in only one, and probe after increasing gaps. Count useful and irrelevant persistence. A read-only graph-off probe should recover equal direct scores for these controlled copies.
3. **System effect:** run complete simulations separately. Their later actions and histories may diverge, so they no longer isolate only one edge update. Switching the graph off at the final tick cannot undo earlier graph-induced ordinary rehearsal or plan contents.
4. **Believability:** blind raters to condition and show enough prior experience to judge context fit. Use appropriate decisions and plausible remembering as distinct judgments. Model scores cannot serve as ground-truth relevance labels.

Use development cases to choose settings and an acceptable false-reminder tradeoff, then freeze them before held-out evaluation. Equal-setting mechanism comparisons and separately tuned stronger baselines answer different questions and should be reported separately. A constructed success demonstrates possibility. A constructed failure refutes a universal guarantee. Neither estimates average effectiveness.

## Evidence and reproduction

- Current formulas, crowding, selection and updates: [associative_memory.py](../code/hou_memory/associative_memory.py).
- Actual prompt assembly, observation query and episode encoding: [memory_agent.py](../code/hou_memory/memory_agent.py).
- Bell example and query variations: [proposal-examples-v10.json](../output/analysis/proposal-examples-v10.json).
- Counterexamples and bounds: [logical-audit-v11.json](../output/analysis/logical-audit-v11.json).
- Reproduce the audit from the repository root: `code/terralingua/.venv/bin/python scripts/logical_memory_audit.py`.
- Reran the 22 current mechanism and actual-agent integration tests: all passed. Ten thousand fixed-seed numerical samples also satisfied score monotonicity and fixed-topology edge-renewal invariants. These checks establish properties of the implementation, not human validity.

The original proposal motivates MiniLM using sparser similarity scores. That is a model-selection rationale from the supplied source, not evidence that MiniLM fails this task or that a sparse score distribution guarantees better associations. No global novelty or general agent-performance claim is established here.
