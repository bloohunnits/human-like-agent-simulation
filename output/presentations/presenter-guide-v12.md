# Presenter guide — relational memory v12

Start with [the team catch-up](../../docs/TEAM_CATCHUP.md). It explains the pinwheel example, score scales, cutoff crossings, implementation and remaining experiments. The following are the actual speaker notes from the final PPTX. Main slides use proposal language.


## Slide 1: Relational memory

Our goal is a simulation whose agents remember and forget in more believable ways. We propose a separately fading associative pathway, so a context can bring back an old experience even when ordinary retrieval is weak. The connection also fades and can strengthen through use. We will test the mechanism first and assess simulation believability separately.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/proposal/proposal.pdf

Hou, Tamoto & Miyashita. CHI EA 2024. https://doi.org/10.1145/3613905.3650839

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 2: The problem: human-like remembering

The supplied proposal makes believability the purpose of the memory model. A place, person or event can act as an associative cue even when an old episode has not been recalled recently. We want selective reminders and plausible lapses rather than maximum recall of everything. The exact equations and learning rules here are project hypotheses. This project does not establish a general improvement in agent intelligence, efficiency or task performance. Human ratings will assess simulation believability; mathematical retrieval examples only establish what the proposed mechanism can do.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/proposal/proposal.pdf

APA Dictionary of Psychology, Associative Memory. https://dictionary.apa.org/associative-memory

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 3: Example: remembering where Mia helped

Mia gives the agent food beside a uniquely named blue pinwheel and says only “I'm usually here.” She never instructs the agent to return for food. Much later, the hungry agent sees the pinwheel without seeing Mia or food. Recovering the earlier episode supplies grounds for a reasonable inference: Mia helped before and might be found here again. The expected action is to approach and look for her; it does not assert that food or Mia is currently there. A different personal history could support another action. The relation connects the observed artifact to the episode; it is not a trained reward value or a universal pinwheel-means-food rule.

The example is a constructed one-episode fixture, scored with the actual MiniLM encoder, episode writer and production retrieval engine. Food-transfer feedback is supplied by the fixture, not verified by a world rollout. The approach behavior has not been tested with an LLM. The stored text includes blue pinwheel and food, so semantic similarity is positive. With hunger explicitly appended to the query, raw cosine is .3702016175. At 100 ticks, direct final score is .0106903 and associative final score is .2164730. The point is loss of access under similarity plus ordinary decay, not that vectors can never recognize the content.

The current live query normally includes observations, messages and optional information; energy reaches the action model separately. This fixture additionally includes “I am hungry. Current goal: find food.” as a stronger comparison. That query change is a fixture only, not a change to the simulation. Seeing a cue can trigger graph access today; hunger alone does not activate an unseen pinwheel. The later chart gives the exact accessibility window.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/memory_agent.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/output/analysis/pinwheel-walkthrough.json

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/scripts/pinwheel_walkthrough.py

## Slide 4: Thesis and testable hypotheses

The thesis is about appropriate context and believable behavior in a simulation. H1 asks whether associative retrieval recovers relevant episodes that the semantic-plus-decay baseline misses under the same prompt budget. H2 asks whether using those connections preserves relevant access across later gaps. Both require useful selectivity, not a higher score alone. Their directional score effects are designed into the rule. The empirical question is whether those changes recover useful context across held-out histories without excessive false reminders or displacement.

Time is a variable: sweep the delay between experience or last recall and the probe. At each delay, compare copies of the same state so one model does not get younger or stronger memories. For H2, give both copies the same ordinary recall events and vary only edge learning. A separate closed-loop simulation then lets histories diverge naturally to measure the whole system. Human ratings are necessary for the believability claim.

Operationally, test improvements across a prespecified collection of histories rather than selecting only successful examples. Label relevant episodes before observing retrieval. Count distracting reminders and displacement alongside relevant recall. Choose an acceptable false-reminder limit on development cases and freeze evaluation criteria before held-out testing. Prompt inclusion is not demonstrated use by the model, and context-appropriate behavior is not identical to human-like memory.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/proposal/proposal.pdf

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 5: Models and their roles

The primary reason for MiniLM in the supplied proposal is its relatively sparse similarity scores in Honda's setting. This concerns the distribution of similarity scores, not sparse vector coordinates or a guarantee on our own data. Local execution is another benefit. Qwen is a practical lower API-cost option for collecting results, not a mandatory cross-model hypothesis. Hold the generation model constant within each comparison. The exact locally available Qwen version is a deployment choice; the proposal and current router name different versions. We train no model weights, but still need labeled evaluation cases.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/proposal/proposal.pdf

Honda et al. HAI 2025 proceedings, published 2026. https://doi.org/10.1145/3765766.3765803

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/embedder.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 6: The memory’s own forgetting curve

Hou combines direct relevance with time since recall and memory-specific strength. Full recall adds tanh(elapsed/(2 tau_m)) to g and resets the recall timestamp. Memories can start with equal g and diverge through different recall histories. The graph extension retains this direct-memory pathway. The illustrated curves use r=.60, tau_m=25 ticks, no intervening recalls and g values 1, 2 and 4. The score mapping is monotonic and our implementation selects deterministically; the chart is not a human probability estimate.

Sources

Hou, Tamoto & Miyashita. CHI EA 2024. https://doi.org/10.1145/3613905.3650839

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/hou_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 7: A separate pathway for associations

This slide shows the one-edge, uncrowded case. On the slide t_i is elapsed ticks since the memory's full recall; the design brief uses t minus the last-recall timestamp. Delta is elapsed ticks since the edge's last qualifying reinforcement. tau_m converts ticks to Hou model time. g is memory-specific; w and h belong to the edge. c is current source activation, zero for an absent source, and beta is a global graph budget. All contributions are bounded. First decay each route separately, then take max and apply the monotonic Hou score mapping. Do not apply the target decay a second time. A weak positive connection does not count as evidence against retrieval. Max ignores it when direct access is stronger. This model has no inhibitory links; several weak cues also cannot add together. General routes and crowding are in the backup equations. This is a proposed Hou-based extension, not a published human-memory formula.

The visible equation is the uncrowded single-edge case. The current general rule also multiplies by f(n)=1 for n<=3 and 1/(1+ln(n/3)) otherwise. Each hub counts all retained episode edges, including faded ones. Once G exceeds M, changing direct relevance while M stays below G has no effect on this score. Max can therefore erase semantic distinctions among linked episodes. Its positive-support interpretation is internally consistent, but selectivity must be evaluated.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/hou_memory.py

## Slide 8: When the pinwheel can bring it back

All points are independent, read-only probes of the same unmodified memory. Both curves use the final retrieval score, not raw cosine or a measured probability. The hunger-aware query has r=.37020161747932434. M(t)=r exp(-t/25); G(t)=.5 × .8 × exp(-t/100). Both pass through F(x)=(1-exp(-x))/(1-exp(-1)); the associative condition uses F(max(M,G)). The node starts at g=1 and the edge at weight .8 and retention 100 ticks, with both clocks at zero. One active observed pinwheel route has no crowding penalty.

With p>.15 required, the equivalent pre-mapping cutoff is .0996193431. Ordinary access reaches the cutoff at 32.8172854 ticks and associative access at 139.0108193 ticks. Thus both qualify at integer ticks 0–32, only the association at 33–139, and neither from 140 onward. The native chart shades the continuous interval between the two threshold crossings. At tick 100 the direct score is .0106903 and graph-supported score is .2164730.

There is NO positive-time crossing of the direct and graph contributions in this example: G(0)=.4 already exceeds M(0)=.3702 and its retention time is longer. Solving equality gives a negative, out-of-domain time (-2.58056 ticks). Do not claim that the graph begins weaker and later overtakes direct similarity. The relevant transition is loss of prompt eligibility. Raw cosine .3702 stays constant for fixed texts; it must not be compared directly to the final .2165 score.

An unrehearsed direct route with even r=1 has final score only .02871 at tick 100 with g=1 and tau=25, so this fixture mainly illustrates the chosen forgetting timescales. A slower ordinary-decay baseline is needed to isolate association specificity in the experiment. No parameter or production query changes were made. Rehearsal, new hub members and competition would alter the displayed window and are separate experiments.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/hou_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/output/analysis/pinwheel-walkthrough.json

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/scripts/pinwheel_walkthrough.py

## Slide 9: Which memories enter the prompt?

Yes, selection includes top-K. The exact rule ranks by score, requires p strictly greater than .15, and returns at most three memories under a total 600 cl100k-base retrieved-text token budget. A too-long memory can be skipped so another eligible shorter memory fits. Fewer than three, including none, may enter the prompt. These are configurable starting values. The plan has a separate 60-token cap.

Every returned memory gets ordinary Hou rehearsal: its own g increases according to spacing and its last-full-recall timestamp resets. Its association learns only if G exceeds M. A direct-route tie gets no graph credit. Update the strongest winning route after selection and deduplicate shared edges once per event. A high-scoring candidate that fails the cutoff, count limit or token limit receives no rehearsal. Simply observing the pinwheel creates no renewal of every old pinwheel link.

Selected, returned, surfaced and fully recalled refer to the same prompt inclusion event in this implementation. To avoid ambiguity we say enters the prompt. A route can receive credit even if that memory would also fit without graph support. Prompt inclusion does not prove the LLM used the memory in its action, so the audit records the exact prompt and response.

The implementation commits rehearsal for the retrieval event before invoking the language model. Retries reuse that committed selection and cannot double-rehearse. The update is therefore credit for assembling the memory input, not a reward for a correct or successful action. A generation failure does not automatically undo this retrieval event. Later graph-off probes must use matched copies: earlier graph-induced recalls may already have changed ordinary memory strength.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/memory_agent.py

## Slide 10: Connection rehearsal and later access

This chart compares use of a connection within the proposed architecture. Both copies recall the episode at tick 60, so their ordinary memory strength g=1.833655 and last-recall timestamp=60 are identical. Both continue to age over the horizontal axis. Only the teal copy updates its winning graph edge at that recall. The direct relevance is .20 and all other starting settings match.

The edge starts at w=.8 and h=100 ticks. Before recall its decayed transmission is .439049. Learning sets w=.551239, h=125 and its edge clock to 60. The no-learning copy retains the original edge history. The chart starts immediately after that recall and shows the full retrieval score with the same bell cue. The associative path dominates both curves for these inputs. At a gap of 90 ticks, the scores are .135 without link learning and .199 with it. The direct-only score is .044 in both. At longer gaps both fall below .15. The plotted probes do not commit any additional recall. This demonstrates the mechanism, not an empirically established duration of human memory.

This chart proves only that the configured link update increases future support with fixed topology, cue and ordinary memory history. It does not establish useful context. Adding episodes to the same hub changes its crowd factor, and competing episodes can exclude this memory from the prompt even above the cutoff. Do not interpret the curve as a live trajectory where every point generates and stores another episode.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/output/analysis/proposal-examples-v10.json

## Slide 11: Experiment conditions

The three presentation conditions correspond to launcher presets hou, independent and learned. All are implemented. The main proposal explains one associative architecture with learning off or on; earlier formula-placement comparisons belong in the design and iteration log.

For H1, take the same memory state at each delay and compare direct-only access with associative access while link learning is off. The time gap varies across tests. For H2, provide equal ordinary recall events to both copies while only one updates connections, then probe with the same cue after increasing gaps. Disable graph support in a read-only final probe to confirm equal direct scores. Frozen replay isolates mechanism effects. Closed-loop simulations subsequently measure the total effect as actions and histories diverge.

Choose parameters on development histories and freeze them for held-out evaluation. Define relevant episodes from the scenario before scoring. Count irrelevant reminders and relevant episodes displaced from limited slots. Wrong cues, shuffled observed-entity links, hub crowding and paraphrases test specificity.

The primary study tests a memory policy. With memory_sources=false, a weighted cue-to-episode lookup table exactly reproduces the graph contribution. Multi-hop graph reasoning and a unique advantage of graph data structures are not tested. The graph-on comparison also changes the effective forgetting timescale, so include a slower-forgetting direct baseline and a goal-aware query as stronger checks before attributing every benefit to associations. Apply the same enriched query to both compared treatments. The simple weighted index is an equivalence control, not a different cognitive mechanism. Additional checks can stay out of the main three-condition table.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/run_hou_experiment.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 12: TerraLingua and the memory audit UI

TerraLingua provides the social simulation. The looping GIF is an environment example, not evidence for the proposed treatment. PowerPoint plays it in Slide Show mode; the PDF shows one frame. We plan an interface that makes retrieval decisions observable, traceable and debuggable: M, G, x, p, winning route, edge w/h/age, target g and clock, cutoff, rank, prompt inclusion and state updates. Log the exact prompt and action so we can investigate unexpected behavior. The UI cannot prove that the LLM used a particular memory. The source cue uses local observations and messages rather than the entire world's events. Keep plans and any recent-history buffer consistent between conditions. k limits memory count, so also cap retrieved tokens.

Sources

Paolo et al. TerraLingua (2026). https://arxiv.org/abs/2603.16910

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/memory_agent.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/webapp.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 13: Evaluation: memory patterns and believability

A worked score crossing verifies the mechanism, not human-memory validity. Define labeled histories and meaningful cues before inspecting outcomes. Include wrong cues, no cue, crowded hubs, old irrelevant events and misleading temporal neighbors. Measure relevant and irrelevant retrieval together under the same k and token budget. For believability, blind raters to condition, randomize presentation order and use anchored judgments of continuity, lapses and associative reminders. Keep the generation model fixed within comparisons and report uncertainty and rater agreement. Inspect whether actions reflect the relevant remembered context as a secondary behavioral measure. Ordinary task accuracy is a diagnostic, not the thesis. The model retrieves stored text exactly; content distortion and clinical trauma mechanisms are outside its claims.

Use three separate outcome levels: what enters the prompt, how the next action fits the agent’s available context, and whether independent raters find the remembering and behavior believable. A higher score proves none of these automatically. Raters should see relevant prior experience when judging context fit but remain blind to the memory condition. Similarity scores must not serve as the ground-truth relevance labels. The bounded plan can still carry old information, so audit plan content and match its rules across conditions.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/proposal/proposal.pdf

Kuroki et al. Shachi. https://arxiv.org/abs/2509.21862

## Slide 14: Proposed implementation and experiments

Build a versioned edge record and a pure scoring function before integrating the simulation. Check graph-off reduction, score bounds, absent-cue behavior, unused-edge forgetting, save/load, read-only previews and exactly-once updates. Then compare associative retrieval against the semantic-plus-decay baseline for H1, and connection learning on versus off with equal ordinary recall events for H2. Integrate stable place tags, source policies, memory granularity and token limits. Use the observed-entity route policy for the main experiments. Reserve additional mechanisms for later design work. The equations and examples describe the proposed system. Treat formula verification as a mechanism check and held-out selectivity as the experiment.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/memory_agent.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

## Slide 15: Full retrieval notation

The exact general route formula is G_i=beta max_P[c_source product_(e in P)L_e f_P]. If no route exists, G=0. Apply beta once per allowed route and a crowd penalty once at a traversed hub. On a route through two physical edges, multiply both aged transmissions. Use r, c, w, beta and f in [0,1], positive time scales, and no future-dated records. The proposed score exactly reduces to Hou with G=0 for the same memory state. F is monotonic, so it changes score display and threshold scale but not rank. To show functional forgetting, tune a positive cutoff; use the same positive cutoff and prompt budget in every condition.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/hou_memory.py

## Slide 16: What changes after a retrieval

Use pre-update scores for all ranking and route attribution. For the example eta=.2, delta_h=25 and h_max=200 ticks. These are illustrative starting constants. Every prompt-included memory receives g += tanh((t-last_recall)/(2 tau_m)) and last_recall=t. Its winning graph route learns only when G>M, with each qualifying edge updated once. A direct tie gets no graph update. Renewal starts from decayed transmission z, so it does not silently restore the old unaged weight. Alternate and unselected routes do not learn. A false reminder can reinforce its association, which is a failure mode to measure.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 17: Graph sources and allowed connections

Use observed entity cues for the main comparisons. Later, direct memory sources can use frozen M values and a fixed seed cutoff/budget. That differs from the prototype's all-memory raw-similarity sourcing and must be controlled separately. A shared-hub route multiplies its two separately aging physical links; the historical prototype used a simplified receiving-edge rule. Never let graph-boosted candidates become new sources during the same pass, and exclude self-return paths. Same-tick all-pairs edges only exist if multiple records share a tick. The writer stores one combined episode per agent turn. Place annotations must be configured explicitly and activated only when observed. The main condition does not use temporal or memory-source routes.

Current retrieval sources come from structured observation, message senders and inventory, plus configured visible places. Stored episode tags also include explicitly named intended action targets, such as trying to create a bell. That records an intention, not proof of a completed encounter or action. The main example uses an actually observed artifact, so it needs no change. Current IDs are normalized names, not a learned identity resolver. Memory-source routes are off in the main condition.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/memory_agent.py

## Slide 18: Time and the experimental controls

Matched does not mean holding time constant throughout an experiment. At a 20-tick gap compare the two copies at 20 ticks, then separately compare at 100 ticks, and so on. Similarly, H2 matches the target memory's g and full-recall timestamp because otherwise a direct-memory rehearsal effect could explain the difference. Connection histories are the intended difference.

In the chart both versions recall at tick 60. At tick 150 they have each gone 90 ticks without another recall. Their direct-only score is .044 in both, but graph-supported scores are .135 and .199 because only one connection was renewed. Controlled probes establish this mechanism. Complete simulation runs cannot keep the subsequent histories identical once the agents behave differently, so analyze those separately as whole-system effects.

Controlled replay uses matched ordinary memory states at each delay. This isolates mechanism effects but cannot model all later behavioral feedback. Separate full simulation runs let histories diverge. Changing only the final graph flag cannot erase earlier graph-induced recalls or plan contents. Report those full trajectories separately and label the fixed-graph chart as a mechanism illustration.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/output/analysis/proposal-examples-v10.json

## Slide 19: Expected limits and failure cases

An unused edge with finite h tends to zero, but a repeatedly qualifying link can remain accessible for a long time. That may be useful rehearsal or an undesirable loop. Track repeated and irrelevant retrieval rates, token crowding and competition with recent events. More recalls of everything would be a failure of selectivity. Max also ignores several weak cues agreeing; a bounded additive combination is a later sensitivity study. Fresh edge strength, beta and h must be calibrated together. The model can recover exact stored text even after a long delay, so it should not be presented as modeling reconstructive recall or emotional conditioning.

The audit reproduces a default-k=3 counterexample: the direct condition returns a scenario-relevant episode with cosine 1 and score .289218. Graph access raises it to .297374, but three scenario-irrelevant episodes rise to .300059, .302766 and .305494 and take all three slots. Every score increases while the useful memory disappears. A separate fixed-age fixture gives p=.216473 for one link at age 100 and p=.146816 for five memories sharing that hub, below .15. At beta=.5 and cutoff=.15, a hub with 167 retained memories cannot by itself produce a passing score even at w=1 and zero edge age. Other cues and direct retrieval can still work. These are mathematical/model limits, not empirical failure rates. They require crowding and distraction sweeps before interpreting long-run persistence.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

## Slide 20: Questions we should be ready to answer

The Mia–pinwheel example concerns a specific episode losing access under semantic similarity plus ordinary decay. Raw similarity is positive (.3702 in the hunger-aware fixture), and both retrieval methods initially include it. The association-only window is integer ticks 33–139 under the fixed defaults. A stronger encoder, richer query or slower ordinary forgetting can alter a comparison. We explicitly test a hunger-aware query in this illustration rather than leaving the agent's need out.

One-hop associations can be represented by an adjacency table or a graph. That equivalence preserves the relational information and does not show that associations lack benefit. The experiment studies independently aging, rehearsed associations, not the superiority of graph storage or multi-hop reasoning.

The cue activates an episode, not an approach reward. The action model may infer from Mia's earlier help and remark that she could be found there again. Its actual action must be evaluated. Max avoids summing overlapping contributions but can suppress semantic differences when the graph wins; retained-link crowding can suppress busy cues. Both remain model choices to test. No formula was changed for this presentation revision.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/associative_memory.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/code/hou_memory/memory_agent.py

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/output/analysis/pinwheel-walkthrough.json

## Slide 21: Sources

Hou: Hou, Tamoto & Miyashita. CHI EA 2024. https://doi.org/10.1145/3613905.3650839

Honda: Honda et al. HAI 2025 proceedings, published 2026. https://doi.org/10.1145/3765766.3765803

Associative competition: Anderson & Reder (1999), The Fan Effect: New Results and New Theories. https://www.andrew.cmu.edu/user/reder/publications/99_jra_lmr_2.pdf

TerraLingua: Paolo et al. TerraLingua (2026). https://arxiv.org/abs/2603.16910

APA: APA Dictionary of Psychology, Associative Memory. https://dictionary.apa.org/associative-memory

Shachi: Kuroki et al. Shachi. https://arxiv.org/abs/2509.21862

EmotionBench: Huang et al. EmotionBench. https://arxiv.org/abs/2308.03656

The supplied course proposal and the independent associative retrieval design brief define our scope. The exact post-decay max, edge aging, reinforcement rule and numerical settings are proposed engineering choices, not equations validated by these references.

Sources

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/proposal/proposal.pdf

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md

https://github.com/bloohunnits/human-like-agent-simulation/blob/main/output/analysis/independent-graph-walkthrough.json
