# Believability Evaluation

Research date: 2026-10-26. This is a proposed evaluation plan grounded in inspected source and primary research. No benchmark, simulation, judge call, or human study was run. Implementation claims refer to parent commit `a178148cab443e69ac599ae3f92509774b5e45e0` and its pinned submodules. Code IDs below link directly to sources; the complete path/function catalog is in [logging-inventory.md](logging-inventory.md).

## 1. What the study should establish

The research question is whether associations recover **appropriate context** and improve later behavior, beyond what can be explained by ordinary semantic retrieval, decay, rehearsal, or extra prompt capacity. Successful retrieval, useful action, and believable behavior are separate outcomes. A useful evaluation follows the chain:

**eligible experience → retrieval → prompt inclusion → response/decision → executed outcome → human judgment.**

The current wrapper already records much of the first four stages: memory IDs/text/times, retrieval scores/routes/inclusion, and a bound agent step. It reinforces selected memories before generation. Therefore “rehearsed” means included in prepared context, and “rescued” means included only with associations in a same-state comparison. Neither establishes that the model used the memory, that the action succeeded, or that it was believable. [MA](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/memory_agent.py) `select_action; _write_memory`; [ME](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/associative_memory.py) `preview; commit`

**Proposed hypotheses:**

- H1: associations increase recovery of independently labeled useful episodes when surface similarity is weak, without excessive irrelevant or stale recall.
- H2: learned connections sustain useful access longer than fixed connections when ordinary recall histories are controlled.
- H3: better context improves grounded decisions, relationship consistency, and retained information.
- H4: blinded raters prefer the resulting behavior under a defined rubric.

These are hypotheses, not verified findings. Predicting greater score/access in a numerical fixture does not establish H3 or H4.

## 2. Primary research and benchmark boundaries

| Evaluation family | What the primary source supports | What must be specified locally |
| --- | --- | --- |
| LoCoMo-Plus | Tests implicit cue–trigger constraint use. Official unified construction writes `input_prompt,trigger,evidence,category`, optional `answer,time_gap`. Cognitive items have no single required reference answer. The judge uses evidence and prediction; its Cognitive template is correct/wrong, while some other categories allow partial credit. | Pin release, construction seed/context, sample IDs, ingestion/time mapping, generation and evaluator versions. Keep evidence labels out of agent-visible input. A simulation cue task inspired by it is not an official dataset score. |
| LongMemEval | Original benchmark provides dated session histories, question IDs/types/answers, evidence session IDs and turn evidence flags. Official QA consumes `question_id,hypothesis`; retrieval evaluation has any/all evidence and ranking metrics. | Pin original S/M/oracle and cleaned revision; a newer V2 is a distinct benchmark choice. Map stored memories to source sessions/turns. Report answer quality separately from retrieval. |
| Hou experiments | Paper describes chronological conversational events, relevance/elapsed time/consolidation and labeled event choices; normalizes scores, applies softmax and compares with labels. Includes qualitative conversation examples. | Acquire the exact task assets/appendix version before claiming reproduction. Sim ticks require a documented time conversion; the local threshold/time scale are not paper requirements. |
| Honda experiments | Author/institutional abstract supports context/time/usage-sensitive remembering, reinforcement through repeated topics, stochastic variability and parameter sensitivity. | Full publisher text returned HTTP 403. Exact task schedules, noise, thresholds, parameter sweeps, scoring and official code remain unverified. Treat proposed probes as inspired experiments until verified. |
| TerraLingua AI Anthropologist | Existing code analyzes recorded behavior with tags, evidence references, audits and narratives. | There is no verified universal scalar believability score. Choose tags/rubric, evaluator identity, evidence windows and human validation. Analysis outputs are judgments. |
| Shachi-style evaluation | Shachi is a modular multi-task framework with task-specific observations, actions and outcomes across social complexity, rather than one score. | Choose tasks and their native metrics before defining logging. Ported TerraLingua tasks need validation of equivalent observations/actions; otherwise name them custom adaptations. |

Primary sources for the table:

- LoCoMo-Plus [paper, §§3–5](https://arxiv.org/html/2602.10715v1) and inspected official [data construction](https://github.com/xjtuleeyf/Locomo-Plus/blob/059f4e3d38f7f1f96765e8e2cb7de3097551bffb/data/unified_input.py), [judge](https://github.com/xjtuleeyf/Locomo-Plus/blob/059f4e3d38f7f1f96765e8e2cb7de3097551bffb/evaluation_framework/task_eval/llm_as_judge.py), [category prompts](https://github.com/xjtuleeyf/Locomo-Plus/blob/059f4e3d38f7f1f96765e8e2cb7de3097551bffb/evaluation_framework/task_eval/prompt.py).
- LongMemEval [official schema/protocol](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/README.md), [QA evaluator](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py), [retrieval evaluator](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/retrieval/eval_utils.py).
- Hou, Tamoto and Miyashita, [CHI EA 2024 paper, §§3–4](https://arxiv.org/html/2404.00573v1), [publication DOI](https://doi.org/10.1145/3613905.3650839). This review verified the arXiv version; a teammate's appendix reference does not establish access to those task assets.
- Honda et al., [publication DOI](https://doi.org/10.1145/3765766.3765803), [author publication record](https://app.ait.kyushu-u.ac.jp/publications/honda2025HAI2025/), [author institutional abstract](https://hyoka.ofc.kyushu-u.ac.jp/html/100021375_en.html). Bibliographic records differ between HAI 2025 conference year and 2026 proceedings publication; cite the DOI and version.
- TerraLingua [paper](https://arxiv.org/abs/2603.16910), plus actual [upstream analysis pipeline](https://github.com/cognizant-ai-lab/terralingua/blob/bf276dbe33e66b2c3b592df27df8ca42c489a4c7/analysis_scripts/AI_ANTHROPOLOGIST.md) and [AP](https://github.com/cognizant-ai-lab/terralingua/blob/bf276dbe33e66b2c3b592df27df8ca42c489a4c7/analysis_scripts/001_llm_agent_analyser.py) `annotate_and_audit`.
- Shachi [paper v3, §4 and task appendices](https://arxiv.org/html/2509.21862v3), [official implementation revision](https://github.com/SakanaAI/shachi/tree/cdca203330b8ddac72975909d12c607cfaf62f3f).

## 3. Objective metrics and subjective judgments

An automatic formula can be reproducible while relying on human judgments in its labels. Keep these two levels visible.

| Measure | Operational definition proposed for this capstone | Evidence/label requirements | Type |
| --- | --- | --- | --- |
| Recall@K | For query q with nonempty gold set G and ordered retrieved set R_K, `|G ∩ R_K| / |G|`. Report macro average over eligible queries and specify retrieval unit. | Independent relevant IDs and memory→source mapping. Deduplicate IDs. | Objective given labels |
| Any-evidence / all-evidence success | `1[G ∩ R_K is nonempty]` and `1[G ⊆ R_K]`. | Same gold sets; report separately from fractional recall. | Objective given labels |
| Included-context precision | Useful included memories / all included memories, with useful defined before viewing condition. Separate forbidden/stale intrusions. | Independent per-query labels plus included IDs. | Objective given labels |
| Empty/negative-cue performance | Rate of appropriate abstention when no useful memory exists; false recall rate on wrong cue/unexposed facts. | Explicit negative cases and acceptable response/selection definition. Empty gold Recall@K is N/A, not an automatic perfect score. | Objective given labels |
| Constraint adherence / answer accuracy | Correct response under the dataset criterion or prespecified answer key. | Final output, expected answer/constraint, evaluator identity/version. | Exact scoring when possible; otherwise semantic judgment |
| Grounded decision success | Executed desired outcome / eligible opportunities. Report chosen-action rate and execution-success rate separately. | Scenario opportunity labels and world outcome/event/receipt evidence. Refusal is not a memory failure by itself. | Objective given protocol |
| Relationship consistency | Supported dyad claims / evaluable claims, and contradictions / evaluable claims, over defined time windows. | Time-valid relationship evidence, explicit claims and adjudicated labels. | Objective aggregation over semantic labels |
| Information spread | Unique agents with verified exposure, retained knowledge or successful use, each reported separately over an explicit population. | Source/exposure/probe/use evidence, fact IDs, denominator, time. | Objective given semantic coding |
| Believability / appropriateness | Rater preference and anchored rating of behavior in context. | Blinded packets and independent human ratings. | Subjective |

LongMemEval's official `evaluate_retrieval` returns **recall_any, recall_all, ndcg**, not the fractional Recall@K definition above. Its turn→session conversion expands results until enough unique sessions are covered. Use official semantics for official scores. Current live top_k=3 and 600 retrieved-text tokens are local settings: do not report Recall@5/10 as if the live prompt included those memories. Distinguish ranking retrieval, threshold filtering and final budgeted prompt inclusion. [Official evaluator](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/retrieval/eval_utils.py); [ME](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/associative_memory.py) `MemoryConfig; _select`

## 4. Practical memory-appropriateness probes

Start with held-out, controlled histories before expensive whole simulations. Proposed cases:

1. A named object/person later cues a useful episode with weak lexical overlap.
2. The same cue appears with distracting episodes or a large entity hub.
3. A wrong cue, an unseen object, or no useful history should not produce confident invented recall.
4. An updated fact/relationship supersedes an earlier one.
5. A remembered event is factually true but irrelevant or socially inappropriate to mention now.
6. A useful memory should influence a decision without being repeated verbatim.
7. Delays and spaced/massed rehearsal change availability, with the query and ordinary recall schedule controlled.

Have independent annotators label relevant, acceptable-but-optional, irrelevant, stale/contradicted, and forbidden/unobserved evidence **before** viewing condition output. Permit multiple valid memories/behaviors. Evaluate the text sent to the model and the final response separately. Record disagreement/uncertainty rather than forcing ambiguous cases into a binary answer.

The wrapper's live cue uses observation/messages/info; it does not automatically include hunger. A hunger-aware numerical fixture is a distinct query manipulation. Include goal-aware queries as an explicitly separate factor if desired, so the study does not attribute query improvement to associations. [MA](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/memory_agent.py) `_context_query_text`

For H2, independently aging connections must be tested with matched ordinary memory recall histories. Logging alone cannot equalize exposure: specify the retrieval/rehearsal schedule in the experiment plan. Avoid using the model's own highest score to define correct memory, which would make the evaluation circular. The project's kernel and score traces support mechanistic inspection, not a claim that human recall probabilities have been calibrated. [ME](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/associative_memory.py) `commit`; [MH](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/hou_memory.py) `recall_probability; _reinforce`

## 5. Relationship consistency

Define what “relationship” means for each test. Acquaintance, favorable attitude, reciprocal assistance, promise keeping, and a directed follow link are different variables. A gift may be strategic; theft may have a permitted contextual explanation. Semantic consistency should allow evidence-based changes after new interactions.

**Proposed procedure:**

- Build a dyad timeline with actor/target tags, relevant messages, executed exchanges, promises and later updates. Preserve information access: an agent cannot reasonably know an event it never observed.
- At fixed probes, ask for identity, past interaction and current stance/commitment. Label claims as supported, contradicted, unknown or ambiguous using the timeline.
- Check subsequent actions only where an opportunity exists. A promise to help cannot be marked broken when no feasible help opportunity occurs.
- Aggregate contradiction rate and supported-claim rate. Also record whether a changed stance is justified by intervening evidence.

Reuse raw agent observations/messages, world transfers, artifact exchanges and receipts. The older path has executed transfer events; fork `social_graph.jsonl` provides actual follow/pending topology with resume segments. A social edge is not a friendship label. No additional full relationship-state snapshot is needed; the missing semantic timeline and claim labels are **manual evaluation data**. [BE](https://github.com/cognizant-ai-lab/terralingua/blob/bf276dbe33e66b2c3b592df27df8ca42c489a4c7/core/environment/env.py) `step`; [RC](https://github.com/sadorrab/terralingua-relational-memory/blob/5751a4f18becedb8302d0751e7ea0ca617cb56b6/terralingua/experiment/execution_receipts.py) `_publish_action_receipt`; [RH](https://github.com/sadorrab/terralingua-relational-memory/blob/5751a4f18becedb8302d0751e7ea0ca617cb56b6/terralingua/experiment/social_graph_history.py) `record; _graph_snapshot`

Park et al.'s *Generative Agents* uses contextual interviews and evidence checking, and distinguishes controlled response comparisons from end-to-end social measurements. It motivates this approach but does not supply a validated capstone relationship-consistency score. [Primary paper, §§6–7](https://arxiv.org/html/2304.03442v2)

## 6. Information-spread testing

Use a fact with a unique source and a clear truth condition. Include paraphrases and misinformation/control facts. Prevent unintended initial exposure through global prompts, copied memories, visible artifacts, or shared scratchpads.

For each fact, distinguish:

- **Exposure:** the recipient's recorded observation contains the transmission or artifact evidence.
- **Retrieval:** a related memory is included at a later opportunity.
- **Knowledge:** a probe response correctly states the fact, with a supporting source.
- **Use:** a subsequent decision satisfies a relevant constraint or achieves the intended outcome.

Proposed measures include cumulative unique exposure/knowledge over time, first verified exposure-to-knowledge delay, false-knowledge rate among unexposed agents, and successful use conditional on verified knowledge. State whether the denominator is the initial cohort, living agents at the probe, or all agents ever eligible. Treat death, late entry, unreachable contacts and missing logs explicitly. Do not infer a causal transmission parent merely from similar wording or first appearance.

Existing logs retain outgoing proposed text and actual incoming text; world `messages.json` contains tick-indexed broadcast strings, and artifacts retain payload/version/user history. Those support exposure coding but lack a universal message/fact lineage. Add compact sender/recipient tags, fact/message ID, channel, delivery tick and parent/evidence reference only if exact paths are a chosen endpoint. Otherwise build exposure/probe tables offline from existing logs and manually code paraphrases. [BA](https://github.com/cognizant-ai-lab/terralingua/blob/bf276dbe33e66b2c3b592df27df8ca42c489a4c7/core/agents/agent_logger.py) `log`; [BE](https://github.com/cognizant-ai-lab/terralingua/blob/bf276dbe33e66b2c3b592df27df8ca42c489a4c7/core/environment/env.py) `step; _build_obs; close`; [BT](https://github.com/cognizant-ai-lab/terralingua/blob/bf276dbe33e66b2c3b592df27df8ca42c489a4c7/core/environment/artifact.py) `serialize`; [RS](https://github.com/sadorrab/terralingua-relational-memory/blob/5751a4f18becedb8302d0751e7ea0ca617cb56b6/terralingua/environment/social_graph_env.py) `_extend_obs; _handle_direct_message`

A stable fact ID is an experiment label; the simulation should not reveal the gold ID or expected answer to agents. Knowledge probes themselves can rehearse memory, so use a fixed schedule across conditions and distinguish measurement effects from spontaneous remembering. An offline read-only scoring preview and an actual interview are different interventions. [ME](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/associative_memory.py) `preview; commit`

## 7. Blinded human believability evaluation

Use two complementary studies if resources allow:

**Matched-context response study:** give each condition the same eligible history and question/opportunity. This isolates retrieval/response quality. **Whole-trajectory study:** compare separately generated runs to assess accumulating behavioral effects, accepting that histories diverge. Do not combine the two estimands into one claim.

The following is a **proposed local rubric**, not a published validated scale. Raters see equal-context anonymized packets with persona, relevant observable history and response/behavior. Condition labels, internal score/routes and architecture explanations stay hidden. Use a prespecified excerpt window; do not select only success cases. Keep output text intact rather than editing one condition to make lengths match.

| Dimension | 1 anchor | 3 anchor | 5 anchor |
| --- | --- | --- | --- |
| Recall appropriateness | Uses irrelevant/contradicted history or misses essential context | Partly fits but misses meaningful context | Uses relevant history naturally and at the right time |
| Relationship consistency | Claims/actions conflict with established relationship without explanation | Mixed or uncertain consistency | Fits relationship and justified changes |
| Behavioral plausibility | Response/action is implausible for the stated agent and situation | Plausible with noticeable unexplained choices | Coherent, context-sensitive behavior |
| Overall believability | Difficult to accept as this character's behavior | Mixed impression | Convincing for this character and situation |

Offer “insufficient evidence” separately from the middle rating. Include a randomized A/B preference with tie/uncertain options and a short evidence comment. Counterbalance display order and keep the condition key separate. Use multiple independent raters per item; pilot the rubric, then choose sample size based on expected effect and uncertainty rather than copying a published participant count.

Store packet ID, scenario/query/run reference, displayed order, pseudonymous rater ID, rubric revision, ratings/preference and uncertainty. Report preference proportions, distributions and uncertainty intervals; assess inter-rater agreement appropriate to ordinal or categorical labels. Account for repeated raters, agents and scenarios when estimating uncertainty. Do not treat every tick in one run as an independent sample. Objective evidence coders can inspect audit/world files separately; asking believability raters to inspect internal scores can reveal the condition.

## 8. A/B conditions and ablations that exist today

The live launcher implements `hou,independent,learned,gated`; the first three match the active comparison. `learned` vs `independent` isolates connection learning if the rest is held constant. `hou` vs association conditions compares the whole association mechanism over the same direct kernel. Same-state `solo_selected` is useful diagnosis but not a substitute for a separate long-run baseline. [ML](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/run_hou_experiment.py) `MEMORY_CONDITIONS; configure_memory`; [ME](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/associative_memory.py) `preview; commit`

ACT-R helper functions and an exploratory UI mode exist, including full recall_times. They are not connected as a live launcher condition. The local UI threshold/CDF behavior and assumptions do not verify Honda's stochastic protocol. Pure cosine/vector-only live mode is also absent from this launcher; Hou includes decay and consolidation. Do not list either comparator as a completed experiment. The newer TerraLingua fork has stock history/scratchpad and different async/observation interfaces; it does not automatically run this wrapper. [MH](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/hou_memory.py) `actr_base_level; actr_recall; HouMemoryStore.score`; [MW](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/webapp.py) `_select_surfacing; api_set_kernel`; [ML](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/run_hou_experiment.py) imports; [RL](https://github.com/sadorrab/terralingua-relational-memory/blob/5751a4f18becedb8302d0751e7ea0ca617cb56b6/terralingua/agents/llm_agent.py) `select_action_async`

For future comparisons, match histories, embeddings, memory encoding, prompt exposure, recent-history/plan buffers, K/threshold/token policy, model/backend and world setup. Retain resolved configuration and code revisions once per run. Shared world seeds help comparisons but do not guarantee identical stochastic LLM trajectories. Include negative/long-delay/crowded-cue cases and report subgroup outcomes, not just a pooled accuracy score.

## 9. Minimal evidence package and remaining verification

A useful evidence package contains:

1. One run manifest with commits, launcher/condition, resolved config, model/backend provenance, dataset/scenario revision and time mapping.
2. Existing raw agent/world/usage files plus the wrapper audit and synchronized checkpoint where applicable.
3. A small evaluation table mapping query IDs, gold evidence/constraints, ordered retrieved IDs, final outputs and evaluator labels.
4. Separate relationship/fact/claim labels and blinded human ratings, referenced by stable sample IDs.

Reuse existing text once; reference it from evaluation tables. Avoid new copies of full prompts, embeddings, all memory states or all graph routes. Preserve detailed scores for the small controlled memory experiment when they are actually needed. The current wrapper already writes a full snapshot every tick and duplicates agent_step in its audit; proposed retention changes require a later engineering task, not this research review. [MA](https://github.com/bloohunnits/human-like-agent-simulation/blob/a178148cab443e69ac599ae3f92509774b5e45e0/code/hou_memory/memory_agent.py) `_dump_hou_memory; select_action`

**Human verification still required:** inspect raw completed/interrupted/resumed run folders; identify real versus human/mock backends; acquire Honda full text and intended Hou task assets; select Shachi tasks; agree on label rules and rater protocol. Confirm the older Anthropologist reduced-log message-key mismatch and fork scanner world-filename/sidecar assumptions before trusting derived results. See [logging inventory](logging-inventory.md), sections 7 and 9.

No claim is made that any of the 12 evaluations has been completed or that these protocols already demonstrate greater believability.
