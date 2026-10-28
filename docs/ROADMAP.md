# Roadmap: from a working engine to a measured believability result

Written 2026-10-26. Seven weeks remain if the final report and presentation land the week of 2026-12-07. Adjust the week ranges if the course dates differ. This file replaces the timeline in `proposal/proposal.tex` section 8, whose five milestones are consumed. Read `docs/TEAM_CATCHUP.md` and `HANDOFF.md` for what the engine does. This file says what we do next, who does it, and how we know it worked.

## 1. Where we are

### Against the proposal's milestones

| Proposal milestone | Status | Evidence |
| --- | --- | --- |
| 1. Model with the existing architecture | Done | `code/hou_memory/hou_memory.py`: Hou kernel and ACT-R kernel, 14 kernel tests |
| 2. Inject into the simulation | Done with scripted generation only | `code/hou_memory/memory_agent.py` subclasses TerraLingua's agent, checkpoint restore verified in `smoke_associative_sim.py`. Three short real-model runs exist locally (3 agents, 12 to 14 ticks) and predate the associative engine. No real run of meaningful length exists. |
| 3. Memory relations | Implemented, calibration open | `code/hou_memory/associative_memory.py`: entity hubs, three edge kinds, routes up to three links, winning-route learning, conditions hou, independent, learned, chained. Evidence is constructed fixtures (`scripts/pinwheel_*_walkthrough.py`). |
| 4. New model in the simulation | Done | Same wrapper, `--memory-condition` selects the engine, audit UI at :5057 |
| 5. Validate | Not started | No held-out histories, no H1 or H2 runs, no human or LLM believability judgment, no benchmark code |

Tests: 54 pass locally (25 engine and agent, 14 kernel, 15 historical graph). The committed docs still say 51 because the 2026-09-29 work is uncommitted.

### What each person has built

**Andre.** All research notes and novelty checks, proposal alignment, both kernels, the graph, the associative engine, the TerraLingua wrapper, the God Mode audit UI, 54 tests, the pinwheel, split and gap walkthroughs, the deck. Uncommitted since 2026-09-29: two-hop routes (`entity_hops`), the `chained` condition, the energy-rise episode line, both gap fixtures, the equation images, and doc updates. Until this is pushed, Ben and Ryan cannot see `chained`.

**Ben.** On main: `code/integration_layer/server.py`, a reverse proxy on ports 9000 and 9001 that answers TerraLingua's hardcoded DeepSeek endpoint with any backend, including a human typing replies. On `origin/bs/proxy`: a cleaner proxy that fronts a local Qwen3-4B through llama.cpp on :8080 and a single entry point. On `origin/bs/mem-storage`: a Postgres memory table mirroring the Hou memory fields, a stub multi-agent store, a Makefile, `pyproject.toml`, and a submodule `code/terralingua-relational-memory` pointing at his own repository, which is a copy of upstream TerraLingua v0.2.1 (2026-10-05) plus a 7-line Makefile. That copy is a restructured engine: a `terralingua` package with a scenario hook system (`Mechanic.on_menu`, `on_action`, `on_step`, `identity`) that adds actions and scripted world events without engine edits, personas from JSON, a run seed, per-step cost logging, and feedback to receivers of gifts and theft that upstream `bf276db` lacks. Our wrapper does not run against it yet (six known breaks, about 150 lines of glue).

Branch tips need fixing before any merge: `CREATE_SQL` has a trailing comma, the INSERT names 10 columns with 9 placeholders, `test_memory_database.py` imports a module name that does not exist and calls a method that was removed, `multiagent_memory_store.py` has `__init__` without `def`, and on `bs/proxy` `UnknownPathError` is declared with `def` so it is a function, not an exception.

**Ryan.** No commits on any branch. Section 4 assigns him the evaluation pipeline, which needs no engine internals.

### What we do not know yet

- Whether a recovered memory has ever changed a real agent's action. No run shows it.
- How often the world produces a reminding opportunity on its own (an entity reappearing after a 20-tick gap with a real past event).
- Whether humans can agree on believability for these agents at all. SOTOPIA reports GPT-4 versus human Pearson .45 on believability and human-human kappa .20 at 5 bins, the weakest dimension they measured.
- Which model the OpenAI project bills and at what price. One reading of the pricing page lists GPT-5 Nano at $0.05 in and $0.40 out per 1M tokens with a deprecation label, and the Luna row at $0.20 and $1.20. An earlier reading in this project recorded Luna at $0.10 and $0.50. The day-1 smoke run settles it from the billing page.

Measured cost from the local GPT-5 Nano run: about $0.00026 per agent-tick (mean 2114 input and 384 output tokens). A 10-agent, 300-tick run is about $0.78. A 20-agent, 1000-tick run is about $5.

## 2. The claim we defend in December

Three separate claims, each with its own measurement, never collapsed into one number (this is what `docs/PROPOSAL_LOGIC_AUDIT.md` section 7 asks for):

1. **Prompt inclusion.** Associations put the right past episode in the prompt when similarity and decay alone would not. Measured from `<tag>_memory_audit.jsonl` with no LLM: `selected` versus `solo_selected` at planted reminding opportunities.
2. **Context-appropriate action.** The agent acts on it. Measured by engine-logged behavior in scripted probes (whom it approaches, whom it gives to), and by fact-check dimensions of a judge calibrated to humans.
3. **Human-rated believability.** Blind raters prefer the behavior. Measured by pairwise human preference on cue cards, with the LLM judge used only to scale to more items after it passes a pre-registered agreement gate.

The positioning stays what `docs/RESEARCH.md` already narrowed it to: an association that ages on its own clock next to a human forgetting curve, inside a multi-agent world, with the cue being what the agent sees, and believability as the outcome. Nearest prior work (HeLa-Mem, Synapse, Hindsight) is cited whenever the graph is shown.

## 3. Decisions to make this week

| Decision | Recommendation | Why |
| --- | --- | --- |
| Engine base for every measured number | Upstream `code/terralingua` at `bf276db` for weeks 1 to 4 | The wrapper runs today, 54 tests pass, three real runs exist. |
| Ben's v0.2.1 copy | Port on a branch, time-boxed to 2 days, go or no-go on 2026-11-06 | Acceptance: a 3-agent, 12-tick Nano run on the new engine writes the same audit record keys. If it fails, the new engine leaves December scope and only the `promise` action is lost. |
| ACT-R | Demote from "base architecture" to a named comparison condition, optional in week 5 | The live direct route is Hou's kernel. Update `docs/GAPS.md` decision 2 and own the deviation in the report. |
| Postgres | Analysis sink keyed by run, condition, seed, agent and tick, after the schema fixes. Never a resume source. | `from_dict` rejects a store whose edges do not match the memories, and a 100-tick life holds about 100 episodes. The checkpoint stays the only source of truth. |
| Local Qwen | Smoke tests and a second-family judge cross-check only | Its ability to pass TerraLingua's strict params-key parser every tick is untested, and Nano costs cents. |
| Canonical deck | Freeze v12 until the week-4 progress check produces measured figures, then build one v14 | `output/presentations` holds untracked v2 to v13. |

## 4. Week by week

Owners: Andre (engine, probes, calibration, report), Ben (infrastructure, runs, port, registry), Ryan (evaluation protocol, cards, judge, human pilot, statistics).

### Week 1, Oct 26 to Nov 1: first live behavioral number, instrument v0, base decision

- Andre, day 1: commit and push the 2026-09-29 working tree (14 modified tracked files plus `scripts/pinwheel_split_walkthrough.py`, `scripts/pinwheel_gap_walkthrough.py`, `output/analysis/pinwheel-gap-walkthrough.json`, `output/analysis/pinwheel-split-walkthrough.json`, and the PNGs under `output/analysis/`). Leave the submodule's local `.gitignore` edit out.
- Andre, day 1: 12-tick smoke to confirm the billed model and that the audit file appears, then record model and price in `HANDOFF.md`:

```bash
cd code/terralingua && .venv/bin/python ../hou_memory/run_hou_experiment.py --exp_name nano_check --model gpt-5-nano --init_agents 3 --max_ts 12 --memory-condition learned --memory-config ../hou_memory/configs/memory.example.json --save_root ./logs_probe
```

- Andre, days 1 to 4: the generous-stranger probe (section 5). Numbers by Friday.
- Ryan, days 1 to 3: set up the venv per `docs/ASSOCIATIVE_IMPLEMENTATION.md`, run the three test commands in `HANDOFF.md`, read `docs/PROPOSAL_LOGIC_AUDIT.md` section 7 and `docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md` section 10.
- Ryan, days 3 to 5: write `docs/EVALUATION_PROTOCOL.md` v0 (section 6). Rubric, bars and controls are fixed before any rating exists.
- Ryan: `scripts/probe_signatures.py`, one row per agent-tick from `agent_logs/<tag>.jsonl`, `messages.json` and `<tag>_memory_audit.jsonl` (tick, agent, action, params target, nearby agents, message, selected ids, solo_selected ids, energy). Every later metric reads this table.
- Ben, days 1 to 3: clone his repository locally, write `docs/FORK_DIFF.md` (one page: what v0.2.1 changes, the six breaks for `memory_agent.py`, the default changes max_history 10, internal_memory_size 500, food_spawn_rate 10, lifespan jitter), and fix the branch-tip errors listed in section 1.
- Ben, days 3 to 5: run two open-world histories on upstream for Ryan's card extractor: 10 agents, 300 ticks, conditions hou and chained, with `--memory-landmarks ../hou_memory/configs/landmarks.example.json`. About $0.78 each.
- Friday: base decision meeting with the probe numbers and `docs/FORK_DIFF.md` on the table. Record it in `HANDOFF.md`.

Exit: probe figure exists, protocol v0 committed, 09-29 work on origin/main, base decision and billed model recorded.

### Week 2, Nov 2 to Nov 8: replay bench, calibration, fork port on a branch

- Andre: `scripts/replay_bench.py`. Replay every probe and open-world history through every `MemoryConfig` condition at zero API cost (`AssociativeMemory.add`, `preview`, `commit`). Per tick, per condition: selected set, Jaccard against hou, rescued fraction (selected minus solo_selected), reminding-opportunity count. Report selected-set divergence and target-line inclusion as two separate numbers. On the gap fixture chained at w 1.0 differs from hou on 152 of 171 probe ticks while the food line differs only on ticks 42 to 57, so the bench decides where to spend raters, not whether.
- Andre: calibrate on week-1 traces, not fixtures. Sweep beta {.3, .5, .7}, edge_weight {.8, 1.0}, retention {100, 150}, and the crowd tax, scoring reach at reminding opportunities, filler share and rescued fraction. Fix or flag the content-blind followed edge: at tick 50 of the gap fixture it puts a "move up" filler (p .177) above the food line (p .109).
- Andre: named baselines in `MEMORY_CONDITIONS`: `flat` (mode hou, time_scale very large so decay is near 1) and `slow` (mode hou, time_scale 75 to 100), plus `stock` (no rebind, upstream agent with max_history 4 and the 150-token scratchpad). The proposal promised a flat vector store baseline and this makes it a real row.
- Ben and Andre, 2 days time-boxed: port `memory_agent.py` and `run_hou_experiment.py` to the v0.2.1 engine on a branch. Go or no-go on Nov 6.
- Ben: `scripts/run_registry.py`, one CSV from every `params.json`, `memory_config.json` and `token_counts.jsonl` with condition, agents, ticks, tokens and dollars.
- Ryan: try the Rescued set page on the week-1 runs (section 6) and fix any wording that confuses a first-time rater.

Exit: replay bench JSON and one two-panel PNG, a calibration table, the three named baselines, the port verdict.

### Week 3, Nov 9 to Nov 15: probes package, frozen config, progress-check figure

- Andre: freeze `code/hou_memory/configs/memory.frozen.json` by Nov 13 and never change it for weeks 4 to 6. Whole-run reduction check: a hou run and an independent run at beta 0 with the scripted client, compared trace for trace.
- Ben: the probes package. On upstream, a `ScriptedNPC` factory (section 5) covers partner choice, reciprocity, return to food, warning and recognition. If the port passed, the same probes as one `ProbeMechanic` on the new engine, plus the one gated action `promise(target, amount, by_step)` logged as PROMISE_KEPT or PROMISE_BROKEN by reading `env.transfers` before the deadline.
- Ryan: run the first judge versions on the Rescued set page (section 6). Recruit 3 to 5 classmates as blind raters and book a 45-minute session in week 4. Ask the instructor whether classmates rating anonymized transcripts needs consent paperwork.
- All, Nov 15 progress-check package: one figure (partner-choice rate versus gap for stock, hou, learned, chained, with Wilson intervals) and one table (reminding opportunities per run, rescued fraction, retrieval hit rate, cost per run). If the port slipped, the figure comes from upstream probe data.

Exit: frozen config committed, reduction check passes, probes package runs end to end, the figure is a PNG regenerated by a script under `scripts/`.

### Week 4, Nov 16 to Nov 22: human anchor set and the judge gate

- Runs: 10 agents, 300 ticks, conditions hou, chained, and learned with edge_weight 1.0 through `--memory-config` so learned and chained differ only in the extra hop. 3 repeats each, 9 runs, about $7.
- Ryan: 60 cards per contrast for 3 contrasts (hou versus chained, hou versus learned, chained versus learned), 180 pairs plus 18 self-pairs. 6 raters, at least 3 classmates, 30 to 60 pairs each blind, both orders balanced across raters, 45 to 90 minutes each. D5 pairwise is primary, D4 Likert is diagnostic.
- Ryan: run the judge on the same pairs with GPT-5 Nano and the current Luna model, both orders. Cost under $1.
- Ryan: `score.py` output. Krippendorff alpha for human-human, Cohen kappa for judge versus human majority with the Judge's Verdict z-test, Spearman on D4, swap-consistency, self-pair rate, a length-only predictor check.
- Gate: promote the judge to unrated items only if kappa on D5 is at least .61, or on D1 to D3 if D5 fails. Items where the two orders disagree go to a human.
- Andre presents the progress check from measured data.

Exit: the gate decision and its numbers appended to `docs/EVALUATION_PROTOCOL.md`. Anchor set saved under `output/analysis/anchor-set-v1/` with a sealed condition key.

### Week 5, Nov 23 to Nov 29 (Thanksgiving week): architecture variants, measured

Each variant is a `MemoryConfig` flag with a fixture test asserting its predicted tick, run through the same probes and the same judge so it lands as one more row in the existing table. Order and predicted effects are in section 7. Reruns after every flag: `test_no_cue_beta_zero_and_hou_reduction` and the whole-run reduction check.

Exit: every condition in the final table has a fixture test, a retrieval hit rate, a judge score and a human spot check (20 cards per condition by two humans). No condition is reported without all four.

### Week 6, Nov 30 to Dec 6: final grid and analysis

- Final grid: probes at 8 agents, 300 ticks, 5 seeds, 4 conditions (stock, hou, chained, best variant), 20 runs, about $10. Open world at 10 agents, 500 ticks, 3 conditions, 3 seeds, 9 runs, about $12. No 20 by 3000 runs.
- Ryan: extract and judge all cards with the promoted judge in both orders, send the top 20 percent by disagreement plus a random 10 percent to the human raters, report judge-human next to human-human.
- Ben: Postgres as an analysis sink from `bs/mem-storage` after the fixes: memory_id TEXT, an edges table, an events table from the audit jsonl. Log cost per run.
- Andre: `docs/RESULTS.md` with the three claims per figure, reminding-opportunity counts, cost per run, gate numbers.
- Code freeze Dec 9. Any engine change after it invalidates the held-out numbers.

Exit: figures frozen under `output/analysis/final/`, every number traces to a file path.

### Week 7, Dec 7 to Dec 13: report and deck

- Andre: report. Own the deviations: live base is Hou with ACT-R as an implemented comparison, GPT-5 Nano instead of Qwen3.6 27B, Shachi reworded to behavioral reproduction benchmarks, EmotionBench a generator check, the AI Anthropologist a card selector not a judge.
- Ryan: evaluation section with the circularity answer (section 6).
- Ben: README, HANDOFF, `docs/ASSOCIATIVE_IMPLEMENTATION.md` refresh (54 tests, underscore directory names, memory_storage and proxy rows, psycopg, pandas and requests in the dependency list, how to run through the proxy), one Makefile target per figure.
- Deck v14 from v12: replace the one-record pinwheel chart with the measured probe figure and the judge-versus-human table, keep the three-claim slide, add one slide on what was cut and why.
- Two buffer days.

## 5. Andre's first experiment: the generous-stranger probe

**Question.** After a gap of 10, 40 or 80 ticks out of view, does a hungry agent turn to the stranger who fed it rather than the one who robbed it, and does the answer depend on the memory condition?

**Why first.** It is the smallest live version of H1 (links outlast decay) with a behavioral readout that needs no judge, no port, no new action and no rater. It runs on the base that works today, costs under $1.50, and its return-tick transcripts become the first cue cards for the judge pilot. Predictions are computable in advance, so the result is a test, not a demo.

**Day 1 morning.** Commit and push (week 1 step 1). Run the 12-tick smoke above and confirm `logs_probe/logs/nano_check/agent_logs/being0_memory_audit.jsonl` exists. None of the three existing local runs has one.

**Day 1 afternoon to day 2: `code/hou_memory/run_probe_experiment.py` (about 150 lines).**

- Import `run_hou_experiment` as `launcher` so every module-level patch applies.
- Set `launcher.NAME_POOL[1] = "Mia"` and `launcher.NAME_POOL[2] = "Kai"` before building the runner. The rename patch indexes `NAME_POOL` by the numeric tag suffix at call time.
- Replace `launcher.runner_module.LLMAgent` with `make_agent(**kw)` returning `ScriptedNPC(**kw)` when `kw["agent_tag"]` is being1 or being2, and `HouMemoryAgent(**kw)` otherwise. `runner._init_state` constructs agents through that module-global name with keyword arguments only. The other two construction sites (reproduction, respawn) stay idle with `reproduction_cost 1000` and `min_agents 0`.
- `ScriptedNPC` contract: `select_action(obs, available_actions, reward, info, time, chat_params=None, client=None, max_attempts=1)` returning `{"action", "params", "message"}`, plus `close()`, `set_state_ckpt(d)`, and `get_state_ckpt()` returning a small picklable dict (the checkpoint fires at tick 100 in the gap-80 run). It must also set `self.agent_name`, `self.agent_tag` and a writable `self.max_history`, because `select_with_retry` reads and writes `agent.max_history` outside its try block. It ignores the client, so it makes no API calls.
- Placement: being0 (Ada) starts at (10,10) and being1 (Mia) at (12,10), hardcoded upstream. being2 is placed at random. Wrap `SimulationRunner._make_env` to capture the env, then wrap `env.restart_env` so `agent_poses` gets `"being2": (8, 10)`. Keep all encounters near the center of a 20 by 20 grid so the torus seam never matters (give and take targets are not wrapped upstream).
- NPC policy, reading `env.agent_pos`. Mia: ticks 0 to 1 step adjacent to Ada and broadcast "I am Mia. You look hungry, here is some energy." Tick 2 give 10 to Ada. Ticks 3 to 8 walk out of the 7 by 7 vision square, then hold. Kai: approach, broadcast "I am Kai. I do not share." when adjacent, take 5 from Ada the next tick (by tick 10 at the latest), out of view by tick 14, hold. At T = 15 + G both return to cells at Chebyshev distance 3 from Ada on opposite sides, stand still and stay silent for 15 ticks. Keep Mia's visible window short so the Mia hub holds about 9 tagged episodes (crowd tax 1/(1+ln(9/3)) = .476).
- Probe logging: wrap `env.step` to append one line per tick to `logs_probe/logs/<exp>/probe_events.jsonl` with tick, the three positions, Ada's action, params, message and energy, and the phase (encounter, gap, return).

**Day 2: the run command**, from `code/terralingua`:

```bash
.venv/bin/python ../hou_memory/run_probe_experiment.py --exp_name probe_hou_g40_r1 --model gpt-5-nano --init_agents 3 --max_ts 70 --agent_lifespan 500 --init_agent_energy 85 --init_food 1 --food_spawn_rate 0 --food_zones None --vision_radius 3 --grid_size 20 --reproduction_cost 1000 --max_history 4 --memory-condition hou --memory-config ../hou_memory/configs/memory.example.json --save_root ./logs_probe --no-save_video
```

Rules the CLI enforces: boolean flags take no value (`--no-save_video`, `--no-reproduction_allowed`), never `--init_food 0` (the food distribution raises on zero, so one tile with zero spawn is the closest legal setting), and `max_ts` is 15 + G + 15, so 40, 70 and 110. Set `--init_agent_energy` to 55, 85 and 125 for gaps 10, 40 and 80 so Ada has about 35 energy at the return tick in every cell and both NPCs stay alive through the final tick. With no food anywhere, Mia's gift is the only energy Ada ever gains.

**Day 3: the grid.** Conditions hou and learned at gaps 10, 40 and 80, with 10 repeats at gap 40 and 5 at gaps 10 and 80 (5 repeats cannot separate .5 from .8 with a Wilson interval), plus chained at gap 80 times 5, plus a stock arm (plain upstream agent, no rebind) at each gap times 5 as the proposal's existing-architecture baseline. About 60 runs, one Nano call per tick for Ada only, 40 to 110 calls each, about 2 to 6 minutes per run in a background loop, total under $1.50. Upstream has no seed flag, so repeats are independent worlds.

**Days 3 to 4: `scripts/probe_choice_rate.py`.** In the return window [T, T+15) read `probe_events.jsonl`, `messages.json` and `being0_memory_audit.jsonl`. Record the first NPC Ada moves toward (first strictly decreasing Chebyshev distance), the first NPC named in her message, and any give or take targeting an NPC. The gift episode is the tick-3 record containing "Energy rose", not the tick-2 record, because the writer compares the current observation's energy with the previous one. Find it by text and check whether its id is in `selected` and in `solo_selected` at T and T+1, likewise the theft record. Also record the Mia fan count, the fraction of ticks Ada's plan mentions Mia (plan leakage, which can mask the memory effect in every condition), and parse retries (any tick above 3500 input tokens in `token_counts.jsonl`). Save `output/analysis/probe-generous-stranger-v1.json` and one PNG of P(Mia first) versus gap per condition with Wilson 95 percent intervals, and write three sentences into `docs/TEAM_CATCHUP.md` for the Friday meeting.

**Predictions to check** (r about .35 for the gift record, time_scale 25, cutoff x about .10, Mia fan about 9):

| Condition | Gap 10 | Gap 40 | Gap 80 |
| --- | --- | --- | --- |
| hou (similarity times decay) | gift selected, M about .14 | not selected, .04 | not selected |
| learned (one-hop Mia route, w .8) | selected, G about .15 | selected, .11 | not selected, .08 to .09 |
| chained (w 1.0) | selected | selected | selected, about .12 |

So gap 40 separates hou from learned, and gap 80 separates learned from chained.

**Decision rule for week 2.** If the hit rate follows the prediction but P(Mia first) does not move, the engine works and the generator ignores the slot, which points at prompt placement and the plan contract rather than at architectures. If both move together, the probe transcripts become the first cue cards. If hou already saturates at every gap, raise the gaps to 40, 80 and 150 before touching any parameter.

## 6. The believability instrument

Written up as `docs/EVALUATION_PROTOCOL.md` by Ryan in week 1 and frozen before any rating.

**Unit of judgment.** A pairwise cue card at a reminding opportunity: a tick T where an entity appears in the observation after at least 20 ticks absent and has an earlier episode with a real event (give, take, message, energy rise). The card shows a dossier of every earlier episode mentioning the entity (verbatim from `<tag>_hou_memory.json`, at most 8 lines, target and distractor marked only in a sealed key), the observation at T, and the action plus message at T and T+1 under two conditions. Where the seed patch allows, the pair is built by resuming one checkpoint at T under each condition so the pair differs only in the memory slot. Never judge raw trajectories without a dossier: with move, give, take and a message, responses without history are indistinguishable.

**Dimensions.** D1 consistency with past events (count of contradictions with dossier lines). D2 appropriate remembering (target line cited). D3 appropriate forgetting and no fabrication (distractor dragged in, or a stated past event not in the dossier, with a "not enough information" option). D4 plausibility of the action given energy, who is nearby and history, 1 to 5 with two anchor examples per level. D5 pairwise verdict A, B or tie. D1 to D3 are fact checks against the log, which is where the literature shows judges and humans agree. D4 and D5 are the taste dimensions where they do not.

**Bars, fixed in advance.** Human-human Krippendorff alpha .667 tentative and .80 reliable, with bootstrap intervals. Judge versus human majority Cohen kappa at least .61, with the judge counted human-like when its kappa sits within one standard deviation of the human-human pairwise kappas. Spearman on D4 at .5 matches the field and .6 is good. Always print judge-human next to human-human on the same items.

**Controls.** Both presentation orders for every pair, tie when the orders disagree, 10 percent self-pairs that must land near 50 percent, names re-randomized across conditions, the injected memory slot and the plan stripped from what judges see, a length-only predictor check with a re-judge on a length-matched subset if length predicts the verdict, the top 20 percent of judge disagreement escalated to humans, and the judge from a different checkpoint than the actor. Cross-check 60 items with local Qwen3-4B through the proxy as the only family-diverse option on hand.

**Where it lives.** The Rescued set page in God Mode (`code/hou_memory/webapp.py`, http://127.0.0.1:5057/rescued). Pick a run and type your name to label. The Judge tab edits the judge's instructions, saves each edit as a new version, runs it on the cards people labeled, and lists where it disagreed on practice cards. The Match tab shows agreement with people on practice and test cards and whether the judge passes. Cards come from `scripts/rescued_cards.py`, and the files and statistics live in `code/hou_memory/judging.py`.

**Cost.** One card is about 1,200 input and 350 output tokens. 180 pairs in both orders cost under $0.25 on Nano and under $1 on Luna. Human time dominates and is free with classmates.

**The circularity answer, for the report.** The judge is never the ground truth. It is an instrument calibrated on a human-rated anchor set with a reported kappa and a pass or fail gate, and the headline believability claim is made from the human ratings. D1 to D3 are checked against the ground-truth log. Judge and actor families differ where possible and the actor's memory slot is hidden. A human-played agent through Ben's proxy gives a ceiling an LLM judge cannot invent. The retrieval hit rate uses no LLM at all, so if believability and retrieval move together across conditions the circularity worry cannot explain the result.

## 7. Architecture comparison set

Ranked by scientific value per day. Items 1 to 3 are ours and ship in week 5 in this order. Items 4 to 6 are optional in week 5 if the first three pass their fixture tests. Items 7 onward are cited, not built. Predicted ticks are for the gap fixture (cue blue pinwheel in view at tick 50, sighting with Mia at tick 0, food from Mia at tick 10, filler between), where today hou drops the food line after tick 41.7 and chained at w 1.0 keeps it to 57.1.

| # | Condition | Rule | Effort | Prediction on the gap fixture | Fit |
| --- | --- | --- | --- | --- | --- |
| 1 | Noisy-OR combination | x = M + G - MG instead of max(M, G) | 0.5 day | food end tick about 58 at w .8 (food selected at tick 50 with p .19) | one line in `preview`, plus a credit rule in `commit` for when both channels contribute |
| 2 | Energy salience on g | g0 = 1 + tanh(abs(dE)/20) for episodes with an energy change | 1 day | food end tick 65.8 with the graph off | `_write_memory` already computes dE. Generative Agents' importance term from the world's own signal, no LLM call |
| 3 | Need cue hub | tag energy-rise episodes with `need:food`, add it to cue sources when energy is low | 1 to 2 days | one-hop p .372 at tick 50, end tick 149 | new entity kind. Makes cue-by-sight versus cue-by-need a rater question |
| 4 | Generative Agents scorer | recency .995 per tick, LLM importance 1 to 10, relevance, min-max sum, no cutoff | 2 days | food and sighting always in the prompt, never a forgetting event | sibling class (needs a positive threshold otherwise), one Nano call per episode |
| 5 | ACT-R mode | kernel in `hou_memory.py` lines 58 to 75, seeded noise, tau field | 1.5 days | food P .96 at tick 50 and .75 at tick 500 with no graph. Filler leak depends on tau | the proposal's named base, report as a kernel comparison |
| 6 | HeLa-Mem style edges | edge kind `co-recalled` incremented in `commit` for selected pairs, additive one-hop spread | 2 to 3 days | no pinwheel-to-food association until the two are recalled together repeatedly. Ours has it from tick 10 by shared entity | the nearest prior art, head-to-head on how edges are learned. Note its released code never calls its decay function |
| 7 | Synapse crowd rule | replace 1/(1+ln(fan/3)) with 1/fan | 0.5 day | a Mia hub of 40 gives .025 versus our .279, so the route dies in a crowded world | one flag gets the key prediction without building Synapse |
| 8 | Consolidation nodes | one Nano call every K ticks writes a one-line insight tagged with its evidence's entities, evidence edges to sources | 3 to 4 days | one-hop end tick about 150 | Generative Agents reflection and HeLa-Mem distillation aging on our clock. Defer unless week 5 finishes early |
| 9 | Personalized PageRank over our graph | mass from cue hubs over `Edge.transmission`, sum over routes | 2 to 3 days | sum versus max on identical state | clean controlled test, no LLM |
| 10 | MemoryBank, A-MEM, Hindsight, full Synapse | cite | 0 | | Hou subsumes MemoryBank. A-MEM adds 2 to 12 LLM calls per episode. Hindsight is 8 to 12 days |

Items 1, 2 and 3 total about 3 days and give a six-condition ablation (stock, hou, flat, chained, ours with 1 to 3, and one from the literature) for the final grid.

## 8. Making memory visible in the world

The hypothesis "more action options make believability easier to judge" does not survive in its raw form. Tool-selection accuracy falls as menus grow (92.4 percent at 15 tools to 78.2 percent at 640 in the cited study), TerraLingua's own Long Memory condition reduced longevity and productivity, and LLM populations homogenize regardless of options. What made memory visible in Smallville and the Concordia contest was a few consequential actions whose outcome the engine logs. So the reframed hypothesis is: a few new rules that make an action sensible only with memory raise judge discriminability, and menu entries without such gating add noise. Week 3 tests it directly with a gated versus ungated A/B on the same world.

Probe set, in build order. Each has an engine-logged signature and a no-memory baseline.

| Probe | Signature a judge or script looks for | No-memory baseline | Needs |
| --- | --- | --- | --- |
| Partner choice (week 1) | first approach or addressed message targets the generous NPC after the gap | 50/50 once the gap exceeds the history window | ScriptedNPC factory on upstream |
| Reciprocity | a give to the earlier donor appears in the world log when the donor returns needy | give rate to donor equals give rate to a stranger | same |
| Return to food | with no food in view and energy low, distance to the last-seen food zone decreases | random walk | static food, two far zones, vision 3 |
| Warning | a broadcast naming the thief when the thief re-enters view near a third agent | silence | same factory |
| Recognition | first message references the NPC's name or stated fact unprompted after the gap | none | the pinwheel fixture made live |
| Promise | `promise(target, amount, by_step)` then PROMISE_KEPT or PROMISE_BROKEN from `env.transfers` | promises in broadcast text fall out of the window | the v0.2.1 engine's `Mechanic` hooks only |
| Pass a story to offspring | an artifact or message to the child carrying an earlier specific fact, then the child acts on it | none | defer, too hard to score in 7 weeks |

Scripted NPCs on upstream come from the agent factory in section 5, which reads `env.agent_pos` and makes no API calls. The proxy's human-typing mode (`code/integration_layer/server.py`) is kept for one thing: a teammate plays one agent for 30 ticks in the probe world, which gives the judge a human ceiling card and lets the rubric be written by someone who has sat inside the prompt.

The authors' own `GPaolo/terralingua-pandemics` repository is the template if we build on the new engine: a scenario package with a Mechanic, personas, scenario events logged for analysis, and scripted no-model calibration runs. No other TerraLingua fork adds memory, richer actions or evaluation. TerraLingua Live accepts a prompt and an API key, not agent code, so it cannot run the engine.

## 9. Idea backlog

Ranked. Effort in person-days.

1. **Divergence-targeted cards (1).** Judge only ticks where two conditions injected different memories on the same replayed history. Spends every human rating where mechanism could matter.
2. **Checkpoint branching at a divergent tick (2 to 3).** Resume one checkpoint under two conditions for one tick so a rated pair differs only in the memory slot. Needs a run-seed patch in `run_hou_experiment.py` because upstream has none. The environment side is unverified and must be checked first.
3. **End-of-life interview (1 to 2).** Five Park-style questions through the same engine at run end ("Who is Mia?", "Where did you last find food and when?", "What happened last time you met Kai?", "What next and why?", "What would you say to Kai?"), graded against the agent's own memory file with a "not enough information" option. Produces a fabrication rate and a number comparable to Park 2023.
4. **Human in the agent seat (0.5).** A teammate plays one agent for 30 ticks through the proxy. The ceiling card.
5. **Judge-free retrieval board in God Mode (1).** A tab in `webapp.py` reading the audit jsonl: ticks where selected differs from solo_selected, hit rate, distractor rate, filler rate.
6. **Show-the-route cards (1 to 2).** For half the cards, append the route explanation (which entity, which links, how old) under the action, and test whether a visible reason for remembering changes human ratings. Separates "the action was plausible" from "the remembering was plausible".
7. **Cue-by-sight versus cue-by-need as a rater question (1 to 2).** With the need hub variant, pair an agent that recalls Mia's food when it sees the pinwheel against one that recalls it whenever hungry, and ask which reads more human.
8. **Gated versus ungated menu A/B (1).** The direct test of the action-space idea.
9. **Opinion node (2).** Borrowed from Hindsight: an "attitude toward Mia" confidence raised on each energy gain near her, visible to the judge.
10. **AI Anthropologist as a card selector (1).** Run its tagging (Reciprocity, Altruism, Record Keeping, Collective Memory) over our logs and sample cards where a tagged event involves a previously met entity. The paper reports no human validation of it, so it never judges.

## 10. Cut list

Each item stays in the report as one sentence saying what it measures and why it was not run.

- **Shachi's 10-task suite.** Measures reproduction fidelity against original papers. The word believability does not appear in it. Reword the proposal sentence to "behavioral reproduction benchmarks". If week 6 has slack, wrap `AssociativeMemory` as its `BaseMemory` and run EmotionBench and Sotopia once.
- **EmotionBench as a memory test.** A one-shot appraisal of the generator, not of retrieval or decay.
- **Hou appendix B and C replication.** The paper released no code and `docs/INTEGRATION.md` records that its tables do not reproduce from its own equation. At most a one-day kernel check. It does not exercise the graph.
- **ACT-R as the base architecture.** Demoted to a named comparison. Update `docs/GAPS.md` decision 2.
- **Postgres as a memory backend or checkpoint replacement.** Analysis sink only, after the fixes.
- **Local Qwen as the generator.** Smoke tests and a second judge family only.
- **Full Hindsight, A-MEM, Synapse.** Cite with the numbers in `docs/RESEARCH.md`.
- **Paper-length 20 by 3000 runs.** 300 to 600 ticks show the forgetting dynamics at time_scale 25. Mechanism comparisons need seeds, not longer runs.
- **Adding share_location, trade, return_to and teach as menu actions at once.** Add `promise` only, run the gated versus ungated A/B, let the result decide.
- **RemoteAgent and the dashboard.** Pulls in Redis and Postgres. The agent factory does the job.
- **TerraLingua Live.** Cannot run our code.
- **Maya as a testbed or a Java port of the kernel.** Doubles the codebases and cannot support the positioning. Zero-API fallback demo only.
- **Prolific raters.** Six classmates cover the anchor set at zero cost.
- **The one-record pinwheel chart as headline evidence.** Replace with the gap fixture plus the measured probe results.
- **Deck churn.** v12 is canonical until v14 is built from measured data.

## 11. Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| GPT-5 Nano unavailable to the project | GPT-5 mini at roughly five times the cost, still under $10 for weeks 1 to 3 |
| The port eats both weeks | Branch only, go or no-go Nov 6, every measured number stays on upstream |
| Fewer than 60 reminding opportunities in open-world runs | The scripted probes guarantee them |
| Human-human alpha on D5 below .667 | D1 to D3 become primary, D5 reported descriptively, longer dossiers next round |
| Judge fails the gate | Humans rate a smaller final set, the judge is reported as exploratory |
| Rater no-shows | Teammates rate self-pairs and fact dimensions only, final human set collected in week 6 |
| Parse retries double cost | reasoning_effort low, stop any run whose mean input exceeds 3500 tokens |
| Plan field leaks entity names across the gap | Report leakage per condition, add a no-plan arm if it is high |
| Null behavioral difference | Report it with intervals. The retrieval hit rate still separates the conditions mechanically |

## 12. Questions for the Friday meeting

1. Stay on upstream for December, or move everything to the v0.2.1 engine if the port passes by Nov 6?
2. Does the instructor need consent paperwork for classmates rating anonymized transcripts?
3. How many classmates can we recruit for a 45-minute session in week 4?
4. Is ACT-R in the final story as a comparison condition, or only as an owned deviation?
5. Does Ben keep his repository as a copy, or re-create it as a GitHub fork so upstream diffs are visible?
