# Independent associative memory: implementation and final audit

The live TerraLingua launcher now defaults to the independently aging associative pathway with winning-route learning. The engine is `code/hou_memory/associative_memory.py`; `memory_agent.py` calls it before every agent decision. The original Hou kernel and historical graph console remain available for comparison. ACT-R, partial refresh of unselected memories, emotional salience, causal edges, and bounded addition are outside this implementation.

See the later [proposal logic audit](PROPOSAL_LOGIC_AUDIT.md) for quantified crowding limits, a default-three-slot displacement counterexample, and the equivalence of the main one-hop graph policy to a weighted lookup index. These are unresolved model tradeoffs, not changes to the implementation.

## Scientific conclusion of the audit

With common inputs, `max(r D, G) >= max(r, G) D`. Score improvement is an invariant, not an empirical discovery. Likewise, the edge update deliberately increases current transmission and, when configured, its retention time. Calling either fact alone a successful research hypothesis would overstate the evidence.

The defensible experiments concern **selectivity and believability**:

- H1: at fixed memory and prompt budgets, does independent associative access recover relevant delayed episodes across held-out histories, without an unacceptable increase in irrelevant reminders? Report target recall, precision, wrong-cue retrieval, displacement of relevant direct memories, and delay curves. Define acceptable tradeoffs before the held-out evaluation.
- H2: with ordinary memory histories matched, does connection rehearsal preserve useful later access across gaps while unused or irrelevant associations still fade? Measure the persistence/false-reminder tradeoff. Compare weight renewal alone against weight renewal plus retention growth.
- Human evaluation: do blinded raters judge the resulting reminders and lapses as more believable? Retrieval scores are not calibrated human probabilities. Exact text replay is not a model of reconstructive or degraded memory.

A counterexample is now tested: every memory's score can increase while the relevant memory loses its only slot to an irrelevant graph-linked episode. The formulas do not guarantee better selection.

## Formula decision

Keep max as the initial combination. It is bounded, gives one clear winning route, and avoids adding duplicate correlated reminders. Its limitation is that multiple weak associations cannot combine. `M + G - M G` is a useful later sensitivity comparison, not a correction that should silently change this experiment.

Each memory keeps its own g and full-recall clock. Each edge keeps w, h, and its own rehearsal clock. For a route, multiply its physical edge transmissions, apply the crowding factor once at its hub, and apply beta once. Compare this with the already-decayed direct contribution. Do not apply target-memory decay again.

The main source policy is explicitly observed entities. Names inside free-form cue text do not automatically activate hubs. Optional memory sources use a fixed number of decayed direct candidates, with no recursive diffusion or self-return. In `gated` mode, exactly the same aged graph input is placed before the target's decay; it is not the older unaged-link prototype.

## What is implemented

- Pure preview, stable memory/edge IDs, deterministic ties, positive cutoff, top-k and an exact `cl100k_base` retrieved-text token cap. Oversized candidates are skipped so shorter eligible episodes can fill remaining slots. Empty retrieval is allowed.
- One committed retrieval per agent tick. Repeated event IDs return the original result; changed requests, stale previews, and backwards time are rejected. API transport retries do not double-rehearse.
- Only selected memories receive Hou rehearsal. Only a selected target's graph-winning route can learn; ties credit direct access. Shared route edges update once per event. Alternate routes and graph-off counterfactual selections are logged.
- Renewal starts from decayed edge transmission, not its old stored weight. h has a finite cap. Unselected memories receive no partial g refresh.
- The prompt contains selected episode text, an intention-only plan capped at 60 tokens, current observation and feedback. It contains no retrieval scores. The separate upstream recent-turn history is omitted by default; `working_history` can explicitly enable a matched recent buffer.
- One episode is encoded per turn. The memory says which action was chosen, not that its outcome already occurred. Next-turn environment feedback is recorded as feedback from the preceding action. Tags can refer to a contemplated action, but the nearby-person/artifact description comes only from observation.
- People, artifacts, inventory items and message senders provide observed entity cues. Optional configured place labels appear only within the actual observation footprint. These are map annotations, not new wells/resources or physical simulation mechanics. Free-form references to absent entities are not extracted.
- Graph state and memories are embedded in the same world checkpoint. Atomic memory dumps serve the viewer; they are never substituted for checkpoint state during resume. Legacy checkpoints without synchronized memory state fail explicitly instead of overwriting history with an empty or future store.
- The integration layer handles headless stdin, optional world checkpoint fields and atomic checkpoint replacement without changing the TerraLingua submodule's source.

## First checkout and dependencies

Use Python 3.13 to match the tested environment. From the repository root:

```bash
git submodule update --init --recursive
python3.13 -m venv code/terralingua/.venv
code/terralingua/.venv/bin/python -m pip install -e code/terralingua
code/terralingua/.venv/bin/python -m pip install -r code/hou_memory/requirements.txt
```

The first `SBERTEmbedder()` construction downloads `sentence-transformers/all-MiniLM-L6-v2`; later runs can use the local cache. No generation API key is required for the test suite, numerical walkthroughs or scripted smoke. A real GPT run needs `OPENAI_API_KEY` in the environment or the local TerraLingua `.env` file. Do not commit credentials.

The [team catch-up](TEAM_CATCHUP.md) explains the current presentation, score scales and remaining empirical questions. ACT-R is outside the active experiment.

## Run it

From the repo root:

```bash
cd code/terralingua
.venv/bin/python ../hou_memory/run_hou_experiment.py \
  --exp_name associative_trial --model gpt-5-nano \
  --init_agents 3 --max_ts 100 --agent_lifespan 500 \
  --memory-condition learned \
  --memory-config ../hou_memory/configs/memory.example.json \
  --memory-landmarks ../hou_memory/configs/landmarks.example.json \
  --save_root ./logs_associative
```

This real-model command uses the configured API account. The implementation verification uses a scripted generation client and incurs no API calls.

`--memory-condition` options share the same initial budgets, topology, parameters and entity-only source policy:

| Option | Retrieval | Connection learning |
| --- | --- | --- |
| `hou` | Direct Hou contribution | Off |
| `gated` | Same aged graph support before target decay | Off |
| `independent` | Separately decayed pathways | Off |
| `learned` (default) | Separately decayed pathways | Winning route only |

JSON overrides can explicitly change any `MemoryConfig` field, including mode/learning. The resolved configuration and landmark map are saved with the experiment; use those values when interpreting results. These defaults are illustrative starting values, not calibrated parameters. Use `--memory-help` for the wrapper's flags. Keep generation settings fixed within a comparison.

Resume with the original name/root and a larger final tick:

```bash
.venv/bin/python ../hou_memory/run_hou_experiment.py \
  --exp_name associative_trial --save_root ./logs_associative \
  --resume --max_ts 200
```

Resume loads the saved memory configuration and landmark map. Do not supply a new memory condition on resume. Start a separate run for another condition. A fresh launch refuses to overwrite an existing experiment directory.

## Inspect actual decisions

```bash
.venv/bin/python ../hou_memory/webapp.py
```

Open `http://127.0.0.1:5057/audit`. Select a run, agent and retrieval tick. The viewer shows pre-update M, G, score, selection reason, winning route, alternatives, edge ages/retention, before/after updates, and the exact prompt/action bound to that retrieval. The raw trace includes x (`combined`). It reads actual committed evidence and never rehearses memories.

Use `--run-dir /absolute/path/to/agent_logs --port 5058` to inspect logs outside the standard TerraLingua directory. `/` is explicitly labeled as the historical what-if console; its sliders are not the new live engine. At tick t the trace scores the store before that turn's new episode is encoded, so its row count can be one smaller than the saved store count.

`*_memory_audit.jsonl` holds completed decision traces. The atomic snapshot also holds the last retrieval, including a retrieval whose generation failed before completing an action. Checkpoint replay may append another record for an existing tick; the viewer uses the most recent entry up to the snapshot's tick. The prompt is attached to the retrieval itself rather than joined to an unrelated old log line.

## Verification

Run the 22 new mechanism and actual-agent integration tests:

```bash
code/terralingua/.venv/bin/python -m unittest discover -s code/hou_memory -p 'test_*memory*.py' -v
```

The existing 14 kernel and 15 historical graph tests are function-style scripts:

```bash
code/terralingua/.venv/bin/python code/hou_memory/test_hou_memory.py
code/terralingua/.venv/bin/python code/hou_memory/test_memory_graph.py
```

Run the real world/agent/checkpoint loop using cached MiniLM and a scripted client:

```bash
cd code/terralingua
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  ../hou_memory/smoke_associative_sim.py --sbert --output /tmp/associative-smoke-new
```

Choose an empty output path. Without `--sbert`, this smoke test uses deterministic vectors. It runs two agents for 8 ticks, restores their exact checkpoint, then continues to tick 12; verifies visibility, memory writes, audit event counts and token limits. This is an integration test, not a believability experiment.

## Remaining limits worth presenting honestly

- Parameters and cutoffs still need development-set calibration and held-out evaluation. Increasing beta/h can preserve wrong reminders as easily as useful ones.
- Max selects the strongest route, not the most reliable semantic explanation. A winning graph route can learn even when direct retrieval would also have earned a slot. The counterfactual audit distinguishes this.
- Cue absence only removes instantaneous graph support. Past graph recalls may already have strengthened the memory itself; final graph-off comparisons are not full baseline trajectories.
- Crowd counts include all retained memories in a hub, even when some edges have faded. They may over-penalize long-lived agents. An activity-weighted fan is a separate model change, not silently substituted here.
- One episode per turn means same-tick links normally do not occur in live writing. Temporal/shared-hub memory-source routes are implemented and tested but off in the primary condition.
- Entity tags are normalized explicit names, not a learned identity resolver. Initial character names are unique across the numeric spawn sequence; duplicate user/chosen character or artifact names can still conflate entities. Avoid ambiguous names in controlled experiments.
- The plan is bounded, rejects recognizable echoes of the retrieval slot, and is prompted to contain intentions, but an LLM can still write historical content into it. Audit that leakage; the code cannot certify semantic compliance.
- Returning text to the prompt is the engineering definition of full recall. It does not prove that the LLM attended to or acted on it.
- Retrieval scans stored episodes; there is no approximate vector index or pruning policy. Repeated near-identical episodes and full audit traces can grow large. Performance/representation changes should preserve the comparison conditions.
