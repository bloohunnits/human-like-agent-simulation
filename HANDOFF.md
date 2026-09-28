# Current handoff — associative memory and proposal v12

Updated 2026-09-28. Read [TEAM_CATCHUP.md](docs/TEAM_CATCHUP.md) first for the research goal, implemented mechanisms, worked example, findings and remaining experiments. Older details are preserved in [the historical handoff](docs/archive/HANDOFF_PRE_V12.md); its uncommitted-work statements and earlier architecture directions are historical.

## Presentation

- [PowerPoint v12](output/presentations/relational-memory-proposal-v12.pptx)
- [PDF v12](output/presentations/relational-memory-proposal-v12.pdf)
- [Slide text v12](output/presentations/slide-text-v12.md), for review without PowerPoint
- [Presenter guide v12](output/presentations/presenter-guide-v12.md), including notes and sources

14 main slides and 7 backups. The styling and existing content remain, with a revised example, the requested removal on the hypothesis slide, one chart added after the equations, and related note/backup clarifications. TerraLingua retains its animated GIF. The PDF is static.

Mia now gives food beside a blue pinwheel and says only "I'm usually here." Later, a hungry agent sees the pinwheel and could recall the encounter, giving it a reason to seek Mia. There is no explicit invitation to return. The approach is an expected behavior to test, not a recorded result.

Measured hunger-aware cosine: .3702016. Final direct/associative scores at tick 100: .0106903/.2164730. Direct access crosses the .15 cutoff at 32.8173 ticks and association access at 139.0108 ticks. Only association qualifies at integer ticks 33–139 in the fixed fixture. There is no positive-time crossover: the association begins stronger. Raw cosine and final retrieval scores are different quantities.

## Runtime and scope

[associative_memory.py](code/hou_memory/associative_memory.py) supplies the live engine to [memory_agent.py](code/hou_memory/memory_agent.py). Ordinary memory and connection state persist with the world checkpoint. Preview is read-only; committed retrievals are idempotent. Default prompt exposure is up to three memories above .15 within 600 retrieved-text tokens, plus a separately capped plan. Observed entity cues are the main sources; optional memory sources are off.

Only included memories receive ordinary rehearsal. Only an included episode's strongest graph route learns when G exceeds M. Updates occur before generation and credit prompt preparation rather than action success. Max and all-retained-link crowding remain unchanged and require evaluation. No production formula or live query changed for v12.

The live embedding query does not automatically include hunger/energy. The fixture explicitly adds hunger-aware text as a stronger probe. Seeing a pinwheel can activate its connection; hunger alone does not currently activate an unseen object.

See [implementation and setup](docs/ASSOCIATIVE_IMPLEMENTATION.md) and [mathematical design](docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md). Main conditions: `hou`, `independent`, `learned`. `gated` is historical; ACT-R is outside the current experiment.

## Verification

Rerun on 2026-09-28:

- 22 current mechanism/actual-agent integration tests, all passed.
- 14 kernel checks and 15 historical graph checks, all passed: **51 total**.
- Pinwheel fixture with cached MiniLM and the stronger hunger-aware query.
- Logical audit: displacement, crowded-cue failure, seven lookup-equivalence fixtures and 10,000 fixed-seed invariant samples.
- Actual TerraLingua smoke: two agents, 12 ticks, identical memory/graph restore at tick 8, real local MiniLM, scripted generation, zero generation API calls.

Receipts: [validation-v12.json](output/analysis/validation-v12.json), [pinwheel-walkthrough.json](output/analysis/pinwheel-walkthrough.json), and [logical-audit-v11.json](output/analysis/logical-audit-v11.json). The audit retains its v11 filename because the model and counterexamples are unchanged. Generated simulation logs remain local and reproducible.

From the repository root, after setup:

```bash
code/terralingua/.venv/bin/python scripts/pinwheel_walkthrough.py
code/terralingua/.venv/bin/python scripts/logical_memory_audit.py
code/terralingua/.venv/bin/python -m unittest discover -s code/hou_memory -p 'test_*memory*.py' -v
code/terralingua/.venv/bin/python code/hou_memory/test_hou_memory.py
code/terralingua/.venv/bin/python code/hou_memory/test_memory_graph.py
code/terralingua/.venv/bin/python code/hou_memory/smoke_associative_sim.py \
  --output output/validation/my-smoke --sbert
```

## Remaining research work

The examples establish an opportunity for delayed access. Typical useful recall, appropriate decisions and greater believability need evaluation. Test labeled held-out histories with distractions, wrong cues and growing hubs. Include goal-aware queries and slower ordinary forgetting. For H2, match ordinary recall events and vary link rehearsal. Then evaluate complete trajectories and blind human judgments.

A graph and weighted lookup can implement the same one-hop association policy. This preserves the relation and its history; it does not show that relational memory lacks benefit. The experiment studies that policy, not the necessity of a graph library.

## Publication scope

The publication includes the implementation/tests, TerraLingua submodule pointer, v12 deck/PDF/text/notes, numerical evidence and reproduction scripts. The v11 PPTX remains under `output/presentations/archive/` as the design reference. Earlier local drafts, model caches, environment files and generated run logs are excluded from the commit.

The pre-existing TerraLingua `.gitignore` modification is left local. No upstream TerraLingua source change is included. Teammates' recently merged proposal/bibliography changes were retained.
