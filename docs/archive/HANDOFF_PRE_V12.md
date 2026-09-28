# Historical snapshot before the v12 publication

This is an archived record. Status, uncommitted-work statements and earlier model directions are superseded by [the current handoff](../../HANDOFF.md). Code paths in prose are relative to the repository root.

# Current handoff: independent associative memory implemented

2026-09-28 logical review: use `output/presentations/relational-memory-proposal-v11.pptx`, its matching PDF and `presenter-guide-v11.md` as the current presentation. Read [PROPOSAL_LOGIC_AUDIT.md](../PROPOSAL_LOGIC_AUDIT.md) before making stronger claims. The independent-decay and rehearsal arithmetic is consistent, and all 22 current mechanism/integration tests passed again. A reproducible audit confirms that default-k=3 graph boosts can displace a relevant memory, hub crowding can defeat apparent persistence, and the entity-only condition is equivalent to a weighted lookup index. Query dependence is now visible in the example slide. The chart explicitly holds the graph fixed. Rehearsal credits prompt preparation, not successful actions. Core production formulas were not changed; one encoder docstring's unsupported literature attribution was removed. Exact counterexamples and bounds are in `output/analysis/logical-audit-v11.json`, produced by `.proposal-build/logical_audit_v11.py`.

2026-09-27 continuation. This section supersedes the older session notes below.

The live agent now uses `code/hou_memory/associative_memory.py`, with independent memory/edge decay, persistent bounded edge learning, entity-only sources by default, positive cutoff, top-k and token limits, and full transaction traces. Four launcher presets (`hou`, `gated`, `independent`, `learned`) share the same source policy and budgets. Optional bounded direct-memory sources are implemented but off in the main condition. No partial neighbor refresh or ACT-R integration was added.

Read [docs/ASSOCIATIVE_IMPLEMENTATION.md](../ASSOCIATIVE_IMPLEMENTATION.md) for the final logic audit, runnable commands, controls and remaining limitations. [The mathematical design](../INDEPENDENT_GRAPH_MEMORY_DESIGN.md) is updated to implemented status. H1/H2 score effects are explicitly mechanism invariants; selective recall and simulation believability are the empirical questions.

The simulation now embeds memory and graph state in the world's checkpoint, so resume cannot load a newer standalone dump. Headless stdin and optional-world-state restore bugs are handled in the launcher; checkpoints and memory dumps are atomic. Retrieval retries are idempotent. Legacy checkpoints without synchronized memory fail explicitly and their dumps remain viewable.

Prompt history no longer bypasses retrieval by default. The plan has a real 60-token cap. The episode writer distinguishes chosen actions from confirmed outcomes. Configured place labels are visible only inside the observation footprint. Initial names remain unique across the spawn sequence.

The new read-only `/audit` page shows actual scores, selections, route alternatives, updates, and the exact prompt/action attached to each retrieval. The old `/` sliders are labeled as the historical what-if console. Launch with `--run-dir` to inspect a specific `agent_logs` directory outside the standard run directory.

Validation: 22 new unit/integration tests plus the existing 14 kernel and 15 historical graph checks passed. A real TerraLingua run with two agents and cached local MiniLM completed 12 ticks, restoring identical memory/graph state at tick 8; generation was scripted, with zero API calls. This validates integration, not cognition or believability. `smoke_associative_sim.py` reproduces it. The final receipt is `output/validation/associative-sim-final/smoke-result.json`; its agent logs are in the sibling `logs/associative_smoke/agent_logs` directory.

Prior presentation revision, 2026-09-28: `output/presentations/relational-memory-proposal-v10-ready.pptx`, its matching PDF, and `presenter-guide-v10.md` are superseded by v11 above. It has 13 main slides and 7 backup slides. It presents the current architecture and three main conditions (`hou`, `independent`, `learned`); old-versus-new formula comparisons stay in the design history. H1 explicitly concerns experiences that semantic similarity and time decay alone miss. H2 concerns useful access across longer gaps. “Selected” is defined as text included in the prompt after cutoff, top-k and token limits.

The new example uses the actual MiniLM, observation formatter, episode writer and retrieval engine. Three conversations about bells rank above Mira's food offer, so similarity-only top-3 misses the offer even though its text also names the bell. Only Mira's episode actually observes that artifact. At the delayed probe, direct-only retrieval returns none while associative retrieval returns the offer. This is a deliberately constructed illustration, not held-out evidence. Exact inputs and numeric results are in `output/analysis/proposal-examples-v10.json`, reproducible offline with `.proposal-build/proposal_examples_v10.py`. A second calculation checks the lasting-connection chart with equal ordinary recall events. Production code was not changed in this presentation revision. The TerraLingua slide preserves the looping 15-frame GIF; the PDF is static.

All work remains uncommitted. Existing uncommitted files and the TerraLingua submodule source were preserved. No paid agent experiment was launched. Defaults still require calibration and held-out evaluation.

---

# Handoff — memory-graph capstone, implementation phase

Written 2026-09-27, closing out the session that took this repo from docs-only
to a working, tested, adversarially-reviewed implementation. Every fact below
was independently re-verified against the actual files and a live test run
before this was written — not transcribed from memory of the session.

**Read this first if you're picking this up cold.** [docs/INTEGRATION.md](../INTEGRATION.md)
is the detailed lab notebook (findings, calibration history, paper-fidelity
check); this file is the map to it.

## Continuation — ACT-R design audit, 2026-09-27

The next-phase work order in [docs/ACTR_MEMORY_DESIGN.md](../ACTR_MEMORY_DESIGN.md)
has been answered in [docs/ACTR_MEMORY_ARCHITECTURE.md](../ACTR_MEMORY_ARCHITECTURE.md).
It contains a primary-source audit, the proposed simulation-independent
architecture, representation and identity policies, isolated encoder/composer
prompts, a numerical competition/failure example, and acceptance tests/ablations.
The original pasted brief remains verbatim.

This continuation changed documentation only. The new architecture is not yet
implemented, and its planned tests have not run. Existing tests were rerun with
the TerraLingua venv: **14 kernel + 15 graph tests passed**. The numerical design
example was checked separately. Source checks used the ACT-R 7.30+ manual,
official tutorial and source distribution; related work required narrowing the
earlier novelty claims.

Next: build the inspectable reference engine and fixture runner from the new
architecture document, then calibrate and evaluate before TerraLingua integration.
All earlier uncommitted work remains uncommitted; this continuation also adds the
architecture document and updates the brief status, README, research note, and
this handoff. No commits or paid simulation runs were made. The remaining sections
below record the preceding implementation session.

## Status in one paragraph

TerraLingua runs as a git submodule, untouched. Our own layer
(`code/hou_memory/`) swaps its stock scratchpad memory for a from-scratch
implementation of the Hou et al. (CHI 2024) recall formula, verified
equation-by-equation against the paper's full text, plus an ACT-R kernel
(implemented, currently hidden pending calibration), plus a memory-relations
graph (entity hubs, three edge types, spreading activation) that provably
reduces to the bare kernel when turned off. A local web console called **God
Mode** shows all of it live, including a running sim. 29 unit tests pass, zero
failures. Two rounds of independent adversarial review (28 agents total) each
found real bugs; all were fixed and re-verified. **Nothing from this session
is committed to git yet** — see "Uncommitted work" below.

## Where everything is

```
code/terralingua/          git submodule, TerraLingua upstream, unmodified
  .venv/                    Python 3.13 venv (terralingua needs 3.13+; system default is 3.12)
  logs_nano/ logs_town/     three past sim runs, each with per-tick frames + video.mp4
  logs_town2/

code/hou_memory/           everything we built
  hou_memory.py             Hou kernel + ACT-R kernel (the math, no I/O)
  memory_graph.py           the relations graph (built on top of the store)
  memory_agent.py           HouMemoryAgent: swaps into TerraLingua's agent loop
  embedder.py                SBERT all-MiniLM-L6-v2, CPU-pinned
  run_hou_experiment.py     launcher: monkeypatches the sim, registers gpt-5-nano,
                             assigns human names, syncs respawned agents
  webapp.py                  God Mode's Flask backend
  static/index.html,style.css   God Mode's frontend
  explore.py                  terminal REPL alternative to God Mode
  test_hou_memory.py (14 tests), test_memory_graph.py (15 tests)

docs/INTEGRATION.md        the detailed findings log — read this for "why"
docs/DESIGN.md              original architecture doc (predates this session)
HANDOFF.md                  this file
```

## How to run it

**God Mode (the console), against whatever runs already exist on disk:**
```bash
cd code/hou_memory
source ../terralingua/.venv/bin/activate
python webapp.py
# -> http://127.0.0.1:5057
```
(Or via the ai-capstone launch config: `code/terralingua/.venv/bin/python code/hou_memory/webapp.py` — same thing, pinned interpreter.)

**A new sim run:**
```bash
cd code/terralingua
source .venv/bin/activate
python ../hou_memory/run_hou_experiment.py \
  --exp_name myrun --init_agents 3 --max_ts 20 \
  --max_history 4 --model gpt-5-nano --save_root ./logs_myrun
```
Needs `code/terralingua/.env` with `OPENAI_API_KEY` (confirmed set in this
environment). `gpt-5-nano` is registered by our launcher — it isn't in
TerraLingua's own model list. Add `--live_render` to watch a pygame window
animate while it runs. God Mode picks up the new run automatically (or turn
on "follow live" in its Agent knob to track a run while it's still ticking).

**Formula-level demos (no sim, no LLM, seconds to run):**
```bash
python run_walkthrough.py      # the g0=1/ticks-with-no-timescale collapse, deliberately
python demo_ada_walkthrough.py # g0 sweep across days
```

**Tests:**
```bash
python test_hou_memory.py   # 14 tests: kernel math, both formulas, serialization
python test_memory_graph.py # 15 tests: graph mechanism, reduction property, review regressions
```
Both currently pass 100%, verified by an independent agent minutes before this
was written (not from session memory).

## What's implemented, precisely

### The Hou kernel (`hou_memory.py`)
`p = (1 - exp(-r·exp(-t/g))) / (1 - e⁻¹)`, matched to the CHI 2024 paper's
Eq. 8 exactly, including the normalizer. `g` grows on recall by
`tanh(elapsed/2)` (paper's S(t)), `g0 = 1` (paper-native — this was NOT true
earlier in the session; see calibration note below).

**Calibration**: time is scaled by `time_scale` (τ), currently `25`, fed into
*both* the decay term and the growth term. This was a real bug caught by the
paper-fidelity check: an earlier version rescaled only `g0`, which left
consolidation ~25× too weak relative to decay. Fixed and pinned by
`test_time_scale_keeps_paper_consolidation_dynamics`.

**Deliberate, documented deviations from the paper** (the paper injects one
memory at threshold 0.86; we surface `top_k=5` above threshold `0.0` because
a sim agent needs richer context than a chatbot turn). The "faithful Hou"
config, if you want to run that ablation, is a **comment in memory_agent.py
only, not the live default** — set `hou_top_k = 1` and
`hou_recall_threshold = 0.86` yourself if you want to actually run it.

### The ACT-R kernel (`hou_memory.py`, same file)
Power-law base-level activation over every past retrieval trace,
`B = ln(Σ max(t−tⱼ, 0.1)^-0.5)`, `A = B + 11·r`, recall probability from the
Gaussian CDF of `(A−τ)/1.2` with `τ = 0`. **Implemented, tested, and wired
into every memory's stored data** (every recall appends a trace) — but its
God Mode display is hidden behind `SHOW_ACTR = false` in `index.html` (one
line) because `τ=0` saturates almost every score near 100%, which reads as
noise rather than signal. Flip that flag back on once τ is calibrated; no
other code changes needed, the underlying state has kept accumulating
correctly the whole time.

### The memory graph (`memory_graph.py`)
Built from data every memory already carries — no LLM calls. Hubs come from
entity tags (`person`/`place`/`object`); edges are `mentions` (memory↔hub),
`co-occurred` (same tick), `followed` (consecutive, skipped for same-tick
pairs since co-occurred covers those). At recall, `r̂ = max(r, received)`,
where `received` arrives through a hub or a direct edge, each route
independently damped by one `λ` per traversal and a fan-tax that only kicks
in above 3 hub members. **`λ = 0` reduces to the bare kernel exactly** — not
by convention, but because `spread()`'s own guard clause returns all-zero
before any hub or edge loop runs; this is asserted by
`test_lambda_zero_reduces_to_bare_kernel` and was re-confirmed live via the
API today.

Reinforcement leaks: memories that surface deepen their own edges; neighbors
that received enough activation (≥0.3) get a fractional strength bump
*without* their recall clock resetting — a leak is not a use. Transmission is
clamped so a worn edge can never push a widened score above the source's own
relevance (an earlier version could push probabilities past 100%; fixed and
covered by `test_transmission_never_amplifies_past_source`). Cue-to-hub
matching is word-boundary regex, not substring (an earlier version let
"bench" fire the "Ben" hub; fixed and covered by
`test_cue_matching_respects_word_boundaries`).

**Acceptance test, passed live against the API** (Petra/Omar hand-made
story, tick 45, `λ=0.7`, cue "picking berries near the well"): Omar's memory
scored 12% solo — below the top-3 surface cut — and 17% with the graph,
tagged `via the well`, crossing the cut. That is the project's thesis
demonstrated on real numbers, not asserted in prose.

### God Mode (`webapp.py` + `static/`)
Flask app, single-threaded on purpose (shared in-memory `STATE`, single-user
console). 14 routes: state, memories (read + cued), recall (commits
reinforcement), add, tick, set_strength, set_topk, set_lambda, set_kernel,
switch, load_sim, load_preset, and a frame server for the world-view image.

Visible controls: Agent picker (+ "follow live" for a running sim), Clock,
Strength g₀ (what-if dial over every memory's decay), Surface limit, **Graph
pull λ** (visible by default), a cue box with Show/Reinforce/Reset, and a
world-view image synced to the clock slider. The Hou/ACT-R ranking toggle
exists in the DOM but is hidden (see ACT-R note above — it's gated twice
over, by inline `display:none` *and* the `SHOW_ACTR` JS flag; flipping only
one won't reveal it). Deep links work: `?run=&agent=&cue=`.

Per-agent LLM cost is shown in dollars, computed from the run's own
`token_counts.jsonl` against a hardcoded OpenAI price table in `webapp.py`
(checked against openai.com on 2026-09-25 — re-check if it's been a while).

## Review history (why you can trust the above)

Three separate adversarial-review passes this session, each: multiple
reviewer agents read the code fresh, a separate verifier agent tried to
*refute* every claimed bug against the actual running code, only confirmed
findings were acted on.

1. **Plan-channel + console review** (15 agents): found and fixed a
   `--resume` bug that would have silently destroyed an agent's entire memory
   history, a plan-contamination bug on repeated LLM parse failures, split
   agent identity on respawn, and several God Mode polling/race issues.
2. **Hou paper-fidelity check** (6 agents, read the actual arXiv full text):
   confirmed the formula, normalizer, and consolidation update are exact
   matches; found the τ-calibration bug described above.
3. **Graph review** (13 agents): found and fixed the substring cue-matching
   bug, a duplicate-tag self-boost bug, the transmission-amplification bug,
   asymmetric edge wear, and a future-memory partial-refresh bug.

Nothing here is "should be fine" — every claim above has a passing test or a
live API call backing it, done today, not recalled.

## Uncommitted work — do this next

**Everything from this entire implementation phase is sitting uncommitted.**
Verified via `git status` just now:

Superproject (`human-like-agent-simulation`):
```
A  .gitmodules
Am code/terralingua        (submodule pointer)
 M docs/DESIGN.md
?? code/hou_memory/         <- all the code described above
?? docs/INTEGRATION.md
```
Submodule (`code/terralingua`): ` M .gitignore` only (the `.venv/`/`.env`
ignore lines added this session) — TerraLingua's own tracked files are
untouched, confirmed.

This wasn't an oversight worth alarm — commits happen when you ask for them —
but it means **a fresh clone of this repo right now would not have any of
this work**. When you're ready: commit the submodule's `.gitignore` change
first (inside `code/terralingua`), then commit the superproject (the
submodule pointer, `.gitmodules`, `docs/`, and `code/hou_memory/` together).

## Known limitations / open items for next session

- **ACT-R display is off** pending threshold (τ) calibration — the same kind
  of tuning pass g₀ went through for Hou. The math and tests are ready.
- **Graph edge weights are session-only** in God Mode — they rebuild from
  the dump on reload. Persisting them into the *live sim loop* (not just the
  console's read side) is the natural next milestone — that's where
  survival-through-connection experiments start running for real, across
  actual multi-hundred-tick lives instead of hand-made stories.
- **Faithful-Hou ablation config** (`top_k=1`, `threshold=0.86`) is
  documented but not wired as a runnable preset — currently requires editing
  `memory_agent.py` by hand.
- **Cost table** in `webapp.py` is a hardcoded snapshot of OpenAI pricing —
  will drift.
- Three past runs (`logs_nano`, `logs_town`, `logs_town2`) exist on disk for
  poking around in God Mode immediately without spending anything on a new run.
