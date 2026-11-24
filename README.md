# Human-like agent simulation

Relational memory for agents in multi-agent simulations. Direct retrieval and associations have separate forgetting and rehearsal processes. We test whether associations recover appropriate context from an agent's own experiences and make its remembering and behavior more believable.

CMSC473/673 project: Andre Atkins, Ben Sadorra and Ryan Shechtman. The [course proposal](proposal/proposal.pdf) supplies the original research context. The implementation and current design describe the active experiment.

## Start here

- **[Team catch-up](docs/TEAM_CATCHUP.md):** thesis, worked example, mathematical distinctions, implementation, findings and remaining experiments.
- **[Current PowerPoint](output/presentations/relational-memory-proposal-v12.pptx)** · **[PDF](output/presentations/relational-memory-proposal-v12.pdf)** · **[Slide text](output/presentations/slide-text-v12.md)** · **[Presenter guide](output/presentations/presenter-guide-v12.md)**.
- **[Handoff](HANDOFF.md):** current files, commands and validation status.

The deck has 14 main slides and 7 backups. Its TerraLingua GIF plays in PowerPoint Slide Show; the PDF is static. Main slides describe the proposed experiment. The team catch-up distinguishes implemented mechanisms from untested behavioral predictions.

## Grade the cards

Grading needs no API key, no TerraLingua install and no recorded worlds. Tested on a fresh clone with Python 3.12 on 2026-11-24.

```bash
git clone https://github.com/bloohunnits/human-like-agent-simulation.git
cd human-like-agent-simulation
python3 -m venv .venv
.venv/bin/python -m pip install -r code/hou_memory/requirements.txt
.venv/bin/python code/hou_memory/webapp.py
```

Then open http://127.0.0.1:5057/rescued and type your first name (Andre, Ben or Ryan). On Windows the Python is `.venv\Scripts\python`.

- The first start downloads a small language model and takes a minute or two. Wait for the line "Rescued set and judge".
- The page shows how many cards are left and saves each answer as you go. Stop any time and start the server again later: it opens on your next card.
- Your answers are saved on your own machine in `output/analysis/rescued/anchor_v4/` as `pairs_<name>.csv` and `ratings_<name>.csv`. They are not in git. When you finish, send those two files to Andre.
- To pick up new cards later, run `git pull` and start the server again. Finished cards stay finished.
- Do not open the page with `?admin` until you are done. That view shows which memory is which.

If everyone is on the same wifi, one person can run `.venv/bin/python code/hou_memory/webapp.py --host 0.0.0.0` and the others open `http://<that computer's address>:5057/rescued`. Then all answers land on one machine and nobody sends files.

How the list is built is in [docs/INTEGRATION.md](docs/INTEGRATION.md), under "The grading list".

## Current system

An observed entity can cue a linked episode even when semantic similarity plus ordinary decay leaves that episode below the retrieval cutoff. The connection has its own clock and also fades. Only memories included in the prompt rehearse; only their winning graph routes can learn.

The main conditions are `hou`, `independent`, and `learned`. The older `gated` condition remains available for design-history comparisons. ACT-R and partial reinforcement of unselected neighbors are outside the active experiment. The read-only `/audit` UI exposes scores, routes, selection decisions, state updates, the exact prompt and the resulting action.

Better context and human believability remain empirical questions. The [logic audit](docs/PROPOSAL_LOGIC_AUDIT.md) identifies competing memories, crowded cues and other tradeoffs. A weighted lookup can represent the same one-hop associations; this equivalence preserves their relational information.

## Reproduce the worked example

After following [environment setup](docs/ASSOCIATIVE_IMPLEMENTATION.md#first-checkout-and-dependencies):

```bash
code/terralingua/.venv/bin/python scripts/pinwheel_walkthrough.py
code/terralingua/.venv/bin/python scripts/logical_memory_audit.py
```

The Mia–pinwheel fixture includes a hunger-aware query. Under its fixed defaults, both methods retrieve the episode initially, only associative retrieval does so at ticks **33–139**, and neither does from tick 140 onward without rehearsal. These are calculated access windows, not measured human probabilities or observed approach behavior.

Verification on 2026-09-28: **51 tests passed**, and a two-agent, 12-tick TerraLingua smoke passed with save/resume, local MiniLM and scripted generation. No generation API calls were made. See [validation evidence](output/analysis/validation-v12.json).

## Repository map

| Location | Purpose |
| --- | --- |
| `code/hou_memory/` | Retrieval engine, simulation wrapper, tests, audit UI and configs |
| `code/terralingua/` | TerraLingua upstream submodule |
| `code/terralingua-relational-memory/` | Fork of a newer vesion of TerraLingua |
| `code/integration-layer/` | Server to mediate between TerraLingua and Model requests |
| `code/memory-storage/` | Manage the databases to hold chat logs and vector embeddings |
| [Implementation](docs/ASSOCIATIVE_IMPLEMENTATION.md) | Setup, run commands, defaults and integration limits |
| [Memory design](docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md) | Equations and design history |
| `scripts/` | Reproducible examples and logical audit |
| `scripts/presentation/` | Presentation source and export instructions |
| `output/presentations/` | Current presentation, PDF, text, notes and GIF |
| `output/analysis/` | Numerical fixtures, chart exports and validation receipt |
| `proposal/` | Original course proposal and bibliography |

Earlier directions remain in [the archived overview](docs/archive/EARLY_PROJECT_OVERVIEW.md), [historical handoff](docs/archive/HANDOFF_PRE_V12.md), and the ACT-R design documents. They are context, not the current experiment specification.
