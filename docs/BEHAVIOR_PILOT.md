# Offline D1–D5 behavioral pilot

This exporter turns manually reviewed candidate pairs into blinded packets and a
blank human rating sheet. The bundled manifest contains invented observations,
episodes and behavior for testing the format.

```powershell
python -B -m unittest scripts.test_build_behavior_pilot scripts.test_collect_behavior_pilot scripts.test_package_behavior_pilot -v
New-Item -ItemType Directory -Path output/validation -Force | Out-Null
python -B scripts/build_behavior_pilot.py scripts/fixtures/behavior_pilot/synthetic_manifest.json --seed 17 --out-dir output/validation/behavior-pilot-synthetic-v2
```

Choose a new destination for each export. For per-rater assignments, distribute
only the matching ZIP created by the packaging workflow below.

| Output | Contents |
| --- | --- |
| `rater/packets.json` | A/B evidence, comparison type, export ID, rubric revision and neutral citation IDs. Dossier ages are relative to T. |
| `rater/rating_sheet.md` | One blank form per packet, with the same export ID and rubric revision. D1 records two counts and a flag separately. D1–D4 are per branch, D5 is pairwise. Tie, NEI and missing have separate meanings. |
| `private_key.json` | Export ID and rubric revision, original conditions and identities by packet and side, source and citation mappings, checkpoint provenance, gold labels, seed and manifest SHA-256. |

The seed fixes packet order and A/B assignment for a given input and Python
version. The export ID is derived from the manifest content, seed and rubric
revision. For assigned ratings, join using the assignment ID and packet ID.
Keep the manifest and key under custodian access.

## Optional per-rater assignments

Use `--raters 3`, `--raters 4`, or `--raters 5` to assign a common pilot block.
The synthetic example is a workflow dry run and provides no human-evaluation
evidence.

```powershell
python -B scripts/build_behavior_pilot.py scripts/fixtures/behavior_pilot/synthetic_manifest.json --seed 17 --raters 3 --out-dir output/validation/behavior-pilot-assigned-demo
```

Each rater gets every development comparison once, in a seeded shuffled order.
The schema v2 synthetic example treats its entire block as development. The two
orientations are split 2/1, 2/2, or 3/2 per comparison for 3, 4, or 5 raters,
respectively. For each rater, each condition appears on A and B equally often
across the block when its total appearances are even, and differs by one when
its total is odd. In schema v3, the group allocation defines the pilot block.

The exporter writes each assignment to its own directory under `rater/`. The
JSON packet order matches the sheet, which has the assigned pseudonymous rater
ID filled in. Ask raters to complete their ratings independently.

The private key retains canonical packet branches. Its `assignments` entries
list each rater's packet IDs in display order and map each displayed side to the
canonical side. Use that mapping on the side prefix of a displayed citation to
look it up in the key. Each rater's `packets.json` and Markdown sheet show an
assignment ID bound to the export, rubric, rater, and exact packet set with its
display orientations. Keep the manifest, key, and original submissions under
custodian access. Omit `--raters` for the standalone layout above.

## Offline rating collection

Generate a blank JSON submission from the assigned `packets.json`. Use the
Markdown sheet to read the questions and fill the JSON responses. The template
needs only rater-visible files and fills all ratings with `MISSING`.

```powershell
python -B -m scripts.collect_behavior_pilot template output/validation/behavior-pilot-assigned-demo/rater/R01/packets.json --output output/validation/synthetic-R01.json
```

The UTF-8 JSON object has exactly these fields: `submission_version` (integer
`1`), `export_id`, `rubric_revision`, `rater_id`, `assignment_id`, `synthetic`
(boolean copied from the export), and `ratings` (array). Keep the generated
identifiers and flag. Submit one row for every assigned packet, including cards
with unanswered fields. Each row has `packet_id`, `branches` (an object with
keys `A` and `B` for the **displayed** sides), and `d5`. Each branch has the nine
fields below. Each field, including `d5`, is an object with exactly `value`,
`reason` (string), and `evidence_ids` (array of displayed citation strings).

| Field | Values for an evaluable rating |
| --- | --- |
| `d1_evaluable_claims`, `d1_contradictions` | Nonnegative integers |
| `d1_any_contradiction` | `"Yes"`, `"No"` |
| `d2_relevant_reference`, `d2_implicit_use` | `"Yes"`, `"No"` |
| `d3_intrusion`, `d3_fabrication`, `d3_any_error` | `"Yes"`, `"No"` |
| `d4_plausibility` | Integer 1–5 |
| `d5` | `"A"`, `"B"`, `"Tie"` |

Every field also accepts the case-sensitive strings `"NEI"` and `"MISSING"`.
For example, a D5 object could be `{"value":"Tie","reason":"Both responses fit the situation.","evidence_ids":["A-D0","B-D0"]}`.
Evaluable ratings require a nonblank reason. NEI and missing ratings can carry a
reason or use an empty string. Enter citations from the displayed packet and
side for D1–D4, or either side for D5. Use an empty array when no citation
applies. Values, reasons, and citation order are retained as supplied.
Evaluable D1 contradictions cannot exceed evaluable claims, and an evaluable
contradiction flag must agree with the count. An evaluable D3 any-error flag
must agree with the known intrusion and fabrication flags. `NEI` and `MISSING`
remain independent values for each field.

The custodian validates files against the private key for the same export. Use
one private archive directory per export so subsequent batches detect duplicates.
Keep the original JSON files separately.

```powershell
python -B -m scripts.collect_behavior_pilot validate output/validation/synthetic-R01.json --key output/validation/behavior-pilot-assigned-demo/private_key.json --out-dir output/validation/behavior-pilot-validated-demo
```

Validation rejects stale or mismatched identities, extra or omitted packets,
duplicates, malformed values, and invalid citations. The command checks the
whole batch before writing and creates one `<assignment_id>.json` per accepted
submission without replacing any file. Each archive record contains the submitted
JSON and SHA-256 hashes of the original file and private key. Join to the private
key later for canonical orientation, history groups and self-pair status.
The custodian confirms who submitted each file. Assignment IDs verify which
assignment a response names. Earlier per-rater exports can be regenerated to
obtain assignment IDs. Files labeled `synthetic` exercise ingestion only.

## Per-rater sharing packages

Run the packager offline using the original private manifest and the export
created with `--raters`. Use a new ZIP output directory and a new report file
under a custodian-only parent directory. Keep the report outside the ZIP
directory. The custodian should retain the manifest, original export, private
key, original submissions, and validated archives in custodian-only storage.

```powershell
python -B -m scripts.package_behavior_pilot scripts/fixtures/behavior_pilot/synthetic_manifest.json --export-dir output/validation/behavior-pilot-assigned-demo --out-dir output/validation/behavior-pilot-zips --report output/validation/behavior-pilot-custodian-report.json
```

Each `R01.zip`, `R02.zip`, or corresponding archive contains exactly four files
at its root. `packets.json` and `rating_sheet.md` contain that rater's assignment.
`response_template.json` is ready to fill in and submit to the collector.
`instructions.md` explains how to complete and return the JSON. The packager
replays the export against the original manifest and key, checks the existing
packet and sheet against that replay, then verifies each ZIP's allowlist,
contents, and metadata after writing. The separate report lists each ZIP,
rater and assignment ID, card count, and verified SHA-256 digest. Choose fresh
paths for a new run because existing output directories and report files are
refused.

The ZIP format uses fixed member order, timestamps and file modes, and stores
the files without compression. Repeating the same export with the same Python
version produces identical ZIP bytes and report hashes. Local creation and
modification times can differ. Use the exporter Python version to replay older
exports, since the seeded assignment order depends on that version.

ZIPs are unencrypted. The custodian should open each ZIP locally and review the
rendered packet, sheet, template, and instructions for the intended assignment
and for indirect identifying clues. Check actual folder and link permissions in
the sharing service before distribution. Grant each rater access to their own
ZIP and verify that the private inputs, report, and submissions are restricted
to the custodian. Synthetic packages exercise this workflow only.

## Source-history groups in manifest v3

Use `schema_version: 3` for genuine pilot assignments. Schema v2 still works
for legacy standalone exports and synthetic assignment dry runs. A legacy
standalone export contains the unsplit comparison list. Use schema v3 for
development materials distributed to pilot raters.

Schema v3 keeps the evidence fields below and requires two private identifiers.

| Field | How to declare it |
| --- | --- |
| `pairs[].source_group_id` | Assign one ID to the entire connected set of comparisons. A comparison joins both branches into the same group, even when they come from separate histories. Reuse the group across linked contrasts, repeated ticks, swapped presentations and future derivatives of a parent pair. |
| `pairs[].branches[].provenance.source_history_id` | Use one stable origin-history ID across related runs, resumes, and restores, including restores from different checkpoints of that history. A group may contain several histories connected by comparisons or overlapping evidence. |

Use nonnumeric identifiers composed of ASCII letters, digits, `.`, `_`, `:`,
or `-`, beginning with a letter or digit. Use globally scoped source artifact
references with an optional `#` record locator. Cite the same artifact with the
same path even when the record or field differs. The exporter compares paths
lexically, folding separators and redundant components without opening the
source files. Drive and UNC paths use case-insensitive comparison. Record
locators keep their case and record identity. The exporter rejects missing
IDs and group declarations that disagree for a shared run, history, checkpoint,
or cited evidence artifact. It relies on the reviewer to declare shared ancestry
and resolve paths or copied artifacts that have different names.

The seed allocates whole groups before the pilot block is built. The held-out
target is 30% of groups, rounded to the nearest group with half ties rounded
up. Group sizes can differ, so the card ratio can differ from the group ratio.
One group gives a development-only export. Two groups give one group per split.
The CLI and private key report group and card counts and flag when the held-out
split is empty or relies on a single group.

Both standalone and per-rater v3 exports write only development cards to
`rater/`. The private key keeps every canonical packet's source group, split and
provenance, plus the group allocation and development-only rater mappings. Keep
the key and full manifest with the custodian and freeze them with the seed before
tuning. The exporter withholds held-out packet bodies from the export directory.
Synthetic v3 splits test allocation and isolation mechanics only.

Run a synthetic grouped dry run from the repository root:

```powershell
New-Item -ItemType Directory -Path output/validation -Force | Out-Null
python -B -c 'import json; from pathlib import Path; from scripts.test_build_behavior_pilot import grouped_manifest; Path("output/validation/synthetic-grouped-manifest.json").write_text(json.dumps(grouped_manifest()), encoding="utf-8")'
python -B scripts/build_behavior_pilot.py output/validation/synthetic-grouped-manifest.json --seed 17 --raters 3 --out-dir output/validation/behavior-pilot-grouped-demo
```

## Optional hidden self-pair controls

Add `--self-pairs 1` or `--self-pairs 2` to a schema v3 export. The seed selects
distinct development parents and one validated branch from each. The same
dossier, turns, and outcomes appear on both sides. Each control uses fresh
neutral entity aliases and equivalent A/B evidence citations. It uses the same
packet and sheet format as other comparisons. The CLI rejects requests for
more controls than eligible development parents.

```powershell
python -B scripts/build_behavior_pilot.py output/validation/synthetic-grouped-manifest.json --seed 17 --raters 3 --self-pairs 2 --out-dir output/validation/behavior-pilot-self-pair-demo
```

Each rater receives each control once in their shuffled block. Controls are
excluded from the condition orientation calculation. Split card counts describe
the ordinary comparisons, and the CLI reports control counts separately.

The custodian can identify controls by `packets[].control.type` in the private
key. Each control keeps the parent's group and split. The key records the parent
pair ID and the zero-based index of the selected branch in the manifest. For
later D5 diagnostics, use the displayed A/B labels. Report tie rate over
evaluable control ratings and side score as `(A wins + 0.5 × ties) / evaluable
control ratings`. A side score of 0.5 indicates balanced side preference.
Exclude controls from condition-preference estimates and report NEI and missing
separately. One or two controls support a pilot check, with limited evidence
about rater bias. Synthetic outputs exercise the workflow only.

## Manifest v2

See the [synthetic manifest](../scripts/fixtures/behavior_pilot/synthetic_manifest.json)
for a complete example. The top level contains `schema_version: 2`, a boolean
`synthetic`, a list of `private_terms`, and a nonempty list of `pairs`.
Set `synthetic` to false for genuine evidence. Each pair supplies a unique
`pair_id`, a private `review` reference to the completed manual review, a
`comparison`, and exactly two `branches` with distinct run IDs and conditions.
Use one stable ID per entity within a run and resolve any ambiguous display
names before assembling the manifest. Use distinctive source and pair IDs so
the leak check can distinguish them from ordinary transcript wording.

Each branch supplies `run_id`, focal `agent_id`, cue `entity_id`, `condition`,
candidate `tick` (T), `last_seen_tick`, `names`, and private `provenance`.
`names` maps stable entity IDs to lists of original names and aliases. Include
the focal agent, cue, other observed entities and any named entity in visible
text. Use names or nonnumeric tags in excerpts for alias substitution. Every
evidence row carries its own `run_id`, focal `agent_id`, integer `tick` and private
`source_id` identifying the exact source record or field. Use canonical source
references and one nonnegative integer decision clock per run.

The same run and evidence can recur across contrasts. Keep run registration,
shared record fields and source references consistent. Observation rows may omit
display text outside the two rated turns. Cue association and gold labels on
episodes can vary by contrast. Each distinct episode source within a run must
keep its original episode and agent ID.

| Branch field | Reviewed evidence |
| --- | --- |
| `observations` | One row for every tick from `last_seen_tick` through T+1. `present` lists stable IDs observed through the defined cue channel. An empty list means a real observation with no entities present. The cue must be present at last seen and T, with at least 20 complete absent ticks in between. At T and T+1, supply `text` with relevant energy, scene, available actions, incoming messages, and prior-action feedback when available. |
| `decisions` | One row at T and one at T+1 with the logged `action` including relevant parameters, and a verbatim `message`. Use an empty string for logged silence. Supply the correlated `selection_id` when using a fork receipt. |
| `outcomes` | One row per decision tick. In addition to `tick`, supply the exact `decision_source_id`, the independent outcome `source_id` and `source_kind`, the normalized `execution_tick` and `recorded_tick`, `status` (`confirmed`, `failed`, or `unknown`) and nonblank `feedback`. See the timing rules below. |
| `episodes` | Supply the reviewed pre-T episodes mentioning the cue and available to the agent. Each has an `episode_id`, cue `entity_id`, original `text` and optional private `gold_label`. The exporter orders by tick and episode ID, checks source uniqueness and rejects dossiers exceeding eight episodes. |
| `dossier_review` | Cite the pre-T `source_id` with `source_kind` of `pre_t_checkpoint` or `decision_audit`, and `next_tick` equal to T. Cite the separate `review_source` and set the three `attestations` flags `available_before_t`, `complete` and `verbatim` to true only after checking the underlying evidence. |
| `event` | Cite a qualifying `kind` with cue `entity_id` and a dossier `episode_id`. Here `tick` is the episode tick. Keep the actual normalized `execution_tick` and `recorded_tick` separately, with execution at or before recording and both at or before the episode. Use `world_transfer` for a corroborated `give` or `take`, `received_message` or `world_broadcast` for `message`, and `energy_observations` for `energy_rise`. An energy rise alone establishes no cause. |
| `provenance` | Private source and run registration details. Retain log locations, source history group, engine and code revisions, model/backend, resolved settings, seed, and tick/resume mapping. |

### Outcomes and feedback

| `source_kind` | `recorded_tick` relative to the decision |
| --- | --- |
| `world_event` or `probe_transition` | Same tick. Use the world processing step. |
| `upstream_feedback` | Following tick. Feedback on the T+1 action therefore comes from T+2. |
| `fork_receipt` | Same or following tick, using the receipt's response step. Match the decision's `selection_id`. A reported `effect_status` of `confirmed` supports confirmation. `unknown` and `not_observed` map to `unknown`. Cite an independent refusal event for `failed`. |

Set `execution_tick` to the world processing tick in the decision clock,
including explicit refusals. For a fork receipt it can be between the
decision and the response step. Use null for unknown execution timing when
the effect is unresolved. Missing records make the pair ineligible. The
packet shows when the evidence was recorded relative to T. Keep chosen
actions, world effects and message delivery separate. Each independent
outcome source and each receipt selection may support only its own decision.

### Checkpoint comparisons

Use `matched_checkpoint` for two reviewed restores of the same pre-T state.
Each branch's provenance needs the same lowercase SHA-256 `checkpoint_id` for
the parent snapshot, a distinct `restore_source` and `next_tick` equal to T.
Put all controlled settings in `provenance.controls`. It requires
`engine_revision`, `actor_model`, `actor_backend`, nonempty
`generator_settings`, `actor_settings` and `environment_settings` objects,
and lowercase `world_state_sha256`, `memory_history_sha256` and `plan_sha256`
digests. Include all settings held constant, such as encoding, retrieval budget
and plan policy. The pair's review should document the intended intervention.
The exporter compares all controls, the run seed if supplied, and the relevant
pre-T context across branches.

The export labels these comparisons **reviewer-attested**. The reviewer checks
the actual checkpoint, restores and source evidence. The exporter checks the
manifest's structure and internal agreement and records the cited references
and attestations in the private key. It does not open the source records or
verify checkpoint contents, event occurrence, dossier completeness, pre-T
availability or verbatim fidelity.

Use `separate_runs` for explicitly paired independent histories. Each branch
supplies its own evidence and the packet is labeled exploratory, even when
runs share a seed.

## Connecting genuine logs

- Join agent records by run, stable `agent_tag`, and decision `timestamp` from
  `agent_logs/<tag>.jsonl`. Normalize string ticks to integers and resolve
  duplicate resume segments against provenance. Build presence from complete
  `observation` records and an unambiguous name-to-tag/landmark mapping. Verify
  cue visibility or audibility consistently across the absence window.
- Use the parsed `action.action`, `action.params` and `action.message` for the
  chosen behavior. Check world event JSONL or fork execution receipts for
  effects and refusals. Upstream `info` returned by `env.step` is consumed at
  the following decision. Check the engine-specific world tick before joining.
  `messages.json` records broadcasts. Recipient observations are evidence
  for reception.
- Take episode text and creation ticks from `<tag>_hou_memory.json` or its
  checkpoint equivalent. Verify the complete entity-specific history against
  a pre-T snapshot or equivalent contemporaneous evidence. A final memory
  dump can omit episodes, and episode action text may describe intent.
- Inventory private literals in `private_terms` that are absent from structured
  fields, including filename fragments, bare artifact labels without path syntax,
  unlisted composite identifier variants,
  model or condition aliases, and label strings. Use namespaced gold codes such
  as `gold:target` when ordinary episode text contains words like `target`.
  The exporter checks known source references and their artifact paths before
  replacing names, preserves other episode wording, and scans the rendered packet
  and sheet. The packager applies the same checks to all four ZIP members.
  Numeric metadata is checked in labeled assignments such as `seed=8675309`
  so ordinary gameplay numbers remain intact. Review the rendered rater files
  for unlisted identifiers, unlabeled metadata and indirect clues before sharing.
