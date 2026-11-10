# Offline retrieval scoring

From the repository root, with Python 3.10 or later:

```text
python scripts/score_retrieval.py scripts/fixtures/retrieval/cases.jsonl scripts/fixtures/retrieval/signatures.csv --k 1 3 5
python -m unittest discover -s scripts -p test_score_retrieval.py -v
```

Use `--field solo_selected_memory_ids` to score the same-state baseline list.
The default field is `selected_memory_ids` and the default cutoffs are 1 and 3.

## Inputs

- **Gold JSONL.** Each case needs `run_dir`, `agent_id`, `timestep`, and
  `relevant_memory_ids` (required evidence). Optional label arrays are
  `optional_memory_ids` (useful context), `irrelevant_memory_ids`, and
  `stale_memory_ids` (superseded or contradicted for this cue). The four label
  sets must be disjoint. An explicit empty relevant set defines a no-answer
  case where no memory should be returned. Leave unresolved cases out of the
  gold file. Extra fields such as the cue and candidate history are for review.
- **Signature CSV.** Use a PR #6 `probe_signatures.py` export, or a CSV with its
  three join columns and the requested retrieval column. Match the `run_dir`
  string exactly as exported, along with the stable `agent_id` and integer
  `timestep`. Retrieval cells contain JSON arrays in audit order or `null`
  when unavailable. A blank cell also means unavailable. Repeated IDs count
  once at their first occurrence. Additional CSV columns are ignored.

For real data, establish which episodes were available at the cue and label
their IDs from the history before viewing retrievals or decisions. The
existing `anchor_v1` cards and cached GPT-5-mini judgments remain inputs to
card evaluation. They rate sampled memories, so they cannot supply a complete
independent retrieval evidence set. The scorer measures prompt inclusion and
leaves decisions, confirmed actions, and outcomes to separate analyses.

## Reading the report

- **Recall@K** averages the fraction of required IDs recovered per positive
  case. Any-evidence and all-evidence success are the shares of positive cases
  with at least one and with every required ID recovered, respectively.
- **Useful-context precision** is the share of judged retrieved IDs labeled
  required or optional. Irrelevant and stale inclusion are separate shares of
  that same denominator. IDs without labels are counted in the reported
  retrieved total, so partial judging is visible.
- **False recall** is the share of observed no-answer cases with a nonempty
  retrieval list. It uses full lists to include observed empty results and
  measures retrieval only. An empty list scores zero evidence hits on a
  positive case. Missing rows and unavailable lists are counted separately and
  excluded from all metrics. Zero denominators display as `N/A`.
- **Depth coverage.** Full-list metrics use each observed list as recorded.
  Every supported `@K` uses the same observed cases for a given field. It takes
  the first K IDs from each list, using the entire list when it is shorter and
  zero IDs when it is empty. A cutoff above the maximum observed depth is
  unsupported. Different fields can have different sets of observed cases.

## Synthetic fixtures

`cases.jsonl` contains 14 constructed histories and cues, with gold IDs assigned
from the histories independently of the hand-authored rankings in
`signatures.csv`. Weak semantic cues draw on the project matrix's LoCoMo-Plus
entry. Long gaps and updated facts draw on its LongMemEval entry. The fixtures
also cover near-miss distractors, irrelevant memories, stale memories, and
no-answer cases. Tick 91 records an empty selection, tick 130 records an
unavailable selection, and tick 142 has no signature row. The example measures
scorer behavior on synthetic inputs only. Official LoCoMo-Plus and LongMemEval
scores require their respective data and evaluation protocols.
