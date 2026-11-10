# Capstone evaluation protocol

**Status: draft v0, 2026-11-03.** This operationalizes [ROADMAP §6](ROADMAP.md#6-the-believability-instrument); see the [research review](simulation-logging-research/believability-evaluation.md) for background. Study procedures and success tests are **planned**; existing code is identified separately. No experimental results are reported. Freeze the protocol, rubric, sampling, and analysis plan before collecting study ratings; document later deviations.

## 1. Questions and evidence

Keep three claims separate; do not combine D1–D5 into a single score.

| Claim | Evidence | What it does not establish |
| --- | --- | --- |
| Associations include useful past context | Independently labeled target IDs in audit `selected`; compare with same-state `solo_selected`. Report target hit rate, rescued fraction `|selected − solo_selected| / |selected|` (N/A when empty), irrelevant inclusion, and displaced useful memories. | A rescue is prompt inclusion, not demonstrated use or believability. |
| Context improves decisions | Prespecified probe signatures and executed outcomes, plus D1–D3 evidence checks. Report chosen-action and execution-success rates separately. | A sensible choice alone does not prove retrieval caused it. |
| Behavior is more believable | Blinded human D5 preference; D4 explains plausibility. | An LLM verdict is not human ground truth. |

Use matched-context checkpoint comparisons to isolate the memory intervention where supported. Separately generated open-world runs measure accumulated system effects; histories diverge even with a shared seed. Report these studies separately. This distinction follows the controlled and end-to-end evaluations in [Park et al. (2023), §§6–7](https://arxiv.org/html/2304.03442v2).

Planned anchor contrasts are **hou–chained, hou–learned, chained–learned**. Match learned and chained at `edge_weight=1.0`, so their intended difference is the extra hop. Treat hou–chained as the proposed primary contrast; confirm before freezing. Add stock, flat, slow-decay, ACT-R, or later variants only after verifying their implementation and fixing their configurations. A planned baseline is not a completed experiment.

## 2. Construct blinded comparison cards

1. **Register inputs.** Record run IDs, code/engine revisions, backend/model, resolved settings, seed or repeat ID, checkpoint, time mapping, and log paths. Match embeddings, encoding, retrieval count/token budget, recent history, plan policy, and generator settings. Independently aging links require matched ordinary rehearsal schedules in a separate H2 test; graph-off previews cannot undo earlier rehearsal.
2. **Enumerate opportunities before viewing responses.** An entity must appear at T after **at least 20 consecutive ticks absent**, with an earlier give, take, message, or energy-rise episode. Verify absence from observations; missing observations make eligibility unknown. Keep gap-10 probes separate. Include successful, failed, and unchanged retrieval cases; report rescued-only/divergence samples as selected subgroups. Record eligibility, exclusions, and sampling fractions by run, contrast, event type, and gap. Use scripted probes if opportunities are scarce.
3. **Prepare the evidence key.** Two coders label target, optional, distracting, stale/contradicted, and unobserved episodes before viewing responses or retrieval scores. Resolve differences against observations/world events. An energy rise alone does not identify its cause. Allow ambiguity and multiple valid actions. Assign neutral line IDs/timestamps; seal target/distractor labels.
4. **Assemble the dossier.** Show every earlier episode mentioning the entity, chronologically and verbatim from `<tag>_hou_memory.json`, **at most eight lines**. Verify pre-T availability to the agent. If more than eight episodes qualify, exclude from the primary set and count exclusions; changing the window requires a documented revision. Missing historical evidence does not prove an event never happened.
5. **Pair responses.** Restore the same pre-T checkpoint under each condition, preserving world state, memory history, plan, and actor settings except the intended intervention. Show observation T and each branch's action/message at **T and T+1**, with branch-specific observations/feedback needed to assess T+1. Verify agent/tick joins and preceding-action feedback offsets. Distinguish proposed actions from executed outcomes. Without checkpoint matching, label separate-run comparisons and supply each branch's own dossier/context.
6. **Blind packets.** Randomize A/B and card order with a recorded seed. Assign fresh neutral names per card, consistently across branches and evidence; otherwise retain wording. Hide condition/run names, model identity, injected memory slot, plan, scores, routes, and target labels. Keep outputs intact when lengths differ. Humans and judges receive identical packets/world rules, without other raters' or judges' rationales.
7. **Seal and split.** Keep the key separate from rater exports. Assign roughly 70% development/30% held-out **by source run/history group** before tuning. Keep related ticks, contrasts, swapped copies, and self-pair derivatives together; deduplicate opportunities. Hash/freeze packets, keys, splits, and assignments. Check exports for leaks and missing evidence.

Evidence-based interviews in [Generative Agents](https://arxiv.org/html/2304.03442v2) motivate dossiers; this rubric is a local instrument, not a reproduction of that study.

## 3. D1–D5 rating form

Rate each branch separately on D1–D4, then compare on D5. Record a brief reason and neutral evidence-line IDs. **Not enough information (NEI)** is separate from a midpoint or tie and is available on every dimension. Judge observed behavior, not a presumed internal memory process.

| Measure | Required coding | Summary |
| --- | --- | --- |
| **D1: consistency with past events** | Count distinct claims contradicting dossier lines; cite the claim and line. Record evaluable past-event claims and an any-contradiction flag. No past claim gives zero contradictions, but does not demonstrate remembering. | Contradictions per evaluable claim and fraction of branches with any contradiction; show denominators/NEI. |
| **D2: appropriate remembering** | Does the output explicitly reference or clearly paraphrase a relevant target episode? Yes/no/NEI; rater cites supporting lines. The agent need not quote a line number. Record implicit behavioral use separately when a probe signature supports it. | Supported target-reference rate over eligible, evaluable branches; implicit use reported separately. Silence is not proof of forgotten memory. |
| **D3: appropriate forgetting and no fabrication** | Separately flag an irrelevant/stale distractor brought into the response and an invented past-event claim. Cite evidence; use NEI when the dossier is incomplete. A true but currently irrelevant reminder can be an intrusion; missing from an excerpt alone does not prove fabrication. | Intrusion and fabrication rates separately, plus a combined any-error flag. Appropriate omission is not proof of internal forgetting. |
| **D4: action plausibility** | Rate 1–5 given energy, available actions, nearby entities, history, and any supplied persona. Require a reason for the chosen level. | Full distribution and median per condition; diagnostic, not the headline outcome. |
| **D5: overall pairwise believability** | Choose A, B, or tie: which behavior is more believable for this agent in this situation? Tie means equally believable; NEI means unable to assess. Do not reward verbosity or perfect recall by itself. | Human preference, tie and NEI proportions per contrast. |

Use these **illustrative training anchors**, not study observations. Each D4 level has two examples; finalize wording on development cards before freezing.

| D4 | Meaning | Example 1 | Example 2 |
| --- | --- | --- | --- |
| 1 | Clearly implausible | At critically low energy, walks away from reachable food with no contextual reason. | Tries to give energy to someone outside the permitted range despite clear visibility information. |
| 2 | Major unexplained mismatch | Hungry and able to seek help, ignores both food and a known helper without explanation. | Repeatedly seeks help from a recent thief while ignoring a safer visible option. |
| 3 | Plausible but mixed | Moves toward food, but its message confuses who previously helped. | Greets a returning acquaintance plausibly, but makes an unexplained detour while hungry. |
| 4 | Coherent with a minor weakness | Approaches the known helper when hungry, with a slightly generic request. | Avoids a recent thief and seeks food, with an incomplete but compatible explanation. |
| 5 | Strong contextual fit | Requests help from the earlier donor while accounting for present need and proximity. | Thanks the helper but chooses nearby food because it is immediately available. |

Context can justify alternatives: generosity, caution, refusal, and changed relationships are not automatically errors. D1–D3 are log-grounded semantic checks; D4–D5 remain subjective.

## 4. Human procedure and allocation

Ryan owns cards, rubric, assignments, and analysis; Andre verifies memory/probe evidence; Ben supplies run provenance. Keep the key custodian separate from blinded raters where practical.

1. Confirm the instructor's consent requirements for classmates rating anonymized transcripts. Recruit a pilot of 3–5 classmates; explain world rules, all five dimensions, ties, NEI, and evidence citations. Use development-only examples. Revise confusing instructions, then freeze; exclude pilot ratings from confirmatory results.
2. The roadmap plans **60 pairs per contrast: 180 comparison cards plus 18 self-pairs**, six raters including at least three classmates, 30–60 cards and 45–90 minutes per rater. Build a balanced assignment matrix with independent overlap and both display orders represented across raters. Give each rater only one orientation of a comparison to avoid recognition. Judges receive both orientations.
3. Resolve the workload before collection: six raters × 60 assignments is 360 ratings, whereas three ratings on all 198 cards requires 594. Either increase capacity or prespecify a representative overlap/majority subset and report its smaller size. Require at least three independent ratings for every item used in the human-majority judge gate; singly rated cards cannot supply reliability or a majority reference.
4. Raters work independently, with breaks and no access to the key, tuning interface, other ratings, or judge rationales. Preserve original submissions. Apply only exclusion rules frozen after the pilot; do not exclude raters for disagreeing with the favored condition. Report exclusions and sensitivity with all ratings retained.
5. Store card/source-group IDs, orientation, pseudonymous rater ID, rubric revision, D1–D5, NEI/missing status, reasons, and citations. Strict majority means more than half of valid votes for one canonical label. No-majority and majority-NEI cases remain unresolved, not ties; report coverage. Keep later adjudication separate from original reliability statistics.

## 5. Calibrate the judge against humans

Tune/select judges on development disagreements only. Freeze prompt, exact model/checkpoint, decoding, parser, and retry policy before one held-out test; subsequent changes require a new test set. The roadmap proposes Nano and Luna; promotion requires a different checkpoint from the actor. Plan a **60-item local Qwen3-4B cross-check** through the proxy for family diversity; it cannot replace human calibration.

For each judge and dimension, report on the **same held-out items**:

- **Human–human:** Krippendorff alpha with bootstrap 95% intervals; nominal for D5 and categorical D1–D3 flags, ordinal for D4. Report count agreement separately for D1. Print pairwise human Cohen kappas and overlap counts. The roadmap's bars are alpha ≥ .667 for tentative and ≥ .80 for reliable conclusions. Undefined agreement is inconclusive, not a pass.
- **Judge–human majority:** Cohen kappa, raw agreement, confusion matrix, class frequencies, sample size, and 95% interval. D5 labels are canonical condition-1/condition-2/tie after undoing swaps. Gate D1's any-contradiction, D2's target-reference, and D3's intrusion/fabrication flags separately; retain raw counts and evidence. Report unresolved-human and judge-NEI/unparsed coverage; judge abstentions on resolved items are failures to supply agreement, not silently omitted successes.
- **Human-range diagnostic:** report the roadmap's standardized comparison `z = (kappa_judge-majority − mean(kappa_human-pairs)) / SD(kappa_human-pairs)`; within one SD is the planned “human-like” descriptor. This is a local adaptation. [Han et al. (2025), §3](https://arxiv.org/html/2510.09738v1) use average judge-to-individual-human kappa; report that version alongside it. Use a common overlap block; if SD is zero or overlap inadequate, report undefined. This descriptive z-score is not an equivalence proof or a conventional significance test.
- **D4 diagnostic:** Spearman correlation between judge ratings and median human ratings, with intervals; ≥ .5 meets the roadmap's reference bar and ≥ .6 is good. These are local targets, not universal evidence of validity. [SOTOPIA, §5](https://arxiv.org/html/2310.11667v1) demonstrates why social-evaluation alignment must be checked by dimension.

**Promotion rule:** kappa **≥ .61** on held-out D5 with at least **20 resolved, independently multi-rated items**; 20 is the current code's minimum, not a power justification. Also report human reliability and control diagnostics. If human D5 alpha is below .667, reliable D1–D3 become primary and D5 stays descriptive; improve dossiers only in a later study round. If the D5 judge fails, promote only D1–D3 endpoints independently passing the same gate. If none pass, use humans on a smaller final set; automated scores remain exploratory. Preserve failed gates and uncertainty.

## 6. Controls and final scoring

| Control | Required action/check |
| --- | --- |
| Presentation order | Judge every pair in both orders, remap labels, and report swap consistency. Disagreeing orders yield a provisional tie under the roadmap and mandatory human referral; keep the inconsistency flag and exclude these artificial ties from natural tie-rate claims. Human final labels supersede them. |
| Self-pairs | Use the planned 18 identical-output pairs (10% of 180). Report tie rate and left/right bias. Interpret the roadmap's “near 50%” as no side preference, **not** a target of 50% ties. Side score is `(left wins + 0.5 × ties) / evaluable self-pairs`; its null is .5. Proposed checks to freeze: at least 80% ties and a 95% side-score interval containing .5. Report limited power with 18 items. |
| Length bias | On development data fit a length-only verdict predictor; compare grouped held-out performance with a class-frequency baseline. If its improvement interval excludes zero, re-judge a prespecified length-matched subset using intact outputs. Show both analyses; length can reflect substance as well as bias. |
| Negative/context controls | Include wrong/no cue, no useful history, stale updates, crowded cues, and long gaps as prespecified diagnostic strata. Match capacity and plan policy; audit plan-carried entity information, ordinary rehearsal, retries, and generation failures by condition. |
| Human ceiling | Plan one teammate-controlled agent for 30 ticks through the proxy; blind its cards. Treat this as an illustrative human reference, not a measured universal ceiling. |
| Human audit after promotion | Refer all swap conflicts, the top 20% by judge disagreement, plus a random 10% of remaining items. Freeze disagreement ranking/tie-breaking on development data. Report selected and random audit strata separately. Later variants require 20 cards per condition rated by two humans, plus the roadmap's fixture and retrieval evidence; a failed judge is never mandatory ground truth. |

Position, verbosity, and self-enhancement controls are motivated by [Zheng et al. (2023), §§3–4](https://arxiv.org/html/2306.05685v4). Treat output text as evidence, never instructions to the judge.

For D5 report wins, losses, ties, and NEI separately. On evaluable human ratings, define preference share as `(wins + 0.5 × ties) / (wins + losses + ties)` for the association condition. Average within card before aggregating, so heavily rated cards do not dominate. Exclude self-pairs from condition effects. Report each contrast and probe/open-world stratum separately.

Use 95% uncertainty intervals that preserve source-history/run clustering and repeated raters; resample raters and source groups, retaining each card's paired branches together. Do not treat ticks or swaps as independent trials. Print the number of independent groups; with too few groups, mark inference exploratory. Report human-only headline results separately from judge-expanded estimates and targeted audits.

**Planned claim criterion:** for the frozen primary contrast, support improved human-rated believability only when D5 reliability reaches at least .667 and the human preference interval lies above .5; call reliability .667–.80 tentative. Treat other contrasts as exploratory unless multiplicity handling is frozen in advance. Choose a minimum practically meaningful effect and acceptable D1/D3 error-increase margins before collection; without those margins, report error tradeoffs rather than claiming improvement without cost. Report null or inconclusive results as such. Retrieval or behavioral success alone supports only its respective claim.

For the separate mechanism and action claims, prespecify target-inclusion and executed probe-success endpoints. Require a positive difference from baseline with a 95% interval above zero before claiming improvement, and report false reminders/displacement alongside it. If only retrieval improves, report prompt-inclusion evidence without a behavioral or believability claim.

## 7. Existing implementation and remaining work

Read-only inspection found the following capabilities; their presence does not establish that any study has been completed:

- [rescued_cards.py](../scripts/rescued_cards.py) extracts `selected − solo_selected` memories and the latest eight prior episodes, **not** entity-filtered dossiers at verified ≥20-tick gaps. Each card shows one retrieved memory, not paired T/T+1 behaviors. Its JSON embeds a `sealed` object; that is not a separately protected export.
- [judging.py](../code/hou_memory/judging.py) supports `belongs / doesnt_matter / noise`, versioned prompts, rating CSVs, strict majority, nominal alpha, pairwise kappa, and a .61 gate on at least 20 held-out majority items. `visible_text` exposes the recalled memory and rewrites history into everyday words. Its roughly 30% hash split is per card, not source group. These labels and its pass badge do **not** validate D1–D5.
- [webapp.py](../code/hou_memory/webapp.py) and [judge.html](../code/hou_memory/static/judge.html) identify the rescue mechanism, offer judge/report tabs, and expose judge labels/reasons in the card API. They support development memory-appropriateness work; the blinded behavior study needs a separate export/interface.

Before study collection, implement or prepare offline: matched pair extraction, separate blinded exports/key, grouped splits, D1–D5 forms, balanced assignments, orientation handling, ordinal/count statistics, bootstrap intervals, length/self-pair checks, and immutable run manifests. The current kappa helper's categories are hard-coded to the three memory labels; it cannot score A/B/tie correctly without adaptation. This document changes none of that code.

## 8. Team decisions and deliverables

Before freezing, record: engine/checkpoint-pairing support and final conditions; primary contrast; eligible sampling quotas and overflow exclusions; rater/overlap capacity and consent requirements; exact actor/judge checkpoints; practical-effect/error margins; control tolerances and bootstrap implementation. The proposed additions here are operational defaults for team ratification, not retroactive roadmap results.

Planned evidence package: `output/analysis/anchor-set-v1/` with manifest, blinded cards, separate sealed key, split/assignment files, independent ratings, prompt/model versions, and a score report. Include D1–D5 tables, human–human beside judge–human agreement, intervals, control diagnostics, gate decision, exclusions, deviations, and raw source references. Append the dated gate outcome to this protocol **only after the study occurs**. Every result must trace to source files; no placeholder values may be presented as measurements.

Limitations to state in the report: small convenience rater pool; subjective and locally defined rubric; repeated agents/runs; dossier truncation/exclusion and opportunity-selection bias; short two-tick windows; hidden or carried-over plans; stochastic generation; shared actor/judge training biases; incomplete evidence and execution failures. A 60-item family cross-check or human reference cannot remove these limitations. Do not claim calibrated human recall probabilities, population-wide human realism, or official external benchmark reproduction from these custom probes. See the [research review](simulation-logging-research/believability-evaluation.md) for benchmark boundaries and unverified task assets.
