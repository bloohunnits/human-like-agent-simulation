"""Offline validation tests using invented evidence only."""

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from collections import Counter
from pathlib import Path

from scripts import build_behavior_pilot as pilot


FIXTURE = Path(__file__).parent / "fixtures" / "behavior_pilot" / "synthetic_manifest.json"


def assignment_manifest():
    """Expand the invented pair into repeated contrasts and a cycle of conditions."""
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    template = manifest["pairs"][0]
    manifest["pairs"] = []
    for index, conditions in enumerate(((1, 2), (2, 1), (1, 3), (3, 2), (2, 4), (3, 4), (3, 4))):
        pair = copy.deepcopy(template)
        pair["pair_id"] = f"SYN-ASSIGNMENT-PAIR-{index}"
        for side, (branch, condition) in enumerate(zip(pair["branches"], conditions)):
            updated = json.loads(json.dumps(branch).replace(branch["run_id"], f"SYN-ASSIGNMENT-RUN-{index}-{side}"))
            updated["condition"] = f"SYN-ASSIGNMENT-CONDITION-{condition}"
            pair["branches"][side] = updated
        manifest["pairs"].append(pair)
    return manifest


def grouped_manifest():
    """Make seven invented cards from five histories, including repeated contrasts and ticks."""
    manifest = assignment_manifest()
    manifest["schema_version"] = 3
    pairs = manifest["pairs"]
    for index, (pair, group) in enumerate(zip(pairs, (0, 0, 0, 1, 2, 3, 4))):
        pair["source_group_id"] = f"SYN-SOURCE-GROUP-{group}"
        checkpoint = hashlib.sha256(f"invented checkpoint {index}".encode()).hexdigest()
        for branch in pair["branches"]:
            branch["provenance"].update(source_history_id=f"SYN-SOURCE-HISTORY-{group}",
                                        checkpoint_id=checkpoint)

    # Reuse one run and checkpoint in a different contrast at the same tick.
    pairs[2]["branches"][0] = copy.deepcopy(pairs[0]["branches"][0])
    pairs[2]["branches"][1]["provenance"]["checkpoint_id"] = pairs[0]["branches"][0]["provenance"]["checkpoint_id"]

    # A separate checkpoint from the same history provides a different tick.
    for branch in pairs[1]["branches"]:
        branch["tick"] += 3
        branch["last_seen_tick"] += 3
        branch["provenance"]["next_tick"] += 3
        branch["dossier_review"]["next_tick"] += 3
        for name in ("observations", "decisions", "outcomes", "episodes"):
            for row in branch[name]:
                row["tick"] += 3
        for row in [*branch["outcomes"], branch["event"]]:
            row["recorded_tick"] += 3
            if row["execution_tick"] is not None:
                row["execution_tick"] += 3
        branch["event"]["tick"] += 3
    return manifest


class BehaviorPilotTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.pair = self.manifest["pairs"][0]
        self.left, self.right = self.pair["branches"]

    def assert_invalid(self, message):
        with self.assertRaisesRegex(ValueError, message):
            pilot.build(self.manifest, 17)

    def add_contrast(self):
        pair = copy.deepcopy(self.pair)
        pair["pair_id"] = "SYN-SECOND-CONTRAST"
        other = json.dumps(pair["branches"][1]).replace(self.right["run_id"], "SYN-THIRD-RUN")
        pair["branches"][1] = json.loads(other.replace(self.right["condition"], "SYN-THIRD-CONDITION"))
        self.manifest["pairs"].append(pair)
        return pair["branches"][0]

    def test_matched_export_preserves_evidence_and_separates_outcomes(self):
        packets, key = pilot.build(self.manifest, 17)
        self.assertTrue(packets["synthetic"])
        packet, private = packets["packets"][0], key["packets"][0]
        self.assertEqual(packet["comparison"], "Matched checkpoint (reviewer-attested)")
        self.assertEqual(private["match_verification"], "reviewer_attested")
        a, b = (packet["branches"][side] for side in "AB")
        self.assertEqual(a["actor"], b["actor"])
        self.assertEqual(a["cue_entity"], b["cue_entity"])
        for side in "AB":
            branch = packet["branches"][side]
            provenance = private["branches"][side]
            episode = branch["dossier"][0]
            self.assertEqual(episode["text"],
                             f'Heard {branch["cue_entity"]} say "There is food to the west."')
            self.assertEqual(episode["ticks_before_T"], 21)
            self.assertEqual(provenance["citations"][episode["cite"]]["gold_label"], "target")
            self.assertEqual([row["when"] for row in branch["turns"]], ["T", "T+1"])
            turn, following = branch["turns"]
            self.assertEqual(following["decision"]["logged_outgoing_message"], "")
            self.assertEqual(following["outcome_feedback"]["status"], "unknown")
            self.assertEqual(following["outcome_feedback"]["recorded_when"], "T+2")
            first_source = provenance["citations"][turn["outcome_feedback"]["cite"]]
            next_source = provenance["citations"][following["outcome_feedback"]["cite"]]
            self.assertEqual(first_source["execution_tick"], self.left["tick"])
            self.assertEqual(next_source["recorded_tick"], self.left["tick"] + 2)
            self.assertEqual(next_source["decision_source_id"],
                             provenance["citations"][following["decision"]["cite"]]["source_id"])
            if provenance["run_id"] == self.left["run_id"]:
                self.assertEqual(turn["decision"]["chosen_action"], "Move west")
                self.assertEqual(turn["decision"]["logged_outgoing_message"],
                                 f'Thanks for the tip, {branch["cue_entity"]}!')
                self.assertEqual(turn["outcome_feedback"]["status"], "confirmed")
            else:
                self.assertEqual(turn["decision"]["chosen_action"], "Move east")
                self.assertEqual(turn["outcome_feedback"]["status"], "failed")

    def test_seed_is_reproducible_and_can_swap_sides(self):
        baseline = pilot.build(self.manifest, 17)
        self.assertEqual(baseline, pilot.build(self.manifest, 17))
        a_conditions = {pilot.build(self.manifest, seed)[1]["packets"][0]["branches"]["A"]["condition"]
                        for seed in range(8)}
        self.assertEqual(a_conditions, {self.left["condition"], self.right["condition"]})

    def test_missing_observation_is_not_absence(self):
        del self.left["observations"][7]
        self.assert_invalid("missing or out-of-window observation")

    def test_missing_presence_list_is_not_absence(self):
        del self.left["observations"][7]["present"]
        self.assert_invalid("present must be list")

    def test_missing_t_plus_one_evidence(self):
        for rows, message in (
            ("observations", "observation"),
            ("decisions", r"T/T\+1 decision"),
            ("outcomes", r"T/T\+1 outcome"),
        ):
            with self.subTest(rows=rows):
                last = self.right[rows].pop()
                self.assert_invalid(message)
                self.right[rows].append(last)
        del self.right["decisions"][-1]["message"]
        self.assert_invalid("message must be str")

    def test_invalid_joins(self):
        for rows, key, bad in (
            ("observations", "agent_id", "wrong-focal-agent"),
            ("decisions", "run_id", self.left["run_id"]),
            ("outcomes", "tick", 23),
            ("outcomes", "recorded_tick", 20),
            ("episodes", "entity_id", self.right["agent_id"]),
        ):
            with self.subTest(rows=rows, key=key):
                record = self.right[rows][0]
                original = record[key]
                record[key] = bad
                self.assert_invalid("join|evidence|feedback tick|cue entity")
                record[key] = original
        self.left["observations"].append(copy.deepcopy(self.left["observations"][7]))
        self.assert_invalid("duplicate evidence tick")

    def test_rejects_short_or_interrupted_gap_and_missing_return(self):
        self.left["last_seen_tick"] = 2
        self.assert_invalid("fewer than 20")
        self.left["last_seen_tick"] = 1
        self.left["observations"][20]["present"] = [self.left["entity_id"]]
        self.assert_invalid("present within claimed absence gap")
        self.left["observations"][20]["present"] = []
        self.left["observations"][21]["present"] = []
        self.assert_invalid("observed at last_seen_tick and T")

    def test_dossier_limit_and_qualifying_event(self):
        event = self.left["event"]
        event["kind"] = "move"
        self.assert_invalid("qualifying earlier event")
        event["kind"] = "message"
        event["episode_id"] = "unjoined"
        self.assert_invalid("event or episode join")
        event["episode_id"] = self.left["episodes"][0]["episode_id"]
        episode = self.left["episodes"][0]
        episode["tick"] = self.left["tick"]
        self.assert_invalid("predate T")
        episode["tick"] = 1
        self.left["episodes"].extend(dict(episode, episode_id=f"other-{i}") for i in range(8))
        self.assert_invalid("dossier must contain")

    def test_decision_source_is_not_outcome_evidence(self):
        self.left["outcomes"][0]["source_id"] = self.left["decisions"][0]["source_id"]
        self.assert_invalid("independent of the chosen action")

    def test_one_outcome_source_cannot_confirm_both_turns(self):
        first, second = self.left["outcomes"]
        second.update(first, tick=second["tick"], execution_tick=second["tick"],
                      recorded_tick=second["tick"], decision_source_id=second["decision_source_id"])
        self.assert_invalid("outcome evidence reused")
        second["source_id"] = second["source_id"].replace("/", "\\")
        self.assert_invalid("outcome evidence reused")

    def test_lexical_source_alias_cannot_confirm_both_turns(self):
        first, second = self.left["outcomes"]
        second.update(first, source_id=first["source_id"].replace("/", "/./", 1),
                      tick=second["tick"], execution_tick=second["tick"],
                      recorded_tick=second["tick"], decision_source_id=second["decision_source_id"])
        self.assert_invalid("outcome evidence reused")

        first["source_id"] = "C:/custodian/private-events.jsonl#23/movement"
        second["source_id"] = r"c:\CUSTODIAN\unused\..\.\private-events.jsonl#23/movement"
        self.assert_invalid("outcome evidence reused")
        second["source_id"] = second["source_id"].replace("#23/", "#24/")
        _, key = pilot.build(self.manifest, 17)
        branch = next(branch for branch in key["packets"][0]["branches"].values()
                      if branch["run_id"] == self.left["run_id"])
        self.assertIn(first["source_id"], {row["source_id"] for row in branch["citations"].values()})
        self.assertIn(second["source_id"], {row["source_id"] for row in branch["citations"].values()})

    def test_equivalent_source_reference_joins_and_preserves_record_identity(self):
        self.add_contrast()
        decision = self.left["decisions"][0]
        self.left["outcomes"][0]["decision_source_id"] = decision["source_id"].replace("/", "/./", 1)
        decision["source_id"] = decision["source_id"].replace("/", "\\.\\", 1)
        pilot.build(self.manifest, 17)
        self.assertNotEqual(pilot.source_identity("audit.json#/Memory"), pilot.source_identity("audit.json#/memory"))
        self.assertNotEqual(pilot.source_identity(r"audit.json#/a\b"), pilot.source_identity("audit.json#/a/b"))
        self.assertNotEqual(pilot.source_identity("audit.json#/name "), pilot.source_identity("audit.json#/name"))
        self.assertNotEqual(pilot.source_identity("C:events.json#1"), pilot.source_identity("C:/events.json#1"))

    def test_outcomes_require_decision_join_and_source_specific_timing(self):
        first, second = self.left["outcomes"]
        source = first["decision_source_id"]
        first["decision_source_id"] = second["decision_source_id"]
        self.assert_invalid("outcome join")
        first["decision_source_id"] = source
        second["recorded_tick"] -= 1
        self.assert_invalid("feedback tick")
        second["recorded_tick"] += 1
        first["recorded_tick"] += 1
        self.assert_invalid("feedback tick")
        first["recorded_tick"] -= 1
        first["execution_tick"] += 1
        self.assert_invalid("execution tick")

    def test_fork_receipt_requires_selection_join_and_effect_evidence(self):
        branch = self.left
        for index, (decision, outcome) in enumerate(zip(branch["decisions"], branch["outcomes"])):
            selection = f"SYN-SELECTION-{index}"
            decision["selection_id"] = selection
            outcome.update(source_kind="fork_receipt", selection_id=selection,
                           status="unknown", effect_status="unknown")
        pilot.build(self.manifest, 17)
        branch["outcomes"][0]["status"] = "confirmed"
        self.assert_invalid("reported effect")
        branch["outcomes"][0]["effect_status"] = "confirmed"
        branch["outcomes"][0]["recorded_tick"] += 1
        branch["outcomes"][0]["execution_tick"] += 1
        pilot.build(self.manifest, 17)
        branch["outcomes"][1]["selection_id"] = branch["decisions"][0]["selection_id"]
        self.assert_invalid("selection join")
        branch["decisions"][1]["selection_id"] = branch["decisions"][0]["selection_id"]
        self.assert_invalid("conflicting receipt selection")

    def test_event_and_dossier_attestation_require_distinct_evidence(self):
        event = self.left["event"]
        event["kind"] = "give"
        self.assert_invalid("qualifying earlier event")
        event.update(kind="message", execution_tick=event["tick"] + 1)
        self.assert_invalid("qualifying earlier event")
        event["execution_tick"] = 0
        review = self.left["dossier_review"]
        review["attestations"]["complete"] = False
        self.assert_invalid("attest pre-T")
        review["attestations"]["complete"] = True
        review["source_kind"] = "final_snapshot"
        self.assert_invalid("pre-T decision boundary")
        review["source_kind"] = "decision_audit"
        review["source_id"] = self.left["episodes"][0]["source_id"]
        self.assert_invalid("distinct pre-T evidence")

    def test_checkpoint_must_be_shared_and_context_must_match(self):
        provenance = self.right["provenance"]
        common = provenance["checkpoint_id"]
        provenance["checkpoint_id"] = "b" * 64
        self.assert_invalid("share a pre-T checkpoint")
        provenance["checkpoint_id"] = common
        self.right["observations"][21]["text"] += " More food is visible."
        self.assert_invalid("observed context")

    def test_matched_restore_requires_different_records_after_normalization(self):
        left = self.left["provenance"]
        right = self.right["provenance"]
        right["restore_source"] = left["restore_source"].replace("/", "/./", 1)
        self.assert_invalid("distinct branch restore")

        left["restore_source"] = "C:/custodian/restores.jsonl#12"
        right["restore_source"] = r"c:\CUSTODIAN\.\restores.jsonl#12"
        self.assert_invalid("distinct branch restore")
        right["restore_source"] = right["restore_source"].replace("#12", "#13")
        pilot.build(self.manifest, 17)

    def test_matched_controls_and_restore_provenance(self):
        provenance = self.right["provenance"]
        original = copy.deepcopy(provenance)
        for field, wrong in (
            ("actor_model", "SYN-DIFFERENT-MODEL"),
            ("actor_backend", "SYN-DIFFERENT-BACKEND"),
            ("engine_revision", "SYN-DIFFERENT-ENGINE"),
            ("generator_settings", {"temperature": 0.7}),
            ("actor_settings", {"working_history": 12}),
            ("environment_settings", {"grid_size": 40}),
            ("world_state_sha256", "c" * 64),
            ("memory_history_sha256", "d" * 64),
            ("plan_sha256", "e" * 64),
        ):
            with self.subTest(field=field):
                provenance["controls"][field] = wrong
                self.assert_invalid("conflicting matched checkpoint controls")
                provenance["controls"][field] = original["controls"][field]
        provenance["seed"] += 1
        self.assert_invalid("conflicting matched checkpoint controls")
        provenance["seed"] = original["seed"]
        provenance["restore_source"] = self.left["provenance"]["restore_source"]
        self.assert_invalid("distinct branch restore")
        provenance["restore_source"] = original["restore_source"]
        del provenance["controls"]
        self.assert_invalid("controls must be dict")

    def test_comparison_type_must_be_explicit(self):
        del self.pair["comparison"]
        self.assert_invalid("comparison must be str")

    def test_explicit_separate_runs_allow_different_ticks_and_context(self):
        self.pair["comparison"] = "separate_runs"
        self.right["provenance"] = {"source_history_id": "SYN-INDEPENDENT-HISTORY"}
        for rows in ("observations", "decisions", "outcomes", "episodes"):
            for row in self.right[rows]:
                row["tick"] += 3
        for row in self.right["outcomes"]:
            row["recorded_tick"] += 3
            if row["execution_tick"] is not None:
                row["execution_tick"] += 3
        for key in ("tick", "last_seen_tick"):
            self.right[key] += 3
        for key in ("tick", "execution_tick", "recorded_tick"):
            self.right["event"][key] += 3
        self.right["dossier_review"]["next_tick"] += 3
        self.right["observations"][-2]["text"] = "Sable Reed has 9 energy. Juniper Moss is visible."
        public, private = pilot.build(self.manifest, 17)
        self.assertEqual(public["packets"][0]["comparison"], "Exploratory separate runs")
        self.assertEqual({branch["tick"] for branch in private["packets"][0]["branches"].values()},
                         {22, 25})

    def test_identical_evidence_can_recur_in_other_contrasts(self):
        self.add_contrast()
        baseline = pilot.build(self.manifest, 17)
        self.assertEqual(baseline, pilot.build(self.manifest, 17))
        pair_orders = {tuple(p["pair_id"] for p in pilot.build(self.manifest, seed)[1]["packets"])
                       for seed in range(8)}
        self.assertEqual(len(pair_orders), 2)

    def test_reused_run_or_turn_cannot_change_across_pairs(self):
        reused = self.add_contrast()
        reused["condition"] = "SYN-CONFLICTING-CONDITION"
        self.assert_invalid("conflicting run registration")
        reused["condition"] = self.left["condition"]
        for kind, field, value in (
            ("observations", "present", []),
            ("decisions", "action", "Stay"),
            ("outcomes", "feedback", "The move failed."),
        ):
            with self.subTest(kind=kind):
                record = reused[kind][-2]
                original = record[field]
                record[field] = value
                self.assert_invalid("conflicting (observation|decision|outcome)")
                record[field] = original

    def test_duplicate_episode_source_with_different_id_is_rejected(self):
        episode = self.left["episodes"][0]
        self.left["episodes"].append(dict(episode, episode_id="SYN-DUPLICATE-MEMORY"))
        self.assert_invalid("duplicate episode source")
        self.left["episodes"].pop()
        reused = self.add_contrast()
        reused["episodes"][0]["episode_id"] = "SYN-DUPLICATE-MEMORY"
        self.assert_invalid("conflicting episode source")

    def test_structured_metadata_does_not_reach_raters(self):
        self.left["decisions"][0]["reasoning"] = "PRIVATE-CHAIN-OF-THOUGHT"
        self.left["observations"][21]["plan"] = "PRIVATE-PLAN"
        public, key = pilot.build(self.manifest, 17)
        visible = json.dumps(public) + pilot.rating_sheet(public)
        for secret in (
            "Sable Reed", "Juniper", "syn-being", "SYN-", "target", "PRIVATE-",
            "gold_label", "source_id", "checkpoint_id", "condition", "recorded_tick",
        ):
            self.assertNotIn(secret, visible)
        self.assertIn(self.left["provenance"]["checkpoint_id"], json.dumps(key))
        sheet = pilot.rating_sheet(public)
        self.assertIn("Synthetic practice material", sheet)
        self.assertIn("A / B / Tie / NEI / MISSING", sheet)
        self.assertIn("Evidence IDs", sheet)
        self.assertIn("| | | |", sheet)

    def test_free_text_leaks_fail_closed_across_evidence_fields(self):
        self.pair["comparison"] = "separate_runs"
        for rows, index, key, leaked in (
            ("decisions", 1, "message", self.right["condition"].lower()),
            ("observations", 22, "text", "prefix_" + self.right["run_id"] + "_suffix"),
            ("episodes", 0, "text", "ＳＹＮ－ＧＯＬＤ－ＡＮＳＷＥＲ"),
            ("outcomes", 1, "feedback", self.right["outcomes"][0]["source_id"]),
            ("outcomes", 0, "feedback", self.right["provenance"]["controls"]["actor_model"]),
        ):
            with self.subTest(field=key, leaked=leaked):
                row = self.left[rows][index]
                original = row[key]
                row[key] += " " + leaked
                try:
                    self.assert_invalid("blinding leak")
                finally:
                    row[key] = original

    def test_blinding_checks_source_before_replacing_names(self):
        self.left["observations"][-1]["source_id"] = r"agent_logs\syn-being17.jsonl#42"
        self.left["decisions"][0]["message"] = "See agent_logs/syn-being17.jsonl#42."
        self.assert_invalid("blinding leak")

    def test_artifact_paths_are_protected_across_visible_evidence(self):
        self.pair["comparison"] = "separate_runs"
        artifact = "C:/custodian-only/private-events.jsonl"
        outcome = self.left["outcomes"][0]
        outcome["source_id"] = artifact + "#23/movement"
        for row, field in (
            (self.left["observations"][-2], "text"),
            (self.left["decisions"][0], "action"),
            (self.left["decisions"][0], "message"),
            (outcome, "feedback"),
            (self.left["episodes"][0], "text"),
        ):
            original = row[field]
            for reference in (artifact, outcome["source_id"], r"c:\CUSTODIAN-ONLY\other\..\private-events.jsonl"):
                with self.subTest(field=field, reference=reference):
                    row[field] = f"{original} See {reference}."
                    self.assert_invalid("blinding leak")
            row[field] = original
        outcome["feedback"] += f" Separate artifact {artifact}.bak. The agent kept ordinary notes."
        self.left["event"]["source_id"] = "./notes#14/message"
        pilot.build(self.manifest, 17)
        outcome["feedback"] += " See ./notes."
        self.assert_invalid("blinding leak")

    def test_embedded_conditions_and_labeled_numbers_are_protected(self):
        decision = self.left["decisions"][0]
        for label, text in (
            (self.left["condition"], self.left["condition"] + "Baseline"),
            ("hou", "houBaseline"),
            ("hou", "seed=8675309"),
            ("hou", "seed=8.675309e6"),
            ("hou", 'generator.temperature: "0.2"'),
            ("hou", "generator.temperature=.20"),
        ):
            with self.subTest(text=text):
                self.left["condition"] = label
                decision["message"] = text
                self.assert_invalid("blinding leak")
        decision["message"] = "Juniper, we should save 8675309 energy and share 0.2 food."
        export, _ = pilot.build(self.manifest, 17)
        self.assertIn("we should save 8675309 energy and share 0.2 food.", pilot.public_json(export))

    def test_final_scan_includes_generated_rater_content(self):
        for term in ("P001", "Pseudonymous"):
            with self.subTest(term=term):
                self.manifest["private_terms"] = [term]
                self.assert_invalid("blinding leak")

    def test_original_identifier_embedded_in_message_fails_after_aliasing(self):
        self.left["decisions"][0]["message"] = "See syn-being17Profile."
        self.assert_invalid("blinding leak")

    def test_export_identity_rubric_and_d1_fields(self):
        export, key = pilot.build(self.manifest, 17)
        other, _ = pilot.build(self.manifest, 1)
        self.assertNotEqual(export["export_id"], other["export_id"])
        sheet = pilot.rating_sheet(export)
        for name in ("export_id", "rubric_revision"):
            self.assertEqual(export[name], key[name])
            self.assertEqual(export[name], export["packets"][0][name])
            self.assertIn(export[name], sheet)
        for side in "AB":
            for name in ("Evaluable past-event claim count", "Contradiction count", "Any contradiction"):
                self.assertIn(f"| {side} | D1 {name} |", sheet)

    def test_cli_writes_separate_files_and_refuses_existing_destination(self):
        tmp = Path(tempfile.gettempdir()).resolve() / f"behavior-pilot-test-{uuid.uuid4().hex}"
        # Inherit the temporary directory ACL on Windows, like the exporter.
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        out = tmp / "pilot"
        command = [sys.executable, str(Path(pilot.__file__)), str(FIXTURE),
                   "--seed", "17", "--out-dir", str(out)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Synthetic pilot: 1 packet(s)", run.stdout)
        self.assertEqual({p.name for p in (out / "rater").iterdir()}, {"packets.json", "rating_sheet.md"})
        key = json.loads((out / "private_key.json").read_text(encoding="utf-8"))
        self.assertEqual(key["seed"], 17)
        self.assertEqual(len(key["manifest_sha256"]), 64)
        duplicate = tmp / "same-export"
        repeat = subprocess.run([*command[:-1], str(duplicate)], capture_output=True, text=True)
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        for relative in ("rater/packets.json", "rater/rating_sheet.md", "private_key.json"):
            self.assertEqual((out / relative).read_bytes(), (duplicate / relative).read_bytes())
        retry = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(retry.returncode, 2)
        self.assertEqual(key, json.loads((out / "private_key.json").read_text(encoding="utf-8")))


class BehaviorPilotAssignmentTests(unittest.TestCase):
    def test_common_block_and_balance_for_each_supported_rater_count(self):
        manifest = assignment_manifest()
        for rater_count in (3, 4, 5):
            with self.subTest(raters=rater_count):
                exports, key = pilot.build_assignments(manifest, 17, rater_count)
                canonical = {packet["packet_id"]: packet for packet in key["packets"]}
                self.assertEqual(len(exports), rater_count)
                self.assertEqual(exports.keys(), key["assignments"].keys())
                orientations = Counter()
                for rater_id, export in exports.items():
                    order = [packet["packet_id"] for packet in export["packets"]]
                    mapping = key["assignments"][rater_id]
                    self.assertEqual(Counter(order), Counter(canonical.keys()))
                    self.assertEqual(order, [entry["packet_id"] for entry in mapping])
                    self.assertEqual(export["rater_id"], rater_id)
                    self.assertEqual(export["export_id"], key["export_id"])
                    condition_counts = Counter()
                    for entry in mapping:
                        packet_id = entry["packet_id"]
                        orientation = entry["display_to_canonical"]
                        self.assertEqual(set(orientation.values()), {"A", "B"})
                        orientations[packet_id, orientation["A"]] += 1
                        for display, side in orientation.items():
                            condition = canonical[packet_id]["branches"][side]["condition"]
                            condition_counts[condition] += 1 if display == "A" else -1
                    self.assertTrue(all(abs(count) <= 1 for count in condition_counts.values()), condition_counts)
                for packet_id in canonical:
                    counts = sorted(orientations[packet_id, side] for side in "AB")
                    self.assertEqual(counts, [rater_count // 2, (rater_count + 1) // 2])

    def test_seeded_assignments_and_presentation_order(self):
        manifest = assignment_manifest()
        baseline = pilot.build_assignments(manifest, 17, 5)
        self.assertEqual(baseline, pilot.build_assignments(manifest, 17, 5))
        exports, key = baseline
        alternative, other_key = pilot.build_assignments(manifest, 18, 5)
        self.assertNotEqual(key["export_id"], other_key["export_id"])
        self.assertNotEqual(key["export_id"], pilot.build_assignments(manifest, 17, 4)[1]["export_id"])
        self.assertNotEqual(key["export_id"], pilot.build(manifest, 17)[1]["export_id"])
        def order(export):
            return tuple(packet["packet_id"] for packet in export["packets"])

        self.assertGreater(len({order(export) for export in exports.values()}), 1)
        self.assertNotEqual(order(exports["R01"]), order(alternative["R01"]))
        for count in (2, 6):
            with self.assertRaisesRegex(ValueError, "3, 4 or 5"):
                pilot.build_assignments(manifest, 17, count)

    def test_orientation_preserves_evidence_and_remaps_citations(self):
        manifest = assignment_manifest()
        original, _ = pilot.build(manifest, 17)
        canonical = {packet["packet_id"]: packet["branches"] for packet in original["packets"]}
        exports, key = pilot.build_assignments(manifest, 17, 3)
        for rater_id, export in exports.items():
            for packet, mapping in zip(export["packets"], key["assignments"][rater_id]):
                for display, side in mapping["display_to_canonical"].items():
                    actual = copy.deepcopy(packet["branches"][display])
                    cited = actual["dossier"] + [record for turn in actual["turns"]
                                                 for record in turn.values() if isinstance(record, dict)]
                    for row in cited:
                        self.assertTrue(row["cite"].startswith(f"{display}-"))
                        row["cite"] = side + row["cite"][1:]
                    self.assertEqual(actual, canonical[packet["packet_id"]][side])

    def test_assignment_metadata_is_scanned_for_leaks(self):
        manifest = assignment_manifest()
        manifest["private_terms"].append("R01")
        pilot.build(manifest, 17)
        with self.assertRaisesRegex(ValueError, "blinding leak"):
            pilot.build_assignments(manifest, 17, 3)

    def test_cli_keeps_the_canonical_key_outside_rater_directories(self):
        tmp = Path(tempfile.gettempdir()).resolve() / f"behavior-assignment-test-{uuid.uuid4().hex}"
        # Inherit the parent ACL so the subprocess can create its output on Windows.
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        out = tmp / "pilot"
        run = subprocess.run(
            [sys.executable, "-B", str(Path(pilot.__file__)), str(FIXTURE),
             "--seed", "17", "--raters", "3", "--out-dir", str(out)],
            capture_output=True, text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual({path.name for path in out.iterdir()}, {"rater", "private_key.json"})
        key = json.loads((out / "private_key.json").read_text(encoding="utf-8"))
        self.assertEqual(set(key["assignments"]), {"R01", "R02", "R03"})
        self.assertIn("condition", key["packets"][0]["branches"]["A"])
        self.assertEqual({path.name for path in (out / "rater").iterdir()}, set(key["assignments"]))
        for directory in (out / "rater").iterdir():
            self.assertEqual({path.name for path in directory.iterdir()}, {"packets.json", "rating_sheet.md"})
            packets = (directory / "packets.json").read_text(encoding="utf-8")
            sheet = (directory / "rating_sheet.md").read_text(encoding="utf-8")
            self.assertIn(f"Pseudonymous rater ID {directory.name}", sheet)
            self.assertIn("workflow dry run provides no human-evaluation evidence", sheet)
            self.assertEqual(json.loads(packets)["rater_id"], directory.name)
            for secret in ("SYN-", "Sable Reed", "display_to_canonical", "assignments", "pair_id",
                           "condition", "run_id", "source_id", "provenance", "gold_label", "seed"):
                self.assertNotIn(secret, packets + sheet)


class BehaviorPilotGroupedSplitTests(unittest.TestCase):
    def test_related_contrasts_and_ticks_stay_together_with_reproducible_allocation(self):
        manifest = grouped_manifest()
        result = pilot.build(manifest, 17)
        self.assertEqual(result, pilot.build(manifest, 17))
        export, key = result
        split = key["split"]
        self.assertEqual(split["counts"]["development"]["groups"], 3)
        self.assertEqual(split["counts"]["held_out"]["groups"], 2)
        self.assertEqual(sum(counts["cards"] for counts in split["counts"].values()), 7)
        shared = [entry for entry in key["packets"] if entry["source_group_id"] == "SYN-SOURCE-GROUP-0"]
        self.assertEqual(len(shared), 3)
        self.assertEqual(len({entry["split"] for entry in shared}), 1)
        self.assertEqual({branch["tick"] for entry in shared for branch in entry["branches"].values()}, {22, 25})
        self.assertEqual(len(export["packets"]), split["counts"]["development"]["cards"])
        for entry in key["packets"]:
            self.assertEqual(entry["split"], split["group_membership"][entry["source_group_id"]])

        manifest["pairs"].reverse()
        self.assertEqual(pilot.build(manifest, 17)[1]["split"], split)
        self.assertNotEqual(pilot.build(manifest, 18)[1]["split"]["group_membership"], split["group_membership"])

    def test_rejects_conflicting_groups_for_any_shared_source(self):
        for kind in ("run", "source history", "checkpoint", "source artifact"):
            with self.subTest(kind=kind):
                manifest = grouped_manifest()
                origin, _, reused, independent, *_ = manifest["pairs"]
                first = origin["branches"][0]
                branch = independent["branches"][0]
                if kind == "run":
                    reused["source_group_id"] = independent["source_group_id"]
                elif kind == "source history":
                    branch["provenance"]["source_history_id"] = first["provenance"]["source_history_id"]
                elif kind == "checkpoint":
                    for item in independent["branches"]:
                        item["provenance"]["checkpoint_id"] = first["provenance"]["checkpoint_id"]
                else:
                    artifact = first["observations"][0]["source_id"].split("#", 1)[0].replace("/", "\\.\\")
                    branch["observations"][0]["source_id"] = artifact + "#200/observation"
                with self.assertRaisesRegex(ValueError, f"conflicting source_group_id declarations for shared {kind}"):
                    pilot.build(manifest, 17)

    def test_missing_or_invalid_group_provenance_and_legacy_behavior(self):
        for field_name in ("source_group_id", "source_history_id"):
            for value in (None, " ", "123"):
                with self.subTest(field=field_name, value=value):
                    manifest = grouped_manifest()
                    pair = manifest["pairs"][0]
                    record = pair if field_name == "source_group_id" else pair["branches"][1]["provenance"]
                    if value is None:
                        del record[field_name]
                    else:
                        record[field_name] = value
                    with self.assertRaisesRegex(ValueError, field_name):
                        pilot.build_assignments(manifest, 17, 3)
        manifest = assignment_manifest()
        manifest["synthetic"] = False
        self.assertEqual(len(pilot.build(manifest, 17)[0]["packets"]), 7)
        with self.assertRaisesRegex(ValueError, "genuine pilot assignments require schema_version 3"):
            pilot.build_assignments(manifest, 17, 3)
        manifest["pairs"][0]["source_group_id"] = "SYN-SOURCE-GROUP-0"
        with self.assertRaisesRegex(ValueError, "source_group_id requires manifest schema_version 3"):
            pilot.build(manifest, 17)

    def test_small_group_counts_report_limitations(self):
        for group_count, card_count in ((1, 3), (2, 4), (3, 5)):
            with self.subTest(groups=group_count):
                manifest = grouped_manifest()
                manifest["pairs"] = manifest["pairs"][:card_count]
                exports, key = pilot.build_assignments(manifest, 17, 3)
                development, held_out = (key["split"]["counts"][name] for name in ("development", "held_out"))
                self.assertTrue(key["split"]["limitation"])
                self.assertEqual(held_out["groups"], int(group_count > 1))
                self.assertEqual(development["groups"] + held_out["groups"], group_count)
                self.assertEqual(development["cards"] + held_out["cards"], card_count)
                self.assertEqual(len(exports["R01"]["packets"]), development["cards"])
                if group_count == 1:
                    self.assertEqual(held_out["cards"], 0)

    def test_ten_groups_split_seven_three_without_breaking_large_groups(self):
        pair_groups = {f"pair-{index}": f"group-{index}" for index in range(10)}
        pair_groups.update({f"extra-{index}": "group-0" for index in range(30)})
        split = pilot.split_groups(pair_groups, 17)
        self.assertIsNone(split["limitation"])
        self.assertEqual(split["counts"]["development"]["groups"], 7)
        self.assertEqual(split["counts"]["held_out"]["groups"], 3)
        large_group_split = split["group_membership"]["group-0"]
        self.assertEqual(split["counts"][large_group_split]["cards"],
                         split["counts"][large_group_split]["groups"] + 30)

    def test_group_ids_are_protected_from_rater_visible_text(self):
        for identifier in ("SYN-SOURCE-GROUP-0", "SYN-SOURCE-HISTORY-0"):
            with self.subTest(identifier=identifier):
                manifest = grouped_manifest()
                manifest["pairs"][0]["branches"][1]["decisions"][0]["message"] = identifier
                with self.assertRaisesRegex(ValueError, "blinding leak"):
                    pilot.build_assignments(manifest, 17, 3)

    def test_grouped_cli_keeps_all_held_out_cards_and_labels_out_of_rater_files(self):
        manifest = grouped_manifest()
        _, baseline = pilot.build(manifest, 17)
        held_out_pairs = {entry["pair_id"] for entry in baseline["packets"] if entry["split"] == "held_out"}
        for pair in manifest["pairs"]:
            if pair["pair_id"] in held_out_pairs:
                pair["branches"][1]["decisions"][0]["message"] = "Withheld transcript marker."
        tmp = Path(tempfile.gettempdir()).resolve() / f"behavior-split-test-{uuid.uuid4().hex}"
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        source = tmp / "manifest.json"
        source.write_text(json.dumps(manifest), encoding="utf-8")
        for rater_count in (None, 3, 4, 5):
            with self.subTest(raters=rater_count):
                out = tmp / f"pilot-{rater_count}"
                command = [sys.executable, "-B", str(Path(pilot.__file__)), str(source), "--seed", "17",
                           "--out-dir", str(out)]
                if rater_count is not None:
                    command.extend(["--raters", str(rater_count)])
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("Development: 3 group(s)", run.stdout)
                self.assertIn("Held-out: 2 group(s)", run.stdout)
                self.assertEqual({path.name for path in out.iterdir()}, {"rater", "private_key.json"})
                key = json.loads((out / "private_key.json").read_text(encoding="utf-8"))
                development = {entry["packet_id"] for entry in key["packets"] if entry["split"] == "development"}
                held_out = {entry["packet_id"] for entry in key["packets"] if entry["split"] == "held_out"}
                self.assertTrue(held_out)
                for mapping in key.get("assignments", {}).values():
                    self.assertEqual(Counter(entry["packet_id"] for entry in mapping), Counter(iter(development)))
                packet_files = list((out / "rater").rglob("packets.json"))
                self.assertEqual(len(packet_files), rater_count or 1)
                for packet_file in packet_files:
                    packets = packet_file.read_text(encoding="utf-8")
                    sheet = packet_file.with_name("rating_sheet.md").read_text(encoding="utf-8")
                    self.assertEqual(Counter(packet["packet_id"] for packet in json.loads(packets)["packets"]),
                                     Counter(iter(development)))
                    for secret in (*held_out, "Withheld transcript marker", "held_out", "source_group_id",
                                   "source_history_id", "SYN-SOURCE-GROUP-", "SYN-SOURCE-HISTORY-"):
                        self.assertNotIn(secret, packets + sheet)


class BehaviorPilotSelfPairTests(unittest.TestCase):
    def test_control_duplicates_all_evidence_with_fresh_aliases_and_side_citations(self):
        manifest = grouped_manifest()
        # Exercise separate-run aliasing for an entity other than the focal agent and cue.
        for pair in manifest["pairs"]:
            pair["comparison"] = "separate_runs"
            for branch in pair["branches"]:
                branch["names"]["syn-bystander"] = ["Elm Frost"]
                branch["decisions"][0]["message"] += " Elm Frost is nearby."
        export, key = pilot.build(manifest, 17, self_pair_count=2)
        visible = {packet["packet_id"]: packet for packet in export["packets"]}
        ordinary = {entry["pair_id"]: entry for entry in key["packets"] if "control" not in entry}
        controls = [entry for entry in key["packets"] if "control" in entry]
        self.assertEqual(len(controls), 2)
        for entry in controls:
            with self.subTest(packet=entry["packet_id"]):
                control = entry["control"]
                parent = ordinary[control["parent_pair_id"]]
                self.assertEqual(control["type"], "self_pair")
                self.assertEqual(entry["source_group_id"], parent["source_group_id"])
                self.assertEqual(entry["split"], parent["split"])
                self.assertEqual(entry["split"], "development")
                source = next(pair for pair in manifest["pairs"] if pair["pair_id"] == parent["pair_id"])
                selected_run = source["branches"][control["selected_branch_index"]]["run_id"]
                parent_side, parent_branch = next((side, branch) for side, branch in parent["branches"].items()
                                                  if branch["run_id"] == selected_run)
                self.assertEqual(entry["branches"]["A"]["aliases"], entry["branches"]["B"]["aliases"])
                self.assertNotEqual(entry["branches"]["A"]["aliases"], parent_branch["aliases"])
                for side in "AB":
                    branch = entry["branches"][side]
                    self.assertEqual(branch["run_id"], selected_run)
                    self.assertEqual({cite[1:]: row for cite, row in branch["citations"].items()},
                                     {cite[1:]: row for cite, row in parent_branch["citations"].items()})
                    self.assertTrue(all(cite.startswith(f"{side}-") for cite in branch["citations"]))
                    expected = json.dumps(visible[parent["packet_id"]]["branches"][parent_side], sort_keys=True)
                    expected = re.sub(r'"cite": "[AB]-', f'"cite": "{side}-', expected)
                    parent_aliases = {alias: branch["aliases"][entity] for entity, alias in parent_branch["aliases"].items()}
                    expected = re.sub(r"Entity \d+-\d+", lambda match: parent_aliases[match[0]], expected)
                    self.assertEqual(json.dumps(visible[entry["packet_id"]]["branches"][side], sort_keys=True), expected)
                self.assertEqual(visible[entry["packet_id"]]["comparison"], visible[parent["packet_id"]]["comparison"])

    def test_seeded_controls_use_distinct_development_parents(self):
        manifest = grouped_manifest()
        manifest["pairs"] = manifest["pairs"][:3]  # One development history fixes eligibility across seeds.
        baseline = pilot.build(manifest, 17, self_pair_count=2)
        self.assertEqual(baseline, pilot.build(manifest, 17, self_pair_count=2))
        self.assertNotEqual(baseline[0]["export_id"], pilot.build(manifest, 17)[0]["export_id"])
        selections = set()
        for seed in (17, 18, 19, 20):
            _, key = pilot.build(manifest, seed, self_pair_count=2)
            ordinary = {entry["pair_id"]: entry for entry in key["packets"] if "control" not in entry}
            controls = [entry for entry in key["packets"] if "control" in entry]
            parents = {entry["control"]["parent_pair_id"] for entry in controls}
            self.assertEqual(len(parents), 2)
            self.assertTrue(all(ordinary[parent]["split"] == "development" for parent in parents))
            selections.add(frozenset(parents))
        self.assertGreater(len(selections), 1)

    def test_zero_controls_preserve_both_export_modes(self):
        manifest = grouped_manifest()
        self.assertEqual(pilot.build(manifest, 17), pilot.build(manifest, 17, self_pair_count=0))
        self.assertEqual(pilot.build_assignments(manifest, 17, 3),
                         pilot.build_assignments(manifest, 17, 3, self_pair_count=0))

    def test_controls_preserve_ordinary_balance_and_occur_once_for_each_rater(self):
        manifest = grouped_manifest()
        for rater_count in (3, 4, 5):
            with self.subTest(raters=rater_count):
                baseline, baseline_key = pilot.build_assignments(manifest, 17, rater_count)
                exports, key = pilot.build_assignments(manifest, 17, rater_count, self_pair_count=2)
                self.assertEqual((exports, key), pilot.build_assignments(manifest, 17, rater_count, self_pair_count=2))
                canonical = {entry["packet_id"]: entry for entry in key["packets"]}
                baseline_canonical = {entry["packet_id"]: entry for entry in baseline_key["packets"]}
                eligible = {packet_id for packet_id, entry in canonical.items() if entry["split"] == "development"}
                self.assertEqual(len(canonical), len(key["packets"]))
                self.assertEqual(len(eligible), 5)
                orders = set()
                for rater, export in exports.items():
                    order = tuple(packet["packet_id"] for packet in export["packets"])
                    orders.add(order)
                    self.assertEqual(Counter(order), Counter(iter(eligible)))
                    self.assertNotEqual(baseline[rater]["export_id"], export["export_id"])
                    self.assertEqual([item["packet_id"] for item in key["assignments"][rater]], list(order))
                    control_count = 0
                    orientations = {}
                    for row in key["assignments"][rater]:
                        entry = canonical[row["packet_id"]]
                        if "control" in entry:
                            control_count += 1
                            self.assertEqual(row["display_to_canonical"], {"A": "A", "B": "B"})
                        else:
                            orientations[entry["pair_id"]] = row["display_to_canonical"]
                    self.assertEqual(control_count, 2)
                    self.assertEqual(orientations, {baseline_canonical[row["packet_id"]]["pair_id"]:
                                                    row["display_to_canonical"]
                                                    for row in baseline_key["assignments"][rater]})
                self.assertGreater(len(orders), 1)

    def test_invalid_counts_and_ordinary_evidence_validation(self):
        for count in (-1, 3, True, "1"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "self_pair_count must be 0, 1 or 2"):
                pilot.build(grouped_manifest(), 17, self_pair_count=count)
        with self.assertRaisesRegex(ValueError, "require schema_version 3"):
            pilot.build_assignments(assignment_manifest(), 17, 3, self_pair_count=1)
        manifest = grouped_manifest()
        manifest["pairs"] = manifest["pairs"][:4]  # Seed 17 leaves one development comparison.
        with self.assertRaisesRegex(ValueError, "only 1 development comparison"):
            pilot.build_assignments(manifest, 17, 3, self_pair_count=2)
        self.assertEqual(len(pilot.build(manifest, 17, self_pair_count=1)[0]["packets"]), 2)
        manifest = grouped_manifest()
        manifest["pairs"][2]["branches"][0]["decisions"][0]["action"] = "Stay"
        with self.assertRaisesRegex(ValueError, "conflicting decision"):
            pilot.build(manifest, 17, self_pair_count=1)
        manifest = grouped_manifest()
        manifest["pairs"][0]["comparison"] = "self_pair"
        with self.assertRaisesRegex(ValueError, "declare matched_checkpoint or separate_runs"):
            pilot.build(manifest, 17, self_pair_count=1)

    def test_cli_exports_controls_in_ordinary_format_with_private_lineage(self):
        tmp = Path(tempfile.gettempdir()).resolve() / f"behavior-controls-test-{uuid.uuid4().hex}"
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        manifest = tmp / "manifest.json"
        manifest.write_text(json.dumps(grouped_manifest()), encoding="utf-8")
        out = tmp / "pilot"
        run = subprocess.run([sys.executable, "-B", str(Path(pilot.__file__)), str(manifest),
                              "--seed", "17", "--raters", "3", "--self-pairs", "2", "--out-dir", str(out)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("3 ordinary comparison(s), 2 self-pair control(s)", run.stdout)
        self.assertEqual({path.name for path in out.iterdir()}, {"rater", "private_key.json"})
        key = json.loads((out / "private_key.json").read_text(encoding="utf-8"))
        controls = {entry["packet_id"] for entry in key["packets"] if "control" in entry}
        self.assertEqual(len(controls), 2)
        for directory in (out / "rater").iterdir():
            packets = (directory / "packets.json").read_text(encoding="utf-8")
            sheet = (directory / "rating_sheet.md").read_text(encoding="utf-8")
            for secret in ("control", "self_pair", "self-pair", "parent_pair_id", "selected_branch_index",
                           "source_group_id", "SYN-", "held_out"):
                self.assertNotIn(secret, packets + sheet)
            records = json.loads(packets)["packets"]
            self.assertEqual(len(records), 5)
            self.assertEqual({frozenset(packet) for packet in records},
                             {frozenset(("export_id", "rubric_revision", "packet_id", "comparison", "branches"))})
            self.assertTrue(controls <= {packet["packet_id"] for packet in records})


if __name__ == "__main__":
    unittest.main()
