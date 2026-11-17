"""Test the offline scorer using invented ratings and histories only."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts import build_behavior_pilot as pilot
from scripts import collect_behavior_pilot as collection
from scripts import score_behavior_pilot as scoring
from scripts.test_build_behavior_pilot import assignment_manifest, grouped_manifest


def mark(field, value):
    field.update(value=value, reason="Synthetic scoring fixture.")


class ScoringWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()).resolve() / f"synthetic-pilot-score-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.key_path = self.root / "private_key.json"
        self.archive = self.root / "validated"
        self.make_export(grouped_manifest(), self_pairs=2)

    def make_export(self, manifest, self_pairs=0):
        self.exports, self.key = pilot.build_assignments(manifest, 17, 3, self_pair_count=self_pairs)
        self.key_path.write_bytes(pilot.public_json(self.key).encode("utf-8"))
        self.submissions = {rater: collection.submission_template(export) for rater, export in self.exports.items()}
        self.cards, _ = scoring.read_design(self.key)
        self.rows = {rater: {row["packet_id"]: row for row in submission["ratings"]}
                     for rater, submission in self.submissions.items()}

    def collect(self, *raters):
        raw_files = []
        for rater in raters:
            raw = self.root / f"synthetic-{rater}.json"
            raw.write_bytes(pilot.public_json(self.submissions[rater]).encode("utf-8"))
            raw_files.append(raw)
        collection.collect(self.key_path, raw_files, self.archive)

    def display_for_condition(self, packet_id, rater, condition):
        row = next(row for row in self.key["assignments"][rater] if row["packet_id"] == packet_id)
        canonical_side = "AB"[self.cards[packet_id].conditions.index(condition)]
        return next(side for side, canonical in row["display_to_canonical"].items() if canonical == canonical_side)

    def set_d5(self, packet_id, votes):
        conditions = sorted(self.cards[packet_id].conditions)
        for rater, vote in zip(self.exports, votes):
            display = self.display_for_condition(packet_id, rater, conditions[vote]) if type(vote) is int else vote
            mark(self.rows[rater][packet_id]["d5"], display)

    def repeated_development_contrast(self):
        packets = [packet_id for packet_id, card in self.cards.items()
                   if card.split == "development" and not card.self_pair
                   and all(condition.endswith(("-3", "-4")) for condition in card.conditions)]
        self.assertEqual(len(packets), 2)
        return packets

    def repeated_contrast_result(self, report):
        return next(row for row in report["splits"]["development"]["contrasts"] if row["cards"] == 2)

    def test_d5_canonicalizes_both_orders_before_agreement_and_preference(self):
        first, second = self.repeated_development_contrast()
        self.set_d5(first, (0, 0, 0))
        self.set_d5(second, (1, 1, 1))
        self.assertEqual({rows[first]["d5"]["value"] for rows in self.rows.values()}, {"A", "B"})
        self.collect(*self.exports)
        report = scoring.score(self.key_path, self.archive)
        d5 = self.repeated_contrast_result(report)["d5"]
        self.assertEqual((d5["condition_1_wins"], d5["condition_1_losses"], d5["ties"]), (3, 3, 0))
        self.assertEqual(d5["condition_1_preference_share"], 0.5)
        self.assertEqual(d5["majority_cards"], {"condition_1": 1, "condition_2": 1})
        self.assertEqual(d5["human_agreement"]["alpha"]["value"], 1)
        self.assertEqual(d5["human_agreement"]["overlap_items"], 2)
        self.assertTrue(all(row["kappa"] == 1 for row in d5["human_agreement"]["pairwise"]))
        self.assertEqual(d5["human_cross_order"], {"cards": 2, "rater_pairs": 4, "matches": 4, "raw_agreement": 1})

    def test_d5_weights_cards_equally_and_counts_nonresponse_separately(self):
        first, second = self.repeated_development_contrast()
        self.set_d5(first, (0, 0, "Tie"))
        self.set_d5(second, (1, "NEI", "MISSING"))
        self.collect(*self.exports)
        d5 = self.repeated_contrast_result(scoring.score(self.key_path, self.archive))["d5"]
        self.assertAlmostEqual(d5["condition_1_preference_share"], 5 / 12)
        self.assertAlmostEqual(d5["condition_2_preference_share"], 7 / 12)
        self.assertEqual(d5["coverage"], {"assigned": 6, "submitted": 6, "unsubmitted": 0,
                                          "missing": 1, "nei": 1, "evaluable": 4})
        self.assertEqual(d5["tie_rate"], 0.25)
        self.assertEqual(d5["nei_rate_among_submitted"], 1 / 6)
        self.assertEqual(d5["human_agreement"]["overlap_items"], 1)
        self.assertEqual(d5["majority_cards"], {"condition_1": 1, "fewer_than_three_evaluable_votes": 1})

    def test_branch_summary_uses_canonical_sides_and_field_specific_denominators(self):
        first, second = self.repeated_development_contrast()
        condition = min(self.cards[first].conditions)
        for rater, claims, contradictions, d4, intrusion in (
            ("R01", 3, 2, 5, "Yes"), ("R02", 1, 0, 3, "NEI"), ("R03", "NEI", 1, "NEI", "No"),
        ):
            side = self.display_for_condition(first, rater, condition)
            branch = self.rows[rater][first]["branches"][side]
            mark(branch["d1_evaluable_claims"], claims)
            mark(branch["d1_contradictions"], contradictions)
            mark(branch["d1_any_contradiction"], "Yes" if contradictions else "No")
            mark(branch["d2_relevant_reference"], "Yes")
            mark(branch["d3_intrusion"], intrusion)
            mark(branch["d4_plausibility"], d4)
        side = self.display_for_condition(second, "R01", condition)
        mark(self.rows["R01"][second]["branches"][side]["d4_plausibility"], 1)
        self.collect(*self.exports)
        branches = self.repeated_contrast_result(scoring.score(self.key_path, self.archive))["branches"]
        summary = branches["by_condition"][condition]
        self.assertEqual(summary["d1_counts"], {"paired_evaluable_branch_ratings": 2, "claims": 4,
                                                "contradictions": 2, "contradictions_per_claim": 0.5})
        self.assertEqual(summary["flags"]["d1_any_contradiction"], {"yes": 2, "no": 1, "rate": 2 / 3})
        self.assertEqual(summary["flags"]["d3_intrusion"], {"yes": 1, "no": 1, "rate": 0.5})
        self.assertEqual(summary["coverage"]["d3_intrusion"]["nei"], 1)
        self.assertEqual(summary["d4"], {"distribution": {"1": 1, "2": 0, "3": 1, "4": 0, "5": 1},
                                         "median": 3, "mean": 3})
        self.assertIsNone(summary["flags"]["d2_implicit_use"]["rate"])
        self.assertEqual(branches["human_agreement"][scoring.D4]["alpha"]["distance"], "ordinal")
        agreement = branches["human_agreement"]["d1_evaluable_claims"]
        self.assertEqual(agreement["overlap_items"], 1)
        self.assertEqual(agreement["pairwise"][0]["mean_absolute_difference"], 2)

    def test_self_pairs_keep_displayed_sides_and_stay_out_of_condition_results(self):
        controls = [packet_id for packet_id, card in self.cards.items() if card.self_pair]
        self.set_d5(controls[0], ("A", "A", "Tie"))
        self.set_d5(controls[1], ("B", "NEI", "MISSING"))
        self.collect(*self.exports)
        development = scoring.score(self.key_path, self.archive)["splits"]["development"]
        control = development["self_pairs"]
        self.assertEqual((control["display_a_wins"], control["display_b_wins"], control["ties"]), (2, 1, 1))
        self.assertEqual(control["tie_rate"], 0.25)
        self.assertEqual(control["display_a_side_score"], 0.625)
        self.assertEqual(control["coverage"], {"assigned": 6, "submitted": 6, "unsubmitted": 0,
                                               "missing": 1, "nei": 1, "evaluable": 4})
        self.assertEqual(development["coverage_including_controls"]["d5"]["evaluable"], 4)
        self.assertTrue(all(row["d5"]["condition_1_preference_share"] is None for row in development["contrasts"]))

    def test_splits_are_isolated_and_absent_files_are_unsubmitted(self):
        self.collect("R01")
        report = scoring.score(self.key_path, self.archive)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["rater_submissions"], {"assigned": 3, "archived": 1})
        development, held_out = (report["splits"][name] for name in scoring.SPLITS)
        self.assertEqual(held_out["status"], "no_held_out_assignments")
        self.assertEqual((held_out["ordinary_cards"], held_out["source_groups"]), (4, 2))
        self.assertTrue(all(row["d5"]["coverage"]["assigned"] == 0 for row in held_out["contrasts"]))
        d5 = development["coverage_including_controls"]["d5"]
        self.assertEqual(d5, {"assigned": 15, "submitted": 5, "unsubmitted": 10, "missing": 5, "nei": 0, "evaluable": 0})
        self.assertEqual(report["judge"]["gate"], "not_evaluated")
        original = self.key_path.read_bytes()
        row = next(packet for packet in self.key["packets"] if packet["split"] == "held_out")
        self.key["assignments"]["R01"][0]["packet_id"] = row["packet_id"]
        self.key["assignment_ids"]["R01"] = pilot.assignment_id(self.key, "R01", self.key["assignments"]["R01"])
        self.key_path.write_bytes(pilot.public_json(self.key).encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "non-development packet"):
            scoring.score(self.key_path, self.archive)
        self.key_path.write_bytes(original)
        row["split"] = "development"
        with self.assertRaisesRegex(ValueError, "disagrees with source-group allocation"):
            scoring.read_design(self.key)

    def test_legacy_synthetic_archive_gets_descriptives_without_group_intervals(self):
        self.make_export(assignment_manifest())
        first = next(iter(self.rows["R01"]))
        self.set_d5(first, ("Tie", "Tie", "Tie"))
        self.collect(*self.exports)
        report = scoring.score(self.key_path, self.archive)
        development = report["splits"]["development"]
        self.assertIsNone(development["source_groups"])
        d5 = next(row["d5"] for row in development["contrasts"] if row["d5"]["ties"])
        self.assertEqual(d5["condition_1_preference_share"], 0.5)
        self.assertEqual(d5["condition_1_preference_interval_95"]["status"], "no_source_group_metadata")

    def test_report_keeps_matched_and_separate_run_cards_separate(self):
        manifest = grouped_manifest()
        manifest["pairs"][-1]["comparison"] = "separate_runs"
        self.make_export(manifest)
        self.collect("R01")
        report = scoring.score(self.key_path, self.archive)
        contrast = [row for row in report["splits"]["development"]["contrasts"]
                    if all(condition.endswith(("-3", "-4")) for condition in row["conditions"])]
        self.assertEqual([(row["comparison"], row["cards"]) for row in contrast],
                         [("matched_checkpoint", 1), ("separate_runs", 1)])

    def test_rejects_key_digest_and_revalidates_archive_payload_identity(self):
        self.collect("R01")
        key_bytes = self.key_path.read_bytes()
        self.key_path.write_bytes(key_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "different private key"):
            scoring.score(self.key_path, self.archive)
        self.key_path.write_bytes(key_bytes)
        archive = next(self.archive.glob("*.json"))
        raw = archive.read_bytes()
        for field, value, error in (
            ("export_id", "wrong", "wrong export ID"),
            ("rubric_revision", "other", "incompatible rubric revision"),
            ("synthetic", False, "synthetic flag"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                record = json.loads(raw)
                record["submission"][field] = value
                archive.write_bytes(pilot.public_json(record).encode("utf-8"))
                scoring.score(self.key_path, self.archive)
        archive.write_bytes(raw)
        record = json.loads(raw)
        record["submission"]["ratings"][0]["d5"]["value"] = "invalid"
        archive.write_bytes(pilot.public_json(record).encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "invalid rating value"):
            scoring.score(self.key_path, self.archive)
        archive.write_bytes(raw)
        duplicate = self.archive / f"{self.key['assignment_ids']['R02']}.json"
        duplicate.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "filename disagrees with assignment ID"):
            scoring.score(self.key_path, self.archive)

    def test_cli_writes_repeatable_local_report_and_preserves_inputs(self):
        first, second = self.repeated_development_contrast()
        self.set_d5(first, (0, 0, 1))
        self.set_d5(second, (1, "Tie", 1))
        self.collect(*self.exports)
        inputs = {path: hashlib.sha256(path.read_bytes()).digest()
                  for path in (self.key_path, *self.archive.glob("*.json"))}
        command = [sys.executable, "-B", "-m", "scripts.score_behavior_pilot", "--key", str(self.key_path),
                   "--validated-dir", str(self.archive), "--output"]
        output = self.root / "synthetic-report.json"
        repeated = self.root / "synthetic-repeat.json"
        for path in (output, repeated):
            run = subprocess.run([*command, str(path)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("synthetic", run.stdout)
        report_bytes = output.read_bytes()
        self.assertEqual(report_bytes, repeated.read_bytes())
        self.assertEqual(subprocess.run([*command, str(output)], capture_output=True).returncode, 2)
        self.assertEqual(subprocess.run([*command, str(self.archive / "score.json")], capture_output=True).returncode, 2)
        self.assertEqual(output.read_bytes(), report_bytes)
        for private_literal in (b"SYN-SOURCE-GROUP", b"SYN-ASSIGNMENT-PAIR", b"assignment_id_key", b"Synthetic scoring fixture"):
            self.assertNotIn(private_literal, report_bytes)
        self.assertEqual(inputs, {path: hashlib.sha256(path.read_bytes()).digest() for path in inputs})


class ScoringStatisticsTests(unittest.TestCase):
    def test_strict_majority_nei_no_majority_and_natural_ties(self):
        cases = (
            (("MISSING", "MISSING", "MISSING"), "no_nonmissing_votes"),
            (("NEI", 0, "NEI", "MISSING"), "majority_nei"),
            ((0, 0, "NEI"), "fewer_than_three_evaluable_votes"),
            ((0, 1, "Tie"), "no_majority"),
            ((0, 0, 1, 1), "no_majority"),
            ((0, "Tie", "Tie"), "tie"),
            ((1, 1, 0), "condition_2"),
        )
        for votes, expected in cases:
            with self.subTest(votes=votes):
                self.assertEqual(scoring.majority(dict(enumerate(votes))), expected)

    def test_nominal_alpha_and_kappa_use_only_paired_ratings(self):
        units = {"card1": {"R01": 0, "R02": 0}, "card2": {"R01": 1, "R02": 1},
                 "card3": {"R01": 1, "R02": 0}, "singleton": {"R01": "Tie", "R02": "NEI"}}
        groups = dict.fromkeys(units, "one_history")
        result = scoring.agreement(units, groups, ["R01", "R02"])
        self.assertEqual(result["evaluable_items"], 4)
        self.assertEqual(result["overlap_items"], 3)
        self.assertAlmostEqual(result["alpha"]["value"], 4 / 9)
        self.assertEqual(result["alpha"]["interval_95"]["status"], "insufficient_groups_or_raters")
        pair = result["pairwise"][0]
        self.assertEqual((pair["overlap_items"], pair["matches"]), (3, 2))
        self.assertAlmostEqual(pair["kappa"], 0.4)

    def test_alpha_weights_variable_rater_counts_and_ordinal_distances(self):
        weighted = [([(1, 1), (1, 1), (2, 1)], 1), ([(3, 1), (3, 1)], 1)]
        self.assertEqual(scoring.krippendorff_alpha(weighted), 0.5)
        self.assertAlmostEqual(scoring.krippendorff_alpha(weighted, ordinal=True), 0.8)
        self.assertIsNone(scoring.krippendorff_alpha([([(1, 3)], 1), ([(2, 2)], 1)]))

    def test_agreement_reports_no_overlap_and_undefined_constant_labels(self):
        disjoint = {"one": {"R01": 0, "R02": "NEI"}, "two": {"R01": "MISSING", "R02": 1}}
        groups = {"one": "first_history", "two": "second_history"}
        result = scoring.agreement(disjoint, groups, ["R01", "R02"])
        self.assertEqual(result["alpha"]["status"], "no_overlap")
        self.assertIsNone(result["alpha"]["value"])
        self.assertEqual(result["pairwise"][0]["status"], "no_overlap")
        self.assertIsNone(result["pairwise"][0]["kappa"])
        constant = {"one": {"R01": "Tie", "R02": "Tie"}, "two": {"R01": "Tie", "R02": "Tie"}}
        result = scoring.agreement(constant, groups, ["R01", "R02"])
        self.assertEqual(result["alpha"]["status"], "no_expected_disagreement")
        self.assertEqual(result["pairwise"][0]["status"], "no_expected_disagreement")
        self.assertEqual(result["pairwise"][0]["raw_agreement"], 1)

    def test_crossed_resampling_keeps_repeated_ticks_and_rater_records_together(self):
        units = {"tick1": {"R01": 1, "R02": 0}, "tick2": {"R01": 1, "R02": 0},
                 "tick3": {"R01": 0, "R02": 1}}
        groups = {"tick1": "first_history", "tick2": "first_history", "tick3": "second_history"}
        weights = list(scoring.weighted_units(units, groups, {"first_history": 3, "second_history": 0},
                                              {"R01": 2, "R02": 0}))
        self.assertEqual(weights, [([(1, 2)], 3), ([(1, 2)], 3)])

        def statistic(sample):
            return scoring.mean_score(sample, {0: 0, 1: 1}, per_card=True)

        estimate = scoring.interval(units, groups, statistic, 0.5)
        self.assertEqual(estimate, scoring.interval(units, groups, statistic, 0.5))
        self.assertEqual((estimate["source_groups"], estimate["raters"]), (2, 2))
        self.assertEqual((estimate["lower"], estimate["upper"]), (0, 1))
        one_rater = {card: {"R01": votes["R01"]} for card, votes in units.items()}
        self.assertEqual(scoring.interval(one_rater, groups, statistic, 0.5)["status"], "insufficient_groups_or_raters")
        alpha = scoring.agreement(units, groups, ["R01", "R02"])["alpha"]
        self.assertEqual(alpha["interval_95"]["status"], "insufficient_defined_draws")
        self.assertIsNone(alpha["interval_95"]["lower"])


if __name__ == "__main__":
    unittest.main()
