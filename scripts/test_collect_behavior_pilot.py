"""Synthetic workflow checks for offline pilot submissions."""

import copy
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
from scripts.test_build_behavior_pilot import assignment_manifest, grouped_manifest


def synthetic_submission(export):
    """Invent ratings solely to exercise ingestion, including every D5 state."""
    submission = collection.submission_template(export)
    for index, (packet, record, choice) in enumerate(
        zip(export["packets"], submission["ratings"], (*pilot.D5_CHOICES, *pilot.RATING_STATES))
    ):
        record["d5"].update(value=choice, reason="Synthetic ingestion example.\nSecond line.",
                            evidence_ids=["B-O0", "A-D0"])
        for side, ratings in record["branches"].items():
            for field, (_, choices) in pilot.BRANCH_MEASURES.items():
                value = 0 if choices is None else choices[0 if side == "A" else -1]
                ratings[field].update(value=value, reason=" Synthetic example only. ",
                                      evidence_ids=[packet["branches"][side]["turns"][0]["observation"]["cite"]])
        record["branches"]["A"]["d1_evaluable_claims"]["value"] = index
        record["branches"]["A"]["d1_contradictions"]["value"] = index // 2
        record["branches"]["A"]["d1_any_contradiction"]["value"] = "Yes" if index >= 2 else "No"
        record["branches"]["A"]["d4_plausibility"]["value"] = index + 1
        record["branches"]["B"]["d1_evaluable_claims"]["value"] = "NEI"
        record["branches"]["B"]["d1_contradictions"]["value"] = "MISSING"
        record["branches"]["A"]["d2_implicit_use"]["value"] = "NEI"
        record["branches"]["B"]["d3_intrusion"]["value"] = "MISSING"
        record["branches"]["B"]["d4_plausibility"]["value"] = "NEI"
    return submission


class BehaviorPilotCollectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = grouped_manifest()
        self.exports, self.key = pilot.build_assignments(self.manifest, 17, 3, self_pair_count=2)
        self.submission = synthetic_submission(self.exports["R01"])

    def test_assignment_ids_bind_identity_exact_packet_set_and_orientation(self):
        self.assertEqual((self.exports, self.key), pilot.build_assignments(self.manifest, 17, 3, self_pair_count=2))
        self.assertEqual(len(set(self.key["assignment_ids"].values())), 3)
        rater_id = "R01"
        mapping = self.key["assignments"][rater_id]
        original = self.key["assignment_ids"][rater_id]
        self.assertEqual(pilot.assignment_id(self.key, rater_id, mapping[::-1]), original)
        self.assertNotEqual(pilot.assignment_id(self.key, rater_id, mapping[:-1]), original)
        self.assertNotEqual(pilot.assignment_id(self.key, "R02", mapping), original)
        for name in ("export_id", "rubric_revision"):
            self.assertNotEqual(pilot.assignment_id({**self.key, name: "different"}, rater_id, mapping), original)
        changed = copy.deepcopy(mapping)
        row = changed[0]
        orientation = row["display_to_canonical"]
        row["display_to_canonical"] = {"A": orientation["B"], "B": orientation["A"]}
        self.assertNotEqual(pilot.assignment_id(self.key, rater_id, changed), original)
        self.key["assignments"][rater_id] = changed
        with self.assertRaisesRegex(ValueError, "private packet and orientation mapping"):
            collection.validate_submission(self.submission, self.key)

    def test_valid_submissions_for_three_four_and_five_raters(self):
        for count in (3, 4, 5):
            exports, key = pilot.build_assignments(self.manifest, 17, count, self_pair_count=2)
            self.assertEqual(len(set(key["assignment_ids"].values())), count)
            for rater_id, export in exports.items():
                with self.subTest(count=count, rater=rater_id):
                    submission = synthetic_submission(export)
                    self.assertEqual({row["d5"]["value"] for row in submission["ratings"]},
                                     {*pilot.D5_CHOICES, *pilot.RATING_STATES})
                    collection.validate_submission(submission, key)
                    self.assertEqual(submission, synthetic_submission(export))
                    self.assertIn(f"Assignment ID {export['assignment_id']}", pilot.rating_sheet(export))
                    self.assertEqual(export["assignment_id"], key["assignment_ids"][rater_id])
                    self.assertTrue(submission["synthetic"])

    def test_wrong_or_stale_identity_is_rejected(self):
        alternative, _ = pilot.build_assignments(self.manifest, 18, 3, self_pair_count=2)
        for field, value, error in (
            ("export_id", alternative["R01"]["export_id"], "wrong export ID"),
            ("rubric_revision", "D1-D5-pilot-v1", "incompatible rubric revision"),
            ("assignment_id", alternative["R01"]["assignment_id"], "assignment ID does not match"),
            ("assignment_id", self.exports["R02"]["assignment_id"], "assignment ID does not match"),
            ("rater_id", "R02", "assignment ID does not match"),
            ("rater_id", "R06", "unknown rater assignment"),
            ("synthetic", False, "synthetic flag"),
        ):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, error):
                collection.validate_submission({**self.submission, field: value}, self.key)

    def test_exact_development_packet_set_and_unique_packet_ratings(self):
        ratings = self.submission["ratings"]
        held_out = next(packet for packet in self.key["packets"] if packet["split"] == "held_out")
        for rows, error in (
            (ratings[:-1], "missing assigned packet"),
            ([*ratings, ratings[0]], "duplicate packet rating"),
            ([{**ratings[0], "packet_id": "P999"}, *ratings[1:]], "unassigned packet"),
            ([{**ratings[0], "packet_id": held_out["packet_id"]}, *ratings[1:]], "unassigned packet"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                collection.validate_submission({**self.submission, "ratings": rows}, self.key)
        self.key["assignments"]["R01"][0]["packet_id"] = held_out["packet_id"]
        with self.assertRaisesRegex(ValueError, "non-development packet"):
            collection.validate_submission(self.submission, self.key)

    def test_rating_domains_reason_citations_and_required_fields(self):
        for field, key, value, error in (
            ("d1_evaluable_claims", "value", -1, "invalid rating value"),
            ("d1_contradictions", "value", True, "invalid rating value"),
            ("d1_any_contradiction", "value", "Tie", "invalid rating value"),
            ("d2_relevant_reference", "value", None, "invalid rating value"),
            ("d3_intrusion", "value", "Unknown", "invalid rating value"),
            ("d4_plausibility", "value", 3.0, "invalid rating value"),
            ("d4_plausibility", "value", 6, "invalid rating value"),
            ("d5", "value", "tie", "invalid rating value"),
            ("d5", "reason", " ", "reason is required"),
            ("d5", "evidence_ids", ["A-O9"], "invalid evidence citation"),
            ("d2_implicit_use", "evidence_ids", ["B-O0"], "invalid evidence citation"),
            ("d3_any_error", "extra", "would otherwise be lost", "invalid .* fields"),
        ):
            with self.subTest(field=field, key=key, value=value), self.assertRaisesRegex(ValueError, error):
                submission = copy.deepcopy(self.submission)
                row = submission["ratings"][0]
                rating = row["d5"] if field == "d5" else row["branches"]["A"][field]
                rating[key] = value
                collection.validate_submission(submission, self.key)

    def test_d1_count_and_flag_consistency(self):
        for claims, contradictions, flag, error in (
            (0, 5, "No", "exceed evaluable claims"),
            (2, -1, "No", "invalid rating value"),
            (1, 1, "No", "flag disagrees"),
            (1, 0, "Yes", "flag disagrees"),
            (0, "NEI", "Yes", "requires evaluable claims"),
            (0, 0, "No", None),
            (1, 1, "Yes", None),
            ("NEI", 1, "Yes", None),
            (2, "MISSING", "Yes", None),
            (2, 1, "NEI", None),
            (0, "NEI", "MISSING", None),
        ):
            with self.subTest(claims=claims, contradictions=contradictions, flag=flag):
                submission = copy.deepcopy(self.submission)
                ratings = submission["ratings"][0]["branches"]["A"]
                for field, value in zip(("d1_evaluable_claims", "d1_contradictions", "d1_any_contradiction"),
                                        (claims, contradictions, flag)):
                    ratings[field]["value"] = value
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        collection.validate_submission(submission, self.key)
                else:
                    before = copy.deepcopy(submission)
                    collection.validate_submission(submission, self.key)
                    self.assertEqual(submission, before)

    def test_d3_combined_flag_consistency(self):
        for intrusion, fabrication, any_error, valid in (
            ("Yes", "No", "No", False),
            ("NEI", "Yes", "No", False),
            ("No", "No", "Yes", False),
            ("No", "No", "No", True),
            ("Yes", "No", "Yes", True),
            ("NEI", "MISSING", "Yes", True),
            ("No", "MISSING", "No", True),
            ("Yes", "NEI", "MISSING", True),
            ("Yes", "MISSING", "NEI", True),
        ):
            with self.subTest(intrusion=intrusion, fabrication=fabrication, any_error=any_error):
                submission = copy.deepcopy(self.submission)
                ratings = submission["ratings"][0]["branches"]["A"]
                for field, value in zip(("d3_intrusion", "d3_fabrication", "d3_any_error"),
                                        (intrusion, fabrication, any_error)):
                    ratings[field]["value"] = value
                if valid:
                    before = copy.deepcopy(submission)
                    collection.validate_submission(submission, self.key)
                    self.assertEqual(submission, before)
                else:
                    with self.assertRaisesRegex(ValueError, "D3 any-error flag disagrees"):
                        collection.validate_submission(submission, self.key)

    def test_template_cli_rejects_malformed_packet_entries_without_writing(self):
        tmp = Path(tempfile.gettempdir()).resolve() / f"pilot-template-{uuid.uuid4().hex}"
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        packets = tmp / "packets.json"
        output = tmp / "template.json"
        command = [sys.executable, "-B", "-m", "scripts.collect_behavior_pilot", "template", str(packets),
                   "--output", str(output)]
        for packet, error in ((None, "packet entry 1 must be an object"),
                              (1, "packet entry 1 must be an object"),
                              ("P001", "packet entry 1 must be an object"),
                              ({}, "packet_id must be str"),
                              ({"packet_id": " "}, "packet_id must be nonblank")):
            with self.subTest(packet=packet):
                packets.write_text(pilot.public_json({**self.exports["R01"], "packets": [packet]}), encoding="utf-8")
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn(error, run.stderr)
                self.assertNotIn("Traceback", run.stderr)
                self.assertFalse(output.exists())

    def test_templates_keep_control_and_orientation_metadata_private(self):
        export = self.exports["R01"]
        blank = collection.submission_template(export)
        collection.validate_submission(blank, self.key)
        visible = pilot.public_json(blank) + pilot.public_json(export) + pilot.rating_sheet(export)
        for secret in ("display_to_canonical", "source_group_id", "source_history_id", "parent_pair_id",
                       "selected_branch_index", "self_pair", "SYN-", self.key["assignment_id_key"]):
            self.assertNotIn(secret, visible)
        controls = {packet["packet_id"] for packet in self.key["packets"] if "control" in packet}
        self.assertEqual(len(controls), 2)
        self.assertTrue(controls <= {row["packet_id"] for row in blank["ratings"]})
        legacy, legacy_key = pilot.build_assignments(assignment_manifest(), 17, 3)
        collection.validate_submission(collection.submission_template(legacy["R01"]), legacy_key)

    def test_citation_validation_uses_the_displayed_orientation(self):
        pair = next(pair for pair in self.manifest["pairs"]
                    if self.key["split"]["group_membership"][pair["source_group_id"]] == "development")
        pair["comparison"] = "separate_runs"
        episodes = pair["branches"][0]["episodes"]
        episodes.append({**episodes[0], "episode_id": "SYN-SECOND-EPISODE",
                         "source_id": episodes[0]["source_id"] + "-second"})
        exports, key = pilot.build_assignments(self.manifest, 17, 3)
        packet_id = next(packet["packet_id"] for packet in key["packets"] if packet["pair_id"] == pair["pair_id"])
        display_sides = set()
        for export in exports.values():
            packet = next(packet for packet in export["packets"] if packet["packet_id"] == packet_id)
            side = next(side for side in "AB" if len(packet["branches"][side]["dossier"]) == 2)
            display_sides.add(side)
            submission = collection.submission_template(export)
            row = next(row for row in submission["ratings"] if row["packet_id"] == packet_id)
            row["d5"]["evidence_ids"] = [f"{side}-H2"]
            collection.validate_submission(submission, key)
            row["d5"]["evidence_ids"] = ["B-H2" if side == "A" else "A-H2"]
            with self.assertRaisesRegex(ValueError, "invalid evidence citation"):
                collection.validate_submission(submission, key)
        self.assertEqual(display_sides, {"A", "B"})

    def test_collection_cli_preserves_originals_and_rejects_duplicates(self):
        tmp = Path(tempfile.gettempdir()).resolve() / f"behavior-collection-test-{uuid.uuid4().hex}"
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        originals = tmp / "originals"
        originals.mkdir()
        archive = tmp / "validated"
        key_path = tmp / "private_key.json"
        key_path.write_text(pilot.public_json(self.key), encoding="utf-8")
        packets = tmp / "packets.json"
        packets.write_text(pilot.public_json(self.exports["R01"]), encoding="utf-8")
        template = tmp / "blank.json"
        cli = [sys.executable, "-B", "-m", "scripts.collect_behavior_pilot"]
        template_cmd = [*cli, "template", str(packets), "--output", str(template)]
        run = subprocess.run(template_cmd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(template.read_bytes()), collection.submission_template(self.exports["R01"]))
        original_template = template.read_bytes()
        self.assertEqual(subprocess.run(template_cmd, capture_output=True).returncode, 2)
        self.assertEqual(template.read_bytes(), original_template)

        response = originals / "synthetic_R01.json"
        response.write_text(pilot.public_json(self.submission), encoding="utf-8")
        raw = response.read_bytes()
        command = [*cli, "validate", str(response), "--key", str(key_path), "--out-dir", str(archive)]
        duplicate_cmd = [*cli, "validate", str(response), str(response), "--key", str(key_path),
                         "--out-dir", str(archive)]
        duplicate = subprocess.run(duplicate_cmd, capture_output=True, text=True)
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("duplicate submission", duplicate.stderr)
        self.assertFalse(archive.exists())
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        validated = archive / f"{self.submission['assignment_id']}.json"
        validated_bytes = validated.read_bytes()
        archived = json.loads(validated_bytes)
        self.assertEqual(archived["submission"], self.submission)
        self.assertEqual(archived["submission_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(archived["private_key_sha256"], hashlib.sha256(key_path.read_bytes()).hexdigest())
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("duplicate submission", repeated.stderr)
        self.assertEqual(response.read_bytes(), raw)
        self.assertEqual(validated.read_bytes(), validated_bytes)

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON field value"):
            collection.read_object(b'{"rating":{"value":"NEI","value":"MISSING"}}')


if __name__ == "__main__":
    unittest.main()
