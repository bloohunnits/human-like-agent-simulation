"""Offline packaging tests using invented behavioral evidence."""

import hashlib
import json
import shutil
import stat
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

from scripts import build_behavior_pilot as pilot
from scripts import collect_behavior_pilot as collection
from scripts import package_behavior_pilot as packaging
from scripts.test_build_behavior_pilot import grouped_manifest
from scripts.test_collect_behavior_pilot import synthetic_submission


class BehaviorPilotPackagingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()).resolve() / f"pilot-zip-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(pilot.public_json(grouped_manifest()), encoding="utf-8")

    def export(self, count=3):
        directory = self.root / f"export-{count}"
        with redirect_stdout(StringIO()):
            pilot.main([str(self.manifest), "--seed", "17", "--raters", str(count), "--self-pairs", "2",
                        "--out-dir", str(directory)])
        return directory

    def package(self, export, name="share"):
        out = self.root / name
        report = self.root / f"{name}-custodian.json"
        packaging.create_packages(self.manifest, export, out, report)
        return out, report

    def test_three_four_and_five_rater_archives_are_blinded_and_collector_compatible(self):
        for count in (3, 4, 5):
            with self.subTest(raters=count):
                export_dir = self.export(count)
                key = json.loads((export_dir / "private_key.json").read_bytes())
                # Unrelated artifacts beside approved files must never be copied.
                (export_dir / "rater" / "R01" / "debug.json").write_text("SYN-DEBUG", encoding="utf-8")
                (export_dir / "rater" / "R01" / "completed.json").write_text("SYN-SUBMISSION", encoding="utf-8")
                share, report = self.package(export_dir, f"share-{count}")
                inventory = json.loads(report.read_bytes())
                self.assertEqual(inventory["export_id"], key["export_id"])
                self.assertEqual(inventory["rubric_revision"], key["rubric_revision"])
                self.assertTrue(inventory["synthetic"])
                self.assertEqual(len(inventory["packages"]), count)
                self.assertEqual({path.name for path in share.iterdir()}, {f"{rater}.zip" for rater in key["assignments"]})
                held = {packet["packet_id"] for packet in key["packets"] if packet["split"] == "held_out"}
                controls = {packet["packet_id"] for packet in key["packets"] if "control" in packet}
                self.assertEqual(len(controls), 2)
                for row in inventory["packages"]:
                    rater = row["rater_id"]
                    path = share / row["filename"]
                    self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertEqual(row["assignment_id"], key["assignment_ids"][rater])
                    with ZipFile(path) as archive:
                        self.assertEqual(archive.namelist(), ["packets.json", "rating_sheet.md", "response_template.json",
                                                             "instructions.md"])
                        files = {member: archive.read(member) for member in archive.namelist()}
                    self.assertEqual(packaging.validate_archive(path, files), row["sha256"])
                    public = json.loads(files["packets.json"])
                    self.assertEqual(public, json.loads((export_dir / "rater" / rater / "packets.json").read_bytes()))
                    template = json.loads(files["response_template.json"])
                    collection.validate_submission(template, key)
                    response = synthetic_submission(public)
                    self.assertEqual(template, collection.submission_template(public))
                    collection.validate_submission(response, key)
                    shown = [packet["packet_id"] for packet in public["packets"]]
                    self.assertEqual(shown, [entry["packet_id"] for entry in key["assignments"][rater]])
                    self.assertEqual(len(shown), row["packet_count"])
                    self.assertTrue(controls <= set(shown))
                    text = b"\n".join(files.values()).decode("utf-8")
                    other_ids = {identifier for other, identifier in key["assignment_ids"].items() if other != rater}
                    other_raters = key["assignments"].keys() - {rater}
                    for secret in (*held, *other_ids, *other_raters, *packaging.PRIVATE_MARKERS,
                                   key["assignment_id_key"], "SYN-", "debug.json", "completed.json"):
                        self.assertNotIn(secret, text)
                    for name in ("rating_sheet.md", "instructions.md"):
                        self.assertIn(row["assignment_id"], files[name].decode("utf-8"))
                    for label in ("Tie", "NEI", "MISSING"):
                        self.assertIn(label, files["instructions.md"].decode("utf-8"))

    def test_repeatable_zip_hashes_and_no_overwrites(self):
        export = self.export()
        out, report = self.package(export)
        second, second_report = self.package(export, "repeat")
        self.assertEqual(report.read_bytes(), second_report.read_bytes())
        for path in out.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
        unused = self.root / "unused.json"
        with self.assertRaisesRegex(ValueError, "output directory already exists"):
            packaging.create_packages(self.manifest, export, out, unused)
        self.assertFalse(unused.exists())
        original = self.root / "original_response.json"
        original.write_bytes(b"synthetic original submission")
        with self.assertRaisesRegex(ValueError, "custodian report already exists"):
            packaging.create_packages(self.manifest, export, self.root / "unused", original)
        self.assertEqual(original.read_bytes(), b"synthetic original submission")
        self.assertFalse((self.root / "unused").exists())
        nested = self.root / "nested"
        with self.assertRaisesRegex(ValueError, "report must be outside"):
            packaging.create_packages(self.manifest, export, nested, nested / "report.json")

    def test_archive_verification_rejects_missing_extra_traversal_and_altered_content(self):
        export = self.export()
        out, _ = self.package(export)
        with ZipFile(out / "R01.zip") as archive:
            expected = {member: archive.read(member) for member in archive.namelist()}
        for name, files, error in (
            ("missing", {key: data for key, data in expected.items() if key != "rating_sheet.md"}, "allowlist"),
            ("extra", {**expected, "private_key.json": b"private"}, "allowlist"),
            ("traversal", {**expected, "../instructions.md": b"private"}, "allowlist"),
            ("backslash", {**expected, r"..\instructions.md": b"private"}, "allowlist"),
            ("wrong-identity", {**expected, "instructions.md": expected["instructions.md"].replace(b"R01", b"R02")},
             "does not match the approved assignment"),
        ):
            with self.subTest(case=name):
                path = self.root / f"{name}.zip"
                with ZipFile(path, "x") as archive:
                    for filename, payload in files.items():
                        member = ZipInfo(filename)
                        member.create_system = 3
                        member.external_attr = 0o100644 << 16
                        archive.writestr(member, payload)
                with self.assertRaisesRegex(ValueError, error):
                    packaging.validate_archive(path, expected)
        path = self.root / "appended.zip"
        path.write_bytes((out / "R01.zip").read_bytes() + b"unexpected trailing content")
        with self.assertRaisesRegex(ValueError, "unexpected bytes"):
            packaging.validate_archive(path, expected)

    def test_rejects_missing_or_mismatched_export_files_before_writing(self):
        export = self.export()
        rater = export / "rater" / "R01"
        sheet = rater / "rating_sheet.md"
        original_sheet = sheet.read_bytes()
        sheet.write_bytes((export / "rater" / "R02" / "rating_sheet.md").read_bytes())
        with self.assertRaisesRegex(ValueError, "rating_sheet.md does not match"):
            self.package(export)
        sheet.write_bytes(original_sheet)
        path = rater / "packets.json"
        original_packets = path.read_bytes()
        packets = json.loads(original_packets)
        packets["packets"][0]["branches"]["A"]["turns"][0]["decision"]["logged_outgoing_message"] = "private_key.json"
        path.write_text(pilot.public_json(packets), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "packets.json does not match"):
            self.package(export)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.package(export)
        self.assertFalse((self.root / "share").exists())
        self.assertFalse((self.root / "share-custodian.json").exists())

    def test_checks_the_private_manifest_and_new_rater_content(self):
        manifest = grouped_manifest()
        manifest["private_terms"].append("response_template.json")
        self.manifest.write_text(pilot.public_json(manifest), encoding="utf-8")
        export = self.export()
        with self.assertRaisesRegex(ValueError, "blinding leak"):
            self.package(export)
        self.manifest.write_text(pilot.public_json(grouped_manifest()), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            self.package(export)
        self.assertFalse((self.root / "share").exists())

    def test_packager_rejects_source_artifact_path_in_replayed_feedback(self):
        manifest = grouped_manifest()
        _, key = pilot.build_assignments(manifest, 17, 3)
        groups = key["split"]["group_membership"]
        pair = next(pair for pair in manifest["pairs"] if groups[pair["source_group_id"]] == "development")
        artifact = "C:/custodian-only/private-events.jsonl"
        outcome = pair["branches"][0]["outcomes"][0]
        outcome["source_id"] = artifact + "#23/movement"
        self.manifest.write_text(pilot.public_json(manifest), encoding="utf-8")
        export = self.export()

        outcome["feedback"] += f" See {artifact}."
        self.manifest.write_text(pilot.public_json(manifest), encoding="utf-8")
        # Get past the raw manifest hash check to exercise the replay's blinding gate.
        key_path = export / "private_key.json"
        key = json.loads(key_path.read_bytes())
        key["manifest_sha256"] = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        key_path.write_text(pilot.public_json(key), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "blinding leak"):
            self.package(export)
        self.assertFalse((self.root / "share").exists())
        self.assertFalse((self.root / "share-custodian.json").exists())

    def test_every_rendered_zip_member_is_scanned_for_source_artifact_paths(self):
        manifest = grouped_manifest()
        artifact = "C:/custodian-only/private-events.jsonl"
        manifest["pairs"][3]["branches"][0]["outcomes"][0]["source_id"] = artifact + "#23/movement"
        self.manifest.write_text(pilot.public_json(manifest), encoding="utf-8")
        exports, key, check = packaging.read_export(self.manifest, self.export())
        export = exports["R01"]
        serialize = pilot.public_json
        clean_files = packaging.package_files(export, key, check)
        for member in packaging.MEMBER_NAMES:
            self.assertNotIn(artifact.encode("utf-8"), clean_files[member])

            def serialize_with_path(record):
                filename = "response_template.json" if "ratings" in record else "packets.json"
                return serialize(record) + (artifact if filename == member else "")

            if member == "rating_sheet.md":
                injection = patch.object(pilot, "rating_sheet", return_value=artifact)
            elif member == "instructions.md":
                injection = patch.object(packaging, "instructions", return_value=artifact)
            else:
                injection = patch.object(pilot, "public_json", side_effect=serialize_with_path)
            with self.subTest(member=member), injection, self.assertRaisesRegex(ValueError, "blinding leak"):
                packaging.package_files(export, key, check)

    def test_rejects_linked_source_files_and_windows_reparse_points(self):
        export = self.export()
        source = export / "rater" / "R01" / "packets.json"
        real_lstat = Path.lstat
        for info in (SimpleNamespace(st_mode=stat.S_IFLNK),
                     SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)):
            def lstat(path):
                return info if path == source else real_lstat(path)

            with patch.object(Path, "lstat", lstat), self.assertRaisesRegex(ValueError, "non-linked entry"):
                self.package(export)
        self.assertFalse((self.root / "share").exists())


if __name__ == "__main__":
    unittest.main()
