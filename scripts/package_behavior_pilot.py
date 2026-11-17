"""Create and verify one offline ZIP per assigned behavioral-pilot rater."""

import argparse
import hashlib
import stat
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from scripts import build_behavior_pilot as pilot
from scripts import collect_behavior_pilot as collection


MEMBER_NAMES = ("packets.json", "rating_sheet.md", "response_template.json", "instructions.md")
PRIVATE_MARKERS = {
    "private_key.json", "display_to_canonical", "assignment_id_key", "source_group_id",
    "source_history_id", "parent_pair_id", "selected_branch_index", "held_out", "self_pair",
}


def require_local(path, *, directory=False):
    """Check the named entry without following symlinks or Windows reparse points."""
    info = path.lstat()
    reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    pilot.require(not reparse and expected_type(info.st_mode), f"expected a local, non-linked entry at {path}")


def instructions(export):
    lines = ["# Behavioral pilot instructions", "",
             f"Export ID {export['export_id']}", f"Assignment ID {export['assignment_id']}",
             f"Rubric revision {export['rubric_revision']}", f"Pseudonymous rater ID {export['rater_id']}", ""]
    if export["synthetic"]:
        lines.extend(["Synthetic practice material for a workflow dry run.", ""])
    lines.extend([
        "1. Read `packets.json` and use `rating_sheet.md` as the guide to each question.",
        "2. Save a copy of `response_template.json` and fill in its values, reasons, and evidence IDs.",
        "   Keep the IDs and all assigned packet rows. Rate the displayed sides independently.",
        "   Enter numeric counts and D4 scores as JSON integers. Enter named choices as JSON strings.",
        "   NEI and MISSING are available for every question.",
        "   Use displayed evidence IDs such as `A-D0`. Supply a brief reason for each evaluable rating.",
        "3. Return the completed JSON file to the custodian using the agreed channel.",
        "",
        "**Tie** means A and B are equally believable on D5.",
        "**NEI** means there is insufficient information to judge the item.",
        "**MISSING** means you left the item unanswered.",
    ])
    return "\n".join(lines) + "\n"


def package_files(export, key, check):
    template = collection.submission_template(export)
    collection.validate_submission(template, key)
    files = {
        "packets.json": pilot.public_json(export),
        "rating_sheet.md": pilot.rating_sheet(export),
        "response_template.json": pilot.public_json(template),
        "instructions.md": instructions(export),
    }
    other_raters = key["assignment_ids"].keys() - {export["rater_id"]}
    other_ids = {key["assignment_ids"][rater] for rater in other_raters}
    private_terms = PRIVATE_MARKERS | other_raters | other_ids | {key["assignment_id_key"], key["manifest_sha256"]}
    check_package = pilot.blinding_check(private_terms, set(), set())
    for contents in files.values():
        check(contents, rendered=True)
        check_package(contents)
    return {name: contents.encode("utf-8") for name, contents in files.items()}


def read_export(manifest_path, export_dir):
    require_local(manifest_path)
    require_local(export_dir, directory=True)
    key_path = export_dir / "private_key.json"
    require_local(key_path)
    key = collection.read_object(key_path.read_bytes())
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    pilot.require(digest == pilot.digest(key, "manifest_sha256"), "manifest does not match the export's private key")
    manifest = collection.read_object(raw)
    assignments = pilot.field(key, "assignments", dict)
    control_count = sum("control" in packet for packet in pilot.field(key, "packets", list))
    exports, expected_key, check = pilot._build_assignments(manifest, pilot.field(key, "seed", int),
                                                           len(assignments), self_pair_count=control_count)
    expected_key["manifest_sha256"] = digest
    # Round-trip the generated key because JSON stores numeric tick keys as strings.
    expected = collection.read_object(pilot.public_json(expected_key))
    pilot.require(pilot.canonical_json(key) == pilot.canonical_json(expected),
                  "private key does not match the replayed assignments")
    return exports, expected, check


def archive_bytes(files):
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name in MEMBER_NAMES:
            member = ZipInfo(name)
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, files[name])
    return buffer.getvalue()


def validate_archive(path, expected):
    """Read every member back and verify its metadata and exact approved bytes."""
    pilot.require(expected.keys() == set(MEMBER_NAMES), "invalid expected package file list")
    require_local(path)
    raw = path.read_bytes()
    with ZipFile(BytesIO(raw)) as archive:
        members = archive.infolist()
        names = [member.orig_filename for member in members]
        pilot.require(len(names) == len(MEMBER_NAMES) and set(names) == set(MEMBER_NAMES),
                      "ZIP members do not match the package allowlist")
        pilot.require(not archive.comment, "unexpected ZIP archive comment")
        for member in members:
            payload = expected[member.filename]
            pilot.require(stat.S_ISREG(member.external_attr >> 16) and not member.extra and not member.comment
                          and not member.flag_bits & 1 and member.compress_type == ZIP_STORED,
                          "unexpected ZIP member metadata")
            pilot.require(member.file_size == len(payload) and archive.read(member) == payload,
                          f"ZIP content does not match the approved assignment for {member.filename}")
    pilot.require(raw == archive_bytes(expected), "unexpected bytes or metadata outside ZIP member contents")
    return hashlib.sha256(raw).hexdigest()


def create_packages(manifest, export_dir, out_dir, report_path):
    share_dir = out_dir.resolve()
    report = report_path.resolve()
    rater_dir = export_dir / "rater"
    pilot.require(not out_dir.exists() and not out_dir.is_symlink(), "package output directory already exists")
    pilot.require(not report_path.exists() and not report_path.is_symlink(), "custodian report already exists")
    pilot.require(report != share_dir and not report.is_relative_to(share_dir),
                  "custodian report must be outside the package directory")
    pilot.require(not report.is_relative_to(rater_dir.resolve()) and not share_dir.is_relative_to(rater_dir.resolve()),
                  "place package output and the report outside the export's rater directories")
    pilot.require(report.parent.is_dir(), "custodian report parent directory must exist")
    exports, key, check = read_export(manifest, export_dir)
    require_local(rater_dir, directory=True)
    approved = {}
    for rater_id, export in exports.items():
        directory = rater_dir / rater_id
        require_local(directory, directory=True)
        files = package_files(export, key, check)
        for name in ("packets.json", "rating_sheet.md"):
            path = directory / name
            require_local(path)
            pilot.require(path.read_text(encoding="utf-8").encode("utf-8") == files[name],
                          f"{rater_id}/{name} does not match the replayed assignment")
        approved[rater_id] = files

    out_dir.mkdir()
    written, packages = [], []
    try:
        for rater_id, files in approved.items():
            path = out_dir / f"{rater_id}.zip"
            with path.open("xb") as output:
                written.append(path)
                output.write(archive_bytes(files))
            checksum = validate_archive(path, files)
            export = exports[rater_id]
            packages.append({"filename": path.name, "rater_id": rater_id, "assignment_id": export["assignment_id"],
                             "packet_count": len(export["packets"]), "sha256": checksum})
        inventory = {"export_id": key["export_id"], "rubric_revision": key["rubric_revision"],
                     "synthetic": key["synthetic"], "packages": packages}
        collection.write_new(report_path, pilot.public_json(inventory).encode("utf-8"))
    except (OSError, ValueError, BadZipFile):
        for path in written:
            path.unlink(missing_ok=True)
        out_dir.rmdir()
        raise
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="original private reviewed manifest")
    parser.add_argument("--export-dir", type=Path, required=True, help="existing export created with --raters")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for rater ZIP files")
    parser.add_argument("--report", type=Path, required=True, help="new custodian-only report file outside --out-dir")
    args = parser.parse_args(argv)
    try:
        inventory = create_packages(args.manifest, args.export_dir, args.out_dir, args.report)
    except (OSError, ValueError, BadZipFile) as error:
        parser.error(str(error))
    print(f"Verified {len(inventory['packages'])} rater ZIP(s) in {args.out_dir}")
    print(f"Custodian report {args.report}")


if __name__ == "__main__":
    main()
