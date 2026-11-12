"""Create and validate offline JSON submissions for assigned D1–D5 pilot cards."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from scripts import build_behavior_pilot as pilot


SUBMISSION_VERSION = 1
IDENTITY_FIELDS = ("export_id", "rubric_revision", "rater_id", "assignment_id")
ASSIGNMENT_ID_PATTERN = re.compile(r"AS-[0-9a-f]{32}")


def exact_fields(record, fields, label):
    pilot.require(type(record) is dict and record.keys() == set(fields), f"invalid {label} fields")


def submission_template(export):
    """Use the rater-visible export only, with every blank value explicitly MISSING."""
    identity = {name: pilot.string(export, name) for name in IDENTITY_FIELDS}
    pilot.require(ASSIGNMENT_ID_PATTERN.fullmatch(identity["assignment_id"]), "invalid assignment ID")
    synthetic = pilot.field(export, "synthetic", bool)
    packets = pilot.field(export, "packets", list)
    packet_ids = []
    for index, packet in enumerate(packets, 1):
        pilot.require(isinstance(packet, dict), f"packet entry {index} must be an object")
        packet_ids.append(pilot.string(packet, "packet_id"))
    pilot.require(bool(packet_ids) and len(set(packet_ids)) == len(packet_ids), "invalid assigned packet list")

    def blank_rating():
        return {"value": "MISSING", "reason": "", "evidence_ids": []}

    ratings = [{"packet_id": packet_id,
                "branches": {side: {measure: blank_rating() for measure in pilot.BRANCH_MEASURES} for side in "AB"},
                "d5": blank_rating()} for packet_id in packet_ids]
    return {"submission_version": SUBMISSION_VERSION, **identity, "synthetic": synthetic, "ratings": ratings}


def assigned_citations(key, rater_id, submitted_id):
    assignments = pilot.field(key, "assignments", dict)
    ids = pilot.field(key, "assignment_ids", dict)
    pilot.digest(key, "assignment_id_key")
    pilot.require(rater_id in assignments and rater_id in ids, "unknown rater assignment")
    pilot.require(submitted_id == ids[rater_id], "assignment ID does not match this rater's assignment")

    version = pilot.field(key, "schema_version", int)
    grouped = version == pilot.GROUPED_SCHEMA_VERSION
    pilot.require(grouped or (version == pilot.SCHEMA_VERSION and key["synthetic"]),
                  "validation requires grouped development cards or a legacy synthetic assignment")
    groups = pilot.field(pilot.field(key, "split", dict), "group_membership", dict) if grouped else {}
    packets = pilot.field(key, "packets", list)
    canonical = {pilot.string(packet, "packet_id"): packet for packet in packets}
    pilot.require(len(canonical) == len(packets), "duplicate packet in private key")
    mapping = pilot.field(assignments, rater_id, list)
    pilot.require(bool(mapping), "empty assignment in private key")
    assigned = {}
    for row in mapping:
        packet_id = pilot.string(row, "packet_id")
        pilot.require(packet_id in canonical and packet_id not in assigned, "invalid packet in private assignment")
        packet = canonical[packet_id]
        if grouped:
            pilot.require(packet.get("split") == "development"
                          and groups.get(pilot.string(packet, "source_group_id")) == "development",
                          "private assignment contains a non-development packet")
        orientation = pilot.field(row, "display_to_canonical", dict)
        pilot.require(orientation in ({"A": "A", "B": "B"}, {"A": "B", "B": "A"}),
                      "invalid orientation in private assignment")
        branches = pilot.field(packet, "branches", dict)
        assigned[packet_id] = {
            display: {display + cite[1:] for cite in pilot.field(pilot.field(branches, side, dict), "citations", dict)}
            for display, side in orientation.items()
        }
    pilot.require(submitted_id == pilot.assignment_id(key, rater_id, mapping),
                  "assignment ID does not match the private packet and orientation mapping")
    return assigned


def validate_rating(record, choices, citations, label):
    exact_fields(record, ("value", "reason", "evidence_ids"), label)
    value = record["value"]
    state = value in pilot.RATING_STATES
    if choices is None:
        valid_value = type(value) is int and value >= 0
    else:
        valid_value = any(type(value) is type(choice) and value == choice for choice in choices)
    pilot.require(state or valid_value, f"invalid rating value for {label}")
    reason = pilot.field(record, "reason", str)
    pilot.require(state or pilot.nonblank(reason), f"reason is required for {label}")
    evidence = pilot.field(record, "evidence_ids", list)
    pilot.require(all(type(cite) is str and cite in citations for cite in evidence),
                  f"invalid evidence citation for {label}")


def validate_branch_consistency(ratings, label):
    claims = ratings["d1_evaluable_claims"]["value"]
    contradictions = ratings["d1_contradictions"]["value"]
    contradiction_flag = ratings["d1_any_contradiction"]["value"]
    if type(claims) is int and type(contradictions) is int:
        pilot.require(contradictions <= claims, f"D1 contradictions exceed evaluable claims for {label}")
    if type(contradictions) is int and contradiction_flag in pilot.YES_NO:
        pilot.require(contradiction_flag == ("Yes" if contradictions else "No"),
                      f"D1 contradiction flag disagrees with the count for {label}")
    pilot.require(not (claims == 0 and contradiction_flag == "Yes"),
                  f"D1 contradiction flag requires evaluable claims for {label}")

    d3_flags = (ratings["d3_intrusion"]["value"], ratings["d3_fabrication"]["value"])
    any_error = ratings["d3_any_error"]["value"]
    pilot.require(not (any_error == "No" and "Yes" in d3_flags)
                  and not (any_error == "Yes" and d3_flags == ("No", "No")),
                  f"D3 any-error flag disagrees with intrusion or fabrication for {label}")


def validate_submission(submission, key):
    """Check one complete assignment without modifying any supplied rating."""
    exact_fields(submission, ("submission_version", *IDENTITY_FIELDS, "synthetic", "ratings"), "submission")
    pilot.require(pilot.field(submission, "submission_version", int) == SUBMISSION_VERSION,
                  "incompatible submission version")
    pilot.require(pilot.string(submission, "export_id") == pilot.string(key, "export_id"), "wrong export ID")
    pilot.require(pilot.string(submission, "rubric_revision") == key.get("rubric_revision") == pilot.RUBRIC_REVISION,
                  "incompatible rubric revision")
    pilot.require(pilot.field(submission, "synthetic", bool) == pilot.field(key, "synthetic", bool),
                  "synthetic flag does not match export")
    rater_id = pilot.string(submission, "rater_id")
    submitted_id = pilot.string(submission, "assignment_id")
    pilot.require(ASSIGNMENT_ID_PATTERN.fullmatch(submitted_id), "invalid assignment ID")
    assigned = assigned_citations(key, rater_id, submitted_id)

    seen = set()
    for record in pilot.field(submission, "ratings", list):
        exact_fields(record, ("packet_id", "branches", "d5"), "packet rating")
        packet_id = pilot.string(record, "packet_id")
        pilot.require(packet_id in assigned, "unassigned packet ID")
        pilot.require(packet_id not in seen, "duplicate packet rating")
        seen.add(packet_id)
        exact_fields(record["branches"], "AB", "branch rating")
        for side, ratings in record["branches"].items():
            exact_fields(ratings, pilot.BRANCH_MEASURES, "D1–D4 rating")
            for name, (_, choices) in pilot.BRANCH_MEASURES.items():
                validate_rating(ratings[name], choices, assigned[packet_id][side], f"{packet_id} {side} {name}")
            validate_branch_consistency(ratings, f"{packet_id} {side}")
        citations = assigned[packet_id]["A"] | assigned[packet_id]["B"]
        validate_rating(record["d5"], pilot.D5_CHOICES, citations, f"{packet_id} D5")
    pilot.require(seen == assigned.keys(), "missing assigned packet rating; use explicit MISSING values")


def read_object(raw):
    record = json.loads(raw, object_pairs_hook=pilot.unique_object)
    pilot.require(type(record) is dict, "expected a JSON object")
    return record


def write_new(path, content):
    output = path.open("xb")
    try:
        with output:
            output.write(content)
    except OSError:
        path.unlink(missing_ok=True)
        raise


def collect(key_path, submissions, out_dir):
    key_raw = key_path.read_bytes()
    key = read_object(key_raw)
    key_digest = hashlib.sha256(key_raw).hexdigest()
    archive = out_dir.resolve()
    pending = {}
    for path in submissions:
        pilot.require(path.resolve().parent != archive, "validated output directory must be separate from submissions")
        raw = path.read_bytes()
        submission = read_object(raw)
        validate_submission(submission, key)
        target = archive / f"{submission['assignment_id']}.json"
        pilot.require(target not in pending and not target.exists(), "duplicate submission for assignment")
        validated = {"submission_sha256": hashlib.sha256(raw).hexdigest(),
                     "private_key_sha256": key_digest, "submission": submission}
        pending[target] = pilot.public_json(validated).encode("utf-8")
    archive.mkdir(mode=0o777 if os.name == "nt" else 0o700, exist_ok=True)
    for target, content in pending.items():
        write_new(target, content)
    return len(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    template = actions.add_parser("template", help="create a blank JSON form from a rater's packets.json")
    template.add_argument("packets", type=Path)
    template.add_argument("--output", type=Path, required=True)
    validate = actions.add_parser("validate", help="check submissions and archive them under custodian access")
    validate.add_argument("submissions", type=Path, nargs="+")
    validate.add_argument("--key", type=Path, required=True)
    validate.add_argument("--out-dir", type=Path, required=True, help="private validated archive for this export")
    args = parser.parse_args(argv)
    try:
        if args.action == "template":
            export = read_object(args.packets.read_bytes())
            write_new(args.output, pilot.public_json(submission_template(export)).encode("utf-8"))
            print(f"Created blank JSON submission at {args.output}")
        else:
            count = collect(args.key, args.submissions, args.out_dir)
            print(f"Validated {count} submission(s) in {args.out_dir}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
