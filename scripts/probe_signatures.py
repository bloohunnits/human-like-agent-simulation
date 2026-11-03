"""Extract a decision table from an existing TerraLingua run (standard library only).

Usage from the repository root::

    python scripts/probe_signatures.py path/to/logs/experiment --out signatures.csv
    python scripts/probe_signatures.py path/to/logs/experiment > signatures.csv
    python -B -m unittest discover -s scripts -p test_probe_signatures.py -v

Inputs are agent_logs/<tag>.jsonl, optional <tag>_memory_audit.jsonl,
messages.json, and agent_names.json. No simulation, model, or judge is imported
or run. Unlike rescued_cards.py, this exports all recorded decision ticks, not
memory cards, and does not recompute retrieval, score probes, or infer outcomes.

Schema/semantics:
* (run_dir, agent_id, timestep) identifies a row. agent_id is the file's stable
  tag, validated against explicit tags. Numeric top-level timestamp/time/tick/
  timestep values must agree and be nonnegative integers. observation.time is
  remaining lifespan and is NEVER a join key. Rows are the union of valid
  decision and audit keys, sorted by timestep then agent_id; gaps are not filled.
* chosen_action, target, parameters, outgoing_message describe a DECISION,
  never confirmed execution/delivery. energy is the pre-decision observation.
  No outcomes are inferred from messages, energy changes, or selected memories;
  use world events/execution receipts for a separate outcome analysis.
* nearby_agents contains observed names (color suffixes removed), not invented
  stable IDs. Both upstream grid and fork grid/graph observations are supported.
  incoming_messages preserves observed channels under their original keys:
  message, incoming_broadcasts, incoming_dms, external_response. Global broadcasts
  are not evidence that this agent received them.
* world_outgoing_messages contains same-tick messages.json text attributable to
  a unique known display name; it stays separate from the chosen outgoing text
  (the engine may truncate it). Ambiguous names are not guessed. Absence of a
  messages tick is unknown, not proof of silence; no T-to-T+1 shift is applied.
* selected_memory_ids and solo_selected_memory_ids preserve audit order. A
  missing/malformed list is null, distinct from a recorded empty list [].
* Structured CSV cells (see JSON_FIELDS) contain JSON; null means unavailable,
  []/{} mean observed empty. Missing scalar cells are blank. CSV uses UTF-8 and
  standard quoting, retaining Unicode, commas, quotes, and message newlines.
* source_references contains run-relative paths plus 1-based JSONL line numbers
  or JSON pointers. issues and stderr diagnostics expose malformed/ambiguous
  inputs. Invalid JSON/identity/timestep records are skipped, not guessed.

Replay policy: keep the last valid record per source/key and preserve every
source reference. Prefer an audit's bound agent_step as a whole over standalone
logs; never mix fields from different attempts. Flag conflicting duplicates.
For legacy audits without agent_step, conflicting decision/audit duplicates make
retrieval-to-decision association ambiguous: memory columns stay null. A bad
bound agent_step also prevents fallback to an unrelated standalone decision.
Missing optional files are harmless; an invalid run path is an error. The CLI
refuses to overwrite an existing output file. Extraction itself is read-only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
from typing import Any, TextIO


FIELDS = (
    "run_id", "run_dir", "agent_id", "agent_name", "timestep", "chosen_action",
    "target", "parameters", "nearby_agents", "incoming_messages",
    "outgoing_message", "world_outgoing_messages", "energy",
    "selected_memory_ids", "solo_selected_memory_ids", "retrieval_event_id",
    "source_references", "issues",
)
JSON_FIELDS = {
    "parameters", "nearby_agents", "incoming_messages", "world_outgoing_messages",
    "selected_memory_ids", "solo_selected_memory_ids", "source_references", "issues",
}
TIME_KEYS = ("timestamp", "time", "tick", "timestep")
CHANNELS = ("message", "incoming_broadcasts", "incoming_dms", "external_response")


def _tick(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("invalid simulation timestep")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid simulation timestep") from None
    if not number.is_finite() or not 0 <= number <= 2**63 - 1 or number != number.to_integral_value():
        raise ValueError("invalid simulation timestep")
    return int(number)


def _time(record: dict) -> int:
    values = [_tick(record[key]) for key in TIME_KEYS if key in record]
    if not values or len(set(values)) != 1:
        raise ValueError("missing or conflicting simulation timesteps")
    return values[0]


def _identity(record: dict, tag: str) -> None:
    if any(record[key] != tag for key in ("agent_tag", "agent_id") if key in record):
        raise ValueError("agent ID conflicts with filename")


def _json_file(path: Path, diagnostics: list[str]) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream, parse_constant=lambda _: None)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        diagnostics.append(f"{path.name}: cannot read JSON ({type(exc).__name__})")
        return None


def _json_lines(path: Path, diagnostics: list[str]):
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda _: None)
                    if not isinstance(record, dict):
                        raise ValueError("expected an object")
                except (UnicodeError, ValueError, RecursionError) as exc:
                    diagnostics.append(f"{path.name}:{line_number}: skipped malformed record ({type(exc).__name__})")
                    continue
                yield line_number, record
    except OSError as exc:
        diagnostics.append(f"{path.name}: cannot read ({type(exc).__name__})")


@dataclass
class _Entry:
    record: dict
    references: list[dict] = field(default_factory=list)
    conflicting: bool = False
    names: set[str] = field(default_factory=set)


def _index(path: Path, tag: str, run: Path, diagnostics: list[str]) -> dict[int, _Entry]:
    result: dict[int, _Entry] = {}
    for line, record in _json_lines(path, diagnostics):
        try:
            _identity(record, tag)
            tick = _time(record)
            event_id = record.get("event_id")
            if path.name.endswith("_memory_audit.jsonl") and isinstance(event_id, str) and ":retrieval:" in event_id:
                event_tag, event_tick = event_id.rsplit(":retrieval:", 1)
                if event_tag != tag or _tick(event_tick) != tick:
                    raise ValueError("retrieval event ID conflicts with agent/timestep")
        except ValueError as exc:
            diagnostics.append(f"{path.name}:{line}: skipped: {exc}")
            continue
        reference = {"path": path.relative_to(run).as_posix(), "line": line}
        names = set()
        for candidate in (record, record.get("agent_step")):
            if not isinstance(candidate, dict):
                continue
            try:
                _identity(candidate, tag)
            except ValueError:
                continue
            name = candidate.get("agent")
            if isinstance(name, str) and name:
                names.add(name)
        previous = result.get(tick)
        if previous is None:
            result[tick] = _Entry(record, [reference], names=names)
        else:
            previous.conflicting |= previous.record != record
            previous.record = record
            previous.references.append(reference)
            previous.names.update(names)
            diagnostics.append(f"{path.name}:{line}: duplicate agent/timestep {tag}/{tick}; using last record")
    return result


def _nearby(observation: dict, issues: list[str]) -> list[str] | None:
    grid = observation.get("observation")
    if not isinstance(grid, dict):
        if "observation" in observation:
            issues.append("malformed_neighborhood")
        return None
    people = set()
    for cell in grid.values():
        items = cell.get("items") if isinstance(cell, dict) else cell
        if not isinstance(items, list):
            issues.append("malformed_neighborhood")
            return None  # do not turn an incomplete neighborhood into an empty one
        for item in items:
            if not isinstance(item, str):
                issues.append("malformed_neighborhood")
                return None
            if not item or item == "X" or item.startswith("A("):
                continue
            try:
                Decimal(item)  # food values, including nonfinite legacy values
                continue
            except InvalidOperation:
                pass
            people.add(item.rsplit("(", 1)[0] if item.endswith(")") else item)
    return sorted(people)


def _memory_ids(audit: dict, key: str, issues: list[str]) -> list[str] | None:
    if key not in audit:
        return None
    value = audit[key]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        issues.append(f"malformed_{key}")
        return None
    return value


def _behavior(log: _Entry | None, audit: _Entry | None, tag: str, tick: int,
              issues: list[str]) -> tuple[dict, dict | None]:
    if audit and "agent_step" in audit.record:
        step = audit.record["agent_step"]
        try:
            if not isinstance(step, dict):
                raise ValueError("expected agent_step object")
            _identity(step, tag)
            if any(key in step for key in TIME_KEYS) and _time(step) != tick:
                raise ValueError("agent_step timestep mismatch")
        except ValueError:
            issues.append("invalid_bound_agent_step")
            return {}, None
        if log and any(step.get(key) != log.record.get(key)
                       for key in ("agent", "agent_tag", "action", "observation")):
            issues.append("bound_step_differs_from_agent_log")
        return step, {**audit.references[-1], "pointer": "/agent_step"}
    if log:
        return log.record, log.references[-1]
    return {}, None


def _world_messages(messages: dict | None, tick: int, tag: str, name: str | None,
                    owners: dict[str, set[str]], issues: list[str]) -> tuple[list[str] | None, dict | None]:
    if messages is None or str(tick) not in messages:
        return None, None
    ref = {"path": "messages.json", "pointer": f"/{tick}"}
    broadcasts = messages[str(tick)]
    if not isinstance(broadcasts, list) or any(not isinstance(text, str) for text in broadcasts):
        issues.append("malformed_world_messages")
        return None, ref
    if not name or owners.get(name) != {tag}:
        issues.append("ambiguous_world_message_sender")
        return None, ref
    result = []
    for text in broadcasts:
        # Longest known name handles names themselves containing ': '.
        matches = [candidate for candidate in owners if text.startswith(candidate + ": ")]
        if matches and max(matches, key=len) == name:
            result.append(text[len(name) + 2:])
    return result, ref


def extract_rows(run_dir: str | Path, diagnostics: list[str] | None = None) -> list[dict[str, Any]]:
    """Read one run into typed rows; append nonfatal problems to diagnostics.

    Does not import TerraLingua or write any files. Missing optional fields are
    None; JSON encoding is deferred to write_csv. Invalid run paths raise
    ValueError. Memory-snapshot files are intentionally not used as tick logs.
    """
    diagnostics = diagnostics if diagnostics is not None else []
    run = Path(run_dir).resolve()
    logs = run / "agent_logs"
    if not logs.is_dir():
        raise ValueError(f"run has no agent_logs directory: {run}")
    agents: dict[tuple[str, int], _Entry] = {}
    audits: dict[tuple[str, int], _Entry] = {}
    for path in sorted(logs.glob("*.jsonl")):
        is_audit = path.name.endswith("_memory_audit.jsonl")
        tag = path.name[:-len("_memory_audit.jsonl")] if is_audit else path.stem
        if path.name == "token_counts.jsonl":
            continue
        destination = audits if is_audit else agents
        destination.update({(tag, tick): entry for tick, entry in _index(path, tag, run, diagnostics).items()})

    names = _json_file(run / "agent_names.json", diagnostics) or {}
    messages = _json_file(run / "messages.json", diagnostics)
    owners: dict[str, set[str]] = {}
    for tag, name in names.items():
        if isinstance(name, str) and name:
            owners.setdefault(name, set()).add(tag)
    # Include names from all attempts, conservatively avoiding reused-name joins.
    for (tag, _), entry in list(agents.items()) + list(audits.items()):
        for name in entry.names:
            owners.setdefault(name, set()).add(tag)

    rows = []
    for tag, tick in sorted(agents.keys() | audits.keys(), key=lambda key: (key[1], key[0])):
        log, audit = agents.get((tag, tick)), audits.get((tag, tick))
        issues: list[str] = []
        refs = {}
        for kind, entry in (("agent_log", log), ("memory_audit", audit)):
            if entry:
                refs[kind] = entry.references
                if len(entry.references) > 1:
                    issues.append(f"duplicate_{kind}")
                if entry.conflicting:
                    issues.append(f"conflicting_{kind}")
        step, behavior_ref = _behavior(log, audit, tag, tick, issues)
        if behavior_ref:
            refs["behavior"] = behavior_ref
        name = step.get("agent", names.get(tag))
        name = name if isinstance(name, str) and name else None
        if "agent" not in step and name:
            refs["agent_name"] = {"path": "agent_names.json", "pointer": "/" + tag.replace("~", "~0").replace("/", "~1")}
        action = step.get("action")
        if not isinstance(action, dict):
            if "action" in step:
                issues.append("malformed_action")
            action = {}
        chosen = action.get("action")
        if chosen is not None and not isinstance(chosen, str):
            issues.append("malformed_chosen_action")
            chosen = None
        params = action.get("params")
        if params is not None and not isinstance(params, dict):
            issues.append("malformed_parameters")
            params = None
        target = next((params[key] for key in ("target", "target_agent", "target_tag")
                       if isinstance(params.get(key), str)), None) if params is not None else None
        outgoing = action.get("message")
        if outgoing is not None and not isinstance(outgoing, str):
            issues.append("malformed_outgoing_message")
            outgoing = None
        observation = step.get("observation")
        if not isinstance(observation, dict):
            if "observation" in step:
                issues.append("malformed_observation")
            observation = {}
        incoming = {key: observation[key] for key in CHANNELS if key in observation}
        for key, value in list(incoming.items()):
            if not isinstance(value, (dict, list, str)):
                incoming[key] = None
                issues.append(f"malformed_{key}")
        energy = observation.get("energy")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not Decimal(str(energy)).is_finite():
            if energy is not None:
                issues.append("malformed_energy")
            energy = None
        audit_data = audit.record if audit else {}
        selected = _memory_ids(audit_data, "selected", issues)
        solo = _memory_ids(audit_data, "solo_selected", issues)
        if log and audit and (log.conflicting or audit.conflicting) and "agent_step" not in audit_data:
            issues.append("ambiguous_legacy_join")
            selected = solo = None
        world, world_ref = _world_messages(messages, tick, tag, name, owners, issues)
        if world_ref:
            refs["world_messages"] = world_ref
        event_id = audit_data.get("event_id")
        nearby = _nearby(observation, issues)
        rows.append({
            "run_id": run.name, "run_dir": str(run), "agent_id": tag,
            "agent_name": name, "timestep": tick, "chosen_action": chosen,
            "target": target, "parameters": params, "nearby_agents": nearby,
            "incoming_messages": incoming or None, "outgoing_message": outgoing,
            "world_outgoing_messages": world, "energy": energy,
            "selected_memory_ids": selected, "solo_selected_memory_ids": solo,
            "retrieval_event_id": event_id if isinstance(event_id, str) else None,
            "source_references": refs, "issues": sorted(set(issues)),
        })
    if not rows:
        diagnostics.append("no valid decision or audit records found")
    return rows


def write_csv(rows: list[dict[str, Any]], stream: TextIO) -> None:
    """Write a header and rows; complex cells use strict JSON, not Python repr."""
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row[key], ensure_ascii=False, allow_nan=False, sort_keys=True)
                         if key in JSON_FIELDS else row[key] for key in FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="existing run folder containing agent_logs/")
    parser.add_argument("--out", "-o", default="-", help="new CSV path; default '-' writes stdout")
    args = parser.parse_args(argv)
    diagnostics: list[str] = []
    try:
        rows = extract_rows(args.run_dir, diagnostics)
        if args.out == "-":
            write_csv(rows, sys.stdout)
        else:
            with Path(args.out).open("x", encoding="utf-8", newline="") as stream:
                write_csv(rows, stream)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for diagnostic in diagnostics:
        print(f"warning: {diagnostic}", file=sys.stderr)
    print(f"extracted {len(rows)} agent-ticks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
