"""Replay recorded runs through the memory math, with no TerraLingua, no
embedder and no model. See docs/MEMORY_REPLAY_BRIEF.md.

Each audit line holds the cue similarity of every memory at that tick, which
does not depend on the condition. Feeding those numbers back in replays the
same history under any condition: decay, graph routes and selection change,
similarity does not. Replaying the recorded condition must reproduce the
recorded selections exactly, which is checked first.

Usage: replay.py <run_dir> [--conditions hou learned chained] [--out DIR]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from itertools import repeat
from pathlib import Path

from associative_memory import MEMORY_CONDITIONS, AssociativeMemory, MemoryConfig
from hou_memory import HouMemoryStore

PLURALS = {"person": "people", "place": "places", "object": "artifacts"}


def entities_of(sources):
    """Entity keys from an audit line back to the entities dict preview takes."""
    out = {}
    for key in sources:
        kind, name = json.loads(key.removeprefix("entity:"))
        out.setdefault(PLURALS[kind], []).append(name)
    return out


def load_agent(agent_logs: Path, tag: str):
    lines = [json.loads(l) for l in (agent_logs / f"{tag}_memory_audit.jsonl").read_text().splitlines() if l.strip()]
    memories = json.loads((agent_logs / f"{tag}_hou_memory.json").read_text())["memories"]
    lines.sort(key=lambda line: line["tick"])
    # A turn missing from the audit (a run stopped mid-error) still changed
    # the store, so the replay ends at the first gap.
    for i in range(1, len(lines)):
        if lines[i]["tick"] != lines[i - 1]["tick"] + 1:
            return lines[:i], memories
    return lines, memories


def replay_agent(lines, memories, config: MemoryConfig, token_counter=None):
    """One row per recorded tick: what this condition would have selected."""
    store = HouMemoryStore(embed_fn=lambda text: None)
    engine = AssociativeMemory(store, config, token_counter)
    # The stale-preview guard hashes the whole store twice a tick. Replay
    # always commits the preview it just made, so the guard is skipped.
    engine._state_digest = lambda: "replay"
    pending = iter(memories)
    upcoming = next(pending, None)
    created = {}
    previous = set()
    out = []
    for line in lines:
        # The live agent scored the store before writing that tick's episode.
        while upcoming is not None and upcoming["t_created"] < line["tick"]:
            mem = engine.add(upcoming["text"], upcoming["t_created"], upcoming.get("meta"))
            created[mem.memory_id] = mem.t_created
            upcoming = next(pending, None)
        r_by_id = {row["id"]: row["relevance"] for row in line["rows"]}
        result = engine.preview(line["cue"], line["tick"], entities_of(line["sources"]),
                                line["event_id"], r_by_id=r_by_id)
        engine.commit(result)
        rescued = [i for i in result.selected if i not in result.solo_selected]
        # Fresh: not already in the prompt, either selected last tick or among
        # the recent turns the prompt shows. The same rule as rescued_cards.py.
        fresh = [i for i in rescued if i not in previous
                 and line["tick"] - created[i] > config.working_history]
        previous = set(result.selected)
        out.append({"tick": line["tick"], "selected": list(result.selected),
                    "solo_selected": list(result.solo_selected), "rescued": rescued, "fresh": fresh})
    return out


def condition_config(recorded: dict, name: str) -> MemoryConfig:
    names = {f.name for f in fields(MemoryConfig)}
    return MemoryConfig(**{**{k: v for k, v in recorded.items() if k in names}, **MEMORY_CONDITIONS[name]})


def replay_tag(agent_logs: Path, tag: str, conditions):
    """Check the recorded condition, then replay one agent under each condition."""
    lines, memories = load_agent(agent_logs, tag)
    if not lines:
        return 0, 0, {}
    recorded = lines[-1]["config"]
    rows = replay_agent(lines, memories, MemoryConfig(**recorded))
    matched = sum(r["selected"] == l["selected"] and r["solo_selected"] == l["solo_selected"]
                  for r, l in zip(rows, lines))
    return len(lines), matched, {name: replay_agent(lines, memories, condition_config(recorded, name))
                                 for name in conditions}


def replay_run(run_dir: Path, conditions):
    agent_logs = run_dir / "agent_logs"
    tags = sorted(p.name.removesuffix("_memory_audit.jsonl") for p in agent_logs.glob("*_memory_audit.jsonl"))
    check = {"ticks": 0, "matched": 0}
    results = {name: {} for name in conditions}
    # One process per agent: chained routes take about a minute per 300-tick agent.
    with ProcessPoolExecutor() as pool:
        for tag, (ticks, matched, by_condition) in zip(tags, pool.map(replay_tag, repeat(agent_logs), tags, repeat(conditions))):
            check["ticks"] += ticks
            check["matched"] += matched
            for name, rows in by_condition.items():
                results[name][tag] = rows
    return check, results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, help="a TerraLingua logs/<exp>/ folder")
    ap.add_argument("--conditions", nargs="+", default=["hou", "learned", "chained"], choices=MEMORY_CONDITIONS)
    ap.add_argument("--out", type=Path, help="write <out>/<run>/<condition>.json")
    args = ap.parse_args()
    check, results = replay_run(args.run_dir, args.conditions)
    print(f"{args.run_dir.name}: recorded condition reproduced on {check['matched']} of {check['ticks']} ticks")
    if check["matched"] != check["ticks"]:
        raise SystemExit("Replay does not match the recording, so the other conditions are not trustworthy")
    for name, by_agent in results.items():
        ticks = [row for rows in by_agent.values() for row in rows]
        rescued = sum(len(row["rescued"]) for row in ticks)
        fresh = sum(len(row["fresh"]) for row in ticks)
        print(f"  {name:12} {rescued} rescues, {fresh} fresh, in {len(ticks)} ticks")
        if args.out:
            path = args.out / args.run_dir.name / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"run": args.run_dir.name, "condition": name, "agents": by_agent}))


if __name__ == "__main__":
    main()
