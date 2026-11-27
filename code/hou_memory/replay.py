"""Replay recorded runs through the memory math, with no TerraLingua, no
embedder and no model. See docs/MEMORY_REPLAY_BRIEF.md.

Each audit line holds the cue similarity of every memory at that tick, which
does not depend on the condition. Feeding those numbers back in replays the
same history under any condition or formula (variants.py): decay, graph
routes and selection change, similarity does not. Replaying the recorded
condition must reproduce the recorded selections exactly, which is checked
first.

What replay cannot show is a change in behavior: the agents acted on the
memories of the recorded condition.

Usage: replay.py <run> [<run> ...] [--variants hou chained noisy_or] [--baseline chained] [--out FILE]
where a run is a logs/<exp>/ folder or just its name under code/terralingua/logs_dev/logs.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from itertools import repeat
from pathlib import Path

try:  # as hou_memory.replay, with code/ on the path
    from .associative_memory import AssociativeMemory, MemoryConfig
    from .hou_memory import HouMemoryStore
    from .variants import VARIANTS, variant_settings
except ImportError:  # on its own, with code/hou_memory on the path
    from associative_memory import AssociativeMemory, MemoryConfig
    from hou_memory import HouMemoryStore
    from variants import VARIANTS, variant_settings

LOGS = Path(__file__).resolve().parent.parent / "terralingua" / "logs_dev" / "logs"

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


def replay_agent(lines, memories, config: MemoryConfig, token_counter=None, engine_class=AssociativeMemory,
                 events=False):
    """One row per recorded tick: what this condition would have selected.
    With events, each row also keeps the audit line the engine would have written."""
    store = HouMemoryStore(embed_fn=lambda text: None)
    engine = engine_class(store, config, token_counter)
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
        event = engine.commit(result)
        rescued = [i for i in result.selected if i not in result.solo_selected]
        # Fresh: not already in the prompt, either selected last tick or among
        # the recent turns the prompt shows. The same rule as rescued_cards.py.
        fresh = [i for i in rescued if i not in previous
                 and line["tick"] - created[i] > config.working_history]
        previous = set(result.selected)
        out.append({"tick": line["tick"], "selected": list(result.selected),
                    "solo_selected": list(result.solo_selected), "rescued": rescued, "fresh": fresh,
                    **({"event": event} if events else {})})
    return out


def recorded_engine(run_dir: Path):
    """The engine class a run was made with. Runs saved before variants used the stock one."""
    try:
        name = json.loads((run_dir / "memory_config.json").read_text()).get("variant")
    except OSError:
        name = None
    return VARIANTS[name].engine if name else AssociativeMemory


def condition_config(recorded: dict, name: str) -> MemoryConfig:
    """The run's own settings under another variant. What the run's own
    variant set goes back to the default first, so learned on a run made with
    chained is one hop at .8 again. Then this variant's settings go on top."""
    names = {f.name for f in fields(MemoryConfig)}
    # The run's own variant: the one with the most settings that all match.
    own = max((variant_settings(v) for v in VARIANTS
               if all(recorded.get(k) == x for k, x in variant_settings(v).items())), key=len, default={})
    return MemoryConfig(**{**{k: v for k, v in recorded.items() if k in names and k not in own},
                           **variant_settings(name)})


def replay_tag(agent_logs: Path, tag: str, names):
    """Check the recorded condition, then replay one agent under each variant."""
    lines, memories = load_agent(agent_logs, tag)
    if not lines:
        return 0, 0, {}
    recorded = lines[-1]["config"]
    rows = replay_agent(lines, memories, MemoryConfig(**recorded), engine_class=recorded_engine(agent_logs.parent))
    matched = sum(r["selected"] == l["selected"] and r["solo_selected"] == l["solo_selected"]
                  for r, l in zip(rows, lines))
    return len(lines), matched, {name: replay_agent(lines, memories, condition_config(recorded, name),
                                                    engine_class=VARIANTS[name].engine)
                                 for name in names}


def replay_run(run_dir: Path, names):
    agent_logs = run_dir / "agent_logs"
    tags = sorted(p.name.removesuffix("_memory_audit.jsonl") for p in agent_logs.glob("*_memory_audit.jsonl"))
    check = {"ticks": 0, "matched": 0}
    results = {name: {} for name in names}
    # One process per agent: chained routes take about a minute per 300-tick agent.
    with ProcessPoolExecutor() as pool:
        for tag, (ticks, matched, by_variant) in zip(tags, pool.map(replay_tag, repeat(agent_logs), tags, repeat(names))):
            check["ticks"] += ticks
            check["matched"] += matched
            for name, rows in by_variant.items():
                results[name][tag] = rows
    return check, results


def moments(by_agent, key):
    """(agent, tick, memory) for every rescue of one kind."""
    return {(tag, row["tick"], i) for tag, rows in by_agent.items() for row in rows for i in row[key]}


def compare(results, baseline):
    """Per variant: counts, and how its fresh rescues differ from the baseline's."""
    base = moments(results[baseline], "fresh") if baseline else set()
    table = {}
    for name, by_agent in results.items():
        fresh = moments(by_agent, "fresh")
        table[name] = {"ticks": sum(len(rows) for rows in by_agent.values()),
                       "rescued": len(moments(by_agent, "rescued")), "fresh": len(fresh),
                       "shared": len(fresh & base), "only_here": len(fresh - base), "only_baseline": len(base - fresh)}
    return table


def run_path(name: str) -> Path:
    path = Path(name)
    return path if (path / "agent_logs").is_dir() else LOGS / name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="logs/<exp>/ folders, or run names under code/terralingua/logs_dev/logs")
    ap.add_argument("--variants", nargs="+", default=["hou", "learned", "chained"], choices=VARIANTS)
    ap.add_argument("--baseline", choices=VARIANTS, help="compare each variant's fresh rescues with this one (default: the first)")
    ap.add_argument("--out", type=Path, help="write every variant's per-tick selections here, as JSON")
    args = ap.parse_args()
    baseline = args.baseline or args.variants[0]
    names = list(dict.fromkeys([baseline, *args.variants]))
    pooled, saved = {}, {}
    for run in args.runs:
        run_dir = run_path(run)
        check, results = replay_run(run_dir, names)
        print(f"{run_dir.name}: recorded condition reproduced on {check['matched']} of {check['ticks']} ticks")
        if check["matched"] != check["ticks"]:
            raise SystemExit("Replay does not match the recording, so the variants are not trustworthy")
        for name, by_agent in results.items():
            pooled.setdefault(name, {}).update({(run_dir.name, tag): rows for tag, rows in by_agent.items()})
        saved[run_dir.name] = results
    print(f"\n{'variant':12} {'ticks':>6} {'rescued':>8} {'fresh':>6}   fresh vs {baseline}: shared, only here, only {baseline}")
    for name, row in compare(pooled, baseline).items():
        print(f"{name:12} {row['ticks']:>6} {row['rescued']:>8} {row['fresh']:>6}   "
              f"{row['shared']:>6} {row['only_here']:>9} {row['only_baseline']:>9}   {VARIANTS[name].about}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"baseline": baseline, "runs": saved}))


if __name__ == "__main__":
    main()
