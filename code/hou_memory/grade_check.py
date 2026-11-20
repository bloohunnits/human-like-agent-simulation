"""Score memory formulas against the human grades, with no sim and no model.

Each graded card is one memory at one moment of a recorded run. This rebuilds
the memory exactly as it was at that moment (stock engine, recorded settings),
then asks each variant to score the store there. Every variant sees the same
history, so only the formula differs.

The grades are few, and every graded memory was picked or dropped by the
recorded formula. Treat this as a check while designing, not a result.

Usage: grade_check.py [--variants hou chained actr] [--cards FOLDER ...]
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from associative_memory import AssociativeMemory, MemoryConfig
from hou_memory import HouMemoryStore
from replay import LOGS, condition_config, entities_of, load_agent
from variants import VARIANTS

CARDS = Path(__file__).resolve().parents[2] / "output" / "analysis" / "rescued"


def load_grades(folders):
    """(run, agent, tick, memory) -> label. A later folder wins on a repeat."""
    grades = {}
    for folder in folders:
        sealed = {c["card_id"]: c["sealed"] for c in map(json.loads, (folder / "cards.jsonl").read_text().splitlines())}
        for ratings in sorted(folder.glob("ratings_*.csv")):
            for row in csv.DictReader(ratings.open()):
                s = sealed[row["card_id"]]
                grades[(s["run"], s["agent_tag"], s["tick"], s["memory_id"])] = row["label"]
    return grades


def score_agent(run, tag, wanted, names):
    """wanted: tick -> memory ids. Returns (variant, tick, memory) -> its row."""
    lines, memories = load_agent(LOGS / run / "agent_logs", tag)
    recorded = lines[-1]["config"]
    stock = MemoryConfig(**recorded)
    engine = AssociativeMemory(HouMemoryStore(embed_fn=lambda text: None), stock)
    engine._state_digest = lambda: "replay"
    pending = iter(memories)
    upcoming = next(pending, None)
    out = {}
    for line in lines:
        if line["tick"] > max(wanted):
            break
        while upcoming is not None and upcoming["t_created"] < line["tick"]:
            engine.add(upcoming["text"], upcoming["t_created"], upcoming.get("meta"))
            upcoming = next(pending, None)
        call = dict(cue=line["cue"], tick=line["tick"], entities=entities_of(line["sources"]),
                    event_id=line["event_id"], r_by_id={row["id"]: row["relevance"] for row in line["rows"]})
        for name in names if line["tick"] in wanted else ():
            # Same store and links, this variant's formulas. Preview changes nothing.
            engine.__class__, engine.config = VARIANTS[name].engine, condition_config(recorded, name)
            result = engine.preview(**call)
            for s in result.scores:
                if s.id in wanted[line["tick"]]:
                    out[(name, line["tick"], s.id)] = {"picked": s.id in result.selected, "value": s.combined,
                                                      "clears": s.score > engine.config.threshold}
        engine.__class__, engine.config = AssociativeMemory, stock
        engine.commit(engine.preview(**call))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="+", default=["hou", "chained", "actr_solo", "actr"], choices=VARIANTS)
    ap.add_argument("--cards", nargs="+", type=Path, help="card folders with ratings_*.csv (default: every one under output/analysis/rescued)")
    args = ap.parse_args()
    grades = load_grades(args.cards or sorted(p for p in CARDS.iterdir() if (p / "cards.jsonl").exists()))
    wanted = {}
    for run, tag, tick, memory in grades:
        wanted.setdefault((run, tag), {}).setdefault(tick, set()).add(memory)
    rows = {}
    for (run, tag), ticks in wanted.items():
        for (name, tick, memory), row in score_agent(run, tag, ticks, args.variants).items():
            rows.setdefault(name, []).append((grades[(run, tag, tick, memory)] == "belongs", row))
    print(f"{len(grades)} graded memories. Good: graded 'belongs'. Rest: graded 'does not matter' or 'noise'.\n")
    print(f"{'variant':12} {'picked good':>12} {'picked rest':>12} {'clears good':>12} {'clears rest':>12} {'ranks good above rest':>22}")
    for name in args.variants:
        good = [row for is_good, row in rows[name] if is_good]
        rest = [row for is_good, row in rows[name] if not is_good]
        count = lambda group, key: f"{sum(r[key] for r in group)}/{len(group)}"
        above = mean((g["value"] > r["value"]) + .5 * (g["value"] == r["value"]) for g in good for r in rest)
        print(f"{name:12} {count(good, 'picked'):>12} {count(rest, 'picked'):>12} {count(good, 'clears'):>12} "
              f"{count(rest, 'clears'):>12} {above:>21.0%}")


if __name__ == "__main__":
    main()
