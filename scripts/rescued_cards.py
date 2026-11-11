"""Build judging cards for rescued memories.

For every tick the engine logs what it put in the prompt (selected) and what
plain decay memory would have put there (solo_selected). A rescue is a memory
id in selected and not in solo_selected that the agent did not already have:
not selected the tick before, and older than the recent turns the prompt
already shows (--include-repeats counts those too). Each moment with a rescue becomes one
card. With --anchor N, N moments are sampled evenly across kinds of rescue and
each gives two cards, the rescued memory and the memory it pushed out, so the
two can be compared blind at the same moment.

Usage: rescued_cards.py <run_dir> [--out DIR] [--history 8] [--min-age 0] [--anchor 60]
where run_dir is a TerraLingua logs/<exp>/ folder.
"""
import argparse
import ast
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# How the agents' own memory lines were written. Graders read them through
# judging.plain(), so a change to this format must be recorded with the set.
MEMORY_TEXT_FORMAT = "raw_json_v1"
TRUE_RESCUE_SCORE = 0.15
HELD_OUT_SLOTS = (0, 2, 5, 7)  # four in every ten sampled moments


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_jsonl(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def as_tick(value):
    return int(value) if float(value).is_integer() else value


def cell_distance(key):
    """Manhattan steps from the agent to a cell key like '(2, -1)'."""
    try:
        rx, ry = ast.literal_eval(str(key))
        return abs(int(rx)) + abs(int(ry))
    except (ValueError, SyntaxError, TypeError):
        return None


def describe_scene(obs):
    """Words, not coordinates. People and artifacts are read the same way
    HouMemoryAgent._extract_entities reads them."""
    if not obs:
        return ["No observation was logged for this tick."]
    people, artifacts, food = {}, set(), []
    for key, contents in (obs.get("observation") or {}).items():
        dist = cell_distance(key)
        for item in contents:
            item = str(item)
            if item.startswith("A("):
                artifacts.add(item.split(": ", 1)[-1])
            elif item in ("X", "") or is_number(item):
                if is_number(item):
                    food.append(dist)
            else:
                name = item.rsplit("(", 1)[0] if item.endswith(")") else item
                if name not in people or (dist is not None and dist < people[name]):
                    people[name] = dist
    lines = []
    energy, left = obs.get("energy"), obs.get("time")
    bits = [f"Energy {energy}." if energy is not None else "Energy unknown."]
    if left is not None:
        bits.append(f"{as_tick(left)} ticks of life left.")
    lines.append(" ".join(bits))
    if people:
        parts = [f"{n} ({d} steps away)" if d is not None else n for n, d in sorted(people.items())]
        lines.append("Nearby: " + ", ".join(parts) + ".")
    else:
        lines.append("Nobody else in sight.")
    lines.append("Artifacts nearby: " + (", ".join(sorted(artifacts)) + "." if artifacts else "none."))
    if food:
        known = [d for d in food if d is not None]
        nearest = f", nearest {min(known)} steps away" if known else ""
        lines.append(f"Food in sight: {len(food)} patches{nearest}.")
    else:
        lines.append("No food in sight.")
    carrying = [str(i).split(": ", 1)[-1] if str(i).startswith("A(") else str(i) for i in obs.get("inventory") or []]
    if carrying:
        lines.append("Carrying: " + ", ".join(carrying) + ".")
    places = obs.get("named_places") or []
    if places:
        lines.append("Visible named places: " + ", ".join(places) + ".")
    heard = obs.get("message") or {}
    if heard:
        lines.append("Incoming messages: " + "; ".join(f'{who} said "{msg}"' for who, msg in heard.items()) + ".")
    else:
        lines.append("Incoming messages: none.")
    return lines


def view_of(obs):
    """What the agent could see, for the card's map. Cells are (row, column)
    from the agent, with rows growing downward, so "up" is one row up."""
    if not obs:
        return None
    cells = []
    for key, contents in (obs.get("observation") or {}).items():
        try:
            row, col = (int(v) for v in key.strip("()").split(","))
        except ValueError:
            continue
        cells.append([row, col, [str(x) for x in contents]])
    return {"radius": obs.get("vision_radius"), "cells": sorted(cells)}


def history_block(memories, tick, count):
    before = sorted((m for m in memories if m.get("t_created", 0) < tick),
                    key=lambda m: (m.get("t_created", 0), m.get("memory_id", "")))
    return [{"ticks_ago": as_tick(tick - m["t_created"]), "text": m["text"]} for m in before[-count:]] if count else []


def route_summary(row):
    out = []
    for r in row.get("routes") or []:
        src = r.get("source", "")
        try:
            kind, name = json.loads(src.split("entity:", 1)[1])
            src = f"{kind} {name}"
        except (ValueError, IndexError, TypeError):
            pass
        out.append(f"via {src}, {len(r.get('edges', []))} hop(s), contribution {r.get('contribution', 0):.3f}")
    return out or ["no graph route recorded"]


def build_card(run, turn, mem_id, kind, history, condition):
    rec, by_id, rows, tag = turn["rec"], turn["by_id"], turn["rows"], turn["tag"]
    tick = rec["tick"]
    mem = by_id.get(mem_id)
    row = rows.get(mem_id, {})
    text = mem["text"] if mem else row.get("text", "<memory text not found>")
    age = as_tick(tick - mem["t_created"]) if mem else None
    step = rec.get("agent_step") or {}
    action = step.get("action") or {}
    return {
        "card_id": "card-" + hashlib.sha1(f"{run}|{tag}|{tick}|{mem_id}".encode()).hexdigest()[:12],
        "history": history_block(turn["memories"], tick, history),
        "scene": describe_scene(step.get("observation")),
        "memory": {"ticks_ago": age, "text": text},
        "views": {"then": view_of((turn["steps"].get(as_tick(mem["t_created"])) or {}).get("observation")) if mem else None,
                  "now": view_of(step.get("observation"))},
        "sealed": {
            "run": run, "condition": condition, "kind": kind, "agent_tag": tag, "agent_name": turn["name"],
            "tick": as_tick(tick), "memory_id": mem_id, "memory_age": age,
            "routes": route_summary(row), "score": row.get("score"), "solo_score": row.get("solo_score"),
            "relevance": row.get("relevance"), "decay": row.get("decay"),
            "direct": row.get("direct"), "graph": row.get("graph"), "reason": row.get("reason"),
            "selected": list(rec.get("selected", [])), "solo_selected": list(rec.get("solo_selected", [])),
            "rescued": turn["rescued"], "displaced": turn["displaced"],
            "displaced_text": [(by_id.get(i) or {}).get("text") for i in turn["displaced"]],
            "next_step": {k: action.get(k) for k in ("action", "params", "message", "reasoning")},
            "plan_after": step.get("internal_memory"),
        },
    }


def stratum(turn):
    """Age of the rescued memory, true rescue or slot swap, and whether a
    person it names is still here. Rare kinds would vanish in a plain sample."""
    rec, mem_id = turn["rec"], turn["rescued"][0]
    row, mem = turn["rows"].get(mem_id, {}), turn["by_id"].get(mem_id) or {}
    age = rec["tick"] - mem.get("t_created", rec["tick"])
    bucket = "1-3" if age <= 3 else "4-8" if age <= 8 else "9+"
    named = re.search(r"Nearby: ([^.]+)\.", mem.get("text", ""))
    people = [n.strip() for n in named.group(1).split(",")] if named else []
    scene = " ".join(describe_scene((rec.get("agent_step") or {}).get("observation")))
    view = "nobody" if not people else "here" if all(p in scene for p in people) else "gone"
    return bucket, (row.get("solo_score") or 1) <= TRUE_RESCUE_SCORE, view


def sample_moments(turns, count, seed):
    """Take moments from each kind in turn. Skip any that reuse a memory or sit
    close enough to a chosen moment to share its history, so no two sampled
    moments are near copies of each other."""
    pool = [t for t in turns if t["displaced"] and
            t["by_id"].get(t["rescued"][0], {}).get("text") != t["by_id"].get(t["displaced"][0], {}).get("text")]
    random.Random(seed).shuffle(pool)
    strata = defaultdict(list)
    for turn in pool:
        strata[stratum(turn)].append(turn)
    chosen, used, spots = [], set(), defaultdict(list)
    while len(chosen) < count and any(strata.values()):
        for key in sorted(strata, key=str):
            while strata[key] and len(chosen) < count:
                turn = strata[key].pop()
                who = (turn["run"], turn["tag"])
                pair = {(who, turn["rescued"][0]), (who, turn["displaced"][0])}
                near = any(abs(turn["rec"]["tick"] - t) <= 4 for t in spots[who])
                if pair & used or near:
                    continue
                chosen.append(turn)
                used |= pair
                spots[who].append(turn["rec"]["tick"])
                break
    return sorted(chosen, key=lambda t: (str(stratum(t)), t["run"], t["tag"], t["rec"]["tick"]))


def assign_groups(cards):
    """Cards sharing a moment or a memory are near copies, so they get one
    group id and always land on the same side of the practice and test split."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cards:
        s = c["sealed"]
        parent[find(("mem", s["run"], s["agent_tag"], s["memory_id"]))] = find(("tick", s["run"], s["agent_tag"], s["tick"]))
    for c in cards:
        s = c["sealed"]
        root = repr(find(("tick", s["run"], s["agent_tag"], s["tick"])))
        s["group"] = "group-" + hashlib.sha1(f"{s['run']}|{root}".encode()).hexdigest()[:12]


def git_state(path):
    run = lambda *args: subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def agent_steps(logs, tag):
    """The per-tick agent log by timestamp. Older audits lack agent_step, and
    the memory's own tick is only here."""
    path = logs / f"{tag}.jsonl"
    if not path.exists():
        return {}
    return {as_tick(r.get("time", r.get("timestamp"))): r for r in load_jsonl(path)
            if is_number(r.get("time", r.get("timestamp")))}


def load_run(run_dir, min_age, include_repeats=False):
    """Every turn in one run where linked memory changed what came to mind."""
    logs = run_dir / "agent_logs"
    if not logs.is_dir():
        sys.exit(f"error: {logs} is not a directory. Pass a TerraLingua logs/<exp>/ folder.")
    params = load_json(run_dir / "params.json", {})
    run = (params.get("run") or {}).get("exp_name") or run_dir.name
    mem_cfg = load_json(run_dir / "memory_config.json", {})
    condition = (mem_cfg.get("memory") or {}).get("mode") or (params.get("agent") or {}).get("memory_mode")
    if not include_repeats:  # the prompt already shows the last working_history turns
        min_age = max(min_age, (mem_cfg.get("memory") or {}).get("working_history", 0) + 1)
    names = load_json(run_dir / "agent_names.json", {})
    tags = sorted(set(names) | {p.name[:-len("_hou_memory.json")] for p in logs.glob("*_hou_memory.json")}
                  | {p.name[:-len("_memory_audit.jsonl")] for p in logs.glob("*_memory_audit.jsonl")})

    turns, ticks_total, rescues_total, per_agent = [], 0, 0, {}
    for tag in tags:
        audit_path = logs / f"{tag}_memory_audit.jsonl"
        if not audit_path.exists():
            continue  # a being that never took a turn
        records = load_jsonl(audit_path)
        dump = load_json(logs / f"{tag}_hou_memory.json", {})
        memories = dump.get("memories", [])
        by_id = {m.get("memory_id"): m for m in memories}
        name = names.get(tag) or dump.get("agent_name") or tag
        steps = agent_steps(logs, tag)
        per_agent[tag] = 0
        previous = set()
        for rec in records:
            ticks_total += 1
            if condition is None:
                condition = (rec.get("config") or {}).get("mode")
            if not isinstance(rec.get("agent_step"), dict):
                rec["agent_step"] = steps.get(as_tick(rec["tick"]))
            selected, solo = list(rec.get("selected", [])), set(rec.get("solo_selected", []))
            rows = {r["id"]: r for r in rec.get("rows", [])}
            rescued = [i for i in selected if i not in solo and
                       not (i in by_id and rec["tick"] - by_id[i]["t_created"] < min_age)
                       and (include_repeats or i not in previous)]
            previous = set(selected)
            rescues_total += len(rescued)
            per_agent[tag] += len(rescued)
            if rescued:
                rescued.sort(key=lambda i: rows.get(i, {}).get("solo_score") or 0)
                turns.append({"run": run, "condition": condition, "tag": tag, "name": name, "rec": rec,
                              "memories": memories, "by_id": by_id, "rows": rows, "steps": steps, "rescued": rescued,
                              "displaced": [i for i in rec.get("solo_selected", []) if i not in set(selected)]})

    rate = 100.0 * rescues_total / ticks_total if ticks_total else 0.0
    print(f"run {run}: {ticks_total} ticks, {rescues_total} rescues ({rate:.1f} per 100) in {len(turns)} moments")
    print("  per agent: " + (", ".join(f"{t}={n}" for t, n in sorted(per_agent.items())) or "none"))
    info = {"run": run, "run_dir": str(run_dir),
            "actor_model": mem_cfg.get("actor_model") or (params.get("agent") or {}).get("model"),
            "actor_effort": mem_cfg.get("effort", "low"), "plan_notes": mem_cfg.get("plan_notes", not mem_cfg.get("paper_prompt", False)),
            "memory_window": (mem_cfg.get("memory") or {}).get("working_history")}
    return turns, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="one or more TerraLingua logs/<exp>/ folders, pooled")
    ap.add_argument("--out", default=None, help="default output/analysis/rescued/<exp> for a single run")
    ap.add_argument("--history", type=int, default=8, help="episodes written before the tick to show")
    ap.add_argument("--min-age", type=int, default=0, help="skip rescues younger than this many ticks")
    ap.add_argument("--include-repeats", action="store_true",
                    help="also count rescues the agent already had: selected last tick or inside its recent turns")
    ap.add_argument("--anchor", type=int, default=0, help="sample this many moments for human grading, two cards each")
    args = ap.parse_args()

    run_dirs = [Path(d).resolve() for d in args.run_dirs]
    if len(run_dirs) > 1 and not args.out:
        sys.exit("error: pass --out when pooling several runs")
    out_dir = Path(args.out) if args.out else ROOT / "output/analysis/rescued" / run_dirs[0].name
    out_dir.mkdir(parents=True, exist_ok=True)
    turns, runs = [], []
    for run_dir in run_dirs:
        run_turns, info = load_run(run_dir, args.min_age, args.include_repeats)
        turns += run_turns
        runs.append(info)

    cards = []
    if args.anchor:
        seed = "|".join(r["run"] for r in runs)
        for i, turn in enumerate(sample_moments(turns, args.anchor, seed)):
            pair = [build_card(turn["run"], turn, turn[kind][0], kind, args.history, turn["condition"] or "unknown")
                    for kind in ("rescued", "displaced")]
            for card in pair:
                card["sealed"]["split"] = "held_out" if i % 10 in HELD_OUT_SLOTS else "tuning"
            cards += pair
    else:  # the strongest rescue of each moment, the rest stay listed in sealed
        cards = [build_card(t["run"], t, t["rescued"][0], "rescued", args.history, t["condition"] or "unknown")
                 for t in turns]
    assign_groups(cards)

    jsonl_path = out_dir / "cards.jsonl"
    with open(jsonl_path, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps({
        "runs": runs, "repo": git_state(ROOT), "engine": git_state(ROOT / "code/terralingua"),
        "memory_text_format": MEMORY_TEXT_FORMAT, "history_lines": args.history, "anchor_moments": args.anchor,
        "cards": len(cards), "kinds": Counter(c["sealed"]["kind"] for c in cards),
        "splits": Counter(c["sealed"].get("split", "by hash") for c in cards),
    }, indent=2) + "\n")

    print(f"cards written: {len(cards)} -> {jsonl_path}")


if __name__ == "__main__":
    main()
