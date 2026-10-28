"""Build judging cards for rescued memories.

For every tick the engine logs what it put in the prompt (selected) and what
plain decay memory would have put there (solo_selected). A rescue is a memory
id in selected and not in solo_selected. Each rescue becomes one card with
three plain-English blocks and one question for a human or LLM judge.

Usage: rescued_cards.py <run_dir> [--out DIR] [--history 8] [--min-age 0]
where run_dir is a TerraLingua logs/<exp>/ folder.
"""
import argparse
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = ("Would a real person in this spot have this come to mind, "
            "and does it matter for what to do next?")
ANSWERS = ["Belongs", "Doesn't matter", "Noise"]


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


def ago(n):
    n = as_tick(n)
    return "1 tick ago" if n == 1 else f"{n} ticks ago"


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


def card_text(card):
    lines = ["What this agent has been through:"]
    lines += [f"- {ago(h['ticks_ago'])}: {h['text']}" for h in card["history"]] or ["- Nothing recorded before this point."]
    lines += ["", "What it sees right now:"] + card["scene"]
    lines += ["", "The memory that came up:", f"From {ago(card['memory']['ticks_ago'])}: {card['memory']['text']}"]
    lines += ["", "Question: " + QUESTION] + [f"- {a}" for a in ANSWERS] + ["Reason (one line):"]
    return "\n".join(lines)


def build_card(run, tag, name, rec, mem_id, memories, by_id, rows, displaced, history, condition):
    tick = rec["tick"]
    mem = by_id.get(mem_id)
    row = rows.get(mem_id, {})
    text = mem["text"] if mem else row.get("text", "<memory text not found>")
    age = as_tick(tick - mem["t_created"]) if mem else None
    card = {
        "card_id": "card-" + hashlib.sha1(f"{run}|{tag}|{tick}|{mem_id}".encode()).hexdigest()[:12],
        "history": history_block(memories, tick, history),
        "scene": describe_scene((rec.get("agent_step") or {}).get("observation")),
        "memory": {"ticks_ago": age, "text": text},
        "question": QUESTION, "answers": ANSWERS,
    }
    card["text"] = card_text(card)
    card["sealed"] = {
        "run": run, "condition": condition, "agent_tag": tag, "agent_name": name,
        "tick": as_tick(tick), "memory_id": mem_id, "memory_age": age,
        "routes": route_summary(row), "score": row.get("score"), "solo_score": row.get("solo_score"),
        "relevance": row.get("relevance"), "decay": row.get("decay"),
        "direct": row.get("direct"), "graph": row.get("graph"), "reason": row.get("reason"),
        "selected": list(rec.get("selected", [])), "solo_selected": list(rec.get("solo_selected", [])),
        "displaced": displaced,
    }
    return card


def agent_steps_fallback(logs, tag):
    """Older audits lack agent_step. Join the per-tick agent log by its timestamp."""
    path = logs / f"{tag}.jsonl"
    if not path.exists():
        return {}
    return {as_tick(r.get("time", r.get("timestamp"))): r for r in load_jsonl(path)
            if is_number(r.get("time", r.get("timestamp")))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="default output/analysis/rescued/<exp>")
    ap.add_argument("--history", type=int, default=8, help="episodes written before the tick to show")
    ap.add_argument("--min-age", type=int, default=0, help="skip rescues younger than this many ticks")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs = run_dir / "agent_logs"
    if not logs.is_dir():
        sys.exit(f"error: {logs} is not a directory. Pass a TerraLingua logs/<exp>/ folder.")
    params = load_json(run_dir / "params.json", {})
    run = (params.get("run") or {}).get("exp_name") or run_dir.name
    mem_cfg = load_json(run_dir / "memory_config.json", {})
    condition = (mem_cfg.get("memory") or {}).get("mode") or (params.get("agent") or {}).get("memory_mode")
    names = load_json(run_dir / "agent_names.json", {})
    tags = sorted(set(names) | {p.name[:-len("_hou_memory.json")] for p in logs.glob("*_hou_memory.json")}
                  | {p.name[:-len("_memory_audit.jsonl")] for p in logs.glob("*_memory_audit.jsonl")})
    out_dir = Path(args.out) if args.out else ROOT / "output/analysis/rescued" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cards, ticks_total, rescues_total, per_agent = [], 0, 0, {}
    for tag in tags:
        audit_path = logs / f"{tag}_memory_audit.jsonl"
        if not audit_path.exists():
            print(f"warning: no audit file for {tag} ({audit_path.name} missing), skipping", file=sys.stderr)
            continue
        records = load_jsonl(audit_path)
        dump = load_json(logs / f"{tag}_hou_memory.json", {})
        memories = dump.get("memories", [])
        by_id = {m.get("memory_id"): m for m in memories}
        name = names.get(tag) or dump.get("agent_name") or tag
        fallback = None
        per_agent[tag] = 0
        for rec in records:
            ticks_total += 1
            if condition is None:
                condition = (rec.get("config") or {}).get("mode")
            if not isinstance(rec.get("agent_step"), dict):
                fallback = agent_steps_fallback(logs, tag) if fallback is None else fallback
                rec["agent_step"] = fallback.get(as_tick(rec["tick"]))
            selected, solo = list(rec.get("selected", [])), set(rec.get("solo_selected", []))
            rescued = [i for i in selected if i not in solo]
            displaced = [i for i in rec.get("solo_selected", []) if i not in set(selected)]
            rows = {r["id"]: r for r in rec.get("rows", [])}
            rescues_total += len(rescued)
            per_agent[tag] += len(rescued)
            for mem_id in rescued:
                mem = by_id.get(mem_id)
                if mem and rec["tick"] - mem["t_created"] < args.min_age:
                    continue
                cards.append(build_card(run, tag, name, rec, mem_id, memories, by_id, rows, displaced,
                                        args.history, condition or "unknown"))

    jsonl_path = out_dir / "cards.jsonl"
    with open(jsonl_path, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    shuffled = list(cards)
    random.Random(run).shuffle(shuffled)
    noun = "card" if len(shuffled) == 1 else "cards"
    md = ["# Rescued memory judging cards", "",
          f"{len(shuffled)} {noun}. Read each card and answer the question with one of the three answers and a one-line reason.", ""]
    for c in shuffled:
        md += [f"## Card {c['card_id']}", "", c["text"], ""]
    md_path = out_dir / "cards.md"
    md_path.write_text("\n".join(md))

    rate = 100.0 * rescues_total / ticks_total if ticks_total else 0.0
    print(f"run {run} ({run_dir})")
    print(f"ticks scanned: {ticks_total}")
    print(f"rescues found: {rescues_total} ({rate:.1f} per 100 ticks)")
    print("per agent: " + (", ".join(f"{t}={n}" for t, n in sorted(per_agent.items())) or "none"))
    print(f"cards written: {len(cards)} -> {jsonl_path} and {md_path}")


if __name__ == "__main__":
    main()
