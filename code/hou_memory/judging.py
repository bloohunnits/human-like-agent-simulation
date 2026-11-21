"""Shared pieces of the believability judge: card sets, versioned judge
instructions, human ratings, judge runs, and agreement statistics.

Used by the Rescued set page in God Mode (webapp.py, /rescued). Files:

  output/analysis/rescued/<run>/cards.jsonl         the rescued set, from scripts/rescued_cards.py
  output/analysis/rescued/<run>/ratings_<rater>.csv human labels
  output/analysis/rescued/<run>/pairs_<rater>.csv   which of two memories helps more
  output/analysis/rescued/<run>/judge_runs/<prompt>__<model>.jsonl  judge labels
  output/analysis/judge_prompts/<prompt>.json     versioned judge instructions

Every card belongs permanently to the tuning group or the held-out group,
decided by its id. People tune the judge by reading its disagreements on
tuning cards only. The judge passes when it agrees with the human majority on
held-out cards, which nobody tuned against.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARDS_ROOT = ROOT / "output/analysis/rescued"
PROMPTS_ROOT = ROOT / "output/analysis/judge_prompts"
LABELS = ("belongs", "doesnt_matter", "noise")
LABEL_NAMES = {"belongs": "Yes, and it matters", "doesnt_matter": "Yes, but it doesn't matter", "noise": "No"}
JUDGE_MODELS = ("gpt-5-mini", "gpt-5.6-luna", "gpt-6-luna", "gpt-5.4-mini", "gpt-5-nano")
GATE_KAPPA, GATE_MIN_CARDS, HELD_OUT_SHARE = 0.61, 20, 0.3
ANCHOR_SET = "anchor_v4"  # the set classmates grade: the newer engine's rescues by any formula, past the warm-up
# A world's first turns are not graded. In the seven graded worlds fewer than
# 1 turn in 100 before turn 30 is anything but a step, and about 6 in 100 after.
WARMUP_TURNS = 30
# Worlds are marked by engine, and people grade the newer one's cards only.
# Only its worlds have gifts, thefts and births as well as notes.
NEWER_ENGINE = "terralingua-relational-memory"
# The three of us each grade every card, so every card gets three grades.
# Names are matched without case.
RATERS = ("andre", "ben", "ryan")
# People and the judge read these same words.
QUESTION = "Out of everything this person remembers, this came to mind. Does that make sense right now?"
ANSWER_HELP = {
    "belongs": "A real person could think of this here, and it helps them pick what to do next.",
    "doesnt_matter": "A real person could think of this here, but it changes nothing about what they do next.",
    "noise": "A real person would not think of this here, or it would send them the wrong way.",
}
# A pair card shows one moment and two memories: the one links brought to mind
# and the one it pushed out. People see them as A and B in an order of their
# own. The answer is stored by memory, never by letter.
PAIR_QUESTION = "Which memory helps this person more right now?"
PAIR_ANSWERS = {"a": "Memory A", "b": "Memory B", "both": "Both help about the same", "neither": "Neither helps"}
NEAR_COPY = 0.73  # share of words two memories have in common. This alike, the pair is a tie and nobody is asked
# People standing together get cards that read almost alike. Moments in one
# world this few turns apart that share this many of their words are one scene.
SCENE_TURNS, SCENE_ALIKE = 4, 0.6
REPLY_FORMAT = ('First give one or two sentences of reasoning, then the label. Reply as json: '
                '{"reason": "...", "label": "belongs" or "doesnt_matter" or "noise"}')

V1 = {
    "name": "v1",
    "parent": None,
    "note": "First version, written before any human labels.",
    "world": (
        "You are helping check whether simulated people remember things the way real people do. "
        "Each case shows what one person in a small survival world did recently (What happened), what they "
        "see right now (Right now), and one memory that just came to their mind (This came to mind). In this world a person loses 1 energy "
        "every turn and dies at 0. They regain energy by stepping on food or when someone gives it to them. "
        "They can move, talk to people in sight, give or take energy, and leave text notes. "
        "Their memories are short notes of what they did, noticed and heard. "
        "If the memory is also listed under What happened, that line is marked. "
        "Being listed there is not by itself a reason to say yes."
    ),
    "task": QUESTION,
    "definitions": ANSWER_HELP,
}


# ---------- files ----------

def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "")).strip("-.")
    if not name:
        raise ValueError("a name is required")
    return name[:60]


def card_sets() -> list[str]:
    if not CARDS_ROOT.exists():
        return []
    return sorted(p.parent.name for p in CARDS_ROOT.glob("*/cards.jsonl"))


def set_dir(card_set: str) -> Path:
    path = CARDS_ROOT / safe_name(card_set)
    if not (path / "cards.jsonl").exists():
        raise ValueError(f"no card set named {card_set}")
    return path


def load_cards(card_set: str) -> list[dict]:
    path = set_dir(card_set) / "cards.jsonl"
    cards = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cards = [c for c in cards if c.get("sealed", {}).get("tick", WARMUP_TURNS) >= WARMUP_TURNS]
    cards.sort(key=lambda c: c["card_id"])
    random.Random(card_set).shuffle(cards)
    return cards


def engines(card_set: str) -> dict[str, str]:
    """Each world in a set, marked newer or older by the engine it ran on.
    A set built before worlds were marked gives nothing."""
    path = set_dir(card_set) / "manifest.json"
    runs = json.loads(path.read_text()).get("runs", []) if path.exists() else []
    return {r["run"]: "newer" if r["engine"] == NEWER_ENGINE else "older" for r in runs if r.get("engine")}


def split_of(card: dict) -> str:
    """Sampled sets store their split. Otherwise hash the group, so near copies
    of one moment or memory never sit on both sides."""
    sealed = card.get("sealed", {})
    if sealed.get("split"):
        return sealed["split"]
    digits = re.sub(r"[^0-9a-f]", "", (sealed.get("group") or card["card_id"]).removeprefix("group-"))[:8] or "0"
    return "held_out" if int(digits, 16) % 100 < HELD_OUT_SHARE * 100 else "tuning"


def text_hash(card: dict) -> str:
    """Covers the text and the maps, so a grade stops counting if either changes."""
    return hashlib.sha1(judge_text(card).encode()).hexdigest()[:12]


def _names(items: list[str]) -> str:
    items = [i.strip() for i in items if i.strip()]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _note(name) -> str:
    """note_for_allies_79 reads as "note for allies"."""
    return '"' + re.sub(r"_\d+$", "", str(name).strip()).replace("_", " ") + '"'


def _notes(names) -> str:
    """Note names in plain words. Notes that share a name are counted, not repeated."""
    counts = Counter(_note(n) for n in names if n.strip())
    return _names([name if n == 1 else f"{name} ({n} notes)" for name, n in counts.items()])


def _note_text(value: str) -> str:
    """The newer engine's note dump, without who made it and when."""
    m = re.match(r"Artifact (\S+)\s+Created by .*?Content: (.*)", value, re.S)
    return f'The note {_note(m.group(1))} says: "{m.group(2).strip()}"' if m else value


def _note_name(action: str, params: dict) -> str:
    """Some models put the note's name in the action: modify_artifact_east_claim_9."""
    return params.get("name") or re.sub(r"^(modify|destroy)_artifact_?", "", action) or "untitled"


def _action(name: str, params: dict) -> str:
    p = params or {}
    if name == "move":
        d = p.get("direction", "")
        return "Stayed put." if d == "stay" else f"Moved {d}."
    if name == "give":
        return f"Tried to give {p.get('target')} {p.get('amount')} energy."
    if name == "take":
        return f"Tried to take {p.get('amount')} energy from {p.get('target')}."
    if name == "create_artifact":
        return f'Wrote a note called {_note(p.get("name"))}.'
    if name == "pickup_artifact":
        return f'Picked up the note {_note(p.get("name"))}.'
    if name == "drop_artifact":
        return f'Dropped the note {_note(p.get("name"))}.'
    if name == "give_artifact":
        return f'Gave the note {_note(p.get("artifact_name"))} to {p.get("target_agent")}.'
    if name == "reproduce":
        return f"Had a child named {p.get('name')}."
    if name == "spawn":
        return f"Asked {p.get('partner')} to have a child."
    if name.startswith(("modify_artifact", "destroy_artifact")):
        return f"{'Rewrote' if name.startswith('modify') else 'Destroyed'} the note {_note(_note_name(name, p))}."
    return f"Did {name.replace('_', ' ')}."


def _flat(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in _flat(item)]
    return [str(value)]


def _speech(text: str) -> str:
    """Two people talking become two plain sentences."""
    return re.sub(r'\.\"\.', '."', text.replace('"; ', '". ')).replace('."."', '."')


def plain(text: str) -> str:
    """Rewrite one stored memory line in everyday words, for people and judge alike."""
    def feedback(m):
        try:
            found = ast.literal_eval(re.sub(r"np\.\w+\(([^()]*)\)", r"\1", m.group(1)))
            values = [v for item in found.values() for v in _flat(item)]
        except (ValueError, SyntaxError):
            values = [m.group(1)]
        values = [_note_text(v) for v in values]
        said = " ".join(v if v.endswith('"') else v.rstrip(".") + "." for v in values)
        said = re.sub(r"Artifact (.+?) content: ", lambda m: f"The note {_note(m.group(1))} says: ", said)
        return "The world said: " + said.replace("another being", "someone")

    chosen = re.search(r"Chose action (\w+): ", text)
    if chosen:
        try:  # a note's text can hold any character, so read the JSON properly
            params, end = json.JSONDecoder().raw_decode(text, chosen.end())
            end += text[end:end + 1] == "."
            text = text[:chosen.start()] + _action(chosen.group(1), params) + text[end:]
        except ValueError:
            pass
    text = re.sub(r"Nearby: ([^.]+)\.", lambda m: f"Near {_names(m.group(1).split(','))}.", text)
    text = re.sub(r"Artifacts here: ([^.]+)\.", lambda m: f"Saw a note: {_notes(m.group(1).split(','))}.", text)
    text = re.sub(r"Visible places: ([^.]+)\.",
                  lambda m: "Could see " + _names([f"the {x.strip()}" for x in m.group(1).split(",")]) + ".", text)
    text = re.sub(r"Feedback from the preceding action: (\{.*?\})\.(?= |$)", feedback, text)
    text = re.sub(r"\bHeard (?=\w+ said \")", "", text)
    return re.sub(r"\s+", " ", _speech(text)).strip()


def _quotes(heard: str) -> list[tuple[str, str]]:
    """'Ada said "hi"; Ben said "ok"' as (speaker, words) pairs."""
    starts = list(re.finditer(r'(?:^|"; )([\w.-]+) said "', heard))
    out = []
    for i, m in enumerate(starts):
        last = i + 1 == len(starts)
        words = heard[m.end():len(heard) if last else starts[i + 1].start()]
        out.append((m.group(1), words[:-1] if last and words.endswith('"') else words))
    return out


def _deed(name: str, params: dict, who: str) -> str:
    p = params or {}
    if name == "move":
        d = p.get("direction", "")
        return f"{who} stays put." if d == "stay" else f"{who} moves {d}."
    if name == "give":
        return f"{who} tries to give {p.get('target')} {p.get('amount')} energy."
    if name == "take":
        return f"{who} tries to take {p.get('amount')} energy from {p.get('target')}."
    wrote = f': "{p["payload"]}"' if p.get("payload") else "."
    if name == "create_artifact":
        return f'{who} writes a note called {_note(p.get("name"))}{wrote}'
    if name == "pickup_artifact":
        return f'{who} picks up the note {_note(p.get("name"))}.'
    if name == "drop_artifact":
        return f'{who} drops the note {_note(p.get("name"))}.'
    if name == "give_artifact":
        return f'{who} gives the note {_note(p.get("artifact_name"))} to {p.get("target_agent")}.'
    if name == "reproduce":
        return f"{who} has a child named {p.get('name')}."
    if name == "spawn":
        return f"{who} asks {p.get('partner')} to have a child" + (f" named {p['name']}." if p.get("name") else ".")
    if name.startswith("modify_artifact"):
        return f"{who} rewrites the note {_note(_note_name(name, p))}{wrote}"
    if name.startswith("destroy_artifact"):
        return f"{who} destroys the note {_note(_note_name(name, p))}."
    return f"{who} does {name.replace('_', ' ')}."


def _world(found: str, who: str) -> str:
    text = plain(f"Feedback from the preceding action: {found}.").removeprefix("The world said: ")
    # "you" is the person, except inside a note or a quote, which keep their words.
    parts = re.split(r'("[^"]*")', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\b[Yy]ou\b", who, re.sub(r"\b[Yy]our\b", f"{who}'s", parts[i]))
    return "".join(parts)


def _turn(text: str, who: str) -> list[str]:
    """One stored memory as script lines, in the order things happened: what
    was around, what the world reported, what was heard, then the person's own
    move and words. Memories are stored in the first person, so the name goes back in."""
    said = world = deed = None
    heard = []
    if m := re.search(r'(?:^|\. )I said "(.*)"\.$', text, re.S):
        said, text = m.group(1), text[:m.start() + 1]
    if m := re.search(r'(?:^|\. )Heard (.*)\.$', text, re.S):
        heard, text = _quotes(m.group(1)), text[:m.start() + 1]
    if m := re.search(r"(?:^|\. )Feedback from the preceding action: (\{.*\})\.$", text, re.S):
        world, text = _world(m.group(1), who), text[:m.start() + 1]
    if m := re.match(r"Chose action (\w+): ", text):
        try:  # a note's text can hold any character, so read the JSON properly
            params, end = json.JSONDecoder().raw_decode(text, m.end())
            deed, text = _deed(m.group(1), params, who), text[end:]
        except ValueError:
            pass
    setting = []
    if m := re.search(r"Nearby: ([^.]+)\.", text):
        people = m.group(1).split(",")
        setting.append(f"{_names(people)} {'is' if len(people) == 1 else 'are'} nearby.")
    if m := re.search(r"Artifacts here: ([^.]+)\.", text):
        notes = m.group(1).split(",")
        setting.append(f"{who} sees {'a note' if len(notes) == 1 else 'notes'}: {_notes(notes)}.")
    if m := re.search(r"Visible places: ([^.]+)\.", text):
        setting.append(f"{who} can see " + _names([f"the {x.strip()}" for x in m.group(1).split(",")]) + ".")
    if m := re.search(r"Energy (rose|fell) from (\S+) to (\S+?)\.(?= |$)", text):
        setting.append(f"{who}'s energy {'rises' if m.group(1) == 'rose' else 'falls'} from {m.group(2)} to {m.group(3)}.")
    lines = [f"({' '.join(setting)})"] if setting else []
    if world:
        lines.append(f"({world})")
    lines += [f'{person}: "{words}"' for person, words in heard]
    if deed:
        lines.append(deed)
    if said is not None:
        lines.append(f'{who}: "{said}"')
    return lines


def _ago(n) -> str:
    return "1 turn ago" if n == 1 else f"{n} turns ago"


def _story(history: list[dict], who: str, memory: dict | None = None) -> list[str]:
    """What happened, oldest first, as a script. Repeated turns fold into one,
    long lines already spoken are not repeated, and the graded memory is marked."""
    memory = memory or {}
    groups = []
    for h in history:
        lines = _turn(h["text"], who)
        marked = h["ticks_ago"] == memory.get("ticks_ago") and h["text"] == memory.get("text")
        if groups and groups[-1]["lines"] == lines and not marked and not groups[-1]["marked"]:
            groups[-1]["last"] = h["ticks_ago"]
            groups[-1]["count"] += 1
        else:
            groups.append({"lines": lines, "first": h["ticks_ago"], "last": h["ticks_ago"], "count": 1, "marked": marked})
    seen, notes = set(), set()

    def once(line):
        if line.startswith("(") and ' says: "' in line:  # a note on the ground, read every turn
            if line in notes:
                return "(The same notes as before.)"
            notes.add(line)
            return line
        m = re.fullmatch(r'([\w.-]+): "(.*)"', line, re.S)
        if not m:
            return line
        if m.groups() in seen and len(m.group(2).split()) > 3:
            return f"{m.group(1)}: (says the same thing again)"
        seen.add(m.groups())
        return line

    out = []
    for g in groups:
        if g["marked"]:  # shown in full, but later turns still count it as already said
            head, lines = f"{_ago(g['first'])} [this is the memory below]", g["lines"]
            [once(x) for x in lines]
        else:
            head = _ago(g["first"]) if g["count"] == 1 else f"{g['first']} to {g['last']} turns ago, the same each turn"
            lines = [once(x) for x in g["lines"]]
        out += [head] + [f"  {x}" for x in lines or ["(nothing)"]] + [""]
    return out[:-1] or ["Nothing yet."]


def _now(scene: list[str], who: str) -> list[str]:
    out = []
    for line in scene:
        if m := re.fullmatch(r"Energy (\d+(?:\.\d+)?)\.(?: (\d+) ticks of life left\.)?", line):
            energy = m.group(1).removesuffix(".0")
            out.append(f"{who} has {energy} energy" + (f" and {m.group(2)} turns of life left." if m.group(2) else "."))
        elif people := re.findall(r"([\w.-]+) \((\d+) steps? away\)", line):
            out.append("(" + _names([f"{name} is {n} step{'' if n == '1' else 's'} away" for name, n in people]) + ".)")
        elif m := re.fullmatch(r"Food in sight: \d+ patch(?:es)?, nearest (\d+) steps? away\.", line):
            out.append(f"(Food is {m.group(1)} step{'' if m.group(1) == '1' else 's'} away.)")
        elif m := re.fullmatch(r"Artifacts nearby: (.+)\.", line):
            if m.group(1) != "none":
                out.append(f"(Notes nearby: {_notes(m.group(1).split(','))}.)")
        elif m := re.fullmatch(r"Incoming messages: (.+)\.", line, re.S):
            if m.group(1) != "none":
                out += [f'{person}: "{words}"' for person, words in _quotes(m.group(1))]
        elif m := re.fullmatch(r"Visible named places: ([^.]+)\.", line):
            out.append(f"({who} can see " + _names([f"the {x.strip()}" for x in m.group(1).split(",")]) + ".)")
        else:
            out.append(f"({line})")
    return out


def visible_text(card: dict) -> str:
    """What a person and the judge both read: what happened, right now, and
    the memory, written as a script with every line naming who did it."""
    who = card.get("sealed", {}).get("agent_name") or "This person"
    memory = card.get("memory", {})
    lines = [f"This is {who}.", "", "What happened:"] + _story(card.get("history", []), who, memory)
    lines += ["", "Right now:"] + [f"  {x}" for x in _now(card.get("scene", []), who)]
    lines += ["", f"This came to {who}'s mind:", _ago(memory.get("ticks_ago"))]
    lines += [f"  {x}" for x in _turn(memory.get("text", ""), who)]
    return "\n".join(lines)


def map_text(view: dict | None, who: str) -> list[str]:
    """The map people see, as a grid of letters for the judge. Top is up."""
    if not view:
        return ["(no map)"]
    r = view.get("radius") or 6
    grid = [["." for _ in range(2 * r + 1)] for _ in range(2 * r + 1)]
    letters = {who: "@"}
    for row, col, items in view["cells"]:
        if abs(row) > r or abs(col) > r:
            continue
        for item in items:
            if re.fullmatch(r"[\d.]+", item):
                mark = "f"
            elif item.startswith("A("):
                mark = "n"
            else:
                if item not in letters:
                    free = [ch for ch in item.upper() + "123456789" if ch.isalnum() and ch not in letters.values()
                            and ch not in "FN"]
                    letters[item] = free[0] if free else "?"
                mark = letters[item]
            if grid[row + r][col + r] in ".fn" or mark not in "fn":
                grid[row + r][col + r] = mark
    grid[r][r] = "@"
    key = [f"@ = {who}"] + [f"{ch} = {name}" for name, ch in letters.items() if name != who] + ["f = food", "n = note"]
    return [" ".join(line) for line in grid] + ["Key: " + ", ".join(key) + "."]


def judge_text(card: dict) -> str:
    """The card text plus its two maps, which people see as pictures."""
    who = card.get("sealed", {}).get("agent_name") or "This person"
    views = card.get("views") or {}
    if not views:
        return visible_text(card)
    lines = [visible_text(card), "", f"Maps of what {who} could see, top is up, one character per square."]
    lines += ["", f"When the memory happened ({_ago(card.get('memory', {}).get('ticks_ago'))}):"]
    lines += map_text(views.get("then"), who)
    lines += ["", "Right now:"] + map_text(views.get("now"), who)
    return "\n".join(lines)


# ---------- judge instructions ----------

def prompts() -> dict[str, dict]:
    PROMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not (PROMPTS_ROOT / "v1.json").exists():
        (PROMPTS_ROOT / "v1.json").write_text(json.dumps(V1, indent=2) + "\n")
    found = {}
    for path in PROMPTS_ROOT.glob("*.json"):
        data = json.loads(path.read_text())
        found[data["name"]] = data
    return dict(sorted(found.items(), key=lambda kv: _version_key(kv[0])))


def _version_key(name: str):
    m = re.fullmatch(r"v(\d+)", name)
    return (0, int(m.group(1))) if m else (1, name)


def save_prompt(parent: str, world: str, task: str, definitions: dict, note: str = "") -> dict:
    existing = prompts()
    numbers = [int(n[1:]) for n in existing if re.fullmatch(r"v\d+", n)]
    name = f"v{max(numbers, default=0) + 1}"
    if set(definitions) != set(LABELS) or not all(str(definitions[l]).strip() for l in LABELS):
        raise ValueError("every label needs a definition")
    data = {"name": name, "parent": parent, "note": note.strip(), "world": world.strip(),
            "task": task.strip(), "definitions": {l: str(definitions[l]).strip() for l in LABELS}}
    (PROMPTS_ROOT / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def system_prompt(prompt: dict, card_id: str) -> str:
    order = list(LABELS)
    random.Random(card_id).shuffle(order)  # the listed order varies by card
    options = "\n".join(f"- {label}: {prompt['definitions'][label]}" for label in order)
    return f"{prompt['world']}\n\n{prompt['task']} Use exactly one label:\n{options}\n\n{REPLY_FORMAT}"


# ---------- human ratings ----------

def normalize(label) -> str | None:
    key = str(label or "").strip().lower().replace("'", "").replace("’", "").replace(" ", "_")
    return {"b": "belongs", "d": "doesnt_matter", "n": "noise", "belongs": "belongs",
            "doesnt_matter": "doesnt_matter", "noise": "noise"}.get(key)


def _read_ratings(path: Path, key: str = "card_id") -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def ratings(card_set: str, stale: list | None = None) -> dict[str, dict[str, dict]]:
    """rater -> card_id -> {label, reason}. A grade stays with its card, the
    same memory at the same moment, for good. One given on an earlier wording
    still counts. It is marked reworded and listed in stale."""
    current = {c["card_id"]: text_hash(c) for c in load_cards(card_set)}
    out = {}
    for path in sorted(set_dir(card_set).glob("ratings_*.csv")):
        rows = {}
        for card_id, row in _read_ratings(path).items():
            if not normalize(row.get("label")) or card_id not in current:
                continue
            reworded = current[card_id] != row.get("text_hash")
            if reworded and stale is not None:
                stale.append(card_id)
            rows[card_id] = {"label": normalize(row["label"]), "reason": row.get("reason", ""),
                             "saved_at": float(row.get("saved_at") or 0), "reworded": reworded}
        out[path.stem.removeprefix("ratings_")] = rows
    return out


def save_rating(card_set: str, rater: str, card_id: str, label: str | None, reason: str = "") -> None:
    rater = safe_name(rater).lower()
    if rater not in RATERS:
        raise ValueError("Type your first name: Andre, Ben or Ryan.")
    card = next((c for c in load_cards(card_set) if c["card_id"] == card_id), None)
    if card is None:
        raise ValueError(f"no card {card_id} in {card_set}")
    path = set_dir(card_set) / f"ratings_{rater}.csv"
    rows = _read_ratings(path) if path.exists() else {}
    label = normalize(label) if label else None
    if label:
        rows[card_id] = {"card_id": card_id, "label": label, "reason": reason.strip(),
                         "text_hash": text_hash(card), "saved_at": round(time.time(), 1)}
    else:
        rows.pop(card_id, None)
    _write_ratings(path, rows)


def _write_ratings(path: Path, rows: dict[str, dict]) -> None:
    _write_rows(path, ["card_id", "label", "reason", "text_hash", "saved_at"], rows)


def _write_rows(path: Path, fields: list[str], rows: dict[str, dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())
    os.replace(tmp, path)


def carry_ratings(old_set: str, new_set: str) -> int:
    """Copy each person's grades for cards that are in both sets, which means
    the same memory at the same moment, so the new set's report has them.
    Grades already in the new set stay."""
    current = {c["card_id"] for c in load_cards(new_set)}
    moved = 0
    for path in sorted(set_dir(old_set).glob("ratings_*.csv")):
        target = set_dir(new_set) / path.name
        rows = _read_ratings(target) if target.exists() else {}
        before = len(rows)
        for card_id, row in _read_ratings(path).items():
            if card_id in current and card_id not in rows and normalize(row.get("label")):
                rows[card_id] = row
        if len(rows) > before:
            _write_ratings(target, rows)
            moved += len(rows) - before
    return moved


def gradable(card_set: str) -> list[dict]:
    """The cards people grade. Once a set marks its worlds, the newer engine's only."""
    mark = engines(card_set)
    return [c for c in load_cards(card_set) if mark.get(c.get("sealed", {}).get("run"), "newer") == "newer"]


# ---------- the grading list ----------
# People grade one fixed list. It has one item per scene, an item is asked of a
# person once, and nothing they graded before comes back in other words.

def _moment(card: dict) -> tuple:
    """One person's one turn. Cards of the same moment differ only in the memory."""
    s = card.get("sealed", {})
    return (s.get("run"), s.get("agent_tag"), s.get("tick")) if s.get("tick") is not None else (card["card_id"],)


def scenes(cards: list[dict]) -> dict[tuple, str]:
    """moment -> scene. People standing together live the same turns, so their
    cards read almost alike. Moments in one world, a few turns apart, that
    share most of their words are one scene, and a scene is graded once."""
    words = {}
    for c in cards:
        if _moment(c) not in words:
            who = c.get("sealed", {}).get("agent_name") or "This person"
            text = "\n".join(_story(c.get("history", []), who) + _now(c.get("scene", []), who))
            words[_moment(c)] = set(re.findall(r"\w+", text.lower())) - {who.lower()}
    parent = {m: m for m in words}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    timed = sorted(m for m in words if len(m) == 3)
    for i, a in enumerate(timed):
        for b in timed[i + 1:]:
            if a[0] == b[0] and abs(a[2] - b[2]) <= SCENE_TURNS and \
                    len(words[a] & words[b]) >= SCENE_ALIKE * len(words[a] | words[b]):
                parent[find(b)] = find(a)
    first = defaultdict(list)
    for m in words:
        first[find(m)].append(m)
    return {m: "scene-" + hashlib.sha1(repr(min(group, key=repr)).encode()).hexdigest()[:12]
            for group in first.values() for m in group}


def pairs(card_set: str) -> list[dict]:
    """Each graded memory that links brought to mind, next to the memory it
    pushed out at that moment. by lists the formulas that made exactly that swap."""
    path = set_dir(card_set) / "moments.json"
    picks = {(m["run"], m["agent_tag"], m["tick"]): m["picks"]
             for m in (json.loads(path.read_text())["moments"] if path.exists() else [])}
    cards = gradable(card_set)
    scene = scenes(cards)
    moments = defaultdict(list)
    for c in cards:
        moments[_moment(c)].append(c)
    out = []
    for key, group in moments.items():
        for linked in (c for c in group if c["sealed"].get("kind") == "rescued"):
            best = None
            for pushed in (c for c in group if c["sealed"].get("kind") == "displaced"):
                a, b = linked["sealed"].get("memory_id"), pushed["sealed"].get("memory_id")
                by = [f for f, p in picks[key].items() if a in p["selected"] and a not in p["solo"]
                      and b in p["solo"] and b not in p["selected"]] if key in picks else [linked["sealed"].get("condition") or "links"]
                if by and (best is None or len(by) > len(best["by"])):
                    best = {"linked": linked, "pushed": pushed, "by": by}
            if best:
                a, b = (set(re.findall(r"\w+", plain(best[side]["memory"]["text"]).lower())) for side in ("linked", "pushed"))
                ids = f"{best['linked']['card_id']}|{best['pushed']['card_id']}"
                out.append({**best, "pair_id": "pair-" + hashlib.sha1(ids.encode()).hexdigest()[:12],
                            "copy": len(a & b) >= NEAR_COPY * max(1, len(a | b)), "scene": scene[key]})
    return out


def _on_card(card: dict) -> bool:
    """The memory is one of the last turns the card already shows."""
    memory = card.get("memory", {})
    return any(h["ticks_ago"] == memory.get("ticks_ago") and h["text"] == memory.get("text") for h in card.get("history", []))


def tasks(card_set: str) -> list[dict]:
    """The list everyone grades, one item per scene. A scene with a swap gives
    a pair card: the swap the most formulas made, near copies left out. A
    scene without one gives a single card for a linked memory that pushed
    nothing out, unless the card already shows that memory as a recent turn.
    Adding worlds to a set adds items. It never changes or removes one."""
    cards = gradable(card_set)
    scene = scenes(cards)
    swaps = defaultdict(list)
    for p in pairs(card_set):
        swaps[p["scene"]].append(p)
    out = []
    for sid, found in swaps.items():
        real = [p for p in found if not p["copy"]]
        if real:
            p = min(real, key=lambda p: (-len(p["by"]), p["pair_id"]))
            out.append({"id": p["pair_id"], "scene": sid, "pair": p})
    alone = defaultdict(list)
    for c in cards:
        if scene[_moment(c)] not in swaps and c.get("sealed", {}).get("kind", "rescued") == "rescued" and not _on_card(c):
            alone[scene[_moment(c)]].append(c)
    for sid, found in alone.items():
        c = min(found, key=lambda c: (-(c.get("memory", {}).get("ticks_ago") or 0), c["card_id"]))
        out.append({"id": c["card_id"], "scene": sid, "card": c})
    return sorted(out, key=lambda t: t["id"])


def graded_before(rater: str) -> tuple[set, int]:
    """Every moment this person has graded in any set, and how many memories
    that is. Read from the grade files each time, so nothing has to be carried
    over for a grade to count."""
    rater = safe_name(rater).lower() if rater else ""
    moments, cards = set(), set()
    for card_set in card_sets() if rater else []:
        folder = set_dir(card_set)
        where = {c["card_id"]: _moment(c) for c in map(json.loads, (folder / "cards.jsonl").read_text().splitlines())}
        if (folder / f"ratings_{rater}.csv").exists():
            for card_id, row in _read_ratings(folder / f"ratings_{rater}.csv").items():
                if card_id in where and normalize(row.get("label")):
                    moments.add(where[card_id])
                    cards.add(card_id)
        if (folder / f"pairs_{rater}.csv").exists():
            answered = _read_ratings(folder / f"pairs_{rater}.csv", "pair_id")
            for p in pairs(card_set):
                if p["pair_id"] in answered:
                    moments.add(_moment(p["linked"]))
                    cards.update((p["linked"]["card_id"], p["pushed"]["card_id"]))
    return moments, len(cards)


def queue(card_set: str, rater: str) -> list[dict]:
    """What one person works through, as the page reads it: the items they
    finished, oldest first, then the ones left, pair cards first. A scene they
    graded before, on any card of any set, is not asked again."""
    rater = safe_name(rater).lower() if rater else ""
    scene = scenes(gradable(card_set))
    seen = {scene[m] for m in graded_before(rater)[0] if m in scene}
    single, double = ratings(card_set).get(rater, {}), pair_ratings(card_set).get(rater, {})
    done, todo = [], []
    for task in tasks(card_set):
        card = task["pair"]["linked"] if "pair" in task else task["card"]
        sealed = card.get("sealed", {})
        item = {"card_id": task["id"], "who": sealed.get("agent_name", ""), "tick": sealed.get("tick"), "mine": None}
        if "pair" in task:
            sides = _sides(rater, task["id"])
            answer = double.get(task["id"])
            item.update(pair=True, text=pair_text(task["pair"], sides), views={"now": (card.get("views") or {}).get("now")})
            if answer:
                letter = {sides[0]: "a", sides[1]: "b"}.get(answer["answer"], answer["answer"])
                item["mine"] = {"label": letter, "reason": answer["reason"]}
        else:
            answer = single.get(task["id"])
            item.update(text=visible_text(card), views=card.get("views"), ago=card.get("memory", {}).get("ticks_ago"), mine=answer)
        if answer:
            done.append((answer["saved_at"], item))
        elif task["scene"] not in seen:
            todo.append(("pair" not in task, hashlib.sha1(f"{rater}|{task['id']}".encode()).hexdigest(), item))
    return [i for _, i in sorted(done, key=lambda d: d[0])] + [i for *_, i in sorted(todo, key=lambda t: t[:2])]


# ---------- pair cards ----------

def pair_text(pair: dict, order=("linked", "pushed")) -> str:
    """One moment and its two memories. No ages and no marks in the story, so
    only the memories themselves tell A from B."""
    card = pair["linked"]
    who = card.get("sealed", {}).get("agent_name") or "This person"
    lines = [f"This is {who}.", "", "What happened:"] + _story(card.get("history", []), who)
    lines += ["", "Right now:"] + [f"  {x}" for x in _now(card.get("scene", []), who)]
    for letter, side in zip("AB", order):
        memory = _turn(pair[side].get("memory", {}).get("text", ""), who)
        lines += ["", f"Memory {letter}:"] + [f"  {x}" for x in memory or ["(nothing)"]]
    return "\n".join(lines)


def pair_hash(pair: dict) -> str:
    """Covers the text and the map of right now, like a card's fingerprint."""
    who = pair["linked"].get("sealed", {}).get("agent_name") or "This person"
    now = (pair["linked"].get("views") or {}).get("now")
    return hashlib.sha1("\n".join([pair_text(pair)] + map_text(now, who)).encode()).hexdigest()[:12]


def _sides(rater: str, pair_id: str) -> tuple[str, str]:
    """Which memory this person sees as A. It never changes, so Back shows the same card."""
    flip = int(hashlib.sha1(f"{rater}|{pair_id}".encode()).hexdigest(), 16) % 2
    return ("pushed", "linked") if flip else ("linked", "pushed")


def pair_ratings(card_set: str) -> dict[str, dict[str, dict]]:
    """rater -> pair_id -> {answer, reason}. answer is linked, pushed, both or
    neither. Like a grade, an answer stays with its pair for good."""
    known = {p["pair_id"] for p in pairs(card_set)}
    out = {}
    for path in sorted(set_dir(card_set).glob("pairs_*.csv")):
        out[path.stem.removeprefix("pairs_")] = {
            k: {"answer": r["answer"], "reason": r.get("reason", ""), "saved_at": float(r.get("saved_at") or 0)}
            for k, r in _read_ratings(path, "pair_id").items() if k in known}
    return out


def save_pair(card_set: str, rater: str, pair_id: str, answer: str | None, reason: str = "") -> None:
    """answer is what the person pressed: a, b, both or neither."""
    rater = safe_name(rater).lower()
    if rater not in RATERS:
        raise ValueError("Type your first name: Andre, Ben or Ryan.")
    pair = next((p for p in pairs(card_set) if p["pair_id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"no pair {pair_id} in {card_set}")
    if answer and answer not in PAIR_ANSWERS:
        raise ValueError("Pick an answer.")
    path = set_dir(card_set) / f"pairs_{rater}.csv"
    rows = _read_ratings(path, "pair_id") if path.exists() else {}
    if answer:
        a, b = _sides(rater, pair_id)
        rows[pair_id] = {"pair_id": pair_id, "answer": {"a": a, "b": b}.get(answer, answer), "reason": reason.strip(),
                         "text_hash": pair_hash(pair), "saved_at": round(time.time(), 1)}
    else:
        rows.pop(pair_id, None)
    _write_rows(path, ["pair_id", "answer", "reason", "text_hash", "saved_at"], rows)


def _implied(linked: str, pushed: str) -> str:
    """What two single grades say about a pair: the better grade wins."""
    a, b = LABELS.index(linked), LABELS.index(pushed)  # belongs is best
    return "linked" if a < b else "pushed" if b < a else "neither" if linked == "noise" else "both"


def head_to_head(card_set: str) -> list[dict]:
    """One row per scene with a swap: what people said about it. A person's
    answer to the scene's pair card counts first. Failing that, a swap in the
    scene whose two memories they graded one at a time before pair cards: the
    better grade wins. how says where the result came from: asked, grades, or
    copy when every swap in the scene is a near copy."""
    rated, asked = ratings(card_set), pair_ratings(card_set)
    mark = engines(card_set)
    task = {t["scene"]: t["id"] for t in tasks(card_set) if "pair" in t}
    swaps = defaultdict(list)
    for p in pairs(card_set):
        swaps[p["scene"]].append(p)
    out = []
    for sid, found in swaps.items():
        found.sort(key=lambda p: (p["pair_id"] != task.get(sid), p["pair_id"]))
        votes, direct = Counter(), False
        for rater in set(rated) | set(asked):
            mine, grades = asked.get(rater, {}), rated.get(rater, {})
            if answer := next((mine[p["pair_id"]]["answer"] for p in found if p["pair_id"] in mine), None):
                votes[answer] += 1
                direct = True
            elif both := next((p for p in found if not p["copy"] and p["linked"]["card_id"] in grades
                               and p["pushed"]["card_id"] in grades), None):
                votes[_implied(grades[both["linked"]["card_id"]]["label"], grades[both["pushed"]["card_id"]]["label"])] += 1
        top = votes.most_common(1)
        result = top[0][0] if top and top[0][1] > sum(votes.values()) / 2 else None if sid in task else "same"
        out.append({"scene": sid, "by": found[0]["by"], "engine": mark.get(found[0]["linked"]["sealed"].get("run")),
                    "result": result, "how": "asked" if direct else "grades" if top else "copy" if result else None})
    return out


def human_majority(rated: dict[str, dict[str, dict]]) -> dict[str, str]:
    votes = defaultdict(Counter)
    for rows in rated.values():
        for card_id, row in rows.items():
            votes[card_id][row["label"]] += 1
    majority = {}
    for card_id, counter in votes.items():
        label, count = counter.most_common(1)[0]
        if count > sum(counter.values()) / 2:
            majority[card_id] = label
    return majority


def compare(card_set: str) -> dict:
    """For each moment in the set, the memories each formula brought to mind,
    and what people said about the ones that were graded. A set has this only
    when it was built from several formulas (rescued_cards.py --variant)."""
    path = set_dir(card_set) / "moments.json"
    if not path.exists():
        return {"formulas": [], "moments": []}
    data = json.loads(path.read_text())
    majority = human_majority(ratings(card_set))
    mark = engines(card_set)
    card_of = {(c["sealed"]["run"], c["sealed"]["agent_tag"], c["sealed"]["tick"], c["sealed"]["memory_id"]): c["card_id"]
               for c in load_cards(card_set)}
    moments = []
    for m in data["moments"]:
        who, picks = m["agent_name"], m["picks"]
        rows = []
        for mem_id, mem in m["memories"].items():
            # yes: came to mind. links: only because of links. out: a linked memory took its place.
            by = {f: "yes" if mem_id in p["solo"] else "links" for f, p in picks.items() if mem_id in p["selected"]}
            by.update({f: "out" for f, p in picks.items() if mem_id in p["solo"] and mem_id not in p["selected"]})
            card = card_of.get((m["run"], m["agent_tag"], m["tick"], mem_id))
            rows.append({"ago": mem["ticks_ago"], "lines": _turn(mem["text"], who), "by": by,
                         "card": bool(card), "grade": majority.get(card)})
        rows.sort(key=lambda r: (-len(r["by"]), r["ago"]))
        moments.append({"run": m["run"], "engine": mark.get(m["run"]), "who": who, "tick": m["tick"],
                        "now": _now(m["scene"], who), "rows": rows})
    return {"formulas": data["formulas"], "moments": moments, "pairs": head_to_head(card_set)}


# ---------- judge runs ----------

def _load_env():
    env_file = ROOT / "code/terralingua/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def ask(client, model: str, prompt: dict, card: dict) -> dict:
    kwargs = {"reasoning_effort": "low"} if model.startswith("gpt-5") and "luna" not in model else {}
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt(prompt, card["card_id"])},
                  {"role": "user", "content": judge_text(card)}],
        response_format={"type": "json_object"},
        **kwargs,
    )
    raw = reply.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        label, reason = normalize(parsed.get("label")), str(parsed.get("reason", "")).strip()
    except json.JSONDecodeError:
        label, reason = None, raw
    usage = reply.usage
    return {"card_id": card["card_id"], "label": label or "unparsed", "reason": reason,
            "text_hash": text_hash(card), "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0}


def run_path(card_set: str, prompt_name: str, model: str) -> Path:
    return set_dir(card_set) / "judge_runs" / f"{safe_name(prompt_name)}__{safe_name(model)}.jsonl"


def run_judge(card_set: str, prompt_name: str, model: str, card_ids=None, progress=None) -> Path:
    if model not in JUDGE_MODELS:
        raise ValueError(f"unknown judge model {model}")
    prompt = prompts().get(prompt_name)
    if prompt is None:
        raise ValueError(f"no judge instructions named {prompt_name}")
    _load_env()
    from openai import OpenAI
    client = OpenAI()
    cards = [c for c in load_cards(card_set) if card_ids is None or c["card_id"] in card_ids]
    out = run_path(card_set, prompt_name, model)
    out.parent.mkdir(exist_ok=True)
    previous = {}
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            previous[row["card_id"]] = row
    todo = [c for c in cards if previous.get(c["card_id"], {}).get("label") in (None, "unparsed")
            or previous[c["card_id"]].get("text_hash") != text_hash(c)]
    done = 0
    if progress:
        progress(done, len(todo))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(lambda card: ask(client, model, prompt, card), todo):
            previous[row["card_id"]] = row
            done += 1
            if progress:
                progress(done, len(todo))
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in previous.values()))
    os.replace(tmp, out)
    return out


def judge_runs(card_set: str) -> dict[str, dict[str, dict]]:
    """'<prompt> / <model>' -> card_id -> row"""
    runs = {}
    folder = set_dir(card_set) / "judge_runs"
    for path in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        prompt_name, _, model = path.stem.partition("__")
        rows = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row["card_id"]] = row
        runs[f"{prompt_name} / {model}"] = rows
    return runs


# ---------- agreement ----------

def cohen_kappa(a: dict, b: dict):
    keys = [k for k in a if k in b]
    if not keys:
        return None, 0, 0
    n = len(keys)
    agree = sum(a[k] == b[k] for k in keys)
    ca, cb = Counter(a[k] for k in keys), Counter(b[k] for k in keys)
    expected = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    kappa = None if expected >= 1 else (agree / n - expected) / (1 - expected)
    return kappa, agree, n


def krippendorff_alpha(raters: dict[str, dict[str, str]]):
    """Nominal alpha over any number of raters, missing ratings allowed."""
    units = defaultdict(list)
    for labels in raters.values():
        for card_id, label in labels.items():
            units[card_id].append(label)
    coincidence = Counter()
    for values in units.values():
        m = len(values)
        if m < 2:
            continue
        for i, x in enumerate(values):
            for j, y in enumerate(values):
                if i != j:
                    coincidence[(x, y)] += 1 / (m - 1)
    totals = Counter()
    for (x, _), w in coincidence.items():
        totals[x] += w
    n = sum(totals.values())
    if n <= 1:
        return None
    observed = sum(w for (x, y), w in coincidence.items() if x != y)
    expected = sum(totals[x] * totals[y] for x in totals for y in totals if x != y) / (n - 1)
    return None if expected == 0 else 1 - observed / expected


def report(card_set: str) -> dict:
    cards = load_cards(card_set)
    todo = tasks(card_set)
    split = {c["card_id"]: split_of(c) for c in cards}
    stale = []
    rated = ratings(card_set, stale)
    majority = human_majority(rated)
    humans = {r: {k: v["label"] for k, v in rows.items()} for r, rows in rated.items()}

    human_pairs = []
    names = list(humans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            kappa, agree, n = cohen_kappa(humans[a], humans[b])
            human_pairs.append({"a": a, "b": b, "kappa": kappa, "agree": agree, "n": n})

    runs = []
    for run_name, rows in judge_runs(card_set).items():
        labels = {k: r["label"] for k, r in rows.items() if r["label"] in LABELS}
        entry = {"run": run_name, "judged": len(labels), "mix": dict(Counter(labels.values())),
                 "tokens": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in rows.values())}
        for group in ("tuning", "held_out"):
            truth = {k: v for k, v in majority.items() if split.get(k) == group}
            kappa, agree, n = cohen_kappa(labels, truth)
            entry[group] = {"kappa": kappa, "agree": agree, "n": n}
        held = entry["held_out"]
        if held["n"] < GATE_MIN_CARDS:
            entry["gate"] = f"not yet: label {GATE_MIN_CARDS - held['n']} more test cards"
        else:
            entry["gate"] = "pass" if (held["kappa"] or 0) >= GATE_KAPPA else "fail"
        runs.append(entry)

    mark = engines(card_set)
    by_engine = {}
    for c in cards:
        if engine := mark.get(c.get("sealed", {}).get("run")):
            entry = by_engine.setdefault(engine, {"cards": 0, "graded": 0, "mix": Counter()})
            entry["cards"] += 1
            if label := majority.get(c["card_id"]):
                entry["graded"] += 1
                entry["mix"][label] += 1

    return {
        "cards": len(cards),
        "engines": by_engine,
        "split": dict(Counter(split.values())),
        "raters": {r: len(rows) for r, rows in rated.items()},
        "reworded": len(stale),
        "list": {"pairs": sum("pair" in t for t in todo), "singles": sum("card" in t for t in todo),
                 "raters": {r: len(rows) for r, rows in pair_ratings(card_set).items()}},
        "majority": {"n": len(majority), **{g: sum(1 for k in majority if split.get(k) == g) for g in ("tuning", "held_out")}},
        "human_pairs": human_pairs,
        "human_alpha": krippendorff_alpha(humans) if len(humans) >= 2 else None,
        "runs": runs,
        "gate_rule": (f"Pass means the judge agrees with people on test cards it was never tuned on: "
                      f"a score of {GATE_KAPPA} or more on at least {GATE_MIN_CARDS} test cards."),
    }


def disagreements(card_set: str, run_name: str) -> list[dict]:
    """Tuning cards only, so the held-out check stays untouched."""
    cards = {c["card_id"]: c for c in load_cards(card_set)}
    rated = ratings(card_set)
    majority = human_majority(rated)
    rows = judge_runs(card_set).get(run_name, {})
    out = []
    for card_id, truth in majority.items():
        row = rows.get(card_id)
        card = cards.get(card_id)
        if not row or not card or split_of(card) != "tuning" or row["label"] == truth:
            continue
        out.append({
            "card_id": card_id,
            "text": visible_text(card),
            "memory": plain(card["memory"]["text"]),
            "people": [{"rater": r, "label": rows_[card_id]["label"], "reason": rows_[card_id]["reason"]}
                       for r, rows_ in rated.items() if card_id in rows_],
            "majority": truth,
            "judge": row["label"],
            "judge_reason": row["reason"],
        })
    return out
