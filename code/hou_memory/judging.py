"""Shared pieces of the believability judge: card sets, versioned judge
instructions, human ratings, judge runs, and agreement statistics.

Used by the Rescued set page in God Mode (webapp.py, /rescued). Files:

  output/analysis/rescued/<run>/cards.jsonl         the rescued set, from scripts/rescued_cards.py
  output/analysis/rescued/<run>/ratings_<rater>.csv human labels
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
ANCHOR_SET = "anchor_v1"  # the set classmates grade
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
    cards.sort(key=lambda c: c["card_id"])
    random.Random(card_set).shuffle(cards)
    return cards


def split_of(card: dict) -> str:
    """Sampled sets store their split. Otherwise hash the group, so near copies
    of one moment or memory never sit on both sides."""
    sealed = card.get("sealed", {})
    if sealed.get("split"):
        return sealed["split"]
    digits = re.sub(r"[^0-9a-f]", "", (sealed.get("group") or card["card_id"]).removeprefix("group-"))[:8] or "0"
    return "held_out" if int(digits, 16) % 100 < HELD_OUT_SHARE * 100 else "tuning"


def text_hash(card: dict) -> str:
    return hashlib.sha1(visible_text(card).encode()).hexdigest()[:12]


def _names(items: list[str]) -> str:
    items = [i.strip() for i in items if i.strip()]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


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
        return f'Wrote a note called "{p.get("name")}".'
    if name == "pickup_artifact":
        return f'Picked up the note "{p.get("name")}".'
    if name == "drop_artifact":
        return f'Dropped the note "{p.get("name")}".'
    if name == "give_artifact":
        return f'Gave the note "{p.get("artifact_name")}" to {p.get("target_agent")}.'
    if name == "reproduce":
        return f"Had a child named {p.get('name')}."
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
        said = " ".join(v.rstrip(".") + "." for v in values)
        said = re.sub(r"Artifact (.+?) content: ", r"Note \1 says: ", said)
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
    text = re.sub(r"Artifacts here: ([^.]+)\.", lambda m: f"Saw a note: {_names(m.group(1).split(','))}.", text)
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
    if name == "create_artifact":
        return f'{who} writes a note called "{p.get("name")}".'
    if name == "pickup_artifact":
        return f'{who} picks up the note "{p.get("name")}".'
    if name == "drop_artifact":
        return f'{who} drops the note "{p.get("name")}".'
    if name == "give_artifact":
        return f'{who} gives the note "{p.get("artifact_name")}" to {p.get("target_agent")}.'
    if name == "reproduce":
        return f"{who} has a child named {p.get('name')}."
    return f"{who} does {name.replace('_', ' ')}."


def _world(found: str, who: str) -> str:
    text = plain(f"Feedback from the preceding action: {found}.").removeprefix("The world said: ")
    text = re.sub(r"\b[Yy]our\b", f"{who}'s", text)
    return re.sub(r"\b[Yy]ou\b", who, text)


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
        setting.append(f"{who} sees {'a note' if len(notes) == 1 else 'notes'}: {_names(notes)}.")
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
    seen = set()

    def once(line):
        m = re.fullmatch(r'([\w.-]+): "(.*)"', line, re.S)
        if not m:
            return line
        if m.groups() in seen and len(m.group(2).split()) > 3:
            return f"{m.group(1)}: (says the same thing again)"
        seen.add(m.groups())
        return line

    out = []
    for g in groups:
        if g["marked"]:
            head, lines = f"{_ago(g['first'])} [this is the memory below]", g["lines"]
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
                out.append(f"(Notes nearby: {_names(m.group(1).split(','))}.)")
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


def _read_ratings(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["card_id"]: r for r in csv.DictReader(f)}


def ratings(card_set: str, stale: list | None = None) -> dict[str, dict[str, dict]]:
    """rater -> card_id -> {label, reason}. A grade counts only for the exact
    text the person read. Grades on older wording go into stale."""
    current = {c["card_id"]: text_hash(c) for c in load_cards(card_set)}
    out = {}
    for path in sorted(set_dir(card_set).glob("ratings_*.csv")):
        rows = {}
        for card_id, row in _read_ratings(path).items():
            if not normalize(row.get("label")):
                continue
            if current.get(card_id) != row.get("text_hash"):
                if stale is not None:
                    stale.append(card_id)
                continue
            rows[card_id] = {"label": normalize(row["label"]), "reason": row.get("reason", ""),
                             "saved_at": float(row.get("saved_at") or 0)}
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
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["card_id", "label", "reason", "text_hash", "saved_at"])
        writer.writeheader()
        writer.writerows(rows.values())
    os.replace(tmp, path)


def for_rater(card_set: str, rater: str) -> list[tuple[dict, dict | None]]:
    """(card, this person's grade), their finished cards first. Then the cards
    with the fewest grades, in an order of their own, so grades spread evenly."""
    rater = safe_name(rater).lower() if rater else ""
    rated = ratings(card_set)
    counts = Counter(card_id for rows in rated.values() for card_id in rows)
    mine = rated.get(rater, {})

    def rank(card):
        cid = card["card_id"]
        if cid in mine:
            return (0, mine[cid]["saved_at"], "")
        return (1, counts[cid], hashlib.sha1(f"{rater}|{cid}".encode()).hexdigest())

    return [(c, mine.get(c["card_id"])) for c in sorted(load_cards(card_set), key=rank)]


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
                  {"role": "user", "content": visible_text(card)}],
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
    split = {c["card_id"]: split_of(c) for c in cards}
    stale = []
    rated = ratings(card_set, stale)
    majority = human_majority(rated)
    humans = {r: {k: v["label"] for k, v in rows.items()} for r, rows in rated.items()}

    pairs = []
    names = list(humans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            kappa, agree, n = cohen_kappa(humans[a], humans[b])
            pairs.append({"a": a, "b": b, "kappa": kappa, "agree": agree, "n": n})

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

    return {
        "cards": len(cards),
        "split": dict(Counter(split.values())),
        "raters": {r: len(rows) for r, rows in rated.items()},
        "stale": len(stale),
        "majority": {"n": len(majority), **{g: sum(1 for k in majority if split.get(k) == g) for g in ("tuning", "held_out")}},
        "human_pairs": pairs,
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
