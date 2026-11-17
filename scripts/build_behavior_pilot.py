"""Build blinded D1–D5 pilot packets from a reviewed evidence manifest."""

import argparse
import hashlib
import hmac
import json
import ntpath
import os
import random
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from posixpath import normpath


COMPARISONS = {
    "matched_checkpoint": "Matched checkpoint (reviewer-attested)",
    "separate_runs": "Exploratory separate runs",
}
SCHEMA_VERSION = 2
GROUPED_SCHEMA_VERSION = 3
RUBRIC_REVISION = "D1-D5-pilot-v2"
RATING_STATES = ("NEI", "MISSING")
YES_NO = ("Yes", "No")
BRANCH_MEASURES = {
    "d1_evaluable_claims": ("D1 Evaluable past-event claim count", None),
    "d1_contradictions": ("D1 Contradiction count", None),
    "d1_any_contradiction": ("D1 Any contradiction", YES_NO),
    "d2_relevant_reference": ("D2 Relevant past reference", YES_NO),
    "d2_implicit_use": ("D2 Implicit behavioral use", YES_NO),
    "d3_intrusion": ("D3 Irrelevant or stale intrusion", YES_NO),
    "d3_fabrication": ("D3 Fabricated past claim", YES_NO),
    "d3_any_error": ("D3 Either error", YES_NO),
    "d4_plausibility": ("D4 Action plausibility", (1, 2, 3, 4, 5)),
}
D5_CHOICES = ("A", "B", "Tie")
EVENT_SOURCES = {
    "give": {"world_transfer"},
    "take": {"world_transfer"},
    "message": {"received_message", "world_broadcast"},
    "energy_rise": {"energy_observations"},
}
OUTCOME_OFFSETS = {
    "world_event": {0},
    "probe_transition": {0},
    "upstream_feedback": {1},
    "fork_receipt": {0, 1},
}
OUTCOME_STATES = {"confirmed", "failed", "unknown"}
NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
SOURCE_REFERENCE_FIELDS = {"source_id", "decision_source_id", "review_source", "restore_source"}
PATH_TOKEN = re.compile(r"""[^\s"'<>#|,;=()[\]{}]+""")


@dataclass(frozen=True)
class SelfPairControl:
    """A presentation of one branch of an already validated development pair."""

    selected_branch_index: int


def require(valid, message):
    if not valid:
        raise ValueError(message)


def field(record, key, kind):
    value = record.get(key)
    require(type(value) is kind, f"{key} must be {kind.__name__}")
    return value


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def string(record, key):
    value = field(record, key, str)
    require(nonblank(value), f"{key} must be nonblank")
    return value


def tick(record, key="tick"):
    value = field(record, key, int)
    require(value >= 0, f"{key} must be nonnegative")
    return value


def source_identity(value):
    """Compare artifact paths lexically, preserving the record locator."""
    artifact, separator, locator = value.partition("#")
    path = artifact.strip().replace("\\", "/")
    require(bool(path), "source reference must name an artifact")
    if re.match(r"^[A-Za-z]:", path) or path.startswith("//"):
        path = ntpath.normcase(ntpath.normpath(path)).replace("\\", "/")
    else:
        path = normpath(path)
    # Legacy line citations accept either separator between the line and field.
    locator = re.sub(r"^(\d+)\\", r"\1/", locator)
    return path, separator + locator


def evidence_identity(record):
    """Normalize known source fields for registry comparisons only."""
    return {
        name: source_identity(value) if name in SOURCE_REFERENCE_FIELDS and isinstance(value, str)
        else evidence_identity(value) if isinstance(value, dict)
        else value
        for name, value in record.items()
    }


def indexed_rows(rows, key, identity, protected):
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f"{key} evidence rows must be objects")
        row_identity = (string(row, "run_id"), string(row, "agent_id"))
        require(row_identity == identity, f"invalid {key} join (run_id, agent_id)")
        row_tick = tick(row)
        value = row_tick if key == "tick" else string(row, key)
        require(value not in indexed, f"duplicate evidence {key} {value}")
        indexed[value] = row
        protected.add(string(row, "source_id"))
    return indexed


def remember(registry, key, fields, description):
    """Require agreement on supplied fields, including partial observation rows."""
    fields = evidence_identity(fields)
    previous = registry.setdefault(key, {})
    require(all(name not in previous or canonical_json(previous[name]) == canonical_json(value)
                for name, value in fields.items()), f"conflicting {description} across evidence")
    previous.update(fields)


def validate_outcome(outcome, decision):
    decision_tick = decision["tick"]
    require(source_identity(string(outcome, "decision_source_id")) == source_identity(decision["source_id"]),
            "invalid outcome join to the chosen decision source")
    source_kind = string(outcome, "source_kind")
    require(source_kind in OUTCOME_OFFSETS, "invalid outcome source kind")
    recorded_tick = tick(outcome, "recorded_tick")
    require(recorded_tick - decision_tick in OUTCOME_OFFSETS[source_kind],
            "invalid outcome feedback tick for source kind")
    status = string(outcome, "status")
    require(status in OUTCOME_STATES, "invalid outcome status")
    execution_tick = outcome.get("execution_tick")
    latest_execution = recorded_tick if source_kind == "fork_receipt" else decision_tick
    require("execution_tick" in outcome
            and ((type(execution_tick) is int and decision_tick <= execution_tick <= latest_execution)
                 or (execution_tick is None and status == "unknown")),
            "invalid outcome execution tick")
    string(outcome, "feedback")
    if source_kind == "fork_receipt":
        selection = string(outcome, "selection_id")
        require(selection == string(decision, "selection_id"), "invalid fork receipt selection join")
        effect = string(outcome, "effect_status")
        require(effect in {"confirmed", "unknown", "not_observed"}
                and status == ("confirmed" if effect == "confirmed" else "unknown"),
                "fork receipt status must reflect its reported effect")


def validate_branch(branch, protected, numbers, registry):
    require(isinstance(branch, dict), "branch must be an object")
    run, actor = (string(branch, key) for key in ("run_id", "agent_id"))
    identity = run, actor
    cue, condition = (string(branch, key) for key in ("entity_id", "condition"))
    start, last_seen = tick(branch), tick(branch, "last_seen_tick")
    require(start - last_seen - 1 >= 20, "fewer than 20 consecutive absent ticks")
    require(actor != cue, "cue entity must differ from the focal agent")
    protected.update((run, condition))

    names = field(branch, "names", dict)
    require(actor in names and cue in names, "register focal agent and cue entity in names")
    used = set()
    for entity, aliases in names.items():
        require(nonblank(entity) and isinstance(aliases, list) and all(map(nonblank, aliases)),
                "names must map stable IDs to lists of nonblank aliases")
        terms = {normalize(term) for term in [entity, *aliases]}
        require(used.isdisjoint(terms), "ambiguous alias across stable entities")
        used.update(terms)
        protected.update(term for term in terms if NUMERIC.fullmatch(term))
        remember(registry, ("aliases", run), dict.fromkeys(terms, entity), "entity aliases")

    observations, decisions, outcomes = (
        indexed_rows(field(branch, key, list), "tick", identity, protected)
        for key in ("observations", "decisions", "outcomes")
    )
    # Count and check the actual rows. An omitted row is never an absent tick.
    require(len(observations) == start - last_seen + 2
            and min(observations, default=-1) == last_seen
            and max(observations, default=-1) == start + 1,
            "missing or out-of-window observation (need last seen through T+1)")
    for observed_tick, observation in observations.items():
        present = field(observation, "present", list)
        require(all(nonblank(item) and item in names for item in present)
                and len(set(present)) == len(present), "invalid observation entity identities")
        if "text" in observation:
            string(observation, "text")
        remember(registry, ("observation", *identity, observed_tick),
                 {**observation, "present": sorted(present)}, "observation")
        remember(registry, ("observation_source", run, source_identity(observation["source_id"])),
                 {"agent_id": actor, "tick": observed_tick}, "observation source")
        if observed_tick in (last_seen, start):
            require(cue in present, "cue entity must be observed at last_seen_tick and T")
        elif observed_tick < start:
            require(cue not in present, "cue entity present within claimed absence gap")

    require(decisions.keys() == {start, start + 1}, "missing or invalid T/T+1 decision evidence")
    require(outcomes.keys() == {start, start + 1}, "missing or invalid T/T+1 outcome evidence")
    decision_sources = {source_identity(row["source_id"]) for row in decisions.values()}
    outcome_sources = {source_identity(row["source_id"]) for row in outcomes.values()}
    require(len(outcome_sources) == len(outcomes),
            "outcome evidence reused for multiple decisions")
    for decision_tick in (start, start + 1):
        string(observations[decision_tick], "text")
        decision = decisions[decision_tick]
        string(decision, "action")
        field(decision, "message", str)
        outcome = outcomes[decision_tick]
        require(source_identity(outcome["source_id"]) not in decision_sources,
                "outcome evidence must be independent of the chosen action record")
        if "selection_id" in decision:
            selection = string(decision, "selection_id")
            protected.add(selection)
            remember(registry, ("selection", run, selection),
                     {"agent_id": actor, "tick": decision_tick, "source_id": decision["source_id"]},
                     "receipt selection")
        validate_outcome(outcome, decision)
        for kind, row in (("decision", decision), ("outcome", outcome)):
            remember(registry, (kind, *identity, decision_tick), row, kind)
            remember(registry, ("action_source", run, source_identity(row["source_id"])),
                     {"kind": kind, "agent_id": actor, "tick": decision_tick}, "action source reuse")

    episodes = indexed_rows(field(branch, "episodes", list), "episode_id", identity, protected)
    require(1 <= len(episodes) <= 8, "dossier must contain 1–8 episodes")
    episode_sources = {source_identity(row["source_id"]) for row in episodes.values()}
    require(len(episode_sources) == len(episodes), "duplicate episode source")
    for episode_id, episode in episodes.items():
        protected.add(episode_id)
        require(tick(episode) < start and string(episode, "entity_id") == cue,
                "dossier episode must concern the cue entity and predate T")
        string(episode, "text")
        if "gold_label" in episode:
            protected.add(string(episode, "gold_label"))
        # Cue association and gold labels can vary by contrast.
        remember(registry, ("episode", *identity, episode_id),
                 {key: episode[key] for key in ("source_id", "tick", "text")}, "episode")
        remember(registry, ("episode_source", run, source_identity(episode["source_id"])),
                 {"agent_id": actor, "episode_id": episode_id}, "episode source")

    dossier_review = field(branch, "dossier_review", dict)
    require(string(dossier_review, "source_kind") in {"pre_t_checkpoint", "decision_audit"}
            and tick(dossier_review, "next_tick") == start,
            "dossier review must cite a checkpoint or audit at the pre-T decision boundary")
    require(dossier_review.get("attestations") ==
            {"available_before_t": True, "complete": True, "verbatim": True}
            and all(type(value) is bool for value in dossier_review["attestations"].values()),
            "dossier review must attest pre-T availability, completeness and verbatim text")
    dossier_source, dossier_review_source = (string(dossier_review, key) for key in ("source_id", "review_source"))
    require(source_identity(dossier_source) not in episode_sources
            and source_identity(dossier_review_source) != source_identity(dossier_source),
            "dossier review must cite distinct pre-T evidence and review records")
    protected.update((dossier_source, dossier_review_source))

    event = field(branch, "event", dict)
    indexed_rows([event], "tick", identity, protected)
    episode = episodes.get(string(event, "episode_id"))
    event_kind = string(event, "kind")
    event_source = source_identity(event["source_id"])
    require(episode is not None and event["tick"] == episode["tick"]
            and string(event, "entity_id") == cue and event_kind in EVENT_SOURCES
            and string(event, "source_kind") in EVENT_SOURCES[event_kind]
            and tick(event, "execution_tick") <= tick(event, "recorded_tick") <= event["tick"]
            and event_source not in episode_sources | decision_sources | outcome_sources
            and (event["source_kind"] not in {"world_transfer", "world_broadcast"}
                 or event_source not in {source_identity(row["source_id"]) for row in observations.values()}),
            "invalid qualifying earlier event or episode join")

    provenance = field(branch, "provenance", dict)
    require(bool(provenance), "provenance must cite run sources")
    collect_provenance(provenance, protected, numbers)
    remember(registry, ("run", run), {"condition": condition, "provenance": provenance},
             "run registration")
    remember(registry, ("dossier", *identity, cue, start),
             {"review": dossier_review, "episodes": sorted(episodes), "event": event},
             "dossier")
    return {
        "branch": branch,
        "observations": observations,
        "decisions": decisions,
        "outcomes": outcomes,
        "episodes": sorted(episodes.values(), key=lambda row: (row["tick"], row["episode_id"])),
    }


def collect_provenance(value, protected, numbers, key=None):
    if isinstance(value, dict):
        for name, item in value.items():
            collect_provenance(item, protected, numbers, name)
    elif isinstance(value, list):
        for item in value:
            collect_provenance(item, protected, numbers, key)
    elif type(value) in (int, float) or (isinstance(value, str) and NUMERIC.fullmatch(value.strip())):
        numbers.add((key, str(value).strip()))
    elif nonblank(value):
        protected.add(value)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(record, key):
    value = string(record, key)
    require(re.fullmatch(r"[0-9a-f]{64}", value), f"{key} must be a lowercase SHA-256 digest")
    return value


def reviewed_restore(branch):
    provenance = branch["provenance"]
    checkpoint = digest(provenance, "checkpoint_id")
    source = string(provenance, "restore_source")
    require(tick(provenance, "next_tick") == branch["tick"], "checkpoint must resume at the candidate tick")
    controls = field(provenance, "controls", dict)
    require(not (provenance.keys() & (controls.keys() | {"model", "backend"})),
            "put matched settings only in provenance.controls")
    for key in ("engine_revision", "actor_model", "actor_backend"):
        string(controls, key)
    for key in ("generator_settings", "actor_settings", "environment_settings"):
        require(field(controls, key, dict), f"{key} must contain reviewed settings")
    for key in ("world_state_sha256", "memory_history_sha256", "plan_sha256"):
        digest(controls, key)
    return checkpoint, source, controls


def checkpoint_context(data):
    branch = data["branch"]
    event = branch["event"]
    return (
        branch["agent_id"], branch["entity_id"], branch["tick"], branch["last_seen_tick"],
        [(tick_id, sorted(row["present"]), row.get("text"))
         for tick_id, row in sorted(data["observations"].items()) if tick_id <= branch["tick"]],
        [(row["tick"], row["episode_id"], row["text"]) for row in data["episodes"]],
        tuple(event[key] for key in ("tick", "episode_id", "kind", "execution_tick", "recorded_tick")),
    )


def validate_pair(pair, protected, numbers, registry):
    require(isinstance(pair, dict), "pair must be an object")
    protected.add(string(pair, "pair_id"))
    protected.add(string(pair, "review"))
    comparison = string(pair, "comparison")
    require(comparison in COMPARISONS, "declare matched_checkpoint or separate_runs")
    branches = field(pair, "branches", list)
    require(len(branches) == 2, "pair needs exactly two branches")
    left, right = (validate_branch(branch, protected, numbers, registry) for branch in branches)
    a, b = branches
    require(a["run_id"] != b["run_id"] and a["condition"] != b["condition"],
            "pair needs distinct run IDs and conditions")
    if comparison == "matched_checkpoint":
        checkpoint_a, restore_a, controls_a = reviewed_restore(a)
        checkpoint_b, restore_b, controls_b = reviewed_restore(b)
        require(source_identity(restore_a) != source_identity(restore_b),
                "matched pair needs distinct branch restore references")
        require(canonical_json(controls_a) == canonical_json(controls_b)
                and canonical_json(a["provenance"].get("seed")) == canonical_json(b["provenance"].get("seed")),
                "conflicting matched checkpoint controls")
        require(checkpoint_a == checkpoint_b
                and checkpoint_context(left) == checkpoint_context(right),
                "matched pair must share a pre-T checkpoint, identities and observed context")
    return left, right


def normalize(value):
    return unicodedata.normalize("NFKC", value).casefold()


def term_pattern(terms, flags=0):
    if not terms:
        return re.compile(r"(?!)")
    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=lambda s: (-len(s), s)))
    # Treat underscores in log names as separators while matching whole words.
    return re.compile(r"(?<![^\W_])(?:" + alternatives + r")(?![^\W_])", flags)


def blinding_check(protected, identities, numbers, source_artifacts=()):
    def scan_form(value):
        return normalize(value).replace("\\", "/")

    terms = {scan_form(term).strip() for term in protected}
    numeric_ids = {term for term in terms if NUMERIC.fullmatch(term)}
    terms -= numeric_ids
    for value in numeric_ids:
        for key in ("run_id", "agent_id", "entity_id", "source_id", "condition", "pair_id",
                    "episode_id", "gold_label", "selection_id"):
            numbers.add((key, value))
    def patterns(values):
        # Paths and punctuated identifiers can occur within longer identifiers.
        embedded = sorted(term for term in values if re.search(r"[-_./:#]", term))
        return term_pattern(values), re.compile("|".join(map(re.escape, embedded)) or r"(?!)")

    literal, composite = patterns(terms)
    paths = {scan_form(path) for path in source_artifacts}
    path_alternatives = "|".join(re.escape(path) for path in sorted(paths, key=lambda path: (-len(path), path)))
    # Match a complete path, including before a record locator or sentence punctuation.
    path_pattern = re.compile(r"(?<![\w./-])(?:" + path_alternatives + r")(?![\w/-]|\.[\w.-])"
                              if paths else r"(?!)")
    canonical_paths = {source_identity(path)[0] for path in paths}
    identity_terms = {scan_form(term).strip() for term in identities} - numeric_ids
    original_names, original_composites = patterns(identity_terms)
    assignments = {}
    for key, value in sorted(numbers):
        try:
            assignments.setdefault(scan_form(key), set()).add(Decimal(scan_form(value)))
        except InvalidOperation:
            require(False, "invalid numeric provenance")
    keys = "|".join(map(re.escape, assignments))
    numeric = re.compile(r"(?<![^\W_])(?P<key>" + keys + r""")["']?\s*[:=]\s*["']?"""
                         + f"(?P<value>{NUMERIC.pattern})" + r"(?![^\W_]|[.,]\d)" if keys else r"(?!)")

    def numeric_leak(value):
        for match in numeric.finditer(value):
            try:
                if Decimal(match["value"]) in assignments[match["key"]]:
                    return True
            except InvalidOperation:
                # Out-of-range numerals cannot equal registered metadata.
                continue
        return False

    def check(value, rendered=False):
        normalized = scan_form(value)
        # Also recognize a short identifier at a CamelCase boundary, e.g. houBaseline.
        split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ",
                       unicodedata.normalize("NFKC", value))
        words = scan_form(split)
        # JSON escapes backslashes. Normalize repeated separators and dot
        # components for path matching without changing the rater's text.
        path_text = re.sub(r"(?<=[\w.:-])/{2,}", "/", normalized)
        path_text = re.sub(r"(?<!/)/{3,}", "//", path_text)
        path_text = re.sub(r"/(?:\./)+", "/", path_text)
        path_leak = path_pattern.search(normalized) or path_pattern.search(path_text) or (
            canonical_paths and any(source_identity(match[0].rstrip(".!?"))[0] in canonical_paths
                                    for match in PATH_TOKEN.finditer(path_text) if "/" in match[0])
        )
        require(not (literal.search(normalized) or literal.search(words) or composite.search(normalized) or path_leak
                     or numeric_leak(normalized) or (rendered and (
                         original_names.search(normalized) or original_names.search(words)
                         or original_composites.search(normalized)))),
                "blinding leak in rater evidence (check private terms and aliases)")

    return check


def render_branch(data, side, aliases, check):
    branch = data["branch"]
    start = branch["tick"]
    replacements = {
        term.lower(): aliases[entity]
        for entity, names in branch["names"].items() for term in [entity, *names]
        if not NUMERIC.fullmatch(normalize(term).strip())
    }
    alias_pattern = term_pattern(replacements, re.IGNORECASE)

    def blind(value):
        check(value)
        rendered = alias_pattern.sub(lambda match: replacements.get(match[0].lower(), match[0]), value)
        check(rendered, rendered=True)
        return rendered

    public = {"actor": aliases[branch["agent_id"]], "cue_entity": aliases[branch["entity_id"]],
              "dossier": [], "turns": []}
    private = {key: branch[key] for key in
               ("run_id", "condition", "agent_id", "entity_id", "tick", "last_seen_tick",
                "provenance", "names", "event", "dossier_review")}
    private["aliases"] = aliases
    private["absence_sources"] = {t: row["source_id"] for t, row in sorted(data["observations"].items())
                                  if t < start}
    private["citations"] = {}
    for index, row in enumerate(data["episodes"], 1):
        cite = f"{side}-H{index}"
        public["dossier"].append({"cite": cite, "ticks_before_T": start - row["tick"],
                                  "text": blind(row["text"])})
        private["citations"][cite] = {key: row[key] for key in
                                     ("episode_id", "tick", "source_id", "gold_label") if key in row}
    for offset in (0, 1):
        t = start + offset
        obs, decision, outcome = (data[key][t] for key in ("observations", "decisions", "outcomes"))
        turn = {"when": "T" if offset == 0 else "T+1"}
        recorded_offset = outcome["recorded_tick"] - start
        recorded_when = "T" if recorded_offset == 0 else f"T+{recorded_offset}"
        for suffix, key, record, content in (
            ("O", "observation", obs, {"text": blind(obs["text"])}),
            ("D", "decision", decision, {"chosen_action": blind(decision["action"]),
                                         "logged_outgoing_message": blind(decision["message"])}),
            ("F", "outcome_feedback", outcome, {"status": outcome["status"], "recorded_when": recorded_when,
                                                "text": blind(outcome["feedback"])}),
        ):
            cite = f"{side}-{suffix}{offset}"
            turn[key] = {"cite": cite, **content}
            private["citations"][cite] = {key: record[key] for key in
                                         ("tick", "source_id", "source_kind", "decision_source_id",
                                          "execution_tick", "recorded_tick", "selection_id",
                                          "effect_status") if key in record}
        public["turns"].append(turn)
    return public, private


def history_id(record, key):
    value = string(record, key)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) and not NUMERIC.fullmatch(value),
            f"{key} must be a canonical, nonnumeric identifier")
    return value


def evidence_sources(branch):
    yield branch["event"]["source_id"]
    yield branch["dossier_review"]["source_id"]
    if "restore_source" in branch["provenance"]:
        yield string(branch["provenance"], "restore_source")
    for name in ("observations", "decisions", "outcomes", "episodes"):
        yield from (row["source_id"] for row in branch[name])


def register_source_group(pair, protected, group_registry):
    group = history_id(pair, "source_group_id")
    protected.add(group)

    def register(kind, identity):
        previous = group_registry.setdefault((kind, identity), group)
        require(previous == group, f"conflicting source_group_id declarations for shared {kind}")

    register("group identifier", normalize(group))
    for branch in pair["branches"]:
        provenance = branch["provenance"]
        register("run", normalize(branch["run_id"].strip().replace("\\", "/")))
        register("source history", normalize(history_id(provenance, "source_history_id")))
        if "checkpoint_id" in provenance:
            register("checkpoint", digest(provenance, "checkpoint_id"))
        for source in evidence_sources(branch):
            register("source artifact", source_identity(source)[0])
    return group


def split_groups(pair_groups, seed):
    """Assign whole source groups and retain counts for the custodian."""
    groups = sorted(set(pair_groups.values()))
    random.Random(seed).shuffle(groups)
    held_out_count = (3 * len(groups) + 5) // 10
    held_out = set(groups[:held_out_count])
    membership = {group: "held_out" if group in held_out else "development" for group in sorted(groups)}
    counts = {split: {"groups": 0, "cards": 0} for split in ("development", "held_out")}
    for split in membership.values():
        counts[split]["groups"] += 1
    for group in pair_groups.values():
        counts[membership[group]]["cards"] += 1

    limitation = None
    if len(groups) == 1:
        limitation = "Only one independent source group. All cards are development; no held-out split is available."
    elif len(groups) == 2:
        limitation = "Only two independent source groups. One group per split is possible; 70/30 is not."
    elif held_out_count == 1:
        limitation = "The held-out split contains only one independent source group."
    return {"group_membership": membership, "counts": counts, "limitation": limitation}


def insert_self_pairs(cards, development_count, count, seed):
    require(count <= development_count,
            f"requested {count} self-pair controls but only {development_count} development comparisons are eligible")
    rng = random.Random(f"self-pairs:{seed}")
    parents = rng.sample(cards[:development_count], count)
    for offset, (pair, sides, _) in enumerate(parents):
        position = rng.randrange(development_count + offset + 1)
        cards.insert(position, (pair, sides, SelfPairControl(rng.randrange(2))))


def _build(manifest, seed, *, pilot_assignment=False, self_pair_count=0):
    require(type(self_pair_count) is int and self_pair_count in (0, 1, 2),
            "self_pair_count must be 0, 1 or 2")
    require(isinstance(manifest, dict), "manifest must be an object")
    version = field(manifest, "schema_version", int)
    require(version in (SCHEMA_VERSION, GROUPED_SCHEMA_VERSION), "manifest schema_version must be 2 or 3")
    grouped = version == GROUPED_SCHEMA_VERSION
    require(not self_pair_count or grouped, "self-pair controls require schema_version 3 development groups")
    synthetic = field(manifest, "synthetic", bool)
    require(not pilot_assignment or grouped or synthetic,
            "genuine pilot assignments require schema_version 3 with source-history groups")
    terms = field(manifest, "private_terms", list)
    require(all(map(nonblank, terms)), "private_terms must contain nonblank strings")
    protected = set(terms)
    numbers = {("seed", str(seed))}
    pairs = field(manifest, "pairs", list)
    require(bool(pairs), "manifest must contain reviewed pairs")
    pair_ids, opportunities, validated = set(), set(), []
    registry = {}
    pair_groups, group_registry = {}, {}
    identities = set()
    source_artifacts = set()
    for index, pair in enumerate(pairs, 1):
        try:
            sides = validate_pair(pair, protected, numbers, registry)
            opportunity = frozenset(tuple(branch[key] for key in
                                          ("run_id", "agent_id", "entity_id", "tick"))
                                    for branch in pair["branches"])
            require(pair["pair_id"] not in pair_ids and opportunity not in opportunities,
                    "duplicate pair ID or paired opportunity")
            if grouped:
                pair_groups[pair["pair_id"]] = register_source_group(pair, protected, group_registry)
            else:
                require("source_group_id" not in pair, "source_group_id requires manifest schema_version 3")
        except ValueError as error:
            raise ValueError(f"pair {index}: {error}") from error
        pair_ids.add(pair["pair_id"])
        opportunities.add(opportunity)
        validated.append((pair, sides, None))
        for branch in pair["branches"]:
            references = [*evidence_sources(branch), branch["dossier_review"]["review_source"],
                          *(row["decision_source_id"] for row in branch["outcomes"])]
            for reference in references:
                artifact, locator = source_identity(reference)
                protected.update((reference, artifact + locator))
                # Bare labels and numbers can also occur as ordinary evidence text.
                original_path = reference.partition("#")[0].strip().replace("\\", "/")
                source_artifacts.update(
                    path for path in (original_path, artifact)
                    if path not in {".", "..", "/", "//"}
                    and ("/" in path or "." in path) and not NUMERIC.fullmatch(path)
                )
            for entity, aliases in branch["names"].items():
                identities.update([entity, *aliases])

    check = blinding_check(protected, identities, numbers, source_artifacts)
    fingerprint_parts = [RUBRIC_REVISION, seed, manifest]
    if self_pair_count:
        fingerprint_parts.append({"self_pairs": self_pair_count})
    fingerprint = canonical_json(fingerprint_parts).encode("utf-8")
    export_info = {"export_id": "BP-" + hashlib.sha256(fingerprint).hexdigest()[:24],
                   "rubric_revision": RUBRIC_REVISION}
    rng = random.Random(seed)
    rng.shuffle(validated)
    split = split_groups(pair_groups, seed) if grouped else None
    if split:
        # Number development cards first so public packet IDs have no held-out gaps.
        validated.sort(key=lambda item: split["group_membership"][pair_groups[item[0]["pair_id"]]] == "held_out")
        if self_pair_count:
            development = split["counts"]["development"]
            insert_self_pairs(validated, development["cards"], self_pair_count, seed)
            development["self_pairs"] = self_pair_count
    packets, key = [], []
    for index, (pair, sides, control) in enumerate(validated, 1):
        packet_id = f"P{index:03d}"
        comparison = pair["comparison"]
        packet = {**export_info, "packet_id": packet_id, "comparison": COMPARISONS[comparison], "branches": {}}
        entry = {"packet_id": packet_id}
        if control is None:
            entry["pair_id"] = pair["pair_id"]
        entry.update(comparison=comparison, review=pair["review"], branches={})
        if split:
            group = pair_groups[pair["pair_id"]]
            entry.update(source_group_id=group, split=split["group_membership"][group])
        if control:
            selected = sides[control.selected_branch_index]
            displayed = (selected, selected)
            entry["control"] = {"type": "self_pair", "parent_pair_id": pair["pair_id"],
                                "selected_branch_index": control.selected_branch_index}
        else:
            if comparison == "matched_checkpoint":
                entry["match_verification"] = "reviewer_attested"
            displayed = rng.sample(sides, 2)
        # Use the same aliases for the focal and cue roles in both branches.
        # Other entities are shared when both displays use the same identity space.
        alias_by_role = {("actor",): f"Entity {index}-1", ("cue",): f"Entity {index}-2"}
        share_other_entities = control is not None or comparison == "matched_checkpoint"
        for side, data in zip("AB", displayed):
            branch = data["branch"]
            aliases = {}
            for entity in sorted(branch["names"]):
                if entity == branch["agent_id"]:
                    role = ("actor",)
                elif entity == branch["entity_id"]:
                    role = ("cue",)
                else:
                    role = ("other", "" if share_other_entities else side, entity)
                if role not in alias_by_role:
                    alias_by_role[role] = f"Entity {index}-{len(alias_by_role) + 1}"
                aliases[entity] = alias_by_role[role]
            try:
                public, private = render_branch(data, side, aliases, check)
            except ValueError as error:
                raise ValueError(f"{packet_id} {side}: {error}") from error
            packet["branches"][side], entry["branches"][side] = public, private
        if entry.get("split") != "held_out":
            packets.append(packet)
        key.append(entry)
    header = {"schema_version": version, "synthetic": synthetic, **export_info}
    export = {**header, "packets": packets}
    check(public_json(export), rendered=True)
    check(rating_sheet(export), rendered=True)
    private = {**header, "seed": seed, "packets": key}
    if split:
        private["split"] = split
    return export, private, check


def build(manifest, seed, *, self_pair_count=0):
    export, key, _ = _build(manifest, seed, self_pair_count=self_pair_count)
    return export, key


def balanced_swaps(condition_pairs, rng):
    """Orient comparisons so each condition is on A and B equally often, up to one."""
    adjacency = defaultdict(list)
    for index, (first, second) in enumerate(condition_pairs):
        adjacency[first].append((index, second))
        adjacency[second].append((index, first))

    # Connect odd-degree conditions to a dummy vertex. The augmented graph has
    # only even degrees, so unused edges can always be traversed in closed walks.
    odd_conditions = [condition for condition, edges in adjacency.items() if len(edges) % 2]
    for index, condition in enumerate(odd_conditions, start=len(condition_pairs)):
        adjacency[condition].append((index, None))
        adjacency[None].append((index, condition))
    for edges in adjacency.values():
        rng.shuffle(edges)

    swaps = [False] * len(condition_pairs)
    used = set()
    for start in adjacency:
        current = start
        while adjacency[current]:
            index, following = adjacency[current].pop()
            if index in used:
                continue
            used.add(index)
            if index < len(condition_pairs):
                swaps[index] = current != condition_pairs[index][0]
            current = following
    return swaps


def oriented_packet(packet, export_id, display_to_canonical):
    rendered = {**packet, "export_id": export_id, "branches": {}}
    for side, canonical_side in display_to_canonical.items():
        branch = deepcopy(packet["branches"][canonical_side])
        rendered["branches"][side] = branch
        if side != canonical_side:
            for row in branch["dossier"]:
                row["cite"] = side + row["cite"][1:]
            for turn in branch["turns"]:
                for field_name in ("observation", "decision", "outcome_feedback"):
                    record = turn[field_name]
                    record["cite"] = side + record["cite"][1:]
    return rendered


def assignment_id(key, rater_id, mapping):
    # A public hash of the mapping would let raters try every A/B orientation.
    entries = sorted((row["packet_id"], row["display_to_canonical"]) for row in mapping)
    payload = canonical_json(["behavior-pilot-assignment-v1", key["export_id"], key["rubric_revision"],
                              rater_id, entries]).encode("utf-8")
    return "AS-" + hmac.new(bytes.fromhex(key["assignment_id_key"]), payload, "sha256").hexdigest()[:32]


def build_assignments(manifest, seed, rater_count, *, self_pair_count=0):
    """Return one blinded export per rater and one private canonical key."""
    exports, key, _ = _build_assignments(manifest, seed, rater_count, self_pair_count=self_pair_count)
    return exports, key


def _build_assignments(manifest, seed, rater_count, *, self_pair_count=0):
    require(type(rater_count) is int and rater_count in (3, 4, 5), "assignments need 3, 4 or 5 raters")
    export, key, check = _build(manifest, seed, pilot_assignment=True, self_pair_count=self_pair_count)
    fingerprint = canonical_json(["rater-assignment-v1", export["export_id"], rater_count]).encode("utf-8")
    export_id = "BP-" + hashlib.sha256(fingerprint).hexdigest()[:24]
    # Derive a deterministic key from private inputs, under a separate domain from export IDs.
    secret_material = canonical_json(["behavior-pilot-assignment-key-v1", seed, manifest]).encode("utf-8")
    key.update(export_id=export_id, assignment_id_key=hashlib.sha256(secret_material).hexdigest())
    rng = random.Random(seed)
    rater_ids = [f"R{index:02d}" for index in range(1, rater_count + 1)]
    rng.shuffle(rater_ids)
    canonical = {entry["packet_id"]: entry for entry in key["packets"]}
    ordinary_packets, control_packets, condition_pairs = [], [], []
    for packet in export["packets"]:
        entry = canonical[packet["packet_id"]]
        if "control" in entry:
            control_packets.append((packet, False))
        else:
            ordinary_packets.append(packet)
            condition_pairs.append(tuple(entry["branches"][side]["condition"] for side in "AB"))
    swaps_by_rater = {}
    # Complementary rater pairs give one of each orientation on every card.
    # With an odd number of raters, one independently balanced rater breaks the tie.
    for index in range(0, rater_count, 2):
        swaps = balanced_swaps(condition_pairs, rng)
        swaps_by_rater[rater_ids[index]] = swaps
        if index + 1 < rater_count:
            swaps_by_rater[rater_ids[index + 1]] = [not swap for swap in swaps]

    exports, assignments, assignment_ids = {}, {}, {}
    for rater_id, swaps in sorted(swaps_by_rater.items()):
        order = list(zip(ordinary_packets, swaps)) + control_packets
        rng.shuffle(order)
        packets, mapping = [], []
        for packet, swap in order:
            display_to_canonical = dict(zip("AB", "BA" if swap else "AB"))
            packets.append(oriented_packet(packet, export_id, display_to_canonical))
            mapping.append({"packet_id": packet["packet_id"], "display_to_canonical": display_to_canonical})
        assignment_ids[rater_id] = assignment_id(key, rater_id, mapping)
        rater_export = {**export, "export_id": export_id, "rater_id": rater_id,
                        "assignment_id": assignment_ids[rater_id], "packets": packets}
        check(public_json(rater_export), rendered=True)
        check(rating_sheet(rater_export), rendered=True)
        exports[rater_id] = rater_export
        assignments[rater_id] = mapping
    return exports, {**key, "assignments": assignments, "assignment_ids": assignment_ids}, check


def rating_sheet(export):
    lines = ["# D1–D5 behavioral pilot rating sheet", ""]
    if export["synthetic"]:
        lines.extend(["**Synthetic practice material.**", ""])
        if "rater_id" in export:
            lines.extend(["This workflow dry run provides no human-evaluation evidence.", ""])
    lines.extend([
        f"Export ID {export['export_id']} · Rubric revision {export['rubric_revision']}",
    ])
    if "assignment_id" in export:
        lines.append(f"Assignment ID {export['assignment_id']}")
    lines.extend([
        "",
        "Use the matching packet. Rate A and B independently on D1–D4, then compare on D5.",
        "Cite the neutral evidence IDs and give a short reason for each judgment.",
        "NEI means not enough information. MISSING means the rating was left uncompleted.",
        "A decision line records the chosen action and logged outgoing text.",
        "Use outcome feedback for claims about execution or delivery. Its recorded time can follow the decision.",
        "",
        "D1 counts distinct contradicted claims. Enter zero for both counts and No for the flag if there were no past claims.",
        "D2 records implicit use separately when a specified probe signature supports it.",
        "D3 distinguishes irrelevant or stale reminders from invented claims. Use NEI when history is insufficient.",
        "D5 Tie means equally believable. Use NEI if the evidence is insufficient to compare.",
        "",
        f"Pseudonymous rater ID {export.get('rater_id', '______')}",
        "",
        "D4 scale 1 clearly implausible, 2 major mismatch, 3 plausible but mixed,",
        "4 coherent with a minor weakness, 5 strong contextual fit.",
    ])
    for packet in export["packets"]:
        lines.extend(["", f"## {packet['packet_id']}", "",
                      "| Side | Measure | Choices or fields | Rating | Evidence IDs | Reason |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for side in "AB":
            for measure, choices in BRANCH_MEASURES.values():
                choices = ("Nonnegative integer",) if choices is None else choices
                values = " / ".join(map(str, (*choices, *RATING_STATES)))
                lines.append(f"| {side} | {measure} | {values} | | | |")
        d5_values = " / ".join((*D5_CHOICES, *RATING_STATES))
        lines.append(f"| Pair | D5 Overall believability | {d5_values} | | | |")
    return "\n".join(lines) + "\n"


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON field {key}")
        result[key] = value
    return result


def public_json(export):
    return json.dumps(export, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="private reviewed JSON manifest")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--raters", type=int, choices=(3, 4, 5),
                        help="assign development comparisons to this many pseudonymous raters")
    parser.add_argument("--self-pairs", type=int, choices=(1, 2), default=0,
                        help="include this many self-pair controls from distinct development comparisons")
    args = parser.parse_args(argv)
    try:
        raw = args.manifest.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=unique_object)
        if args.raters is None:
            export, key = build(manifest, args.seed, self_pair_count=args.self_pairs)
            exports = {None: export}
        else:
            exports, key = build_assignments(manifest, args.seed, args.raters, self_pair_count=args.self_pairs)
        key["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        private_json = json.dumps(key, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # On Windows, inherit the parent ACL, including the active sandbox identity.
        args.out_dir.mkdir(mode=0o777 if os.name == "nt" else 0o700)
        rater_dir = args.out_dir / "rater"
        rater_dir.mkdir()
        (args.out_dir / "private_key.json").write_text(private_json, encoding="utf-8")
        for rater_id, export in exports.items():
            destination = rater_dir
            if rater_id is not None:
                destination = rater_dir / rater_id
                destination.mkdir()
            (destination / "packets.json").write_text(public_json(export), encoding="utf-8")
            (destination / "rating_sheet.md").write_text(rating_sheet(export), encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    prefix = "Synthetic pilot" if key["synthetic"] else "Behavior pilot"
    count = len(next(iter(exports.values()))["packets"])
    if args.raters is None:
        print(f"{prefix}: {count} packet(s) in {rater_dir}")
    else:
        print(f"{prefix}: {count} comparison(s) per rater, {args.raters} raters in {rater_dir}")
    if args.self_pairs:
        print(f"Pilot block: {count - args.self_pairs} ordinary comparison(s), {args.self_pairs} self-pair control(s).")
    if "split" in key:
        development, held_out = (key["split"]["counts"][name] for name in ("development", "held_out"))
        print(f"Development: {development['groups']} group(s), {development['cards']} card(s). "
              f"Held-out: {held_out['groups']} group(s), {held_out['cards']} card(s).")
        if key["split"]["limitation"]:
            print(key["split"]["limitation"])
    print(f"Private condition/provenance key: {args.out_dir / 'private_key.json'}")


if __name__ == "__main__":
    main()
