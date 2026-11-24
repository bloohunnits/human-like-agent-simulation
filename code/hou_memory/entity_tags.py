"""The relational tags of one turn, read from what the agent observes.

No simulation import, so the agent wrapper and anything outside the
simulation (code/memory_storage, a proxy) can share it. Text is never searched
for names: only structured fields of the observation and the action are read.
"""

from __future__ import annotations


def heard(obs: dict) -> dict:
    """Messages by sender. The newer engine calls them incoming_broadcasts."""
    return obs.get("message") or obs.get("incoming_broadcasts") or {}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def extract_entities(obs: dict, action: dict | None = None) -> dict:
    """Structured observed cues; actions add explicit episode tags only.
    Pass no action for what is in view now, the cue of a recall."""
    action = action or {}
    people, artifacts = set(), set()

    for contents in obs.get("observation", {}).values():
        for item in contents:
            item = str(item)
            if item.startswith("A("):  # artifact, rendered "A(text): name"
                artifacts.add(item.split(": ", 1)[-1])
            elif item in ("X", "") or _is_number(item):
                continue  # off-map marker or a food energy value - not a named entity
            else:
                # Color is an appearance, not a new entity identity.
                people.add(item.rsplit("(", 1)[0] if item.endswith(")") else item)

    for item in obs.get("inventory", []):
        if str(item).startswith("A("):
            artifacts.add(str(item).split(": ", 1)[-1])

    for sender in heard(obs):
        people.add(sender)

    params = action.get("params", {}) or {}
    act = action.get("action")
    if act in ("give", "take") and params.get("target"):
        people.add(str(params["target"]))
    elif act == "reproduce" and params.get("name"):
        people.add(str(params["name"]))
    elif act == "create_artifact" and params.get("name"):
        artifacts.add(str(params["name"]))

    return {"people": sorted(people), "artifacts": sorted(artifacts),
            "places": sorted(set(obs.get("named_places", [])))}
