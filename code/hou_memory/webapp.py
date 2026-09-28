"""Local web UI over the Hou memory kernel - the same data explore.py
reads (real sim dumps, auto-detected; falls back to the toy story), but as
a page with plain-English labels, color, and a bar per memory instead of
a terminal table.

    python webapp.py
    -> open http://127.0.0.1:5057

No LLM calls happen here. Real SBERT embeddings, real Hou math, same
HouMemoryStore as everything else in this folder.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from embedder import SBERTEmbedder
from explore import (
    DEFAULT_G0,
    TERRALINGUA_ROOT,
    find_experiments,
    latest_tick_in,
    load_experiment,
    preset_story,
)
from hou_memory import ACTR_TAU, actr_recall, recall_probability
from memory_graph import MemoryGraph

app = Flask(__name__, static_folder="static")

STATE = {}
EXTRA_EXPERIMENTS = []


def _experiments():
    return list(dict.fromkeys(EXTRA_EXPERIMENTS + find_experiments(TERRALINGUA_ROOT)))


def _init_state():
    embedder = SBERTEmbedder()
    STATE["embedder"] = embedder
    experiments = _experiments()
    STATE["experiments"] = experiments
    if not (experiments and _load_experiment(experiments[0])):
        _load_preset(DEFAULT_G0)


# USD per 1M tokens. Filled from openai.com pricing, see INTEGRATION.md;
# update there when rates change. None = unknown model, UI shows tokens only.
PRICES = {  # developers.openai.com/api/docs/pricing, checked 2026-09-25
    "o4-mini": {"in": 1.10, "out": 4.40},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5.1": {"in": 1.25, "out": 10.00},
}


def _run_model(dir_path: Path):
    import json as _json

    try:
        p = _json.load(open(dir_path.parent / "params.json"))
        return p.get("agent", {}).get("model")
    except Exception:
        return None


def _read_usage(dir_path: Path):
    """Per-agent token burn from TerraLingua's own token_counts.jsonl:
    one line per (timestep, agent) with input/output token totals."""
    import json as _json

    f = dir_path / "token_counts.jsonl"
    if not f.exists():
        return {}
    # Dedupe by (agent, timestep), keeping the LAST line: a resumed run
    # re-appends lines for replayed ticks and would double-count.
    latest = {}
    try:
        for line in open(f):
            try:
                row = _json.loads(line)
            except ValueError:
                continue  # a line mid-write during a live run
            latest[(row["agent_tag"], row.get("timestep"))] = row
    except OSError:
        pass
    usage = {}
    for (tag, _ts), row in latest.items():
        u = usage.setdefault(tag, {"inTokens": 0, "outTokens": 0, "ticks": 0})
        u["inTokens"] += row.get("total_input_tokens", 0)
        u["outTokens"] += row.get("total_output_tokens", 0)
        u["ticks"] += 1
    return usage


def _load_experiment(dir_path: Path):
    import json as _json

    stores = load_experiment(dir_path, STATE["embedder"])
    # Per-agent extras written into the dumps: human name and current plan.
    names, plans = {}, {}
    for f in dir_path.glob("*_hou_memory.json"):
        try:
            payload = _json.load(open(f))
            names[payload["agent_tag"]] = payload.get("agent_name", payload["agent_tag"])
            plans[payload["agent_tag"]] = payload.get("plan", "")
        except Exception:
            pass
    if not stores:
        # every dump unreadable (or the dir vanished) - keep current state
        print(f"no readable dumps in {dir_path}, not switching")
        return False
    tag = sorted(stores)[0]
    STATE["mode"] = "sim"
    STATE["experiment_dir"] = dir_path
    STATE["stores"] = stores
    STATE["names"] = names
    STATE["plans"] = plans
    STATE["usage"] = _read_usage(dir_path)
    STATE["run_model"] = _run_model(dir_path)
    STATE["tag"] = tag
    STATE["store"] = stores[tag]
    STATE["owner"] = names.get(tag, tag)
    STATE["tick"] = latest_tick_in(STATE["store"])
    STATE["graph"] = MemoryGraph(STATE["store"])
    return True


def _load_preset(g0: float):
    STATE["mode"] = "preset"
    STATE["experiment_dir"] = None
    STATE["stores"] = None
    STATE["names"] = {}  # else _namify rewrites Petra's story with sim names
    STATE["plans"] = {}
    STATE["usage"] = {}
    STATE["run_model"] = None
    STATE["tag"] = None
    STATE["store"] = preset_story(STATE["embedder"], g0)
    STATE["owner"] = "Petra"
    STATE["tick"] = 6.0
    STATE["graph"] = MemoryGraph(STATE["store"])


def _finite_positive(value, lo=0.1, hi=1000.0):
    """Parse a slider value; None if unusable. Guards the kernel, which
    raises on g <= 0 - one bad POST must not 500 every later request."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return min(max(x, lo), hi)


def _namify(s):
    """Show friendly names everywhere, even in memories written before the
    agent was renamed (the step-0 edge): swap any leftover beingN tag for its
    name. Longest tags first so being10 isn't clobbered by being1."""
    names = STATE.get("names") or {}
    for tag in sorted(names, key=len, reverse=True):
        s = s.replace(tag, names[tag])
    return s


def _row(mem, r, p, tick, surfaced):
    entities = (mem.meta or {}).get("entities", {}) or {}
    a, actr_p = actr_recall(r if r is not None else 1.0, mem.recall_times, tick)
    return {
        "actrA": None if a == float("-inf") else round(a, 1),
        "actrP": actr_p,
        "id": id(mem),
        "text": _namify(mem.text),
        "r": r,
        "p": p,
        "g": mem.g,
        "age": max(tick - mem.t_last_recalled, 0.0),
        "createdAt": mem.t_created,
        "recallCount": mem.recall_count,
        "surfaced": surfaced,
        "owner": STATE.get("owner"),
        "event": (mem.meta or {}).get("event"),
        "people": list(dict.fromkeys(_namify(x) for x in entities.get("people", []))),
        "artifacts": entities.get("artifacts", []),
        "places": entities.get("places", []),
    }


def _sim_agents(dir_path: Path):
    """List (tag, name) for each agent dump in a run, name from the dump."""
    import json as _json

    out = []
    for f in sorted(dir_path.glob("*_hou_memory.json")):
        tag = f.stem.replace("_hou_memory", "")
        name = tag
        try:
            name = _json.load(open(f)).get("agent_name", tag)
        except Exception:
            pass
        out.append({"tag": tag, "name": name})
    return out


def _run_name(dir_path: Path) -> str:
    # .../logs_town/logs/town/agent_logs -> "town"
    parts = dir_path.parts
    return parts[-2] if len(parts) >= 2 else str(dir_path)


def _frames_dir():
    d = STATE.get("experiment_dir")
    return Path(d).parent / "frames" if d else None


def _frame_count():
    d = _frames_dir()
    if not d or not d.exists():
        return 0
    try:
        return len(list(d.glob("[0-9]*.png")))
    except OSError:
        return 0


def _run_mtime(dir_path):
    """Newest dump mtime for the loaded run - the follow-live poll compares
    this and skips the reload when nothing changed (finished runs go quiet,
    and your dials stop being reverted every four seconds)."""
    if not dir_path:
        return 0
    try:
        return max((f.stat().st_mtime for f in Path(dir_path).glob("*_hou_memory.json")), default=0)
    except OSError:
        return 0


def _state_payload():
    experiments = _experiments()
    STATE["experiments"] = experiments
    latest = latest_tick_in(STATE["store"])
    return {
        "mode": STATE["mode"],
        "tick": STATE["tick"],
        # Slider headroom past the last event so you can push time forward
        # and watch memories fade.
        "maxTick": int(latest) + 20,
        "runName": _run_name(STATE["experiment_dir"]) if STATE["experiment_dir"] else None,
        "experimentDir": str(STATE["experiment_dir"]) if STATE["experiment_dir"] else None,
        "currentAgent": STATE["tag"],
        "currentAgentName": STATE.get("owner"),
        "availableAgents": sorted(STATE["stores"]) if STATE["stores"] else [],
        "sims": [
            {"index": i, "path": str(d), "runName": _run_name(d), "agents": _sim_agents(d)}
            for i, d in enumerate(experiments)
        ],
        "memoryCount": len(STATE["store"].memories),
        "topK": STATE["store"].top_k,
        "recallThreshold": STATE["store"].recall_threshold,
        "g0": STATE["store"].g0,
        "kernel": STATE.get("kernel", "hou"),
        "lambda": STATE.get("lambda", 0.5),
        "graphStats": STATE["graph"].stats() if STATE.get("graph") else None,
        "runModel": STATE.get("run_model"),
        "plan": _namify(((STATE.get("plans") or {}).get(STATE["tag"], "") or "").removeprefix("Current plan:").strip()) if STATE["tag"] else "",
        "runMtime": _run_mtime(STATE.get("experiment_dir")),
        "frameCount": _frame_count(),
        "usage": [
            {
                "tag": t,
                "name": (STATE.get("names") or {}).get(t, t),
                "inTokens": u["inTokens"],
                "outTokens": u["outTokens"],
                "ticks": u["ticks"],
                "perTick": round((u["inTokens"] + u["outTokens"]) / u["ticks"]) if u["ticks"] else 0,
                "costUsd": (
                    round((u["inTokens"] * PRICES[m]["in"] + u["outTokens"] * PRICES[m]["out"]) / 1e6, 4)
                    if (m := STATE.get("run_model")) in PRICES else None
                ),
            }
            for t, u in sorted((STATE.get("usage") or {}).items())
        ],
    }


@app.route("/api/frame/<int:n>")
def api_frame(n):
    """The rendered world picture for tick n, from the run's own frames."""
    d = _frames_dir()
    if d:
        f = d / f"{n:05d}.png"
        if f.exists():
            return send_file(f, mimetype="image/png", max_age=3600)
    return ("no frame", 404)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/audit")
def audit_page():
    return send_from_directory("static", "audit.html")


@app.route("/api/live_audit")
def api_live_audit():
    """Actual committed sim evidence, independent of the legacy what-if console."""
    import json

    directory = STATE.get("experiment_dir")
    tag = request.args.get("agent", STATE.get("tag"))
    if not directory or tag not in (STATE.get("stores") or {}):
        return jsonify({"error": "Select a simulation agent first."}), 400
    path = Path(directory) / f"{tag}_hou_memory.json"
    try:
        with open(path) as f:
            snapshot = json.load(f)
    except (OSError, ValueError):
        return jsonify({"error": "Snapshot temporarily unavailable."}), 409
    engine = snapshot.get("associative")
    if engine is None:
        return jsonify({"error": "Legacy run: no committed associative traces. The main page is its historical what-if console."}), 404
    latest = engine.get("last_event")
    tick = request.args.get("tick", type=float)
    latest_tick = latest["tick"] if latest else -1
    if tick is None:
        tick = latest_tick
    events = {}
    audit_file = Path(directory) / f"{tag}_memory_audit.jsonl"
    if audit_file.exists():
        with open(audit_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row["tick"] <= latest_tick:
                        events[row["tick"]] = row
                except (ValueError, KeyError):
                    continue  # a concurrent append may have a partial last line
    if latest:
        events[latest_tick] = latest  # atomic snapshot is authoritative
    trace = events.get(tick)
    prompt = trace.get("agent_step") if trace else None
    return jsonify({"agent": snapshot["agent_name"], "config": engine["config"],
                    "ticks": sorted(events), "trace": trace, "agent_step": prompt,
                    "edge_count": len(engine["edges"]), "memory_count": len(snapshot["memories"]),
                    "note": "Actual pre-update scores and committed updates. Duplicate ticks after resume use the latest record."})


@app.route("/api/state")
def api_state():
    return jsonify(_state_payload())


def _kernel_sort_key(pair):
    """Rank by the active kernel: Hou's p, or ACT-R's activation."""
    _mem, row = pair
    if STATE.get("kernel") == "actr":
        return (row["actrP"], row["actrA"] if row["actrA"] is not None else float("-inf"))
    return (row["p"],)


def _select_surfacing(pairs, store):
    """Which memories make the cut under the active kernel: Hou's threshold
    on p, or ACT-R's activation clearing tau. Returns the surfacing mems."""
    if STATE.get("kernel") == "actr":
        eligible = [(m, r) for m, r in pairs if r["actrA"] is not None and r["actrA"] > ACTR_TAU]
    else:
        eligible = [(m, r) for m, r in pairs if r["p"] > store.recall_threshold]
    return [m for m, _r in eligible[: store.top_k]]


@app.route("/api/set_lambda", methods=["POST"])
def api_set_lambda():
    """Spread damping. 0 turns the graph off - by construction the scores
    then equal the bare kernel's, which is the ablation."""
    data = request.get_json(force=True, silent=True) or {}
    v = _finite_positive(data.get("value"), lo=0.0, hi=1.0)
    if v is None:
        return jsonify({"error": "value must be 0..1"}), 400
    STATE["lambda"] = v
    return jsonify(_state_payload())


@app.route("/api/set_kernel", methods=["POST"])
def api_set_kernel():
    data = request.get_json(force=True, silent=True) or {}
    k = data.get("value")
    if k not in ("hou", "actr"):
        return jsonify({"error": "value must be hou or actr"}), 400
    STATE["kernel"] = k
    return jsonify(_state_payload())


def _cue_pairs(store, tick, cue):
    """Score every present memory twice: solo (direct r) and with the graph
    (r widened by spread). The active score p is the graph one, which
    equals solo exactly when lambda = 0."""
    lam = STATE.get("lambda", 0.5)
    graph = STATE.get("graph")
    full = store.score(store._embed_fn(cue), tick)  # aligned with store.memories
    direct = [r if m.t_created <= tick else 0.0 for m, r, _p in full]
    received = graph.spread(cue, direct, lam) if graph else [(0.0, None)] * len(full)
    ts = store.time_scale
    pairs = []
    for idx, (mem, r, solo_p) in enumerate(full):
        if mem.t_created > tick:
            continue
        rec, via = received[idx]
        elapsed = max(tick - mem.t_last_recalled, 0.0) / ts
        graph_p = recall_probability(max(r, rec), elapsed, mem.g)
        row = _row(mem, r, solo_p, tick, False)
        row["p"] = graph_p
        row["soloP"] = solo_p
        row["graphP"] = graph_p
        row["via"] = via if rec > r else None
        row["received"] = round(rec, 3)
        row["_idx"] = idx
        pairs.append((mem, row))
    return pairs, received


def _mark_surfacing(pairs, store):
    """Surface by the active (graph) score, and flag rescues: memories that
    made the cut only because the graph widened their cue."""
    surfacing = {id(m) for m in _select_surfacing(pairs, store)}
    solo_sorted = sorted(pairs, key=lambda pr: pr[1]["soloP"], reverse=True)
    solo_set = {id(m) for m, row in
                [pr for pr in solo_sorted if pr[1]["soloP"] > store.recall_threshold][: store.top_k]}
    hou_active = STATE.get("kernel", "hou") == "hou"
    for mem, row in pairs:
        row["surfaced"] = id(mem) in surfacing
        # rescue is defined against Hou's solo ranking; under another
        # ranking kernel the comparison is incoherent, so don't claim it
        row["rescued"] = hou_active and row["surfaced"] and id(mem) not in solo_set
        del row["_idx"]
    return surfacing


@app.route("/api/memories")
def api_memories():
    """No cue -> raw standing (ceiling scores at r=1). Cue given -> real
    scores against every memory, surfacing marked per the active kernel."""
    store = STATE["store"]
    tick = STATE["tick"]
    cue = request.args.get("cue", "").strip()
    # Time honesty: at clock t, memories recorded after t don't exist yet.
    present = [m for m in store.memories if m.t_created <= tick]
    if not cue:
        pairs = [(mem, _row(mem, None, recall_probability(1.0, max(tick - mem.t_last_recalled, 0.0) / store.time_scale, mem.g), tick, None))
                 for mem in present]
        pairs.sort(key=_kernel_sort_key, reverse=True)
        return jsonify({"cue": None, "rows": [r for _m, r in pairs]})

    pairs, _received = _cue_pairs(store, tick, cue)
    pairs.sort(key=_kernel_sort_key, reverse=True)
    _mark_surfacing(pairs, store)
    return jsonify({"cue": cue, "rows": [r for _m, r in pairs]})


@app.route("/api/recall", methods=["POST"])
def api_recall():
    """Commit version of /api/memories?cue=...: actually reinforces."""
    data = request.get_json(force=True)
    cue = data.get("cue", "").strip()
    if not cue:
        return jsonify({"error": "cue required"}), 400
    store = STATE["store"]
    tick = STATE["tick"]
    # Same scoring as the preview, but committed: reinforce what surfaces.
    # Done here (not via store.recall) so the clock filter applies - a memory
    # from the future can't be reinforced by a past-time recall.
    pairs, received = _cue_pairs(store, tick, cue)
    pairs.sort(key=_kernel_sort_key, reverse=True)
    idx_of = {id(m): row["_idx"] for m, row in pairs}
    surfacing = _select_surfacing(pairs, store)
    for mem in surfacing:
        # Never move a memory's clock backward: reinforcing at a rewound
        # clock would rewrite t_last_recalled into the past and permanently
        # weaken it. At a past clock the reinforce is skipped, not rewound.
        if tick >= mem.t_last_recalled:
            store._reinforce(mem, tick)
    graph = STATE.get("graph")
    if graph and STATE.get("lambda", 0.5) > 0:
        graph.reinforce({idx_of[id(m)] for m in surfacing if id(m) in idx_of}, received, tick)
    repairs, _rec2 = _cue_pairs(store, tick, cue)
    repairs.sort(key=_kernel_sort_key, reverse=True)
    _mark_surfacing(repairs, store)
    return jsonify({"cue": cue, "rows": [r for _m, r in repairs], "committed": True})


@app.route("/api/add", methods=["POST"])
def api_add():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    STATE["store"].add(text, t=STATE["tick"])
    if STATE.get("graph"):
        STATE["graph"].add_memory()  # incremental: keeps session edge wear
    else:
        STATE["graph"] = MemoryGraph(STATE["store"])
    return jsonify(_state_payload())


@app.route("/api/tick", methods=["POST"])
def api_tick():
    data = request.get_json(force=True, silent=True) or {}
    tick = _finite_positive(data.get("value"), lo=0.0, hi=100000.0)
    if tick is None:
        return jsonify({"error": "value must be a number"}), 400
    STATE["tick"] = tick
    return jsonify(_state_payload())


@app.route("/api/set_strength", methods=["POST"])
def api_set_strength():
    """What-if lever: set every loaded memory's strength (g) to one value so
    you can drag it and watch the decay curve move. Overrides each memory's
    own recorded strength for exploration - reloading the agent (switch or
    load_sim) restores the real values from disk."""
    data = request.get_json(force=True, silent=True) or {}
    g = _finite_positive(data.get("value"))
    if g is None:
        return jsonify({"error": "value must be a positive number"}), 400
    store = STATE["store"]
    store.g0 = g
    for mem in store.memories:
        mem.g = g
    return jsonify(_state_payload())


@app.route("/api/set_topk", methods=["POST"])
def api_set_topk():
    """How many of the highest-scoring memories surface into the prompt.
    In a live run this is fixed per agent (5); here it's a knob so you can
    see how a tighter or looser head changes what makes the cut."""
    data = request.get_json(force=True, silent=True) or {}
    k = _finite_positive(data.get("value"), lo=1, hi=20)
    if k is None:
        return jsonify({"error": "value must be a number from 1 to 20"}), 400
    STATE["store"].top_k = int(k)
    return jsonify(_state_payload())


@app.route("/api/switch", methods=["POST"])
def api_switch():
    data = request.get_json(force=True, silent=True) or {}
    tag = data.get("tag")
    if not STATE["stores"] or tag not in STATE["stores"]:
        return jsonify({"error": "no such agent"}), 400
    # Reload from disk so the strength-dial what-if doesn't stick to an
    # agent you navigate away from and back to.
    if STATE.get("experiment_dir"):
        _load_experiment(STATE["experiment_dir"])
    if tag not in STATE["stores"]:
        return jsonify({"error": "no such agent"}), 400
    STATE["tag"] = tag
    STATE["store"] = STATE["stores"][tag]
    STATE["owner"] = STATE.get("names", {}).get(tag, tag)
    STATE["tick"] = latest_tick_in(STATE["store"])
    STATE["graph"] = MemoryGraph(STATE["store"])
    return jsonify(_state_payload())


@app.route("/api/load_sim", methods=["POST"])
def api_load_sim():
    """Address a run by its path (stable) rather than its position in a
    list that re-sorts by mtime whenever a run is actively writing; the
    index is only a fallback for old clients."""
    data = request.get_json(force=True, silent=True) or {}
    experiments = _experiments()
    target = None
    path = data.get("path")
    if path:
        target = next((d for d in experiments if str(d) == path), None)
        if target is None:
            return jsonify({"error": "that run is no longer on disk"}), 400
    if target is None and "index" in data:
        try:
            idx = int(data["index"])
        except (TypeError, ValueError):
            return jsonify({"error": "bad index"}), 400
        if 0 <= idx < len(experiments):
            target = experiments[idx]
    if target is None:
        return jsonify({"error": "no such sim"}), 400
    if not _load_experiment(target):
        return jsonify({"error": "run has no readable dumps"}), 400
    return jsonify(_state_payload())


@app.route("/api/load_preset", methods=["POST"])
def api_load_preset():
    data = request.get_json(force=True, silent=True) or {}
    g0 = _finite_positive(data.get("g0", DEFAULT_G0))
    if g0 is None:
        return jsonify({"error": "g0 must be a positive number"}), 400
    _load_preset(g0)
    return jsonify(_state_payload())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Explicit agent_logs directory to inspect")
    parser.add_argument("--port", type=int, default=5057)
    args = parser.parse_args()
    if args.run_dir:
        directory = args.run_dir.resolve()
        if not list(directory.glob("*_hou_memory.json")):
            parser.error("--run-dir must contain memory dumps")
        EXTRA_EXPERIMENTS.append(directory)
    _init_state()
    print(f"\nSimulation audit: http://127.0.0.1:{args.port}/audit\n")
    # threaded=False: STATE is a plain dict shared by every request; the
    # console is single-user and the client serializes its calls anyway
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)
