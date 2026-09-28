"""Interactive REPL over the bare Hou kernel. Real SBERT embeddings, real
Hou math, zero LLM calls at query time. Type a command, see the numbers
immediately, type another.

    python explore.py              # auto-loads the most recent sim run, if any
    python explore.py --preset     # force the small synthetic story instead
    python explore.py --sim <dir>  # load a specific experiment's agent_logs dir

Every `run_hou_experiment.py` run now dumps each agent's full memory store
(every memory, its embedding, g, and recall history) to
<log_dir>/agent_logs/<agent_tag>_hou_memory.json every tick. This tool reads
that file back - no sim needs to be running, and a killed/crashed run still
leaves a loadable snapshot.

Commands:
  list              every memory's raw standing right now (g, age, p at r=1)
  cue <text>        score ALL memories against a cue at the current tick.
                    Read-only - nothing is reinforced. Shows exactly why
                    each one would or wouldn't surface.
  recall <text>     same scoring, but commits it: whatever surfaces gets
                    its clock reset and g bumped, same as a real recall.
  add <text>        write a new memory at the current tick
  tick <n>          move "now" to tick n (time passes, nothing else happens)
  sims              list every sim run found on disk, newest first
  load <n or path>  load experiment #n from `sims`, or an explicit dir/file
  switch <tag>      switch to a different agent within the loaded experiment
  preset [g0]       drop the real data and load the small synthetic story,
                    optionally with a different g0 (paper-native default 1;
                    time is calibrated by the story's fixed time_scale 25)
  help              show this
  quit

Columns: r = relevance (cosine sim to your cue). g = strength (consolidation).
age = ticks since this memory was last recalled. p = recall probability -
the actual Hou output, what decides if this memory would make it into an
agent's prompt.
"""

import json
import sys
from pathlib import Path

from embedder import SBERTEmbedder
from hou_memory import HouMemoryStore, recall_probability

DEFAULT_G0 = 1.0  # paper-native; time is calibrated via time_scale instead
TERRALINGUA_ROOT = Path(__file__).resolve().parent.parent / "terralingua"


def _ent(people=(), places=(), artifacts=()):
    return {"entities": {"people": list(people), "places": list(places),
                         "artifacts": list(artifacts)}, "event": None}


def preset_story(embedder, g0: float) -> HouMemoryStore:
    """A tiny hand-made story, tagged with entities the way sim memories are,
    so the example previews the feature (and the places field, which sim
    memories can't fill because TerraLingua has no named places)."""
    store = HouMemoryStore(embed_fn=embedder, top_k=3, recall_threshold=0.05, g0=g0,
                           time_scale=25.0)
    store.add("Met Omar at the well. He showed me the berry patch nearby.", t=0,
              meta=_ent(people=["Omar"], places=["the well", "the berry patch"]))
    store.add("Picked berries at the patch.", t=1, meta=_ent(places=["the berry patch"]))
    store.add("Picked berries at the patch.", t=2, meta=_ent(places=["the berry patch"]))
    store.add("Picked berries at the patch.", t=3, meta=_ent(places=["the berry patch"]))
    store.add("Ate strange mushrooms found in the woods.", t=6,
              meta=_ent(places=["the woods"]))
    store.add("Was sick all night after dinner.", t=6, meta=_ent())
    return store


def find_experiments(root: Path) -> list[Path]:
    """Every agent_logs dir that has at least one *_hou_memory.json dump,
    newest first (by the newest file inside it)."""
    groups: dict[Path, float] = {}
    if not root.exists():
        return []
    for f in root.rglob("*_hou_memory.json"):
        mtime = f.stat().st_mtime
        groups[f.parent] = max(groups.get(f.parent, 0.0), mtime)
    return sorted(groups, key=lambda d: groups[d], reverse=True)


def load_experiment(dir_path: Path, embedder) -> dict[str, HouMemoryStore]:
    stores = {}
    for f in sorted(dir_path.glob("*_hou_memory.json")):
        try:
            payload = json.load(open(f))
            stores[payload["agent_tag"]] = HouMemoryStore.from_dicts(
                payload["memories"],
                embed_fn=embedder,
                top_k=payload["top_k"],
                recall_threshold=payload["recall_threshold"],
                g0=payload["g0"],
                time_scale=payload.get("time_scale", 1.0),
            )
        except Exception as e:
            # A truncated or malformed dump (e.g. a run killed mid-write on an
            # old non-atomic writer) skips that one agent, never the whole app.
            print(f"skipping unreadable dump {f.name}: {e}")
    return stores


def latest_tick_in(store: HouMemoryStore) -> float:
    if not store.memories:
        return 0.0
    return max(max(m.t_created, m.t_last_recalled) for m in store.memories)


def print_row(r, p, g, age, created, text, surfaced):
    marker = "-> SURFACE" if surfaced else "          "
    r_str = f"{r:.2f}" if r is not None else "  - "
    print(f"  {marker}  r={r_str}  p={p:.3f}  g={g:6.2f}  age={age:5.1f}  born@t={created:<4}  {text[:90]}")


def cmd_list(store, current_tick):
    print(f"\n-- all memories, standing as of tick {current_tick} (no query) --")
    if not store.memories:
        print("  <empty>")
        return
    for mem in store.memories:
        age = current_tick - mem.t_last_recalled
        p_ceiling = recall_probability(1.0, max(age, 0.0) / store.time_scale, mem.g)
        print_row(None, p_ceiling, mem.g, age, mem.t_created, mem.text, False)
    print("  (p here is the ceiling at r=1 - the best this memory could ever score right now)")


def cmd_score(store, cue, current_tick, commit):
    query_embedding = store._embed_fn(cue)
    scored = store.score(query_embedding, current_tick)
    scored.sort(key=lambda triple: triple[2], reverse=True)
    if commit:
        surfaced = store.recall(cue, current_tick)
        surfaced_set = {id(mem) for mem, _r, _p in surfaced}
    else:
        would_surface = [t for t in scored if t[2] > store.recall_threshold][: store.top_k]
        surfaced_set = {id(mem) for mem, _r, _p in would_surface}

    verb = "RECALL (committed)" if commit else "CUE (read-only)"
    print(f'\n-- {verb}: "{cue}"  @ tick {current_tick}  ({len(store.memories)} memories total, showing all) --')
    for mem, r, p in scored:
        age = current_tick - mem.t_last_recalled
        print_row(r, p, mem.g, age, mem.t_created, mem.text, id(mem) in surfaced_set)
    if not surfaced_set:
        print("  (nothing crossed the recall_threshold - this cue surfaces nothing)")
    if commit:
        print("  (g and clock updated for the SURFACE rows above)")


def print_sims(experiments: list[Path]):
    if not experiments:
        print("  no sim runs found under", TERRALINGUA_ROOT)
        return
    for i, d in enumerate(experiments):
        n_agents = len(list(d.glob("*_hou_memory.json")))
        print(f"  [{i}] {d}  ({n_agents} agent(s))")


def load_into(state: dict, dir_path: Path, embedder):
    stores = load_experiment(dir_path, embedder)
    if not stores:
        print(f"  no *_hou_memory.json files in {dir_path}")
        return
    state["experiment_dir"] = dir_path
    state["stores"] = stores
    tag = sorted(stores)[0]
    state["current_tag"] = tag
    state["store"] = stores[tag]
    state["current_tick"] = latest_tick_in(state["store"])
    print(f"  loaded {dir_path}")
    print(f"  agents: {sorted(stores)} (showing '{tag}' - use 'switch <tag>' to change)")
    print(f"  {len(state['store'].memories)} memories, now at tick {state['current_tick']}")


def main():
    args = sys.argv[1:]
    force_preset = "--preset" in args
    sim_arg = None
    if "--sim" in args:
        sim_arg = args[args.index("--sim") + 1]

    print(__doc__)
    embedder = SBERTEmbedder()
    state = {}

    experiments = find_experiments(TERRALINGUA_ROOT)

    if sim_arg:
        load_into(state, Path(sim_arg), embedder)
    elif force_preset or not experiments:
        if not experiments and not force_preset:
            print("\n  no real sim data found yet - falling back to the synthetic preset.")
            print("  run one with: python run_hou_experiment.py --init_agents 2 --max_ts 20 ...\n")
        state["store"] = preset_story(embedder, DEFAULT_G0)
        state["current_tick"] = 6.0
        state["experiment_dir"] = None
    else:
        print(f"\nFound {len(experiments)} sim run(s):")
        print_sims(experiments)
        load_into(state, experiments[0], embedder)

    print("\nType 'help' for commands.\n")

    while True:
        try:
            line = input(f"[t={state['current_tick']}] > ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        store = state["store"]
        current_tick = state["current_tick"]

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd == "list":
            cmd_list(store, current_tick)
        elif cmd == "cue":
            if not rest:
                print("  usage: cue <text>")
                continue
            cmd_score(store, rest, current_tick, commit=False)
        elif cmd == "recall":
            if not rest:
                print("  usage: recall <text>")
                continue
            cmd_score(store, rest, current_tick, commit=True)
        elif cmd == "add":
            if not rest:
                print("  usage: add <text>")
                continue
            store.add(rest, t=current_tick)
            print(f"  added at tick {current_tick}: {rest}")
        elif cmd == "tick":
            try:
                state["current_tick"] = float(rest)
                print(f"  now at tick {state['current_tick']}")
            except ValueError:
                print("  usage: tick <number>")
        elif cmd == "sims":
            experiments = find_experiments(TERRALINGUA_ROOT)
            print_sims(experiments)
        elif cmd == "load":
            if not rest:
                print("  usage: load <n from `sims`, or a directory path>")
                continue
            if rest.isdigit():
                experiments = find_experiments(TERRALINGUA_ROOT)
                idx = int(rest)
                if idx >= len(experiments):
                    print(f"  no such index (found {len(experiments)} sim(s), try 'sims')")
                    continue
                load_into(state, experiments[idx], embedder)
            else:
                load_into(state, Path(rest), embedder)
        elif cmd == "switch":
            stores = state.get("stores")
            if not stores:
                print("  no multi-agent experiment loaded (use 'load' first)")
                continue
            if rest not in stores:
                print(f"  unknown tag {rest!r}, available: {sorted(stores)}")
                continue
            state["current_tag"] = rest
            state["store"] = stores[rest]
            state["current_tick"] = latest_tick_in(state["store"])
            print(f"  switched to '{rest}', {len(state['store'].memories)} memories, tick {state['current_tick']}")
        elif cmd == "preset":
            g0 = float(rest) if rest else DEFAULT_G0
            state["store"] = preset_story(embedder, g0)
            state["current_tick"] = 6.0
            state["experiment_dir"] = None
            print(f"  loaded synthetic preset: {len(state['store'].memories)} memories, g0={g0}, tick=6")
        else:
            print(f"  unknown command: {cmd!r} (try 'help')")


if __name__ == "__main__":
    main()
