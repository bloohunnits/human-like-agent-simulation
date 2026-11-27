"""Check AssociativeMemoryStore against recorded worlds, with no model and no embedder.

Every recorded turn of every agent goes through the store on the in-memory
memory driver. The cue similarities are the ones the run recorded, so nothing
is embedded. The store must pick what the run picked on every turn.

Run from code/: python -m memory_storage.replay_check <run> [<run> ...]
A run is a logs/<exp>/ folder or its name under code/terralingua/logs_dev/logs.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import count, repeat
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

import pandas as pd

from hou_memory.associative_memory import MemoryConfig
from hou_memory.replay import entities_of, load_agent, recorded_engine, run_path
from memory_storage.associative_memory_store import AssociativeMemoryStore
from memory_storage.in_memory_database import InMemoryMemoryDatabase


class RecordedSimilarity:
    """Stands in for the embedding driver with the similarities of one recorded turn."""

    def __init__(
        self,
        ):
        self.relevance = {}

    def upsert_memories(
        self,
        memories: pd.DataFrame,
        )-> int:
        """ Nothing is embedded in a replay """
        return len(memories)

    def recall_by_similarity(
        self,
        embedding,
        k: int = 5,
        agent_id: Optional[str] = None,
        )-> pd.DataFrame:
        """ The recorded similarity of every memory to this turn's cue """
        return pd.DataFrame({"memory_id": list(self.relevance), "relevance": list(self.relevance.values())})


def rising_ids(
    )-> Callable:
    """ UUID strings in rising order, so equal scores break as they did in the run """
    numbers = count(1)
    return lambda: str(UUID(int=next(numbers)))


def check_agent(
    agent_logs: Path,
    tag: str,
    )-> tuple:
    """
    Replay one agent through the store

    Returns the turns replayed, the turns whose picks match the recording and
    the first turn that differs as (tick, recorded, picked)
    """
    lines, memories = load_agent(agent_logs, tag)
    if not lines:
        return 0, 0, None
    similarity = RecordedSimilarity()
    store = AssociativeMemoryStore(
        InMemoryMemoryDatabase(), similarity, embed=lambda text: [1.0],
        config=MemoryConfig(**lines[-1]["config"]), engine_class=recorded_engine(agent_logs.parent),
        new_id=rising_ids(),
    )
    recorded, stored = {}, {}
    pending = iter(memories)
    upcoming = next(pending, None)
    matched, first = 0, None
    for line in lines:
        # The live agent scored the store before writing that tick's episode.
        while upcoming is not None and upcoming["t_created"] < line["tick"]:
            memory_id = store.add_memory(tag, upcoming["text"], upcoming["t_created"], upcoming.get("meta"))
            recorded[memory_id] = upcoming["memory_id"]
            stored[upcoming["memory_id"]] = memory_id
            upcoming = next(pending, None)
        similarity.relevance = {stored[row["id"]]: row["relevance"] for row in line["rows"]}
        picked = store.recall_memories(tag, line["cue"], line["tick"], entities=entities_of(line["sources"]))
        picked = [recorded[memory_id] for memory_id in picked["memory_id"]]
        if picked == line["selected"]:
            matched += 1
        elif first is None:
            first = (line["tick"], line["selected"], picked)
    return len(lines), matched, first


def main(
    )-> None:
    if not sys.argv[1:]:
        raise SystemExit(__doc__)
    differs = False
    with ProcessPoolExecutor() as pool:
        for run in sys.argv[1:]:
            agent_logs = run_path(run) / "agent_logs"
            tags = sorted(p.name.removesuffix("_memory_audit.jsonl") for p in agent_logs.glob("*_memory_audit.jsonl"))
            turns = matched = 0
            for tag, (seen, same, first) in zip(tags, pool.map(check_agent, repeat(agent_logs), tags)):
                turns, matched = turns + seen, matched + same
                if first:
                    print(f"  {tag} first differs at tick {first[0]}: the run picked {first[1]}, the store {first[2]}")
            print(f"{agent_logs.parent.name}: the store picked what the run picked on {matched} of {turns} turns")
            differs = differs or matched != turns
    raise SystemExit(1 if differs else 0)


if __name__ == "__main__":
    main()
