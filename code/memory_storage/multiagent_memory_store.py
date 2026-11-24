"""Hou-level memory for many agents, apart from the simulation and the LLM.

The drivers, the embedder, the id source and the clock are injected, so the
store runs and is tested with no database, no model and no @patch.
"""

import time
from random import getrandbits
from typing import Callable, Optional
from uuid import UUID

import pandas as pd

from hou_memory.hou_memory import HouMemoryStore, Memory, recall_probability


RECALL_COLUMNS = ["memory_id", "content", "t_created", "relevance", "score", "meta"]
_last_id = 0


def ordered_id(
    )-> str:
    """
    A new UUID string that sorts after every id this process made before

    Memories written at one tick are read back in id order, so they reload in
    the order they were written and their links keep their strength
    """
    global _last_id
    _last_id = max(_last_id + 1, time.time_ns())
    return str(UUID(int=(_last_id << 64) | getrandbits(64)))


class AgentMemories(HouMemoryStore):
    """One agent's memories, with ids from the injected source so that they
    are unique across agents. Vectors live in the embedding driver."""

    new_id: Callable = None

    def add(
        self,
        text: str,
        t: float,
        meta: Optional[dict] = None,
        )-> Memory:
        memory = super().add(text, t, meta)
        memory.memory_id = self.new_id()
        return memory


class MultiAgentMemoryStore:
    """
    Intermediary between agents and the memory databases
    Ranks by Hou's recall probability. A subclass changes the ranking
    """

    def __init__(
        self,
        memory_database,
        embedding_database,
        embed: Callable,
        new_id: Callable = ordered_id,
        clock: Optional[Callable] = None,
        top_k: int = 5,
        recall_threshold: float = 0.0,
        g0: float = 1.0,
        time_scale: float = 1.0,
        ):
        """
        Parameters:
            memory_database: driver for memory rows and connection rows
            embedding_database: driver for vectors and similarity search
            embed: turns a text into a vector
            new_id: gives a new UUID string for each memory. Ids must rise in
                the order they are made
            clock: gives the current time when a call passes no tick
            time_scale: time units per unit of Hou decay
        """
        self.memory_database = memory_database
        self.embedding_database = embedding_database
        self.embed = embed
        self.new_id = new_id
        self.clock = clock
        self.top_k = top_k
        self.recall_threshold = recall_threshold
        self.g0 = g0
        self.time_scale = time_scale
        self._agents = {}

    def add_memory(
        self,
        agent_id: str,
        content: str,
        tick: Optional[float] = None,
        meta: Optional[dict] = None,
        )-> str:
        """
        Store a new memory and return its id

        Parameters:
            tick: when it happened. The injected clock is used when omitted
            meta: free metadata. meta["entities"] holds the relational tags
        """
        agent_id, tick = str(agent_id), self._time(tick)
        memory = self._write(agent_id, self._memories(agent_id), content, tick, dict(meta or {}))
        self.embedding_database.upsert_memories(pd.DataFrame([{
            "memory_id": memory.memory_id,
            "agent_id": agent_id,
            "embedding": [float(value) for value in self.embed(content)],
        }]))
        self._save(agent_id, [memory])
        return memory.memory_id

    def recall_memories(
        self,
        agent_id: str,
        cue: str,
        tick: Optional[float] = None,
        top_k: Optional[int] = None,
        reinforce: bool = True,
        entities: Optional[dict] = None,
        )-> pd.DataFrame:
        """
        Return the memories that come to mind for a cue, best first

        Parameters:
            cue: text describing the present moment
            top_k: number of memories to retrieve. The store default when omitted
            reinforce: strengthen what was recalled, as the paper does
            entities: what is in view. Used by stores that rank with links
        """
        agent_id, tick = str(agent_id), self._time(tick)
        store = self._memories(agent_id)
        relevance = {}
        if store.memories:
            found = self.embedding_database.recall_by_similarity(
                self.embed(cue), k=len(store.memories), agent_id=agent_id)
            relevance = {str(i): max(0.0, float(r)) for i, r in zip(found["memory_id"], found["relevance"])}
        picked = self._rank(agent_id, store, cue, relevance, tick, top_k, reinforce, entities)
        return pd.DataFrame(
            [[m.memory_id, m.text, m.t_created, r, score, m.meta] for m, r, score in picked],
            columns=RECALL_COLUMNS,
        )

    def reinforce_memories(
        self,
        agent_id: str,
        memory_ids: list[str],
        tick: Optional[float] = None,
        )-> int:
        """ Strengthen the given memories as if recalled now; return how many """
        agent_id, tick = str(agent_id), self._time(tick)
        store = self._memories(agent_id)
        wanted = {str(memory_id) for memory_id in memory_ids}
        found = [m for m in store.memories if m.memory_id in wanted]
        for memory in found:
            store._reinforce(memory, tick)
        self._save(agent_id, found)
        return len(found)

    def _time(
        self,
        tick: Optional[float],
        )-> float:
        """ The given tick, or the injected clock's """
        if tick is not None:
            return float(tick)
        if self.clock is None:
            raise ValueError("Pass a tick or inject a clock")
        return float(self.clock())

    def _memories(
        self,
        agent_id: str,
        )-> AgentMemories:
        """ The agent's memories, read from the driver on first use. A driver
        orders by creation time only, so ties are put in id order here """
        if agent_id not in self._agents:
            rows = self.memory_database.list_memories(agent_id).to_dict(orient="records")
            rows.sort(key=lambda row: (float(row["t_created"]), str(row["memory_id"])))
            store = AgentMemories.from_dicts(
                [{
                    "text": row["content"],
                    "embedding": None,
                    "t_created": float(row["t_created"]),
                    "t_last_recalled": float(row["t_created"] if pd.isna(row["t_last_recalled"]) else row["t_last_recalled"]),
                    "g": float(row["g"]),
                    "recall_count": int(row["recall_count"]),
                    "meta": row["meta"] if isinstance(row["meta"], dict) else {},
                    "memory_id": str(row["memory_id"]),
                } for row in rows],
                embed_fn=lambda text: None, top_k=self.top_k, recall_threshold=self.recall_threshold,
                g0=self.g0, time_scale=self.time_scale,
            )
            store.new_id = self.new_id
            self._agents[agent_id] = store
        return self._agents[agent_id]

    def _write(
        self,
        agent_id: str,
        store: AgentMemories,
        content: str,
        tick: float,
        meta: dict,
        )-> Memory:
        """ Add one memory to the agent's state. A subclass also links it """
        return store.add(content, tick, meta)

    def _rank(
        self,
        agent_id: str,
        store: AgentMemories,
        cue: str,
        relevance: dict,
        tick: float,
        top_k: Optional[int],
        reinforce: bool,
        entities: Optional[dict],
        )-> list:
        """
        Pick the recalled memories as (memory, relevance, score), best first

        Parameters:
            relevance: memory id to cue similarity, from the embedding driver
        """
        scored = []
        for memory in store.memories:
            r = relevance.get(memory.memory_id, 0.0)
            elapsed = max(tick - memory.t_last_recalled, 0.0) / self.time_scale
            scored.append((memory, r, recall_probability(r, elapsed, memory.g)))
        scored.sort(key=lambda s: (-s[2], s[0].memory_id))
        picked = [s for s in scored if s[2] > self.recall_threshold][:self.top_k if top_k is None else top_k]
        if reinforce:
            for memory, _, _ in picked:
                store._reinforce(memory, tick)
            self._save(agent_id, [memory for memory, _, _ in picked])
        return picked

    def _save(
        self,
        agent_id: str,
        memories: list,
        )-> None:
        """ Write the current state of these memories through the driver """
        if memories:
            self.memory_database.upsert_memories(pd.DataFrame([{
                "memory_id": m.memory_id,
                "agent_id": agent_id,
                "content": m.text,
                "t_created": m.t_created,
                "t_last_recalled": m.t_last_recalled,
                "g": m.g,
                "recall_count": m.recall_count,
                "meta": m.meta,
            } for m in memories]))
