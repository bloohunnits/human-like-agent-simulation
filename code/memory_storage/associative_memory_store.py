"""The memory store ranked by the associative engine in code/hou_memory."""

from dataclasses import replace
from typing import Callable, Optional
from uuid import uuid4

import pandas as pd

from hou_memory.associative_memory import AssociativeMemory, MemoryConfig
from hou_memory.hou_memory import Memory
from memory_storage.multiagent_memory_store import AgentMemories, MultiAgentMemoryStore


class AssociativeMemoryStore(MultiAgentMemoryStore):
    """
    Ranks with similarity, decay and the links between memories
    Links are persisted as connection rows and reloaded with the memories
    """

    def __init__(
        self,
        memory_database,
        embedding_database,
        embed: Callable,
        config: Optional[MemoryConfig] = None,
        engine_class: type = AssociativeMemory,
        token_counter: Optional[Callable] = None,
        new_id: Callable = lambda: str(uuid4()),
        clock: Optional[Callable] = None,
        ):
        """
        Parameters:
            config: the engine's settings, including slots, cutoff and time scale
            engine_class: AssociativeMemory or a subclass with other formulas
            token_counter: counts the tokens of the recalled text against
                config.token_budget. The engine's tiktoken counter when omitted
        """
        self.config = config or MemoryConfig()
        self.engine_class = engine_class
        self.token_counter = token_counter
        self._engines = {}
        super().__init__(
            memory_database, embedding_database, embed, new_id, clock,
            top_k=self.config.top_k, recall_threshold=self.config.threshold,
            g0=self.config.g0, time_scale=self.config.time_scale,
        )

    def _engine(
        self,
        agent_id: str,
        store: AgentMemories,
        )-> AssociativeMemory:
        """ The agent's engine. Its links are rebuilt from the memories, then
        take their strength and clock from the stored connection rows """
        if agent_id not in self._engines:
            engine = self.engine_class(store, self.config, self.token_counter)
            for row in self.memory_database.list_edges(agent_id).to_dict(orient="records"):
                edge = engine.edges.get(str(row["edge_id"]))
                if edge is not None:
                    edge.weight = float(row["weight"])
                    edge.retention = float(row["retention"])
                    edge.last_reinforced = float(row["last_reinforced"])
            self._engines[agent_id] = engine
        return self._engines[agent_id]

    def _write(
        self,
        agent_id: str,
        store: AgentMemories,
        content: str,
        tick: float,
        meta: dict,
        )-> Memory:
        engine = self._engine(agent_id, store)
        known = set(engine.edges)
        memory = engine.add(content, tick, meta)
        self._save_edges(agent_id, [edge for edge_id, edge in engine.edges.items() if edge_id not in known])
        return memory

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
        engine = self._engine(agent_id, store)
        if top_k is not None and top_k != engine.config.top_k:
            engine.config = replace(engine.config, top_k=top_k)
        by_id = {m.memory_id: m for m in store.memories}
        result = engine.preview(
            cue, tick, entities, event_id=self.new_id(),
            r_by_id={memory_id: relevance.get(memory_id, 0.0) for memory_id in by_id},
        )
        scores = {score.id: score for score in result.scores}
        if reinforce:
            event = engine.commit(result)
            self._save(agent_id, [by_id[update["id"]] for update in event["memory_updates"]])
            self._save_edges(agent_id, [engine.edges[update["id"]] for update in event["edge_updates"]])
        return [(by_id[i], scores[i].relevance, scores[i].score) for i in result.selected]

    def _save_edges(
        self,
        agent_id: str,
        edges: list,
        )-> None:
        """ Write the current state of these links through the driver """
        if edges:
            self.memory_database.replace_edges(pd.DataFrame([{
                "edge_id": edge.id,
                "agent_id": agent_id,
                "node_a": edge.a,
                "node_b": edge.b,
                "kind": edge.kind,
                "created": edge.created,
                "last_reinforced": edge.last_reinforced,
                "weight": edge.weight,
                "retention": edge.retention,
            } for edge in edges]))
