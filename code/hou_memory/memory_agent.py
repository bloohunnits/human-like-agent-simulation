"""Drop-in replacement for TerraLingua's LLMAgent that swaps the built-in
memory for independently aging associative retrieval. The Hou kernel remains
the direct pathway and a configurable comparison condition.

Nothing in TerraLingua's own code is edited. run_hou_experiment.py rebinds
the `LLMAgent` name inside core.experiment.runner's namespace to this class
before the runner constructs any agents.
"""

from __future__ import annotations

import json
import os
import sys
import copy
from pathlib import Path

_TERRALINGUA_ROOT = Path(__file__).resolve().parent.parent / "terralingua"
if str(_TERRALINGUA_ROOT) not in sys.path:
    sys.path.insert(0, str(_TERRALINGUA_ROOT))

from core.agents.llm_agent import LLMAgent  # noqa: E402

from embedder import SBERTEmbedder  # noqa: E402
from hou_memory import HouMemoryStore  # noqa: E402
from associative_memory import AssociativeMemory, MemoryConfig  # noqa: E402


class HouMemoryAgent(LLMAgent):
    # One SBERT model shared by every agent in the process - loading it
    # per-agent would multiply startup cost by agent count for no reason.
    _shared_embedder: SBERTEmbedder | None = None
    memory_config = MemoryConfig()

    # The plan cap. TerraLingua's 150-token scratchpad doubled as working
    # plan AND long-term memory; we scope the field to intention only
    # (docs/INTEGRATION.md finding 4) and a 60-token cap forces triage so
    # it can't quietly become a decay-proof life archive again.
    hou_plan_tokens = 60

    PLAN_CONTRACT = """- Plan (the internal_memory field):
    - Use the internal_memory field for your PLAN: your current intention, what you are doing now and next.
    - Keep it under 60 tokens. Keep it unchanged when nothing has changed; rewrite it when your intention changes.
    - Never record past events in it. What has happened to you is remembered for you automatically and will come to mind when relevant."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.memory_config
        self.hou_store = HouMemoryStore(
            embed_fn=self._embedder(), top_k=cfg.top_k, g0=cfg.g0,
            recall_threshold=cfg.threshold, time_scale=cfg.time_scale,
        )
        self.memory_engine = AssociativeMemory(self.hou_store, cfg, self._count_tokens)
        self._completed_tick = None
        self._completed_action = None
        self.current_plan = ""
        self.use_internal_memory = True
        self.internal_memory_size = self.hou_plan_tokens
        self._rescope_scratchpad_to_plan()

    def _rescope_scratchpad_to_plan(self) -> None:
        """Rewrite the rendered system prompt so the internal_memory field is
        a plan, not a diary. String surgery on TerraLingua's rendered
        template - their files stay untouched. Tolerant of the block being
        absent (custom system prompts pass through unchanged)."""
        sp = self.system_prompt
        sp = sp.replace(
            "Your INTERNAL MEMORY from the previous timestep",
            "Your PLAN, plus any memories that come to mind",
        )
        start = sp.find("- Internal memory :")
        if start != -1:
            ends = [i for m in ("- Artifacts", "- Inventory", "** Final remarks")
                    if (i := sp.find(m, start + 1)) != -1]
            end = min(ends) if ends else len(sp)
            sp = sp[:start] + self.PLAN_CONTRACT + "\n" + sp[end:]
        self.system_prompt = sp

    def _make_prompt(self, formatted_obs, available_actions, internal_memory, info):
        """Relabel the per-step template's memory slot and reply-format hint
        to match the plan contract, again without editing TerraLingua."""
        # The upstream history window otherwise bypasses memory retrieval.
        history = self.history
        count = self.memory_engine.config.working_history
        self.history = history[-count:] if count else []
        try:
            prompt = super()._make_prompt(
                formatted_obs=formatted_obs, available_actions=available_actions,
                internal_memory=internal_memory, info=info,
            )
        finally:
            self.history = history
        prompt = prompt.replace(
            "Previous INTERNAL MEMORY:", "YOUR PLAN AND WHAT COMES TO MIND:"
        )
        prompt = prompt.replace(
            'internal_memory: "<internal memory object containing things you wish to remember in the next turn. Limited to 600 tokens. Keep it concise.>"',
            'internal_memory: "<your PLAN: current intention, what you are doing now and next. Under 60 tokens. Never past events.>"',
        )
        return prompt

    @classmethod
    def _embedder(cls) -> SBERTEmbedder:
        if cls._shared_embedder is None:
            cls._shared_embedder = SBERTEmbedder()
        return cls._shared_embedder

    def select_action(
        self, obs, available_actions, reward, info, time, chat_params, client, max_attempts=5
    ):
        # Upstream's logger rewrites observation keys in-place for JSON.
        # Keep the caller's live environment observation untouched.
        obs = copy.deepcopy(obs)
        formatted_obs = self._format_observation(obs)
        query_text = self._context_query_text(formatted_obs, info)
        cue_entities = self._extract_entities(obs, {})
        event_id = f"{self.agent_tag}:retrieval:{time}"
        preview = self.memory_engine.preview(query_text, time, cue_entities, event_id)
        trace = self.memory_engine.commit(preview)
        if self._completed_tick == time:
            return self._completed_action
        # A retry uses the original pre-update selection, not newly boosted scores.
        by_id = {m.memory_id: m for m in self.hou_store.memories}
        rows = {r["id"]: r for r in trace["rows"]}
        surfaced = [(by_id[i], rows[i]["relevance"], rows[i]["score"]) for i in trace["selected"]]
        injected_slot = self._render_memory_slot(surfaced)
        self.internal_memory = injected_slot

        action = super().select_action(
            obs=obs,
            available_actions=available_actions,
            reward=reward,
            info=info,
            time=time,
            chat_params=chat_params,
            client=client,
            max_attempts=max_attempts,
        )

        # After a successful parse, self.internal_memory holds what the LLM
        # wrote in the internal_memory field - under the plan contract, its
        # current plan. When every parse attempt failed, the base class
        # leaves the slot exactly as we injected it, and capturing THAT
        # would nest surfaced memories into the plan forever - so only
        # accept a value that actually changed. (The per-tick agent log also
        # records it, so churn and contamination are measurable post-run.)
        raw = str(self.internal_memory or "").strip()
        if raw and raw != injected_slot.strip():
            # Small models sometimes echo the slot label back; strip it.
            if raw.lower().startswith("current plan:"):
                raw = raw[len("current plan:"):].strip()
            self.current_plan = self.internal_memory_encoder.decode(
                self.internal_memory_encoder.encode(raw, disallowed_special=())[:self.hou_plan_tokens])

        entities = self._extract_entities(obs, action)
        self._write_memory(obs, action, time, entities, info)
        self._completed_tick = time
        self._completed_action = action
        # Bind the prompt/action to this retrieval, avoiding ambiguous joins
        # against old same-tick log records after a checkpoint replay.
        trace["agent_step"] = self.logger.data_dict[str(time)]
        with open(self._audit_path(), "a") as f:
            f.write(json.dumps(trace, allow_nan=False) + "\n")
        self._dump_hou_memory()  # written every tick, not just at close() - a
        # killed/crashed run still leaves a real, loadable snapshot for explore.py
        return action

    def set_state_ckpt(self, state_ckpt: dict):
        """Restore memory from the SAME checkpoint as the world, never a newer dump."""
        super().set_state_ckpt(state_ckpt)
        payload = state_ckpt.get("associative_memory")
        if payload is None:
            raise ValueError("This legacy checkpoint has no synchronized memory state. Start a new experiment; the old dump remains viewable.")
        self.hou_store = HouMemoryStore.from_dicts(payload["memories"], self._embedder())
        self.memory_engine = AssociativeMemory.from_dict(self.hou_store, payload["associative"], self._count_tokens)
        self.current_plan = payload["plan"]
        self._completed_tick = payload.get("completed_tick")
        self._completed_action = payload.get("completed_action")
        self.use_internal_memory = True
        self._rescope_scratchpad_to_plan()

    def get_state_ckpt(self):
        state = super().get_state_ckpt()
        state["associative_memory"] = self._memory_payload()
        return state

    def _count_tokens(self, text):
        return len(self.internal_memory_encoder.encode(text, disallowed_special=()))

    def validate_internal_memory(self, internal_memory):
        raw = str(internal_memory)
        if "what comes to mind:" in raw.casefold():
            return self.current_plan  # reject an echoed retrieval slot before truncation
        tokens = self.internal_memory_encoder.encode(raw, disallowed_special=())
        return self.internal_memory_encoder.decode(tokens[:self.hou_plan_tokens])

    def _format_observation(self, obs):
        formatted = super()._format_observation(obs)
        places = obs.get("named_places", [])
        if places:
            formatted["observation"] += "\nVisible named places: " + ", ".join(places)
        return formatted

    def _audit_path(self):
        return Path(self.logger.log_dir) / f"{self.agent_tag}_memory_audit.jsonl"

    def close(self):
        self._dump_hou_memory()
        super().close()

    def _hou_dump_path(self) -> Path:
        return Path(self.logger.log_dir) / f"{self.agent_tag}_hou_memory.json"

    def _memory_payload(self):
        return {
            "agent_tag": self.agent_tag,
            "agent_name": self.agent_name,
            "g0": self.hou_store.g0,
            "top_k": self.hou_store.top_k,
            "recall_threshold": self.hou_store.recall_threshold,
            "time_scale": self.hou_store.time_scale,
            "plan": getattr(self, "current_plan", ""),
            "memories": self.hou_store.to_dicts(),
            "associative": self.memory_engine.to_dict(),
            "completed_tick": self._completed_tick,
            "completed_action": self._completed_action,
        }

    def _dump_hou_memory(self) -> None:
        payload = self._memory_payload()
        # Atomic: write to a temp file, then rename. A run killed mid-write
        # must never leave a truncated dump that the console can't load.
        path = self._hou_dump_path()
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, allow_nan=False)
        os.replace(tmp, path)

    def _context_query_text(self, formatted_obs: dict, info: dict | None) -> str:
        parts = [
            f"Observation: {formatted_obs['observation']}",
            f"Incoming messages: {formatted_obs['message']}",
        ]
        if info:
            parts.append(f"Additional info: {info}")
        return "\n".join(parts)

    def _render_surfaced(self, surfaced) -> str:
        if not surfaced:
            return "<nothing comes to mind>"
        lines = []
        for mem, r, p in surfaced:
            lines.append(self.memory_engine.memory_line(mem.text))
        return "".join(lines).rstrip("\n")

    def _render_memory_slot(self, surfaced) -> str:
        """What goes into the prompt's memory slot: the agent's own plan
        first (guaranteed, verbatim), then whatever the kernel surfaced."""
        plan = self.current_plan or "<no plan yet>"
        return f"Current plan: {plan}\n\nWhat comes to mind:\n{self._render_surfaced(surfaced)}"

    @staticmethod
    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    def _extract_entities(self, obs: dict, action: dict) -> dict:
        """Structured observed cues; actions add explicit episode tags only."""
        people, artifacts = set(), set()

        for contents in obs.get("observation", {}).values():
            for item in contents:
                item = str(item)
                if item.startswith("A("):  # artifact, rendered "A(text): name"
                    artifacts.add(item.split(": ", 1)[-1])
                elif item in ("X", "") or self._is_number(item):
                    continue  # off-map marker or a food energy value - not a named entity
                else:
                    # Color is an appearance, not a new entity identity.
                    people.add(item.rsplit("(", 1)[0] if item.endswith(")") else item)

        for item in obs.get("inventory", []):
            if str(item).startswith("A("):
                artifacts.add(str(item).split(": ", 1)[-1])

        for sender in obs.get("message", {}):
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

    def _describe_action(self, action: dict) -> str:
        act = action.get("action")
        p = action.get("params", {}) or {}
        if act == "move":
            d = p.get("direction", "somewhere")
            return "Stayed put." if d == "stay" else f"Moved {d}."
        if act == "give":
            return f"Gave {p.get('amount', '?')} energy to {p.get('target', 'someone')}."
        if act == "take":
            return f"Took {p.get('amount', '?')} energy from {p.get('target', 'someone')}."
        if act == "reproduce":
            return f"Reproduced, creating {p.get('name', 'a child')}."
        if act == "create_artifact":
            return f"Left an artifact called \"{p.get('name', 'untitled')}\"."
        if act == "pickup_artifact":
            return f"Picked up the artifact \"{p.get('name', 'one')}\"."
        return f"Did: {act}."

    def _write_memory(self, obs: dict, action: dict, time: int, entities: dict, info=None) -> None:
        """Readable English, not a coordinate dump: what I did, who/what was
        around, what was said."""
        # The world has not executed this turn yet. Do not record a requested
        # transfer, movement or creation as a confirmed outcome.
        parts = [f"Chose action {action.get('action')}: {json.dumps(action.get('params', {}), sort_keys=True)}."]

        observed = self._extract_entities(obs, {})
        others = observed["people"]
        arts = observed["artifacts"]
        if others:
            parts.append(f"Nearby: {', '.join(others)}.")
        if arts:
            parts.append(f"Artifacts here: {', '.join(arts)}.")
        if entities["places"]:
            parts.append(f"Visible places: {', '.join(entities['places'])}.")
        outcomes = {k: v for k, v in (info or {}).items() if k != "available_actions"}
        if outcomes:
            parts.append(f"Feedback from the preceding action: {outcomes}.")

        heard = obs.get("message", {})
        if heard:
            said = "; ".join(f"{who} said \"{msg}\"" for who, msg in heard.items())
            parts.append(f"Heard {said}.")
        if action.get("message"):
            parts.append(f'I said "{action["message"]}".')

        text = " ".join(parts)
        self.memory_engine.add(
            text, tick=time, meta={"entities": entities, "event": action.get("action"), "action_is_intended": True}
        )
