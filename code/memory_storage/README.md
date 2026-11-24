# Memory storage

Memory for many agents, apart from the simulation and the LLM. Nothing here imports TerraLingua or calls a model.

## Parts

| File | What it holds |
| --- | --- |
| `multiagent_memory_store.py` | `MultiAgentMemoryStore`, the base store. Ranks by Hou's recall probability. |
| `associative_memory_store.py` | `AssociativeMemoryStore`. Ranks with the associative engine and saves its links. |
| `memory_database.py` | PostgreSQL driver for memory rows and connection rows. |
| `embedding_database.py` | Qdrant driver for vectors and similarity search. |
| `in_memory_database.py` | The same two drivers in process memory, for tests and runs with no database. |
| `replay_check.py` | Replays recorded worlds through `AssociativeMemoryStore` and checks it picks what each run picked. No model, no embedder. |

## The store

- `add_memory` stores a text with its time and metadata and returns its id.
- `recall_memories` returns the memories that come to mind for a cue, best first, and strengthens them unless told not to.
- `reinforce_memories` strengthens given memories without a recall.

Every call names the agent. The drivers, the embedder, the id source and the clock are injected.

## Driver conventions

- Rows go in and out as pandas DataFrames with fixed columns.
- Memory ids are UUID strings, unique across agents. The default id source makes them rise in the order made.
- A memory driver has `upsert_memories`, `recall_by_id`, `list_memories`, `list_edges` and `replace_edges`.
- An embedding driver has `upsert_memories`, `recall_by_id` and `recall_by_similarity`.
- `recall_by_id` takes an optional `agent_id` in both.

## Where the relational tags come from

From `extract_entities` in `code/hou_memory/entity_tags.py`, which our TerraLingua wrapper calls each turn. Base TerraLingua does not make them and no prompt or model does. The module imports nothing from the simulation, so a proxy can call it too.

It reads them from the structured observation TerraLingua hands the agent each turn:

- people: names in the visible cells and senders of messages heard
- artifacts: notes in view or carried
- places: `named_places`, set by the launcher's landmarks file

A written memory also gets the target of the agent's own give or take, the name of a child and the name of a note it creates. Text is never searched for names.

The store receives tags from its caller: in `meta["entities"]` on `add_memory`, and as `entities` on `recall_memories` for what is in view.

## Is the existing memory code tangled with the simulation

No. `hou_memory.py` needs numpy only. `associative_memory.py` needs `hou_memory.py` only. Both stores reuse them unchanged. The simulation ties are all in `memory_agent.py`, which this folder does not import.

`associative_memory.py`, `variants.py`, `replay.py` and `entity_tags.py` load as `hou_memory.<name>` from `code/` and on their own from `code/hou_memory`.

One tie was to the in-memory list: memory ids were list positions. The stores pass their own id source so ids are unique across agents.

## Checked against recorded worlds

On 2026-11-24 `replay_check.py` matched every turn of four recorded worlds: `newengine_1` 834 of 834, `newengine_2` 1380 of 1380, `newengine_3` 1093 of 1093 and `paper_mini_300` 1910 of 1910.

## Limits

- An agent's memories are read from the driver on first use and then kept in the process. The drivers are written on every change.
- ACT-R recall traces are not stored. After a reload they are rebuilt from the creation time and the last recall.
- The engine's last retrieval is not stored, so its one-retrieval-per-tick guard starts over after a reload.
- Memories of one tick reload in id order. An injected id source must make ids that rise in the order made, or links between same-tick memories lose their strength on reload.
- Calls are synchronous.

## Next step

The reverse proxy. Per TerraLingua request it would call `extract_entities` on the observation, call `recall_memories` with the observation as the cue and those tags, add the result to the prompt, and after the model answers call `add_memory` with the turn's text and the tags of the observation and the action.
