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

## The store

- `add_memory` stores a text with its time and metadata and returns its id.
- `recall_memories` returns the memories that come to mind for a cue, best first, and strengthens them unless told not to.
- `reinforce_memories` strengthens given memories without a recall.

Every call names the agent. The drivers, the embedder, the id source and the clock are injected.

## Driver conventions

- Rows go in and out as pandas DataFrames with fixed columns.
- Memory ids are UUID strings, unique across agents.
- A memory driver has `upsert_memories`, `recall_by_id`, `list_memories`, `list_edges` and `replace_edges`.
- An embedding driver has `upsert_memories`, `recall_by_id` and `recall_by_similarity`.
- `recall_by_id` takes an optional `agent_id` in both.

## Where the relational tags come from

From our TerraLingua wrapper, `HouMemoryAgent._extract_entities` in `code/hou_memory/memory_agent.py`. Base TerraLingua does not make them and no prompt or model does.

The wrapper reads them from the structured observation TerraLingua hands the agent each turn:

- people: names in the visible cells and senders of messages heard
- artifacts: notes in view or carried
- places: `named_places`, set by the launcher's landmarks file

A written memory also gets the target of the agent's own give or take, the name of a child and the name of a note it creates. Text is never searched for names.

The store receives tags from its caller: in `meta["entities"]` on `add_memory`, and as `entities` on `recall_memories` for what is in view.

## Is the existing memory code tangled with the simulation

No. `hou_memory.py` needs numpy only. `associative_memory.py` needs `hou_memory.py` only. Both stores reuse them unchanged. The simulation ties are all in `memory_agent.py`, which this folder does not import.

One tie was to the in-memory list: memory ids were list positions. The stores pass their own id source so ids are unique across agents.

## Limits

- An agent's memories are read from the driver on first use and then kept in the process. The drivers are written on every change.
- ACT-R recall traces are not stored. After a reload they are rebuilt from the creation time and the last recall.
- The engine's last retrieval is not stored, so its one-retrieval-per-tick guard starts over after a reload.
- Two memories of one agent at the same tick can reload in a different order. Their links then keep default strength.
- Calls are synchronous.

## Next step

The reverse proxy. Per TerraLingua request it would call `recall_memories` with the observation as the cue and the tags in view, add the result to the prompt, and after the model answers call `add_memory` with the turn's text and tags. The tag extraction has to move out of `memory_agent.py` for that.
