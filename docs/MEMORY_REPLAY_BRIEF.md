# Running the memory math without the sim

A brief for the replay bench: feed recorded runs back through the memory engine under any condition, with no TerraLingua, no embedder, and no API cost.

## The one code change

`AssociativeMemory.preview` takes an optional `r_by_id` dict. When it is given, the embed and cosine steps are skipped and those numbers are used as `relevance`. Everything after that line is unchanged. Build the store with a stub `embed_fn` that returns `None` so `add` works without a model. `commit` never touches embeddings.

## Where the numbers come from

Each recorded run folder under `code/terralingua/logs_dev/logs/<run>/agent_logs/` has, per agent:

- `<tag>_memory_audit.jsonl`: one line per tick. Use `tick`, `sources`, and `rows[].id` with `rows[].relevance`. That is r by id for that tick.
- `<tag>_hou_memory.json`: the episodes. Use `memories[].text`, `t_created`, `meta`. One episode per tick, in order.

Relevance is the cosine between the cue text and the episode text. It does not depend on the memory condition, so one recorded world drives every condition. Decay, graph contribution and selection are what change.

## The loop, per agent

```
for each audit line, in tick order:
    r = {row.id: row.relevance for row in line.rows}
    result = engine.preview(cue=None, tick=line.tick, entities=line.sources, r_by_id=r)
    engine.commit(result)
    engine.add(episode created at this tick)   # from the dump, same order the live agent used
```

Preview before add. At tick t the live agent scored the store before writing that tick's episode.

## What to compare

Run the loop once per condition (hou, learned, chained) on the same agent. Per tick record `selected`, `solo_selected`, and their difference, the rescued set.

Sanity check first: replaying the recorded condition must reproduce the recorded `selected` exactly. If it does not, the ordering or the config is off, and nothing else is trustworthy until it matches.

## Config

Build each condition's `MemoryConfig` from the `MEMORY_CONDITIONS` table and the run's `memory_config.json`. Keep `time_scale` 25 and `g0` 1.

## Scope

Pure functions and the replay loop only. No TerraLingua import, no embedder, no agent class. Output one JSON per run and condition. Figures come later.

## What this answers and what it does not

It answers whether associations put a past episode in the prompt when similarity and decay alone would not, on real histories. It cannot show whether the agent would have acted differently. The recorded agent chose its actions seeing the recorded condition's memories. That still needs the sim and the model.
