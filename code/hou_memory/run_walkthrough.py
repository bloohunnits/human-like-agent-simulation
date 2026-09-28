"""The smallest possible run of the Hou kernel: one command, six memories,
time measured in raw sim ticks with NO time_scale, which is exactly the
degenerate configuration this script exists to demonstrate (the live agent
now runs g0=1 with time_scale=25; see INTEGRATION.md). No LLM, no
TerraLingua - just the formula and SBERT.

    python run_walkthrough.py

Ticks vs days: demo_ada_walkthrough.py found that g0=1 collapses to zero
almost immediately when t is counted in days. This script deliberately
stays in ticks to check whether the same default is sane at the
granularity the live agent actually uses, before touching any tick-to-day
conversion.
"""

from embedder import SBERTEmbedder
from hou_memory import HouMemoryStore

CHECK_AT_TICK = 15
DEAD_THRESHOLD = 0.01  # p_at_r1 below this = "not coming back without a strong cue"

CUES = ["Ben", "the berry patch", "mushrooms", "feeling sick"]


def build_store() -> HouMemoryStore:
    store = HouMemoryStore(embed_fn=SBERTEmbedder(), top_k=3, recall_threshold=0.05)

    store.add("Ada meets Ben at the well. He shows her the berry patch nearby.", t=0)
    store.add("Ada picks berries at the patch.", t=1)
    store.add("Ada picks berries at the patch.", t=2)
    store.add("Ada picks berries at the patch.", t=3)
    store.add("Ada eats strange mushrooms she found in the woods.", t=6)
    store.add("Ada is sick all night after her dinner.", t=6)

    return store


def main():
    store = build_store()

    print(f"=== Cues at tick {CHECK_AT_TICK} (read-only, no reinforcement) ===\n")
    for cue in CUES:
        query_embedding = store._embed_fn(cue)
        scored = sorted(
            store.score(query_embedding, t=CHECK_AT_TICK), key=lambda t: t[2], reverse=True
        )
        top = [triple for triple in scored if triple[2] > store.recall_threshold][:3]
        print(f'Cue: "{cue}"')
        if not top:
            print("  <nothing surfaced>")
        for mem, r, p in top:
            print(f"  r={r:.2f} p={p:.2f} g={mem.g:.2f}  {mem.text}")
        print()

    snapshot = store.strength_snapshot(t=CHECK_AT_TICK)
    alive = [row for row in snapshot if row["p_at_r1"] >= DEAD_THRESHOLD]
    dead = [row for row in snapshot if row["p_at_r1"] < DEAD_THRESHOLD]
    avg_g = sum(row["g"] for row in snapshot) / len(snapshot)
    avg_p = sum(row["p_at_r1"] for row in snapshot) / len(snapshot)

    print("=== KEY METRICS ===")
    print(f"g0                : 1.0 (paper default)")
    print(f"time unit         : ticks")
    print(f"total memories    : {len(snapshot)}")
    print(f"checked at tick   : {CHECK_AT_TICK}")
    print(f"still alive (p_at_r1 >= {DEAD_THRESHOLD}): {len(alive)}/{len(snapshot)}")
    print(f"avg g             : {avg_g:.2f}")
    print(f"avg p_at_r1       : {avg_p:.3f}")
    print()
    print("Per memory:")
    for row in snapshot:
        status = "alive" if row["p_at_r1"] >= DEAD_THRESHOLD else "dead"
        print(
            f"  [{status:5s}] g={row['g']:.2f}  p_at_r1={row['p_at_r1']:.3f}  {row['text']}"
        )


if __name__ == "__main__":
    main()
