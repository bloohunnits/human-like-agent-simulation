"""Runs the README's 90-day Ada walkthrough through the bare Hou kernel
(no graph, no edges - the piece implemented so far) to see, concretely,
what it gets right on its own and what it needs the graph for.

No LLM calls. Real SBERT embeddings, real Hou math. t is in days.

Also sweeps g0 (initial consolidation strength), because the first run of
this at the paper's default g0=1 produced a degenerate result: every
memory was already unrecallable by day 90, mushroom-sickness included -
p = [1-exp(-r*exp(-t/g))]/[1-e^-1] collapses to ~0 once t/g gets much
past ~5, and with g0=1 that happens within days. g and t must share a
scale; the paper never pins one down (open question in docs/DESIGN.md).
Comparing g0=1 against a couple of larger values makes that sensitivity
concrete instead of theoretical.
"""

from embedder import SBERTEmbedder
from hou_memory import HouMemoryStore


def build_ada_store(embedder, g0: float) -> HouMemoryStore:
    store = HouMemoryStore(embed_fn=embedder, top_k=3, recall_threshold=0.05, g0=g0)

    # Day 1: Ada meets Ben, he shows her the berry patch.
    store.add(
        "Ada meets Ben at the well. He shows her the berry patch nearby.",
        t=1,
    )

    # Days 2-30: mundane, near-identical berry-picking trips. Ben is not
    # mentioned again - nothing here recalls him, because there is no
    # graph yet to route a "patch" trip back to "Ben" without saying his name.
    for day in range(2, 31):
        store.add(f"Ada picks berries at the patch on day {day}.", t=day)

    # Day 45: a non-semantic causal pair. The two memories share no words.
    store.add("Ada eats strange mushrooms she found in the woods.", t=45)
    store.add("Ada is sick all night after her dinner.", t=45)

    return store


def report(store: HouMemoryStore, label: str):
    print(f"\n########## g0={store.g0}  ({label}) ##########")

    print("\n-- Day 90: what surfaces for each cue (read-only, no reinforcement) --")
    for cue in [
        "Ben",
        "the berry patch",
        "picking berries",
        "should Ada eat mushrooms again?",
        "Ada feeling ill",
    ]:
        query_embedding = store._embed_fn(cue)
        scored = sorted(store.score(query_embedding, t=90), key=lambda t: t[2], reverse=True)
        top = [triple for triple in scored if triple[2] > store.recall_threshold][:3]
        print(f'Cue: "{cue}"')
        if not top:
            print("  <nothing surfaced>")
        for mem, r, p in top:
            print(f"  r={r:.2f} p={p:.2f} g={mem.g:.2f}  {mem.text}")

    print("\n-- Raw standing at day 90 (no query, r=1 ceiling) --")
    for row in store.strength_snapshot(t=90):
        print(
            f"  g={row['g']:5.2f}  recalls={row['recall_count']}  "
            f"p_at_r1={row['p_at_r1']:.3f}  {row['text'][:65]}"
        )


def main():
    embedder = SBERTEmbedder()  # shared - loading SBERT twice is wasted time
    for g0, label in [
        (1.0, "paper default"),
        (10.0, "recalibrated: a memory should coast for ~1-2 weeks unaided"),
        (30.0, "recalibrated: coast for ~a month unaided"),
    ]:
        report(build_ada_store(embedder, g0), label)


if __name__ == "__main__":
    main()
