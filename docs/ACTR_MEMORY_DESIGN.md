# ACT-R-inspired cognitive memory: target design

Status: design audit completed; implementation pending. Written 2026-09-27 as the design brief
for evolving the ACT-R kernel beyond what currently exists in
[`code/hou_memory/hou_memory.py`](../code/hou_memory/hou_memory.py).

## Where this sits relative to what's built

The current `actr_recall()` implementation is a simplified base-level
activation (`B = ln(Σ age^-d)`) plus one similarity term
(`A = B + w·r`), scored against a fixed threshold. It captures history-based
decay and a spacing-vs-massing distinction, but it is **not** the system this
document describes: it has no structured chunks, no distinct spreading-
activation sources, no partial-matching mismatch scale, and no separation
between "what gets encoded," "what gets requested," and "what gets
retrieved." It is currently hidden in God Mode behind `SHOW_ACTR = false`
pending exactly the kind of calibration work this document scopes out.

Section 8 has now been answered in
[ACTR_MEMORY_ARCHITECTURE.md](ACTR_MEMORY_ARCHITECTURE.md): source-checked audit,
proposed architecture, isolated runtime prompts, worked examples, and planned
tests/ablations. The supplied brief below remains verbatim. No new ACT-R runtime
has been implemented by this design phase.

This document is the fuller target: an architecture where LLM generation and
embedding similarity fill specific, bounded gaps in ACT-R's declarative
memory, while activation, history, competition, and retrieval failure stay
under the memory engine's control rather than the LLM's. It is
simulation-agnostic on purpose — it does not assume TerraLingua's action/obs
schema, and integrating it is a separate step from writing it.

Companion docs: [DESIGN.md](DESIGN.md) (the graph architecture, implemented),
[RESEARCH.md](RESEARCH.md) (paper survey), [INTEGRATION.md](INTEGRATION.md)
(what's actually running and tested today).

The brief below is reproduced exactly as given, unedited, in a fenced block
so nothing gets reflowed, re-cased, or reworded.

---

## The brief, as given

```text
Help me develop an ACT-R-inspired cognitive memory system for an
open-ended LLM-agent simulation, eventually integrated into TerraLingua.

The objective is to preserve the integrity of ACT-R's memory mechanisms
while adapting the representation, retrieval-request composition, and
semantic comparison mechanisms that may be too rigid for our setting.

We are not trying to replace ACT-R-style recall with ordinary vector
retrieval. We are investigating where LLM generation and embedding
similarity can supply missing capabilities while leaving activation,
memory history, competition, and retrieval failure in control.

Keep the design exploratory and simulation-agnostic. Do not assume
TerraLingua APIs, prescribe an exhaustive ontology, or jump directly
into implementation.


1. BACKGROUND: WHAT ACT-R IS

ACT-R—Adaptive Control of Thought–Rational—is a cognitive architecture:
a framework for modeling human cognition, not merely a search algorithm
or a vector database.

It combines symbolic representations with numerical mechanisms.
Declarative knowledge is represented as structured chunks. Buffers
provide access to information currently available to the system.
Production rules respond to buffer contents and direct cognitive
operations, including memory requests.

A chunk is not inherently a word, token, or fixed-length text passage.
It is a structured representation with slots and values, which can
include references to other chunks.

Classic ACT-R declarative retrieval is not inherently based on cosine
similarity between a query embedding and whole-memory embeddings.
A structured request specifies desired information; matching and
activation determine what is retrieved.

However, ACT-R is not simply a rigid exact-match database either.
Partial matching can allow imperfect matches, and current context
can influence recall through spreading activation.

Our focus is the declarative-memory subsystem and its interfaces.
Do not imply that retaining its equations reproduces the entire
ACT-R cognitive architecture.


2. THE MEMORY MECHANISMS WE WANT TO PRESERVE

Use this conceptual activation decomposition:

A_i = B_i + SA_i + P_i + epsilon_i

Each term should retain a distinct responsibility.

BASE-LEVEL ACTIVATION

A standard history-based form is:

B_i(t) = beta_i + ln(sum_r [(t - t_ir)^(-d)])

Here:
- i identifies a persistent memory.
- t_ir is the time of a recorded reference/presentation event.
- t - t_ir is the positive elapsed time since that event.
- d controls decay.
- beta_i is an optional constant offset.

Intuition: repeated references contribute strength, and those
contributions weaken with elapsed time.

Base-level activation represents historical accessibility, not
semantic similarity to the current request. An irrelevant memory
can have high base activation.

Keep this mechanism. Changing the query must not erase a memory's
identity, history, or accumulated accessibility.

Define which simulation events count as encoding, reuse, recall,
or rehearsal. Distinguish that policy from ACT-R's actual reference
and chunk-merging mechanics. Merely scoring or inspecting a
candidate must not automatically strengthen it.

SPREADING ACTIVATION

A simplified form is:

SA_i = sum_j [W_j * S_ji]

Here:
- j identifies a currently available activation source.
- W_j is its allocated source activation.
- S_ji is its association strength to memory i.

Intuition: current context can support an otherwise less accessible
memory through existing associations.

In ACT-R, these sources are tied to represented buffer contents,
including chunk-valued slots. They are not automatically every
word appearing in a natural-language query.

Preserve bounded source activation. Generating additional words
or synonymous cues must not create unlimited contextual support.

Do not equate association strength with semantic similarity.
Explain what establishes an association in the baseline mechanism
before proposing any changes to it.

Do not silently replace direct ACT-R-style spreading with recursive,
multi-hop graph diffusion. Treat that as a separate extension.

PARTIAL MATCHING

For ordinary positive slot-value comparisons, a simplified form is:

P_i = mp * sum_k [s(q_k, v_ik)]

Here:
- q_k is a requested slot value.
- v_ik is the candidate memory's corresponding value.
- s measures their similarity on a mismatch-compatible scale.
- mp controls the influence of mismatches.

A common convention gives exact matches 0 and mismatches negative
values, with -1 representing the default maximum difference.
Negated constraints and other comparison operators require their
appropriate handling rather than blindly applying this simplified
formula.

ACT-R supports explicitly supplied similarities and programmable
similarity functions. Do not claim that all similarities must be
entered manually.

TOTAL ACTIVATION AND RETRIEVAL

Preserve competition among eligible memories, an explicit retrieval
threshold, noise where appropriate, and activation-dependent
retrieval timing.

The engine must be able to retrieve a less-than-perfect match or
fail to retrieve anything. Do not let the LLM silently correct
those outcomes into optimal recall.

These numerical mechanisms belong in code, including controlled
randomness where required—not in an LLM's improvised scoring.


3. THE TWO MAIN ADAPTATIONS WE ARE CONSIDERING

A. LLM-assisted representation and request composition.

B. Embedding-derived similarity for appropriate partial matches.

These address different problems:

Generation answers:
"How should the current experience and information need be represented?"

Similarity answers:
"How different are this requested value and this candidate value?"

Activation answers:
"Given history, context, matching, and noise, what actually comes back?"

Do not collapse those questions into one semantic-relevance score.


4. ADAPTATION A: GENERATING COMPATIBLE STRUCTURE AND REQUESTS

The representation problem exists on both sides of retrieval.

Stored experiences need usable structure. Current experiences and
information needs need compatible structure. Generating an excellent
query is insufficient if stored memories use incompatible roles,
identities, or levels of abstraction.

Explore how an LLM could help interpret experiences and compose
structured retrieval inputs without becoming the memory retriever.

Keep two related functions distinguishable:

Encoding:
Interpret an agent-visible experience into a proposed structured
representation for the memory system.

Request composition:
Interpret the current observation, existing goal, and bounded working
context into a request for information and relevant contextual sources.

The system should validate these proposals and control persistence,
identity binding, buffer updates, and reference-history updates.

Do not assume the LLM needs to generate ACT-R programming syntax.
A small intermediate representation may be more appropriate.

Keep the retrieval request separate from the contextual information
that can supply spreading activation. Mentioning a concept in the
request does not automatically create a source or an association.

Explore a modest shared structural contract with open-ended content,
rather than either an exhaustive hand-coded universe or completely
unconstrained prose.

Address granularity in terms of meaning and relationships, not word
count. Preserve who did what to whom, relevant negation, and whether
an event was observed, reported, hypothetical, or uncertain.

Avoid turning every contextual detail into a mandatory constraint.
Distinguish genuinely required properties, partially matchable
properties, and context-only information.

Unknown information must remain unknown. Do not turn an unresolved
person into a guessed identity or an unspecified value into a
requirement that a slot be empty.

Recognize that deciding what information to request replaces part
of task-specific cognitive control, not merely text formatting.
Label that substitution honestly.


5. ADAPTATION B: EMBEDDINGS INSIDE PARTIAL MATCHING

Manually maintaining every semantic similarity in an open-ended
environment would be impractical. A programmable comparator could
instead derive selected similarities from embeddings:

s(q_k, v_ik) = f(cos(E(q_k), E(v_ik)))

This is a proposed way to supply partial-match values—not a
replacement for the activation equation.

Explore how to map scores onto the chosen mismatch scale, preserve
exact matches, and calibrate the mismatch parameter against the
other activation terms.

A compatible numerical range is necessary but does not establish
psychological validity. Test the resulting behavior.

Do not embed every value indiscriminately. Entity identity, numbers,
times, semantic categories, and relational roles may require
different comparison policies.

Two people must not become the same person because their names or
descriptions have similar embeddings. Likewise, paraphrases of one
episode should not automatically create unrelated memory identities.

Keep partial-match similarity distinct from contextual association.
An embedding comparator does not automatically explain where S_ji
comes from.

Additional uses of generation or embeddings may be considered, but
only when they solve a clearly identified problem. For each proposed
use, explain which architectural responsibility changes and why the
existing mechanism is insufficient.


6. ISOLATION AND ARCHITECTURAL INTEGRITY

The request-composition LLM should receive only the current
agent-visible observation, an existing goal, bounded working context,
and legitimately available representation/identity information.

It must not receive the entire memory archive, hidden simulator state,
another agent's private state, candidate answers, or a target outcome.

Previously recalled information may be present in working context
when the surrounding model legitimately places it there. Do not
confuse that with unrestricted access to long-term memory.

Enforce access restrictions in code. A prompt alone is not isolation.

Allow ordinary language interpretation, but require unsupported
personal-history claims and unresolved references to remain absent
or explicitly uncertain.

The LLM must not:
- Search or rank the memory archive.
- Invent the episode it hopes to retrieve.
- Set activation strengths or learning histories.
- Reformulate requests indefinitely until a desired answer appears.

The memory engine must not:
- Exact-filter away all candidates intended for partial matching.
- Treat candidate inspection as recall.
- Let duplicate cues inflate activation.
- Let representation changes reset or arbitrarily merge histories.
- Make base-level activation mathematically present but behaviorally
  irrelevant through overly restrictive candidate selection.

We care about preserving causal influence, not merely retaining
familiar equations in the code.


7. ILLUSTRATIVE EXAMPLE

Current observation:
"John asks whether he can borrow some food."

Existing goal:
"Decide whether to lend it."

The interpretation component may represent John, a borrowing
request, food, and the agent's decision context.

The request composer may seek a previous interaction involving
John, with borrowing-related information contributing where
appropriate.

It should not automatically require every retrieved experience to
involve food; an experience involving borrowed money may matter.

It must not generate:
"Recall the time John failed to repay me."

That would introduce a specific past episode unless it was already
legitimately available in working context.

The memory engine—not the LLM—then evaluates eligible memories
using their histories, contextual associations, partial-match
penalties, and noise.

A relevant memory may surface, a competing memory may surface, or
retrieval may fail.

Use this example to expose design choices, not to force one
particular schema or guarantee the correct memory wins.


8. WHAT I WANT YOU TO PRODUCE

First, audit this proposal for logical mistakes and distinguish
standard ACT-R mechanisms from our proposed adaptations.

Then explain the smallest coherent architecture that connects:
experience interpretation, stored representation, request composition,
contextual sources, matching, activation, recall, and learning updates.

Identify which responsibilities belong to the LLM, embedding
comparators, deterministic adapters, and the memory engine.

Discuss representation granularity and request breadth without
prematurely fixing an ontology or simulation-specific schema.

Provide contrasting examples, including paraphrases, ambiguous
identities, role reversals, and information that is relevant without
sharing every surface word.

Draft isolated runtime prompts for the generation responsibilities,
keeping memory encoding and request composition distinguishable.

Propose focused tests and ablations that establish whether:
- Memory history still changes retrieval with the request held fixed.
- Contextual associations affect recall independently of base history.
- Embedding partial matching helps without erasing identities or roles.
- Near-matches survive candidate generation.
- Redundant cues do not manufacture activation.
- Hidden memory or simulator information leaks into request formation.

Begin with an inspectable memory store and an exhaustive
eligible-candidate baseline before introducing approximate search.

Do not assume that better task performance means more human-like
memory. Identify what behavioral evidence would support a claim
of cognitive fidelity.

Use primary ACT-R sources to verify architectural claims and
identify version-specific assumptions. Do not assume this hybrid
is novel merely because we have proposed it.

GUIDING PRINCIPLE

Use generation to express experiences and information needs.
Use vector similarity where a semantic comparison is needed.
Keep ACT-R-style activation, memory history, and retrieval dynamics
responsible for what the agent actually recalls.
```

---

## Status

Section 8's design deliverables are complete in
[ACTR_MEMORY_ARCHITECTURE.md](ACTR_MEMORY_ARCHITECTURE.md), dated 2026-09-27.
The next milestone is the inspectable, simulation-independent reference engine
and fixture runner described there. Proposed acceptance tests are not yet
implemented; the existing Hou/graph implementation remains unchanged.
