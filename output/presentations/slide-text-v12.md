# Slide text — relational memory v12

Slides 1–14 are the main presentation. Slides 15–21 are backups. Text below comes from the PPTX; chart values are available in the editable charts and numerical evidence.


## Slide 1: Relational memory

in multi-agent simulations

Modeling human-like memory

for more believable simulations

Andre Atkins, Ben Sadorra, and Ryan Shechtman

CMSC473/673

## Slide 2: The problem: human-like remembering

A familiar place can bring back an old experience.

The pattern we want to model

An experience can be hard to recall.

A specific context can bring it back.

Other experiences continue to fade.

The gap in a simple retrieval rule

Text similarity finds matching content.

Recency favors recent experiences.

Personal associations need their own history.

Our goal is believable remembering and forgetting in a social simulation.

## Slide 3: Example: remembering where Mia helped

Earlier

Mia gives the agent food beside a blue pinwheel.

She says, “I’m usually here.”

Much later

The hungry agent notices the same pinwheel.

Mia and food are outside its current view.

The reminder

The pinwheel brings back that encounter,

giving the agent a reason to look for Mia there.

Expected behavior: approach the location and seek Mia’s help.

## Slide 4: Thesis and testable hypotheses

Associative memory should give agents more appropriate

context for their decisions, making their behavior more

believable and consistent with their own experiences.

H1  Associative recovery

Associations will help agents retrieve

past experiences relevant to their

present situation, including experiences

that semantic similarity and time decay

alone would miss.

H2  Lasting accessibility

Recalling experiences through their

associations will help preserve relevant

context across longer gaps.

## Slide 5: Models and their roles

Role

Planned model

Rationale

Direct similarity

SBERT all-MiniLM-L6-v2

Sparser similarity scores in the cited setup.

Small enough to run locally.

Actions and messages

GPT-5 Nano

API generation for repeated

matched simulation runs.

Lower API-cost option

A locally available Qwen model

Local generation for larger collections

when the hardware permits.

SBERT scores the cue against memory text.

The generation model receives memories before acting.

## Slide 6: The memory’s own forgetting curve

Direct relevance rᵢ

Cue-to-memory

embedding similarity

Elapsed time

Time since full recall

Memory strength gᵢ

Each memory has its own g.

A higher g slows forgetting.

Calculated Hou curves at relevance .60. These are model scores, not measured human recall rates.


Chart labels:

- Series: g = 1
- Series: g = 2
- Series: g = 4
- Axis: Retrieval score
- Axis: Ticks since full recall

## Slide 7: A separate pathway for associations

Direct-memory contribution

Mᵢ = rᵢ exp(−tᵢ / (τₘ gᵢ))

rᵢ  cue-to-memory cosine similarity

gᵢ  this memory’s strength

τₘ  tick-to-memory time scale

One uncrowded connection

Gᵢ = β c w exp(−Δ / h)

c  current cue activation

w  connection weight

h  connection retention time

xᵢ = max(Mᵢ, Gᵢ)

pᵢ = (1 − exp(−xᵢ)) / (1 − exp(−1))

Each route ages separately. Max takes the stronger contribution. β limits the graph contribution.

## Slide 8: When the pinwheel can bring it back

Ticks 0–32

Either method can

retrieve the encounter.

Ticks 33–139

Only the association

keeps it above .15.

From tick 140

Both fall below

the cutoff.

One fixed episode and visible cue. No intervening recall, new connections or competing memories.


Chart labels:

- Series: Similarity + decay
- Series: With association
- Series: Cutoff .15
- Axis: Final retrieval score
- Axis: Ticks since the encounter

## Slide 9: Which memories enter the prompt?

1

Score the memories

Compare direct and associative contributions before changing any strengths.

2

Fill the memory slots

Take up to 3 highest-scoring memories above .15 that fit within 600 tokens.

3

Apply recall updates

Rehearse each included memory. Update its strongest graph route only if it beats direct access.

“Selected” means its text enters the agent’s prompt.

## Slide 10: Connection rehearsal and later access

Both recall the same

episode once

Only one also strengthens

the connection.

The score clears the

cutoff longer.

Both links still fade.

Fixed graph and cue. Separate probes, with no new episodes or intervening recalls.


Chart labels:

- Series: No link update
- Series: Link strengthened once
- Series: Cutoff .15
- Axis: Retrieval score
- Axis: Ticks after the recall

## Slide 11: Experiment conditions

Condition

What it tests

Semantic similarity and ordinary decay

Baseline remembering and forgetting

Associative retrieval, link learning off

H1: relevant experiences recovered

through surviving connections

Associative retrieval, link learning on

H2: useful context preserved after

a connection helps recall an experience

We will vary the time gap and the number of competing memories.

Additional checks: goal-aware queries, slower ordinary forgetting, wrong cues and crowded places.

## Slide 12: TerraLingua and the memory audit UI

The agent’s input

Local observations and messages

State, plan and recalled experiences

Planned audit trail

Direct and associative contributions

Winning route and edge age

Memories included and left out

Changes to memory and link strength

Trace each retrieval and state update, then inspect the exact prompt and resulting action.

## Slide 13: Evaluation: memory patterns and believability

Question

Evidence we will collect

H1: does context recover relevant old episodes?

Relevant recall, false reminders and displacement

H2: does rehearsal preserve useful access?

Useful and distracting reminders

after longer gaps

Do decisions fit the recalled context?

Prompt and action traces, plus blind ratings

Does the simulation feel more believable?

Blind ratings of remembering and behavior

Prompt inclusion, appropriate decisions and believability are separate outcomes.

## Slide 14: Proposed implementation and experiments

1

Reference scoring and traces

Separate the memory and edge clocks. Verify the worked examples.

2

Controlled comparisons

Vary delays and distractions. Test retrieval and connection learning.

3

Simulation and human review

Integrate the audit UI and evaluate believable remembering.

Initial scope: independent associative access and connection learning.

## Slide 15: Full retrieval notation

q: current cue     mᵢ: stored episode     E: SBERT embedding

rᵢ = max(0, cos(E(q), E(mᵢ)))

Mᵢ = rᵢ exp(−(t − tᵢ,last) / (τₘ gᵢ))

Lₑ = wₑ exp(−(t − tₑ,last) / hₑ)

Gᵢ = β max over routes [source cue × edge transmissions × crowd penalty]

xᵢ = max(Mᵢ, Gᵢ)       pᵢ = (1 − exp(−xᵢ)) / (1 − exp(−1))

The prompt receives at most k memories above the cutoff that fit within the token budget.

## Slide 16: What changes after a retrieval

State

Update

When

Memory strength g and clock

Increase g by the Hou spacing rule.

Reset its last-full-recall time.

The memory enters the prompt.

Connection strength w

Start from current transmission z.

Set w = z + η(1 − z).

Its memory enters the prompt

and this graph route beats direct access.

Connection duration h and clock

Raise h to a finite cap.

Record this reinforcement time.

Once per qualifying edge

per retrieval event.

Memories left out

No recall update.

Their connections continue aging.

Cutoff, slot limit or token limit

excludes the memory.

A preview or merely seeing the cue changes no existing memory or connection strength.

## Slide 17: Graph sources and allowed connections

Connection

Meaning

Initial scope

Entity to episode

A current entity matches

an episode’s entity tag.

Main condition: explicitly

observed entity cues.

Same tick

Two separately stored events

share a creation tick.

Optional. The writer normally

stores one episode per turn.

Adjacent episodes

One stored episode follows another.

Optional. Immediately previous

or next stored episode.

Memory through one entity hub

A direct source cues another episode

through two mentions edges.

Optional. Fixed seed budget

and no recursive spreading.

Example: receiving food beside a pinwheel links the encounter to that object. The object did not cause the help.

## Slide 18: Time and the experimental controls

Test

What varies

Same at each time gap

H1: associative recovery

Time gap, competing memories

and graph access on or off

Identical memory state,

cue and prompt budget.

H2: lasting accessibility

Time since the same recall event

and connection learning on or off

Equal ordinary recalls.

Only one copy updates its link.

Believability in the simulation

Memory condition across

repeated simulation runs

Same generation settings

and prompt budgets.

Time still causes decay. We vary the gap and compare at each gap.

## Slide 19: Expected limits and failure cases

Failure case

What we expect or need to measure

No active associated cue

No support through that cue. Other routes may still work.

Several episodes share a cue

Graph scores can override differences in text relevance.

A cue accumulates many episodes

Crowding can push even a surviving link below the cutoff.

Repeated false reminder

An irrelevant association can strengthen and recur.

Aging of stored content

Access changes. Stored text stays intact.

Cue-triggered retrieval is a modeling hypothesis. Clinical trauma and emotion require additional mechanisms.

## Slide 20: Questions we should be ready to answer

Question

Answer

Does the example prove vector search fails?

Similarity is positive. Ordinary decay

puts the old encounter below the cutoff.

Is graph structure necessary here?

A graph or weighted lookup can store

the same cue-to-episode associations.

Does reinforcement prove better memory?

It raises support by design. Useful recall

and believability still need evaluation.

What does selected mean?

Its text enters the prompt after the cutoff

and the memory and token limits.

What can max lose?

When graph support dominates, differences

in direct text relevance can stop affecting rank.

## Slide 21: Sources

Source

Role in the project

Hou, Tamoto & Miyashita (2024)

Direct-memory decay and rehearsal

Honda et al., HAI 2025 proceedings

Embedding choice and memory-model context

Anderson & Reder, 1999

Associative competition and the fan effect

Paolo et al., TerraLingua, 2026

Persistent multi-agent simulation environment

APA Dictionary of Psychology

Associative-memory motivation

Kuroki et al., Shachi

Agent-believability evaluation context

Huang et al., EmotionBench

Secondary behavioral-evaluation context

Course proposal and design brief

Project goal, proposed equations and worked examples

Full references appear in the speaker notes. The independent-pathway equations are our proposal.
