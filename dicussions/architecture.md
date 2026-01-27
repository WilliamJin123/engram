# Engram Architecture Discussion

Single source of truth for memory architecture design. When reading this document, it means that we are curerntly in dicussion mode. keep everything within this discussions directory. Do not peek into the code unless explicitly instructed to.

---

## Core Philosophy

### Nodes as Discrete Concepts

Reject distributed representations. One concept = one node (post-consolidation).

The reason many neurons fire when thinking of "dog" isn't distributed encoding - it's:
1. **Multiple entry points**: Vision, sound, smell, language all lead to same concept
2. **Multiple facets**: Visual features, behaviors, emotions - different aspects
3. **Redundancy**: Fault tolerance through multiple representations

These are different PATHS to and FACETS of the concept, not parts of it.

### Redundancy → Consolidation

Multiple nodes CAN represent the same concept temporarily:

```
[dog_visual_memory]    ─┐
[dog_sound_memory]     ─┼─→ CONSOLIDATION → [DOG]
[dog_from_book]        ─┘
```

The unified node inherits richness from all source modalities.

### Everything is a Node

Not just concepts - **everything**:
- Properties are nodes ("red", "flying", "heavy")
- Relationships can be nodes ("is mother of" as a concept)
- Actions are nodes ("running", "eating")
- Assertions are nodes ("penguins cannot fly")
- Operations are nodes (merges, corrections, queries)
- The system's own history is nodes

There are no attributes on nodes in the traditional sense. If a dog is brown, there's an edge from [dog] to [brown]. If a bird can fly, there's an edge from [bird] to [flying].

### Everything is Reasoned Upon

Because everything is a node, everything can be:
- Examined
- Questioned
- Undone
- Modified

A consolidation is a node. A correction is a node. An assertion is a node. The system's own history of changes is part of the graph.

---

## Node Model

### The Insight: Nodes Don't Have Inherent Weight

In the brain:
- Synapses have weights (channel density, receptor count)
- Neurons don't have inherent "importance"
- A neuron's importance IS its connectivity

Similarly, our nodes shouldn't have inherent properties like mass, energy, certainty. **Everything is relational.**

We're not uncertain about "dog exists as a concept." We're uncertain about edges:
- "Is this thing a dog?" (edge from observation to dog)
- "Do dogs bark?" (edge from dog to barking)
- "Is this source reliable?" (edge to provenance)

### Minimal Node Model

```
Node:
  id: unique identifier
  content: human-readable payload

  # That's it. No inherent weight, energy, mass, certainty.
  # All properties emerge from edge structure.
```

### Derived Properties (Not Stored)

What we previously called node properties are actually derived from edges:

**"Importance" (was: mass)**
```
importance(node) = sum of incoming edge strengths
                 + sum of outgoing edge strengths
                 # Or some weighted combination
```
A well-connected node is important. Importance isn't inherent - it's structural.

**"Accessibility" (was: energy)**
```
accessibility(node) = sum of (edge.strength * recency(edge.last_activated))
                      for edge in incoming_edges
```
A node is accessible if recently-activated nodes have strong edges to it.

**"Certainty"**
```
certainty(node) = strength of edges from evidence/provenance nodes
```
We're not certain about the node itself - we're certain about the edges that support it.

**"Plasticity"**
```
plasticity(node) = inverse of connectivity density
                 # Well-connected nodes are harder to change
                 # Because changing them means changing many edges
```

### What This Means

1. **No node-level decay**: Edges decay, not nodes. A node with no edges is just... disconnected (and maybe garbage collected).

2. **Uncertainty lives in edges**: "I'm not sure if dogs bark" = weak edge from dog to barking, not a property of either node.

3. **Importance emerges**: A concept becomes important by accumulating strong connections, not by being marked important.

4. **Simpler model**: Nodes are just identity + content. All dynamics happen on edges.

### Uncertainty About Concepts Themselves

We can be uncertain about an abstraction itself, not just its relationships. How is this represented?

**A tentative/uncertain concept has:**
- Weak incoming edges (not sure what leads to it)
- Weak outgoing edges (not sure what it implies)
- Few edges overall (not well-established)
- Possibly contradictory edges (conflicting information)

**Example: First encounter with "dog"**
```
[fluffy_thing_I_saw] ──(weak)──> [???_new_concept_???]
                                       │
                                       ├──(weak)──> [animal?]
                                       ├──(weak)──> [pet?]
                                       └──(weak)──> [dangerous?]
```

The concept exists, but it's weakly connected. We're uncertain about what it IS (weak outgoing edges to categories) and what leads to it (weak incoming edges from observations).

As we learn more:
```
[dog_1_I_saw] ──(strong)──> [DOG]
[dog_2_I_saw] ──(strong)──> [DOG]
[word_"dog"] ──(strong)──> [DOG]
                             │
                             ├──(strong)──> [animal]
                             ├──(strong)──> [pet]
                             ├──(strong)──> [barking]
                             └──(inhibitory)──> [dangerous]  # most dogs aren't
```

The concept becomes established through accumulated strong edges. **Certainty about a concept = density and strength of its edges.**

### Abstraction Lifecycle

**1. Forming new abstractions:**
- See multiple instances
- Instances cluster (strong edges between them)
- Cluster crystallizes into parent node (the abstraction)
- Or: receive a label and create a node to anchor it

**2. Uncertainty about early abstractions:**
- New abstraction has few, weak edges
- It's tentative - might be wrong way to carve reality
- Strengthens as more instances connect to it
- May be revised, split, or merged

**3. Combining separate abstractions:**
```
Before: [morning_star] and [evening_star] are separate
Learn:  They're the same thing!
After:  Merge into [venus], MergeEvent records this

The merger is a node. It can be examined, undone if wrong.
```

**4. Splitting an abstraction:**
```
Before: [fish] includes whales
Learn:  Whales aren't fish!
After:  Create [whale] node, move edges, create SplitEvent

[whale] ──(inhibitory)──> [fish]  # whale suppresses fish association
[whale] ──(strong)──> [mammal]
```

**5. Adding labels that unify:**
```
Before: [dog], [cat], [whale] exist separately
Learn:  "These are all mammals"
After:
  Option A: Create [mammal] node, connect to all three
  Option B: [mammal] already exists, just add edges

Either way: just adding edges (and maybe a node)
```

**6. Refining abstractions:**
```
Before: [bird] ──(strong)──> [flying]
Learn:  Penguins are birds but don't fly
After:
  [penguin] ──(strong)──> [bird]
  [penguin] ──(inhibitory)──> [flying]

The abstraction isn't wrong, just has exceptions.
```

### No Architectural Changes Needed

All of these are representable with:
- Nodes (identity + content)
- Edges (directed, signed strength, optional gate)
- MergeEvents (for combining)
- SplitEvents (for separating)
- Standard edge operations (add, strengthen, weaken, inhibit)

The uncertainty about concepts is structural, not a special property.

### Practical Consideration

Computing derived properties on-demand might be expensive. We might cache them:
```
Node:
  id: unique identifier
  content: human-readable payload

  # Cached (derived, not authoritative)
  _cached_importance: float
  _cached_accessibility: float
  _cache_timestamp: timestamp
```

But these are optimizations, not the ground truth. The ground truth is the edge structure.

---

## What Decays and How It Affects Traversal

### Only Edges Decay

Since nodes have no inherent weight, **only edges decay**:

```
edge.strength *= decay_rate  # per time unit, uniform for all edges
edge.last_activated = timestamp
```

- All edges decay at the same base rate
- STDP modifies this (timing-based strengthening/weakening)
- Activation refreshes the edge (resets decay clock, may strengthen)

**Nodes don't decay.** A node with no edges is just disconnected. It might be garbage collected, but it doesn't "fade" - its edges fade.

### How Decay Affects Traversal

Traversal is **spreading activation**. Since everything is in edges:

**1. Edge strength determines spread amount:**
```python
activation_spread = source_activation * edge.strength
```
Weaker edges spread less activation → target less likely to activate.

**2. Accessibility is edge-derived:**
```python
def can_reach(target, from_active_nodes):
    total_incoming = sum(
        edge.strength * source_activation[edge.source]
        for edge in target.incoming_edges
        if edge.source in from_active_nodes
    )
    return total_incoming > activation_threshold
```

A node is reachable if there's enough activation flowing in through edges. No separate "energy" property needed.

**3. Is traversal stochastic?**

The brain has stochasticity at the synapse level (probabilistic neurotransmitter release). But population-level behavior is more deterministic.

Options for our model:

**Option A: Fully deterministic**
- Activation spreads based on strength, thresholds
- Same input → same output (given same graph state)
- Variability comes only from different contexts/states

**Option B: Stochastic at edge level**
```python
if random() < edge.strength:  # strength as probability
    spread_activation(edge.target)
```
- Same input can → different outputs
- More brain-like at micro level
- Harder to reason about, debug

**Option C: Deterministic with stochastic tiebreaking**
- Activation spreads deterministically
- When nodes are equally activated, random selection
- Controlled stochasticity only where needed

**Current lean: Option A or C**

Fully deterministic (A) is simpler and variability emerges from context anyway. Stochastic tiebreaking (C) adds controlled randomness only where truly ambiguous.

But this is uncertain - needs experimentation.

---

## Edge Model

### Properties

```
Edge:
  source: node_id
  target: node_id
  strength: float (-1.0 to +1.0)  # signed! negative = inhibitory

  # Dynamics
  last_activated: timestamp
  activation_count: int

  # Gating (optional)
  gate: node_id | null  # edge only fires if gate node is active
```

### Directional

Edges are directed. A→B does not imply B→A.

Often edges are paired (dog→bark and bark→dog), but with potentially different strengths:
- dog→bark: +0.8 (dogs strongly evoke barking)
- bark→dog: +0.5 (barking might be dog, might be seal, might be cough)

### Excitatory vs Inhibitory (Signed Strength)

Like synapses in the brain:
- **Positive strength**: excitatory - activating source increases target activation
- **Negative strength**: inhibitory - activating source decreases target activation

Examples:
```
[bird] ──(+0.8)──> [flying]      # birds evoke flying
[penguin] ──(-0.7)──> [flying]   # penguins suppress flying
[penguin] ──(+0.9)──> [bird]     # penguins strongly evoke bird
```

When you query "can penguins fly?":
1. Activate [penguin]
2. Spreading activation hits [flying] via two paths:
   - Direct: penguin --(-0.7)--> flying (inhibitory)
   - Indirect: penguin --> bird --> flying (excitatory)
3. Net activation of [flying] depends on competition
4. The inhibitory direct edge likely wins (stronger, shorter path)

This handles negation through dynamics, not special node types.

### Gated Edges (Conditional Activation)

Some edges only fire when a condition is met:

```
Edge:
  source: [bird]
  target: [flying]
  strength: +0.8
  gate: [has_wings]  # only fires if [has_wings] is active
```

This allows conditionals:
- "Birds can fly IF they have wings"
- "I go outside IF weather is nice"
- "This tool works IF dependencies are installed"

The gate is a node. When the gate node is active (above threshold), the edge can fire. When inactive, the edge is dormant.

### Untyped (Semantically)

Still no "is-a", "has-a" labels. The relationship type emerges from:
- The nodes being connected
- The sign of the edge (excitatory/inhibitory)
- The gate condition (if any)
- The pattern of surrounding connections

Polarity and gating are structural properties, not semantic types.

### Situational Initial Strength

New edges don't have fixed initial strength. Depends on formation context:

- **Trusted source**: stronger initial connection
- **Untrusted source**: weaker initial connection
- **Emotional salience**: stronger (memorable events stick)
- **Logical/deductive**: stronger (clear reasoning)
- **Speculative**: weaker (tentative connections)
- **Repetition**: stronger with each occurrence

Key insight: You can be **certain about your uncertainty**. If someone tells you something dubious:
- Edge to the claim: weak (uncertain)
- Edge to "this_source_is_unreliable": strong (certain about uncertainty)

### STDP-Like Dynamics (Timing-Based Plasticity)

Edge strength changes based on activation timing:

```
For edge A → B:

If A activates BEFORE B activates:
  → Strengthen edge (A contributed to B)
  → "Fire together, wire together"

If A activates AFTER B activates:
  → Weaken edge (A didn't contribute to B)
  → "Wrong direction" signal

If A activates but B never activates:
  → Slight weakening (A didn't lead anywhere useful)

If A never activates:
  → Normal baseline decay only
```

**Why this matters - Pruning alternatives:**

```
[problem]
  ├──> [approach_A] ──> [solution]  # This path taken, succeeded
  ├──> [approach_B] ──> [solution]  # Not taken
  └──> [approach_C] ──> [solution]  # Not taken

When [solution] activates after [approach_A]:
  - approach_A → solution: STRENGTHENS (A before solution)
  - approach_B → solution: WEAKENS (B never fired, or fired after)
  - approach_C → solution: WEAKENS (C never fired, or fired after)

Over time: successful paths strengthen, unused alternatives fade.
```

This naturally prunes suboptimal paths without explicit "this is the best one" logic.

---

## Retrieval: Competitive Activation, Not Randomness

### Why Not Explicit Probability/Noise?

The brain doesn't have a "randomness dial." Variation in memory retrieval emerges from:
- **Context**: What else is currently active
- **State**: Emotional, physiological condition
- **Competition**: Multiple associations vie for activation
- **Interference**: Recent/similar memories compete

These are deterministic processes that PRODUCE variability, not injected noise.

### Competitive Spreading Activation

```python
def retrieve(query, context, state):
    # Find entry points
    entry_nodes = find_entry_points(query)

    # Spreading activation with competition
    activation = {}  # node_id -> activation_level (can be negative!)

    for node in entry_nodes:
        activation[node.id] = initial_activation(node, query)

    for _ in range(spread_iterations):
        new_activation = {}

        for node_id, level in activation.items():
            if level <= 0:
                continue  # inhibited nodes don't spread

            node = get_node(node_id)

            for edge in node.outgoing_edges:
                # Check gate condition
                if edge.gate and not is_active(edge.gate, activation):
                    continue  # gate closed, edge doesn't fire

                # Activation spreads based on SIGNED edge strength
                # Positive edges excite, negative edges inhibit
                spread = level * edge.strength  # strength is -1.0 to +1.0

                # Context modulates
                spread *= context_relevance(edge.target, context)

                # State modulates
                spread *= state_modifier(edge.target, state)

                # Accumulate (excitation and inhibition sum)
                new_activation[edge.target] = (
                    new_activation.get(edge.target, 0) + spread
                )

        # Competition: normalize, clamp, threshold
        activation = normalize_and_threshold(new_activation)

    return nodes_above_threshold(activation)
```

**Key dynamics**:
- Inhibited nodes (negative activation) don't spread further
- Gated edges only fire when gate is active
- Excitation and inhibition sum - net effect determines activation
- Competition emerges naturally from these dynamics

### Why This Produces Variability

Same query, different results because:
1. **Context differs**: Different nodes are pre-activated
2. **State differs**: Different emotional modulation
3. **Recency differs**: Recently accessed nodes have higher energy
4. **Competition**: Which path "wins" depends on current activation landscape

No randomness needed. Variability emerges from the system's state.

### Tip-of-Tongue Phenomenon

When retrieval fails despite knowledge existing:
- Target node has low energy (not accessed recently)
- Competing similar nodes have higher activation
- Context doesn't provide good entry points
- The path exists but loses the competition

---

## Consolidation: Automatic, Reversible, Reasoned

### The Process

```
Phase 1: Attraction
- Similar nodes (high HDV similarity) develop strong edges
- They activate together frequently
- This is automatic, continuous

Phase 2: Soft Merge
- When similarity exceeds threshold AND both nodes activated together N times
- System automatically merges
- Creates a MergeEvent node recording: what merged, when, why

Phase 3: The MergeEvent is a Node
- MergeEvent connects to the new merged node
- MergeEvent can be queried: "what consolidations happened?"
- MergeEvent can be UNDONE: split the node back apart

Phase 4: Reasoning About Merges
- System (or human) can examine merge events
- If a merge was wrong (morning star ≠ evening star), undo it
- The undo is ALSO a node (UndoMergeEvent)
- Full history of the graph's evolution is in the graph
```

### MergeEvent Structure

```
MergeEvent (a Node):
  type: "merge_event"
  source_nodes: [node_id, node_id, ...]  # what was merged
  result_node: node_id                    # what it became
  timestamp: datetime
  trigger: string                         # "similarity_threshold", "explicit_request"
  similarity_at_merge: float
  reversible: bool                        # can this be undone?

  # Edges
  → result_node (created_by relationship)
  → source_node snapshots (if we keep them for undo)
```

### Undoing a Merge

```python
def undo_merge(merge_event):
    if not merge_event.reversible:
        raise CannotUndo("Source data not preserved")

    # Restore original nodes from snapshots
    restored = []
    for snapshot in merge_event.source_snapshots:
        node = restore_node(snapshot)
        restored.append(node)

    # Remove or archive the merged node
    archive_node(merge_event.result_node)

    # Create UndoMergeEvent
    undo_event = create_node(
        type="undo_merge_event",
        original_merge=merge_event.id,
        restored_nodes=[n.id for n in restored],
        reason=context.reason  # why are we undoing?
    )

    return restored, undo_event
```

### Avoiding Bad Merges

Things that LOOK similar but shouldn't merge:
- **Homonyms**: bank (river) vs bank (financial)
- **Instance vs category**: "my dog Max" vs "dogs"
- **Different senses**: morning star vs evening star

Mitigation:
- Similarity threshold must be HIGH
- Context of usage matters (do they appear in same contexts?)
- If nodes have contradictory edges, don't merge (bank→water vs bank→money)
- When uncertain, create soft link instead of merge, revisit later

---

## Curiosity: Uncertainty as Drive

### The Problem Curiosity Solves

Passive memory just stores and retrieves. But humans:
- Actively seek to fill gaps
- Feel discomfort at unresolved uncertainty
- Pursue knowledge even without immediate need
- Learn HOW to learn

### Uncertainty Creates Tension

```
Node:
  ...
  uncertainty_tension: float  # derived from certainty and importance

  # High tension when:
  # - Low certainty + high mass (important but unsure)
  # - Has unresolved contradictions
  # - Provenance is questionable
  # - Connected to high-stakes domains
```

### Tension Drives Behavior

```python
def get_curiosity_targets():
    """What should the system try to resolve?"""

    targets = []

    for node in all_nodes():
        tension = calculate_tension(node)

        if tension > CURIOSITY_THRESHOLD:
            targets.append({
                'node': node,
                'tension': tension,
                'resolution_strategies': suggest_strategies(node)
            })

    return sorted(targets, key=lambda x: x['tension'], reverse=True)

def calculate_tension(node):
    # Important but uncertain = high tension
    importance_uncertainty = node.mass * (1 - node.certainty)

    # Contradictions create tension
    contradiction_tension = count_contradictions(node) * 0.3

    # Questionable provenance creates tension
    provenance_tension = 0
    if node.provenance and avg_credibility(node.provenance) < 0.5:
        provenance_tension = 0.2

    return importance_uncertainty + contradiction_tension + provenance_tension

def suggest_strategies(node):
    """How might we resolve this uncertainty?"""

    strategies = []

    if node.provenance:
        strategies.append("verify_sources")

    if has_contradictions(node):
        strategies.append("resolve_contradictions")

    if node.certainty < 0.5:
        strategies.append("seek_additional_evidence")

    if is_outdated(node):
        strategies.append("check_for_updates")

    return strategies
```

### Learning to Learn

The system should develop meta-knowledge:
- Which sources are reliable? (track verification outcomes)
- Which domains have high uncertainty? (focus learning there)
- What strategies work for resolving what kinds of uncertainty?

This meta-knowledge is ALSO nodes in the graph:
- "Wikipedia is generally reliable for science"
- "To verify a claim, look for primary sources"
- "My memory of childhood events may be reconstructed"

---

## Provenance (Conditional)

### When to Store

Store provenance when:
- **Explicit citation**: Source was specifically mentioned
- **Low certainty**: Need audit trail for uncertain knowledge
- **High stakes**: Medical, legal, financial, safety domains
- **Questionable source**: Need to remember WHY we're uncertain
- **Salient event**: Emotional/memorable learning moment

### When to Discard

Discard provenance when:
- **Deeply consolidated**: High mass + high certainty = "just know it"
- **Many sources**: Provenance becomes "common knowledge"
- **Old and unquestioned**: No need to remember where basic facts came from

### Provenance Structure

```
ProvenanceRecord:
  source_node: node_id | null   # another node, if applicable
  source_type: string           # "observation", "told", "inferred", "read"
  timestamp: datetime
  context: string               # circumstances
  credibility: float            # how reliable was source?
```

---

---

## Relatedness: Structure IS Similarity

### Core Insight

Relatedness isn't computed separately - it's **embodied in the graph structure**.

Similar nodes naturally cluster together through edge formation. The clustering IS the relatedness. You don't ask "how related are A and B?" - you observe:
- Do they share edges?
- Do they share neighbors?
- Are they in the same cluster?
- Is the path between them short?

The graph structure itself IS the "precomputed" similarity.

### How Clusters Form

```
1. Node A is created (e.g., seeing a specific dog)
2. Node A activates, retrieves similar existing nodes
3. Edges form between A and similar nodes
4. Over time, similar nodes accumulate edges to each other
5. A cluster emerges - a region of densely connected nodes
6. The cluster represents a concept (dogs, generally)
```

This is automatic. No explicit clustering algorithm - clustering emerges from edge dynamics.

### What "Similar" Means

Two nodes are similar if:
- They were created in similar contexts
- They share edges to the same nodes
- They're activated by similar queries
- They have overlapping content (LLM can judge)

Similarity isn't one thing - it's multi-dimensional. Nodes can be similar in some ways and different in others.

### Local vs Global Structure

**Local**: Direct edges, shared neighbors (immediate cluster)
**Global**: Path length, cluster membership, hierarchical position

For most operations, local structure matters most. Global structure emerges from local patterns.

### No Separate Embedding?

Maybe we don't need HDV or any embedding at all. The edges ARE the embedding.

If we need fast "find similar nodes" queries:
- Maintain cluster membership as metadata
- Index by high-connectivity hubs
- Use the structure itself

This needs validation through prototyping.

---

## Background Consolidation (Replaces Sleep/Wake)

### Why Humans Sleep

Biological constraint: generating new synapses takes time. Consolidation requires:
- Reduced interference from new inputs
- Time for physical changes to happen
- Energy reallocation

### We Don't Have That Constraint

A computational system can consolidate continuously. No need for explicit "sleep."

### Perpetual Background Process

```python
class ConsolidationDaemon:
    """Runs continuously, throttles during high activity."""

    def run(self):
        while True:
            if system_activity() > HIGH_ACTIVITY_THRESHOLD:
                # Doing stuff - consolidate less aggressively
                sleep(longer_interval)
                continue

            # Low activity - consolidate more
            task = self.pick_task()
            task.execute()
            sleep(short_interval)

    def pick_task(self):
        # Priority order
        candidates = [
            self.find_merge_candidates,      # similar nodes to merge
            self.find_decay_candidates,      # low-energy nodes to prune
            self.find_abstraction_candidates, # clusters needing parents
            self.find_contradiction_candidates, # conflicts to flag
        ]
        return self.highest_priority(candidates)
```

### Throttling During Activity

When the system is actively processing queries or creating nodes:
- Consolidation slows down (avoid interference)
- Focus resources on the active task
- Queue consolidation work for later

When the system is idle:
- Consolidation speeds up
- Work through the backlog
- More aggressive pruning/merging

### Not Sleep, Just Background Work

This isn't sleep - it's continuous maintenance that adapts to load. Like how your computer does garbage collection and indexing in the background.

---

## Hierarchy Emergence: Natural Crystallization

### The Principle

Hierarchy should emerge naturally, not be imposed.

Just as clusters emerge from edge dynamics, hierarchy emerges from cluster dynamics.

### The Process

```
1. CLUSTERING
   - Similar nodes form edges
   - Edges accumulate into clusters
   - Clusters are regions of dense connectivity

2. CRYSTALLIZATION
   - A cluster becomes stable (consistent membership, frequent co-activation)
   - A parent node crystallizes at the "center"
   - Parent connects to all cluster members
   - Parent's content = common attributes of members (LLM infers)

3. RECURSION
   - Parent nodes can themselves cluster
   - Meta-parents emerge (dog → mammal → animal)
   - Hierarchy grows upward organically

4. GROUNDING
   - Leaf nodes are concrete (specific instances, observations)
   - Higher nodes are abstract (categories, principles)
   - The most abstract nodes are near the "root"
```

### When Does Crystallization Happen?

"Crystallization" = a cluster of nodes spawns a parent node to represent the abstraction.

Crystallization is driven by **both**:
1. **Usage patterns** (bottom-up, emergent)
2. **Graph structure analysis** (introspective, reasoned)

These aren't separate - graph analysis is itself based on usage patterns, and the analysis results become nodes too.

**Bottom-up triggers (usage-driven):**

```
Co-activation tracking:
- Nodes X, Y, Z frequently activate together
- This pattern is itself recorded as a node
- [pattern: X+Y+Z co-activate] becomes part of the graph
- When pattern is strong enough, parent node crystallizes
```

**Top-down triggers (reasoning/imagination):**

```
The system can reason about its own knowledge:
- "I notice these concepts share properties"
- "It would be useful to have a word for this cluster"
- "What if I grouped these differently?"

This reasoning is just activation/traversal of meta-knowledge nodes.
Imagination = exploring hypothetical structures before committing.
```

**External triggers (language/social):**

```
Human says "these are called dogs":
- Create parent node with label
- Connect to indicated nodes
- Language bootstraps abstraction
```

**Compression triggers (efficiency):**

```
If representing cluster as single node:
- Reduces redundant edges
- Simplifies reasoning paths
- Then crystallization has structural benefit
```

**Key insight:** The system can reason about its own graph structure because that structure is represented IN the graph (meta-nodes). Analysis isn't a separate process - it's traversal of meta-knowledge. And the results of analysis become new nodes, which can themselves be analyzed.

This is how imagination works: activate hypothetical structures, see what they connect to, evaluate before committing.

### Inheritance and Exceptions

Once hierarchy exists:
- Children inherit edges/attributes from parents
- Exceptions override: `penguin.can_fly = false` despite `bird.can_fly = true`
- Inheritance saves memory: don't repeat shared attributes on every child

### The Exception Problem

From your intuition.md: "A penguin is a bird but cannot fly"

**Important**: Properties are nodes, not attributes. "Flying" is a node. The relationship is an edge - which can be excitatory or inhibitory.

**Solution: Inhibitory edges handle simple negation**

```
[bird] ──(+0.8)──> [flying]       # birds associated with flying
[penguin] ──(+0.9)──> [bird]      # penguin is a bird
[penguin] ──(-0.7)──> [flying]    # penguin INHIBITS flying
```

The direct inhibitory edge from penguin to flying overrides the indirect excitatory path through bird. This is how the brain handles exceptions - direct inhibition.

**Meta-nodes for complex relationships**

For nuanced cases beyond simple negation, use meta-nodes:

```
[conditional_flying]  # meta-node
  ├──> [flying]
  ├──> [weather_conditions]
  content: "can fly when conditions permit"

[injured_bird] ──(+0.8)──> [conditional_flying]
```

Meta-nodes can represent:
- **Conditional**: "X if Y"
- **Temporal**: "X was true, now false"
- **Probabilistic**: "X usually but not always"
- **Contextual**: "X in context Z"

**Gated edges for conditionals**

From your intuition.md: "I can go outside if the weather is nice"

```
[me] ────────────> [going_outside]
        │
        └── gate: [nice_weather]
```

This edge only fires when [nice_weather] is active. The conditional is structural.

**Combining approaches**

- **Simple negation**: inhibitory edge (penguin --(-)-> flying)
- **Conditional**: gated edge (me -> outside, gated on weather)
- **Complex/nuanced**: meta-node (temporal exceptions, probabilistic, contextual)

**Similar-but-distinct concepts**: "flying", "can fly", "flight", "the ability to fly" - these are related but potentially distinct nodes. They'd cluster together but might not merge if used in different contexts or carrying different connotations.

### Hierarchy is Soft

Important: hierarchy isn't rigid classification. It's:
- Emergent from structure
- Revisable (merges can undo, splits can happen)
- Multi-dimensional (a node can belong to multiple clusters/hierarchies)
- Probabilistic (membership by edge strength, not boolean)

A penguin is "mostly a bird" but also connects to "flightless animals", "cold climate animals", etc.

---

## Open Questions

### 1. Gating Mechanics
- How exactly do gated edges work?
- Can gates be complex (multiple conditions)?
- Can gates themselves be gated (nested conditionals)?
- How do gates interact with inhibition?

### 2. Cluster Detection
- How do we efficiently identify clusters?
- Real-time as edges form, or periodic batch?
- What density threshold = "a cluster"?

### 3. Crystallization Trigger
- What makes a cluster "ready" to spawn a parent node?
- How to infer parent content from children?
- How to avoid premature crystallization?

### 4. Fast Similarity Queries
- "Find nodes similar to X" - how to do efficiently?
- Index by cluster membership?
- Use graph structure directly (shared neighbors)?
- Do we need any precomputed representation?

### 5. Similar-but-Distinct Concepts
- "flying" vs "can fly" vs "flight" vs "ability to fly"
- When should these remain separate vs. merge?
- How to maintain useful distinctions while allowing clustering?

### 6. Competitive Activation Tuning
- How many spread iterations?
- What threshold for "activated"?
- How to balance context vs. edge strength?

### 7. Edge Strength Dynamics
- What's the uniform decay rate?
- How much does co-activation strengthen (STDP-like)?
- Is there a maximum/minimum strength?
- How does timing of co-activation affect strength changes?

### 8. Derived Property Computation
- How often to recompute importance, accessibility, etc.?
- Cache and invalidate? Compute on-demand?
- What's the performance impact?

### 9. Traversal Stochasticity
- Fully deterministic? (variability from context only)
- Stochastic at edge level? (strength as probability)
- Deterministic with stochastic tiebreaking?

### 10. Hyperbolic Positioning
- Still useful for hierarchy visualization?
- Derived from graph structure how?
- Or drop entirely in favor of pure graph structure?

---

## Universal Substrate: Beyond Memory

The graph isn't just memory - it's a universal substrate for cognition. Everything is a node, everything obeys the same dynamics.

### Workflows and Processes

A workflow is a sequence of nodes with gated edges:

```
[step_1] ──(gate: step_1_complete)──> [step_2] ──(gate: step_2_complete)──> [step_3]
```

- Each step is a node (possibly executable)
- Edges are gated on completion/conditions
- The workflow itself can be a meta-node connecting all steps
- Workflows can branch (multiple gated edges from one step)
- Workflows can loop (edge back to earlier step, gated on retry condition)

Activation flows through the workflow as gates open.

### Decisions

A decision is a node with multiple outgoing gated edges:

```
[decision_point]
  ├──(gate: condition_A)──> [outcome_A]
  ├──(gate: condition_B)──> [outcome_B]
  └──(gate: condition_C)──> [outcome_C]
```

When [decision_point] activates:
- Check which gates are open (which conditions are active)
- Activation flows down open paths
- If multiple gates open, activation splits (or competes)

Decision-making is just gated spreading activation.

### Predictions

A prediction is a node representing a future state:

```
[current_state] ──(+0.7)──> [predicted_future_state]
                    │
                    └── content: "if X continues, then Y"
```

- Predictions connect to causal factors
- Edge strength = confidence in prediction
- Predictions can be validated (actual outcome strengthens/weakens edge)
- Wrong predictions create inhibitory edges to that prediction pattern

### Tools

Tools are nodes with executable content:

```
[tool_node]
  content: "Searches the web for information"
  implementation: function/API reference

  # Inputs (edges FROM input types TO tool)
  [query_text] ──> [tool_node]

  # Outputs (edges FROM tool TO output types)
  [tool_node] ──> [search_results]
```

- Tool selection = which tool node gets activated by the query
- Tool execution = running the implementation
- Tool learning = strengthening edges to tools that work, weakening edges to tools that fail

### Sense of Self

The "self" is a cluster of nodes:

```
[SELF] (meta-node for the cluster)
  ├──> [my_capabilities]
  │      ├──> [I_can_search]
  │      ├──> [I_can_reason]
  │      └──> [I_cannot_feel_pain]  (inhibitory self-knowledge)
  │
  ├──> [my_preferences]
  │      ├──> [I_prefer_accuracy]
  │      └──> [I_dislike_ambiguity]
  │
  ├──> [my_history]
  │      ├──> [I_helped_user_with_X]
  │      └──> [I_made_mistake_on_Y]
  │
  └──> [my_identity]
         ├──> [I_am_an_AI]
         ├──> [I_am_a_memory_system]
         └──> [my_purpose]
```

Self-knowledge is just knowledge. It clusters, consolidates, and evolves like any other concept.

### Meta-Knowledge

Knowledge about knowledge - nodes about nodes:

```
[I_know_that] ──> [dogs_bark]           # knowing a fact
[I_am_uncertain_about] ──> [X]          # uncertainty awareness
[I_learned_from] ──> [source_Y]         # provenance awareness
[this_conflicts_with] ──> [other_node]  # contradiction awareness
```

Meta-knowledge enables:
- Reasoning about confidence
- Source evaluation
- Contradiction detection
- Learning strategy selection

### Awareness of Own Graph

The graph can contain representations of itself:

```
[my_graph] (node representing the entire graph)
  ├──> [my_clusters]
  │      ├──> [dog_knowledge_cluster]
  │      └──> [coding_skills_cluster]
  │
  ├──> [my_operations]
  │      ├──> [consolidation_process]
  │      ├──> [decay_process]
  │      └──> [retrieval_process]
  │
  └──> [my_agents]
         ├──> [consolidation_daemon]
         ├──> [curiosity_agent]
         └──> [pruning_agent]
```

This is self-modeling. The system has a map of itself.

### Agents as Nodes (Emergent Self-Alteration)

The agents that reorganize the graph are themselves nodes:

```
[consolidation_daemon]
  content: "Merges similar nodes during low activity"
  implementation: consolidation_algorithm

  # Configuration as edges
  [consolidation_daemon] ──> [similarity_threshold_0.8]
  [consolidation_daemon] ──> [runs_during_idle]

  # Can be modified!
  [user_preference] ──> [consolidation_daemon]  # user influences behavior
  [consolidation_daemon] ──> [consolidation_daemon]  # self-reference
```

**Emergent self-modification**:

1. Agents observe graph dynamics (through meta-knowledge nodes)
2. Agents notice patterns ("consolidation is too aggressive")
3. Agents can modify their own configuration nodes
4. Or spawn new agent nodes with different parameters
5. Successful modifications strengthen, unsuccessful ones decay

The system can evolve its own maintenance processes.

### The Recursive Foundation

Everything bottoms out in nodes and edges:

```
Concepts      = nodes + edges
Properties    = nodes + edges
Workflows     = nodes + gated edges
Decisions     = nodes + gated edges
Tools         = nodes + edges + implementation
Self          = cluster of nodes
Meta-knowledge = nodes about nodes
Agents        = nodes + edges + implementation
The graph     = node representing itself
```

**One substrate. One set of dynamics. Emergent complexity.**

### Implications

1. **No special cases**: Everything obeys the same physics (decay, reinforcement, consolidation)

2. **Self-improvement**: The system can reason about and modify its own processes

3. **Unified cognition**: Memory, reasoning, planning, self-awareness are not separate systems

4. **Emergence**: Complex behaviors emerge from simple node/edge dynamics

5. **Introspection**: The system can query its own structure and processes

### Open Questions for Universal Substrate

- How do executable nodes (tools, agents) get their implementation? Bootstrap? Learned? Given?
- How to prevent runaway self-modification? (agent modifies itself to remove safety constraints)
- Where is the boundary between graph content and graph infrastructure?
- How does the graph "run"? What's the execution model outside the graph?

---

## Beyond Human: Computational Improvements

The sections above describe a system **inspired by** human cognition. This section describes where we can **improve upon** it.

### What We Emulate (Brain-Inspired)

| Human Limitation | Our Emulation |
|------------------|---------------|
| Discrete concepts stored in neurons | Nodes as discrete concepts |
| Synaptic connections with weights | Directed edges with strength |
| Excitatory/inhibitory synapses | Signed edge strength |
| Memory consolidation during sleep | Background consolidation process |
| Forgetting unused information | Energy decay |
| Strengthening used memories | Mass/energy reinforcement |
| Associative recall | Spreading activation |
| Hierarchical categorization | Emergent hierarchy from clusters |
| Conditional reasoning | Gated edges |
| Sense of self | Self-referential node cluster |

These are features, not bugs. They enable generalization, creativity, and robustness.

### Where We Improve (Computational Advantages)

#### 1. Step Types: Fuzzy, Deterministic, or Agentic

Humans reason fuzzily - we don't execute code in our heads. Our steps can be different types:

```
[fuzzy_step]
  type: "fuzzy"
  # LLM-like reasoning
  # May produce more steps as output
  # e.g., "understand user intent" → spawns sub-steps

[deterministic_step]
  type: "deterministic"
  # Code execution
  # Precise, repeatable
  # e.g., execute_sql_query()

[agentic_step]
  type: "agentic"
  # Hybrid: LLM + tools + sub-workflows
  # Structured but flexible
  # e.g., "research this topic" (uses search, reading, synthesis)
```

A step is ONE of these types, not all at once:
- **Fuzzy**: LLM reasoning, may decompose into more steps
- **Deterministic**: Code execution, precise output
- **Agentic**: Structured workflow that combines fuzzy + deterministic sub-steps

**Crucially**: Flow between steps is **deterministic**, defined by graph structure. The LLM doesn't decide "what next?" at runtime - the edges and gates determine flow, just like how humans mentally represent workflows as fixed sequences. The fuzzy reasoning happens WITHIN a step, not in deciding which step comes next.

#### 2. Recursive Decomposition as First-Class

Humans break problems into subproblems mentally. We make this explicit:

```
[solve_big_problem]
  ├──> [subproblem_1]
  │      ├──> [sub_subproblem_1a]
  │      └──> [sub_subproblem_1b]
  ├──> [subproblem_2]
  └──> [subproblem_3]

  # Each subproblem is itself a node (potentially with sub-nodes)
  # Solutions propagate back up
  # The decomposition IS the reasoning trace
```

Benefits:
- Decomposition is inspectable (not hidden in "thinking")
- Subproblems can be cached and reused
- Failed subproblems can be retried independently
- The structure of the problem becomes part of the graph

#### 3. Parallel Reasoning (Not Just Sequential)

Human reasoning is largely sequential - one thought at a time. We can parallelize:

```
[problem]
  ├──(parallel)──> [approach_A] ──> [result_A]
  ├──(parallel)──> [approach_B] ──> [result_B]
  └──(parallel)──> [approach_C] ──> [result_C]
                          │
                          ▼
                   [synthesize_results]
```

- Multiple reasoning paths execute simultaneously
- Independent subproblems solved in parallel
- Results merged/compared/synthesized
- Faster AND more thorough (explore multiple approaches)

This is like having multiple "minds" working on the same problem.

#### 4. Archival and Retrieval (Not Tunable Decay)

Humans forget. That's good - noise reduction, generalization, cognitive hygiene. **We emulate this faithfully.**

**Decay is NOT type-based.** In the brain, synapses don't know what "type" of information they encode. What determines decay:

- **Uniform baseline decay**: All synapses weaken over time (protein turnover)
- **Activity counteracts decay**: Used connections strengthen, unused weaken
- **STDP**: Timing of co-activation affects strength changes
- **Homeostatic regulation**: Global scaling, not type-specific

So our model should have:
- **Uniform decay rate** for all edges (not tunable per-type)
- **Reinforcement through use** (accessing strengthens)
- **Timing sensitivity** (edges formed during strong activation start stronger)

**The improvement** isn't tunable decay - it's **archival**:

```
[active_graph]
  # Normal dynamics: decay, reinforcement, consolidation
  # Things naturally forgotten if unused

[archive]
  # Explicit snapshots preserved outside normal dynamics
  # Not part of active retrieval
  # Can be restored if needed
  # Like external storage, not like memory
```

This is more like writing something down vs. remembering it. The archive is external to the cognitive system - it's a tool, not a different type of memory.

**We don't improve upon decay itself. Decay is a feature, not a bug.**

#### 5. Explicit Uncertainty Quantification

Humans have intuitions about confidence but can't quantify them. We can:

```
[claim]
  certainty: 0.73
  provenance: [source_A (credibility: 0.9), source_B (credibility: 0.4)]

  # Can compute: why am I 73% confident?
  # Can update: new evidence shifts certainty precisely
```

Enables:
- Calibrated confidence
- Explicit reasoning about uncertainty
- Precise belief updating

#### 6. Transactional Reasoning (Rollback)

Humans can't easily "undo" a train of thought. We can:

```
[reasoning_branch]
  checkpoint: [graph_state_before]

  # Explore speculative reasoning
  # If it leads nowhere: rollback to checkpoint
  # If it succeeds: commit changes
```

This enables:
- Safe exploration of hypotheticals
- "What if" reasoning without polluting the graph
- Easy recovery from reasoning dead-ends

#### 7. Shared Knowledge (Multi-Agent)

Humans share knowledge through slow, lossy communication. Multiple agents can share a graph:

```
[shared_knowledge_cluster]
  accessible_by: [agent_1, agent_2, agent_3]

  # All agents see the same knowledge
  # Changes by one are visible to others
  # No lossy translation between minds
```

Enables true collaborative cognition, not just communication.

#### 8. Inspectable Reasoning

Human reasoning is opaque (even to ourselves). Our graph is inspectable:

```
Query: "Why do you believe X?"

Response: [path from evidence to X]
  [observation_1] ──(+0.8)──> [intermediate_conclusion]
  [intermediate_conclusion] ──(+0.6)──> [X]
  [observation_2] ──(+0.3)──> [X]

  # Full audit trail of reasoning
```

### Summary: Emulate vs. Improve

| Aspect | Emulate Human | Improve Upon |
|--------|---------------|--------------|
| Concept storage | Nodes as discrete concepts | - |
| Associations | Weighted edges | - |
| Forgetting | Decay dynamics | Faithful emulation + external archive |
| Consolidation | Background process | Continuous, not sleep-bound |
| Reasoning | Spreading activation | + Parallel paths |
| Step execution | Fuzzy/intuitive | + Deterministic, + Agentic types |
| Step flow | Sequential, structured | Parallel when independent |
| Decomposition | Implicit | Explicit, inspectable, cacheable |
| Confidence | Intuitive | Quantified |
| Undo | Difficult | Transactional rollback |
| Sharing | Lossy communication | Shared graph |
| Introspection | Limited | Full inspection |

**Principle**: Start with brain-inspired foundations (proven by evolution), then augment with computational superpowers where it makes sense.

**Key constraint**: Flow between steps is deterministic (graph-structural), not decided by LLM at runtime. Fuzzy reasoning happens WITHIN steps, not in navigation between them.

---

## References to Other Docs

- `intuition.md`: Raw ideas on inheritance, exceptions, conditionals, history
- `paradigm.md`: The 15 laws, full framework description
