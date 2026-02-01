# Quantum-Inspired Agentic Memory

Discussion document for agentic memory applications. Keep everything in this directory.

---

## Scope

A practical application of the Engram substrate for LLM-based agents. **Philosophically separate** from the base model (which seeks emergent, pure representations from primitives), but shares the same substrate dynamics.

Here we accept imported representations (text from LLMs) and leverage LLMs for semantic heavy lifting while the substrate handles structure.

---

## Why Current Agentic Memory Fails

| Approach | Problem |
|----------|---------|
| RAG / Vector DB | No structure - just chunks ranked by similarity |
| Conversation buffer | No consolidation - grows unbounded or truncates |
| Summary chains | Lossy - can't recover details, no episodic access |
| Knowledge graphs | Rigid schemas - miss fuzzy associations |

All lack: temporal structure, binding (who said what when), graceful decay, interference-based retrieval.

---

## Design Principles

**Division of labor.** LLMs are good at meaning; substrates are good at structure. Use each for what it does best.

| Substrate handles | LLM handles |
|-------------------|-------------|
| Storage, binding, retrieval | Semantic interpretation |
| Temporal structure, decay | Importance judgment |
| Interference, phase dynamics | Topic labeling / clustering |
| Entanglement mechanics | Reflection / abstraction |
| Fast, deterministic | Slow, expensive, smart |

**Text as primary representation.** No LLM embeddings - they import biases and create opaque dependencies. Store actual text; let the substrate handle structure, let LLMs judge meaning when needed.

**Encoding is simple; dynamics are rich.**
```
Text chunk → tokenize → hash to sparse pattern → store with metadata

The magic isn't in encoding. It's in:
- Entanglement (binding text to context: speaker, time, topic)
- Phase relationships (temporal sequences, causation)
- Interference (related memories activate together)
- Decoherence (old, unused memories fade to classical stability)
```

**Structure through binding, not schema.**
```
Traditional KG:  (entity)--[relation]-->(entity)  ← rigid
Engram:          TEXT₁ ⊗ SPEAKER ⊗ TIME ⊗ TOPIC   ← flexible entanglement
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LLM Agent                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Memory Interface                       │
│                                                         │
│  store(text, context)    ───► LLM: importance? topics?  │
│  recall(query)           ───► LLM: rerank? summarize?   │
│  reflect(scope, depth)   ───► LLM: abstract, reorganize │
│                                                         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Engram Substrate                      │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Patterns  │  │ Entanglement│  │  Coherence  │     │
│  │   (sparse)  │◀▶│   (binding) │◀▶│  (dynamics) │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                         │
│  - Fast, deterministic retrieval                        │
│  - Structural queries (what's connected to X?)          │
│  - Temporal ordering and decay                          │
│  - Interference-based activation spreading              │
└─────────────────────────────────────────────────────────┘
```

The LLM is the **judge**; the substrate is the **court records**.

---

## Core Operations

### Store (with LLM pre-processing)

```python
def store(text: str, context: dict):
    # LLM judges importance and extracts structure (cheap, fast model)
    analysis = llm.analyze(f"""
        Text: {text}
        Context: {context}

        Return JSON:
        - importance: 0-1 (how likely to be needed later?)
        - topics: list of topic strings
        - entities: list of named entities
        - supersedes: does this update/contradict recent memory?
    """)

    # Substrate handles the actual storage
    pattern = text_to_pattern(text)
    pattern.coherence = analysis.importance

    # Entangle with topics
    for topic in analysis.topics:
        topic_pattern = get_or_create_topic(topic)
        entangle(pattern, topic_pattern)

    # Entangle with entities
    for entity in analysis.entities:
        entity_pattern = get_or_create_entity(entity)
        entangle(pattern, entity_pattern)

    # Handle supersession (updates/corrections)
    if analysis.supersedes:
        old = find_superseded(analysis.supersedes)
        old.coherence *= 0.5  # decay old version
        entangle(pattern, old, relation="updates")

    # Entangle with context
    entangle(pattern, speaker_pattern(context.speaker))
    entangle(pattern, time_pattern(context.time))

    substrate.store(pattern)
```

### Recall (with LLM post-processing)

```python
def recall(query: str, context: dict = None) -> list[Memory]:
    # Substrate does fast structural retrieval
    query_pattern = text_to_pattern(query)
    candidates = substrate.activate(query_pattern, top_k=20)

    # Include entangled context with each candidate
    for c in candidates:
        c.context = substrate.get_entangled(c, types=["speaker", "time", "topic"])

    # LLM reranks based on semantic relevance to query + current context
    ranked = llm.rerank(f"""
        Query: {query}
        Current context: {context}

        Candidates (with their stored context):
        {format_candidates(candidates)}

        Return: ordered indices by relevance
        Optional: brief synthesis if multiple candidates relate
    """)

    return [candidates[i] for i in ranked.indices]
```

### Reflect (LLM-driven, scoped)

```python
def reflect(scope: str = "recent", depth: str = "light"):
    """
    scope: "recent" | "topic:X" | "entity:X" | "session" | "all"
    depth: "light" (summarize) | "deep" (reorganize)
    """

    # Get relevant patterns from substrate
    if scope == "recent":
        patterns = substrate.get_recent(hours=24)
    elif scope.startswith("topic:"):
        topic = scope.split(":", 1)[1]
        patterns = substrate.get_entangled_with(get_topic(topic))
    elif scope.startswith("entity:"):
        entity = scope.split(":", 1)[1]
        patterns = substrate.get_entangled_with(get_entity(entity))
    elif scope == "session":
        patterns = substrate.get_current_session()
    else:
        patterns = substrate.get_all(coherence_threshold=0.1)

    if depth == "light":
        # Summarize and create abstraction pattern
        result = llm.analyze(f"""
            Memories to consolidate:
            {format_patterns(patterns)}

            Return:
            - summary: concise summary of key points
            - insights: any patterns or notable observations
        """)

        # Create abstraction pattern entangled with sources
        abstract = text_to_pattern(result.summary)
        abstract.coherence = 0.8
        abstract.metadata["type"] = "abstraction"
        for p in patterns:
            entangle(abstract, p, relation="abstracts")
        substrate.store(abstract)

        return result

    elif depth == "deep":
        # LLM proposes reorganization
        result = llm.analyze(f"""
            Memories to analyze:
            {format_patterns(patterns)}

            Propose reorganization:
            - new_topics: [{name, description, member_ids}] - new clusters to create
            - contradictions: [{id1, id2, description, resolution}] - conflicts found
            - redundancies: [{ids, merged_text}] - duplicates to merge
            - abstractions: [{source_ids, abstraction_text}] - patterns to abstract
            - importance_updates: [{id, new_importance, reason}] - reweight
        """)

        apply_reorganization(result)
        return result
```

---

## What This Enables

| Capability | Mechanism |
|------------|-----------|
| "What did the user say about X?" | Query + entanglement retrieval |
| "How has user's view on X changed?" | Temporal phase relationships + LLM synthesis |
| "What's contradictory in my knowledge?" | Deep reflect finds phase opposition |
| "What patterns recur across conversations?" | Reflect abstraction |
| "What's most relevant right now?" | Coherence-weighted activation + LLM rerank |
| Graceful context window | Old stuff decays but remains accessible |
| Automatic organization | LLM-driven topic clustering on reflect |

---

## Why the Substrate Still Matters

Even with LLM doing semantic heavy lifting, the substrate is essential:

| Substrate value | Why LLM can't replace it |
|-----------------|-------------------------|
| Binding structure | LLM has no persistent state |
| Temporal ordering | LLM sees flat context window |
| Interference / resonance | LLM does similarity, not activation spread |
| Graceful decay | LLM has hard context cutoff |
| Fast retrieval | LLM is slow and expensive |
| Entanglement queries | "What's connected to X?" is structural |
| Coherence dynamics | Salience changes over time |

The substrate provides **structure and persistence**. The LLM provides **judgment and meaning**.

---

## Cost Control

LLM calls are expensive. Strategies to mitigate:

| Strategy | Implementation |
|----------|----------------|
| Use small models | `haiku` for routine analysis, `sonnet` for deep reflection |
| Batch operations | Analyze multiple stores together |
| Cache judgments | Same text → same analysis (memoize by hash) |
| Lazy evaluation | Don't analyze until retrieval proves it matters |
| Threshold triggers | Only call LLM when thresholds hit |

### Lazy Importance Pattern

```python
def store_lazy(text: str, context: dict):
    """Store with deferred LLM analysis."""
    pattern = text_to_pattern(text)
    pattern.coherence = 0.5  # neutral default
    pattern.metadata["needs_analysis"] = True
    pattern.metadata["raw_context"] = context
    substrate.store(pattern)

def on_retrieval(pattern):
    """Analyze on first retrieval - now we know it matters."""
    if pattern.metadata.get("needs_analysis"):
        analysis = llm.analyze(pattern.text, pattern.metadata["raw_context"])
        pattern.coherence = analysis.importance
        apply_entanglements(pattern, analysis)
        pattern.metadata["needs_analysis"] = False
```

### Batched Store

```python
class MemoryBuffer:
    """Buffer stores and analyze in batches."""

    def __init__(self, batch_size=10, flush_interval=60):
        self.buffer = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval

    def store(self, text, context):
        self.buffer.append((text, context))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        # Single LLM call for entire batch
        analysis = llm.analyze_batch(self.buffer)

        for (text, context), item_analysis in zip(self.buffer, analysis):
            pattern = text_to_pattern(text)
            pattern.coherence = item_analysis.importance
            apply_entanglements(pattern, item_analysis)
            substrate.store(pattern)

        self.buffer = []
```

---

## Encoding Without Embeddings

Simple hash-based encoding. The LLM handles semantics; this just needs to support overlap detection.

```python
def text_to_pattern(text: str, dim: int = 10000, k: int = 100) -> SparsePattern:
    """Hash text to sparse pattern. No ML, no embeddings."""
    tokens = tokenize(text)  # simple whitespace + punctuation

    # Each token activates bits (seeded by token hash for consistency)
    active_bits = set()
    for token in tokens:
        seed = hash(token)
        rng = Random(seed)
        bits_per_token = max(1, k // len(tokens))
        active_bits.update(rng.sample(range(dim), bits_per_token))

    # Phase from token position (preserves word order)
    phases = {}
    for i, token in enumerate(tokens):
        seed = hash(token)
        rng = Random(seed)
        for bit in rng.sample(range(dim), max(1, k // len(tokens))):
            if bit in active_bits:
                phases[bit] = (i / len(tokens)) * 2 * pi

    return SparsePattern(bits=active_bits, phases=phases, coherence=1.0)
```

Properties:
- Similar text → overlapping bits (Jaccard similarity)
- Word order preserved in phase
- Deterministic (same text → same pattern)
- No external dependencies

---

## Integration Examples

### Drop-in for existing agents

```python
from engram import AgentMemory

memory = AgentMemory(
    llm_client=anthropic_client,
    analysis_model="claude-3-haiku",      # cheap for routine
    reflection_model="claude-sonnet-4-20250514",   # smart for deep work
)

# Store observations
memory.store("User prefers dark mode", {"speaker": "observation"})

# Recall relevant memories
memories = memory.recall("What are the user's UI preferences?")

# Background consolidation
insights = memory.reflect(scope="session", depth="light")
```

### With conversation history

```python
# Instead of sliding window or summarization
for message in conversation:
    memory.store(message.content, {
        "speaker": message.role,
        "time": message.timestamp,
    })

# Retrieve relevant history for context
relevant = memory.recall(current_query, context={"task": current_task})

# Format for LLM context
context = "\n".join(f"[{m.context.speaker}]: {m.text}" for m in relevant)
```

### Reflection triggers

```python
# Event-triggered reflection (not fixed schedule)
class ReflectionTrigger:
    def __init__(self, memory):
        self.memory = memory
        self.unreflected_count = 0
        self.last_reflection = time.now()

    def on_store(self):
        self.unreflected_count += 1

        # Trigger conditions
        if self.unreflected_count > 50:
            self.memory.reflect(scope="recent", depth="light")
            self.unreflected_count = 0

    def on_idle(self, idle_seconds):
        if idle_seconds > 300 and self.unreflected_count > 10:
            self.memory.reflect(scope="session", depth="light")
            self.unreflected_count = 0

    def on_session_end(self):
        self.memory.reflect(scope="session", depth="deep")
```

---

## Multi-Agent Memory

### The Unified Substrate Model

Multiple agents share **one substrate with one global coherence map**. The system isn't a collection of separate minds sharing notes—it's **one mind with multiple foci of attention**.

```
┌─────────────────────────────────────┐
│                                     │
│       Unified Substrate             │
│   (patterns + global coherence)     │
│                                     │
│    ┌─────┐  ┌─────┐  ┌─────┐       │
│    │Focus│  │Focus│  │Focus│       │
│    │  A  │  │  B  │  │  C  │       │
│    └─────┘  └─────┘  └─────┘       │
│                                     │
└─────────────────────────────────────┘
```

### Why Unified (Not Per-Agent Coherence)

| Per-agent coherence | Unified coherence |
|---------------------|-------------------|
| Conflicts hidden between agents | Conflicts surface as phase opposition |
| Agents diverge over time | Agents naturally converge |
| Coordination requires explicit sync | Alignment emerges from shared salience |
| Each agent has separate "truth" | One organizational truth with uncertainty |

**Core principle: Conflicts are features, not bugs.** If Agent A stores "technology X is promising" and Agent B stores "technology X has fatal flaws," both patterns exist with phase opposition. The system *knows* it has contradictory information. `reflect(depth="deep")` surfaces and resolves the contradiction.

### What Is an "Agent"?

An agent is not a separate entity with its own memory. It's a **query context**—a lens through which the unified substrate is accessed.

| Agent property | Implementation |
|----------------|----------------|
| Role | Flavors how queries are formulated |
| Focus | Topics this agent prioritizes in retrieval |
| Attribution | `stored_by` metadata tracks who contributed what |
| Perspective | Different agents may weight the same patterns differently in LLM reranking |

The agents aren't having a conversation *between* separate minds. They're different **aspects** of one mind examining a problem from different angles.

### Scoped Visibility (When Needed)

Even in the unified model, some patterns may need restricted access:

| Scope | Use case | Mechanism |
|-------|----------|-----------|
| `draft` | Work in progress, not ready for others | Metadata flag, excluded until finalized |
| `sensitive` | Role-restricted information | Clearance level in metadata |
| `ephemeral` | Per-query scratch space | Not stored to substrate at all |

These are **metadata filters**, not separate substrates. Patterns still exist in one place; visibility is controlled at query time.

### Organizational Consciousness

This model treats the multi-agent system as one cognitive entity:

| Human analogy | Multi-agent equivalent |
|---------------|------------------------|
| Different brain regions | Different agents (researcher, critic, executor) |
| Unified consciousness | Unified substrate + coherence |
| Internal conflict | Contradictory patterns, phase opposition |
| Resolving conflict | Reflection, coherence competition, measurement |
| Attention | Global coherence allocation |

---

## Measurement and Collapse

### What Triggers Collapse?

In quantum mechanics, measurement correlates system state with a macroscopic record, forcing commitment to a specific outcome. In Engram:

**Measurement = committing to a specific interpretation with external consequences.**

| Event | Collapse? | Reason |
|-------|-----------|--------|
| Activation/query | No | Just computing overlap, no commitment |
| LLM reranking | Partial | LLM selects what's relevant—a soft measurement |
| Agent acts on memory | Yes | External consequence, irreversible |
| Output to user | Yes | Definite commitment, recorded |

### Observe vs Measure

| Mode | What happens | Use case |
|------|--------------|----------|
| **Observe** | Compute activations, return candidates, no state change | Exploration, hypotheticals, "what if" |
| **Measure** | Commit to interpretation, selected patterns shift toward classical | Decision-making, action |

### Collapse Effects

When a pattern is measured (selected and acted upon):

1. **Coherence decays slightly** — commitment to this interpretation reduces superposition
2. **Phase relationships lock in** — the interference pattern that led to selection is reinforced
3. **Recency updated** — pattern is marked as recently relevant

Unmeasured patterns retain their quantum character—they can still interfere in future queries.

### Why Collapse Matters

**Collapse stabilizes frequently-used knowledge.** Patterns that get measured often become more classical (stable, reliable, less prone to interference). Patterns never measured retain uncertainty and may fade through natural decoherence.

**Collapse signals importance for consolidation.** Measured patterns are candidates for abstraction during reflection—they proved useful, so they're worth remembering.

---

## Persistence and Crystallization

### The Metaphor

| Quantum mechanics | Engram |
|-------------------|--------|
| Quantum state (fragile, needs isolation) | High coherence (active interference) |
| Classical state (stable, can be written down) | Low coherence (stable, phase dormant) |

**Crystallization = controlled decoherence for cold storage.**

### How It Works

To persist the substrate:
1. Optionally prune patterns below a coherence threshold (too faded to keep)
2. Serialize patterns with their bits, phases, coherence, metadata, and entanglement links
3. Store as JSON, MessagePack, or any portable format

To restore:
1. Load patterns back into substrate
2. Optionally "reheat" coherence based on time elapsed or importance
3. Restore entanglement links between patterns
4. Resume dynamics

### What Gets Preserved

| Component | Preserved? | Notes |
|-----------|------------|-------|
| Active bits | Yes | The pattern identity |
| Phases | Yes | Even if dormant, may matter when reheated |
| Coherence | Yes | Snapshot of salience at freeze time |
| Metadata | Yes | Attribution, topics, timestamps |
| Entanglements | Yes | As pattern ID references |
| Source text | Yes | For reconstruction and LLM re-analysis |

### Reheating Strategies

| Strategy | Description |
|----------|-------------|
| **Preserve exactly** | Restore coherence as saved |
| **Time decay** | Reduce coherence based on storage duration |
| **Importance boost** | High-importance patterns get reheated to active coherence |
| **Query-driven** | Start cold, reheat patterns as they're accessed |

Crystallization enables: pause/resume execution, serverless operation, memory sharing between systems, backup and recovery.

---

## Open Questions

1. **Entanglement granularity**: Entangle at sentence level? Paragraph? Let LLM decide chunk boundaries?
2. **Reflection depth heuristics**: When is "light" sufficient vs "deep" needed?
3. **Contradiction resolution authority**: When should the agent be consulted vs auto-resolve?
4. **Evaluation metrics**: How do we measure if this is actually better than RAG?
5. **Text encoding strategy**: Hash-based (fast, no semantics) vs LLM-assigned codes (semantic, slower) vs hybrid?
6. **Very long sequences**: Does phase encoding degrade beyond 100-1000 items?
