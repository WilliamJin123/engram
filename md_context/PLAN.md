# Building Engram: Implementation Guide

## Phase 0: Validate Core Assumptions First

Before building the system, prove the primitives work in isolation.

### HDV Validation

```python
# Test 1: Binding preserves retrievability
a = random_bipolar(10000)
b = random_bipolar(10000)
bound = bind(a, b)
recovered = unbind(bound, a)
assert similarity(recovered, b) > 0.9

# Test 2: Bundling preserves queryability
items = [random_bipolar(10000) for _ in range(10)]
bundled = bundle(items)
for item in items:
    assert similarity(bundled, item) > 0.3

# Test 3: Bundling degrades gracefully
for n in [10, 50, 100, 200]:
    items = [random_bipolar(10000) for _ in range(n)]
    bundled = bundle(items)
    avg_sim = mean([similarity(bundled, item) for item in items])
    print(f"n={n}, avg_similarity={avg_sim}")
    # Watch where this breaks down

# Test 4: Local propagation converges
# Create small graph, propagate signals, verify it stabilizes
```

Don't proceed until you understand the noise characteristics of your chosen dimensionality.

### Hyperbolic Validation

```python
# Test: Hierarchy encodes correctly
# Create known tree, embed in Poincaré ball
# Verify: parents closer to origin than children
# Verify: siblings closer to each other than to cousins
# Verify: cone queries return correct ancestors
```

---

## Phase 1: The Minimal Graph

Build the stupidest thing that works.

### Node Structure

```python
@dataclass
class Node:
    id: str
    content: str
    
    base_hdv: np.ndarray      # fixed at creation
    hdv: np.ndarray           # evolves via propagation
    
    mass: float = 0.1
    energy: float = 1.0
    
    properties: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime
    last_accessed: datetime
```

### Edge Structure

```python
@dataclass
class Edge:
    source: str           # node id
    target: str           # node id
    edge_type: str        # "IS_A", "HAS", "OWNS", etc.
    weight: float = 1.0

@dataclass 
class HyperEdge:
    id: str
    edge_type: str
    participants: Dict[str, str]  # role -> node_id
    weight: float = 1.0
```

### Graph Store

```python
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]
    hyperedges: List[HyperEdge]
    
    # Edge type HDVs (fixed)
    edge_type_hdvs: Dict[str, np.ndarray]
    
    def add_node(self, content, context=None) -> Node:
        ...
    
    def add_edge(self, source, target, edge_type) -> Edge:
        ...
    
    def get_neighbors(self, node_id) -> List[Tuple[Node, str]]:
        ...
    
    def similarity_search(self, query_hdv, k=10) -> List[Node]:
        ...
```

No abstractions yet. No sleep agents. No hierarchy. Just nodes, edges, HDVs.

### Validation Checkpoint

- Create 100 nodes manually
- Connect them with edges
- Query by HDV similarity
- Verify sensible results

---

## Phase 2: Propagation Dynamics

Add the local update mechanism.

```python
class Node:
    ...
    
    def receive_signal(self, source_hdv, edge_type_hdv, strength=1.0):
        influence = bind(edge_type_hdv, source_hdv)
        learning_rate = 0.05 / (1 + self.mass)
        self.hdv = normalize(self.hdv + learning_rate * strength * influence)
    
    def propagate(self, graph, decay=0.5):
        if decay < 0.01:
            return
        
        for neighbor, edge_type in graph.get_neighbors(self.id):
            edge_hdv = graph.edge_type_hdvs[edge_type]
            neighbor.receive_signal(self.hdv, edge_hdv, strength=decay)
            # Don't recurse here - queue for batch processing

class Graph:
    ...
    propagation_queue: List[Tuple[Node, float]]  # (node, strength)
    
    def process_propagation(self, max_steps=1000):
        """Process queued propagations in batch."""
        steps = 0
        while self.propagation_queue and steps < max_steps:
            node, strength = self.propagation_queue.pop(0)
            if strength > 0.01:
                node.propagate(self, decay=strength * 0.5)
            steps += 1
```

### Validation Checkpoint

- Create node A connected to B connected to C
- Modify A's HDV directly
- Run propagation
- Verify: B changes more than C, D (unconnected) unchanged
- Verify: System stabilizes (propagation queue empties)

---

## Phase 3: Decay and Reinforcement

Add the energy/mass dynamics.

```python
class Graph:
    ...
    
    def tick(self, dt=1.0):
        """Time passes. Energy decays."""
        dead = []
        
        for node in self.nodes.values():
            # Energy decays
            node.energy *= (0.99 ** dt)
            
            # Mass decays slower
            node.mass *= (0.999 ** dt)
            
            if node.energy < 0.01:
                dead.append(node.id)
        
        # Prune dead nodes
        for node_id in dead:
            self.remove_node(node_id)
    
    def access(self, node_id):
        """Node was used. Reinforce it."""
        node = self.nodes[node_id]
        node.energy = min(1.0, node.energy + 0.2)
        node.last_accessed = now()
    
    def reinforce(self, node_id, success=True):
        """Node was used successfully. Increase mass."""
        node = self.nodes[node_id]
        if success:
            node.mass += 0.1
```

### Validation Checkpoint

- Create 100 nodes
- Access only 20 of them repeatedly
- Run tick() 1000 times
- Verify: accessed nodes survive, others die
- Verify: frequently accessed nodes have higher mass

---

## Phase 4: Basic Queries

Before adding intelligence, make sure you can find things.

```python
class Graph:
    ...
    
    def query_similar(self, query_hdv, k=10, min_energy=0.1):
        """Find similar nodes by HDV."""
        candidates = [
            (n, similarity(query_hdv, n.hdv))
            for n in self.nodes.values()
            if n.energy > min_energy
        ]
        candidates.sort(key=lambda x: -x[1])
        return candidates[:k]
    
    def query_by_content(self, content, k=10):
        """LLM encodes content to HDV, then similarity search."""
        query_hdv = self.encode_content(content)  # LLM boundary
        return self.query_similar(query_hdv, k)
    
    def get_property(self, node_id, prop):
        """Get property with inheritance."""
        node = self.nodes[node_id]
        
        if prop in node.properties:
            return node.properties[prop]
        
        # Walk up IS_A edges
        for parent_id in self.get_parents(node_id):
            result = self.get_property(parent_id, prop)
            if result is not None:
                return result
        
        return None
```

### Validation Checkpoint

- Create nodes for: fido, rex, bella (dogs), whiskers (cat)
- Manually connect them with IS_A, HAS edges
- Query "furry pet" - verify dogs and cat returned
- Query "barks" - verify only dogs returned

---

## Phase 5: First Sleep Agent (Clustering)

Now add intelligence, but simple.

```python
class ClusteringAgent:
    def __init__(self, graph, llm=None):
        self.graph = graph
        self.llm = llm  # optional, for naming
    
    def find_clusters(self, min_similarity=0.6, min_size=3):
        """Find groups of similar nodes without parents."""
        orphans = [n for n in self.graph.nodes.values() 
                   if not self.graph.get_parents(n.id)]
        
        clusters = []
        used = set()
        
        for node in orphans:
            if node.id in used:
                continue
            
            # Find similar orphans
            cluster = [node]
            for other in orphans:
                if other.id != node.id and other.id not in used:
                    if similarity(node.hdv, other.hdv) > min_similarity:
                        cluster.append(other)
            
            if len(cluster) >= min_size:
                clusters.append(cluster)
                used.update(n.id for n in cluster)
        
        return clusters
    
    def create_parent(self, cluster):
        """Extract shared properties, create parent node."""
        
        # Analyze shared properties
        shared, variable, sparse = analyze_properties(cluster)
        
        # Create parent
        parent = self.graph.add_node(
            content=None,  # unnamed for now
            context=[(n, "CHILD") for n in cluster]
        )
        parent.properties = shared
        parent.hdv = bundle([n.hdv for n in cluster])
        
        # Link children
        for child in cluster:
            self.graph.add_edge(child.id, parent.id, "IS_A")
            # Remove redundant properties from child
            for prop in shared:
                if child.properties.get(prop) == shared[prop]:
                    del child.properties[prop]
        
        # Name it (LLM boundary)
        if self.llm:
            parent.content = self.llm.name_cluster(shared, cluster)
        
        return parent
    
    def run(self):
        """One pass of clustering."""
        clusters = self.find_clusters()
        created = []
        for cluster in clusters:
            parent = self.create_parent(cluster)
            created.append(parent)
        return created
```

### Validation Checkpoint

- Create 20 dog nodes, 15 cat nodes, 10 bird nodes (manually, with similar properties)
- Run clustering agent
- Verify: three parent nodes created
- Verify: children linked correctly
- Verify: shared properties moved to parents

---

## Phase 6: More Sleep Agents

Add incrementally, validating each.

### Contradiction Detector

```python
class ContradictionAgent:
    def find_contradictions(self, similarity_threshold=0.7):
        """Find nodes that are similar but have conflicting properties."""
        contradictions = []
        
        nodes = list(self.graph.nodes.values())
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                sim = similarity(a.hdv, b.hdv)
                if sim > similarity_threshold:
                    conflicts = find_property_conflicts(a, b)
                    if conflicts:
                        contradictions.append((a, b, conflicts))
        
        return contradictions
```

### Pruning Agent

```python
class PruningAgent:
    def find_dead_nodes(self, energy_threshold=0.05):
        return [n for n in self.graph.nodes.values() 
                if n.energy < energy_threshold]
    
    def find_orphan_clusters(self):
        """Nodes disconnected from main graph."""
        ...
    
    def find_redundant_nodes(self):
        """Nodes that are ~identical to another node."""
        ...
```

### Abstraction Agent

```python
class AbstractionAgent:
    def find_implicit_principles(self):
        """Look for patterns in constraints/rules that suggest higher principle."""
        ...
    
    def propose_abstraction(self, pattern):
        """Create a more abstract node that captures the pattern."""
        ...
```

---

## Phase 7: Working Agent Integration

Now connect to actual usage.

```python
class WorkingAgent:
    def __init__(self, graph, llm):
        self.graph = graph
        self.llm = llm
    
    def query(self, question):
        """Answer question using graph."""
        # Encode question
        query_hdv = self.encode(question)
        
        # Find relevant nodes
        relevant = self.graph.query_similar(query_hdv, k=20)
        
        # Access them (reinforcement)
        for node, _ in relevant:
            self.graph.access(node.id)
        
        # LLM synthesizes answer
        context = self.format_nodes(relevant)
        return self.llm.answer(question, context)
    
    def add_knowledge(self, content, source="user"):
        """Add new knowledge to graph."""
        # LLM extracts structured info
        entities, properties, relationships = self.llm.parse(content)
        
        # Create nodes
        for entity in entities:
            node = self.graph.add_node(entity.content)
            node.properties = entity.properties
        
        # Create edges
        for rel in relationships:
            self.graph.add_edge(rel.source, rel.target, rel.type)
        
        # Propagate
        self.graph.process_propagation()
```

---

## On Seeding Primitives

### Do you need to pre-install abstract concepts?

**Arguments for seeding:**
- Bootstrap problem: clustering needs similar nodes to exist first
- Edge types need base HDVs anyway
- Core concepts (IS_A, HAS, etc.) are universal

**Arguments against seeding:**
- You want emergence, not engineering
- Seeded concepts might not match actual usage
- Bias toward your ontology vs discovered ontology

### Recommendation: Minimal Seeding

Seed only structural primitives, not domain concepts:

```python
def initialize_graph():
    graph = Graph()
    
    # Edge type HDVs (required)
    graph.register_edge_type("IS_A")
    graph.register_edge_type("HAS")
    graph.register_edge_type("CAUSES")
    graph.register_edge_type("CONTRADICTS")
    graph.register_edge_type("RELATED")
    
    # Meta types (for self-description)
    graph.add_node("NODE_TYPE", properties={"meta": True})
    graph.add_node("EDGE_TYPE", properties={"meta": True})
    graph.add_node("AGENT", properties={"meta": True})
    
    # That's it. No "animal", no "tool", no domain concepts.
    
    return graph
```

Let "dog" emerge from seeing dogs. Let "combat_system" emerge from combat nodes clustering. Don't impose structure.

---

## Validation Strategy

### Level 1: Unit Tests

- HDV operations preserve expected properties
- Propagation converges
- Decay works as expected

### Level 2: Integration Tests

- Create known graph, verify queries return correct results
- Add contradictory information, verify detection
- Simulate usage patterns, verify survival of useful nodes

### Level 3: Behavioral Tests

- Feed system 100 dog facts, verify "dog" concept emerges
- Feed system contradictory facts, verify resolution or flagging
- Simulate active development session, verify coherence maintained

### Level 4: Adversarial Tests

- Feed garbage, verify graceful degradation
- Rapidly change structure, verify stability
- Create pathological graphs (huge clusters, long chains), verify performance

### Level 5: Real Usage

- Actually use it for game design
- Track: query relevance, contradiction frequency, hierarchy quality
- Adjust parameters based on actual behavior

---

## Build Order Summary

```
1. HDV primitives        → validate math works
2. Graph + nodes + edges → validate storage works
3. Propagation           → validate locality works
4. Decay/reinforcement   → validate dynamics work
5. Queries               → validate retrieval works
6. Clustering agent      → validate emergence works
7. More sleep agents     → validate consolidation works
8. Working agent         → validate real usage works
9. Iterate based on actual behavior
```

Don't skip validation. Each layer depends on the previous being solid.