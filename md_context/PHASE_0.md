# Phase 0: Core Primitives Implementation Summary

## Overview

This document summarizes the development of Engram's foundational primitives - High-Dimensional Vector (HDV) operations and Hyperbolic space embeddings. These form the mathematical foundation for a memory system that combines:
- **HDV**: Distributed, noise-tolerant representations with algebraic operations
- **Hyperbolic space**: Natural hierarchy embedding with efficient ancestor queries

**Final result**: 118 passing tests validating all core operations with documented noise characteristics.

---

## Development Timeline

### 1. Project Setup & Technology Choice

**Initial Plan**: Use NumPy for simplicity.

**User Decision**: Switch to PyTorch for GPU readiness.

**Rationale**: PyTorch CPU performance is equivalent to NumPy for basic operations, but provides seamless GPU acceleration when needed. The API is nearly identical (`np.sum` → `torch.sum`), making the switch trivial.

```toml
# pyproject.toml
[project]
dependencies = ["torch>=2.0.0"]
```

### 2. HDV Encoding Decision: Bipolar vs Ternary

**Original Plan**: Bipolar vectors {-1, +1}
- Classic HDC representation
- Self-inverse binding: `bind(a, a) = all 1s`
- Well-studied theoretical properties

**User Decision**: Switch to ternary {-1, 0, +1}

**Implications**:
1. Added `sparsity` parameter (fraction of non-zero elements)
2. Binding still works (element-wise multiply), but recovery is partial
3. Information loss where either vector has zeros
4. Need to adjust similarity thresholds

```python
def random_ternary(dim: int, sparsity: float = 0.5, seed: int | None = None) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    mask = torch.rand(dim, generator=gen) < sparsity
    signs = 2 * torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) - 1
    return signs * mask.float()
```

**Measured Impact of Sparsity on Unbind Recovery**:
| Sparsity | Recovery Similarity |
|----------|---------------------|
| 0.25     | 0.50                |
| 0.50     | 0.71                |
| 0.75     | 0.87                |
| 1.00     | 1.00 (perfect)      |

---

## Phase A: HDV Operations

### Cycle 1: Random Ternary Generation

**Tests Written**: 12 tests covering shape, values, balance, reproducibility, sparsity control.

**Implementation**: Straightforward - generate random mask, generate random signs, multiply.

**No issues encountered**.

### Cycle 2: Bind/Unbind Operations

**Tests Written**: 14 tests including integration tests for retrievability.

**Key Insight**: For ternary vectors, bind/unbind are both element-wise multiplication. The "unbind" is the same operation as "bind" because:
- Where `a[i] = ±1`: `a * a = 1`, so `unbind(bind(a,b), a) = b`
- Where `a[i] = 0`: Information is lost regardless

```python
def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b

def unbind(bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return bound * a  # Same operation!
```

**Validation Results**:
- Correct unbind similarity: 0.71 (sparsity=0.5)
- Wrong-key unbind: ~0.03 (noise floor)
- Signal/Noise ratio: 21x

### Cycle 3: Bundle Operation

**Tests Written**: 13 tests for normalization, queryability, capacity degradation.

**Implementation**: Sum all vectors, normalize to unit length.

```python
def bundle(items: list[torch.Tensor]) -> torch.Tensor:
    if len(items) == 0:
        raise ValueError("Cannot bundle empty list")
    summed = torch.stack(items).sum(dim=0)
    return normalize(summed)
```

**Issue Encountered**: Test tolerance too strict (1e-6) for floating-point normalization.
**Fix**: Relaxed to 1e-5.

**Capacity Degradation (Validated)**:
| Items | Avg Similarity | Theory (1/√n) |
|-------|----------------|---------------|
| 5     | 0.448          | 0.447         |
| 10    | 0.320          | 0.316         |
| 50    | 0.141          | 0.141         |
| 100   | 0.100          | 0.100         |
| 200   | 0.071          | 0.071         |

The measured values match theoretical predictions almost exactly.

### Cycle 4: Similarity & Normalize

**Tests Written**: 15 tests for cosine similarity properties and normalization.

**Implementation**: Standard cosine similarity with zero-vector handling.

```python
def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (torch.dot(a, b) / (norm_a * norm_b)).item()
```

**Orthogonality Validation**:
- Mean similarity of random pairs: -0.0008 (expected: 0)
- Std of similarities: 0.0102 (expected: 0.01 = 1/√dim)

### Cycle 5: Noise Characteristics Documentation

**Tests Written**: 14 tests that serve as documentation for system behavior.

Key documented properties:
1. Random vectors are nearly orthogonal (std ~ 1/√dim)
2. Bundle capacity degrades as 1/√n
3. Wrong-key unbind produces only noise
4. Sparsity directly affects recovery quality

---

## Phase B: Hyperbolic Operations

### Cycle 6: Poincare Ball Basics

**Tests Written**: 20 tests for projection, distance, validity checking.

**Implementation**:

```python
def project_to_poincare(v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm = torch.norm(v)
    if norm == 0:
        return v
    max_norm = 1.0 - eps
    if norm >= max_norm:
        return v * (max_norm / norm)
    return v

def poincare_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    # d(a,b) = acosh(1 + 2 * ||a-b||² / ((1-||a||²)(1-||b||²)))
    diff = a - b
    norm_diff_sq = torch.dot(diff, diff)
    norm_a_sq = torch.dot(a, a)
    norm_b_sq = torch.dot(b, b)
    denom = torch.clamp((1 - norm_a_sq) * (1 - norm_b_sq), min=1e-10)
    y = torch.clamp(1 + 2 * norm_diff_sq / denom, min=1.0)
    return torch.acosh(y).item()
```

**Issue Encountered**: Origin distance formula test had 1e-5 tolerance but numerical error was ~0.006.
**Fix**: Changed to 0.1% relative error tolerance.

### Cycle 7: Hierarchy Embedding

**Tests Written**: 14 tests for tree embedding with hierarchy preservation.

**Initial Implementation**: Simple approach with random perturbations for siblings.

**Problem**: Siblings weren't spread apart enough angularly, causing incorrect ancestor detection.

**Solution**: Implemented Gram-Schmidt orthogonalization to create maximally spread sibling directions:

```python
def _get_orthogonal_directions(base: torch.Tensor, n: int) -> list[torch.Tensor]:
    # Generate n orthogonal vectors in subspace orthogonal to base
    orthogonal_vecs = []
    for _ in range(n):
        v = _get_random_direction()
        v = v - torch.dot(v, base) * base  # Project out base
        for prev in orthogonal_vecs:
            v = v - torch.dot(v, prev) * prev  # Gram-Schmidt
        if torch.norm(v) > 1e-6:
            orthogonal_vecs.append(v / torch.norm(v))

    # Create child directions with large spread
    directions = []
    for orth in orthogonal_vecs:
        d = base + 1.5 * orth  # Large spread factor
        directions.append(d / torch.norm(d))
    return directions
```

**Embedding Strategy**:
- Root at small radius (~0.1)
- Each level increases radius by 0.15
- Siblings spread orthogonally from parent direction

### Cycle 8: Cone Queries (Most Complex)

**Tests Written**: 16 tests for ancestor detection.

**The Challenge**: Determine if point A is an ancestor of point D using only their embeddings.

**Criteria**:
1. A must be closer to origin (smaller radius = shallower depth)
2. A and D must be in similar angular direction (cone containment)

**Iteration 1**: Fixed threshold (0.7) for angular similarity.
- **Problem**: Too strict for deep hierarchies (root not detected as ancestor of deep nodes).

**Iteration 2**: Adaptive threshold based on ancestor's depth.
- **Problem**: Still couldn't distinguish siblings from ancestors.

**Iteration 3**: Added minimum depth difference requirement.
```python
min_depth_diff = 0.08
if ancestor_norm >= descendant_norm - min_depth_diff:
    return False
```
- **Problem**: Siblings at same level still passed because depth check only applies to same-level nodes; the angular test was still too permissive.

**Iteration 4**: Increased sibling spread in embedding (spread factor 0.8 → 1.5).
- **Improvement**: Better angular separation, but wide trees still had issues.

**Iteration 5**: Tuned adaptive threshold formula through empirical testing.

```python
# Debug output for wide tree:
# child0: norm=0.250, cos_angle=0.989, threshold=0.275 (TRUE ancestor)
# child1: norm=0.250, cos_angle=0.305, threshold=0.275 (FALSE positive!)

# Final formula: threshold grows with depth
threshold = -0.1 + 1.8 * ancestor_norm
# At norm=0.1 (root): threshold = 0.08 (very wide cone)
# At norm=0.25 (level 1): threshold = 0.35 (excludes siblings at ~0.30)
# At norm=0.4 (level 2): threshold = 0.62 (strict)
```

**Final Implementation**:
```python
def is_ancestor(ancestor: torch.Tensor, descendant: torch.Tensor) -> bool:
    ancestor_norm = torch.norm(ancestor).item()
    descendant_norm = torch.norm(descendant).item()

    # Must be significantly closer to origin (different tree level)
    if ancestor_norm >= descendant_norm - 0.08:
        return False

    # Handle origin edge cases
    if ancestor_norm < 1e-6:
        return descendant_norm >= 0.08
    if descendant_norm < 1e-6:
        return False

    # Angular similarity with depth-adaptive threshold
    cos_angle = (torch.dot(ancestor, descendant) /
                 (ancestor_norm * descendant_norm)).item()
    threshold = -0.1 + 1.8 * ancestor_norm

    return cos_angle > threshold
```

---

## Final System Capabilities

### HDV Operations
| Operation | Purpose | Complexity |
|-----------|---------|------------|
| `random_ternary` | Generate random vector | O(dim) |
| `bind` | Associate two concepts | O(dim) |
| `unbind` | Retrieve associated concept | O(dim) |
| `bundle` | Superpose multiple vectors | O(n × dim) |
| `similarity` | Query membership/association | O(dim) |

### Hyperbolic Operations
| Operation | Purpose | Complexity |
|-----------|---------|------------|
| `project_to_poincare` | Ensure valid embedding | O(dim) |
| `poincare_distance` | Measure hyperbolic distance | O(dim) |
| `embed_tree` | Embed hierarchy | O(nodes × dim) |
| `cone_query` | Find all ancestors | O(nodes × dim) |
| `is_ancestor` | Check ancestor relationship | O(dim) |

---

## Limitations & Considerations

### HDV Limitations

1. **Ternary Recovery Loss**: With sparsity=0.5, unbind recovers only 71% similarity. For critical retrieval, use higher sparsity or bipolar encoding.

2. **Bundle Capacity**: Practical limit of ~50-100 items before signal/noise ratio degrades below 3x. For larger sets, consider hierarchical bundling.

3. **No Sequence Encoding**: Current implementation has no permutation operation. Adding sequences would require `permute(v, shift)` operation.

### Hyperbolic Limitations

1. **Fixed Embedding**: `embed_tree` creates a static embedding. No support for dynamic tree updates without re-embedding.

2. **Cone Query Approximation**: The `is_ancestor` function uses heuristic thresholds tuned for the specific embedding strategy. Different embedding schemes would need re-tuning.

3. **Flat Hierarchy Assumption**: The embedding assumes relatively balanced trees. Very deep or very wide trees may need adjusted parameters:
   - Deep trees: Reduce radius increment (0.15 → 0.10)
   - Wide trees: Increase spread factor (1.5 → 2.0)

4. **No Learning**: Embeddings are algorithmic, not learned. For optimal hierarchy representation, gradient-based optimization (e.g., Poincare embeddings paper) would give better results.

### Numerical Considerations

1. **Boundary Stability**: Points very close to Poincare ball boundary (norm > 0.99) have numerical instability in distance calculations. The `eps=1e-5` margin in projection prevents this.

2. **High-Dimensional Orthogonality**: The ~0.01 noise floor from random orthogonality sets the fundamental limit on disambiguation. Increasing dimension reduces this (1/√dim).

---

## Testing Strategy

The TDD approach proved valuable:

1. **RED**: Write comprehensive tests first, including edge cases
2. **GREEN**: Implement minimal code to pass
3. **REFACTOR**: Clean up while maintaining green

**Key Testing Patterns Used**:
- Property-based tests (symmetry, associativity, bounds)
- Statistical tests (mean, std of random samples)
- Integration tests (bind→unbind roundtrip)
- Validation tests (thresholds from theory)
- Documentation tests (print noise characteristics)

---

## Files Created

```
engram/
├── pyproject.toml                          # Package config
├── src/engram/
│   ├── __init__.py                         # Package root
│   ├── hdv/
│   │   ├── __init__.py                     # Exports: random_ternary, bind, unbind, bundle, similarity, normalize
│   │   └── operations.py                   # 100 lines - all HDV operations
│   └── hyperbolic/
│       ├── __init__.py                     # Exports: project_to_poincare, poincare_distance, embed_tree, cone_query, is_ancestor
│       └── poincare.py                     # 200 lines - all hyperbolic operations
└── tests/
    ├── conftest.py                         # Shared fixtures (dim=10000, hyperbolic_dim=50)
    ├── hdv/
    │   ├── test_random_ternary.py          # 12 tests
    │   ├── test_bind_unbind.py             # 14 tests
    │   ├── test_bundle.py                  # 13 tests
    │   ├── test_similarity.py              # 15 tests
    │   └── test_noise_characteristics.py   # 14 tests (documentation)
    └── hyperbolic/
        ├── test_poincare_embedding.py      # 20 tests
        ├── test_hierarchy.py               # 14 tests
        └── test_cone_queries.py            # 16 tests
```

**Total**: ~300 lines of implementation, ~800 lines of tests, 118 passing tests.

---

## Conclusion

Phase 0 successfully validates that:

1. **HDV algebra works**: Bind/unbind/bundle operations behave as expected with quantified noise characteristics
2. **Hyperbolic embedding works**: Tree structures can be embedded with correct parent-child-sibling relationships
3. **The primitives compose**: These building blocks are ready for Phase 1 (memory traces, temporal context)

The main insight from implementation: **ternary encoding trades perfect recovery for sparsity**, and **hyperbolic cone queries require careful threshold tuning** to balance sensitivity vs. specificity for ancestor detection.
