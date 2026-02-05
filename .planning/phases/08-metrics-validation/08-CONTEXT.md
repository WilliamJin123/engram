# Phase 8: Metrics & Validation - Context

**Gathered:** 2026-02-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Quantify how interference retrieval performs against baselines (cosine similarity, random, recency) and document degradation curves across noise levels. Phase 7 measured system behavior; Phase 8 compares retrieval methods to validate the interference approach.

</domain>

<decisions>
## Implementation Decisions

### Baseline definitions
- **Cosine similarity baseline**: Claude's discretion on sparse vs dense representation (pick most fair comparison)
- **Random baseline**: Test BOTH random ranking (shuffle results) and random pattern selection
- **Recency baseline**: Include as third baseline - ranks by access time
- **Memory state**: Test baselines on BOTH clean and noisy memory to show degradation across all methods

### Degradation curves
- **Granularity**: Both 4-level summary (NONE/LOW/MEDIUM/HIGH) AND fine-grained curves (10+ data points)
- **X-axis**: Two curve types - total noise patterns AND near-miss count specifically
- **Y-axis**: Both target rank AND Recall@K metrics
- **Output**: Generate matplotlib plots (PNG/SVG) saved alongside data

### Success metrics
- **Gap threshold**: Statistical significance (p < 0.05 across multiple trials)
- **Assertions**: Soft metrics only - document results without failing tests (Phase 8 is measurement)
- **Criteria**: Keep all three from Phase 7 - strict (top-1), relaxed (top-3), and relative
- **Trial count**: Claude's discretion based on runtime and statistical needs

### Documentation format
- **Location**: Markdown report AND raw JSON AND graphs all in tests/stress/ directory
- **Generation**: On-demand flag (pytest --update-report) - not auto-regenerated on every run
- **Comparison**: Standalone report - Phase 8 is self-contained, Phase 7 is separate reference

### Claude's Discretion
- Cosine baseline implementation (sparse vs dense vectors)
- Trial count for statistical significance (balance runtime vs confidence)
- Exact fine-grained data points for curves
- Statistical test choice (t-test, Mann-Whitney, etc.)

</decisions>

<specifics>
## Specific Ideas

- Use Phase 7's TEST_SUMMARY.md metrics as reference (degradation curve: 1->6->9 rank for 1->3->5 near-misses)
- The key question is: "Does interference retrieval beat cosine/random/recency under noise?"
- Phase 8 validates the APPROACH, not the current parameter tuning

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 08-metrics-validation*
*Context gathered: 2026-02-05*
