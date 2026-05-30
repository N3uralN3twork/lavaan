# lav_model_gradient.R Optimization Design

## Goal

Optimize `R/lav_model_gradient.R` for the full FastLavaan 25-scenario profiling
suite while preserving lavaan behavior exactly.

The optimization is successful if it achieves either:

- at least a 10% reduction in pooled median latency across all 25 scenarios,
  computed from the profiling workbook's `latency-samples` sheet; or
- at least a 10% median-latency reduction in one or more scenarios, with pooled
  25-scenario latency non-worse.

Correctness and regression gates are mandatory. A speedup is not accepted if it
changes analytic-gradient results, breaks unit/equivalence tests, or creates a
confirmed scenario regression.

## Measurement Contract

Use the parent FastLavaan profiling harness as the authoritative benchmark
source:

- run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`;
- compare candidate runs against an existing baseline label or a fresh baseline;
- use the workbook `latency-samples` sheet for pooled latency metrics;
- use scenario-level comparison and confirmation output to identify regressions;
- keep `--workers 1` for authoritative latency when running direct scripted
  checks, unless using the Makefile's established comparison workflow.

The main acceptance comparison should use the full 25-scenario suite. Smaller
scenario screens may be used while developing, but they cannot justify keeping
the final patch by themselves.

## Correctness Gates

Before accepting the optimization:

- `make equivalence` must pass from the parent FastLavaan workspace.
- `make unit_tests` must pass from the parent FastLavaan workspace.
- `benchmark/profiling/check-gradient-equivalence.R` must pass.
- `git diff --check` must pass in the nested `lavaan` package repo.

The focused gradient equivalence check compares the current implementation to
`HEAD:R/lav_model_gradient.R`, so use it before and after candidate edits to
isolate behavioral changes.

## Optimization Approach

Use a structural, reusable optimization pass inside `R/lav_model_gradient.R`.
Prefer changes that reduce repeated work across all supported paths rather than
scenario-shaped shortcuts.

Promising areas include:

- hoisting repeated slot reads and simple flags out of hot loops;
- avoiding repeated `names()`, `[[ ]]`, and index-list lookup work inside
  group/model-matrix loops;
- reusing already-computed implied, omega, delta, and conditional-x pieces when
  contracts allow it;
- replacing avoidable intermediate matrices or conversions with cheaper
  equivalent forms;
- keeping branch-specific behavior for ML, WLS/DWLS/ULS/GLS/DLS/NTRLS,
  conditional-x, multilevel, categorical, RAM, composites, and group weights.

Do not broaden the scope into unrelated profiling harness changes or unrelated
lavaan internals unless profiling proves `lav_model_gradient.R` cannot reach
the target on its own.

## Implementation Boundaries

Keep edits scoped to `R/lav_model_gradient.R` unless a tightly coupled helper is
required for a correctness-preserving performance win.

Do not change public APIs, returned gradient shapes, equality-constraint
handling, `type = "free"` versus non-free behavior, group-weight semantics,
or warning/error behavior.

Any fallback for numerically fragile paths must preserve the current result
contract. If a risky path cannot be optimized with confidence, leave it alone.

## Validation Flow

1. Capture or identify the comparison baseline.
2. Run a focused pre-edit gradient equivalence/smoke check if needed.
3. Make one optimization family at a time.
4. Run focused gradient equivalence.
5. Run `make equivalence`.
6. Run `make unit_tests`.
7. Run the full 25-scenario benchmark comparison.
8. Inspect `latency-samples`, scenario comparison, stability confirmation, and
   p90 movement before deciding whether to keep the patch.

If the first structural pass does not produce a clear win, use the workbook and
pprof/stage evidence to choose the next smallest reusable optimization target.

