# lav_model_gradient.R Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the full 25-scenario FastLavaan suite by optimizing gradient-related R paths while preserving exact analytic-gradient behavior.

**Architecture:** Start with low-risk callback-context reuse around `lav_model_gradient()` and `lav_model_estimate()`, because it removes repeated setup across optimizer gradient calls without changing math. Then benchmark; only if the target is not met, move deeper into LISREL derivative helpers such as `lav_lisrel_df_dmlist()` with focused equivalence checks after each optimization family.

**Tech Stack:** R, lavaan S4 internals, parent FastLavaan profiling harness, `testthat`, `pkgload`, `bench`, `readxl`/`openxlsx2`, `make`.

---

## File Structure

- Modify `R/lav_model_gradient.R`: add reusable gradient-context helpers, accept `optim_context`, reuse cached group weights, model-matrix indices, and conditional-x sample statistics.
- Modify `R/lav_model_estimate.R`: build one gradient context per estimation call and pass it into `lav_model_gradient()` from the optimizer callback.
- Read `../tests/unit/test-lav_model_gradient_context.R`: reuse the existing parent-harness contract test. It skips while `lav_model_gradient_context()` is absent, then becomes active once the helper exists.
- Modify `R/lav_representation_lisrel.R` only after Task 2 measurement shows `lav_lisrel_df_dmlist()` remains a live hotspot; otherwise leave it unchanged.
- Do not modify profiling harness files unless a benchmark-reading helper is needed for local analysis; prefer one-off R commands for workbook inspection.

## Task 1: Pre-Edit Baseline And Current-State Checks

**Files:**
- Read: `docs/superpowers/specs/2026-05-30-lav-model-gradient-optimization-design.md`
- Read: `R/lav_model_gradient.R`
- Read: `R/lav_model_estimate.R`
- Read: `R/lav_representation_lisrel.R`
- Read: `../benchmark/profiling/check-gradient-equivalence.R`

- [ ] **Step 1: Confirm worktree scope**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan/lavaan`:

```sh
git status --short
```

Expected: only planned files are dirty. If unrelated user edits appear, leave them untouched and avoid formatting those files.

- [ ] **Step 2: Verify focused gradient equivalence before source edits**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript benchmark/profiling/check-gradient-equivalence.R
```

Expected: prints `Gradient equivalence check passed.`

- [ ] **Step 3: Verify the current parent unit test state**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
make unit_tests
```

Expected: parent unit tests pass. The `test-lav_model_gradient_context.R` test may currently skip the context check because `lav_model_gradient_context()` is absent.

## Task 2: Add Gradient Context Reuse

**Files:**
- Modify: `R/lav_model_gradient.R`
- Modify: `R/lav_model_estimate.R`
- Test: `../tests/unit/test-lav_model_gradient_context.R`

- [ ] **Step 1: Record the current context-test behavior**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript -e "library(testthat); pkgload::load_all('lavaan', export_all = FALSE); testthat::test_file('tests/unit/test-lav_model_gradient_context.R', reporter = 'summary')"
```

Expected before implementation: the test suite passes with the context test skipped because `lav_model_gradient_context()` is absent.

- [ ] **Step 2: Add group-weight and context helpers**

In `R/lav_model_gradient.R`, after `lav_model_gradient_conditional_x_sample_cache()`, add:

```r
lav_model_gradient_group_weights <- function(lavmodel = NULL,
                                             lavsamplestats = NULL,
                                             group_weight = TRUE) {
  nblocks <- lavmodel@nblocks
  if (!group_weight) {
    return(rep(1.0, nblocks))
  }

  estimator <- lavmodel@estimator
  estimator_args <- lavmodel@estimator.args
  nobs <- unlist(lavsamplestats@nobs, use.names = FALSE)
  ntotal <- lavsamplestats@ntotal

  if (estimator %in% c("ML", "PML", "FML", "MML", "REML", "NTRLS", "catML")) {
    nobs / ntotal
  } else if (estimator == "DLS") {
    if (estimator_args$dls.FtimesNminus1) {
      (nobs - 1) / ntotal
    } else {
      nobs / ntotal
    }
  } else {
    (nobs - 1) / ntotal
  }
}

lav_model_gradient_context <- function(lavmodel = NULL,
                                       lavsamplestats = NULL,
                                       lavdata = NULL,
                                       group_weight = TRUE) {
  nblocks <- lavmodel@nblocks

  list(
    mm_idx = lav_model_get_mm_idx(lavmodel),
    group_w = lav_model_gradient_group_weights(
      lavmodel = lavmodel,
      lavsamplestats = lavsamplestats,
      group_weight = group_weight
    ),
    conditional_x_sample_cache = if (lavmodel@conditional.x) {
      lav_model_gradient_conditional_x_sample_cache(
        lavsamplestats = lavsamplestats,
        nblocks = nblocks
      )
    } else {
      NULL
    }
  )
}
```

- [ ] **Step 3: Wire `optim_context` into `lav_model_gradient()`**

Update the `lav_model_gradient()` signature in `R/lav_model_gradient.R`:

```r
                               ceq_simple = FALSE,
                               implied = NULL,
                               optim_context = NULL) {
```

Replace the current early `mm_idx <- lav_model_get_mm_idx(lavmodel)` assignment and group-weight block with:

```r
  if (is.null(optim_context)) {
    optim_context <- lav_model_gradient_context(
      lavmodel = lavmodel,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata,
      group_weight = group_weight
    )
  }

  mm_idx <- optim_context$mm_idx
  group_w <- optim_context$group_w
```

Keep the existing local reads of estimator, representation, meanstructure, categorical, conditional-x, and estimator args. Remove only the old inline group-weight calculation that this helper replaces.

- [ ] **Step 4: Reuse the conditional-x sample cache**

In the conditional-x ML branch of `R/lav_model_gradient.R`, replace:

```r
    conditional_x_sample_cache <-
      lav_model_gradient_conditional_x_sample_cache(
        lavsamplestats = lavsamplestats,
        nblocks = lavmodel@nblocks
      )
```

with:

```r
    conditional_x_sample_cache <- optim_context$conditional_x_sample_cache
    if (is.null(conditional_x_sample_cache)) {
      conditional_x_sample_cache <-
        lav_model_gradient_conditional_x_sample_cache(
          lavsamplestats = lavsamplestats,
          nblocks = lavmodel@nblocks
        )
    }
```

- [ ] **Step 5: Build and pass one context from the optimizer**

In `R/lav_model_estimate.R`, after the `share_implied` / `use_fast_ml_objective` setup, add:

```r
  optim_context <- lav_model_gradient_context(
    lavmodel = lavmodel,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    group_weight = group_weight
  )
```

Then add the argument to the `lav_model_gradient()` call inside `gradient_function()`:

```r
      implied = if (share_implied) state else NULL,
      optim_context = optim_context
```

- [ ] **Step 6: Run the focused context test**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript -e "library(testthat); pkgload::load_all('lavaan', export_all = FALSE); testthat::test_file('tests/unit/test-lav_model_gradient_context.R', reporter = 'summary')"
```

Expected: the test passes and the trace proves `lav_model_gradient()` did not call `lav_model_gradient_conditional_x_sample_cache()` when a context cache is supplied.

- [ ] **Step 7: Run focused equivalence gates**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript benchmark/profiling/check-gradient-equivalence.R
Rscript benchmark/profiling/check-estimate-state-cache-equivalence.R 25
```

Expected: both scripts pass.

- [ ] **Step 8: Commit the context optimization**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan/lavaan`:

```sh
git add -- R/lav_model_gradient.R R/lav_model_estimate.R
git commit -m "Optimize gradient callback context reuse"
```

Expected: one commit containing only the context helper and optimizer wiring.

## Task 3: Measure Task 2 Before Deeper Derivative Work

**Files:**
- Read: `../benchmark/runs/V6-Baseline/profiling-summary.xlsx`
- Read: candidate workbook generated by this task

- [ ] **Step 1: Run package correctness gates**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
make equivalence
make unit_tests
```

Expected: both pass.

- [ ] **Step 2: Run a directional scenario screen**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript benchmark/profiling/run-scenario-screen.R --run-label gradient-context-screen --scenarios cfa_holzinger_ml,mimic_holzinger_covariates_ml,conditional_x_ml,fiml_holzinger_ml,twolevel_demo_ml --iterations 60 --warmup-iterations 10 --workers 1 --compare-label V6-Baseline
```

Expected: no failed scenarios; context-heavy scenarios should be non-worse. Treat this as directional only.

- [ ] **Step 3: Run the full 25-scenario comparison**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript benchmark/profiling/run-profiling.R --run-label gradient-context-full --mode full --scenarios all --warmup-iterations 10 --blocks 3 --block-iterations 200 --workers 1 --compare-label V6-Baseline --confirm-regressions always --confirm-iterations 100 --confirm-workers 1
```

Expected: workbook at `benchmark/runs/gradient-context-full/profiling-summary.xlsx`.

- [ ] **Step 4: Compute pooled median and p90 from `latency-samples`**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript -e "library(readxl); base <- read_xlsx('benchmark/runs/V6-Baseline/profiling-summary.xlsx', sheet = 'latency-samples'); cand <- read_xlsx('benchmark/runs/gradient-context-full/profiling-summary.xlsx', sheet = 'latency-samples'); q <- function(x, p) as.numeric(quantile(as.numeric(x[['elapsed_sec']]), probs = p, na.rm = TRUE, type = 8)) * 1000; out <- data.frame(metric = c('pooled_median_ms','pooled_p90_ms'), baseline = c(q(base, .5), q(base, .9)), candidate = c(q(cand, .5), q(cand, .9))); out[['ratio']] <- out[['candidate']] / out[['baseline']]; print(out, row.names = FALSE)"
```

Expected: report baseline and candidate pooled values. Keep Task 2 if pooled median is lower, pooled p90 is non-worse, and no confirmed regressions appear. If pooled median is at least 10% lower, the main performance goal is met.

## Task 4: Optimize The LISREL ML Gradient Hot Path If Needed

**Files:**
- Modify: `R/lav_model_gradient.R`
- Modify: `R/lav_representation_lisrel.R`
- Test: `../benchmark/profiling/check-gradient-equivalence.R`

Only run this task if Task 3 does not meet the target and `hotspot-rollup` or `stage-share-summary` still points to `lav_model_gradient()` / `lav_lisrel_df_dmlist()`.

- [ ] **Step 1: Inspect current hotspot evidence**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript -e "library(readxl); wb <- 'benchmark/runs/gradient-context-full/profiling-summary.xlsx'; hs <- read_xlsx(wb, sheet = 'hotspot-rollup'); print(hs[grep('lav_model_gradient|lav_lisrel_df_dmlist|lav_model_omega', paste(hs[['top_self_function']], hs[['top_total_function']])), ], n = 25)"
```

Expected: if no rows mention gradient/LISREL derivative helpers, stop this task and choose the next evidence-backed hotspot instead.

- [ ] **Step 2: Hoist repeated free-index metadata in `lav_model_gradient()`**

In `R/lav_model_gradient.R`, after local `nx_free <- lavmodel@nx.free`, add:

```r
  nblocks <- lavmodel@nblocks
  nblocks_gt1 <- nblocks > 1L
  ceq_simple_only <- lavmodel@ceq.simple.only
  m_free_idx_list <- lavmodel@m.free.idx
  x_free_idx_list <- lavmodel@x.free.idx
  x_unco_idx_list <- lavmodel@x.unco.idx
  glist_names <- names(glist)
```

Then replace repeated occurrences of:

```r
lavmodel@nblocks
lavmodel@ceq.simple.only
lavmodel@m.free.idx[[mm]]
lavmodel@x.free.idx[[mm]]
lavmodel@x.unco.idx[[mm]]
names(glist[mm_in_group])
```

with the corresponding local variables:

```r
nblocks
ceq_simple_only
m_free_idx_list[[mm]]
x_free_idx_list[[mm]]
x_unco_idx_list[[mm]]
glist_names[mm_in_group]
```

Expected: no behavior change; less repeated S4 slot and name work inside hot loops.

- [ ] **Step 3: Add a focused free-gradient fill helper if profiling still shows list extraction overhead**

If the ML direct path still spends measurable time extracting `dx_group` matrices, add this helper in `R/lav_model_gradient.R` near `lav_model_gradient_context()`:

```r
lav_model_gradient_fill_free_group <- function(dx = NULL,
                                               dx_group = NULL,
                                               mm_in_group = NULL,
                                               mm_names = NULL,
                                               m_free_idx_list = NULL,
                                               x_idx_list = NULL,
                                               group_weight = 1.0) {
  for (pos in seq_along(mm_in_group)) {
    mm <- mm_in_group[[pos]]
    dx_mm <- dx_group[[mm_names[[pos]]]]
    if (!identical(group_weight, 1.0)) {
      dx_mm <- group_weight * dx_mm
    }
    dx[x_idx_list[[mm]]] <- dx_mm[m_free_idx_list[[mm]]]
  }
  dx
}
```

Use it only in the ML direct `type == "free"` branch, with `x_idx_list` set to `x_unco_idx_list` when `ceq_simple_only` is true and `x_free_idx_list` otherwise.

- [ ] **Step 4: Run focused gradient equivalence**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript benchmark/profiling/check-gradient-equivalence.R
```

Expected: `Gradient equivalence check passed.`

- [ ] **Step 5: Commit the LISREL/gradient hot-loop cleanup**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan/lavaan`:

```sh
git add -- R/lav_model_gradient.R R/lav_representation_lisrel.R
git commit -m "Tighten gradient hot-loop metadata reuse"
```

Expected: one commit with only the derivative hot-loop cleanup.

## Task 5: Final Gates And Acceptance Benchmark

**Files:**
- Read: `../benchmark/runs/gradient-optimization-final/profiling-summary.xlsx`
- Modify: none unless a gate exposes a necessary source fix.

- [ ] **Step 1: Run full correctness gates**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
make equivalence
make unit_tests
```

Expected: both pass.

- [ ] **Step 2: Run whitespace sanity**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan/lavaan`:

```sh
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run the final full comparison**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript benchmark/profiling/run-profiling.R --run-label gradient-optimization-final --mode full --scenarios all --warmup-iterations 10 --blocks 3 --block-iterations 200 --workers 1 --compare-label V6-Baseline --confirm-regressions always --confirm-iterations 100 --confirm-workers 1
```

Expected: no failed scenarios, no confirmed regressions, workbook at `benchmark/runs/gradient-optimization-final/profiling-summary.xlsx`.

- [ ] **Step 4: Compute final pooled metrics from `latency-samples`**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript -e "library(readxl); base <- read_xlsx('benchmark/runs/V6-Baseline/profiling-summary.xlsx', sheet = 'latency-samples'); cand <- read_xlsx('benchmark/runs/gradient-optimization-final/profiling-summary.xlsx', sheet = 'latency-samples'); q <- function(x, p) as.numeric(quantile(as.numeric(x[['elapsed_sec']]), probs = p, na.rm = TRUE, type = 8)) * 1000; out <- data.frame(metric = c('pooled_median_ms','pooled_p90_ms'), baseline = c(q(base, .5), q(base, .9)), candidate = c(q(cand, .5), q(cand, .9))); out[['delta_pct']] <- 100 * (out[['candidate']] / out[['baseline']] - 1); print(out, row.names = FALSE)"
```

Expected: accept if pooled median is at least 10% lower, or one or more scenario medians are at least 10% lower with pooled median non-worse and p90 non-worse.

- [ ] **Step 5: Inspect regression confirmation**

Run from `C:/Users/miqui/OneDrive/R Projects/FastLavaan`:

```sh
Rscript -e "library(readxl); wb <- 'benchmark/runs/gradient-optimization-final/profiling-summary.xlsx'; sheets <- excel_sheets(wb); if ('stability-summary' %in% sheets) print(read_xlsx(wb, sheet = 'stability-summary'), n = 50); if ('comparison-summary' %in% sheets) print(read_xlsx(wb, sheet = 'comparison-summary'), n = 50)"
```

Expected: no confirmed scenario regressions. If any confirmed regression appears, revert or fix the responsible task before claiming success.
