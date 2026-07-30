test_that("LISREL objective-gradient diagnostics preserve the native result", {
  ns <- asNamespace("lavaan")
  prepare_backend <- get("lav_rust_prepare_backend", ns)
  full <- get("lav_model_objective_gradient_lisrel_ml", ns)
  diagnostics <- get("lav_model_objective_gradient_lisrel_ml_diagnostics", ns)
  build_plan <- get("lav_model_objective_gradient_lisrel_plan", ns)
  skip_if_not(prepare_backend(), "Rust backend is not available")

  old_backend <- getOption("lavaan.backend", default = NULL)
  old_kernels <- getOption("lavaan.rust.kernels", default = NULL)
  on.exit({
    options(lavaan.backend = old_backend)
    options(lavaan.rust.kernels = old_kernels)
  }, add = TRUE)
  options(lavaan.backend = "rust")
  options(lavaan.rust.kernels = "model_objective")

  model <- 'visual =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed =~ x7 + x8 + x9'
  fit <- lavaan::cfa(model, data = lavaan::HolzingerSwineford1939, meanstructure = TRUE)
  result <- full(
    fit@Model, fit@Model@GLIST, fit@SampleStats, fit@Data
  )
  diagnostic_result <- diagnostics(
    fit@Model, fit@Model@GLIST, fit@SampleStats, fit@Data
  )
  plan <- build_plan(fit@Model, fit@SampleStats, fit@Data)
  cached_result <- full(
    fit@Model, fit@Model@GLIST, fit@SampleStats, fit@Data, plan = plan
  )

  expect_equal(diagnostic_result$result$objective, result$objective, tolerance = 1e-10)
  expect_equal(diagnostic_result$result$fx.group, result$fx.group, tolerance = 1e-10)
  expect_equal(diagnostic_result$result$gradient, result$gradient, tolerance = 1e-10)
  expect_s3_class(plan, "lavaan_lisrel_ml_plan")
  expect_equal(cached_result$objective, result$objective, tolerance = 1e-10)
  expect_equal(cached_result$fx.group, result$fx.group, tolerance = 1e-10)
  expect_equal(cached_result$gradient, result$gradient, tolerance = 1e-10)
  expect_named(
    diagnostic_result$timings,
    c(
      "r_packing_ms", "r_bridge_call_ms", "bridge_conversion_ms",
      "model_plan_free_map_ms", "implied_moments_ms",
      "factorization_objective_ms", "direct_score_gradient_ms",
      "result_buffer_ms", "r_ffi_and_result_conversion_ms"
    )
  )
  expect_true(all(is.finite(diagnostic_result$timings)))
  expect_true(all(diagnostic_result$timings >= 0))
})

test_that("LISREL Rust bridge reports invalid native contracts", {
  ns <- asNamespace("lavaan")
  prepare_backend <- get("lav_rust_prepare_backend", ns)
  native <- get("lav_rust_model_objective_gradient_lisrel_ml", ns)
  skip_if_not(prepare_backend(), "Rust backend is not available")

  expect_error(
    native(
      numeric(), numeric(), numeric(), numeric(), numeric(), numeric(),
      numeric(), numeric(), 1L, 1L, 0L, 1, 1, integer(), integer(),
      integer(), integer(), 0L, FALSE
    ),
    "nfree must be positive"
  )
})
