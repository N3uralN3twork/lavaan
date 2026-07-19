test_that("model estimate helpers match the R fallback expressions", {
  values <- c(3, 5, 7)
  k0 <- c(1, 1, 1)
  K <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  parscale <- c(2, 4, 5)
  dx <- c(2, 4, 8)

  expect_equal(as.numeric((values - k0) %*% K), c(28, 64))
  expect_equal(
    as.numeric(K %*% c(2, 3) + c(1, -1, 0.5)) / parscale,
    c(7.5, 4.5, 4.9)
  )
  expect_equal(as.numeric((dx / c(2, 4, 8)) %*% K) / 10, c(0.6, 1.5))
})

test_that("model estimate Rust wrappers preserve R fallback formulas", {
  skip_if_not(lav_rust_prepare_backend(), "Rust backend is not available")

  old_backend <- getOption("lavaan.backend", default = NULL)
  old_kernels <- getOption("lavaan.rust.kernels", default = NULL)
  on.exit({
    options(lavaan.backend = old_backend)
    options(lavaan.rust.kernels = old_kernels)
  }, add = TRUE)
  options(lavaan.backend = "rust")
  options(lavaan.rust.kernels = "model_estimate")

  values <- c(3, 5, 7)
  k0 <- c(1, 1, 1)
  K <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  parscale <- c(2, 4, 5)
  dx <- c(2, 4, 8)

  expect_equal(
    lav_rust_model_estimate_pack(values, k0, K),
    as.numeric((values - k0) %*% K),
    tolerance = 1e-12
  )
  expect_equal(
    lav_rust_model_estimate_unpack_unscale(c(2, 3), c(1, -1, 0.5), K, parscale),
    as.numeric(K %*% c(2, 3) + c(1, -1, 0.5)) / parscale,
    tolerance = 1e-12
  )
  expect_equal(
    lav_rust_model_estimate_gradient_postprocess(dx, c(2, 4, 8), K, 10, TRUE),
    as.numeric((dx / c(2, 4, 8)) %*% K) / 10,
    tolerance = 1e-12
  )
})
