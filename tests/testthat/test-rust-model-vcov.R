test_that("model vcov helpers match the R fallback expressions", {
  old_backend <- getOption("lavaan.backend", default = NULL)
  old_kernels <- getOption("lavaan.rust.kernels", default = NULL)
  on.exit({
    options(lavaan.backend = old_backend)
    options(lavaan.rust.kernels = old_kernels)
  }, add = TRUE)
  options(lavaan.backend = "r")
  options(lavaan.rust.kernels = "none")

  delta <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  a1 <- matrix(c(2, 0.5, 1, 0.5, 3, -1, 1, -1, 4), nrow = 3)
  left <- matrix(c(1, 3, 2, 4), nrow = 2)
  middle <- matrix(c(2, 0.5, 0.5, 3), nrow = 2)
  jac <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  vcov <- diag(c(2, 3, 4))

  expect_equal(lav_model_vcov_delta_A_delta(delta, a1), crossprod(delta, a1) %*% delta)
  expect_equal(lav_model_vcov_sandwich(left, middle), left %*% middle %*% left)
  expect_equal(
    lav_model_vcov_jacobian_vcov_jacobian_t(jac, vcov),
    jac %*% vcov %*% t(jac)
  )
})

test_that("model vcov Rust bridge matches the R fallback expressions", {
  skip_if_not(lav_rust_prepare_backend(), "Rust backend is not available")

  old_backend <- getOption("lavaan.backend", default = NULL)
  old_kernels <- getOption("lavaan.rust.kernels", default = NULL)
  on.exit({
    options(lavaan.backend = old_backend)
    options(lavaan.rust.kernels = old_kernels)
  }, add = TRUE)
  options(lavaan.backend = "rust")
  options(lavaan.rust.kernels = "model_vcov")

  delta <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  a1 <- matrix(c(2, 0.5, 1, 0.5, 3, -1, 1, -1, 4), nrow = 3)
  left <- matrix(c(1, 3, 2, 4), nrow = 2)
  middle <- matrix(c(2, 0.5, 0.5, 3), nrow = 2)
  jac <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  vcov <- diag(c(2, 3, 4))

  expect_equal(lav_model_vcov_delta_A_delta(delta, a1), crossprod(delta, a1) %*% delta)
  expect_equal(lav_model_vcov_sandwich(left, middle), left %*% middle %*% left)
  expect_equal(
    lav_model_vcov_jacobian_vcov_jacobian_t(jac, vcov),
    jac %*% vcov %*% t(jac)
  )
})
