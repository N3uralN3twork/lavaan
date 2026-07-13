# lavaan Rust workspace

This directory is the package-local home for Rust kernels that will eventually
back selected `lavaan/R` helpers.

## Layout

- `lavaan-kernels/`: the Rust crate for numeric helpers.
- `lavaan/R/`: keeps the reference R implementations for validation until the
  Rust bridge is turned on.

## Working rule

Keep the exported R function names stable. The migration should happen behind
the existing wrappers so validation can compare the R and Rust paths directly.
