//! Compatibility CLI for the existing matrix equivalence harness.
use lavaan_kernels::linalg::{multiply, transpose};
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() != Some("delta_a_delta") {
        return Err("expected delta_a_delta".into());
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut tokens = input.split_whitespace();
    let rows: usize = tokens.next().ok_or("missing row count")?.parse()?;
    let cols: usize = tokens.next().ok_or("missing column count")?.parse()?;
    let delta_len = rows.checked_mul(cols).ok_or("matrix size overflow")?;
    let weight_len = rows.checked_mul(rows).ok_or("matrix size overflow")?;
    let expected = delta_len
        .checked_add(weight_len)
        .ok_or("matrix size overflow")?;
    let values: Vec<f64> = tokens.map(str::parse).collect::<Result<_, _>>()?;
    if values.len() != expected {
        return Err("invalid matrix payload length".into());
    }
    let (delta, weight) = values.split_at(delta_len);
    let out = multiply(
        &transpose(delta, rows, cols),
        &multiply(weight, delta, rows, rows, cols),
        cols,
        rows,
        cols,
    );
    print!("{cols} {cols}");
    for value in out {
        print!(" {value:.17e}");
    }
    println!();
    Ok(())
}
