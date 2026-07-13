use lavaan_kernels::{
    helpers::square_matrix_from_column_major,
    delta_a_delta,
    lav_matrix_antidiag_idx,
    lav_matrix_diag_idx,
    lav_matrix_diag_prepost,
    lav_matrix_diagh_idx,
    lav_matrix_lower2full,
    lav_matrix_upper2full,
    lav_matrix_vec,
    lav_matrix_vech,
    lav_matrix_vech_col_idx,
    lav_matrix_vech_idx,
    lav_matrix_vech_reverse,
    lav_matrix_vech_row_idx,
    lav_matrix_vechru,
    lav_matrix_vechru_idx,
    lav_matrix_vechru_reverse,
    lav_matrix_vechr,
    lav_matrix_vechr_idx,
    lav_matrix_vechr_reverse,
    lav_matrix_vechu,
    lav_matrix_vechu_idx,
    lav_matrix_vechu_reverse,
    lav_matrix_vecr,
};
use std::env;
use std::io::{self, Read};

fn parse_next<T: std::str::FromStr>(parts: &mut std::str::SplitWhitespace<'_>, label: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let token = parts
        .next()
        .ok_or_else(|| format!("missing {label}"))?;
    token
        .parse::<T>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn parse_bool_flag(parts: &mut std::str::SplitWhitespace<'_>, label: &str) -> Result<bool, String> {
    Ok(parse_next::<usize>(parts, label)? != 0)
}

fn print_vec_f64(values: Vec<f64>) {
    print!("{}", values.len());
    for value in values {
        print!(" {value:.17}");
    }
    println!();
}

fn print_vec_usize(values: Vec<usize>) {
    print!("{}", values.len());
    for value in values {
        print!(" {value}");
    }
    println!();
}

fn print_mat(mat: &faer::Mat<f64>) {
    print!("{} {}", mat.nrows(), mat.ncols());
    for col in 0..mat.ncols() {
        for row in 0..mat.nrows() {
            let value = mat[(row, col)];
            print!(" {value:.17}");
        }
    }
    println!();
}

fn run_delta_a_delta(input: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let m: usize = parse_next(&mut parts, "row count m")?;
    let n: usize = parse_next(&mut parts, "column count n")?;

    let delta_len = m
        .checked_mul(n)
        .ok_or_else(|| "delta dimensions overflow usize".to_string())?;
    let a1_len = m
        .checked_mul(m)
        .ok_or_else(|| "a1 dimensions overflow usize".to_string())?;

    let mut delta = Vec::with_capacity(delta_len);
    for index in 0..delta_len {
        let value: f64 = parse_next(&mut parts, &format!("delta value {index}"))?;
        delta.push(value);
    }

    let mut a1 = Vec::with_capacity(a1_len);
    for index in 0..a1_len {
        let value: f64 = parse_next(&mut parts, &format!("a1 value {index}"))?;
        a1.push(value);
    }

    if parts.next().is_some() {
        return Err("unexpected trailing input for delta_a_delta".to_string());
    }

    let result = delta_a_delta(&delta, &a1, m, n)?;

    print!("{n} {n}");
    for value in result {
        print!(" {value:.17}");
    }
    println!();
    Ok(())
}

fn run_lav_matrix_diag_prepost(input: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let dim: usize = parse_next(&mut parts, "matrix dimension")?;

    let matrix_len = dim
        .checked_mul(dim)
        .ok_or_else(|| "matrix dimensions overflow usize".to_string())?;
    let mut matrix = Vec::with_capacity(matrix_len);
    for index in 0..matrix_len {
        let value: f64 = parse_next(&mut parts, &format!("matrix value {index}"))?;
        matrix.push(value);
    }

    let mut diag = Vec::with_capacity(dim);
    for index in 0..dim {
        let value: f64 = parse_next(&mut parts, &format!("diag value {index}"))?;
        diag.push(value);
    }

    if parts.next().is_some() {
        return Err("unexpected trailing input for lav_matrix_diag_prepost".to_string());
    }

    let a = square_matrix_from_column_major(&matrix, dim)?;
    let result = lav_matrix_diag_prepost(&a, &diag);

    print_mat(&result);
    Ok(())
}

fn run_vec(input: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let len: usize = parse_next(&mut parts, "vector length")?;
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        let value: f64 = parse_next(&mut parts, &format!("vector value {index}"))?;
        values.push(value);
    }
    if parts.next().is_some() {
        return Err("unexpected trailing input for vec".to_string());
    }
    print_vec_f64(lav_matrix_vec(&values));
    Ok(())
}

fn run_vecr(input: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let nrow: usize = parse_next(&mut parts, "matrix row count")?;
    let ncol: usize = parse_next(&mut parts, "matrix column count")?;
    let matrix = read_matrix_values(&mut parts, nrow, ncol)?;
    if parts.next().is_some() {
        return Err("unexpected trailing input for vecr".to_string());
    }
    let values = lav_matrix_vecr(&matrix, nrow, ncol)?;
    print_vec_f64(values);
    Ok(())
}

fn read_matrix_values(
    parts: &mut std::str::SplitWhitespace<'_>,
    nrow: usize,
    ncol: usize,
) -> Result<Vec<f64>, String> {
    let matrix_len = nrow
        .checked_mul(ncol)
        .ok_or_else(|| "matrix dimensions overflow usize".to_string())?;
    let mut matrix = Vec::with_capacity(matrix_len);
    for index in 0..matrix_len {
        let value: f64 = parse_next(parts, &format!("matrix value {index}"))?;
        matrix.push(value);
    }
    Ok(matrix)
}

fn run_triangular_extract(input: &str, command: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let dim: usize = parse_next(&mut parts, "matrix dimension")?;
    let values = read_matrix_values(&mut parts, dim, dim)?;
    let diagonal = parse_bool_flag(&mut parts, "diagonal flag")?;
    if parts.next().is_some() {
        return Err(format!("unexpected trailing input for {command}"));
    }

    let values = match command {
        "vech" => lav_matrix_vech(&values, dim, diagonal)?,
        "vechr" => lav_matrix_vechr(&values, dim, diagonal)?,
        "vechu" => lav_matrix_vechu(&values, dim, diagonal)?,
        "vechru" => lav_matrix_vechru(&values, dim, diagonal)?,
        _ => return Err(format!("unknown command {command}")),
    };
    print_vec_f64(values);
    Ok(())
}

fn run_index_command(input: &str, command: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let n: usize = parse_next(&mut parts, "matrix dimension")?;
    let diagonal = match command {
        "diag_idx" | "diagh_idx" | "antidiag_idx" => true,
        _ => parse_bool_flag(&mut parts, "diagonal flag")?,
    };
    if parts.next().is_some() {
        return Err(format!("unexpected trailing input for {command}"));
    }

    let values = match command {
        "diag_idx" => lav_matrix_diag_idx(n),
        "diagh_idx" => lav_matrix_diagh_idx(n),
        "antidiag_idx" => lav_matrix_antidiag_idx(n),
        "vech_idx" => lav_matrix_vech_idx(n, diagonal),
        "vech_row_idx" => lav_matrix_vech_row_idx(n, diagonal),
        "vech_col_idx" => lav_matrix_vech_col_idx(n, diagonal),
        "vechr_idx" => lav_matrix_vechr_idx(n, diagonal),
        "vechu_idx" => lav_matrix_vechu_idx(n, diagonal),
        "vechru_idx" => lav_matrix_vechru_idx(n, diagonal),
        _ => return Err(format!("unknown command {command}")),
    };
    print_vec_usize(values);
    Ok(())
}

fn run_triangle_reverse(input: &str, command: &str) -> Result<(), String> {
    let mut parts = input.split_whitespace();
    let diagonal = parse_bool_flag(&mut parts, "diagonal flag")?;
    let mut values = Vec::new();
    for token in parts {
        let value: f64 = token
            .parse()
            .map_err(|error| format!("invalid triangle value: {error}"))?;
        values.push(value);
    }

    let mat = match command {
        "vech_reverse" => lav_matrix_vech_reverse(&values, diagonal)?,
        "vechru_reverse" => lav_matrix_vechru_reverse(&values, diagonal)?,
        "upper2full" => lav_matrix_upper2full(&values, diagonal)?,
        "vechr_reverse" => lav_matrix_vechr_reverse(&values, diagonal)?,
        "vechu_reverse" => lav_matrix_vechu_reverse(&values, diagonal)?,
        "lower2full" => lav_matrix_lower2full(&values, diagonal)?,
        _ => return Err(format!("unknown command {command}")),
    };

    print_mat(&mat);
    Ok(())
}

fn print_usage_and_exit() {
    eprintln!(
        "usage: lav_matrix <delta_a_delta|diag_prepost|diagh_idx|antidiag_idx|vec|vecr|vech|vechr|vechu|vechru|diag_idx|vech_idx|vech_row_idx|vech_col_idx|vechr_idx|vechu_idx|vechru_idx|vech_reverse|vechru_reverse|upper2full|vechr_reverse|vechu_reverse|lower2full> < stdin\n\
Expected input is whitespace-separated numeric values."
    );
    std::process::exit(2);
}

fn main() {
    let mut args = env::args().skip(1);
    let command = args.next();

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .unwrap_or_else(|error| {
            eprintln!("failed to read stdin: {error}");
            std::process::exit(1);
        });

    let result = match command.as_deref() {
        Some("delta_a_delta") | Some("delta") => run_delta_a_delta(&input),
        Some("diag_prepost") | Some("lav_matrix_diag_prepost") => {
            run_lav_matrix_diag_prepost(&input)
        }
        Some("vec") | Some("lav_matrix_vec") => run_vec(&input),
        Some("vecr") | Some("lav_matrix_vecr") => run_vecr(&input),
        Some("vech") | Some("lav_matrix_vech") => run_triangular_extract(&input, "vech"),
        Some("vechr") | Some("lav_matrix_vechr") => run_triangular_extract(&input, "vechr"),
        Some("vechu") | Some("lav_matrix_vechu") => run_triangular_extract(&input, "vechu"),
        Some("vechru") | Some("lav_matrix_vechru") => run_triangular_extract(&input, "vechru"),
        Some("diag_idx") | Some("lav_matrix_diag_idx")
        | Some("diagh_idx") | Some("lav_matrix_diagh_idx")
        | Some("antidiag_idx") | Some("lav_matrix_antidiag_idx")
        | Some("vech_idx") | Some("lav_matrix_vech_idx")
        | Some("vech_row_idx") | Some("lav_matrix_vech_row_idx")
        | Some("vech_col_idx") | Some("lav_matrix_vech_col_idx")
        | Some("vechr_idx") | Some("lav_matrix_vechr_idx")
        | Some("vechu_idx") | Some("lav_matrix_vechu_idx")
        | Some("vechru_idx") | Some("lav_matrix_vechru_idx") => run_index_command(&input, command.as_deref().unwrap()),
        Some("vech_reverse") | Some("lav_matrix_vech_reverse")
        | Some("vechru_reverse") | Some("lav_matrix_vechru_reverse")
        | Some("upper2full") | Some("lav_matrix_upper2full")
        | Some("vechr_reverse") | Some("lav_matrix_vechr_reverse")
        | Some("vechu_reverse") | Some("lav_matrix_vechu_reverse")
        | Some("lower2full") | Some("lav_matrix_lower2full") => {
            run_triangle_reverse(&input, command.as_deref().unwrap())
        }
        Some(_) | None => {
            print_usage_and_exit();
            Ok(())
        }
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
