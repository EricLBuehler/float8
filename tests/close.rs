use float8::f8;

#[test]
fn test_all_close() {
    for n in 0..100 {
        let inp = (n as f64) * 0.1;
        let num = f8::from_f64(inp);
        // println!("out = {}, inp = {}", num.to_f16(), inp);
        println!("diff = {:.3}, inp = {inp}", num.to_f16().to_f64() - inp);
    }
}

#[test]
fn test_cmp() {
    let mut v: Vec<f8> = vec![];
    v.push(f8::ONE);
    v.push(f8::INFINITY);
    v.push(f8::NEG_INFINITY);
    v.push(f8::NAN);
    v.push(f8::MAX_SUBNORMAL);
    v.push(-f8::MAX_SUBNORMAL);
    v.push(f8::ZERO);
    v.push(f8::NEG_ZERO);
    v.push(f8::NEG_ONE);
    v.push(f8::MIN_POSITIVE);

    v.sort_by(|a, b| a.total_cmp(&b));

    assert!(v
        .into_iter()
        .zip(
            [
                f8::NEG_INFINITY,
                f8::NEG_ONE,
                -f8::MAX_SUBNORMAL,
                f8::NEG_ZERO,
                f8::ZERO,
                f8::MAX_SUBNORMAL,
                f8::MIN_POSITIVE,
                f8::ONE,
                f8::INFINITY,
                f8::NAN
            ]
            .iter()
        )
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}
