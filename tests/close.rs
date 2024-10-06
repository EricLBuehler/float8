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
