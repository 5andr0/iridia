mod inputs;
#[test]
fn real_data_inputs_are_declared() {
    primase_testdata::check_declared(&inputs::REGISTRY, env!("CARGO_MANIFEST_DIR"));
}
