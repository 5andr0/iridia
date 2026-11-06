//! vkd3d's HLSL shader tests (`tests/hlsl/*.shader_test`), read as runs of its Windows runners by
//! `suites/vkd3d/vkd3d_cases.py` into case files ([`cases`]), and run here ([`run`]).
pub mod cases;
pub mod run;
