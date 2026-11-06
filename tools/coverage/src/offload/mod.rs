//! LLVM's offload-test-suite (`test/**/*.test`), read for DirectX with DXC by `suites/offload/offload_cases.py` into a
//! case file ([`cases`]), and run here as lit, the offloader, FileCheck and imgdiff would run it ([`run`]): the
//! pipeline the offloader reads and prints ([`pipeline`], [`yaml`]), its result rules ([`check`]), the program's root
//! signature ([`rts0`]), lit's feature expressions ([`features`]), FileCheck ([`filecheck`]) and imgdiff ([`image`]).
pub mod cases;
pub mod check;
pub mod features;
pub mod filecheck;
pub mod image;
pub mod pipeline;
pub mod rts0;
pub mod run;
pub mod yaml;
