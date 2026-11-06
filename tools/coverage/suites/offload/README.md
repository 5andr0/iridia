# LLVM's offload-test-suite, as DirectX tests

`offload.json.gz` holds the tests of [llvm/offload-test-suite](https://github.com/llvm/offload-test-suite) (`test/` at commit `3c2fb76`) as `offload_cases.py` reads them for lit's DirectX configuration with DXC on Windows: each test's RUN lines as steps (a compile, an offloader run with its pipeline, an imgdiff, or a line this runner does not model), its `REQUIRES`, `UNSUPPORTED` and `XFAIL` lines, its FileCheck lines, and the programs dxc 1.9.2609 made of its shaders. Every buffer's `Data` is the bytes the offloader's YAML reader makes of it, base64.

The golden images the image tests compare with are not copied: the file names each by its path in [llvm/offload-golden-images](https://github.com/llvm/offload-golden-images) and its SHA-256 at commit `ede8992`, and the runner reads them from a checkout of that repository at that commit (`iridia-coverage offload offload.json.gz <checkout>`).

To make it again:

```
python3 -I offload_cases.py extract <offload-test-suite> <offload-golden-images> <dxc> offload.json.gz <commit> <golden commit>
```

The tests these cases come from are the LLVM project's, under the Apache License 2.0 with LLVM Exceptions (`LICENSE`, the suite's own).
