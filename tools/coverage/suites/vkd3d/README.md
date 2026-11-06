# vkd3d's HLSL shader tests, as Windows runs

The case files here hold the runs of [vkd3d](https://gitlab.winehq.org/wine/vkd3d)'s HLSL shader tests (`tests/hlsl` at commit `1e329f31`) as `vkd3d_cases.py` reads them for vkd3d's Windows runners, one run per test file, runner and shader model, every step in order with the values the test expects its probes to read back:

| File | Runners | Models | Compiler |
|---|---|---|---|
| `d3d9.json.gz` | `d3d9` | 2.0, 3.0 | d3dcompiler_47 (`D3DCompile`, 10.0.26100.4654) |
| `dxbc.json.gz` | `d3d11`, `d3d12` | 4.0 to 5.1 | d3dcompiler_47 |
| `dxil.json.gz` | `d3d12-dxc` | 6.0 to 6.6 | dxc 1.9.2609 |

Each file also holds the programs those compilers made of the tests' shaders (by digest) and the runs left out, each with the reason. The `.listings.json.gz` beside each holds Microsoft's listing of every program (`D3DDisassemble`, `dxc -dumpbin`), which the coverage table compares dx's listings with. The files are gzipped JSON, written reproducibly.

To make them again (the compilers are Microsoft's: fxc under Wine on Linux, see the knowledge base's wine-shader-tools playbook):

```
python3 -I vkd3d_cases.py extract <vkd3d>/tests/hlsl <work> d3d9 d3d11 d3d12 d3d12-dxc
python -I vkd3d_cases.py compile-fxc <d3dcompiler_47.dll> <work>        # Windows Python
python3 -I vkd3d_cases.py compile-dxc <dxc> <work>
python3 -I vkd3d_cases.py merge <work> . <vkd3d commit>
```

`d3d9.json.gz` holds every program and every expectation of `iridia-d3d9-ir/tests/vkd3d/cases.json` (the 1,100 programs, 1,887 probes and 1,869 draws are the same), and more: the runs at the models vkd3d's own runs leave out, and the vertex programs of draws.

The probes and the shader sources these cases come from are vkd3d's test files, by the vkd3d authors, under the GNU Lesser General Public License 2.1 or later.
