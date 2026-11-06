# vkd3d's Direct3D 9 cases

`cases.json` holds the Direct3D 9 runs of [vkd3d](https://gitlab.winehq.org/wine/vkd3d)'s HLSL shader tests (`tests/hlsl` at commit `1e329f3`), as `../oracles/vkd3d_cases.py` reads them for vkd3d's Direct3D 9 runner on Windows. Each run lists its constant writes, draws and probes, with the value the test expects there. The file also holds the 1,100 programs Microsoft's `D3DCompile` (d3dcompiler_47) makes of the tests' shaders at models 2.0 and 3.0, and the 61 runs left out, each with the reason.

The probes and the shader sources these cases come from are vkd3d's test files, by the vkd3d authors, under the GNU Lesser General Public License 2.1 or later. `../vkd3d.rs` runs the cases.
