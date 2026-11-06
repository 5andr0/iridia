"""Independent text oracle: Microsoft D3DDisassemble, with no shader format logic.

    python disassemble.py d3dcompiler_47.dll input.bin output.asm [input.bin output.asm ...]

A Windows Python: on Windows, or under Wine (`winpy`, primase-oracles tools/wine). The DLL is a path to Microsoft's
d3dcompiler_47.dll (the oracle home's sets were made with 10.0.26100.4654, modding-archive/tools/fxc-10.0.26100.4654);
a bare name is refused, since under Wine it loads Wine's own D3DCompiler, whose listing is not Microsoft's.
Inputs must be privately extracted raw shaders; neither they nor game disassembly belong in this repository.
"""
import ctypes
import sys
from pathlib import Path, PureWindowsPath

if len(sys.argv) < 4 or len(sys.argv) % 2:
    sys.exit(__doc__)
dll = sys.argv[1]
if len(PureWindowsPath(dll).parts) < 2:
    sys.exit(f"{dll}: give the path of Microsoft's d3dcompiler_47.dll, not a name the loader resolves")
compiler = ctypes.WinDLL(dll)
compiler.D3DDisassemble.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
]
compiler.D3DDisassemble.restype = ctypes.c_long

for source, listing in zip(sys.argv[2::2], sys.argv[3::2]):
    data = Path(source).read_bytes()
    blob = ctypes.c_void_p()
    buffer = ctypes.create_string_buffer(data)
    result = compiler.D3DDisassemble(buffer, len(data), 0, None, ctypes.byref(blob))
    if result < 0:
        raise RuntimeError(f"{source}: D3DDisassemble failed: {result & 0xffffffff:#x}")
    table = ctypes.cast(blob, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    try:
        pointer = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(table[3])(blob)
        size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(table[4])(blob)
        Path(listing).write_bytes(ctypes.string_at(pointer, size).rstrip(b"\0"))
    finally:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])(blob)
