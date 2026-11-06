"""Independent token oracle: Microsoft's D3DAssemble and D3DDisassemble, with no shader format logic.

Windows: python assemble.py d3dcompiler_47.dll fixtures-folder
Assembles every <name>.asm in the folder to <name>.bin (the raw token stream) and writes Microsoft's disassembly of
it to <name>.dis; a source the assembler refuses gets <name>.err with its messages instead.
"""
import ctypes
import sys
from pathlib import Path

compiler = ctypes.WinDLL(sys.argv[1])
blob_p = ctypes.POINTER(ctypes.c_void_p)
compiler.D3DAssemble.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
    blob_p, blob_p,
]
compiler.D3DAssemble.restype = ctypes.c_long
compiler.D3DDisassemble.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_char_p, blob_p]
compiler.D3DDisassemble.restype = ctypes.c_long


def take(blob):
    """The bytes of an ID3DBlob, which is then released."""
    if not blob:
        return b""
    table = ctypes.cast(blob, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    try:
        pointer = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(table[3])(blob)
        size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(table[4])(blob)
        return ctypes.string_at(pointer, size)
    finally:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])(blob)


for source in sorted(Path(sys.argv[2]).glob("*.asm")):
    text = source.read_bytes()
    shader, messages = ctypes.c_void_p(), ctypes.c_void_p()
    result = compiler.D3DAssemble(text, len(text), source.name.encode(), None, None, 0,
                                  ctypes.byref(shader), ctypes.byref(messages))
    errors = take(messages)
    if result < 0:
        source.with_suffix(".err").write_bytes(errors or f"{result & 0xffffffff:#x}".encode())
        continue
    tokens = take(shader)
    source.with_suffix(".bin").write_bytes(tokens)
    listing = ctypes.c_void_p()
    if compiler.D3DDisassemble(tokens, len(tokens), 0, None, ctypes.byref(listing)) >= 0:
        source.with_suffix(".dis").write_bytes(take(listing).rstrip(b"\0"))
