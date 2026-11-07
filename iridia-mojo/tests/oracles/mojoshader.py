"""Make MonoGame's GLSL of Direct3D 9 programs: the oracle pairs iridia-mojo's real-data test reads (tests/oracle.rs).

MonoGame's effect compiler (mgfxc 3.8.1, `ShaderData.CreateGLSL` in Tools/MonoGame.Effect.Compiler/Effect/
ShaderData.mojo.cs) writes an OpenGL effect's shaders by handing each Direct3D 9 program to its MojoShader build
(`MOJOSHADER_parse("glsl", ...)` of `libmojoshader_64.dll`) and editing the text three ways: `#version 110` removed, the
GL_ES precision block put first, and `#extension GL_OES_standard_derivatives : enable` before that where the text
calls `dFdx` or `dFdy`. This does the same with the same DLL, so each program's GLSL is what MonoGame ships for it, and
writes beside it what the effect's tables would say (iridia-mojo's `Hints`: the constants as CreateGLSL re-indexes
MojoShader's symbols, each sampler's slot and name, each attribute's semantic) and what MojoShader states of the
uniform array (its uniforms, in row order).

    python -I mojoshader.py <libmojoshader_64.dll> <input .zip or folder> <pattern> <output folder>

`pattern` picks the programs among the input's files (fnmatch: `*.[pv]s`, `programs/d3d9/*.d3d9`). The output folder
gets `programs.jsonl`, one line per program in name order: its `name` (its path in the input) and `sha256`, then either
its `glsl`, `stage`, `target`, `hints` and `uniforms`, or `refused` (MojoShader's error); and `MANIFEST.toml`, which
records the DLL's SHA-256, the input and the counts. The programs' bytes are the input's: a consumer reads them from
the input set by name, held to the hash. The output folder must not exist: an oracle entry is written once.
"""
import ctypes as C
import fnmatch
import hashlib
import json
import os
import sys
import zipfile


class Error(C.Structure):
    _fields_ = [("error", C.c_char_p), ("filename", C.c_char_p), ("error_position", C.c_int)]


class Uniform(C.Structure):
    _fields_ = [("type", C.c_int), ("index", C.c_int), ("array_count", C.c_int), ("constant", C.c_int),
                ("name", C.c_char_p)]


class Sampler(C.Structure):
    _fields_ = [("type", C.c_int), ("index", C.c_int), ("name", C.c_char_p), ("texbem", C.c_int)]


class Attribute(C.Structure):
    _fields_ = [("usage", C.c_int), ("index", C.c_int), ("name", C.c_char_p)]


class TypeInfo(C.Structure):
    _fields_ = [("parameter_class", C.c_int), ("parameter_type", C.c_int), ("rows", C.c_uint), ("columns", C.c_uint),
                ("elements", C.c_uint), ("member_count", C.c_uint), ("members", C.c_void_p)]


class Symbol(C.Structure):
    _fields_ = [("name", C.c_char_p), ("register_set", C.c_int), ("register_index", C.c_uint),
                ("register_count", C.c_uint), ("info", TypeInfo)]


class ParseData(C.Structure):
    _fields_ = [
        ("error_count", C.c_int), ("errors", C.POINTER(Error)), ("profile", C.c_char_p), ("output", C.c_char_p),
        ("output_len", C.c_int), ("instruction_count", C.c_int), ("shader_type", C.c_int), ("major_ver", C.c_int),
        ("minor_ver", C.c_int), ("uniform_count", C.c_int), ("uniforms", C.POINTER(Uniform)),
        ("constant_count", C.c_int), ("constants", C.c_void_p), ("sampler_count", C.c_int),
        ("samplers", C.POINTER(Sampler)), ("attribute_count", C.c_int), ("attributes", C.POINTER(Attribute)),
        ("output_count", C.c_int), ("outputs", C.POINTER(Attribute)), ("swizzle_count", C.c_int),
        ("swizzles", C.c_void_p), ("symbol_count", C.c_int), ("symbols", C.POINTER(Symbol)),
        ("preshader", C.c_void_p), ("malloc", C.c_void_p), ("free", C.c_void_p), ("malloc_data", C.c_void_p),
    ]


# MOJOSHADER_usage, which is D3DDECLUSAGE's order, by the names a Direct3D 9 program's semantics have
SEMANTICS = ["POSITION", "BLENDWEIGHT", "BLENDINDICES", "NORMAL", "PSIZE", "TEXCOORD", "TANGENT", "BINORMAL",
             "TESSFACTOR", "POSITIONT", "COLOR", "FOG", "DEPTH", "SAMPLE"]
SYMREGSET_BOOL, SYMREGSET_INT4, SYMREGSET_FLOAT4, SYMREGSET_SAMPLER = 0, 1, 2, 3
UNIFORM_TYPES = {0: "float", 1: "int", 2: "bool"}
SAMPLER_TYPES = {0: "2d", 1: "cube", 2: "volume", 3: "1d"}


def text(p):
    return p.decode("ascii") if p is not None else None


def monogame_glsl(output, vertex):
    """CreateGLSL's three edits of MojoShader's text."""
    code = output.replace("#version 110", "")
    precision = "precision highp float;\r\n" if vertex else "precision mediump float;\r\n"
    code = "#ifdef GL_ES\r\n" + precision + "precision mediump int;\r\n#endif\r\n" + code
    if "dFdx" in code or "dFdy" in code:
        code = "#extension GL_OES_standard_derivatives : enable\r\n" + code
    return code


def hints(p, vertex):
    """What the effect's tables say of the program, as CreateGLSL makes them from MojoShader's lists."""
    symbols = [p.symbols[i] for i in range(p.symbol_count)]
    symbols = [dict(name=text(s.name), set=s.register_set, index=s.register_index, count=s.register_count,
                    elements=s.info.elements, columns=s.info.columns, rows=s.info.rows) for s in symbols]
    # CreateGLSL's order: by register, the arrays first ("mojoshader puts array objects first"), then each register
    # set numbered from zero in that order
    symbols.sort(key=lambda s: s["index"] + (1024 if s["elements"] == 1 else 0))
    sampler_symbols = {s["index"]: s["name"] for s in symbols if s["set"] == SYMREGSET_SAMPLER}
    next_index = {SYMREGSET_BOOL: 0, SYMREGSET_INT4: 0, SYMREGSET_FLOAT4: 0}
    constants = []
    for s in symbols:
        if s["set"] not in next_index:
            continue
        index = next_index[s["set"]]
        next_index[s["set"]] += s["count"]
        if s["set"] == SYMREGSET_FLOAT4:
            constants.append(dict(name=s["name"], offset=index * 16, bytes=s["count"] * 16, width=s["columns"]))
    samplers = []
    for i in range(p.sampler_count):
        s = p.samplers[i]
        samplers.append(dict(glsl_name=text(s.name), slot=s.index, name=sampler_symbols.get(s.index, text(s.name)),
                             type=SAMPLER_TYPES.get(s.type, str(s.type))))
    attributes = []
    for i in range(p.attribute_count):
        a = p.attributes[i]
        semantic = SEMANTICS[a.usage] if 0 <= a.usage < len(SEMANTICS) else str(a.usage)
        attributes.append(dict(glsl_name=text(a.name), semantic=semantic, index=a.index))
    uniforms = []
    for i in range(p.uniform_count):
        u = p.uniforms[i]
        uniforms.append(dict(type=UNIFORM_TYPES.get(u.type, str(u.type)), register=u.index,
                             array_count=u.array_count, constant=bool(u.constant), name=text(u.name)))
    stage = "vertex" if vertex else "pixel"
    return dict(stage=stage, target=f"{'vs' if vertex else 'ps'}_{p.major_ver}_{p.minor_ver}",
                hints=dict(constants=constants, samplers=samplers, attributes=attributes), uniforms=uniforms)


def programs(source, pattern):
    """The (name, bytes) of the input's files that match `pattern`, by name."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            names = sorted(n for n in z.namelist() if fnmatch.fnmatchcase(n, pattern))
            return [(n, z.read(n)) for n in names]
    out = []
    for root, _, files in os.walk(source):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), source).replace(os.sep, "/")
            if fnmatch.fnmatchcase(rel, pattern):
                out.append(rel)
    return [(n, open(os.path.join(source, n), "rb").read()) for n in sorted(out)]


def main():
    dll_path, source, pattern, out = sys.argv[1:5]
    if os.path.exists(out):
        sys.exit(f"{out} exists: an entry is written once")
    dll = C.CDLL(dll_path)
    dll.MOJOSHADER_parse.restype = C.POINTER(ParseData)
    dll.MOJOSHADER_parse.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_int,
                                     C.c_void_p, C.c_void_p, C.c_void_p]
    dll_sha = hashlib.sha256(open(dll_path, "rb").read()).hexdigest()
    os.makedirs(out)
    counts = {"programs": 0, "glsl": 0, "refused": 0}
    names = set()
    lines = []
    for name, data in programs(source, pattern):
        if name in names:
            sys.exit(f"two programs named {name}")
        names.add(name)
        counts["programs"] += 1
        vertex = len(data) >= 4 and int.from_bytes(data[2:4], "little") == 0xFFFE
        line = dict(name=name, sha256=hashlib.sha256(data).hexdigest())
        # MonoGame's build does not export MOJOSHADER_freeParseData, so the parse data is not freed (mgfxc does not free it)
        p = dll.MOJOSHADER_parse(b"glsl", data, len(data), None, 0, None, 0, None, None, None).contents
        if p.error_count > 0:
            e = p.errors[0]
            line["refused"] = f"{text(e.error)} (at {e.error_position})"
            counts["refused"] += 1
        else:
            line["glsl"] = monogame_glsl(text(p.output), vertex)
            line.update(hints(p, vertex))
            counts["glsl"] += 1
        lines.append(json.dumps(line, sort_keys=True))
    with open(os.path.join(out, "programs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    with open(os.path.join(out, "MANIFEST.toml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# MonoGame's GLSL of Direct3D 9 programs (iridia-mojo/tests/oracles/mojoshader.py)\n")
        f.write('tool = "libmojoshader_64.dll of dotnet-mgfxc 3.8.1.303 (MonoGame\'s effect compiler, NuGet)"\n')
        f.write(f'tool_sha256 = "{dll_sha}"\n')
        f.write(f'input = "{os.path.basename(source)}"\n')
        f.write(f'pattern = "{pattern}"\n')
        for k, v in counts.items():
            f.write(f"{k} = {v}\n")
    print(counts)


if __name__ == "__main__":
    main()
