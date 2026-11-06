"""vkd3d's HLSL shader tests as Direct3D 9 cases: the shaders Microsoft's compiler makes of them, and what vkd3d's tests
expect a Direct3D 9 device on Windows to draw.

    python3 vkd3d_cases.py extract <vkd3d>/tests/hlsl <work>     (Linux) the runs and the shaders to compile
    python vkd3d_cases.py compile <d3dcompiler_47.dll> <work>     (Windows) D3DCompile and D3DDisassemble each shader
    python3 vkd3d_cases.py merge <work> <cases.json> <vkd3d commit>  (Linux) the committed cases

The test files are read as vkd3d's shader runner reads them for its Direct3D 9 runner on Windows (tests/shader_runner.c
and tests/shader_runner_d3d9.c): models 2.0 and 3.0, the tags `windows` and `d3d9`, the capabilities `clip-planes`,
`fog` and `point-size`, and `todo` and `bug` ignored, since on Windows every check must hold. A line qualified by a tag
of the hardware (`nvidia`, `amd`, `intel`, `warp`) or by a version of one is left out, and so is everything after it
in its run when it changes state, since what it does depends on the machine. This script has no Direct3D 9 format
logic: the shaders are Microsoft's compiler's, and what a case expects is the test file's.

A run is one test file at one model. It is `selected` where vkd3d's runner would run its tests at that model (the
models of the file's header, and the first model of shader model 1 to 3 at which the shader compiles) and `extended`
where the runner would not but the model is in the range the file requires and the shader compiles: the same
expectations, at a model vkd3d's own runs leave out.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

MODELS = ["2.0", "3.0", "4.0", "4.1", "5.0", "5.1", "6.0", "6.1", "6.2", "6.5", "6.6"]
D3D9_MODELS = (0, 1)
TAGS = ["amd", "d3d11", "d3d12", "d3d9", "glsl", "intel", "llvm", "llvmpipe", "mesa", "msl", "mvk", "nvidia",
        "opengl", "radeonsi", "radv", "spirv", "vulkan", "warp", "windows"]
HARDWARE = {"amd", "intel", "nvidia", "warp"}
PRESENT = {"d3d9", "windows"}
CAPS = ["barycentrics", "clip-planes", "cull-distance", "depth-bounds", "descriptor-indexing", "descriptors",
        "filter-minmax", "float64", "fog", "geometry-shader", "int64", "mesh-shader", "min-lod-clamp",
        "native-16-bit", "no-refactoring", "point-size", "rov", "rt-vp-array-index", "stencil-export",
        "tessellation-shader", "texture-swizzle", "wave-ops"]
D3D9_CAPS = {"clip-planes", "fog", "point-size"}
OPTIONS = [(0, "none"), (1 << 3, "row-major"), (1 << 4, "column-major"), (1 << 12, "backcompat"),
           (1 << 13, "ieee-strictness"), (1 << 20, "unbounded-descriptor-arrays")]
SHADER_SECTIONS = {"[compute-shader": "cs", "[pixel-shader": "ps", "[pixel shader": "ps", "[vertex-shader": "vs",
                   "[vertex shader": "vs", "[effect": "fx", "[hull-shader": "hs", "[domain-shader": "ds",
                   "[geometry-shader": "gs", "[amplification-shader": "as", "[mesh-shader": "ms"}
UNKNOWN = "unknown"


class Unknown(Exception):
    """A line whose meaning depends on the machine the test runs on."""


class Malformed(Exception):
    pass


def f32_bits(text):
    """The bits of the float strtof makes of `text`: the nearest float, ties to even, by exact arithmetic."""
    t = text.strip().lower()
    sign = 0
    if t.startswith(("-", "+")):
        sign = 0x8000_0000 if t[0] == "-" else 0
        t = t[1:]
    if t in ("inf", "infinity"):
        return sign | 0x7f80_0000
    if t.startswith("nan"):
        return sign | 0x7fc0_0000
    value = Fraction(float.fromhex(t)) if t.startswith("0x") else Fraction(t)
    if value == 0:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    exponent = max(exponent, -126)
    if exponent > 127:
        return sign | 0x7f80_0000
    scaled = value / Fraction(2) ** (exponent - 23)
    whole, rest = divmod(scaled.numerator, scaled.denominator)
    if 2 * rest > scaled.denominator or (2 * rest == scaled.denominator and whole & 1):
        whole += 1
    if whole >= 1 << 24:
        whole >>= 1
        exponent += 1
        if exponent > 127:
            return sign | 0x7f80_0000
    if whole < 1 << 23:
        return sign | whole
    return sign | ((exponent + 127) << 23) | (whole - (1 << 23))


FLOAT = re.compile(r"\s*([+-]?(?:0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*(?:[pP][+-]?\d+)?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
                   r"|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN]))")
INTEGER = re.compile(r"\s*([+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*))")


def read_float(line):
    m = FLOAT.match(line)
    if not m:
        raise Malformed(f"float at '{line}'")
    return f32_bits(m.group(1)), line[m.end():]


def read_integer(line, signed):
    m = INTEGER.match(line)
    if not m:
        raise Malformed(f"integer at '{line}'")
    text = m.group(1)
    negative = text.startswith("-")
    digits = text.lstrip("+-")
    value = int(digits, 16) if digits[:2].lower() == "0x" else int(digits, 8) if digits.startswith("0") else int(digits)
    value = -value if negative else value
    return value & 0xffff_ffff, line[m.end():]


def match(line, token):
    """vkd3d's match_string: `token` after leading space, followed by space or a parenthesis; the rest or None."""
    s = line.lstrip()
    if not s.startswith(token):
        return None
    after = s[len(token):]
    if after and not (after[0].isspace() or after[0] in "()]"):
        return None
    if not after:
        return None
    return after.lstrip()


def match_rest(line, token):
    rest = match(line, token)
    return None if rest is None else rest.lstrip()


def model_string(s):
    for n, name in enumerate(MODELS):
        if s.startswith(name):
            return n, s[len(name):]
        if s[:1] == name[0] and name[1:] == ".0" and s[1:2] != "." and not s[1:2].isdigit():
            return n, s[1:]
    raise Malformed(f"shader model '{s}'")


def tag(s):
    """A tag at the start of `s`, with any version comparison: (holds or UNKNOWN, rest) or None."""
    negate = s.startswith("!")
    if negate:
        s = s[1:]
    for name in reversed(TAGS):
        if not s.startswith(name):
            continue
        s = s[len(name):]
        compared = re.match(r"(<=|>=|!=|<|>|=)(\d+(?:\.\d+){0,2})", s)
        if compared:
            s = s[compared.end():]
        if name in HARDWARE or (compared and name in PRESENT):
            return UNKNOWN, s
        holds = name in PRESENT
        return (not holds) if negate else holds, s
    return None


def conj(a, b):
    if a is False or b is False:
        return False
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    return True


def disj(a, b):
    if a is True or b is True:
        return True
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    return False


def qualifier(s, model):
    """check_qualifier_args at `s`: (holds, model mask, rest)."""
    if not s.startswith("("):
        return True, ~0, s
    s = s[1:]
    holds, mask, first = False, 0, True
    while not s.startswith(")"):
        if not first and s.startswith("|"):
            s = s[1:]
        first = False
        part, sub = True, ~0
        while not s.startswith(")") and not s.startswith("|"):
            s = s.lstrip()
            matched = True
            if s.startswith("sm>="):
                m, s = model_string(s[4:])
                sub &= ~((1 << m) - 1)
                part = conj(part, model >= m)
            elif s.startswith("sm<"):
                m, s = model_string(s[3:])
                sub &= (1 << m) - 1
                part = conj(part, model < m)
            elif s.startswith("sm="):
                m, s = model_string(s[3:])
                sub &= 1 << m
                part = conj(part, model == m)
            else:
                t = tag(s)
                if t is None:
                    matched = False
                else:
                    part = conj(part, t[0])
                    s = t[1]
            s = s.lstrip()
            if matched and s.startswith("&"):
                s = s[1:]
            elif not (s.startswith(")") or s.startswith("|")):
                raise Malformed(f"qualifier '{s}'")
        holds = disj(holds, part)
        mask |= sub
    return holds, mask, s[1:]


def match_with_args(line, token, model):
    """match_string_with_args: (holds, rest) or None."""
    s = line.lstrip()
    if not s.startswith(token):
        return None
    after = s[len(token):]
    if not after or not (after[0].isspace() or after[0] in "()]"):
        return None
    holds, _, rest = qualifier(after, model)
    return holds, rest.lstrip()


def literal(line):
    """read_literal: (kind, [u32 bits], rest)."""
    for kind, signed in (("f32", None), ("rgba", None), ("u32", False), ("i32", True)):
        rest = match(line, kind)
        if rest is None or not rest.startswith("("):
            continue
        rest = rest[1:]
        values = []
        while True:
            if kind in ("f32", "rgba"):
                v, rest = read_float(rest)
            else:
                v, rest = read_integer(rest, signed)
            values.append(v)
            rest = rest.lstrip()
            if rest.startswith(","):
                rest = rest[1:]
                continue
            break
        rest = rest.lstrip()
        if not rest.startswith(")"):
            raise Malformed(f"literal close at '{rest}'")
        return ("f32" if kind == "rgba" else kind), values, rest[1:]
    return None


class Run:
    """One test file at one model, read as shader_runner.c's run_shader_tests_for_model reads it."""

    def __init__(self, name, model, header_mask):
        self.name, self.model, self.header_mask = name, model, header_mask
        self.minimum, self.maximum = 0, len(MODELS) - 1
        self.options, self.skip, self.missing_caps, self.uncertain = 0, False, False, False
        self.hrs = [True] * len(MODELS)
        self.shaders = {}
        self.steps = []
        self.stopped = None
        self.target = False
        self.jobs = None

    def job(self, ps):
        """The compile job of a draw with `ps`: compile_hlsl compiles at the draw, with the options then in force."""
        if ps["form"] == "d3dbc-hex":
            key = ("hex", ps["source"])
            job = {"hex": ps["source"], "from": f"{self.name}:{ps['start']}"}
        elif ps["form"] == "hlsl":
            profile = f"ps_{MODELS[self.model].replace('.', '_')}"
            key = (ps["source"], profile, self.options)
            job = {"source": ps["source"], "profile": profile, "options": self.options,
                   "from": f"{self.name}:{ps['start']}"}
        else:
            key = ("form", ps["form"])
            job = {"form": ps["form"], "from": f"{self.name}:{ps['start']}"}
        index = self.jobs["index"].setdefault(key, len(self.jobs["list"]))
        if index == len(self.jobs["list"]):
            self.jobs["list"].append(job)
        return index

    def stop(self, line_number, reason):
        if self.stopped is None:
            self.stopped = {"line": line_number, "reason": reason}

    def test_mask(self):
        mask = self.header_mask
        if not mask & 0b11:
            for j in range(self.minimum, min(self.maximum, 1) + 1):
                if self.hrs[j]:
                    mask |= 1 << j
                    break
        return mask

    def runs_tests(self):
        """should_skip_execution for a run, and whether vkd3d's own runner would select this model."""
        if self.skip or self.uncertain:
            return None
        if not self.minimum <= self.model <= self.maximum:
            return None
        if not self.hrs[self.model]:
            return None
        return "selected" if self.test_mask() >> self.model & 1 else "extended"

    def require(self, line, number):
        for op, kind in (("shader model >=", ">="), ("shader model =", "="), ("shader model <", "<")):
            rest = match(line, op)
            if rest is None:
                continue
            m, _ = model_string(rest.lstrip())
            if kind == "=":
                self.maximum, self.minimum = min(self.maximum, m), max(self.minimum, m)
            elif kind == "<":
                self.maximum = min(self.maximum, m - 1)
            else:
                self.minimum = max(self.minimum, m)
            return
        rest = match(line, "options:")
        if rest is not None:
            self.options = 0
            for value, name in OPTIONS:
                r = match(rest, name)
                if r is not None:
                    self.options |= value
                    rest = r
            return
        if match(line, "format") is not None:
            self.missing_caps = True
            return
        for cap in CAPS:
            if match(line, cap) is not None:
                if cap not in D3D9_CAPS:
                    self.missing_caps = True
                return
        s, holds_any = line.strip(), False
        while s:
            holds = True
            while s and not s.startswith("|"):
                s = s.lstrip()
                t = tag(s)
                if t is None:
                    raise Malformed(f"require '{line.strip()}'")
                holds, s = conj(holds, t[0]), t[1].lstrip()
                if s.startswith("&"):
                    s = s[1:]
            if s.startswith("|"):
                s = s[1:]
            holds_any = disj(holds_any, holds)
        if holds_any is UNKNOWN:
            self.uncertain = True
        elif not holds_any:
            self.skip = True

    def shader_header(self, rest, kind):
        self.hrs = [True] * len(MODELS)
        form = "hlsl"
        s = rest
        while s and not s.startswith("]"):
            s = s.lstrip()
            for word in ("todo", "fail", "notimpl"):
                if s.startswith(word):
                    _, mask, s = qualifier(s[len(word):], self.model)
                    if word != "todo":
                        for n in range(len(MODELS)):
                            if mask >> n & 1:
                                self.hrs[n] = False
                    break
            else:
                for word in ("d3dbc-hex", "dxbc-tpf-hex", "dxbc-dxil-hex"):
                    if s.startswith(word):
                        form, s = word, s[len(word):]
                        break
                else:
                    raise Malformed(f"shader header '{rest}'")
        return {"kind": kind, "form": form}

    def test_line(self, line, number):
        """parse_test_directive: one step, or a stop where the line is not one this reading models."""
        skip, more = False, True
        while more:
            more = False
            for word in ("todo", "bug"):
                m = match_with_args(line, word, self.model)
                if m is not None:
                    line, more = m[1], True
            m = match_with_args(line, "winbug", self.model)
            if m is not None:
                line, more, skip = m[1], True, True
            m = match_with_args(line, "if", self.model)
            if m is not None:
                if m[0] is UNKNOWN:
                    raise Unknown(line)
                if not m[0]:
                    skip = True
                line, more = m[1], True
        if skip:
            return
        rest = match_rest(line, "clear")
        if rest is not None:
            r = match_rest(rest, "rtv-0")
            lit = literal(r) if r is not None else None
            if lit is None or lit[0] != "f32" or len(lit[1]) != 4:
                return self.stop(number, f"clear of another resource: {line.strip()}")
            self.target = True
            return self.steps.append({"clear": lit[1], "line": number})
        rest = match_rest(line, "draw quad")
        if rest is not None:
            vs = self.shaders.get("vs")
            ps = self.shaders.get("ps")
            if vs is not None:
                return self.stop(number, "a draw with a vertex shader of the test's")
            if ps is None:
                return self.stop(number, "a draw with no pixel shader")
            self.target = True
            return self.steps.append({"draw": self.job(ps), "line": number})
        rest = match_rest(line, "uniform")
        if rest is not None:
            m = re.match(r"\s*(\d+)\s+float4\s+", rest)
            if not m:
                return self.stop(number, f"uniform of a type this reading does not model: {line.strip()}")
            rest = rest[m.end():]
            values = []
            for _ in range(4):
                v, rest = read_float(rest)
                values.append(v)
            return self.steps.append({"write": "float", "at": int(m.group(1)), "bits": values, "line": number})
        rest = match_rest(line, "write")
        if rest is not None:
            bank = None
            for name, b in (("cbv-f32", "float"), ("cbv-i32", "int"), ("cbv-bool", "bool"), ("cbv-0", "float"),
                            ("cbv-1", "int"), ("cbv-2", "bool")):
                r = match_rest(rest, name)
                if r is not None:
                    bank, rest = b, r
                    break
            m = re.match(r"\s*\(\s*(\d+)\s*\)", rest or "")
            lit = literal(rest[m.end():]) if bank and m else None
            if lit is None:
                return self.stop(number, f"write this reading does not model: {line.strip()}")
            return self.steps.append({"write": bank, "at": int(m.group(1)), "bits": lit[1], "line": number})
        rest = match_rest(line, "probe")
        if rest is not None:
            for name in ("rtv-0",):
                r = match_rest(rest, name)
                if r is not None:
                    rest = r
            if re.match(r"\s*(rtv|dsv|srv|uav|vb|cbv)", rest):
                return self.stop(number, f"probe of another resource: {line.strip()}")
            m = (re.match(r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", rest)
                 or re.match(r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", rest))
            if not m:
                return self.stop(number, f"probe this reading does not model: {line.strip()}")
            g = [int(x) for x in m.groups()]
            rect = g if len(g) == 4 else [g[0], g[1], g[0] + 1, g[1] + 1]
            lit = literal(rest[m.end():])
            if lit is None or lit[0] != "f32":
                return self.stop(number, f"probe of a type this reading does not model: {line.strip()}")
            u = re.match(r"\s*(\d+)", lit[2])
            if not self.target:
                return self.stop(number, "a probe before any draw or clear")
            return self.steps.append({"probe": rect, "expect": lit[1], "ulps": int(u.group(1)) if u else 0,
                                      "line": number})
        return self.stop(number, f"state this reading does not model: {line.split()[0] if line.split() else ''}")


def header_mask(lines):
    mask = 0
    in_header = False
    for line in lines:
        if line == "[header]\n":
            in_header = True
        elif line.startswith("["):
            break
        elif line[:1] not in ("%", "\n") and in_header:
            rest = match_rest(line, "test-shader-models")
            if rest is None or not rest.startswith("("):
                raise Malformed(line)
            for word in rest[1:].split(")")[0].split():
                mask |= 1 << MODELS.index(word)
    return mask


def read_file(path, model, jobs):
    """One run of `path` at `model`; shaders to compile are added to `jobs`."""
    lines = path.read_text().splitlines(keepends=True)
    run = Run(path.name, model, header_mask(lines))
    run.jobs = jobs
    state, source, start, current, section_tests = None, [], 0, None, None
    number = 0

    def close():
        nonlocal source
        if state == "require" and run.maximum < run.minimum:
            run.skip = True
        if state == "require" and run.missing_caps:
            run.skip = True
        if state == "shader":
            current["source"] = "".join(source)
            current["start"] = start
            run.shaders[current["kind"]] = current
        source = []

    for raw in lines + [None]:
        number += 1
        if raw is None or raw.startswith("["):
            close()
            if raw is None:
                break
            start, line = number, raw
            kind = next((k for p, k in SHADER_SECTIONS.items() if match(line, p) is not None), None)
            if kind is not None:
                state = "shader"
                prefix = next(p for p, k in SHADER_SECTIONS.items() if match(line, p) is not None)
                current = run.shader_header(line.lstrip()[len(prefix):], kind)
            elif line == "[require]\n":
                state = "require"
                run.minimum, run.maximum = 0, len(MODELS) - 1
                run.options, run.skip, run.missing_caps, run.uncertain = 0, False, False, False
            elif line == "[test]\n":
                state = "test"
                section_tests = run.runs_tests()
                if section_tests is not None:
                    run.steps.append({"section": section_tests, "line": number})
            elif re.match(r"\[(rtv|dsv)", line):
                state = "resource"
                if run.runs_tests() is not None:
                    run.stop(number, "a render target or depth buffer of the test's")
            elif re.match(r"\[(srv|uav|vb|sampler)[- ]?\d*\]", line) or line in (
                    "[input-layout]\n", "[descriptors]\n", "[header]\n", "[preproc]\n", "[preproc fail]\n"):
                state = "other"
            else:
                raise Malformed(f"{path.name}:{number}: section {line.strip()}")
            continue
        if raw[:1] in ("%", "\n"):
            continue
        if state == "shader":
            source.append(raw)
        elif state == "require":
            run.require(raw, number)
        elif state == "test" and section_tests is not None and run.stopped is None:
            try:
                run.test_line(raw, number)
            except Unknown:
                word = raw.split()
                touches_state = not any(w.startswith("probe") for w in word)
                if touches_state:
                    run.stop(number, "a state change that depends on the machine")
    return run


def extract(tests, work):
    jobs, runs = {"index": {}, "list": []}, []
    for path in sorted(Path(tests).glob("*.shader_test")):
        for model in D3D9_MODELS:
            try:
                run = read_file(path, model, jobs)
            except Malformed as e:
                runs.append({"file": path.name, "model": MODELS[model], "malformed": str(e)})
                continue
            runs.append({"file": path.name, "model": MODELS[model], "steps": run.steps, "stopped": run.stopped})
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    jobs = jobs["list"]
    (work / "jobs.json").write_text(json.dumps(jobs, indent=1))
    (work / "runs.json").write_text(json.dumps(runs, indent=1))
    print(f"{len(runs)} runs, {len(jobs)} shaders")


def compile_jobs(dll, work):
    import ctypes
    compiler = ctypes.WinDLL(dll)
    blob_p = ctypes.POINTER(ctypes.c_void_p)
    compiler.D3DCompile.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint,
                                    blob_p, blob_p]
    compiler.D3DCompile.restype = ctypes.c_long
    compiler.D3DDisassemble.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_char_p, blob_p]
    compiler.D3DDisassemble.restype = ctypes.c_long

    def take(blob):
        if not blob:
            return b""
        table = ctypes.cast(blob, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        try:
            pointer = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(table[3])(blob)
            size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(table[4])(blob)
            return ctypes.string_at(pointer, size)
        finally:
            ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])(blob)

    work = Path(work)
    out = []
    for job in json.loads((work / "jobs.json").read_text()):
        if "source" not in job:
            out.append({})
            continue
        text = job["source"].encode()
        shader, messages = ctypes.c_void_p(), ctypes.c_void_p()
        hr = compiler.D3DCompile(text, len(text), None, None, None, b"main", job["profile"].encode(),
                                 job["options"], 0, ctypes.byref(shader), ctypes.byref(messages))
        errors = take(messages).rstrip(b"\0").decode(errors="replace")
        entry = {"hr": hr & 0xffff_ffff, "messages": errors}
        if hr >= 0:
            tokens = take(shader)
            entry["tokens"] = tokens.hex()
            listing = ctypes.c_void_p()
            if compiler.D3DDisassemble(tokens, len(tokens), 0, None, ctypes.byref(listing)) >= 0:
                entry["disassembly"] = take(listing).rstrip(b"\0").decode(errors="replace")
        out.append(entry)
    (work / "compiled.json").write_text(json.dumps(out, indent=1))


def hex_tokens(text):
    """parse_hex: whitespace separated words of eight hex digits, `%` to the end of a line a comment."""
    words = []
    for line in text.splitlines():
        for word in line.split("%")[0].split():
            if len(word) != 8:
                raise Malformed(f"hex word '{word}'")
            words.append(int(word, 16))
    return b"".join(w.to_bytes(4, "little") for w in words)


def merge(work, out, commit):
    import hashlib
    work = Path(work)
    jobs = json.loads((work / "jobs.json").read_text())
    compiled = json.loads((work / "compiled.json").read_text())
    runs = json.loads((work / "runs.json").read_text())
    programs, origins = {}, {}
    names = []
    for job, done in zip(jobs, compiled):
        if "hex" in job:
            tokens = hex_tokens(job["hex"])
        elif "form" in job:
            names.append((f"a {job['form']} shader", job["from"]))
            continue
        elif done.get("tokens") is not None:
            tokens = bytes.fromhex(done["tokens"])
        else:
            names.append((f"refused by the compiler ({done['hr']:#x})", job["from"]))
            continue
        name = hashlib.sha1(tokens).hexdigest()[:16]
        programs[name] = tokens.hex()
        origins.setdefault(name, []).append(job["from"])
        names.append((name, job["from"]))
    cases, left_out = [], []
    for run in runs:
        if "malformed" in run:
            left_out.append({"file": run["file"], "model": run["model"], "reason": run["malformed"]})
            continue
        section = None
        for step in run["steps"]:
            if "section" in step:
                section = {"file": run["file"], "model": run["model"], "run": step["section"],
                           "line": step["line"], "steps": []}
                cases.append(section)
                continue
            step = dict(step)
            if "draw" in step:
                name, origin = names[step["draw"]]
                if name not in programs:
                    step = {"unrunnable": f"{origin}: {name}", "line": step["line"]}
                else:
                    step["draw"] = name
            section["steps"].append(step)
        if run["stopped"]:
            left_out.append({"file": run["file"], "model": run["model"], **run["stopped"]})
    cases = [c for c in cases if any("probe" in s for s in c["steps"])]
    Path(out).write_text(json.dumps({
        "source": f"vkd3d {commit} tests/hlsl, read for its Direct3D 9 runner on Windows",
        "programs": dict(sorted(programs.items())),
        "origins": dict(sorted(origins.items())),
        "cases": cases,
        "left_out": left_out,
    }, indent=1) + "\n")
    print(f"{len(programs)} programs, {len(cases)} cases, {len(left_out)} runs stopped or malformed")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "extract":
        extract(sys.argv[2], sys.argv[3])
    elif command == "compile":
        compile_jobs(sys.argv[2], sys.argv[3])
    elif command == "merge":
        merge(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
