"""vkd3d's HLSL shader tests as runs of a Direct3D runner on Windows: the programs Microsoft's compilers make of the
tests' shaders, and every step a run takes with what the test expects it to read back.

    python3 vkd3d_cases.py extract <vkd3d>/tests/hlsl <work> <runner>...   (Linux) the runs and the shaders to compile
    python vkd3d_cases.py compile-fxc <dll> <work> [<previous work>]      (Windows Python) D3DCompile, D3DDisassemble
    python3 vkd3d_cases.py compile-dxc <dxc> <work> [<previous work>]      (Linux) dxc and its -dumpbin listing
    python3 vkd3d_cases.py merge <work> <out dir> <vkd3d commit>           (Linux) d3d9, dxbc, dxil .json.gz

A runner is one of vkd3d's Windows runners and its model range: `d3d9` (2.0 and 3.0), `d3d11` (4.0 to 5.0), `d3d12`
(4.0 to 5.1, compiled by d3dcompiler_47) and `d3d12-dxc` (6.0 to 6.6, compiled by dxcompiler). The test files are read
as vkd3d's shader runner reads them (tests/shader_runner.c, at the commit the output names) for that runner: its tags
(`windows` and the runner's), `todo` and `bug` ignored since on Windows every check must hold, `winbug` lines left out.
A runner here is a device with every capability vkd3d's runner can report for it (the capabilities a device lacks only
remove runs). A line qualified by a tag of the hardware (`amd`, `intel`, `nvidia`, `warp`, or a version of a runner) is
left out where it is a probe, and stops its run where it changes state, since what follows depends on the machine.

This script has no bytecode logic: the programs are Microsoft's compilers', and what a run expects is the test file's.

A run is one test file at one model on one runner, with every step in order: vkd3d's runner keeps its constant buffers,
resources and render targets from one [test] section to the next. Each [test] section is marked `selected` where
vkd3d's runner would run its tests at that model, `extended` where it would not but the model is in the range the file
requires and the shaders compile (the same expectations at a model vkd3d's own runs leave out), and not run otherwise.
"""
import json
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

FORMAT = 2
MODELS = ["2.0", "3.0", "4.0", "4.1", "5.0", "5.1", "6.0", "6.1", "6.2", "6.5", "6.6"]
TAGS = ["amd", "d3d11", "d3d12", "d3d9", "glsl", "intel", "llvm", "llvmpipe", "mesa", "msl", "mvk", "nvidia",
        "opengl", "radeonsi", "radv", "spirv", "vulkan", "warp", "windows"]
HARDWARE = {"amd", "intel", "nvidia", "warp"}
CAPS = ["barycentrics", "clip-planes", "cull-distance", "depth-bounds", "descriptor-indexing", "descriptors",
        "filter-minmax", "float64", "fog", "geometry-shader", "int64", "mesh-shader", "min-lod-clamp",
        "native-16-bit", "no-refactoring", "point-size", "rov", "rt-vp-array-index", "stencil-export",
        "tessellation-shader", "texture-swizzle", "wave-ops"]
D3D9_CAPS = {"clip-planes", "fog", "point-size"}
# shader_runner_d3d11.c's d3d11_runner_init: what any device reports, with the optional features present
D3D11_CAPS = {"cull-distance", "filter-minmax", "float64", "geometry-shader", "no-refactoring", "rov",
              "rt-vp-array-index", "stencil-export", "tessellation-shader"}
# shader_runner_d3d12.c's d3d12_runner_init_caps, every optional feature present
D3D12_CAPS = set(CAPS) - D3D9_CAPS
RUNNERS = {
    "d3d9": {"models": (0, 1), "tags": {"d3d9", "windows"}, "caps": D3D9_CAPS, "compiler": "fxc"},
    "d3d11": {"models": (2, 4), "tags": {"d3d11", "windows"}, "caps": D3D11_CAPS, "compiler": "fxc"},
    "d3d12": {"models": (2, 5), "tags": {"d3d12", "windows"}, "caps": D3D12_CAPS, "compiler": "fxc"},
    "d3d12-dxc": {"models": (6, 10), "tags": {"d3d12", "windows"}, "caps": D3D12_CAPS, "compiler": "dxc"},
}
OPTIONS = [(0, "none"), (1 << 3, "row-major"), (1 << 4, "column-major"), (1 << 12, "backcompat"), (1 << 13, "ieee-strictness"),
           (1 << 20, "unbounded-descriptor-arrays")]
BACKCOMPAT = 1 << 12
SHADER_SECTIONS = [("[compute-shader", "cs"), ("[pixel-shader", "ps"), ("[pixel shader", "ps"),
                   ("[vertex-shader", "vs"), ("[effect", "fx"), ("[hull-shader", "hs"), ("[domain-shader", "ds"),
                   ("[geometry-shader", "gs"), ("[amplification-shader", "as"), ("[mesh-shader", "ms")]
FORMATS = [("unknown", 0), ("r32g32b32a32-float", 16), ("r32g32b32a32-uint", 16), ("r32g32b32a32-sint", 16),
           ("r32g32b32-float", 12), ("r32g32-float", 8), ("r32g32-uint", 8), ("r32g32-sint", 8),
           ("d32-float-s8x24-uint", 8), ("x32-typeless-g8x24-uint", 8), ("r8g8b8a8-unorm", 4), ("r8g8b8a8-uint", 4),
           ("r32-typeless", 4), ("r32-float-shadow", 4), ("r32-float", 4), ("r32-uint", 4), ("r32-sint", 4),
           ("r8-uint", 1), ("b8g8r8a8-unorm", 4)]
COMPARISONS = ["less equal", "not equal", "greater equal", "never", "less", "equal", "greater", "always"]
STENCIL_OPS = ["keep", "zero", "replace", "increment-clamp", "decrement-clamp", "invert", "increment-wrap",
               "decrement-wrap"]
FILTERS = ["point point point", "point point linear", "point linear point", "point linear linear",
           "linear point point", "linear point linear", "linear linear point", "linear linear linear"]
ADDRESS = ["border", "clamp", "mirror_once", "mirror", "wrap"]
TOPOLOGIES = ["triangle list", "triangle strip", "point list", "1 control point patch list",
              "2 control point patch list", "3 control point patch list", "4 control point patch list"]
# the vertex shader `draw quad` gives a draw without one, and the triangle it draws (shader_runner.c)
QUAD_VS = "float4 main(float4 pos : position) : sv_position\n{\n    return pos;\n}"
QUAD = [-2.0, -2.0, -2.0, 4.0, 4.0, -2.0]
RENDER_TARGET = (640, 480)
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


def f64_words(text):
    """The two words, low first, of the double strtod makes of `text`."""
    import struct
    t = text.strip().lower()
    if t.lstrip("+-").startswith("0x"):
        value = float.fromhex(t)
    else:
        value = float(t)
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return [bits & 0xffff_ffff, bits >> 32]


FLOAT = re.compile(r"\s*([+-]?(?:0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*(?:[pP][+-]?\d+)?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
                   r"|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN]))")
INTEGER = re.compile(r"\s*([+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*))")


def read_float(line):
    m = FLOAT.match(line)
    if not m:
        raise Malformed(f"float at '{line}'")
    return f32_bits(m.group(1)), line[m.end():]


def read_double(line):
    m = FLOAT.match(line)
    if not m:
        raise Malformed(f"double at '{line}'")
    return f64_words(m.group(1)), line[m.end():]


def read_integer(line):
    """strtol/strtoul with base 0: the value and the rest."""
    m = INTEGER.match(line)
    if not m:
        raise Malformed(f"integer at '{line}'")
    text = m.group(1)
    negative = text.startswith("-")
    digits = text.lstrip("+-")
    value = int(digits, 16) if digits[:2].lower() == "0x" else int(digits, 8) if digits.startswith("0") else int(digits)
    return (-value if negative else value), line[m.end():]


def strtoul(line):
    """strtoul(line, &rest, 10) or None where it reads nothing."""
    m = re.match(r"\s*([+-]?\d+)", line)
    if not m:
        return None, line
    return int(m.group(1)) & 0xffff_ffff, line[m.end():]


def strtof(line):
    m = FLOAT.match(line)
    if not m:
        return None, line
    return f32_bits(m.group(1)), line[m.end():]


def match(line, token):
    """vkd3d's match_string: `token` after leading space, followed by space or a parenthesis; the rest or None."""
    s = line.lstrip()
    if not s.startswith(token):
        return None
    after = s[len(token):]
    if not after or not (after[0].isspace() or after[0] in "()]"):
        return None
    return after.lstrip()


def model_string(s):
    for n, name in enumerate(MODELS):
        if s.startswith(name):
            return n, s[len(name):]
        if s[:1] == name[0] and name[1:] == ".0" and s[1:2] != "." and not s[1:2].isdigit():
            return n, s[1:]
    raise Malformed(f"shader model '{s}'")


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


class Runner:
    def __init__(self, name):
        spec = RUNNERS[name]
        self.name = name
        self.first, self.last = spec["models"]
        self.tags = spec["tags"]
        self.caps = spec["caps"]
        self.compiler = spec["compiler"]

    def tag(self, s):
        """match_shader_runner_tag at `s`: (holds or UNKNOWN, rest) or None."""
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
            if name in HARDWARE or (compared and name in self.tags):
                return UNKNOWN, s
            holds = name in self.tags
            return (not holds) if negate else holds, s
        return None


class Run:
    """One test file at one model on one runner, read as shader_runner.c's run_shader_tests_for_model reads it."""

    def __init__(self, name, runner, model, header_mask, jobs):
        self.name, self.runner, self.model, self.header_mask, self.jobs = name, runner, model, header_mask, jobs
        self.minimum, self.maximum = runner.first, runner.last
        self.options, self.skip, self.missing_caps, self.uncertain = 0, False, False, False
        self.native16 = False
        self.denorm = None
        self.hrs = [True] * len(MODELS)
        self.shaders = {}
        self.steps = []
        self.stopped = None
        self.resources = set()

    # -- the qualifiers of a line --

    def qualifier(self, s):
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
                    part = conj(part, self.model >= m)
                elif s.startswith("sm<"):
                    m, s = model_string(s[3:])
                    sub &= (1 << m) - 1
                    part = conj(part, self.model < m)
                elif s.startswith("sm="):
                    m, s = model_string(s[3:])
                    sub &= 1 << m
                    part = conj(part, self.model == m)
                else:
                    t = self.runner.tag(s)
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

    def with_args(self, line, token):
        """match_string_with_args: None where `token` is not there, else (holds, rest)."""
        s = line.lstrip()
        if not s.startswith(token):
            return None
        after = s[len(token):]
        if not after or not (after[0].isspace() or after[0] in "()]"):
            return None
        holds, _, rest = self.qualifier(after)
        return holds, rest.lstrip()

    # -- shaders and the programs a step runs --

    def job(self, stage):
        """The compile job of `stage`'s current shader: compile_hlsl compiles at the step, with the options in force."""
        sh = self.shaders[stage]
        model = MODELS[self.model]
        origin = f"{self.name}:{sh['start']}"
        if sh["form"] != "hlsl":
            key = ("hex", sh["form"], sh["source"])
            job = {"hex": sh["source"], "form": sh["form"], "from": origin}
        else:
            options = self.options & ~BACKCOMPAT if self.model >= 5 else self.options
            profile = f"{stage}_{model.replace('.', '_')}"
            dxc = self.model >= 6
            key = (sh["source"], profile, options, dxc and self.native16, dxc and self.denorm)
            job = {"source": sh["source"], "profile": profile, "options": options, "from": origin,
                   "compiler": "dxc" if dxc else "fxc"}
            if dxc and self.native16:
                job["native16"] = True
            if dxc and self.denorm:
                job["denorm"] = self.denorm
        index = self.jobs["index"].setdefault(key, len(self.jobs["list"]))
        if index == len(self.jobs["list"]):
            self.jobs["list"].append(job)
        return index

    def stop(self, line_number, reason):
        if self.stopped is None:
            self.stopped = {"line": line_number, "reason": reason}

    def step(self, step, number):
        step["line"] = number
        self.steps.append(step)

    # -- which [test] sections run --

    def test_mask(self):
        """get_runner_test_mask for TEST_ACTION_RUN."""
        mask = self.header_mask
        sm1 = (1 << 2) - 1
        sm4 = ((1 << 6) - 1) & ~sm1
        sm6 = ((1 << len(MODELS)) - 1) & ~(sm1 | sm4)
        if "hs" in self.shaders:
            sm4 &= ~((1 << 4) - 1)
        for group in (sm1, sm4, sm6):
            if not mask & group:
                for j in range(self.minimum, self.maximum + 1):
                    if group >> j & 1 and self.hrs[j]:
                        mask |= 1 << j
                        break
        return mask

    def runs_tests(self):
        """should_skip_execution for a run, and whether vkd3d's own runner would select this model."""
        if self.skip:
            return None
        if self.uncertain:
            return UNKNOWN
        if not self.minimum <= self.model <= self.maximum:
            return None
        if not self.hrs[self.model]:
            return None
        return "selected" if self.test_mask() >> self.model & 1 else "extended"

    # -- sections --

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
            # one pass over the table in its order, as the C loop takes it: "column-major row-major" is column-major
            self.options = 0
            for value, name in OPTIONS:
                r = match(rest, name)
                if r is not None:
                    self.options |= value
                    rest = r
            return
        if match(line, "format") is not None:
            return  # every format capability is present
        for cap in CAPS:
            if match(line, cap) is not None:
                if cap == "native-16-bit":
                    self.native16 = True
                if cap not in self.runner.caps:
                    self.missing_caps = True
                return
        s, holds_any = line.strip(), False
        while s:
            holds = True
            while s and not s.startswith("|"):
                s = s.lstrip()
                t = self.runner.tag(s)
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
        """read_shader_directive: the source form, and the models at which the shader fails to compile."""
        self.hrs = [True] * len(MODELS)
        form = "hlsl"
        s = rest
        while s and not s.startswith("]"):
            s = s.lstrip()
            for word in ("todo", "fail", "notimpl"):
                if s.startswith(word):
                    _, mask, s = self.qualifier(s[len(word):])
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

    # -- one line of a [test] section --

    def resource_ref(self, line):
        """parse_resource_reference: ((type, slot), rest) or None."""
        for name, slot in (("cbv-f32", 0), ("cbv-i32", 1), ("cbv-bool", 2)):
            r = match(line, name)
            if r is not None:
                return ("cbv", slot), r
        m = re.match(r"\s*cbv-(\d+)", line)
        if m:
            return ("cbv", int(m.group(1))), line[m.end():]
        r = match(line, "dsv")
        if r is not None:
            return ("dsv", 0), r
        for kind in ("rtv", "srv", "uav", "vb"):
            m = re.match(rf"\s*{kind}-(\d+)", line)
            if m:
                return (kind, int(m.group(1))), line[m.end():]
        return None

    def literal(self, line):
        """read_literal: ({kind, count, words}, rest), the words the union's memory holds."""
        def items(rest, reader, most):
            if not rest.startswith("("):
                raise Malformed(f"literal at '{rest}'")
            rest, values = rest[1:], []
            while True:
                v, rest = reader(rest)
                values.append(v)
                rest = rest.lstrip()
                if rest.startswith(",") and len(values) < most:
                    rest = rest[1:]
                    continue
                break
            if not rest.startswith(")"):
                raise Malformed(f"literal close at '{rest}'")
            return values, rest[1:]

        for kind in ("f32", "rgba"):
            r = match(line, kind)
            if r is not None:
                v, rest = items(r, read_float, 4)
                if len(v) == 3:
                    raise Malformed(f"literal of three floats: '{line}'")
                return {"kind": "f32", "count": len(v), "words": v}, rest
        r = match(line, "f64")
        if r is not None:
            v, rest = items(r, read_double, 2)
            return {"kind": "f64", "count": len(v), "words": [w for d in v for w in d]}, rest
        for kind, bits in (("u8", 8), ("u16", 16), ("i16", 16)):
            r = match(line, kind)
            if r is not None:
                v, rest = items(r, read_integer, 4)
                if len(v) == 3:
                    raise Malformed(f"literal of three: '{line}'")
                packed = 0
                for n, x in enumerate(v):
                    packed |= (x & ((1 << bits) - 1)) << (n * bits)
                words = [(packed >> (32 * n)) & 0xffff_ffff for n in range((len(v) * bits + 31) // 32)]
                return {"kind": kind, "count": len(v), "words": words}, rest
        for kind in ("u32", "i32"):
            r = match(line, kind)
            if r is not None:
                v, rest = items(r, read_integer, 4)
                if len(v) == 3:
                    raise Malformed(f"literal of three: '{line}'")
                return {"kind": kind, "count": len(v), "words": [x & 0xffff_ffff for x in v]}, rest
        for kind in ("u64", "i64"):
            r = match(line, kind)
            if r is not None:
                v, rest = items(r, read_integer, 2)
                words = []
                for x in v:
                    x &= (1 << 64) - 1
                    words += [x & 0xffff_ffff, x >> 32]
                return {"kind": kind, "count": len(v), "words": words}, rest
        return None

    def default_target(self):
        """set_default_target: a 640 x 480 float target where the run has neither its own nor a depth buffer."""
        if ("rtv", 0) in self.resources or ("dsv", 0) in self.resources:
            return None
        self.resources.add(("rtv", 0))
        return {"resource": {"type": "rtv", "slot": 0, "format": "r32g32b32a32-float", "dimension": "2d",
                             "width": RENDER_TARGET[0], "height": RENDER_TARGET[1], "levels": 1, "default": True}}

    def draw_programs(self):
        """The programs a draw runs, by stage."""
        return {stage: self.job(stage) for stage in ("vs", "hs", "ds", "gs", "ps") if stage in self.shaders}

    def test_line(self, line, number):
        """parse_test_directive: one step, or a stop where the line is not one this reading models."""
        skip, more = False, True
        while more:
            # shader_runner.c's loop: a qualifier that does not hold is still read past, but does not repeat the loop
            more = False
            for word in ("todo", "bug"):
                m = self.with_args(line, word)
                if m is not None:
                    line = m[1]
                    more = more or m[0] is not False
            m = self.with_args(line, "winbug")
            if m is not None:
                if m[0] is UNKNOWN:
                    raise Unknown(line)
                line = m[1]
                if m[0]:
                    skip, more = True, True
            m = self.with_args(line, "if")
            if m is not None:
                if m[0] is UNKNOWN:
                    raise Unknown(line)
                if not m[0]:
                    skip = True
                line, more = m[1], True
        if skip:
            return

        for word in ("dispatch-mesh", "dispatch"):
            rest = match(line, word)
            if rest is None:
                continue
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)", rest)
            if not m:
                raise Malformed(line)
            groups = [int(g) for g in m.groups()]
            if word == "dispatch":
                if "cs" not in self.shaders:
                    return self.stop(number, "a dispatch with no compute shader")
                return self.step({"dispatch": groups, "programs": {"cs": self.job("cs")}}, number)
            target = self.default_target()
            if target:
                self.step(target, number)
            programs = {s: self.job(s) for s in ("as", "ms", "ps") if s in self.shaders}
            return self.step({"dispatch_mesh": groups, "programs": programs}, number)
        rest = match(line, "clear")
        if rest is not None:
            ref = self.resource_ref(rest)
            if ref is None:
                raise Malformed(line)
            (kind, slot), rest = ref
            if kind == "rtv":
                target = self.default_target()
                if target:
                    self.step(target, number)
            lit, rest = self.literal(rest)
            step = {"clear": f"{kind}-{slot}" if kind != "dsv" else "dsv", "value": lit["words"]}
            if kind == "dsv":
                s, _ = self.literal(rest)
                step["stencil"] = s["words"][0]
            return self.step(step, number)
        rest = match(line, "depth-bounds")
        if rest is not None:
            lo, rest = strtof(rest)
            hi, _ = strtof(rest)
            return self.step({"state": {"depth_bounds": [lo, hi]}}, number)
        rest = match(line, "depth")
        if rest is not None:
            return self.step({"state": {"depth": self.comparison(rest)[0]}}, number)
        rest = match(line, "draw quad")
        if rest is not None:
            if ("hs" in self.shaders) != ("ds" in self.shaders):
                raise Malformed("a hull shader without a domain shader")
            target = self.default_target()
            if target:
                self.step(target, number)
            if "vs" not in self.shaders:
                # shader_runner.c keeps the quad's vertex shader as the run's from here on
                self.shaders["vs"] = {"kind": "vs", "form": "hlsl", "source": QUAD_VS, "start": 0}
            return self.step({"draw": self.draw_programs(), "quad": QUAD}, number)
        rest = match(line, "draw")
        if rest is not None:
            if ("hs" in self.shaders) != ("ds" in self.shaders):
                raise Malformed("a hull shader without a domain shader")
            target = self.default_target()
            if target:
                self.step(target, number)
            for topology in TOPOLOGIES:
                r = match(rest, topology)
                if r is not None:
                    rest = r
                    break
            else:
                raise Malformed(f"topology '{rest}'")
            counts = []
            for _ in range(4):
                v, rest = strtoul(rest)
                if v is None:
                    break
                counts.append(v)
            if not counts:
                raise Malformed(line)
            counts += [1, 0, 0][len(counts) - 1:]
            if "vs" not in self.shaders:
                return self.stop(number, "a draw with no vertex shader")
            return self.step({"draw": self.draw_programs(), "topology": topology, "vertices": counts[0],
                              "instances": counts[1], "first_vertex": counts[2], "first_instance": counts[3]},
                             number)
        rest = match(line, "copy")
        if rest is not None:
            (a, sa), rest = self.resource_ref(rest)
            (b, sb), rest = self.resource_ref(rest)
            return self.step({"copy": [f"{a}-{sa}", f"{b}-{sb}"]}, number)
        rest = match(line, "probe")
        if rest is not None:
            ref = self.resource_ref(rest)
            if ref is None:
                ref = (("rtv", 0), rest)
            (kind, slot), rest = ref
            if (kind, slot) not in self.resources:
                raise Malformed(f"probe of an undefined resource: {line.strip()}")
            layer = 0
            m4 = re.match(r"\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", rest)
            m3 = re.match(r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", rest)
            m2 = re.match(r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", rest)
            m1 = re.match(r"\s*\(\s*(\d+)\s*\)", rest)
            if m4:
                rect, rest = [int(x) for x in m4.groups()], rest[m4.end():]
            elif m3:
                g = [int(x) for x in m3.groups()]
                rect, layer, rest = [g[0], g[1], g[0] + 1, g[1] + 1], g[2], rest[m3.end():]
            elif m2:
                g = [int(x) for x in m2.groups()]
                rect, rest = [g[0], g[1], g[0] + 1, g[1] + 1], rest[m2.end():]
            elif m1:
                g = int(m1.group(1))
                rect, rest = [g, 0, g + 1, 1], rest[m1.end():]
            else:
                raise Malformed(f"probe arguments: {line.strip()}")
            lit, rest = self.literal(rest)
            step = {"probe": rect, "target": "dsv" if kind == "dsv" else f"{kind}-{slot}", "kind": lit["kind"],
                    "count": lit["count"], "expect": lit["words"]}
            if lit["kind"] in ("f32", "f64"):
                u = re.match(r"\s*(\d+)", rest)
                step["ulps"] = int(u.group(1)) if u else 0
            if layer:
                step["layer"] = layer
            return self.step(step, number)
        rest = match(line, "write")
        if rest is not None:
            ref = self.resource_ref(rest)
            if ref is None or ref[0][0] != "cbv":
                raise Malformed(f"write: {line.strip()}")
            (_, slot), rest = ref
            m = re.match(r"\s*\(\s*(\d+)\s*\)", rest)
            if not m:
                raise Malformed(f"write arguments: {line.strip()}")
            lit, _ = self.literal(rest[m.end():])
            return self.step({"write": slot, "at": int(m.group(1)), "bits": lit["words"]}, number)
        rest = match(line, "uniform")
        if rest is not None:
            m = re.match(r"\s*(\d+)\s+float4\s+", rest)
            if not m:
                raise Malformed(f"uniform: {line.strip()}")
            rest = rest[m.end():]
            values = []
            for _ in range(4):
                v, rest = read_float(rest)
                values.append(v)
            return self.step({"write": 0, "at": int(m.group(1)), "bits": values}, number)
        rest = match(line, "sample mask")
        if rest is not None:
            v, _ = read_integer(rest)
            return self.step({"state": {"sample_mask": v & 0xffff_ffff}}, number)
        rest = match(line, "stencil")
        if rest is not None:
            r = rest.lstrip()
            if not r.startswith("("):
                raise Malformed(line)
            func, r = self.comparison(r[1:])
            ops = []
            for _ in range(3):
                for op in STENCIL_OPS:
                    x = match(r, op)
                    if x is not None or r.lstrip().startswith(op + ")"):
                        ops.append(op)
                        r = x if x is not None else r.lstrip()[len(op):]
                        break
                else:
                    raise Malformed(f"stencil op '{r}'")
            return self.step({"state": {"stencil": {"func": func, "fail": ops[0], "depth_fail": ops[1],
                                                    "pass": ops[2]}}}, number)
        rest = match(line, "stencil-ref")
        if rest is not None:
            lit, _ = self.literal(rest)
            return self.step({"state": {"stencil_ref": lit["words"][0]}}, number)
        rest = match(line, "alpha test")
        if rest is not None:
            func, rest = self.comparison(rest)
            ref, _ = strtof(rest)
            return self.step({"state": {"alpha_test": [func, ref]}}, number)
        rest = match(line, "shade mode")
        if rest is not None:
            return self.step({"state": {"flat": match(rest, "flat") is not None}}, number)
        rest = match(line, "clip-plane")
        if rest is not None:
            index, rest = strtoul(rest)
            if match(rest, "disable") is not None:
                return self.step({"state": {"clip_plane": [index, None]}}, number)
            plane = []
            for _ in range(4):
                v, rest = read_float(rest)
                plane.append(v)
            return self.step({"state": {"clip_plane": [index, plane]}}, number)
        rest = match(line, "point-size")
        if rest is not None:
            values = []
            for _ in range(3):
                v, rest = strtof(rest)
                values.append(v)
            return self.step({"state": {"point_size": values}}, number)
        rest = match(line, "point-sprite")
        if rest is not None:
            return self.step({"state": {"point_sprite": match(rest, "on") is not None}}, number)
        rest = match(line, "fog-colour")
        if rest is not None:
            values = []
            for _ in range(4):
                v, rest = read_float(rest)
                values.append(v)
            return self.step({"state": {"fog_colour": values}}, number)
        rest = match(line, "fog")
        if rest is not None:
            for mode in ("disable", "none", "linear", "exp", "exp2"):
                r = match(rest, mode)
                if r is not None:
                    rest = r
                    break
            else:
                raise Malformed(line)
            fog = {"mode": mode}
            for word, value in (("ortho", True), ("non-ortho", False)):
                r = match(rest, word)
                if r is not None:
                    fog["ortho"], rest = value, r
            if mode == "linear":
                fog["start"], rest = read_float(rest)
                fog["end"], rest = read_float(rest)
            elif mode in ("exp", "exp2"):
                fog["density"], rest = read_float(rest)
            return self.step({"state": {"fog": fog}}, number)
        rest = match(line, "viewport")
        if rest is not None:
            index, rest = read_integer(rest)
            parts = rest.strip().strip("()").split(",")
            values = [read_float(p)[0] for p in parts[:4]]
            return self.step({"state": {"viewport": [index, values]}}, number)
        rest = match(line, "bump")
        if rest is not None:
            index, rest = strtoul(rest)
            r = match(rest, "f32")
            if r is None or not r.startswith("("):
                raise Malformed(line)
            r = r[1:]
            matrix = []
            for n in range(4):
                v, r = read_float(r)
                matrix.append(v)
                r = r.lstrip()
                if n < 3:
                    if not r.startswith(","):
                        raise Malformed(line)
                    r = r[1:]
            if not r.startswith(")"):
                raise Malformed(line)
            scale, r = read_float(r[1:])
            offset, _ = read_float(r)
            return self.step({"state": {"bump": [index, matrix, scale, offset]}}, number)
        rest = match(line, "denorm")
        if rest is not None:
            for mode in ("preserve", "ftz", "any"):
                if match(rest, mode) is not None or rest.strip() == mode:
                    self.denorm = mode
                    return
            raise Malformed(line)
        rest = match(line, "projected")
        if rest is not None:
            index, rest = strtoul(rest)
            return self.step({"state": {"projected": [index, match(rest, "enable") is not None]}}, number)
        raise Malformed(f"test directive '{line.strip()}'")

    def comparison(self, s):
        for name in COMPARISONS:
            r = match(s, name)
            if r is not None:
                return name, r
        raise Malformed(f"comparison '{s}'")


def header_mask(lines):
    mask = 0
    in_header = False
    for line in lines:
        if line == "[header]\n":
            in_header = True
        elif line.startswith("["):
            break
        elif line[:1] not in ("%", "\n") and in_header:
            rest = match(line, "test-shader-models")
            if rest is None or not rest.startswith("("):
                raise Malformed(line)
            for word in rest[1:].split(")")[0].split():
                mask |= 1 << MODELS.index(word)
    return mask


def resource_line(resource, line):
    """parse_resource_directive."""
    rest = match(line, "format")
    if rest is not None:
        for name, size in FORMATS:
            if match(rest, name) is not None or rest.strip() == name:
                resource["format"] = name
                return
        raise Malformed(f"format '{rest.strip()}'")
    rest = match(line, "stride")
    if rest is not None:
        resource["stride"] = int(rest.split()[0])
        resource["format"] = "unknown"
        return
    rest = match(line, "size")
    if rest is not None:
        m = re.match(r"\(\s*(\w+)\s*,\s*([\d\s,]+)\)", rest)
        if not m:
            raise Malformed(f"size '{rest}'")
        kind = m.group(1)
        n = [int(x) for x in m.group(2).replace(",", " ").split()]
        resource.update({"height": 1, "depth": 1, "layers": 1})
        if kind in ("buffer", "raw_buffer", "counter_buffer"):
            resource["dimension"], resource["width"] = "buffer", n[0]
            if kind == "raw_buffer":
                resource["raw"] = True
            if kind == "counter_buffer":
                resource["counter"], resource["stride"], resource["format"] = True, 4, "unknown"
        elif kind == "2d":
            resource["dimension"], resource["width"], resource["height"] = "2d", n[0], n[1]
        elif kind == "2dms":
            resource["dimension"], resource["samples"], resource["width"], resource["height"] = "2d", n[0], n[1], n[2]
        elif kind == "2darray":
            resource["dimension"], resource["width"], resource["height"], resource["layers"] = "2d", n[0], n[1], n[2]
        elif kind == "3d":
            resource["dimension"], resource["width"], resource["height"], resource["depth"] = "3d", n[0], n[1], n[2]
        elif kind == "cube":
            resource["dimension"], resource["width"], resource["height"], resource["layers"] = "cube", n[0], n[0], 6
        else:
            raise Malformed(f"size '{rest}'")
        return
    rest = match(line, "levels")
    if rest is not None:
        resource["levels"] = int(rest.split()[0])
        return
    rest = match(line, "min-lod-clamp")
    if rest is not None:
        resource["min_lod_clamp"] = f32_bits(rest.split()[0])
        return
    rest = match(line, "swizzle")
    if rest is not None:
        resource["swizzle"] = rest.strip()[:4]
        return
    # data: each word strtoul(.., 0), or strtof where the number has a point
    s = line
    while True:
        s = s.lstrip()
        if not s:
            break
        m = re.match(r"[+-]?(0[xX][0-9a-fA-F]+|\d+)", s)
        fm = FLOAT.match(s)
        if s.startswith(".") or (m and s[m.end():m.end() + 1] == "."):
            word, s = f32_bits(fm.group(1)), s[fm.end():]
        elif m:
            text = m.group(0)
            neg = text.startswith("-")
            digits = text.lstrip("+-")
            v = int(digits, 16) if digits[:2].lower() == "0x" else int(digits, 8) if digits.startswith("0") and \
                len(digits) > 1 else int(digits)
            word, s = (-v if neg else v) & 0xffff_ffff, s[m.end():]
        else:
            break
        resource.setdefault("data", []).append(word)


def input_layout_line(line):
    """parse_input_layout_directive: slot, format, semantic name and index."""
    slot, rest = strtoul(line)
    rest = rest.lstrip()
    for name, size in FORMATS:
        r = match(rest, name)
        if r is not None:
            break
    else:
        raise Malformed(f"input layout '{line.strip()}'")
    words = r.split()
    element = {"slot": slot, "format": name, "size": size, "name": words[0]}
    element["index"] = int(words[1]) if len(words) > 1 and words[1].isdigit() else 0
    return element


def descriptors_line(line):
    m = re.match(r"\s*([tu])\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*,\s*space\s*(\d+)\s*,\s*(srv|uav)\s*(\d+)", line)
    if not m:
        raise Malformed(f"descriptors '{line.strip()}'")
    t, lo, hi, space, _, target = m.groups()
    return {"type": "srv" if t == "t" else "uav", "register": int(lo), "count": int(hi) - int(lo) + 1,
            "space": int(space), "target": int(target)}


def read_file(path, runner, model, jobs):
    """One run of `path` at `model` on `runner`; shaders to compile are added to `jobs`."""
    lines = path.read_text().splitlines(keepends=True)
    run = Run(path.name, runner, model, header_mask(lines), jobs)
    state, source, start, current, section_tests = None, [], 0, None, None
    resource, sampler, layout, descriptors = None, None, None, None
    number = 0

    def close():
        nonlocal source
        if state == "require":
            if run.maximum < run.minimum:
                run.skip = True
            elif run.missing_caps:
                run.skip = True
        elif state == "shader":
            current["source"] = "".join(source)
            current["start"] = start
            run.shaders[current["kind"]] = current
        elif state == "resource":
            if resource["type"] == "vb":
                resource["width"] = 4 * len(resource.get("data", []))
            if not run.skip and not run.uncertain:
                run.resources.add((resource["type"], resource["slot"]))
                run.step({"resource": resource}, start)
        elif state == "sampler" and not run.skip and not run.uncertain:
            run.step({"sampler": sampler}, start)
        elif state == "input-layout":
            run.step({"input_layout": layout}, start)
        elif state == "descriptors":
            run.step({"descriptors": descriptors}, start)
        source = []

    for raw in lines + [None]:
        number += 1
        if raw is None or raw.startswith("["):
            close()
            if raw is None:
                break
            start, line = number, raw
            kind = next((k for p, k in SHADER_SECTIONS if match(line, p) is not None), None)
            index = re.match(r"\[(sampler|rtv|srv|uav|vb)[- ](\d+)\]\n", line)
            if kind is not None:
                state = "shader"
                prefix = next(p for p, k in SHADER_SECTIONS if match(line, p) is not None)
                current = run.shader_header(line.lstrip()[len(prefix):], kind)
            elif line == "[require]\n":
                state = "require"
                run.minimum, run.maximum = runner.first, runner.last
                run.options, run.skip, run.missing_caps, run.uncertain = 0, False, False, False
                run.native16 = False
            elif index and index.group(1) == "sampler":
                state = "sampler"
                sampler = {"slot": int(index.group(2)), "filter": "point point point",
                           "address": ["clamp", "clamp", "clamp"]}
            elif index:
                state = "resource"
                kind, slot = index.group(1), int(index.group(2))
                resource = {"type": kind, "slot": slot}
                if kind != "vb":
                    resource.update({"format": "r32g32b32a32-float", "levels": 1})
                else:
                    resource["dimension"] = "buffer"
            elif line == "[dsv]\n":
                state = "resource"
                resource = {"type": "dsv", "slot": 0, "format": "d32-float", "levels": 1}
            elif line == "[test]\n":
                state = "test"
                section_tests = run.runs_tests()
                if section_tests is UNKNOWN:
                    run.stop(number, "a [require] that depends on the machine")
                    section_tests = None
                if run.stopped is None:
                    run.step({"section": section_tests}, number)
            elif line == "[input-layout]\n":
                state, layout = "input-layout", []
            elif line == "[descriptors]\n":
                state, descriptors = "descriptors", []
            elif line in ("[header]\n", "[preproc]\n", "[preproc fail]\n"):
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
        elif state == "resource":
            resource_line(resource, raw)
        elif state == "sampler":
            sampler_line(sampler, raw)
        elif state == "input-layout":
            layout.append(input_layout_line(raw))
        elif state == "descriptors":
            descriptors.append(descriptors_line(raw))
        elif state == "test" and section_tests is not None and run.stopped is None:
            try:
                run.test_line(raw, number)
            except Unknown:
                if not raw.split() or "probe" not in raw:
                    run.stop(number, "a state change that depends on the machine")
    return run


def sampler_line(sampler, line):
    """parse_sampler_directive."""
    rest = match(line, "address")
    if rest is not None:
        modes = []
        for _ in range(3):
            for mode in ADDRESS:
                r = match(rest, mode)
                if r is not None or rest.strip() == mode:
                    modes.append(mode)
                    rest = r if r is not None else ""
                    break
            else:
                raise Malformed(f"address '{rest}'")
        sampler["address"] = modes
        return
    rest = match(line, "filter")
    if rest is not None:
        reduction = "comparison" if sampler.get("comparison") else "standard"
        for word in ("minimum", "maximum"):
            r = match(rest, word)
            if r is not None and reduction == "standard":
                reduction, rest = word, r
        for name in FILTERS:
            if rest.strip().startswith(name):
                sampler["filter"], sampler["reduction"] = name, reduction
                return
        raise Malformed(f"filter '{rest}'")
    rest = match(line, "comparison")
    if rest is not None:
        for name in COMPARISONS:
            if match(rest, name) is not None or rest.strip() == name:
                sampler["comparison"], sampler["reduction"] = name, "comparison"
                return
        raise Malformed(f"comparison '{rest}'")
    raise Malformed(f"sampler '{line.strip()}'")


def extract(tests, work, runners):
    jobs, runs = {"index": {}, "list": []}, []
    for path in sorted(Path(tests).glob("*.shader_test")):
        for name in runners:
            runner = Runner(name)
            for model in range(runner.first, runner.last + 1):
                try:
                    run = read_file(path, runner, model, jobs)
                except Malformed as e:
                    runs.append({"file": path.name, "runner": name, "model": MODELS[model], "malformed": str(e)})
                    continue
                runs.append({"file": path.name, "runner": name, "model": MODELS[model], "steps": run.steps,
                             "stopped": run.stopped})
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    jobs = jobs["list"]
    (work / "jobs.json").write_text(json.dumps(jobs, indent=1))
    (work / "runs.json").write_text(json.dumps(runs, indent=1))
    print(f"{len(runs)} runs, {len(jobs)} shaders "
          f"({sum(1 for j in jobs if j.get('compiler') == 'fxc')} fxc, "
          f"{sum(1 for j in jobs if j.get('compiler') == 'dxc')} dxc)")


def job_key(job):
    """What a compile depends on: a job without the line it came from."""
    return json.dumps({k: v for k, v in job.items() if k != "from"}, sort_keys=True)


def previous_results(previous, compiler):
    """The results of an earlier work directory's compiles by `compiler`, by job key, to reuse."""
    if previous is None:
        return {}
    previous = Path(previous)
    done = json.loads((previous / f"compiled-{compiler}.json").read_text())
    jobs = json.loads((previous / "jobs.json").read_text())
    return {job_key(jobs[int(n)]): entry for n, entry in done.items()}


def compile_fxc(dll, work, previous=None):
    """D3DCompile and D3DDisassemble each fxc job (Windows Python: ctypes on d3dcompiler_47.dll)."""
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
    out = {}
    reuse = previous_results(previous, "fxc")
    for n, job in enumerate(json.loads((work / "jobs.json").read_text())):
        if job.get("compiler") != "fxc":
            continue
        if job_key(job) in reuse:
            out[str(n)] = reuse[job_key(job)]
            continue
        text = job["source"].encode()
        shader, messages = ctypes.c_void_p(), ctypes.c_void_p()
        try:
            hr = compiler.D3DCompile(text, len(text), None, None, None, b"main", job["profile"].encode(),
                                     job["options"], 0, ctypes.byref(shader), ctypes.byref(messages))
        except OSError as e:
            # the compiler faulted on this source (ctypes turns the access violation into OSError): a run that
            # compiles it would stop here on Windows too
            out[str(n)] = {"crashed": str(e)}
            print(f"job {n}: d3dcompiler faulted: {e}", flush=True)
            continue
        errors = take(messages).rstrip(b"\0").decode(errors="replace")
        entry = {"hr": hr & 0xffff_ffff, "messages": errors}
        if hr >= 0:
            tokens = take(shader)
            entry["tokens"] = tokens.hex()
            listing = ctypes.c_void_p()
            if compiler.D3DDisassemble(tokens, len(tokens), 0, None, ctypes.byref(listing)) >= 0:
                entry["disassembly"] = take(listing).rstrip(b"\0").decode(errors="replace")
        out[str(n)] = entry
    (work / "compiled-fxc.json").write_text(json.dumps(out, indent=1))


def compile_dxc(dxc, work, previous=None):
    """dxc each dxc job with the arguments vkd3d's dxc_compile passes (tests/utils.h), and its -dumpbin listing."""
    work = Path(work)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reuse = previous_results(previous, "dxc")
        for n, job in enumerate(json.loads((work / "jobs.json").read_text())):
            if job.get("compiler") != "dxc":
                continue
            if job_key(job) in reuse:
                out[str(n)] = reuse[job_key(job)]
                continue
            source, binary = tmp / "shader.hlsl", tmp / "shader.dxil"
            source.write_text(job["source"])
            binary.unlink(missing_ok=True)
            args = [dxc, "-T", job["profile"], "-E", "main", "-Qstrip_reflect", "-Qstrip_debug",
                    "-flegacy-macro-expansion"]
            options = job["options"]
            if options & (1 << 3):
                args.append("-Zpr")
            if options & (1 << 4):
                args.append("-Zpc")
            if options & (1 << 12):
                args.append("-Gec")
            if not options & (1 << 20):
                args.append("-flegacy-resource-reservation")
            if options & (1 << 13):
                args.append("-Gis")
            if job.get("native16"):
                args.append("-enable-16bit-types")
            if job.get("denorm"):
                args += ["-denorm", job["denorm"]]
            done = subprocess.run(args + ["-Fo", str(binary), str(source)], capture_output=True, text=True)
            entry = {"hr": 0 if done.returncode == 0 else 0x80004005, "messages": done.stderr}
            if done.returncode == 0:
                entry["tokens"] = binary.read_bytes().hex()
                listing = subprocess.run([dxc, "-dumpbin", str(binary)], capture_output=True, text=True)
                if listing.returncode == 0:
                    entry["disassembly"] = listing.stdout
            out[str(n)] = entry
    (work / "compiled-dxc.json").write_text(json.dumps(out, indent=1))


def hex_tokens(text):
    """parse_hex: whitespace separated words of eight hex digits, `%` to the end of a line a comment."""
    words = []
    for line in text.splitlines():
        for word in line.split("%")[0].split():
            if len(word) != 8:
                raise Malformed(f"hex word '{word}'")
            words.append(int(word, 16))
    return b"".join(w.to_bytes(4, "little") for w in words)


def write_gz(path, text):
    """Write `text` gzipped, reproducibly (no name, no time in the header)."""
    import gzip
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    Path(path).write_bytes(buf.getvalue())


def merge(work, out, commit):
    import hashlib
    work = Path(work)
    jobs = json.loads((work / "jobs.json").read_text())
    compiled = {}
    for name in ("compiled-fxc.json", "compiled-dxc.json"):
        if (work / name).exists():
            compiled.update(json.loads((work / name).read_text()))
    runs = json.loads((work / "runs.json").read_text())
    programs, origins, listings = {}, {}, {}
    names = []
    for n, job in enumerate(jobs):
        done = compiled.get(str(n), {})
        if "hex" in job:
            tokens = hex_tokens(job["hex"])
        elif job.get("compiler") and str(n) not in compiled:
            raise SystemExit(f"job {n} ({job['from']}, {job['profile']}) was not compiled")
        elif done.get("tokens") is not None:
            tokens = bytes.fromhex(done["tokens"])
        elif "crashed" in done:
            names.append((None, f"{job['from']}: {job['profile']}: the compiler faulted ({done['crashed']})"))
            continue
        else:
            names.append((None, f"{job['from']}: {job['profile']} refused by the compiler ({done['hr']:#x})"))
            continue
        name = hashlib.sha1(tokens).hexdigest()[:16]
        programs[name] = tokens.hex()
        origins.setdefault(name, [])
        if job["from"] not in origins[name]:
            origins[name].append(job["from"])
        if done.get("disassembly") is not None:
            listings[name] = done["disassembly"]
        names.append((name, None))
    cases, left_out = [], []
    for run in runs:
        where = {"file": run["file"], "runner": run["runner"], "model": run["model"]}
        if "malformed" in run:
            left_out.append({**where, "reason": run["malformed"]})
            continue
        steps = []
        for step in run["steps"]:
            step = dict(step)
            if "programs" in step or "draw" in step:
                field = "draw" if "draw" in step else "programs"
                stages = {}
                refused = []
                for stage, job in step[field].items():
                    name, why = names[job]
                    if name is None:
                        refused.append(why)
                    else:
                        stages[stage] = name
                if refused:
                    step = {"unrunnable": "; ".join(refused), "line": step["line"]}
                else:
                    step[field] = stages
            steps.append(step)
        if not any(s.get("section") for s in steps):
            if run["stopped"]:
                left_out.append({**where, **run["stopped"]})
            continue
        cases.append({**where, "steps": steps})
        if run["stopped"]:
            left_out.append({**where, **run["stopped"]})
    cases = [c for c in cases if any("probe" in s for s in c["steps"])]
    # one file per bytecode, each with the programs its runs use
    files = {"d3d9": ["d3d9"], "dxbc": ["d3d11", "d3d12"], "dxil": ["d3d12-dxc"]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stem, runners in files.items():
        mine = [c for c in cases if c["runner"] in runners]
        used = set()
        for c in mine:
            for step in c["steps"]:
                for field in ("draw", "programs"):
                    if isinstance(step.get(field), dict):
                        used.update(step[field].values())
        kept = {k: v for k, v in programs.items() if k in used}
        write_gz(out / f"{stem}.json.gz", json.dumps({
            "format": FORMAT,
            "source": f"vkd3d {commit} tests/hlsl, read for its Windows runners {', '.join(runners)}",
            "programs": dict(sorted(kept.items())),
            "origins": {k: v for k, v in sorted(origins.items()) if k in used},
            "cases": mine,
            "left_out": [r for r in left_out if r["runner"] in runners],
        }, indent=1) + "\n")
        write_gz(out / f"{stem}.listings.json.gz", json.dumps(
            {k: v for k, v in sorted(listings.items()) if k in used}, indent=1) + "\n")
        print(f"{stem}: {len(kept)} programs, {len(mine)} runs, "
              f"{sum(1 for r in left_out if r['runner'] in runners)} runs stopped or malformed")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "extract":
        extract(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif command == "compile-fxc":
        compile_fxc(*sys.argv[2:5])
    elif command == "compile-dxc":
        compile_dxc(*sys.argv[2:5])
    elif command == "merge":
        merge(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
