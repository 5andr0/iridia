"""LLVM's offload-test-suite (github.com/llvm/offload-test-suite) as cases: each test's RUN lines as lit runs them for
the DirectX API with DXC, the programs dxc makes of its shaders, its pipeline description and what it checks.

    python3 offload_cases.py extract <offload-test-suite> <golden-images> <dxc> <out.json.gz> <commit> <golden commit>

The tests are read as lit reads them (test/lit.cfg.py): `split-file` splits a test into its files, `%if <feature> %{
.. %} %else %{ .. %}` in a RUN line is taken for the configuration DirectX + DXC on Windows, `%dxc_target` is dxc
with the arguments the line gives. `REQUIRES`, `UNSUPPORTED` and `XFAIL` are kept as the expressions they are: what a
device has is the runner's to say. The golden images are not copied: a case names each by its path in llvm/offload-golden-images and its SHA-256,
and the runner reads them from a checkout of that repository at the commit the file names. A pipeline's YAML is kept as
data, every buffer's `Data` written as the bytes the
offloader's YAML reader makes of it for its `Format` (lib/Support/Pipeline.cpp: `Hex*` and `Float16` as hexadecimal
integers, integers by their C syntax, floats by strtof and strtod). This script has no bytecode logic: the programs
are dxc's, and what a test expects is the test's (its `Results`, its FileCheck lines, its golden image).
"""
import base64
import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

FEATURES_TRUE = {"DirectX", "DXC", "Windows"}
FEATURES_FALSE = {"Clang", "Vulkan", "Metal", "Darwin", "Linux"}

FORMATS = {
    "Hex8": ("B", 1, "hex"), "Hex16": ("H", 2, "hex"), "Hex32": ("I", 4, "hex"), "Hex64": ("Q", 8, "hex"),
    "UInt16": ("H", 2, "int"), "UInt32": ("I", 4, "int"), "UInt64": ("Q", 8, "int"),
    "Int16": ("h", 2, "int"), "Int32": ("i", 4, "int"), "Int64": ("q", 8, "int"),
    "Float16": ("H", 2, "hex"), "Float32": ("f", 4, "float"), "Depth32": ("f", 4, "float"),
    "Float64": ("d", 8, "double"), "Bool": ("I", 4, "int"),
}


def llvm_integer(text):
    """StringRef::getAsInteger with radix 0: 0x, 0b, 0o and a leading 0 for octal; the yaml reader also takes a sign."""
    t = text.strip()
    neg = t.startswith("-")
    t = t.lstrip("+-")
    low = t.lower()
    if low.startswith("0x"):
        v = int(t[2:], 16)
    elif low.startswith("0b"):
        v = int(t[2:], 2)
    elif low.startswith("0o"):
        v = int(t[2:], 8)
    elif len(t) > 1 and t.startswith("0"):
        v = int(t[1:], 8)
    else:
        v = int(t)
    return -v if neg else v


def strtof_bits(text):
    """The bits of the float strtof makes of `text`, by exact arithmetic (nan and inf with their sign)."""
    from fractions import Fraction
    t = text.strip().lower()
    sign = 0
    if t.startswith(("-", "+")):
        sign = 0x8000_0000 if t[0] == "-" else 0
        t = t[1:]
    if t in ("inf", "infinity", ".inf"):
        return sign | 0x7f80_0000
    if t in ("nan", ".nan"):
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


def strtod_bytes(text):
    t = text.strip().lower()
    neg = t.startswith("-")
    body = t.lstrip("+-")
    if body in ("nan", ".nan"):
        bits = 0x7ff8_0000_0000_0000 | ((1 << 63) if neg else 0)
        return struct.pack("<Q", bits)
    if body in ("inf", "infinity", ".inf"):
        return struct.pack("<d", float("-inf") if neg else float("inf"))
    value = float.fromhex(t) if body.startswith("0x") else float(t)
    return struct.pack("<d", value)


def scalar_bytes(fmt, text):
    code, size, how = FORMATS[fmt]
    if how == "hex":
        return (llvm_integer(text) & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
    if how == "int":
        return (llvm_integer(text) & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
    if how == "float":
        return strtof_bits(text).to_bytes(4, "little")
    return strtod_bytes(text)


def data_bytes(fmt, data):
    """A buffer's `Data` as bytes: a list of scalars, or (for an array of buffers) a list of lists."""
    if fmt not in FORMATS:
        raise ValueError(f"buffer format {fmt}")
    return b"".join(scalar_bytes(fmt, str(x)) for x in data)


def convert_buffers(pipeline):
    """Every buffer's Data (and FillValue) as base64 bytes beside the YAML."""
    for b in pipeline.get("Buffers") or []:
        fmt = b.get("Format")
        data = b.get("Data")
        if data is not None:
            if data and isinstance(data[0], list):
                b["Data"] = [base64.b64encode(data_bytes(fmt, d)).decode() for d in data]
                b["DataArrays"] = True
            else:
                b["Data"] = base64.b64encode(data_bytes(fmt, data)).decode()
        if "FillValue" in b:
            b["FillValue"] = base64.b64encode(scalar_bytes(fmt, str(b["FillValue"]))).decode()
    for p in pipeline.get("PushConstants") or []:
        for v in p.get("Values") or []:
            if "Data" in v:
                v["Data"] = base64.b64encode(data_bytes(v["Format"], v["Data"])).decode()
    return pipeline


def split_file(text):
    """split-file: `#--- name` starts a file, up to the next; the lines before the first belong to none. Each file keeps
    its line numbers by blank lines in front, as split-file does without --no-leading-lines."""
    files, current, lines = {}, None, text.splitlines(keepends=True)
    for n, line in enumerate(lines):
        m = re.match(r"^(?://|#)--- (\S+)", line)
        if m:
            current = m.group(1)
            files[current] = "\n" * (n + 1)
            continue
        if current is not None:
            files[current] += line
    return files


def lit_if(line, holds):
    """lit's `%if feature %{ a %} %else %{ b %}`, outermost first."""
    while True:
        m = re.search(r"%if\s+([^%]+?)\s*%\{", line)
        if not m:
            return line
        start, depth, i = m.end(), 1, m.end()
        while depth and i < len(line):
            if line.startswith("%{", i):
                depth += 1
                i += 2
            elif line.startswith("%}", i):
                depth -= 1
                i += 2
            else:
                i += 1
        then_part, rest = line[start:i - 2], line[i:]
        else_part = ""
        em = re.match(r"\s*%else\s*%\{", rest)
        if em:
            j, depth = em.end(), 1
            while depth and j < len(rest):
                if rest.startswith("%{", j):
                    depth += 1
                    j += 2
                elif rest.startswith("%}", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            else_part, rest = rest[em.end():j - 2], rest[j:]
        chosen = then_part if holds(m.group(1)) else else_part
        line = line[:m.start()] + chosen + rest


def feature_holds(expr):
    """A lit feature expression for DirectX + DXC on Windows, the features of a device left false here (a RUN line's
    %if names compiler and API features only)."""
    py = expr.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", expr)
    env = {n: n in FEATURES_TRUE for n in names}
    for n in names:
        py = re.sub(rf"\b{re.escape(n)}\b", repr(env[n]), py)
    return eval(py, {"__builtins__": {}})


def run_lines(text):
    """The RUN lines, a trailing backslash joining a line to the next."""
    out, pending = [], ""
    for line in text.splitlines():
        m = re.match(r"^\s*(?://|#)\s*RUN:\s?(.*)$", line)
        if not m:
            continue
        part = m.group(1)
        if part.rstrip().endswith("\\"):
            pending += part.rstrip()[:-1] + " "
            continue
        out.append((pending + part).strip())
        pending = ""
    return out


def directives(text, word):
    found = []
    for line in text.splitlines():
        m = re.match(rf"^\s*(?://|#)\s*{word}:\s*(.*)$", line)
        if m:
            found.append(m.group(1).strip())
    return found


def checks(text):
    """FileCheck lines by prefix: (prefix, kind, pattern)."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?://|#)\s*([A-Z][A-Z0-9_-]*?)(-NEXT|-NOT|-LABEL|-SAME|-DAG|-EMPTY)?:\s?(.*)$", line)
        if m and m.group(1) not in ("RUN", "REQUIRES", "UNSUPPORTED", "XFAIL"):
            out.append({"prefix": m.group(1), "kind": (m.group(2) or "").lstrip("-") or "CHECK",
                        "pattern": m.group(3)})
    return out


def write_gz(path, text):
    """Write `text` gzipped, reproducibly (no name, no time in the header)."""
    import gzip
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    Path(path).write_bytes(buf.getvalue())


def extract(suite, golden, dxc, out, commit, golden_commit):
    suite, golden = Path(suite), Path(golden)
    tests = sorted(p for p in (suite / "test").rglob("*") if p.suffix in (".test", ".yaml") and p.is_file()
                   and "Inputs" not in p.parts and p.name not in ("CMakeLists.txt",) and
                   not p.name.startswith("lit."))
    programs, cases, images = {}, [], {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for path in tests:
            text = path.read_text()
            name = str(path.relative_to(suite / "test"))
            lines = run_lines(text)
            if not lines:
                continue
            case = {"test": name, "requires": directives(text, "REQUIRES"),
                    "unsupported": directives(text, "UNSUPPORTED"), "xfail": directives(text, "XFAIL"),
                    "steps": [], "checks": checks(text)}
            files = split_file(text)
            work = tmp / f"t{len(cases)}"
            work.mkdir()
            objects = {}
            for line in lines:
                line = lit_if(line, feature_holds)
                line = line.replace("%t/", f"{work}/").replace("%t", f"{work}/t").replace("%s", str(path))
                if line.startswith("split-file"):
                    for fname, body in files.items():
                        (work / fname).parent.mkdir(parents=True, exist_ok=True)
                        (work / fname).write_text(body)
                    continue
                command = line.split("|")[0].strip()
                expect_failure = command.startswith("not ")
                command = command[4:].strip() if expect_failure else command
                filecheck = None
                if "| FileCheck" in line:
                    fc = line.split("| FileCheck", 1)[1]
                    m = re.search(r"--check-prefix(?:es)?=(\S+)", fc)
                    filecheck = m.group(1).split(",") if m else ["CHECK"]
                words = command.split()
                tool = words[0] if words else ""
                if tool in ("%dxc_target", "%dxc_target_lib"):
                    args = words[1:]
                    if "-Fo" not in args:
                        case["steps"].append({"unmodelled": f"a compile without -Fo: {command}"})
                        continue
                    target = args[args.index("-Fo") + 1]
                    args = [a for a in args]
                    done = subprocess.run([dxc] + args, capture_output=True, text=True, cwd=work)
                    shown = [a.replace(str(work) + "/", "%t/") for a in args]
                    step = {"compile": shown}
                    if done.returncode == 0 and Path(target).exists():
                        blob = Path(target).read_bytes()
                        h = hashlib.sha1(blob).hexdigest()[:16]
                        programs[h] = base64.b64encode(blob).decode()
                        objects[target] = h
                        step["program"] = h
                    else:
                        step["refused"] = (done.stderr or done.stdout).replace(str(work) + "/", "%t/")[:4000]
                    if expect_failure:
                        step["expect_failure"] = True
                    case["steps"].append(step)
                elif tool == "%offloader":
                    args = words[1:]
                    pipeline_path = next((a for a in args if a.endswith(".yaml")), None)
                    objs = [a for a in args if a in objects or a.endswith(".o")]
                    step = {"offload": True, "programs": [objects.get(o) for o in objs]}
                    if expect_failure:
                        step["expect_failure"] = True
                    if filecheck:
                        step["filecheck"] = filecheck
                        # what FileCheck reads: stdout, and stderr too where the line sends it there
                        if "2>&1" in line.split("|")[0]:
                            step["stderr"] = True
                    if "-r" in args:
                        step["render"] = args[args.index("-r") + 1]
                    try:
                        loaded = yaml.load(Path(pipeline_path).read_text(), Loader=yaml.BaseLoader)
                        step["pipeline"] = convert_buffers(loaded)
                    except Exception as e:  # noqa: BLE001 - a pipeline this reader cannot read is reported
                        step["pipeline_error"] = f"{type(e).__name__}: {e}"
                    case["steps"].append(step)
                elif tool == "imgdiff":
                    args = words[1:]
                    golden_path = next((a for a in args if "%goldenimage_dir" in a), None)
                    rules = args[args.index("-rules") + 1] if "-rules" in args else None
                    step = {"imgdiff": golden_path.replace("%goldenimage_dir/", "") if golden_path else None}
                    if rules:
                        step["rules"] = yaml.load(Path(rules).read_text(), Loader=yaml.BaseLoader)
                    if golden_path:
                        rel = golden_path.replace("%goldenimage_dir/", "")
                        p = golden / rel
                        if p.exists():
                            # the image is fetched at run time (its repository at the commit the file names); the
                            # digest checks it is the one this case was read with
                            images[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
                        else:
                            step["missing"] = True
                    case["steps"].append(step)
                else:
                    case["steps"].append({"unmodelled": command.replace(str(work) + "/", "%t/")[:300]})
            cases.append(case)
    write_gz(out, json.dumps({
        "format": 1,
        "source": f"llvm/offload-test-suite {commit} test/, read for DirectX with DXC; golden images "
                  f"llvm/offload-golden-images {golden_commit}",
        "programs": dict(sorted(programs.items())),
        "images": dict(sorted(images.items())),
        "cases": cases,
    }, indent=1) + "\n")
    print(f"{len(cases)} tests, {len(programs)} programs, {len(images)} golden images")


if __name__ == "__main__":
    if len(sys.argv) != 8 or sys.argv[1] != "extract":
        raise SystemExit(__doc__)
    extract(*sys.argv[2:])
