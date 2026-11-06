"""Build the SM5.1 fixtures here with Microsoft's fxc and dxc, and write fxc's dump of each SM5.1 build (a Windows
Python: on Windows, or under Wine with primase-oracles' `tools/wine/winpy`).

Each `.hlsl` states its builds as `// fxc <arguments>` and `// dxc <arguments>` lines; each is run here, in this
folder, as written. Then `<name>.51.asm` is fxc's `-dumpbin` of every `<name>.51.dxbc`, as fxc prints it, with trailing
whitespace removed. fxc names the d3dcompiler_47.dll it loaded in its first line when that is not the system's
("... 10.1 (using <path>)"); that note is dropped, so the dump is the same whichever folder the builds come from.

    python -I generate.py --fxc PATH --dxc PATH

The fixtures are those of fxc 10.0.26100.4654 with its d3dcompiler_47.dll (modding-archive/tools/fxc-10.0.26100.4654)
and the Windows SDK 10.0.26100.4654's dxc.exe with its dxcompiler.dll and dxil.dll
(modding-archive/tools/dxc-win-1.7.2308.16-sdk-10.0.26100.4654); both write the same bytes under Wine as on Windows.
The Linux dxc 1.7.2308 release does not: its containers differ. Under Wine give the tools as Wine paths (Z:\\...).
"""
import argparse, glob, os, re, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fxc", required=True)
    ap.add_argument("--dxc", required=True)
    a = ap.parse_args()
    tools = {"fxc": a.fxc, "dxc": a.dxc}
    failed = 0
    for source in sorted(glob.glob(os.path.join(HERE, "*.hlsl"))):
        for line in open(source, encoding="utf-8"):
            m = re.match(r"//\s*(fxc|dxc)\s+(-.*)", line)
            if not m:
                continue
            r = subprocess.run([tools[m.group(1)], *shlex.split(m.group(2))], cwd=HERE, capture_output=True,
                               text=True)
            if r.returncode != 0:
                failed += 1
                print(f"FAIL {m.group(0)}: {r.stderr.strip() or r.stdout.strip()}")
    for build in sorted(glob.glob(os.path.join(HERE, "*.51.dxbc"))):
        r = subprocess.run([a.fxc, "-dumpbin", os.path.basename(build)], cwd=HERE, capture_output=True)
        if r.returncode != 0:
            failed += 1
            print(f"FAIL fxc -dumpbin {os.path.basename(build)}")
            continue
        lines = [l.rstrip() for l in r.stdout.decode("utf-8").splitlines()]
        lines[0] = re.sub(r" \(using .*\)$", "", lines[0])
        with open(build[: -len(".dxbc")] + ".asm", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
