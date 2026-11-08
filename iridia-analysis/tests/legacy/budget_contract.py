"""Exercise the real supervisor on Windows and Linux, using a tiny framed-protocol worker."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import time


def worker():
    while True:
        length = sys.stdin.buffer.read(4)
        if not length:
            return
        data = sys.stdin.buffer.read(struct.unpack("<I", length)[0]).decode()
        if data == "idle":
            time.sleep(0.6)
        else:
            until = time.process_time() + (0.12 if data == "work" else 5)
            while time.process_time() < until:
                pass
        print("REPLY\t" + json.dumps({"Ok": data}), flush=True)


def check(case):
    script = Path(__file__).resolve()
    child = subprocess.Popen([sys.executable, "-I", str(script.with_name("capped.py")),
                              "--cpu-seconds", "0.2", "--", sys.executable, "-I", str(script), "worker"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        requests = {"idle": ["idle"], "reset": ["work", "work"], "busy": ["busy"]}[case]
        data = b"".join(struct.pack("<I", len(x)) + x.encode() for x in requests)
        started = time.monotonic()
        out, err = child.communicate(data, timeout=120)
        elapsed = time.monotonic() - started
        replies = [json.loads(line[6:]) for line in out.splitlines() if line.startswith(b"REPLY\t")]
        if case == "busy":
            assert child.returncode == 124, (child.returncode, out, err)
            assert not replies and b"legacy CPU budget exceeded" in err, (out, err)
        else:
            assert child.returncode == 0, (child.returncode, out, err)
            assert [r["answer"]["Ok"] for r in replies] == requests, replies
            assert all(0 <= r["cpu_seconds"] < 0.2 for r in replies), replies
            if case == "idle":
                assert elapsed >= 0.6, elapsed
            else:
                assert sum(r["cpu_seconds"] for r in replies) >= 0.2, replies
        print(f"{case}: elapsed {elapsed:.3f} s; replies {replies}; stderr {err.decode().strip()}")
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker()
    else:
        check(sys.argv[1])
