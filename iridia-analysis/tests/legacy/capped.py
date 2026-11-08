"""Persistent legacy worker, bounded by 512 MiB and per-request worker CPU time."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import queue
import signal
import struct
import subprocess
import sys
import threading

LIMIT = 512 * 1024 * 1024
SUPERVISOR_PID = os.getpid()


def windows_memory_cap():
    from ctypes import wintypes as w

    class Io(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("a", "b", "c", "d", "e", "f")]

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("processes", w.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", w.DWORD), ("scheduling", w.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", Io), ("process_limit", ctypes.c_size_t),
                    ("job_limit", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                    ("peak_job", ctypes.c_size_t)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = w.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.GetCurrentProcess.restype = w.HANDLE
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.CloseHandle.argtypes = [w.HANDLE]
    job = k.CreateJobObjectW(None, None)
    info = Extended()
    # JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    info.basic.flags = 0x200 | 0x2000
    info.job_limit = LIMIT
    if not job or not k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    # Keep the handle open until process exit: closing it kills this supervisor and its worker.
    return job


def posix_child():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    if sys.platform.startswith("linux"):
        # Rust may kill the supervisor at its hard protocol deadline. Its worker must die too.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
        if os.getppid() != SUPERVISOR_PID:
            os.kill(os.getpid(), signal.SIGKILL)


def worker_clock(child):
    if os.name == "nt":
        from ctypes import wintypes as w
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        k.GetProcessTimes.restype = w.BOOL

        def seconds():
            created, exited, kernel, user = (w.FILETIME() for _ in range(4))
            if not k.GetProcessTimes(int(child._handle), ctypes.byref(created), ctypes.byref(exited),
                                     ctypes.byref(kernel), ctypes.byref(user)):
                raise ctypes.WinError(ctypes.get_last_error())
            return ((kernel.dwHighDateTime << 32) + kernel.dwLowDateTime
                    + (user.dwHighDateTime << 32) + user.dwLowDateTime) / 10_000_000
        return seconds
    if sys.platform.startswith("linux"):
        ticks = os.sysconf("SC_CLK_TCK")
        stat = Path(f"/proc/{child.pid}/stat")

        def seconds():
            # comm can contain spaces and parentheses. Fields after its last ')' start at state (3).
            fields = stat.read_text().rsplit(") ", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / ticks
        return seconds

    def seconds():
        # Other POSIX hosts expose cumulative worker CPU time as [[days-]hh:]mm:ss.
        value = subprocess.check_output(["ps", "-p", str(child.pid), "-o", "time="], text=True).strip()
        days, _, value = value.rpartition("-")
        total = 0
        for part in value.split(":"):
            total = total * 60 + float(part)
        return total + (int(days) * 86400 if days else 0)
    return seconds


def serve(command, cpu_seconds):
    job = windows_memory_cap() if os.name == "nt" else None
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             preexec_fn=posix_child if os.name != "nt" else None)
    clock = worker_clock(child)
    replies = queue.Queue()

    def read_replies():
        try:
            for line in child.stdout:
                if line.startswith(b"REPLY\t"):
                    replies.put(line[6:])
        finally:
            replies.put(None)

    reader = threading.Thread(target=read_replies, daemon=True)
    reader.start()
    try:
        while True:
            length = sys.stdin.buffer.read(4)
            if not length:
                return 0
            if len(length) != 4:
                raise RuntimeError("truncated legacy request length")
            size = struct.unpack("<I", length)[0]
            data = sys.stdin.buffer.read(size)
            if len(data) != size:
                raise RuntimeError("truncated legacy request")
            started = clock()
            child.stdin.write(length + data)
            child.stdin.flush()
            while True:
                try:
                    reply = replies.get(timeout=0.01)
                except queue.Empty:
                    used = clock() - started
                    if used >= cpu_seconds:
                        print(f"legacy CPU budget exceeded ({cpu_seconds:g} s; worker {used:.6f} s)",
                              file=sys.stderr, flush=True)
                        return 124
                    continue
                if reply is None:
                    return child.wait() or 1
                used = clock() - started
                if used >= cpu_seconds:
                    print(f"legacy CPU budget exceeded ({cpu_seconds:g} s; worker {used:.6f} s)",
                          file=sys.stderr, flush=True)
                    return 124
                result = {"cpu_seconds": used, "answer": json.loads(reply)}
                print("REPLY\t" + json.dumps(result, separators=(",", ":")), flush=True)
                break
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        reader.join()
        # Keep the job handle live until the worker has ended.
        assert job is None or job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-seconds", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.cpu_seconds <= 0:
        parser.error("a command and positive CPU budget are required")
    return serve(command, args.cpu_seconds)


if __name__ == "__main__":
    sys.exit(main())
