"""Run the legacy oracle with a 512 MiB address/commit budget. No shader is read until the cap is in place."""
import os
import subprocess
import sys

LIMIT = 512 * 1024 * 1024

if os.name != "nt":
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    os.execv(sys.argv[1], sys.argv[1:])

import ctypes
from ctypes import wintypes as w

k = ctypes.WinDLL("kernel32", use_last_error=True)

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

k.CreateJobObjectW.restype = w.HANDLE
k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
k.GetCurrentProcess.restype = w.HANDLE
k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]

def checked(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

job = k.CreateJobObjectW(None, None)
checked(job)
info = Extended()
# JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
info.basic.flags = 0x200 | 0x2000
info.job_limit = LIMIT
checked(k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)))
# Assign this supervisor before spawning the worker: the child inherits membership, without an allocation race.
checked(k.AssignProcessToJobObject(job, k.GetCurrentProcess()))
status = subprocess.call(sys.argv[1:])
checked(k.QueryInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info), None))
print(f"legacy peak commit: {info.peak_job} bytes", file=sys.stderr)
sys.exit(status)
