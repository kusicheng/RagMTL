                                                                                

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import sys
import time

MAX_DISK_BYTES = 2_000_000_000_000
MAX_RAM_BYTES = 32_000_000_000


class BudgetExceeded(RuntimeError):
    pass


def tree_bytes(root: Path) -> int:
    pass                                                                           

                                                                           
                                                                               
                                                                                
       
    root = Path(root)
    total = 0
    if not root.exists():
        return 0
    if root.is_symlink():
        raise ValueError("Budget root cannot be a symlink")
    resolved_root = root.resolve()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            scanner = os.scandir(directory)
        except FileNotFoundError:                                          
            continue
        with scanner as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                                                                          
                                                                            
                    is_link = entry.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)
                    if is_link:
                        target = Path(entry.path).resolve()
                        if not target.is_relative_to(resolved_root):
                            raise ValueError(f"Link escapes the budget root: {entry.path}")
                        if target.is_file():
                            total += target.stat().st_size
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    else:
                        total += info.st_size
                except FileNotFoundError:                               
                    pass
    return total


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        handle = get_process()
        if not get_memory(handle, ctypes.byref(c), c.cb):
            raise OSError("Cannot measure process peak working set")
        return int(c.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class ResourceBudget:
    def __init__(self, root: str | Path, *, disk_bytes: int = MAX_DISK_BYTES,
                 ram_bytes: int = MAX_RAM_BYTES, reserve_bytes: int = 64_000_000):
        self.root = Path(root).resolve()
        if not 0 < disk_bytes <= MAX_DISK_BYTES:
            raise ValueError("Disk budget must be in (0, 2 TB decimal]")
        if not 0 < ram_bytes <= MAX_RAM_BYTES:
            raise ValueError("RAM budget must be in (0, 32 GB decimal]")
        if not isinstance(reserve_bytes, int) or not 0 <= reserve_bytes < disk_bytes:
            raise ValueError("Reserve bytes must be nonnegative and below disk limit")
        self.disk_limit = disk_bytes
        self.ram_limit = ram_bytes
        self.reserve = reserve_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self.peak_disk_bytes = 0
        self.started = time.perf_counter()

    def contains(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"Data must be inside the accounted root {self.root}")
        return p

    def check(self, additional_bytes: int = 0) -> dict:
        if additional_bytes < 0:
            raise ValueError("Additional bytes must be nonnegative")
        used = tree_bytes(self.root)
        self.peak_disk_bytes = max(used, self.peak_disk_bytes)
        rss = peak_rss_bytes()
        if used + additional_bytes + self.reserve > self.disk_limit:
            raise BudgetExceeded("Physical disk budget would be exceeded (includes reserve)")
        if shutil.disk_usage(self.root).free < additional_bytes + self.reserve:
            raise BudgetExceeded("Insufficient free disk space")
        if rss >= self.ram_limit:
            raise BudgetExceeded("Process peak RAM exceeds configured limit")
        return {"physical_bytes": used, "peak_observed_physical_bytes": self.peak_disk_bytes,
                "process_peak_rss_bytes": rss, "disk_limit_bytes": self.disk_limit,
                "ram_limit_bytes": self.ram_limit,
                "elapsed_seconds": time.perf_counter() - self.started,
                "scope": "single process and accounted data root; host total must be monitored separately"}
