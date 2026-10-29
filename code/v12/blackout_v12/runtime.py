"""Portable process budgets and a cross-process CUDA learner lease."""
from contextlib import contextmanager, nullcontext
import ctypes
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from .io import ROOT

_PROCESS_JOB = None
_MANAGED_PROCESSES = []


def child_environment():
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED="0", PYTHONUTF8="1")
    paths = [str(ROOT / "code/v12"), str(ROOT / "code/shared")]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + env.get("PYTHONPATH", "").split(os.pathsep)))
    return env


def process_options(detached=False):
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW
        if detached:
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        return dict(creationflags=flags)
    return dict(start_new_session=detached)


@contextmanager
def process_tree_guard():
    """Own only this supervisor's jobs; Windows closes their descendants on exit."""
    global _PROCESS_JOB
    handle = None
    if os.name == 'nt':
        from ctypes import wintypes
        class BasicLimits(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t),
                        ('maximum', ctypes.c_size_t), ('active', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]
        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('basic', BasicLimits), ('io', Counters),
                        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                        ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            if handle:
                kernel.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        _PROCESS_JOB = handle
    try:
        yield
    finally:
        _PROCESS_JOB = None
        if handle:
            kernel.CloseHandle(handle)
        else:
            for process in _MANAGED_PROCESSES:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        _MANAGED_PROCESSES.clear()


def managed_popen(*args, **kwargs):
    """Attach explicit experiment jobs before they can outlive the supervisor."""
    guarded = os.name == 'nt' and _PROCESS_JOB
    if guarded:
        # Python's Windows venv redirector can create a child immediately.
        # Suspend before assignment so that child also inherits the Job object.
        kwargs['creationflags'] = kwargs.get('creationflags', 0) | 0x4
    elif os.name != 'nt':
        kwargs['start_new_session'] = True
    process = subprocess.Popen(*args, **kwargs)
    if guarded:
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        try:
            if not kernel.AssignProcessToJobObject(_PROCESS_JOB, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            _resume_process_threads(process.pid)
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
    _MANAGED_PROCESSES.append(process)
    return process


def _resume_process_threads(pid):
    """Resume our suspended child using documented Toolhelp/Thread APIs."""
    from ctypes import wintypes
    class ThreadEntry(ctypes.Structure):
        _fields_ = [('size',wintypes.DWORD),('usage',wintypes.DWORD),
                    ('thread',wintypes.DWORD),('owner',wintypes.DWORD),
                    ('priority',wintypes.LONG),('delta',wintypes.LONG),('flags',wintypes.DWORD)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD,wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Thread32First.argtypes = kernel.Thread32Next.argtypes = [wintypes.HANDLE,ctypes.POINTER(ThreadEntry)]
    kernel.OpenThread.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(0x4,0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = False
    try:
        entry = ThreadEntry(); entry.size = ctypes.sizeof(entry)
        more = kernel.Thread32First(snapshot,ctypes.byref(entry))
        while more:
            if entry.owner == pid:
                thread = kernel.OpenThread(0x2,False,entry.thread)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if kernel.ResumeThread(thread) == 0xffffffff:
                        raise ctypes.WinError(ctypes.get_last_error())
                    resumed = True
                finally:
                    kernel.CloseHandle(thread)
            more = kernel.Thread32Next(snapshot,ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    if not resumed:
        raise RuntimeError('Cannot locate suspended experiment process thread')


def hardware_identity():
    import torch
    physical = None
    memory = None
    processor = platform.processor()
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in ("total", "available", "page", "page_available", "virtual", "virtual_available", "extended")]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = status.total
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores | ConvertTo-Json -Compress"], capture_output=True, text=True, timeout=10, **process_options())
            rows = json.loads(result.stdout)
            rows = rows if isinstance(rows, list) else [rows]
            physical = sum(row["NumberOfCores"] for row in rows)
            processor = "; ".join(row["Name"].strip() for row in rows)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            pass
        if sys.platform == "darwin":
            try:
                physical = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True))
            except (OSError, ValueError):
                pass
        elif Path("/proc/cpuinfo").exists():
            pairs = set()
            for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
                items = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                items = {k.strip(): v.strip() for k, v in items.items()}
                if "core id" in items:
                    pairs.add((items.get("physical id", "0"), items["core id"]))
            physical = len(pairs) or None
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            gpus.append(dict(index=index, name=device.name, total_memory=device.total_memory))
    driver_gpu = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, **process_options())
        if result.returncode == 0:
            driver_gpu = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return dict(platform=platform.platform(), processor=processor, physical_cores=physical,
                logical_cpus=os.cpu_count() or 1, memory_bytes=memory, python=sys.version.split()[0],
                torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpus=gpus, driver_gpu=driver_gpu,
                mps_available=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))


class FileLease:
    """OS file locks release automatically after process death; no stale PID lock."""
    def __init__(self, path, blocking=True):
        self.path, self.blocking, self.stream = Path(path), blocking, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        while True:
            try:
                self.stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if not self.blocking:
                    self.stream.close()
                    raise RuntimeError("Resource is already in use: " + str(self.path))
                time.sleep(.1)

    def __exit__(self, *_):
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def learner_lease(device):
    return FileLease(ROOT / "logs/v12/.cuda-learner.lock") if str(device).startswith("cuda") else nullcontext()


def budget(config, hardware):
    cores = (hardware['logical_cpus'] if config.get('cpu_budget_basis','logical') == 'logical'
             else hardware['physical_cores'] or hardware['logical_cpus'])
    cpu_limit = max(1, cores - int(config.get("cpu_reserve", 1)))
    memory_gib = (hardware.get("memory_bytes") or 16 * 1024**3) / 1024**3
    memory_limit = max(1, int((memory_gib - config.get("memory_reserve_gib", 6)) / config.get("worker_memory_gib", 2)))
    cap = min(cpu_limit, memory_limit, config.get("worker_cap", 8))
    requested = config.get("workers", "auto")
    workers = cap if requested == "auto" else min(int(requested), cap)
    parallel = min(max(1, int(config.get("parallel_runs", 2))), workers)
    return dict(worker_cap=cap, selected_workers=max(1, workers), parallel_runs=parallel,
                workers_per_run=max(1, workers // parallel), actor_threads=1,
                method="logical-CPU/RAM ceiling; measured throughput determines actual concurrency")


def configure_worker():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
