"""Read-only macOS on-screen window audit, independent of Unity internals."""
import ctypes
import os
import subprocess
import sys


def group_pids(group=None):
    group = os.getpgrp() if group is None else group
    result = subprocess.run(["ps", "-axo", "pid=,pgid="], text=True, capture_output=True, check=True)
    return {int(line.split()[0]) for line in result.stdout.splitlines() if len(line.split()) == 2 and int(line.split()[1]) == group}


def visible_windows():
    if sys.platform != "darwin":
        return {"verified": False, "reason": "macOS window audit unavailable"}
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    ptr = ctypes.c_void_p
    cg.CGWindowListCopyWindowInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    cg.CGWindowListCopyWindowInfo.restype = ptr
    cf.CFArrayGetCount.argtypes = [ptr]; cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetValueAtIndex.argtypes = [ptr, ctypes.c_long]; cf.CFArrayGetValueAtIndex.restype = ptr
    cf.CFDictionaryGetValue.argtypes = [ptr, ptr]; cf.CFDictionaryGetValue.restype = ptr
    cf.CFStringCreateWithCString.argtypes = [ptr, ctypes.c_char_p, ctypes.c_uint32]; cf.CFStringCreateWithCString.restype = ptr
    cf.CFNumberGetValue.argtypes = [ptr, ctypes.c_int, ptr]; cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ptr]
    keys = [cf.CFStringCreateWithCString(None, k.encode(), 0x08000100) for k in
            ("kCGWindowOwnerPID", "kCGWindowLayer", "kCGWindowNumber")]
    windows = cg.CGWindowListCopyWindowInfo(1 | 16, 0)
    if not windows:
        for key in keys: cf.CFRelease(key)
        return {"verified": False, "reason": "CGWindowList unavailable"}
    owned = group_pids()
    found = []
    try:
        for i in range(cf.CFArrayGetCount(windows)):
            row = cf.CFArrayGetValueAtIndex(windows, i)
            values = []
            for key in keys:
                value = ctypes.c_int(0)
                number = cf.CFDictionaryGetValue(row, key)
                if number: cf.CFNumberGetValue(number, 9, ctypes.byref(value))
                values.append(value.value)
            if values[0] in owned and values[1] == 0:
                found.append({"pid": values[0], "window_id": values[2]})
        return {"verified": True, "onscreen_owned_windows": found}
    finally:
        cf.CFRelease(windows)
        for key in keys: cf.CFRelease(key)
