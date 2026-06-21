"""
Hardware auto-detection and auto-configuration for VAMToolbox.

`detect_system()` probes CPU / RAM / GPU.  `recommend_config()` turns those facts
into sensible runtime settings (GPU vs CPU backend, VRAM chunk budget, rebin worker
count, RAM budget for slab sizing).  `autoconfigure()` does both and (optionally)
applies the settings to the live library, so a run is tuned to the machine it lands
on without hand-editing anything.

    import vamtoolbox
    info, rec = vamtoolbox.util.hardware.autoconfigure()   # detect + apply + print
"""
import os
import subprocess


# ───────────────────────────── detection ──────────────────────────────────────
def _ram_gb():
    """(total, available) RAM in GB.  psutil -> Win32 GlobalMemoryStatusEx -> (nan, nan)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 1), round(vm.available / 1e9, 1)
    except Exception:
        pass
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MS(); m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.ullTotalPhys / 1e9, 1), round(m.ullAvailPhys / 1e9, 1)
    except Exception:
        return float("nan"), float("nan")


def _detect_gpus():
    """List of {name, vram_total_gb, vram_free_gb} via nvidia-smi (empty if none)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            gpus = []
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    name, tot, free = parts
                    gpus.append({"name": name,
                                 "vram_total_gb": round(float(tot) / 1024, 1),
                                 "vram_free_gb": round(float(free) / 1024, 1)})
            return gpus
    except Exception:
        pass
    return []


def _astra_cuda_ok():
    """True if astra is importable and a usable CUDA device is present."""
    try:
        import astra  # noqa: F401
    except Exception:
        return False
    # astra exposes use_cuda() on most builds; fall back to a GPU presence check.
    try:
        return bool(astra.use_cuda())
    except Exception:
        return len(_detect_gpus()) > 0


def _metal_ok():
    """True if an Apple Metal device + metalcompute are usable (Apple Silicon)."""
    try:
        from vamtoolbox.projector.metalbackend import metal_available
        return metal_available()
    except Exception:
        return False


def detect_system():
    """Probe the machine.  Returns a dict of hardware facts (never raises)."""
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        pass
    total_gb, avail_gb = _ram_gb()
    gpus = _detect_gpus()
    return {
        "cpu_logical": logical,
        "cpu_physical": physical,
        "ram_total_gb": total_gb,
        "ram_avail_gb": avail_gb,
        "gpus": gpus,
        "cuda": _astra_cuda_ok(),
        "metal": _metal_ok(),
    }


# ──────────────────────────── recommendation ──────────────────────────────────
def recommend_config(info=None):
    """Map hardware facts to runtime settings."""
    if info is None:
        info = detect_system()
    cores = info["cpu_logical"]
    use_cuda = bool(info["cuda"] and info["gpus"])
    # Apple Metal is the preferred CPU-alternative when CUDA is absent
    # (e.g. Apple Silicon); the projectorconstructor selects it automatically.
    use_metal = bool(info.get("metal") and not use_cuda)

    rec = {
        "use_cuda": use_cuda,
        "use_metal": use_metal,
        # CPU projector: sparse matrix path is the fast CPU option (lever 6).
        "cpu_backend": "sparse",
        # Rebin workers: pin all cores on small boxes; leave ~2 free on big ones so
        # the machine stays usable during the ~tens-of-seconds rebin spike.
        "rebin_jobs": -1 if cores <= 8 else max(1, cores - 2),
        # RAM budget the slab sizer should plan against (use available, not total).
        "ram_budget_gb": info["ram_avail_gb"],
        "vram_budget_bytes": None,
    }
    if use_cuda:
        free = info["gpus"][0]["vram_free_gb"]
        # Chunked projector working set: ~40% of free VRAM, capped to a sane window.
        rec["vram_budget_bytes"] = int(min(max(free * 0.4, 1.0), 8.0) * 1e9)
    return rec


def autoconfigure(apply=True, verbose=True):
    """Detect hardware, compute recommended settings, optionally apply them live.

    Applies: the chunked-CUDA projector VRAM budget and the rebin worker count.
    Returns (info, rec) so the caller can also use `use_cuda` / `ram_budget_gb`
    (e.g. to drive the z-slab decision).
    """
    import vamtoolbox

    info = detect_system()
    rec = recommend_config(info)

    if apply:
        vamtoolbox.geometry.REBIN_N_JOBS = rec["rebin_jobs"]
        if rec["vram_budget_bytes"] is not None:
            try:
                vamtoolbox.projector.Projector3DParallelCUDA.\
                    Projector3DParallelCUDAAstraChunked.VRAM_BUDGET_BYTES = rec["vram_budget_bytes"]
            except Exception:
                pass

    if verbose:
        print(summary(info, rec))
    return info, rec


def summary(info=None, rec=None):
    """Human-readable hardware + recommendation report."""
    if info is None:
        info = detect_system()
    if rec is None:
        rec = recommend_config(info)
    lines = ["=" * 66, "  VAMToolbox hardware auto-configuration", "=" * 66]
    lines.append(f"  CPU         : {info['cpu_physical']} cores "
                 f"({info['cpu_logical']} logical)")
    lines.append(f"  RAM         : {info['ram_avail_gb']} GB free / "
                 f"{info['ram_total_gb']} GB total")
    if info["gpus"]:
        for i, g in enumerate(info["gpus"]):
            lines.append(f"  GPU {i}       : {g['name']}  "
                         f"({g['vram_free_gb']} GB free / {g['vram_total_gb']} GB)")
    elif info.get("metal"):
        lines.append("  GPU         : Apple Metal device (metalcompute)")
    else:
        lines.append("  GPU         : none detected")
    lines.append("-" * 66)
    if rec["use_cuda"]:
        backend = "CUDA (astra GPU)"
    elif rec.get("use_metal"):
        backend = "Apple Metal (GPU parallel-beam)"
    else:
        backend = "CPU sparse-matrix"
    lines.append(f"  -> backend     : {backend}")
    if rec["vram_budget_bytes"]:
        lines.append(f"  -> VRAM budget : {rec['vram_budget_bytes'] / 1e9:.1f} GB / chunk")
    lines.append(f"  -> rebin jobs  : {rec['rebin_jobs']}  "
                 f"({'all cores' if rec['rebin_jobs'] == -1 else 'capped'})")
    lines.append(f"  -> RAM budget  : {rec['ram_budget_gb']} GB (for slab sizing)")
    lines.append("=" * 66)
    return "\n".join(lines)


if __name__ == "__main__":
    autoconfigure(apply=False)
