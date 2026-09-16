from __future__ import annotations

import json
import re
from typing import Any
import sys

from env import Env, ModuleNotAvailable, getattr_path, read_text, unknown, major_minor

# The NVIDIA-built PyTorch wheels for Jetson carry a local version segment —
# the part after "+" — that names the NVIDIA container release. A wheel from
# plain PyPI has no such segment. This is a hint, not a proof, which is why the
# probe reports the tag itself alongside the interpretation.
_NV_LOCAL_TAG = re.compile(r"(?:^|\.)nv\d", re.IGNORECASE)

# `# R36 (release), REVISION: 5.0, GCID: ...`
_L4T_RELEASE = re.compile(r"R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(r"REVISION:\s*([\d.]+)")

# hepler function
def _split_local_version(raw: str) -> dict[str, Any]:
    if not raw:
        return {"raw": raw, "public": None, "local": None, "nvidia_build": False}
    public, sep, local = raw.partition("+")
    local = local if sep else None
    return {
        "raw": raw,
        "public": public or None,
        "local": local,
        "nvidia_build": bool(local and _NV_LOCAL_TAG.search(local)),
    }

# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

def probe_torch(env: Env) -> dict[str, Any]:
    try:
        torch = env.importer("torch")
    except ModuleNotAvailable as e:
        return unknown("import torch", f"torch not installed: {e}")

    version_raw = getattr_path(torch, "__version__", None)
    if version_raw is None:
        return unknown("import torch", "torch.__version__ missing")

    split_result = _split_local_version(str(version_raw))
    cuda_available = getattr_path(torch, "cuda.is_available", lambda: False)()
    cuda_version = None
    device_name = None
    if cuda_available:
        cuda_version_raw = getattr_path(torch.cuda, "version", None)
        if cuda_version_raw is not None:
            cuda_version = str(cuda_version_raw)
        else:
            try:
                cuda_version = torch.version.cuda
            except AttributeError:
                cuda_version = None
        device_name = getattr_path(torch.cuda, "get_device_name", lambda x: None)(0)

    diagnosis = "torch is installed and sees the GPU" if cuda_available else "torch installed but GPU not visible"
    if split_result.get("nvidia_build") is False and "+" not in str(version_raw):
        diagnosis += " — possible stock PyPI wheel (no NVIDIA local version tag)"

    return {
        "value": str(version_raw),
        "source": "import torch",
        "status": "ok",
        "version": split_result,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "diagnosis": diagnosis,
    }


def probe_cuda(env: Env) -> dict[str, Any]:
    src = "/usr/local/cuda/version.json"
    raw = read_text(env.root, src)
    if raw is None:
        return unknown(src, "version.json missing or unreadable")
    try:
        data = json.loads(raw)
    except Exception as e:
        return unknown(src, f"invalid JSON: {e}")
    version = data.get("cuda", {}).get("version") if isinstance(data.get("cuda"), dict) else data.get("version")
    if version is None:
        return unknown(src, "version key missing")
    line = major_minor(str(version))
    return {
        "value": str(version),
        "source": src,
        "status": "ok",
        "line": line,
    }


def probe_opencv(env: Env) -> dict[str, Any]:
    try:
        cv2 = env.importer("cv2")
    except ModuleNotAvailable as e:
        return unknown("import cv2", f"cv2 not installed: {e}")

    version = getattr_path(cv2, "__version__", None)
    cuda_enabled = False
    cuda_devices = 0
    try:
        cuda_devices = getattr_path(cv2, "cuda.getCudaEnabledDeviceCount", lambda: 0)()
        cuda_enabled = cuda_devices > 0
    except Exception:
        pass

    detail = "the cv2.cuda namespace exists but reports no devices — this is a non-CUDA build" if cuda_devices == 0 else f"CUDA enabled with {cuda_devices} device(s)"

    return {
        "value": str(version) if version else None,
        "source": "import cv2",
        "status": "ok",
        "cuda_devices": cuda_devices,
        "cuda_enabled": cuda_enabled,
        "detail": detail,
    }


def probe_tensorrt(env: Env) -> dict[str, Any]:
    try:
        tensorrt = env.importer("tensorrt")
    except ModuleNotAvailable as e:
        # Check if we're in a venv without system-site-packages
        python = env.python
        in_venv = python.prefix != python.base_prefix
        detail = "isolated virtual environment missing --system-site-packages" if in_venv else f"tensorrt not installed: {e}"
        return unknown("import tensorrt", detail)

    version = getattr_path(tensorrt, "__version__", None)
    line = major_minor(str(version)) if version else None
    return {
        "value": str(version) if version else None,
        "source": "import tensorrt",
        "status": "ok",
        "line": line,
    }


def probe_l4t(env: Env) -> dict[str, Any]:
    src = "/etc/nv_tegra_release"
    raw = read_text(env.root, src)
    if raw is None:
        return unknown(src, "file missing or unreadable")
    release_match = _L4T_RELEASE.search(raw)
    revision_match = _L4T_REVISION.search(raw)
    if not release_match or not revision_match:
        return unknown(src, "expected R... (release) and REVISION: patterns not found")
    value = f"{release_match.group(1)}.{revision_match.group(1)}"
    line = f"{release_match.group(1)}.{revision_match.group(1).split('.')[0]}"
    return {
        "value": value,
        "source": src,
        "status": "ok",
        "line": line,
        "raw": raw,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Env.real()
    # out = probe_l4t(env)
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Env.real()

    # testing probes
    report = {
        "probe_torch": probe_torch(env),
        "probe_cuda": probe_cuda(env),
        "probe_opencv": probe_opencv(env),
        "probe_tensorrt": probe_tensorrt(env),
        "probe_l4t": probe_l4t(env),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

