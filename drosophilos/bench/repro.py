"""Reproducibility metadata for performance benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..sim.model import Params


def circuit_hash(pl) -> str:
    """Hash a pipeline's canonical topology and topology-affecting initial constants."""
    edges = sorted((int(s), int(d), int(q), int(delay))
                   for s, d, q, delay in zip(pl.net.src, pl.net.dst, pl.net.quanta, pl.net.delay))
    payload = {
        "neurons": int(pl.net.n),
        "edges": edges,
        # Biases are per-neuron initial constants.  Preserve neuron order (rather than
        # sorting values) because assigning a bias to a different role changes the image.
        # Canonicalize sub-nanovolt representation noise without preserving -0.0.
        "bias": [0.0 if (value := round(float(bias), 9)) == 0.0 else value
                 for bias in getattr(pl.net, "bias", ())],
        # Constant values select the image rails lit at load time.  The edge table is
        # unchanged when a constant changes, but the built circuit image is not.
        "constants": sorted((str(k), int(v)) for k, v in getattr(pl, "const_values", {}).items()),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _cpu_brand() -> str:
    brand = platform.processor()
    if brand:
        return brand
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def record(*, source_path: str, pl, inputs: list[int], initial_state: dict | None,
           pacing: str, mul: str, copies: int, width: int, height: int,
           frames_requested: int, params: Params, backend: str, device: str,
           dtype: str, simulator: str, gpu_name: str | None = None,
           perturbation=None, seed: int | None = None,
           cmdline: list[str] | None = None) -> dict:
    """Return the stable environment and workload portion of a benchmark record."""
    import torch

    path = Path(source_path)
    source = path.read_bytes()
    dirty = _git(["status", "--porcelain"])
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": None if dirty is None else bool(dirty),
        "source_path": str(path),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "circuit_sha256": None if pl is None else circuit_hash(pl),
        "inputs": list(inputs),
        "initial_state": initial_state,
        "pacing": pacing,
        "mul": mul,
        "copies": int(copies),
        "width": int(width),
        "height": int(height),
        "frames_requested": int(frames_requested),
        "neurons_per_node": None if pl is None else int(pl.net.n),
        "edges_per_node": None if pl is None else int(pl.net.nnz),
        "model_params": dataclasses.asdict(params),
        "perturbation": perturbation,
        "seed": seed,
        "backend": backend,
        "simulator": simulator,
        "device": device,
        "dtype": dtype,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_brand": _cpu_brand(),
            "gpu_name": gpu_name,
            "memory_bytes": _memory_bytes(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "cmdline": list(sys.argv if cmdline is None else cmdline),
    }
