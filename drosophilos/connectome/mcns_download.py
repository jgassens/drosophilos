"""Bulk download of the MCNS v1.0 flat-connectome tables (Janelia, CC-BY 4.0, no login).

Only the three tables the loader needs. Verified 2026-09-13 at
https://male-cns.janelia.org/download/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",  # 13 MB
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",  # 42 MB
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",  # 1.1 GB
}
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "mcns"


def path_of(key: str, data_dir: Path = DEFAULT_DIR) -> Path:
    return data_dir / FILES[key]


def download(keys=("annotations", "neurotransmitters", "weights"), data_dir: Path = DEFAULT_DIR) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for key in keys:
        dest = path_of(key, data_dir)
        if dest.exists() and dest.stat().st_size > 0:
            out.append(dest)
            continue
        url = BASE + FILES[key]
        print(f"downloading {url} -> {dest}", file=sys.stderr)
        subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dest), url], check=True)
        out.append(dest)
    return out


if __name__ == "__main__":
    keys = sys.argv[1:] or ("annotations", "neurotransmitters", "weights")
    for p in download(keys):
        print(p)
