"""Program image manifest: what was run, on which graph, with which interventions.

Four graphs are reported separately (plan §Machine definition): original (anatomical,
under the loader's rules), retained (after silencing whole cells), active-nonzero (after
zeroed weights), and the silencing list. Structural edits must be empty for Profile 2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..sim.model import Params


@dataclass
class ParameterEdit:
    kind: str  # "weight" | "threshold" | "bias" | "gain"
    pre: int | None  # bodyId for weights
    post: int | None  # bodyId
    anatomical: float | None  # signed anatomical quanta (weights) or the default value
    new: float  # new quanta (weights) or new value
    bound: str  # the rule that permitted it, e.g. "0 <= q <= 4 * count * 16"
    reason: str = ""


@dataclass
class Manifest:
    connectome: dict
    profile: int
    params: dict
    circuit_bodies: list[int]
    graphs: dict  # original / retained / active_nonzero: summaries
    silencing: list[dict] = field(default_factory=list)
    parameter_edits: list[ParameterEdit] = field(default_factory=list)
    structural_edits: list[dict] = field(default_factory=list)
    ports: dict = field(default_factory=lambda: {"inputs": [], "outputs": []})
    execution_mode: str = "isolated"
    isolation_classification: str = "not yet assessed"
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.profile == 2 and self.structural_edits:
            raise ValueError("Profile 2 forbids structural edits (added / rerouted edges)")
        if self.profile == 1 and (self.parameter_edits or self.silencing or self.structural_edits):
            raise ValueError("Profile 1 forbids every intervention beyond declared inputs")
        for e in self.parameter_edits:
            if e.kind == "weight" and e.anatomical is not None and e.new * e.anatomical < 0:
                raise ValueError(f"sign flip on {e.pre}->{e.post} is not a parameter edit")

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return json.dumps(d, indent=1, default=_default)

    def save(self, path: Path) -> None:
        self.validate()
        Path(path).write_text(self.to_json())


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def params_dict(params: Params) -> dict:
    d = dataclasses.asdict(params)
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d
