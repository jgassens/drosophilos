"""MCNS v1.0 loader with explicit neuron-selection and edge-filter rules.

Rules (the manifest records them verbatim):
  NEURON_RULE: a body is a neuron iff its `superclass` annotation is non-null
               (166,700 bodies in v1.0; `status == "Traced"` would give 165,122).
  EDGE_RULE:   an edge (pre -> post) is kept iff both bodies are neurons and its
               synapse count in the min-confidence-0.5 weights table is >= min_syn.
  SIGN_RULE:   sign is taken from the presynaptic neuron's `consensus_nt`:
               gaba, glutamate, histamine -> -1 ; everything else, including
               "unclear", -> +1 (Shiu et al. 2024 convention, extended to histamine,
               which gates chloride channels at fly photoreceptor synapses).
  DELAY_RULE:  every edge uses the default delay (no per-synapse delay data).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from ..sim.model import QUANTA_PER_SYNAPSE, Params, Topology
from .mcns_download import DEFAULT_DIR, FILES, path_of

NEURON_RULE = "superclass is not null"
EDGE_RULE = "both endpoints are neurons and syn_count >= min_syn (weights table, minconf 0.5)"
SIGN_RULE = "presynaptic consensus_nt in {gaba, glutamate, histamine} -> -1, else +1"
DELAY_RULE = "all edges: default_delay_ms"
INHIBITORY = {"gaba", "glutamate", "histamine"}


@dataclass
class MCNS:
    neurons: pd.DataFrame  # one row per neuron, positional index = neuron id in the simulator
    pre: np.ndarray  # int32 neuron indices
    post: np.ndarray
    count: np.ndarray  # int32 synapse counts (>= min_syn)
    min_syn: int
    version: str = "MCNS v1.0"

    # ---- lookups -------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.neurons)

    def index_of(self, body_ids) -> np.ndarray:
        bodies = self.neurons["bodyId"].to_numpy()
        pos = np.searchsorted(bodies, body_ids)
        pos = np.clip(pos, 0, len(bodies) - 1)
        if not np.all(bodies[pos] == body_ids):
            raise KeyError("some bodyIds are not neurons under NEURON_RULE")
        return pos

    def bodies_of(self, idx) -> np.ndarray:
        return self.neurons["bodyId"].to_numpy()[np.asarray(idx)]

    def select(self, **conds) -> np.ndarray:
        """Indices of neurons whose annotation columns equal the given values."""
        m = np.ones(self.n, dtype=bool)
        for col, val in conds.items():
            col = col.rstrip("_")
            m &= (self.neurons[col] == val).to_numpy()
        return np.nonzero(m)[0]

    @property
    def sign(self) -> np.ndarray:
        return self.neurons["sign"].to_numpy()

    # ---- graphs --------------------------------------------------------------------
    def topology(self, params: Params = Params(), min_syn: int | None = None) -> Topology:
        """Shared anatomical topology: quanta = sign(pre) * count * 16, default delay."""
        keep = slice(None) if min_syn is None else self.count >= min_syn
        pre, post, cnt = self.pre[keep], self.post[keep], self.count[keep]
        quanta = (self.sign[pre] * cnt * QUANTA_PER_SYNAPSE).astype(np.int32)
        delay = np.full(len(pre), params.default_delay_steps, dtype=np.int32)
        return Topology.from_edges(self.n, pre, post, quanta, delay)

    def graph_summary(self, min_syn: int | None = None) -> dict:
        keep = slice(None) if min_syn is None else self.count >= min_syn
        pre, post, cnt = self.pre[keep], self.post[keep], self.count[keep]
        h = hashlib.sha256()
        for arr in (pre, post, cnt):
            h.update(np.ascontiguousarray(arr).tobytes())
        return {
            "n_neurons": int(self.n),
            "n_edges": int(len(pre)),
            "total_synapses": int(cnt.sum()),
            "min_syn": int(self.min_syn if min_syn is None else min_syn),
            "sha256": h.hexdigest(),
        }

    def summary(self) -> dict:
        nt = self.neurons["nt"].value_counts(dropna=False).to_dict()
        return {
            "version": self.version,
            "neuron_rule": NEURON_RULE,
            "edge_rule": EDGE_RULE,
            "sign_rule": SIGN_RULE,
            "delay_rule": DELAY_RULE,
            "n_neurons": int(self.n),
            "superclass_counts": self.neurons["superclass"].value_counts().to_dict(),
            "nt_counts": {str(k): int(v) for k, v in nt.items()},
            "fraction_inhibitory_neurons": float((self.sign < 0).mean()),
            "graph_min_syn_1": self.graph_summary(1) if self.min_syn <= 1 else None,
            "graph_min_syn_5": self.graph_summary(5),
        }


def _detect_weight_columns(names: list[str]) -> tuple[str, str, str]:
    pre = next((c for c in names if "pre" in c.lower()), None)
    post = next((c for c in names if "post" in c.lower()), None)
    weight = next((c for c in names if c.lower() in ("weight", "syn_count", "count", "n")), None)
    if weight is None:
        weight = [c for c in names if c not in (pre, post)][-1]
    if pre is None or post is None:
        raise ValueError(f"cannot identify pre/post columns in {names}")
    return pre, post, weight


def load_mcns(data_dir: Path = DEFAULT_DIR, min_syn: int = 1, cache: bool = True) -> MCNS:
    data_dir = Path(data_dir)
    cache_path = data_dir / f"mcns_v1.0_edges_minsyn{min_syn}.npz"
    neurons_path = data_dir / "mcns_v1.0_neurons.parquet"
    if cache and cache_path.exists() and neurons_path.exists():
        z = np.load(cache_path)
        neurons = pd.read_parquet(neurons_path)
        return MCNS(neurons, z["pre"], z["post"], z["count"], min_syn=min_syn)

    ann = pf.read_table(path_of("annotations", data_dir)).to_pandas()
    ann = ann[ann["superclass"].notna()].copy()  # NEURON_RULE
    keep_cols = ["bodyId", "type", "flywireType", "hemibrainType", "superclass", "class",
                 "subclass", "somaSide", "rootSide", "status", "statusLabel", "instance",
                 "somaNeuromere", "group"]
    ann = ann[[c for c in keep_cols if c in ann.columns]].sort_values("bodyId").reset_index(drop=True)
    ann["statusLabel"] = ann["statusLabel"].astype(str)

    nt = pf.read_table(path_of("neurotransmitters", data_dir)).to_pandas()
    nt = nt[["body", "consensus_nt", "predicted_nt", "predicted_nt_confidence", "celltype_predicted_nt"]]
    nt = nt.rename(columns={"body": "bodyId", "consensus_nt": "nt", "predicted_nt_confidence": "nt_conf"})
    ann = ann.merge(nt, on="bodyId", how="left")
    ann["nt"] = ann["nt"].fillna("unclear")
    ann["sign"] = np.where(ann["nt"].isin(INHIBITORY), -1, 1).astype(np.int8)

    # 151.9 M segment-to-segment rows: stream in batches, keep neuron-to-neuron rows only
    wt = pf.read_table(path_of("weights", data_dir))
    c_pre, c_post, c_w = _detect_weight_columns(wt.schema.names)
    bodies = ann["bodyId"].to_numpy()
    pres, posts, counts = [], [], []
    for batch in wt.to_batches(max_chunksize=8_000_000):
        pre_b = batch.column(c_pre).to_numpy()
        post_b = batch.column(c_post).to_numpy()
        w = batch.column(c_w).to_numpy()
        pi = np.clip(np.searchsorted(bodies, pre_b), 0, len(bodies) - 1)
        qi = np.clip(np.searchsorted(bodies, post_b), 0, len(bodies) - 1)
        ok = (bodies[pi] == pre_b) & (bodies[qi] == post_b) & (w >= min_syn)  # EDGE_RULE
        pres.append(pi[ok].astype(np.int32))
        posts.append(qi[ok].astype(np.int32))
        counts.append(w[ok].astype(np.int32))
    del wt
    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    count = np.concatenate(counts)
    order = np.lexsort((post, pre))
    pre, post, count = pre[order], post[order], count[order]

    if cache:
        np.savez(cache_path, pre=pre, post=post, count=count)
        ann.to_parquet(neurons_path)
    return MCNS(ann, pre, post, count, min_syn=min_syn)


if __name__ == "__main__":
    m = load_mcns()
    print(json.dumps(m.summary(), indent=1, default=str))
