"""Stage H1: turn a placement of a netlist (embed_netlist.Placement) into a simulable image on
real MCNS neurons, and measure whether the placed circuit computes.

The image keeps the netlist's own indexing: image neuron i *is* designed neuron i, hosted on
the real neuron `placement.mapping[i]` (its bodyId is recorded) or, when the placement left it
out, on a synthetic neuron (a Profile 3 neuron). So every harness written for the netlist
(`protocol.run.run_transactions`, the adder's own test) runs unchanged on the image's topology.

Edges, under the H0 Profile 2 rules (embed_h0.Policy):
  carried    a designed edge (src -> dst, q quanta) whose hosts have an anatomical edge of the
             right sign with count * QUANTA_PER_SYNAPSE * k >= |q| for a scale k <= k_max. It is
             carried at exactly the designed quanta; the scale k is a parameter edit
             (manifest.ParameterEdit), recorded per edge and as a histogram.
  Profile 3  every other designed edge, added at its designed quanta and labelled (src role,
             dst role, quanta, reason). `profile3` selects which of them an image gets; the
             rest are simply absent (that is how conditions B-D of the H1 report are built).
  parasitic  every anatomical edge among the hosts that is not a carried designed edge: zeroed
             when policy.zero_parasitic (a documented zero-weight edit), else kept at its
             anatomical quanta (count * QUANTA_PER_SYNAPSE, sign from the host's transmitter).
  biases     the netlist's per-neuron tonic bias (mV above rest; the flip-flop latch's members
             and its excitatory proxy p, protocol.flipflop -- any biased designed neuron, whatever
             its motif) goes onto the host as a Profile 2 parameter edit
             (ParameterEdit "bias", anatomical 0 -> designed mV; `biases` in the manifest counts)
             and into the image topology's `bias`; a synthetic neuron keeps its bias as part of
             its own (Profile 3) definition.

`simulate_channel` runs a four-phase channel (the adder) on an image with the channel's own
drive and decode; `run_h1_conditions` is the H1 measurement (docs/h1_placement.md).

The same image inside the whole brain: `full_graph_topology` applies the image's edits in place
on the entire MCNS topology (carried edges rescaled, Profile 3 edges added, parasitic edges
zeroed, optionally the hosts' output edges to the rest of the brain and their input edges from
it zeroed too) and returns it with the index map designed neuron -> full-graph neuron. The
channel's harness only knows designed indices, so `CircuitView` presents a full-graph RefSim
through that map: injection goes in as designed indices, spikes come back as designed indices,
and everything else in the brain is invisible to the harness. `bench/h1_fullgraph.py` is the
whole-brain experiment (a silent, a Poisson-background or a burst-driven surround, as H0's).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..lib.netlist import Netlist
from ..sim.model import QUANTA_PER_SYNAPSE, Params, Topology
from ..sim.ref64 import RefSim
from ..sim.trace import SpikeTrace
from .embed_h0 import Policy
from .embed_netlist import Placement, autapse_reason, role_key
from .manifest import Manifest, ParameterEdit, params_dict
from .mcns import MCNS

SCALE_BINS = (0.5, 1.0, 2.0, 3.0, 4.0)  # histogram of the scale k = |q| / (count * 16) of carried edges


@dataclass
class ImageEdge:
    src: int  # designed (= image) index
    dst: int
    quanta: int
    src_role: str
    dst_role: str
    reason: str
    cls: str  # "src role key -> dst role key"
    count: int = 0  # anatomical synapses under it (0 for a Profile 3 edge with no anatomical edge)
    delay: int = 0  # steps (the designed delay for a Profile 3 edge, the default for a parasitic one)


@dataclass
class Image:
    n: int
    topology: Topology
    real: np.ndarray  # designed index -> real neuron index, -1 for a synthetic neuron
    bodies: np.ndarray  # designed index -> bodyId, -1 for a synthetic neuron
    synthetic: list  # designed indices hosted by a synthetic (Profile 3) neuron
    carried: list  # ParameterEdit per carried designed edge (anatomical quanta -> designed quanta)
    scales: dict  # designed edge index -> scale k
    profile3: list  # ImageEdge: the Profile 3 edges this image has
    omitted: list  # ImageEdge: designed edges neither carried nor added (excluded by `profile3`)
    parasitic: list  # ImageEdge: anatomical edges among the hosts that are not carried designed edges
    zero_parasitic: bool
    counts: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    carried_syn: list = field(default_factory=list)  # (src, dst, quanta, delay) per carried designed synapse, designed indices
    biases: list = field(default_factory=list)  # ParameterEdit per biased designed neuron on a real host (0 mV -> designed mV)

    @property
    def profile(self) -> int:
        return 3 if (self.profile3 or self.synthetic) else 2


# ----------------------------------------------------------------------------------------
# Profile 3 selectors: which missing edges an image is given
# ----------------------------------------------------------------------------------------
def all_missing(e: ImageEdge) -> bool:
    return True


def none_missing(e: ImageEdge) -> bool:
    return False


def from_sources(roles) -> Callable[[ImageEdge], bool]:
    """Missing edges whose source has one of these exact roles (e.g. the broadcast neurons)."""
    roles = set(roles)
    return lambda e: e.src_role in roles


def any_of(*selectors) -> Callable[[ImageEdge], bool]:
    return lambda e: any(s(e) for s in selectors)


def of_class(*classes) -> Callable[[ImageEdge], bool]:
    """Missing edges of these classes, written as "src role key -> dst role key"."""
    classes = set(classes)
    return lambda e: e.cls in classes


# ----------------------------------------------------------------------------------------
# the image
# ----------------------------------------------------------------------------------------
def build_image(net: Netlist, m: MCNS, placement: Placement, policy: Policy = Policy(),
                profile3: Callable[[ImageEdge], bool] | None = None, params: Params | None = None) -> Image:
    """The placed netlist as a topology on the netlist's own indices (module docstring).
    `profile3`: which missing designed edges to add (default: all of them)."""
    params = params or net.params
    select = profile3 or all_missing
    n = net.n
    real = np.full(n, -1, np.int64)
    for d, r in placement.mapping.items():
        real[int(d)] = int(r)
    hosts = np.flatnonzero(real >= 0)
    if len(set(real[hosts].tolist())) != len(hosts):
        raise ValueError("two designed neurons share a host")
    bodies_all = m.neurons["bodyId"].to_numpy()
    bodies = np.where(real >= 0, bodies_all[np.maximum(real, 0)], -1).astype(np.int64)
    synthetic = [int(d) for d in np.flatnonzero(real < 0)]
    sign = m.sign
    C = sp.csr_matrix((m.count.astype(np.int64), (m.pre, m.post)), shape=(m.n, m.n))
    C.sum_duplicates()
    reasons = {(int(s), int(d)): r for s, d, _, r in placement.missing}

    def count(s: int, d: int) -> int:
        return int(C[s, d]) if s >= 0 and d >= 0 else 0

    src_t, dst_t, q_t, dl_t = [], [], [], []
    carried_edges, scales, p3, omitted, carried_syn = [], {}, [], [], []
    carried_pairs = set()
    # designed synapses are aggregated per (src, dst) pair before the bound check: a pair the
    # netlist enters twice (the adder's Q.reset_inh -> Q.comp.c2_0.L.u/.v, -2716 twice) must be
    # carried at its SUM, and the sum is what the scale bound applies to — checked per entry, the
    # image carried both on a 47-synapse edge and the full-graph topology summed them to a
    # weight seven times the anatomical one, past k_max (Kimi's review, 2026-09-16)
    agg: dict = {}
    for e, (s, d, q, dl) in enumerate(zip(net.src, net.dst, net.quanta, net.delay)):
        key = (int(s), int(d))
        if key in agg:
            if agg[key][1] != int(dl):
                raise ValueError(f"designed synapses {s} -> {d} with different delays cannot share one anatomical edge")
            agg[key][0] += int(q); agg[key][2].append(e)
        else:
            agg[key] = [int(q), int(dl), [e]]
    n_pairs = len(agg)
    for (s, d), (q, dl, rows) in agg.items():
        e = rows[0]
        rs, rd = int(real[s]), int(real[d])
        cnt = count(rs, rd)
        req = policy.req_count(q)
        if s == d:
            nt = m.neurons["nt"].iat[rs] if rs >= 0 else None
            host_sign = 1 if nt in policy.excitatory_nt else -1 if nt in policy.inhibitory_nt else 0
            detail = autapse_reason(q, cnt, host_sign, policy, rs >= 0)
        elif rs < 0 or rd < 0:
            detail = "endpoint unplaced"
        elif cnt == 0:
            detail = f"no anatomical edge (needs {req} synapses)"
        elif int(sign[rs]) * q < 0:
            detail = f"wrong sign (host transmitter; {cnt} synapses)"
        elif cnt < req:
            detail = f"anatomical edge too weak ({cnt} synapses, needs {req})"
        else:
            detail = None
        if detail is None:
            k = abs(q) / (cnt * QUANTA_PER_SYNAPSE)
            assert k <= policy.k_max + 1e-9
            scales[e] = k
            carried_pairs.add((rs, rd))
            carried_edges.append(ParameterEdit("weight", int(bodies[s]), int(bodies[d]), int(sign[rs]) * cnt * QUANTA_PER_SYNAPSE, int(q),
                                               f"0 <= |q| <= {policy.k_max} * count * {QUANTA_PER_SYNAPSE}, sign preserved",
                                               f"{net.roles[s]} -> {net.roles[d]} (scale {k:.3f})"))
            carried_syn.append((int(s), int(d), int(q), int(dl)))
            src_t.append(s); dst_t.append(d); q_t.append(int(q)); dl_t.append(int(dl))
            continue
        reason = reasons.get((int(s), int(d)))  # the placement's audit, with the data's detail
        edge = ImageEdge(int(s), int(d), int(q), net.roles[s], net.roles[d], f"{reason}: {detail}" if reason else detail,
                         detail if s == d else f"{role_key(net.roles[s])} -> {role_key(net.roles[d])}", cnt, int(dl))
        if select(edge):
            p3.append(edge)
            src_t.append(s); dst_t.append(d); q_t.append(int(q)); dl_t.append(int(dl))
        else:
            omitted.append(edge)

    # parasitic: anatomical edges among the hosts that are not carried designed edges
    parasitic = []
    if len(hosts):
        real_hosts = real[hosts]
        sub = C[real_hosts][:, real_hosts].tocoo()
        for i, j, cnt in zip(sub.row.tolist(), sub.col.tolist(), sub.data.tolist()):
            rs, rd = int(real_hosts[i]), int(real_hosts[j])
            if (rs, rd) in carried_pairs or (rs == rd and (int(hosts[i]), int(hosts[j])) not in agg):
                continue
            s, d = int(hosts[i]), int(hosts[j])
            reason = "under a designed edge (too weak or wrong sign)" if any(a == s and b == d for a, b in zip(net.src, net.dst)) else "not designed"
            parasitic.append(ImageEdge(s, d, int(sign[rs]) * int(cnt) * QUANTA_PER_SYNAPSE, net.roles[s], net.roles[d], reason,
                                       f"{role_key(net.roles[s])} -> {role_key(net.roles[d])}", int(cnt), params.default_delay_steps))
            if not policy.zero_parasitic:
                src_t.append(s); dst_t.append(d); q_t.append(int(sign[rs]) * int(cnt) * QUANTA_PER_SYNAPSE); dl_t.append(params.default_delay_steps)
    bias = np.zeros(n, np.float64)  # the netlist's tonic biases (mV), image index = designed index
    bias[: len(net.bias)] = np.asarray(net.bias, dtype=np.float64)[:n]
    topo = Topology.from_edges(n, src_t, dst_t, q_t, dl_t, bias=bias if np.any(bias) else None)
    bias_edits = [ParameterEdit("bias", None, int(bodies[d]), 0.0, float(bias[d]), "tonic bias, mV added to the resting potential",
                                f"{net.roles[d]} bias {bias[d]:g} mV") for d in np.flatnonzero(bias != 0).tolist() if real[d] >= 0]

    hist = {}
    lo = 0.0
    for hi in SCALE_BINS:
        hist[f"({lo}, {hi}]"] = int(sum(1 for k in scales.values() if lo < k <= hi))
        lo = hi
    by_class: dict = {}
    for e in p3:
        by_class[e.cls] = by_class.get(e.cls, 0) + 1
    omitted_by_class: dict = {}
    for e in omitted:
        omitted_by_class[e.cls] = omitted_by_class.get(e.cls, 0) + 1
    counts = {
        "neurons": n, "placed": int(len(hosts)), "synthetic_neurons": len(synthetic),
        "designed_edges": net.nnz, "designed_pairs": n_pairs, "carried": len(carried_edges),
        "profile3_edges": len(p3), "profile3_by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "omitted_missing_edges": len(omitted), "omitted_by_class": dict(sorted(omitted_by_class.items(), key=lambda kv: -kv[1])),
        "parasitic": len(parasitic), "parasitic_zeroed": len(parasitic) if policy.zero_parasitic else 0,
        "parasitic_kept": 0 if policy.zero_parasitic else len(parasitic),
        "parasitic_under_designed": sum(1 for e in parasitic if e.reason.startswith("under")),
        "parasitic_abs_quanta": int(sum(abs(e.quanta) for e in parasitic)),
        "scales": hist, "scale_max": round(max(scales.values()), 3) if scales else None,
        "topology_edges": topo.nnz,
        "biases": len(bias_edits),  # biased designed neurons on real hosts: one parameter edit each
        "biases_synthetic": int(sum(1 for d in np.flatnonzero(bias != 0).tolist() if real[d] < 0)),
    }
    img = Image(n, topo, real, bodies, synthetic, carried_edges, scales, p3, omitted, parasitic, policy.zero_parasitic, counts,
                carried_syn=carried_syn, biases=bias_edits)
    if placement.carried and placement.carried != len(carried_edges):
        counts["placement_carried_mismatch"] = placement.carried  # the audit and the image disagree: report it
    img.manifest = image_manifest(net, m, img, policy, params)
    return img


def image_manifest(net: Netlist, m: MCNS, img: Image, policy: Policy, params: Params) -> dict:
    """A manifest in the style of connectome/manifest.py: the four graphs (original, retained,
    active-nonzero, silencing), the parameter edits (carried edges rescaled, the hosts' tonic
    biases, parasitic edges zeroed) and the structural edits (Profile 3 edges and synthetic neurons), plus the counts."""
    edits = list(img.carried) + list(img.biases)
    if img.zero_parasitic:
        edits += [ParameterEdit("weight", int(img.bodies[e.src]), int(img.bodies[e.dst]), e.quanta, 0, "documented zero weight",
                                f"parasitic edge among circuit neurons ({e.reason})") for e in img.parasitic]
    structural = [{"kind": "add_edge", "pre": int(img.bodies[e.src]), "post": int(img.bodies[e.dst]), "pre_role": e.src_role,
                   "post_role": e.dst_role, "quanta": e.quanta, **({"delay": e.delay} if e.src == e.dst else {}),
                   "class": e.cls, "reason": e.reason} for e in img.profile3]
    structural += [{"kind": "add_neuron", "role": net.roles[d], "reason": "unplaced by the search"} for d in img.synthetic]
    g0 = m.graph_summary()
    n_par_zero = len(img.parasitic) if img.zero_parasitic else 0
    active = {**g0, "n_edges": g0["n_edges"] - n_par_zero + len(img.profile3), "n_neurons": g0["n_neurons"] + len(img.synthetic),
              "note": "original minus zeroed parasitic edges, plus Profile 3 edges; carried designed edges rescaled"}
    man = Manifest(
        connectome={"id": "MCNS", "version": m.version, **{k: v for k, v in m.summary().items() if k.endswith("_rule")}},
        profile=img.profile, params=params_dict(params),
        circuit_bodies=[int(b) for b in img.bodies if b >= 0],
        graphs={"original": g0, "retained": g0, "active_nonzero": active,
                "image": {"n_neurons": img.n, "n_edges": img.topology.nnz, "note": "isolated: circuit neurons only, netlist indices"}},
        silencing=[], parameter_edits=edits, structural_edits=structural,
        execution_mode="isolated",
        isolation_classification="isolated image (no surround); parasitic edges " + ("zeroed" if img.zero_parasitic else "kept at anatomical quanta"),
        notes=[f"counts: {json.dumps(img.counts)}"],
    )
    man.validate()
    d = dataclasses.asdict(man)
    d["counts"] = img.counts
    return d


# ----------------------------------------------------------------------------------------
# saving / loading a mapping
# ----------------------------------------------------------------------------------------
MAPPING_META_KEYS = (
    "builder", "width", "ordered", "watchdog_hops", "storage", "rate_robust",
    "experimental_autapse", "ordered_xor_oneshot", "k_max", "isolation_weight",
    "region", "restarts", "seed", "latch_feedback", "dale_feedback_version",
    "dale_biased_version", "dale_refractory_version",
    "isolation_in", "isolation_out", "echo_weight",
)


def _mapping_json(path: str) -> dict:
    """Read a saved mapping."""
    return json.loads(Path(path).read_text())


def load_mapping_meta(path: Path) -> dict:
    """The recognized, top-level build/placement metadata in a saved mapping.

    Old mapping files did not have the Stage-H contract and simply return whichever of these
    keys they happen to contain (notably the old flip-flop mapping's ``storage``).
    """
    d = _mapping_json(str(Path(path)))
    return {key: d[key] for key in MAPPING_META_KEYS if key in d}


def save_mapping(path: Path, net: Netlist, m: MCNS, pl: Placement, meta: dict | None = None) -> None:
    bodies = m.neurons["bodyId"].to_numpy()
    out = {**(meta or {}), "summary": pl.summary(net), "strategy": pl.strategy, "order_note": pl.order_note,
           "mapping": {net.roles[d]: {"designed": int(d), "neuron_index": int(r), "bodyId": int(bodies[r])} for d, r in sorted(pl.mapping.items())},
           "unplaced": [net.roles[d] for d in pl.unplaced],
           "missing": [{"src": net.roles[s], "dst": net.roles[d], "quanta": int(q), "reason": r} for s, d, q, r in pl.missing]}
    Path(path).write_text(json.dumps(out, indent=1))


def load_mapping(path: Path, net: Netlist, m: MCNS) -> Placement:
    """A Placement rebuilt from a saved mapping (roles -> bodyIds); `carried` and `missing` are
    taken from the file, `build_image` re-derives both from the data."""
    d = _mapping_json(str(Path(path)))
    saved_mapping_roles = set(d["mapping"])
    saved_unplaced_roles = set(d.get("unplaced", []))
    saved_roles = saved_mapping_roles | saved_unplaced_roles
    net_roles = set(net.roles)
    if saved_roles != net_roles:
        raise ValueError(
            "mapping roles do not match rebuilt netlist: "
            f"{len(saved_mapping_roles)} mapped + {len(saved_unplaced_roles)} unplaced "
            f"({len(saved_roles)} unique saved) vs {len(net_roles)} netlist roles"
        )
    roles = {r: i for i, r in enumerate(net.roles)}
    mapping = {roles[role]: int(m.index_of(np.array([v["bodyId"]]))[0]) for role, v in d["mapping"].items()}
    missing = [(roles[e["src"]], roles[e["dst"]], int(e["quanta"]), e["reason"]) for e in d.get("missing", [])]
    unplaced = [roles[r] for r in d.get("unplaced", [])]
    return Placement(mapping, unplaced, int(d.get("summary", {}).get("carried", 0)), missing,
                     int(d.get("summary", {}).get("parasitic_to_zero", 0)), 0.0, strategy=d.get("strategy", "loaded"))


# ----------------------------------------------------------------------------------------
# the image inside the whole brain
# ----------------------------------------------------------------------------------------
@dataclass
class FullGraph:
    """The whole MCNS topology with an image's edits applied in place (`full_graph_topology`)."""
    topology: Topology  # m.n real neurons, then the image's synthetic neurons
    index_map: np.ndarray  # designed index -> full-graph index
    n_real: int  # m.n; indices >= n_real are the synthetic neurons
    counts: dict

    @property
    def circuit(self) -> np.ndarray:
        """The circuit's full-graph indices, in designed order (= index_map)."""
        return self.index_map

    def designed_of(self, n_full: int | None = None) -> np.ndarray:
        """Inverse map: full-graph index -> designed index, -1 outside the circuit."""
        inv = np.full(n_full or self.topology.n, -1, np.int64)
        inv[self.index_map] = np.arange(len(self.index_map))
        return inv


def _locate(base: Topology, pre: int, post: int) -> range:
    """Synapse indices of every `pre -> post` entry of a CSR-by-source topology (sorted by dst)."""
    lo, hi = int(base.indptr[pre]), int(base.indptr[pre + 1])
    sub = base.dst[lo:hi]
    a, b = int(np.searchsorted(sub, post, "left")), int(np.searchsorted(sub, post, "right"))
    return range(lo + a, lo + b)


def validate_surround_settings(surround_scale_exc: float = 1.0, surround_scale_inh: float = 1.0,
                               surround_input_cap: float | None = None) -> None:
    """Gains and the optional cap (synapse-equivalent weights) are finite and nonnegative."""
    for name, value in (("surround_scale_exc", surround_scale_exc),
                        ("surround_scale_inh", surround_scale_inh),
                        ("surround_input_cap", surround_input_cap)):
        if value is None and name == "surround_input_cap":
            continue
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _scale_surround(base: Topology, quanta: np.ndarray, eligible: np.ndarray,
                    exc: float, inh: float, cap: float | None) -> dict:
    """Scale non-image anatomy, cap gain-scaled excitation, then round once (ties to even).

    Host-to-host anatomy is excluded, even if parasitic edges were kept. The cap sums only
    eligible excitation; designed inputs never consume it. Boundary cuts are applied later.
    """
    edges = np.flatnonzero(eligible)
    original = quanta[edges]
    positive = original > 0
    with np.errstate(over="ignore", invalid="ignore"):
        weights = original.astype(np.float64) * np.where(positive, exc, inh)
    if not np.all(np.isfinite(weights)):
        raise ValueError("surround weight overflow")
    capped_neurons = 0
    if cap is not None:
        posts = base.dst[edges[positive]]
        # Divide before summing to express the cap in single-synapse weights, not quanta.
        totals = np.bincount(posts, weights=weights[positive] / QUANTA_PER_SYNAPSE, minlength=base.n)
        if not np.all(np.isfinite(totals)):
            raise ValueError("surround input total overflow")
        over = totals > cap
        capped_neurons = int(over.sum())
        factors = np.ones(base.n, np.float64)
        factors[over] = cap / totals[over]
        weights[positive] *= factors[posts]
    rounded = np.rint(weights)
    if np.any(np.abs(rounded) > np.iinfo(np.int32).max):
        raise ValueError("surround quanta overflow")
    quanta[edges] = rounded.astype(np.int64)
    return {"surround_scale_exc": float(exc), "surround_scale_inh": float(inh),
            "surround_input_cap": None if cap is None else float(cap),
            "surround_eligible_edges": int(len(edges)),
            "surround_changed_edges": int(np.count_nonzero(quanta[edges] != original)),
            "surround_capped_neurons": capped_neurons,
            "surround_zeroed_edges": int(np.count_nonzero((original != 0) & (rounded == 0))),
            "surround_zeroed_by_rounding": int(np.count_nonzero((weights != 0) & (rounded == 0))),
            "surround_rounding": "nearest integer quantum, ties to even"}


def full_graph_topology(m: MCNS, params: Params, image: Image, zero_outputs: bool = False, zero_inputs: bool = False,
                        base: Topology | None = None, cut_outputs_frac: float = 0.0,
                        cut_inputs_frac: float = 0.0, cut_inputs_by: str = "synapses",
                        attribution_in: Path | None = None,
                        cut_inputs_from_types: Collection[str] | None = None,
                        surround_scale_exc: float = 1.0, surround_scale_inh: float = 1.0,
                        surround_input_cap: float | None = None) -> FullGraph:
    """The whole MCNS topology (`m.topology(params)`, or `base` if the caller already has it)
    with the image's edits applied in place, as h0_run.full_topology does for H0's circuit:

      carried    each carried designed synapse rescales its hosts' anatomical edge to the designed
                 quanta (two designed synapses on one pair sum onto the one edge) at the designed delay;
      parasitic  zeroed when the image zeroes them (kept otherwise: they are the anatomy);
      Profile 3  added as new synapses at the designed quanta and delay; the image's synthetic
                 neurons are appended after the m.n real ones;
      biases     the image topology's per-neuron bias goes onto each host (and synthetic neuron)
                 at its full-graph index, on top of the base topology's own biases if it has any;
      zero_outputs   also zero every edge from a host to a neuron outside the circuit (H0's
                 outputs-zeroed condition: the circuit cannot disturb the brain);
      zero_inputs    also zero every edge from outside the circuit into a host (the isolated case
                 embedded: the brain cannot disturb the circuit; a control).
      cut_outputs_frac / cut_inputs_frac  zero just the strongest fraction of those respective
                 anatomical boundary edges.  Ties use (post, pre) order so a cut ladder is
                 reproducible; a non-zero fraction cuts at least one edge.
      cut_inputs_by="attribution"  rank inputs by descending score from an h_fault_attr NPZ,
                 then synapse count, then (post, pre). Missing scores are zero. BodyIds identify
                 pairs across runs; foreign pairs, duplicate pairs and invalid scores are rejected.
      cut_inputs_from_types  zero every incoming anatomical edge whose outside presynaptic
                 MCNS neuron has one of these exact ``type`` strings.  This is exclusive with
                 input fractions and ``zero_inputs``.
      surround_scale_exc / surround_scale_inh  multiply positive / negative anatomical weights
                 only when at least one endpoint is outside the image. All host-to-host edges,
                 carried and Profile 3 edges are untouched. After gains, surround_input_cap
                 proportionally caps eligible positive input per destination, in synapse-weight
                 units (quanta / QUANTA_PER_SYNAPSE). Round once to nearest quantum, ties to even;
                 boundary cuts follow. Non-default settings and rounding counts appear in counts;
                 defaults (1, 1, None) preserve the legacy topology AND counts byte for byte.
    `base` must be the anatomical topology from `MCNS.topology` (quanta = sign * synapse count * QUANTA_PER_SYNAPSE), because the cut fractions recover synapse counts from it.

    Returns the topology, the index map designed -> full-graph, and the counts of each edit."""
    validate_surround_settings(surround_scale_exc, surround_scale_inh, surround_input_cap)
    for name, fraction in (("cut_outputs_frac", cut_outputs_frac), ("cut_inputs_frac", cut_inputs_frac)):
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{name} must be a finite float in [0, 1], got {fraction!r}")
    if zero_outputs and cut_outputs_frac:
        raise ValueError("zero_outputs and cut_outputs_frac cannot be combined")
    if cut_inputs_from_types is not None:
        if isinstance(cut_inputs_from_types, str):
            raise ValueError("cut_inputs_from_types must be a collection of type strings, not a string")
        requested_types = list(dict.fromkeys(cut_inputs_from_types))
        if not all(isinstance(t, str) for t in requested_types):
            raise ValueError("cut_inputs_from_types must contain only type strings")
    else:
        requested_types = None
    if zero_inputs and (cut_inputs_frac or requested_types is not None):
        raise ValueError("zero_inputs cannot be combined with cut_inputs_frac or cut_inputs_from_types")
    if cut_inputs_frac and requested_types is not None:
        raise ValueError("cut_inputs_frac and cut_inputs_from_types cannot be combined")
    if cut_inputs_by not in ("synapses", "attribution"):
        raise ValueError("cut_inputs_by must be synapses or attribution")
    if (cut_inputs_by == "attribution") != (attribution_in is not None):
        raise ValueError("cut_inputs_by=attribution requires attribution_in (and vice versa)")
    base = base if base is not None else m.topology(params)
    if base.n != m.n:
        raise ValueError("base topology is not the connectome's")
    n_syn = len(image.synthetic)
    n_full = m.n + n_syn
    index_map = image.real.astype(np.int64).copy()
    for k, d in enumerate(image.synthetic):
        index_map[d] = m.n + k
    quanta = base.quanta.astype(np.int64)  # int64 while summing; back to int32 at the end
    delay = base.delay.copy()

    # carried designed synapses: the anatomical edge takes the designed quanta (summed per pair) and delay
    per_edge: dict = {}
    for s, d, q, dl in image.carried_syn:
        ks = _locate(base, int(index_map[s]), int(index_map[d]))
        if len(ks) == 0:
            raise ValueError(f"carried designed edge {s} -> {d} has no anatomical edge between its hosts")
        k = ks[0]
        per_edge.setdefault(k, [0, dl, list(ks)])
        per_edge[k][0] += int(q)
        if per_edge[k][1] != dl:
            raise ValueError(f"two designed synapses on one anatomical edge with different delays ({s} -> {d})")
    for k, (q, dl, ks) in per_edge.items():
        for kk in ks:
            quanta[kk] = 0
        quanta[k] = q
        delay[k] = dl
    n_delay_edits = sum(1 for k, (q, dl, ks) in per_edge.items() if dl != base.delay[k])

    n_par = 0
    if image.zero_parasitic:
        for e in image.parasitic:
            for k in _locate(base, int(index_map[e.src]), int(index_map[e.dst])):
                quanta[k] = 0
                n_par += 1

    hosts = index_map[image.real >= 0]
    in_circuit = np.zeros(n_full, bool)
    in_circuit[index_map] = True
    output_sel = np.isin(base.src, hosts) & ~in_circuit[base.dst]
    input_sel = in_circuit[base.dst] & ~in_circuit[base.src]
    surround_report = None
    if surround_scale_exc != 1 or surround_scale_inh != 1 or surround_input_cap is not None:
        eligible = ~(in_circuit[base.src] & in_circuit[base.dst])
        surround_report = _scale_surround(base, quanta, eligible, surround_scale_exc,
                                          surround_scale_inh, surround_input_cap)

    def cut_boundary(sel: np.ndarray, fraction: float, zero: bool, attribution: bool = False) -> dict:
        """Zero all or a strongest deterministic prefix of one anatomical boundary."""
        edges = np.flatnonzero(sel)
        # MCNS.topology sets quanta to sign * count * QUANTA_PER_SYNAPSE.  `base` is never
        # edited, so this recovers anatomical counts without a per-edge Python lookup.
        counts = np.abs(base.quanta[edges]).astype(np.int64) // QUANTA_PER_SYNAPSE
        scores = None
        if attribution:
            with np.load(attribution_in, allow_pickle=False) as saved:
                pre, post, score = (saved[key] for key in ("pre_bodyId", "post_bodyId", "score"))
            if any(a.ndim != 1 for a in (pre, post, score)) or not (len(pre) == len(post) == len(score)):
                raise ValueError("attribution arrays must be equal-length vectors")
            if not np.issubdtype(pre.dtype, np.integer) or not np.issubdtype(post.dtype, np.integer):
                raise ValueError("attribution bodyIds must be integers")
            if not (np.issubdtype(score.dtype, np.integer) or np.issubdtype(score.dtype, np.floating)):
                raise ValueError("attribution scores must be real numbers")
            if not np.all(np.isfinite(score)) or np.any(score < 0):
                raise ValueError("attribution scores must be finite and nonnegative")
            integer_scores = np.issubdtype(score.dtype, np.integer)
            if integer_scores and np.any(score > np.iinfo(np.int64).max):
                raise ValueError("integer attribution scores must fit int64")
            try:
                keys = m.index_of(pre).astype(np.int64) * m.n + m.index_of(post)
            except KeyError as exc:
                raise ValueError("attribution contains foreign bodyIds") from exc
            if len(np.unique(keys)) != len(keys):
                raise ValueError("attribution contains duplicate pairs")
            boundary_keys = base.src[edges].astype(np.int64) * m.n + base.dst[edges]
            pos = np.searchsorted(boundary_keys, keys)
            if len(keys) and (np.any(pos >= len(edges)) or np.any(boundary_keys[np.minimum(pos, len(edges) - 1)] != keys)):
                raise ValueError("attribution contains pairs outside this incoming boundary")
            scores = np.zeros(len(edges), np.int64 if integer_scores else np.float64)
            scores[pos] = score
        n_cut = len(edges) if zero else math.ceil(Fraction(str(fraction)) * len(edges))
        if n_cut:
            # np.lexsort's final key is primary: stronger first, then post and pre ascending.
            rank_keys = (base.src[edges], base.dst[edges], -counts)
            if scores is not None:
                rank_keys += (-scores,)
            rank_order = np.lexsort(rank_keys)
            cut = edges[rank_order[:n_cut]]
            cut_counts = counts[rank_order[:n_cut]]
            quanta[cut] = 0
            cut_synapses = int(cut_counts.sum())
            weakest = int(cut_counts.min())
        else:
            cut_synapses, weakest = 0, None
        total_synapses = int(counts.sum())
        return {"edges": int(len(edges)), "synapses": total_synapses,
                "cut_edges": int(n_cut), "cut_synapses": cut_synapses,
                "remaining_edges": int(len(edges) - n_cut), "remaining_synapses": total_synapses - cut_synapses,
                "weakest_cut_synapses": weakest}

    outputs = cut_boundary(output_sel, float(cut_outputs_frac), bool(zero_outputs))
    inputs = cut_boundary(input_sel, float(cut_inputs_frac), bool(zero_inputs), cut_inputs_by == "attribution")
    type_cut_report = None
    if requested_types is not None:
        neuron_types = m.neurons["type"].astype(str).to_numpy()
        source_matches = ~in_circuit[:m.n] & np.isin(neuron_types, requested_types)
        type_sel = input_sel & source_matches[base.src]
        cut = np.flatnonzero(type_sel)
        cut_counts = np.abs(base.quanta[cut]).astype(np.int64) // QUANTA_PER_SYNAPSE
        quanta[cut] = 0
        all_input_edges = np.flatnonzero(input_sel)
        all_input_counts = np.abs(base.quanta[all_input_edges]).astype(np.int64) // QUANTA_PER_SYNAPSE
        cut_synapses = int(cut_counts.sum())
        inputs = {"edges": int(len(all_input_edges)), "synapses": int(all_input_counts.sum()),
                  "cut_edges": int(len(cut)), "cut_synapses": cut_synapses,
                  "remaining_edges": int(len(all_input_edges) - len(cut)),
                  "remaining_synapses": int(all_input_counts.sum()) - cut_synapses,
                  "weakest_cut_synapses": int(cut_counts.min()) if len(cut) else None}
        matched_types = set(neuron_types[source_matches])
        missing_types = [t for t in requested_types if t not in matched_types]
        type_cut_report = {"cut_inputs_from_types": requested_types,
                           "cut_inputs_from_types_matched_neurons": int(source_matches.sum()),
                           "cut_inputs_from_types_unmatched": missing_types}

    src = base.src
    dst = base.dst
    if image.profile3:
        src = np.concatenate([src, index_map[[e.src for e in image.profile3]].astype(np.int32)])
        dst = np.concatenate([dst, index_map[[e.dst for e in image.profile3]].astype(np.int32)])
        quanta = np.concatenate([quanta, np.array([e.quanta for e in image.profile3], np.int64)])
        delay = np.concatenate([delay, np.array([e.delay for e in image.profile3], np.int32)])
    if len(quanta) and abs(quanta).max() > np.iinfo(np.int32).max:
        raise ValueError("quanta overflow")
    bias = None
    n_bias = n_bias_syn = 0
    if base.bias is not None or image.topology.bias is not None:
        bias = np.zeros(n_full, np.float64)
        if base.bias is not None:
            bias[: m.n] = base.bias
        if image.topology.bias is not None:
            biased = np.flatnonzero(image.topology.bias)
            bias[index_map[biased]] = image.topology.bias[biased]
            n_bias = int((image.real[biased] >= 0).sum())
            n_bias_syn = int(len(biased) - n_bias)
    topo = Topology.from_edges(n_full, src, dst, quanta.astype(np.int32), delay, bias=bias)
    counts = {"n": n_full, "n_real": int(m.n), "synthetic_neurons": n_syn, "topology_edges": topo.nnz,
              "carried_synapses": len(image.carried_syn), "carried_edges_rescaled": len(per_edge), "delay_edits": int(n_delay_edits),
              "profile3_added": len(image.profile3), "parasitic_zeroed": n_par,
              # The legacy names are retained, but now mean all boundary edges actually cut.
              "outputs_zeroed": outputs["cut_edges"], "inputs_zeroed": inputs["cut_edges"],
              "zero_outputs": bool(zero_outputs), "zero_inputs": bool(zero_inputs),
              "cut_outputs_frac": float(cut_outputs_frac), "cut_inputs_frac": float(cut_inputs_frac),
              "outputs_edges": outputs["edges"], "outputs_synapses": outputs["synapses"],
              "outputs_cut_edges": outputs["cut_edges"], "outputs_cut_synapses": outputs["cut_synapses"],
              "outputs_remaining_edges": outputs["remaining_edges"], "outputs_remaining_synapses": outputs["remaining_synapses"],
              "outputs_weakest_cut_synapses": outputs["weakest_cut_synapses"],
              "inputs_edges": inputs["edges"], "inputs_synapses": inputs["synapses"],
              "inputs_cut_edges": inputs["cut_edges"], "inputs_cut_synapses": inputs["cut_synapses"],
              "inputs_remaining_edges": inputs["remaining_edges"], "inputs_remaining_synapses": inputs["remaining_synapses"],
              "inputs_weakest_cut_synapses": inputs["weakest_cut_synapses"],
              "biases": n_bias, "biases_synthetic": n_bias_syn,
              "nonzero_edges": int((topo.quanta != 0).sum())}
    if cut_inputs_by == "attribution":
        counts["cut_inputs_by"] = cut_inputs_by
    if surround_report is not None:
        counts.update(surround_report)
    if type_cut_report is not None:
        counts.update(type_cut_report)
        if type_cut_report["cut_inputs_from_types_unmatched"]:
            counts["warnings"] = ["requested input source type matched no outside neuron: " + t
                                  for t in type_cut_report["cut_inputs_from_types_unmatched"]]
    return FullGraph(topo, index_map, int(m.n), counts)


class CircuitView:
    """A RefSim seen through an index map: the harness injects into and reads spikes of *designed*
    indices, the simulator runs the whole topology. Mirrors the parts of RefSim that
    protocol.run.run_transactions uses (`step`, `step_index`, `add_events`, `_spk_step`,
    `_spk_neuron`, `trace`), with the spike lists holding only the circuit's spikes, renumbered.
    `sim` is the underlying RefSim (its `trace` has every neuron of the topology)."""

    def __init__(self, sim: RefSim, index_map: np.ndarray):
        self.sim = sim
        self.index_map = np.asarray(index_map, np.int64)
        self.n = int(len(self.index_map))
        self.inverse = np.full(sim.n, -1, np.int64)
        self.inverse[self.index_map] = np.arange(self.n)
        self._spk_step: list[np.ndarray] = []
        self._spk_node: list[np.ndarray] = []
        self._spk_neuron: list[np.ndarray] = []
        self._seen = len(sim._spk_step)

    @property
    def step_index(self) -> int:
        return self.sim.step_index

    @property
    def params(self) -> Params:
        return self.sim.params

    def add_events(self, node: int, steps, neurons, quanta) -> None:
        neurons = self.index_map[np.asarray(neurons, np.int64).ravel()]
        self.sim.add_events(node, steps, neurons, quanta)

    def _absorb(self) -> None:
        """Copy the circuit's share of the underlying simulator's new spike batches."""
        while self._seen < len(self.sim._spk_step):
            n_idx = self.sim._spk_neuron[self._seen]
            d = self.inverse[n_idx]
            keep = d >= 0
            if keep.any():
                self._spk_step.append(self.sim._spk_step[self._seen][keep])
                self._spk_node.append(self.sim._spk_node[self._seen][keep])
                self._spk_neuron.append(d[keep])
            self._seen += 1

    def step(self) -> None:
        self.sim.step()
        self._absorb()

    def run(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            self.step()

    def drop_underlying_spikes(self) -> int:
        """Forget the whole-brain spike record so far (the circuit's own is kept): a chained run of
        many additions would otherwise hold every surround spike of every addition. Returns the
        number of batches dropped; the caller reads what it needs from `sim` first."""
        self._absorb()
        n = len(self.sim._spk_step)
        self.sim._spk_step.clear(); self.sim._spk_node.clear(); self.sim._spk_neuron.clear()
        self._seen = 0
        return n

    @property
    def trace(self) -> SpikeTrace:
        """The circuit's spikes, in designed indices."""
        self._absorb()
        if not self._spk_step:
            return SpikeTrace.empty()
        return SpikeTrace.from_arrays(np.concatenate(self._spk_step), np.concatenate(self._spk_node), np.concatenate(self._spk_neuron))


def circuit_sim(topology: Topology, params: Params, index_map=None):
    """A simulator the channel's harness can drive by designed index: the RefSim itself when the
    topology is the isolated image (identity map), else a CircuitView over the full topology."""
    sim = RefSim(topology, params)
    if index_map is None:
        return sim
    index_map = np.asarray(index_map, np.int64)
    if len(index_map) == topology.n and np.array_equal(index_map, np.arange(topology.n)):
        return sim
    return CircuitView(sim, index_map)


def power_on(ch, sim) -> None:
    """Inject a channel's power-on events into a made simulator.

    ``CircuitView.add_events`` deliberately accepts designed indices, so this is correct both
    for an isolated image and for its image embedded in a full graph.
    """
    for step, neuron, quanta in ch.power_on_events(0):
        sim.add_events(0, [step], [neuron], [quanta])


def underlying(sim) -> RefSim:
    return sim.sim if isinstance(sim, CircuitView) else sim


def trace_since(sim: RefSim, batch: int) -> SpikeTrace:
    """The RefSim's spikes from its `batch`-th recorded spike batch on (batches are per step, in
    order), without rebuilding the whole trace: what one addition of a chained run produced."""
    if batch >= len(sim._spk_step):
        return SpikeTrace.empty()
    return SpikeTrace.from_arrays(np.concatenate(sim._spk_step[batch:]), np.concatenate(sim._spk_node[batch:]),
                                  np.concatenate(sim._spk_neuron[batch:]))


# ----------------------------------------------------------------------------------------
# the surround, driven as H0 drives it (h0_run.run_transaction)
# ----------------------------------------------------------------------------------------
FORCE_SPIKE = 100_000  # quanta that force a spike on the next step (h0_run's `big`)


def surround_events(kind: str, neurons: np.ndarray, params: Params, start: int, n_steps: int, rng: np.random.Generator,
                    rate_hz: float = 2.0, burst_n: int = 1000, burst_at: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """External events (steps, neurons, quanta) for one surround condition over [start, start + n_steps):
      silent   none;
      poisson  every neuron of `neurons` fires independently with probability rate_hz * dt / 1000 per
               step (H0's Bernoulli-per-cell background, drawn sparsely: a binomial count per step, then
               that many distinct neurons -- the same distribution without the dense n_steps x N table);
      burst    `burst_n` of `neurons` drawn without replacement fire together at start + burst_at
               (H0: 1,000 cholinergic neurons 5 ms after DATA).
    Every event carries FORCE_SPIKE quanta, so the driven neuron spikes on the next step."""
    neurons = np.asarray(neurons, np.int64)
    if kind == "silent" or len(neurons) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    if kind == "poisson":
        p = rate_hz * params.dt / 1000.0
        hits = rng.binomial(len(neurons), p, size=n_steps)
        steps = np.repeat(np.arange(start, start + n_steps, dtype=np.int64), hits)
        picks = np.concatenate([rng.choice(len(neurons), h, replace=False) for h in hits if h]) if hits.any() else np.empty(0, np.int64)
        return steps, neurons[picks], np.full(len(steps), FORCE_SPIKE, np.int64)
    if kind == "burst":
        picks = rng.choice(neurons, min(burst_n, len(neurons)), replace=False)
        return np.full(len(picks), start + burst_at, np.int64), picks, np.full(len(picks), FORCE_SPIKE, np.int64)
    raise ValueError(f"unknown surround {kind!r}")


def surround_populations(m: MCNS, exclude=()) -> dict:
    """H0's driven populations: the sensory neurons (background) and the cholinergic neurons (burst),
    minus `exclude` (the circuit's hosts: a forced spike into a host would be a fault of the
    experiment, not of the surround)."""
    ex = np.zeros(m.n, bool)
    if len(exclude):
        ex[np.asarray(list(exclude), np.int64)] = True
    sensory = np.flatnonzero(m.neurons["superclass"].str.contains("sensory", na=False).to_numpy() & ~ex)
    exc = np.flatnonzero((m.neurons["nt"] == "acetylcholine").to_numpy() & ~ex)
    return {"sensory": sensory, "cholinergic": exc,
            "excluded_hosts": {"sensory": int((m.neurons["superclass"].str.contains("sensory", na=False).to_numpy() & ex).sum()),
                               "cholinergic": int(((m.neurons["nt"] == "acetylcholine").to_numpy() & ex).sum())}}


def boundary_envelope_ms(topo: Topology, trace: SpikeTrace, circuit: np.ndarray, start: int, end: int, params: Params) -> dict:
    """What crosses into the circuit from outside over steps [start, end): every spike of a non-circuit
    neuron, through each of its non-zero synapses onto a circuit neuron, counted at its arrival step
    as one presynaptic spike, split by transmitter sign (the sign of the synapse's quanta, which is
    the presynaptic neuron's sign under SIGN_RULE), binned per ms. Vectorised (h0_run.boundary_envelope
    loops per spike; a 1.2 s whole-brain run has millions of surround spikes)."""
    circ = np.asarray(circuit, np.int64)
    in_circ = np.zeros(topo.n, bool)
    in_circ[circ] = True
    n_ms = int(np.ceil((end - start) * params.dt))
    out = {"exc_per_ms": np.zeros(n_ms, np.int64), "inh_per_ms": np.zeros(n_ms, np.int64),
           "exc_quanta_per_ms": np.zeros(n_ms, np.int64), "inh_quanta_per_ms": np.zeros(n_ms, np.int64)}
    ev = trace.events
    sel = (ev["step"] >= start) & (ev["step"] < end) & ~in_circ[ev["neuron"]]
    steps, src = ev["step"][sel], ev["neuron"][sel]
    per_neuron_max = np.zeros(len(circ), np.int64)
    n_src = int(len(np.unique(src)))
    if len(src):
        # the synapses into the circuit, CSR by source: a slice per spiking neuron
        into = np.flatnonzero(in_circ[topo.dst] & (topo.quanta != 0))
        into_src = topo.src[into]
        lo = np.searchsorted(into_src, src, "left")
        hi = np.searchsorted(into_src, src, "right")
        lens = hi - lo
        total = int(lens.sum())
        if total:
            rep = np.repeat(np.arange(len(src)), lens)
            offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
            syn = into[lo[rep] + offs]
            arrive = steps[rep] + topo.delay[syn] - start
            ms = np.minimum((arrive * params.dt).astype(np.int64), n_ms - 1)
            q = topo.quanta[syn].astype(np.int64)
            exc = q > 0
            np.add.at(out["exc_per_ms"], ms[exc], 1)
            np.add.at(out["inh_per_ms"], ms[~exc], 1)
            np.add.at(out["exc_quanta_per_ms"], ms[exc], q[exc])
            np.add.at(out["inh_quanta_per_ms"], ms[~exc], q[~exc])
            pos = np.full(topo.n, -1, np.int64)
            pos[circ] = np.arange(len(circ))
            per_step = np.zeros((end - start + 1, len(circ)), np.int64)
            np.add.at(per_step, (np.minimum(arrive, end - start), pos[topo.dst[syn]]), np.abs(q))
            per_neuron_max = per_step.max(axis=0)
    res = {k: v.tolist() for k, v in out.items()}
    res.update({"surround_spikes": int(len(src)), "surround_neurons_recruited": n_src,
                "exc_spikes_in": int(out["exc_per_ms"].sum()), "inh_spikes_in": int(out["inh_per_ms"].sum()),
                "max_exc_per_ms": int(out["exc_per_ms"].max()), "max_inh_per_ms": int(out["inh_per_ms"].max()),
                "max_exc_quanta_per_ms": int(out["exc_quanta_per_ms"].max()), "max_inh_quanta_per_ms": int(out["inh_quanta_per_ms"].min()),
                "total_abs_quanta": int(np.abs(out["exc_quanta_per_ms"]).sum() + np.abs(out["inh_quanta_per_ms"]).sum()),
                "per_neuron_max_abs_quanta_per_step": per_neuron_max.tolist()})
    return res


# ----------------------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------------------
def run_words(ch, sim, words: list[int], expected: list[int], params: Params, max_steps_per_tx: int = 12000) -> tuple[list, list, dict]:
    """One batch of words through one simulator (a RefSim on the isolated image or a CircuitView
    on the full graph) with the channel's own harness. Returns the records, per record what the
    output register reached (rail taps lit, Q neurons fired, arithmetic neurons fired) and the
    circuit's spikes in its window, and the harness's stats."""
    from ..protocol.run import run_transactions

    net = ch.net
    q_taps = [x for pair in ch.consumer.rail_taps for x in pair]
    q_prefix = net.roles[ch.consumer.completion.u].split(".")[0] + "."
    rs, _, st = run_transactions(ch, params, words, expected=expected, sim=sim, max_steps_per_tx=max_steps_per_tx)
    ev = sim.trace.events
    reached = []
    for r in rs:
        end = r.ready_step if r.ready_step is not None else sim.step_index
        win = ev["neuron"][(ev["step"] >= r.load_step) & (ev["step"] <= end)]
        fired = set(win.tolist())
        reached.append({"q_bits_lit": sum(1 for x in q_taps if x in fired),
                        "q_neurons_fired": sum(1 for x in fired if net.roles[x].startswith(q_prefix)),
                        "arithmetic_fired": sum(1 for x in fired if net.roles[x].startswith("add.")),
                        "circuit_spikes": int(len(win)), "end_step": int(end)})
    return rs, reached, st


def tally(recs: list, reached: list, n: int, params: Params, spikes: int) -> dict:
    """The H1 table's tallies over a channel's transaction records."""
    completed = sum(1 for r in recs if r.ready_step is not None)
    correct = sum(1 for r in recs if r.status == "valid" and r.decoded == r.word)
    wrong = sum(1 for r in recs if r.status == "valid" and r.decoded is not None and r.decoded != r.word)
    faults = sum(1 for r in recs if r.status == "fault")
    timeouts = sum(1 for r in recs if r.status == "timeout")
    no_accept = sum(1 for r in recs if r.accept_step is None)
    return {"n": n, "attempted": len(recs), "correct": correct, "wrong": wrong, "faults": faults, "timeouts": timeouts,
            "no_accept": no_accept, "completed": completed, "missing_completion": n - completed,
            "fault_spikes": int(sum(r.fault_spikes for r in recs)), "timeout_spikes": int(sum(r.timeout_spikes for r in recs)),
            "accept_ms": [round(a) for a in ((r.accept_step - r.load_step) * params.dt for r in recs if r.accept_step is not None)],
            "cycle_ms": [round((r.ready_step - r.load_step) * params.dt) for r in recs if r.ready_step is not None],
            "total_spikes": int(spikes), "statuses": [r.status for r in recs], "reached": reached, "records": recs}


def simulate_channel(ch, image: Image, words: list[int], expected: list[int], params: Params | None = None,
                     max_steps_per_tx: int = 12000, fresh_each: bool = True, topology: Topology | None = None,
                     index_map=None) -> dict:
    """Drive the channel's transactions through the image with the channel's own harness
    (protocol.run.run_transactions: same injection, same decode). By default on the image's own
    isolated topology, whose indices are the netlist's; with `topology` and `index_map` (a
    `FullGraph`'s) on the whole brain, through a CircuitView, so the harness still injects and
    decodes by designed index. `fresh_each`: every word in its own simulator from rest, so a word
    whose transaction never completes does not stop the words after it (the harness stops at the
    first transaction without READY); False chains them in one simulator, which also tests the
    reset between words. Returns the records and the tallies the H1 table reports."""
    params = params or ch.net.params
    topology = image.topology if topology is None else topology
    recs, reached, spikes = [], [], 0
    batches = [([w], [x]) for w, x in zip(words, expected)] if fresh_each else [(list(words), list(expected))]
    for ws, xs in batches:
        sim = circuit_sim(topology, params, index_map)
        # the harness is handed a made simulator and so skips its own power-on injection
        # (protocol.run.run_transactions): a flip-flop channel needs its pulse here, by designed index
        power_on(ch, sim)
        rs, re, st = run_words(ch, sim, ws, xs, params, max_steps_per_tx)
        recs += rs
        reached += re
        spikes += st["total_spikes"]
    return tally(recs, reached, len(words), params, spikes)


BROADCAST = ("Q.reset_inh", "P.reset_inh", "P.wd.cancel_inh")


def h1_conditions(policy: Policy = Policy()) -> dict:
    """The five conditions of the H1 report: (selector of missing edges, policy)."""
    base = {k: v for k, v in policy.__dict__.items()}
    zero = Policy(**{**base, "zero_parasitic": True})
    kept = Policy(**{**base, "zero_parasitic": False})
    fanout = any_of(from_sources(BROADCAST), of_class("edge_inh -> edge"), from_sources(("add.actd.d10",)))
    return {
        "A": ("all designed edges (missing ones as Profile 3), parasitic zeroed", all_missing, zero),
        "B": ("carried edges only", none_missing, zero),
        "C": ("carried + the three broadcast neurons' missing edges", from_sources(BROADCAST), zero),
        "D": ("C + missing relay-inhibitor and actd.d10 fan-out edges", fanout, zero),
        "E": ("A with parasitic edges kept at anatomical weights", all_missing, kept),
        "F": ("all missing edges except the three broadcast neurons' (the logic completed, no resets)",
              lambda e: e.src_role not in BROADCAST, zero),
    }


def adder_cases(width: int, n_cases: int, seed: int = 0) -> tuple[list, list, list]:
    """`n_cases` random additions of two `width`-bit operands and a carry-in: (cases, producer
    words, expected sums). The channel's own `width` is the producer's 2 * width + 1 rails."""
    from ..lib.adder import operand_word

    rng = np.random.default_rng(seed)
    cases = [(int(rng.integers(0, 1 << width)), int(rng.integers(0, 1 << width)), int(rng.integers(0, 2))) for _ in range(n_cases)]
    return cases, [operand_word(a, b, c, width) for a, b, c in cases], [a + b + c for a, b, c in cases]


def run_h1_conditions(ch, m: MCNS, placement: Placement, width: int = 4, n_cases: int = 50, seed: int = 0,
                      policy: Policy = Policy(), conditions: dict | None = None, verbose: bool = True) -> dict:
    """Build the image for each condition and run `n_cases` random `width`-bit additions through it."""
    _, words, expected = adder_cases(width, n_cases, seed)
    out = {}
    for name, (desc, select, pol) in (conditions or h1_conditions(policy)).items():
        img = build_image(ch.net, m, placement, pol, profile3=select)
        res = simulate_channel(ch, img, words, expected, fresh_each=True)
        res.pop("records")
        if res["correct"] == n_cases:  # chained as well: the reset between words is part of computing
            chained = simulate_channel(ch, img, words, expected, fresh_each=False)
            res["chained_correct"] = chained["correct"]
            res["chained_cycle_ms"] = chained["cycle_ms"]
        res["profile3_edges"] = img.counts["profile3_edges"]
        res["profile3_by_class"] = img.counts["profile3_by_class"]
        res["parasitic_kept"] = img.counts["parasitic_kept"]
        res["description"] = desc
        out[name] = res
        if verbose:
            print(f"[h1] {name} ({desc}): correct {res['correct']}/{n_cases}, wrong {res['wrong']}, faults {res['faults']}, "
                  f"timeouts {res['timeouts']}, no ACCEPT {res['no_accept']}, completions missing {res['missing_completion']}, "
                  f"Profile 3 edges {res['profile3_edges']}, parasitic kept {res['parasitic_kept']}, "
                  f"chained {res.get('chained_correct')}, reached {res['reached'][:3]}", flush=True)
    return out
