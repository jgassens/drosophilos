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

`simulate_channel` runs a four-phase channel (the adder) on an image with the channel's own
drive and decode; `run_h1_conditions` is the H1 measurement (docs/h1_placement.md).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..lib.netlist import Netlist
from ..sim.model import QUANTA_PER_SYNAPSE, Params, Topology
from ..sim.ref64 import RefSim
from .embed_h0 import Policy
from .embed_netlist import Placement, role_key
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
    carried_edges, scales, p3, omitted = [], {}, [], []
    carried_pairs = set()
    for e, (s, d, q, dl) in enumerate(zip(net.src, net.dst, net.quanta, net.delay)):
        rs, rd = int(real[s]), int(real[d])
        cnt = count(rs, rd)
        req = policy.req_count(q)
        if rs < 0 or rd < 0:
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
            src_t.append(s); dst_t.append(d); q_t.append(int(q)); dl_t.append(int(dl))
            continue
        reason = reasons.get((int(s), int(d)))  # the placement's audit, with the data's detail
        edge = ImageEdge(int(s), int(d), int(q), net.roles[s], net.roles[d], f"{reason}: {detail}" if reason else detail,
                         f"{role_key(net.roles[s])} -> {role_key(net.roles[d])}", cnt)
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
            if rs == rd or (rs, rd) in carried_pairs:
                continue
            s, d = int(hosts[i]), int(hosts[j])
            reason = "under a designed edge (too weak or wrong sign)" if any(a == s and b == d for a, b in zip(net.src, net.dst)) else "not designed"
            parasitic.append(ImageEdge(s, d, int(sign[rs]) * int(cnt) * QUANTA_PER_SYNAPSE, net.roles[s], net.roles[d], reason,
                                       f"{role_key(net.roles[s])} -> {role_key(net.roles[d])}", int(cnt)))
            if not policy.zero_parasitic:
                src_t.append(s); dst_t.append(d); q_t.append(int(sign[rs]) * int(cnt) * QUANTA_PER_SYNAPSE); dl_t.append(params.default_delay_steps)
    topo = Topology.from_edges(n, src_t, dst_t, q_t, dl_t)

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
        "designed_edges": net.nnz, "carried": len(carried_edges),
        "profile3_edges": len(p3), "profile3_by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "omitted_missing_edges": len(omitted), "omitted_by_class": dict(sorted(omitted_by_class.items(), key=lambda kv: -kv[1])),
        "parasitic": len(parasitic), "parasitic_zeroed": len(parasitic) if policy.zero_parasitic else 0,
        "parasitic_kept": 0 if policy.zero_parasitic else len(parasitic),
        "parasitic_under_designed": sum(1 for e in parasitic if e.reason.startswith("under")),
        "parasitic_abs_quanta": int(sum(abs(e.quanta) for e in parasitic)),
        "scales": hist, "scale_max": round(max(scales.values()), 3) if scales else None,
        "topology_edges": topo.nnz,
    }
    img = Image(n, topo, real, bodies, synthetic, carried_edges, scales, p3, omitted, parasitic, policy.zero_parasitic, counts)
    if placement.carried and placement.carried != len(carried_edges):
        counts["placement_carried_mismatch"] = placement.carried  # the audit and the image disagree: report it
    img.manifest = image_manifest(net, m, img, policy, params)
    return img


def image_manifest(net: Netlist, m: MCNS, img: Image, policy: Policy, params: Params) -> dict:
    """A manifest in the style of connectome/manifest.py: the four graphs (original, retained,
    active-nonzero, silencing), the parameter edits (carried edges rescaled, parasitic edges
    zeroed) and the structural edits (Profile 3 edges and synthetic neurons), plus the counts."""
    edits = list(img.carried)
    if img.zero_parasitic:
        edits += [ParameterEdit("weight", int(img.bodies[e.src]), int(img.bodies[e.dst]), e.quanta, 0, "documented zero weight",
                                f"parasitic edge among circuit neurons ({e.reason})") for e in img.parasitic]
    structural = [{"kind": "add_edge", "pre": int(img.bodies[e.src]), "post": int(img.bodies[e.dst]), "pre_role": e.src_role,
                   "post_role": e.dst_role, "quanta": e.quanta, "class": e.cls, "reason": e.reason} for e in img.profile3]
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
    d = json.loads(Path(path).read_text())
    roles = {r: i for i, r in enumerate(net.roles)}
    mapping = {roles[role]: int(m.index_of(np.array([v["bodyId"]]))[0]) for role, v in d["mapping"].items()}
    missing = [(roles[e["src"]], roles[e["dst"]], int(e["quanta"]), e["reason"]) for e in d.get("missing", [])]
    unplaced = [roles[r] for r in d.get("unplaced", [])]
    return Placement(mapping, unplaced, int(d.get("summary", {}).get("carried", 0)), missing,
                     int(d.get("summary", {}).get("parasitic_to_zero", 0)), 0.0, strategy=d.get("strategy", "loaded"))


# ----------------------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------------------
def simulate_channel(ch, image: Image, words: list[int], expected: list[int], params: Params | None = None,
                     max_steps_per_tx: int = 12000, fresh_each: bool = True) -> dict:
    """Drive the channel's transactions through the image with the channel's own harness
    (protocol.run.run_transactions: same injection, same decode; the image's indices are the
    netlist's). `fresh_each`: every word in its own simulator from rest, so a word whose
    transaction never completes does not stop the words after it (the harness stops at the
    first transaction without READY); False chains them in one simulator, which also tests
    the reset between words. Returns the records and the tallies the H1 table reports."""
    from ..protocol.run import run_transactions

    params = params or ch.net.params
    net = ch.net
    q_taps = [x for pair in ch.consumer.rail_taps for x in pair]
    q_prefix = net.roles[ch.consumer.completion.u].split(".")[0] + "."
    recs, reached, spikes = [], [], 0
    batches = [([w], [x]) for w, x in zip(words, expected)] if fresh_each else [(list(words), list(expected))]
    for ws, xs in batches:
        sim = RefSim(image.topology, params)
        rs, sim, st = run_transactions(ch, params, ws, expected=xs, sim=sim, max_steps_per_tx=max_steps_per_tx)
        recs += rs
        spikes += st["total_spikes"]
        ev = sim.trace.events
        for r in rs:
            end = r.ready_step if r.ready_step is not None else sim.step_index
            win = ev["neuron"][(ev["step"] >= r.load_step) & (ev["step"] <= end)]
            fired = set(win.tolist())
            reached.append({"q_bits_lit": sum(1 for x in q_taps if x in fired),
                            "q_neurons_fired": sum(1 for x in fired if net.roles[x].startswith(q_prefix)),
                            "arithmetic_fired": sum(1 for x in fired if net.roles[x].startswith("add."))})
    n = len(words)
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
