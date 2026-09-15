"""Stage H1: what splitting the broadcast neurons into trees buys in the fly wiring.

The placed adder (docs/h1_placement.md) carries 61-71 % of its edges; the three broadcast
neurons Q.reset_inh (115 targets), P.reset_inh (41) and P.wd.cancel_inh (102) carry a few
percent of theirs because no inhibitory MCNS neuron reaches that many targets at the needed
synapse counts. `Netlist.split_hubs(max_fanout)` replaces each by a tree of identical copies
(same inputs, same spikes, no added hop), so every member needs only max_fanout targets.

For max_fanout in (8, 16) and k_max in (4, 8): split the adder netlist, place it
(`place_netlist`, restarts=4, time_limit_s=300, seed=0), build the image and run conditions
  A: every missing edge added, parasitic zeroed (the mapping computes)
  B: carried edges only (does the bare placement compute a first sum? a second?)
  F: all missing edges except the hub class (the logic completed, the resets left as placed)
with 20 additions each. Edge classes: "hub" = an edge out of a broadcast neuron or one of
its copies (the former hub edges, now spread over the tree), "tree_in" = the inputs the split
duplicated onto copies (trigger / relay -> copy; the copies of those in turn), "hard" =
everything else. Writes docs/h1_split.json; `python -m drosophilos.bench.h1_split` reproduces it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..connectome.embed_h0 import Policy
from ..connectome.embed_image import BROADCAST, build_image, none_missing, run_h1_conditions
from ..connectome.embed_netlist import place_netlist, role_key
from ..connectome.mcns import MCNS, load_mcns
from ..lib.adder import build_adder_channel
from ..sim.model import Params

MAX_FANOUTS = (8, 16)
K_MAX_VALUES = (4, 8)
_COPY = re.compile(r"(\.c\d+)+$")


def root_role(role: str) -> str:
    """'Q.reset_inh.c3.c1' -> 'Q.reset_inh'."""
    return _COPY.sub("", role)


def edge_class(net, e: int, nnz0: int) -> str:
    if root_role(net.roles[net.src[e]]) in BROADCAST:
        return "hub"
    return "tree_in" if e >= nnz0 else "hard"


def run_one(max_fanout: int, k_max: float, m: MCNS, verbose: bool = True, restarts: int = 4, time_limit_s: float = 300.0,
            n_cases: int = 20) -> dict:
    policy = Policy(k_max=float(k_max))
    ch = build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100)
    net = ch.net
    n0, nnz0 = net.n, net.nnz
    copies = net.split_hubs(max_fanout)
    roles = net.roles
    hub_ids = {roles.index(h) for h in BROADCAST}
    tree_of = {h: [h] + copies.get(h, []) for h in hub_ids}

    t0 = time.time()
    pl = place_netlist(net, m, policy=policy, restarts=restarts, time_limit_s=time_limit_s, seed=0, verbose=False)
    search_seconds = time.time() - t0

    missing_pairs = {(s, d) for s, d, _, _ in pl.missing}
    carried = [(int(s), int(d)) not in missing_pairs and int(s) in pl.mapping and int(d) in pl.mapping
               for s, d in zip(net.src, net.dst)]
    by_class: dict = {}
    missing_by_class: dict = {}  # by root roles (the placer's own role_key reads a copy 'x.c3' as 'c')
    for e in range(net.nnz):
        c = by_class.setdefault(edge_class(net, e, nnz0), [0, 0])
        c[0] += carried[e]
        c[1] += 1
        s, d = net.src[e], net.dst[e]
        cls = f"{role_key(root_role(roles[s]))} -> {role_key(root_role(roles[d]))}" + (" (tree input)" if e >= nnz0 else "")
        mc = missing_by_class.setdefault(cls, [0, 0])
        mc[0] += not carried[e]
        mc[1] += 1
    # per broadcast neuron: how the tree fared
    hubs: dict = {}
    for h, tree in tree_of.items():
        outs = [e for e in range(net.nnz) if net.src[e] in tree]
        ins = [e for e in range(net.nnz) if net.dst[e] in tree]
        members_placed = sum(1 for x in tree if x in pl.mapping)
        per_member = {roles[x]: [sum(carried[e] for e in outs if net.src[e] == x), sum(1 for e in outs if net.src[e] == x),
                                 int(x in pl.mapping)] for x in tree}
        hubs[roles[h]] = {
            "members": len(tree), "members_placed": members_placed,
            "out_carried": sum(carried[e] for e in outs), "out_total": len(outs),
            "out_carried_original": sum(carried[e] for e in outs if net.src[e] == h),
            "out_carried_copies": sum(carried[e] for e in outs if net.src[e] != h),
            "in_carried": sum(carried[e] for e in ins), "in_total": len(ins),
            "in_carried_copies": sum(carried[e] for e in ins if net.dst[e] != h),
            "in_total_copies": sum(1 for e in ins if net.dst[e] != h),
            "copies_with_all_inputs": sum(1 for x in tree if x != h and all(carried[e] for e in ins if net.dst[e] == x)),
            "per_member": per_member,
        }

    hub_sel = lambda e: root_role(e.src_role) not in BROADCAST  # noqa: E731
    zero = Policy(k_max=float(k_max), zero_parasitic=True)
    conditions = {
        "A": ("all designed edges (missing ones as Profile 3), parasitic zeroed", lambda e: True, zero),
        "B": ("carried edges only", none_missing, zero),
        "F": ("all missing edges except the hub class (the broadcast neurons' and their copies')", hub_sel, zero),
    }
    t1 = time.time()
    res = run_h1_conditions(ch, m, pl, width=4, n_cases=n_cases, seed=0, conditions=conditions, verbose=verbose)
    sim_seconds = time.time() - t1
    img_A = build_image(net, m, pl, zero, profile3=lambda e: True)

    out = {
        "max_fanout": max_fanout, "k_max": k_max,
        "netlist": {"neurons": net.n, "neurons_unsplit": n0, "synapses": net.nnz, "synapses_unsplit": nnz0,
                    "copies": {roles[k]: len(v) for k, v in copies.items()},
                    "max_out_degree": int(net.out_degrees().max())},
        "placement": {
            "carried": pl.carried, "carried_fraction": round(pl.carried / net.nnz, 4),
            "by_class": {k: {"carried": v[0], "total": v[1]} for k, v in sorted(by_class.items())},
            "neurons_placed": len(pl.mapping), "neurons_total": net.n,
            "parasitic_to_zero": pl.parasitic, "parasitic_zeroed_image": img_A.counts["parasitic_zeroed"],
            "search_seconds": round(search_seconds, 1), "sim_seconds": round(sim_seconds, 1),
        },
        "hubs": hubs,
        "conditions": {
            name: {"correct": r["correct"], "n": r["n"], "profile3_edges": r["profile3_edges"],
                   "no_accept": r["no_accept"], "missing_completion": r["missing_completion"],
                   "chained_correct": r.get("chained_correct"), "reached": r["reached"][:3],
                   "profile3_by_class_top": dict(sorted(r["profile3_by_class"].items(), key=lambda kv: -kv[1])[:12])}
            for name, r in res.items()
        },
        "missing_by_class_top": {k: v for k, v in sorted(missing_by_class.items(), key=lambda kv: -kv[1][0])[:15]},
    }
    return out


def run_all(max_fanouts=MAX_FANOUTS, k_max_values=K_MAX_VALUES, verbose: bool = True, path: Path | None = None) -> dict:
    m = load_mcns()
    results = {}
    for mf in max_fanouts:
        for k in k_max_values:
            if verbose:
                print(f"[h1_split] max_fanout = {mf}, k_max = {k}", flush=True)
            results[f"{mf}x{k}"] = run_one(mf, k, m, verbose=verbose)
            if path is not None:
                path.write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "docs" / "h1_split.json"
    run_all(path=path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
