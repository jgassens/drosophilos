"""Stage H1: the placement at the next scale -- one resident kernel cell -- and what limits it.

The 4-bit adder (614 neurons, 1,146 edges; docs/h1_placement.md) is placed on MCNS with 61 % of
its edges carried. The next real target is a kernel: `build_pipeline` (drosophilos/lib/kernel.py)
turns a loop body into cells, each an input/master register with a datapath, and the smallest
one -- a single 8-bit ADD cell with its input register, `c1 = input + 1` -- is five times the
adder (3,170 neurons, 5,768 synapse entries). This bench places that cell with `place_netlist`
(k_max 4, 2 restarts, 1,800 s, seed 0) as built and after `Netlist.split_hubs(max_fanout=16)`,
and records what the search reads (latches, relays, chains, hubs with their fan-outs, the
static candidate domains against the connectome's strong-edge core) and what it gets (carried
edges by class, neurons placed, parasitic edges, seconds, peak memory).

No simulation. Each placement runs in a child process whose resident memory the parent polls
(`ps`); a child above MEMORY_LIMIT_GB is killed and the bench falls back to the 4-bit cell
(`_Design`'s static setup is one dense boolean row per designed neuron, n x 166,700 bytes: 0.5 GB
for the 8-bit cell before the anatomy's own caches). Writes docs/h1_cell.json;
`python -m drosophilos.bench.h1_cell` reproduces it (`--n 4` for the small cell only).
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..sim.model import Params

WIDTH = 8
FALLBACK_WIDTH = 4
K_MAX = 4.0
RESTARTS = 2
TIME_LIMIT_S = 1800.0
MAX_FANOUT = 16
MEMORY_LIMIT_GB = 8.0
HUB_DEG = 20
PYTHON = "/Users/jeremiahgassensmith/programming/drosophilos/.venv/bin/python"
_COPY = re.compile(r"(\.c\d+)+$")


def root_role(role: str) -> str:
    """'c1.Q.reset_inh.c3.c1' -> 'c1.Q.reset_inh'."""
    return _COPY.sub("", role)


def cell_netlist(n: int):
    """The smallest real kernel: one ADD cell, `c1 = input + k1`, with its input register."""
    from ..lib.kernel import build_pipeline
    pl = build_pipeline(Params(), n, [{"name": "c1", "op": "ADD", "a": "input", "b": ("const", "k1")}],
                        consts={"k1": 1}, outputs=["c1"])
    return pl.net


def describe(net, policy=None, hub_deg: int = HUB_DEG) -> dict:
    """What the search reads off the netlist: edge classes and motifs (the placer's own `_Design`)."""
    from ..connectome.embed_h0 import Policy
    from ..connectome.embed_netlist import _Design
    policy = policy or Policy(k_max=K_MAX)
    D = _Design(net, policy, hub_deg)
    kinds: dict = {}
    chain_lengths: list = []
    relay_inhibitors = 0
    for mo in D.motifs:
        kinds[mo.kind] = kinds.get(mo.kind, 0) + 1
        if mo.kind == "chain":
            chain_lengths.append(len(mo.nodes))
        elif mo.kind == "relay":
            relay_inhibitors += len(mo.nodes) - 1
    outdeg = np.bincount(D.src, minlength=net.n)
    indeg = np.bincount(D.dst, minlength=net.n)
    hubs = sorted(({"role": net.roles[h], "out": int(outdeg[h]), "in": int(indeg[h]),
                    "sign": int(D.sign[h])} for h in np.flatnonzero(D.hub)), key=lambda x: -x["out"])
    pairs = {(int(s), int(d)) for s, d in zip(D.src, D.dst)}
    reqs = sorted({int(r) for r in D.req})
    return {
        "neurons": int(net.n), "synapses": int(net.nnz), "edges": len(pairs),
        "hard": int(D.hard.sum()), "hub_soft": int(D.soft.sum()), "impossible": int(D.impossible.sum()),
        "mixed_sign": [net.roles[d] for d in D.mixed],
        "motifs": kinds, "latch_members": 2 * kinds.get("latch", 0), "relay_inhibitors": relay_inhibitors,
        "chain_lengths": sorted(chain_lengths, reverse=True),
        "hubs": hubs, "max_out_degree": int(outdeg.max()), "max_in_degree": int(indeg.max()),
        "required_counts": reqs,  # anatomical synapses a designed edge needs, k_max applied
    }


def domain_stats(net, m, policy) -> dict:
    """The static candidate sets (the search's layer 2, after arc consistency) against the
    connectome: how many distinct real neurons the cell's latch members, relays and chains can
    draw on, and how many real mutual pairs exist at the latch threshold. This is the
    resource the question is about -- whether ten times the adder runs the strong-edge core out."""
    from ..connectome.embed_netlist import _Anatomy, _Design, _Search, role_key

    class _Probe(_Search):
        """The search's own static setup, with its arc consistency instrumented: which designed
        neuron lost its last candidate to which neighbour, and whether that neighbour still had
        candidates (a real conflict) or was already empty (the emptiness spreading)."""

        def _arc_consistency(self, rounds: int):
            Dd, Aa = self.D, self.A
            self.first_empty: dict = {}
            self.empty_after_round: list = []
            for rnd in range(rounds):
                changed = False
                for e in np.flatnonzero(Dd.hard):
                    s_, d_, r = int(Dd.src[e]), int(Dd.dst[e]), int(Dd.req[e])
                    Ar, ArT = Aa.A(r)
                    for x, other, M in ((s_, d_, Ar), (d_, s_, ArT)):
                        dense = np.zeros(Aa.n, np.int32)
                        dense[self.dom[other]] = 1
                        supp = (M @ dense) > 0
                        new = self.dom[x][supp[self.dom[x]]]
                        if len(new) < len(self.dom[x]):
                            if len(new) == 0:
                                self.first_empty[x] = (rnd, other, int(len(self.dom[other])), r)
                            self.dom[x] = new
                            changed = True
                self.empty_after_round.append(int(sum(len(d) == 0 for d in self.dom)))
                if not changed:
                    break

    t0 = time.time()
    A = _Anatomy(m, policy)
    D = _Design(net, policy, HUB_DEG)
    S = _Probe(D, A, np.random.default_rng(0), hub_first=False, cand_cap=48, chain_budget=20_000,
               max_backtracks=8, backtrack_depth=3, deadline=time.time() + 60.0)
    setup_s = time.time() - t0
    out: dict = {"setup_seconds": round(setup_s, 1)}
    roots = [(net.roles[x], net.roles[o], sz, r) for x, (_, o, sz, r) in S.first_empty.items() if sz > 0]
    out["arc_consistency"] = {
        "empty_domains_after_round": S.empty_after_round,
        "emptied_by_a_neighbour_with_candidates": len(roots),  # the real conflicts; the rest is spread
        "emptied_by_an_empty_neighbour": len(S.first_empty) - len(roots),
        "root_conflicts": [{"neuron": x, "neighbour": o, "neighbour_candidates": sz, "required_count": r} for x, o, sz, r in roots[:40]],
        "root_conflict_classes": dict(sorted(
            {f"{role_key(root_role(x))} <- {role_key(root_role(o))}": sum(1 for y, oo, _, _ in roots if (role_key(root_role(y)), role_key(root_role(oo))) == (role_key(root_role(x)), role_key(root_role(o)))) for x, o, _, _ in roots}.items())),
    }
    by_kind: dict = {}
    for mo in D.motifs:
        b = by_kind.setdefault(mo.kind, {"nodes": 0, "domain_sizes": [], "union": set(), "empty": 0})
        for x in mo.nodes:
            b["nodes"] += 1
            b["domain_sizes"].append(int(len(S.dom[x])))
            b["union"].update(S.dom[x].tolist())
            b["empty"] += int(len(S.dom[x]) == 0)
    for k, b in by_kind.items():
        ds = np.array(b["domain_sizes"]) if b["domain_sizes"] else np.zeros(1)
        out[k] = {"designed_nodes": b["nodes"], "distinct_real_candidates": len(b["union"]),
                  "domain_median": float(np.median(ds)), "domain_min": int(ds.min()), "domain_max": int(ds.max()),
                  "empty_domains": b["empty"]}
    # the latch resource itself: real mutual pairs at the loop threshold, and how many are
    # available to the cell's latch members after the context filters
    latch_thr = min(int(D.req[e]) for mo in D.motifs if mo.kind == "latch" for e in mo.edges) if by_kind.get("latch") else None
    if latch_thr is not None:
        mask, pairs = A.mutual(latch_thr)
        latch_union = by_kind["latch"]["union"]
        in_dom = np.array([a in latch_union and b in latch_union for a, b in pairs]) if len(pairs) else np.zeros(0, bool)
        out["latch_threshold"] = latch_thr
        out["real_mutual_pairs"] = int(len(pairs))
        out["real_mutual_members"] = int(mask.sum())
        out["mutual_pairs_both_in_some_latch_domain"] = int(in_dom.sum())
        out["designed_latches"] = int(len([1 for mo in D.motifs if mo.kind == "latch"]))
    return out


def latch_capacity(m, k_max_values=(4, 8, 16), loop_quanta: int = 3621) -> dict:
    """The connectome's supply of latches, per weight bound: real excitatory neurons in a mutual
    pair at the loop threshold, the pairs, and -- the number that bounds any one-neuron-per-role
    placement -- the largest set of pairwise disjoint pairs (a maximum matching; 1,223 pairs over
    771 neurons cannot host 1,223 latches). Also the largest strongly connected component of the
    >= threshold cholinergic graph (the "strong-edge core")."""
    import scipy.sparse.csgraph as cg
    from ..connectome.embed_h0 import Policy
    from ..connectome.embed_netlist import _Anatomy
    out: dict = {}
    for k in k_max_values:
        pol = Policy(k_max=float(k))
        A = _Anatomy(m, pol)
        thr = pol.req_count(loop_quanta)
        mask, pairs = A.mutual(thr)
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_edges_from(map(tuple, pairs.tolist()))
            disjoint, how = len(nx.max_weight_matching(G, maxcardinality=True)), "maximum matching"
        except ImportError:  # greedy lower bound
            used = np.zeros(A.n, bool)
            disjoint = 0
            for a, b in pairs.tolist():
                if not used[a] and not used[b]:
                    used[a] = used[b] = True
                    disjoint += 1
            how = "greedy matching (lower bound; networkx not installed)"
        Aexc, _ = A.A(thr, "exc", "exc")
        _, lab = cg.connected_components(Aexc, directed=True, connection="strong")
        out[str(k)] = {"threshold": thr, "mutual_pairs": int(len(pairs)), "mutual_members": int(mask.sum()),
                       "max_disjoint_pairs": int(disjoint), "matching": how, "strong_core": int(np.bincount(lab).max())}
    return out


def edge_classes(net, D, nnz0: int | None, former_hubs: set) -> dict:
    """Distinct (src, dst) pair -> class. Placer classes: hard / hub (soft: a hub's many edges) /
    impossible (mixed sign). After a split: 'hub' also covers every edge out of a former hub's
    tree (the fan-out the split spread over copies) and 'tree_in' the inputs it duplicated."""
    cls: dict = {}
    for e in range(len(D.src)):
        key = (int(D.src[e]), int(D.dst[e]))
        if key in cls:
            continue
        if D.impossible[e]:
            c = "impossible"
        elif root_role(net.roles[key[0]]) in former_hubs:
            c = "hub"
        elif D.soft[e]:
            c = "hub"
        elif nnz0 is not None and e >= nnz0:
            c = "tree_in"
        else:
            c = "hard"
        cls[key] = c
    return cls


def split_at_smallest_bound(net, max_fanout: int, step: int = 4, cap: int = 64):
    """`split_hubs(max_fanout)`, or, when the transform reports a cycle (a hub that feeds its own
    driver: the 8-bit cell's fault latch drives the register's reset trigger, whose inhibitor
    inhibits the fault latch, so bounding any of the three at 16 raises the others without end),
    the smallest larger bound in steps of `step` that terminates; a rejected split leaves the
    netlist untouched. Returns (copies, bound, note)."""
    errors = []
    bound = max_fanout
    while bound <= cap:
        try:
            copies = net.split_hubs(bound)
        except ValueError as e:
            errors.append(f"max_fanout {bound}: {e}")
            bound += step
            continue
        note = None if not errors else "; ".join(errors) + f"; split at max_fanout {bound} instead"
        return copies, bound, note
    raise ValueError("; ".join(errors))


def place_cell(n: int, split: bool, m=None, restarts: int = RESTARTS, time_limit_s: float = TIME_LIMIT_S,
               k_max: float = K_MAX, max_fanout: int = MAX_FANOUT, verbose: bool = True, with_domains: bool = True) -> dict:
    """Build the cell (split or not), place it, and return the record docs/h1_cell.json stores."""
    from ..connectome.embed_h0 import Policy
    from ..connectome.embed_netlist import _Design, place_netlist
    from ..connectome.mcns import load_mcns
    policy = Policy(k_max=float(k_max))
    net = cell_netlist(n)
    nnz0 = None
    former_hubs: set = set()
    copies: dict = {}
    D0 = _Design(net, policy, HUB_DEG)
    outdeg0 = np.bincount(D0.src, minlength=net.n)
    unsplit_hubs = {net.roles[h] for h in np.flatnonzero(D0.out_hub)}
    split_note = None
    if split:
        nnz0 = net.nnz
        copies, max_fanout, split_note = split_at_smallest_bound(net, max_fanout)
        former_hubs = {net.roles[k] for k in copies if net.roles[k] in unsplit_hubs}
    desc = describe(net, policy)
    if m is None:
        m = load_mcns()
    rec: dict = {"width": n, "split": split, "max_fanout": max_fanout if split else None, "k_max": k_max,
                 "restarts": restarts, "time_limit_s": time_limit_s, "netlist": desc}
    if split:
        rec["max_fanout"] = max_fanout
        rec["split_note"] = split_note
        rec["split_copies"] = {net.roles[k]: len(v) for k, v in sorted(copies.items(), key=lambda kv: -len(kv[1]))}
        rec["split_neurons_added"] = sum(len(v) for v in copies.values())
        rec["split_synapses_added"] = net.nnz - nnz0
        rec["split_originals_out_degree"] = {net.roles[k]: int(outdeg0[k]) for k in copies}
    if with_domains:
        rec["domains"] = domain_stats(net, m, policy)
        if verbose:
            print(f"[h1_cell] domains: {rec['domains']}", flush=True)
    t0 = time.time()
    pl = place_netlist(net, m, policy=policy, restarts=restarts, time_limit_s=time_limit_s, seed=0, verbose=verbose)
    search_s = time.time() - t0
    D = _Design(net, policy, HUB_DEG)
    cls = edge_classes(net, D, nnz0, former_hubs)
    missing = {(int(s), int(d)) for s, d, _, _ in pl.missing}
    by_class: dict = {}
    for key, c in cls.items():
        b = by_class.setdefault(c, {"carried": 0, "total": 0})
        b["total"] += 1
        b["carried"] += key not in missing
    for b in by_class.values():
        b["fraction"] = round(b["carried"] / max(1, b["total"]), 4)
    # each hub (or its tree): outputs carried
    hub_out: dict = {}
    roots = {}
    for d in range(net.n):
        roots.setdefault(root_role(net.roles[d]), []).append(d)
    for h in desc["hubs"] if not split else [{"role": r} for r in sorted(former_hubs)]:
        tree = roots[root_role(h["role"])] if split else [net.roles.index(h["role"])]
        outs = [k for k in cls if k[0] in tree]
        ins = [k for k in cls if k[1] in tree]
        hub_out[h["role"]] = {"members": len(tree), "members_placed": sum(1 for x in tree if x in pl.mapping),
                              "out_carried": sum(1 for k in outs if k not in missing), "out_total": len(outs),
                              "in_carried": sum(1 for k in ins if k not in missing), "in_total": len(ins)}
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_gb = maxrss / 1e9 if sys.platform == "darwin" else maxrss / 1e6  # macOS reports bytes, Linux KB
    rec["placement"] = {
        "carried": pl.carried, "edges": pl.edges, "carried_fraction": round(pl.carried / max(1, pl.edges), 4),
        "by_class": dict(sorted(by_class.items())),
        "neurons_placed": len(pl.mapping), "neurons_total": net.n, "unplaced_by_role": {
            k: v[1] for k, v in sorted(pl.by_role.items(), key=lambda kv: -kv[1][1]) if v[1]},
        "parasitic_to_zero": pl.parasitic,
        "search_seconds": round(search_s, 1), "placement_seconds": round(pl.seconds, 1), "strategy": pl.strategy,
        "order_note": pl.order_note, "peak_memory_gb": round(maxrss_gb, 2),
        "motifs": pl.motifs, "motifs_search": pl.motifs_search,
        "missing_by_class_top": dict(list(pl.missing_by_class.items())[:20]),
        "hubs": hub_out,
        "exposure": pl.exposure,
    }
    return rec


# ---- child process with a memory watch --------------------------------------------------
def _rss_gb(pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) * 1024 / 1e9 if out else 0.0
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def run_child(n: int, split: bool, out: Path, restarts: int, time_limit_s: float, memory_limit_gb: float = MEMORY_LIMIT_GB,
              verbose: bool = True) -> dict:
    """Run place_cell in a child, polling its resident memory; kill it above the limit or at
    1.5 x the time limit plus setup. Returns the record, or {"failed": reason, ...}."""
    cmd = [PYTHON, "-m", "drosophilos.bench.h1_cell", "--worker", "--n", str(n), "--restarts", str(restarts),
           "--time-limit", str(time_limit_s), "--out", str(out)] + (["--split"] if split else [])
    t0 = time.time()
    hard_deadline = t0 + 1.5 * time_limit_s + 600.0
    p = subprocess.Popen(cmd, stdout=None if verbose else subprocess.DEVNULL, stderr=None if verbose else subprocess.DEVNULL)
    peak = 0.0
    reason = None
    while p.poll() is None:
        time.sleep(2.0)
        peak = max(peak, _rss_gb(p.pid))
        if peak > memory_limit_gb:
            reason = f"memory {peak:.2f} GB above {memory_limit_gb} GB"
        elif time.time() > hard_deadline:
            reason = f"wall time {time.time() - t0:.0f} s above 1.5 x the limit + 600 s"
        if reason:
            p.kill()
            p.wait()
            break
    wall = time.time() - t0
    if reason is None and p.returncode != 0:
        reason = f"exit code {p.returncode}"
    if reason:
        return {"width": n, "split": split, "failed": reason, "wall_seconds": round(wall, 1), "peak_memory_polled_gb": round(peak, 2)}
    rec = json.loads(out.read_text())
    rec["wall_seconds"] = round(wall, 1)
    rec["peak_memory_polled_gb"] = round(peak, 2)
    return rec


def run_all(n: int = WIDTH, path: Path | None = None, restarts: int = RESTARTS, time_limit_s: float = TIME_LIMIT_S,
            fallback: int | None = FALLBACK_WIDTH, verbose: bool = True) -> dict:
    """The n-bit cell unsplit and split, then the fallback width (the 4-bit cell) unsplit and split
    -- always, since it is cheap and is the one size at which `split_hubs(16)` terminates; a failed
    n-bit run (memory, time, error) is recorded as such and the fallback stands in for it."""
    scratch = (path.parent if path else Path(".")) / ".h1_cell_tmp"
    scratch.mkdir(exist_ok=True)
    results: dict = {"k_max": K_MAX, "restarts": restarts, "time_limit_s": time_limit_s, "max_fanout": MAX_FANOUT,
                     "memory_limit_gb": MEMORY_LIMIT_GB, "runs": {}, "fallback": None}
    widths = [n] + ([fallback] if fallback is not None and fallback != n else [])
    for width in widths:
        for split in (False, True):
            key = f"n{width}_{'split' if split else 'unsplit'}"
            if verbose:
                print(f"[h1_cell] {key}", flush=True)
            rec = run_child(width, split, scratch / f"{key}.json", restarts, time_limit_s, verbose=verbose)
            results["runs"][key] = rec
            if "failed" in rec and width == n and fallback is not None:
                results["fallback"] = f"{key}: {rec['failed']}; the n = {fallback} cell stands in"
                if verbose:
                    print(f"[h1_cell] {results['fallback']}", flush=True)
            if path is not None:
                path.write_text(json.dumps(results, indent=1))
    for f in scratch.glob("*.json"):
        f.unlink()
    scratch.rmdir()
    from ..connectome.mcns import load_mcns
    results["latch_capacity"] = latch_capacity(load_mcns())
    if path is not None:
        path.write_text(json.dumps(results, indent=1))
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=WIDTH)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT_S)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--worker", action="store_true", help="place one configuration and write --out (used by run_all)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args(argv)
    if args.worker:
        rec = place_cell(args.n, args.split, restarts=args.restarts, time_limit_s=args.time_limit)
        args.out.write_text(json.dumps(rec, indent=1))
        print(f"[h1_cell] wrote {args.out}: carried {rec['placement']['carried']} / {rec['placement']['edges']} "
              f"in {rec['placement']['search_seconds']} s, peak {rec['placement']['peak_memory_gb']} GB", flush=True)
        return
    path = args.out or Path(__file__).resolve().parents[2] / "docs" / "h1_cell.json"
    run_all(args.n, path=path, restarts=args.restarts, time_limit_s=args.time_limit,
            fallback=None if args.no_fallback else FALLBACK_WIDTH)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
