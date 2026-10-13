"""Stage H: census MCNS self-connections for configuration F.

Run with ``python -m drosophilos.bench.h_autapse_census --data-dir DATA/mcns``.
The report asks whether a real neuron's anatomical self-connection can carry the
zero-delay inhibitory feedback which configuration F adds to latch members and
to ordered sum-XOR transaction relays.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..connectome.embed_h0 import Policy
from ..connectome.mcns import load_mcns
from ..lib.adder import build_adder_channel
from ..lib.netlist import Drive
from ..protocol.latch import EXPERIMENTAL_AUTAPSE_GAIN
from ..sim.model import Params, QUANTA_PER_SYNAPSE


BINS = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9),
        ("10-19", 10, 19), ("20-49", 20, 49), (">=50", 50, math.inf))


def sign_name(value) -> str:
    """The MCNS sign label used in this report (including missing/zero as unknown)."""
    if pd.isna(value) or value == 0:
        return "unknown"
    return "inhibitory" if value < 0 else "excitatory"


def self_edge_table(pre: np.ndarray, post: np.ndarray, count: np.ndarray,
                    neurons: pd.DataFrame) -> pd.DataFrame:
    """Return annotations for every anatomical self-edge; inputs are MCNS-style arrays."""
    keep = np.asarray(pre) == np.asarray(post)
    indices = np.asarray(pre)[keep].astype(np.int64)
    cols = [c for c in ("bodyId", "type", "superclass", "class", "nt", "predicted_nt", "nt_conf", "sign")
            if c in neurons.columns]
    out = neurons.iloc[indices][cols].copy().reset_index(drop=True)
    out.insert(0, "neuron_index", indices)
    out["synapse_count"] = np.asarray(count)[keep].astype(np.int64)
    out["sign_label"] = [sign_name(x) for x in out.get("sign", pd.Series(0, index=out.index))]
    return out


def count_categories(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows:
        return {}
    return {str(k): int(v) for k, v in rows[column].fillna("unknown").value_counts().items()}


def count_bins(counts: np.ndarray) -> dict[str, int]:
    values = np.asarray(counts)
    return {label: int(((values >= lo) & (values <= hi)).sum()) for label, lo, hi in BINS}


def supply_summary(rows: pd.DataFrame) -> dict:
    return {"total": int(len(rows)), "by_sign": count_categories(rows, "sign_label"),
            "by_transmitter": count_categories(rows, "nt"),
            "by_superclass": count_categories(rows, "superclass"),
            "by_synapse_count_bin": count_bins(rows["synapse_count"].to_numpy())}


def autapse_demands(params: Params = Params(), k_max: float = 4.0) -> dict:
    """Configuration-F self-edge quanta and imported Profile-2 count thresholds."""
    drive = Drive.from_params(params)
    q_latch = -round(EXPERIMENTAL_AUTAPSE_GAIN * drive.loop)
    q_oneshot = -round(4.4 * drive.loop)
    policy = Policy(k_max=k_max)
    return {"k_max": k_max, "quanta_per_synapse": QUANTA_PER_SYNAPSE,
            "drive_loop_quanta": drive.loop,
            "latch_autapse": {"quanta": int(q_latch), "required_synapse_count": policy.req_count(q_latch)},
            "oneshot_transaction": {"quanta": int(q_oneshot), "required_synapse_count": policy.req_count(q_oneshot)}}


def fit_summary(rows: pd.DataFrame, demand: dict) -> dict:
    """Counts of inhibitory self-edges meeting each imported Profile-2 threshold."""
    inh = rows[rows["sign_label"] == "inhibitory"].copy()
    out = {"inhibitory_self_edges": int(len(inh)), "excitatory_self_edges": int((rows["sign_label"] == "excitatory").sum())}
    for name in ("latch_autapse", "oneshot_transaction"):
        req = demand[name]["required_synapse_count"]
        good = inh[inh["synapse_count"] >= req]
        # This is the multiplier needed by every existing inhibitory self-edge.
        k_needed = np.abs(demand[name]["quanta"]) / (inh["synapse_count"].to_numpy() * QUANTA_PER_SYNAPSE) if len(inh) else np.array([])
        out[name] = {"required_synapse_count": int(req), "carriable_self_edges": int(len(good)),
                     "carriable_body_ids": [int(x) for x in good["bodyId"]],
                     "k_max_needed_by_existing_inhibitory_self_edge": [
                         {"bodyId": int(b), "synapse_count": int(c), "k_max": round(float(k), 6)}
                         for b, c, k in zip(inh["bodyId"], inh["synapse_count"], k_needed)]}
    return out


def latch_matches(net, drive: Drive) -> list[tuple[int, int]]:
    """The exact non-mutating latch recognition used by add_experimental_autapses."""
    edges = {(s, d): q for s, d, q in zip(net.src, net.dst, net.quanta)}
    return [(u, v) for (u, v), q in edges.items()
            if q == drive.loop and edges.get((v, u)) == drive.loop
            and net.roles[u].endswith(".u")
            and net.roles[v] == net.roles[u].removesuffix(".u") + ".v"]


def adder_demand() -> dict:
    """Count configuration-F demands on the specified 4-bit flip-flop adder build."""
    channel = build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100, storage="flipflop")
    latches = latch_matches(channel.net, channel.drive)
    # Gates.ripple_adder_ordered makes one sum xor per bit; oneshot=True makes one
    # transaction relay (and thus one self-edge) for each of its two late rails.
    sum_xors, rails_per_sum = 4, 2
    return {"netlist_neurons": channel.net.n, "matching_excitatory_latches": len(latches),
            "latch_autapse_self_edges": 2 * len(latches), "sum_xor2_ordered": sum_xors,
            "rails_per_sum_xor": rails_per_sum, "transaction_self_edges": sum_xors * rails_per_sum}


def raw_self_edge_check(data_dir: Path, known_bodies: np.ndarray) -> dict:
    """Count raw feather self rows and which survive MCNS's both-endpoints-neurons rule."""
    import pyarrow.dataset as ds
    path = data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    dataset = ds.dataset(path, format="feather")
    names = dataset.schema.names
    pre = "body_pre" if "body_pre" in names else next(c for c in names if "pre" in c.lower())
    post = "body_post" if "body_post" in names else next(c for c in names if "post" in c.lower())
    total = kept = 0
    bodies = np.sort(np.asarray(known_bodies))
    for batch in dataset.scanner(columns=[pre, post], batch_size=2_000_000).to_batches():
        a, b = batch.column(0).to_numpy(), batch.column(1).to_numpy()
        same = a == b
        total += int(same.sum())
        x = a[same]
        pos = np.searchsorted(bodies, x)
        ok = (pos < len(bodies)) & (bodies[np.minimum(pos, len(bodies) - 1)] == x)
        kept += int(ok.sum())
    return {"raw_self_edges": total, "raw_self_edges_with_both_bodies_in_neuron_table": kept,
            "excluded_outside_neuron_table": total - kept}


def make_report(data_dir: Path) -> dict:
    m = load_mcns(data_dir=data_dir)
    rows = self_edge_table(m.pre, m.post, m.count, m.neurons)
    demand = autapse_demands()
    return {"stage": "H autapse census", "inputs": {
            "edge_cache": str(data_dir / "mcns_v1.0_edges_minsyn1.npz"),
            "neurons": str(data_dir / "mcns_v1.0_neurons.parquet"),
            "raw_weights": str(data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
            "params": dataclasses.asdict(Params()), "k_max": 4.0},
            "mcns": {"version": m.version, "data_dir": str(data_dir),
            "neuron_rule": "superclass is not null", "edge_rule": "both endpoints are neurons and syn_count >= min_syn"},
            "placement_rule": {"quoted": "0 <= |q| <= k_max * count * QUANTA_PER_SYNAPSE; the presynaptic host transmitter sign must match the designed edge sign.",
                               "implementation": "drosophilos.connectome.embed_h0.Policy.req_count(q)"},
            "supply": {"self_edges": rows.to_dict(orient="records"), "summary": supply_summary(rows),
                       "raw_source_check": raw_self_edge_check(data_dir, m.neurons["bodyId"].to_numpy())},
            "demand": demand, "fit": fit_summary(rows, demand), "adder_demand": adder_demand()}


def markdown(report: dict) -> str:
    s, f, d, a = report["supply"]["summary"], report["fit"], report["demand"], report["adder_demand"]
    lines = ["# Stage H autapse census", "", "An autapse is a neuron-to-itself anatomical connection.", "",
             "| row | self-edges |", "| --- | ---: |",
             f"| supply total | {s['total']} |", f"| inhibitory supply | {s['by_sign'].get('inhibitory', 0)} |",
             f"| excitatory supply | {s['by_sign'].get('excitatory', 0)} |",
             f"| carriable latch autapse at k_max 4 | {f['latch_autapse']['carriable_self_edges']} |",
             f"| carriable one-shot autapse at k_max 4 | {f['oneshot_transaction']['carriable_self_edges']} |",
             f"| demand on the 4-bit adder (latch / transaction) | {a['latch_autapse_self_edges']} / {a['transaction_self_edges']} |", "",
             f"Profile 2 allows a designed edge only when the presynaptic host has its sign and `0 <= |q| <= k_max * count * QUANTA_PER_SYNAPSE`; `Policy.req_count` supplies the count conversion.",
             f"The latch feedback is {d['latch_autapse']['quanta']} quanta and needs {d['latch_autapse']['required_synapse_count']} synapses at k_max 4; the transaction feedback is {d['oneshot_transaction']['quanta']} quanta and needs {d['oneshot_transaction']['required_synapse_count']}.",
             f"The raw table has {report['supply']['raw_source_check']['raw_self_edges']} self rows, of which {report['supply']['raw_source_check']['excluded_outside_neuron_table']} have a body outside the MCNS neuron table, leaving {s['total']}.",
             "Cholinergic (excitatory) self-edges are shown as supply but cannot carry these inhibitory designed edges."]
    return "\n".join(lines) + "\n"


def main() -> None:
    from ..connectome.mcns_download import DEFAULT_DIR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()
    report = make_report(args.data_dir)
    root = Path(__file__).resolve().parents[2]
    out = root / "docs" / "stage_h"; out.mkdir(parents=True, exist_ok=True)
    (out / "autapse_census.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "autapse_census.md").write_text(markdown(report))
    print(json.dumps(report["supply"]["summary"], indent=2))


if __name__ == "__main__":
    main()
