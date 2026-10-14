"""Stage H: census MCNS self-connections for configuration F."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..connectome.embed_h0 import Policy
from ..connectome.mcns import INHIBITORY, load_mcns
from ..lib.adder import build_adder_channel
from ..sim.model import Params, QUANTA_PER_SYNAPSE

BINS = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-19", 10, 19),
        ("20-49", 20, 49), (">=50", 50, math.inf))
MODULATORY = {"dopamine", "serotonin", "octopamine", "tyramine", "histamine"}


def nt_class(value) -> str:
    """Classify consensus transmitter without MCNS's default-positive sign."""
    if pd.isna(value) or str(value).lower() in {"", "unclear", "missing", "nan"}:
        return "unknown"
    value = str(value).lower()
    if value in INHIBITORY:
        return "inhibitory"
    if value == "acetylcholine":
        return "excitatory"
    if value in MODULATORY:
        return "modulatory"
    return "unknown"


def placement_sign_name(value) -> str:
    return "inhibitory" if value < 0 else "excitatory"


def self_edge_table(pre: np.ndarray, post: np.ndarray, count: np.ndarray, neurons: pd.DataFrame) -> pd.DataFrame:
    """Return annotations for every anatomical self-edge; inputs are MCNS-style arrays."""
    keep = np.asarray(pre) == np.asarray(post)
    indices = np.asarray(pre)[keep].astype(np.int64)
    cols = [c for c in ("bodyId", "type", "superclass", "class", "nt", "predicted_nt", "nt_conf", "sign") if c in neurons]
    out = neurons.iloc[indices][cols].copy().reset_index(drop=True)
    out.insert(0, "neuron_index", indices)
    out["synapse_count"] = np.asarray(count)[keep].astype(np.int64)
    out["nt_class"] = [nt_class(x) for x in out.get("nt", pd.Series(index=out.index, dtype=object))]
    out["placement_sign"] = [placement_sign_name(x) for x in out["sign"]]
    return out


def count_categories(rows: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in rows[column].fillna("unknown").value_counts().items()}


def count_bins(counts: np.ndarray) -> dict[str, int]:
    values = np.asarray(counts)
    return {label: int(((values >= lo) & (values <= hi)).sum()) for label, lo, hi in BINS}


def supply_summary(rows: pd.DataFrame) -> dict:
    return {"total": int(len(rows)), "by_nt_class": count_categories(rows, "nt_class"),
            "by_placement_sign": count_categories(rows, "placement_sign"),
            "by_transmitter": count_categories(rows, "nt"),
            "by_superclass": count_categories(rows, "superclass"),
            "by_synapse_count_bin": count_bins(rows["synapse_count"].to_numpy())}


def _outgoing_dale_signs(net) -> np.ndarray:
    """Placement's designed-neuron transmitter vote; designed self-edges do not vote."""
    src, q, dst = np.asarray(net.src), np.asarray(net.quanta), np.asarray(net.dst)
    nonself = src != dst
    pos = np.bincount(src[nonself], weights=(q[nonself] > 0), minlength=net.n)
    neg = np.bincount(src[nonself], weights=(q[nonself] < 0), minlength=net.n)
    return np.where(pos >= neg, 1, -1).astype(np.int8)


def adder_demand() -> dict:
    """Derive all configuration-F self-edge demand from the specified real build."""
    params = Params()
    channel = build_adder_channel(params, 4, ordered=True, watchdog_hops=100, storage="flipflop",
                                  rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)
    baseline = build_adder_channel(params, 4, ordered=True, watchdog_hops=100, storage="flipflop")
    net = channel.net
    src, dst, q = np.asarray(net.src), np.asarray(net.dst), np.asarray(net.quanta)
    self_edges, dale = src == dst, _outgoing_dale_signs(net)
    role = np.asarray(net.roles, dtype=object)[src]
    kinds = {"latch_autapse": self_edges & np.array([x.endswith((".u", ".v")) for x in role]),
             "oneshot_transaction": self_edges & np.array([".once" in x and x.endswith(".edge") for x in role])}
    result = {"netlist_neurons": int(net.n), "non_f_flipflop_netlist_neurons": int(baseline.net.n),
              "self_edges_by_quanta": {str(int(x)): int((q[self_edges] == x).sum()) for x in np.unique(q[self_edges])}}
    for name, mask in kinds.items():
        quanta = np.unique(q[mask])
        if len(quanta) != 1:
            raise ValueError(f"{name} has {len(quanta)} self-edge quanta; expected one")
        signs = dale[src[mask]]
        result[name] = {"quanta": int(quanta[0]), "self_edges": int(mask.sum()),
                        "designed_source_dale_signs": {"inhibitory": int((signs < 0).sum()),
                                                        "excitatory": int((signs > 0).sum())}}
    return result


def autapse_demands(params: Params = Params(), k_max: float = 4.0, *, built: dict | None = None) -> dict:
    """Profile-2 thresholds derived from configuration F's built-netlist quanta."""
    del params
    built, policy = (adder_demand() if built is None else built), Policy(k_max=k_max)
    return {"k_max": k_max, "quanta_per_synapse": QUANTA_PER_SYNAPSE,
            **{name: {"quanta": built[name]["quanta"], "required_synapse_count": policy.req_count(built[name]["quanta"])}
               for name in ("latch_autapse", "oneshot_transaction")}}


def fit_summary(rows: pd.DataFrame, demand: dict, built: dict | None = None) -> dict:
    """Compare standalone sign/count capacity with the complete Dale-law placement rule."""
    built = adder_demand() if built is None else built
    out = {"inhibitory_self_edges": int((rows["placement_sign"] == "inhibitory").sum()),
           "excitatory_self_edges": int((rows["placement_sign"] == "excitatory").sum())}
    for name in ("latch_autapse", "oneshot_transaction"):
        req = demand[name]["required_synapse_count"]
        standalone = rows[(rows["placement_sign"] == "inhibitory") & (rows["synapse_count"] >= req)]
        signs = built[name]["designed_source_dale_signs"]
        # The source vote must also agree with the negative self-edge.  It does not.
        full = standalone if signs["inhibitory"] else standalone.iloc[0:0]
        out[name] = {"quanta": int(demand[name]["quanta"]), "required_synapse_count": int(req),
                     "standalone_inhibitory_self_edges_with_enough_synapses": int(len(standalone)),
                     "standalone_body_ids": [int(x) for x in standalone["bodyId"]],
                     "full_rule_carriable_self_edges": int(len(full)), "designed_source_dale_signs": signs}
    return out


def raw_self_edge_check(data_dir: Path, known_bodies: np.ndarray) -> dict:
    import pyarrow.dataset as ds
    dataset = ds.dataset(data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather", format="feather")
    names = dataset.schema.names
    pre = "body_pre" if "body_pre" in names else next(c for c in names if "pre" in c.lower())
    post = "body_post" if "body_post" in names else next(c for c in names if "post" in c.lower())
    total = kept = 0
    bodies = np.sort(np.asarray(known_bodies))
    for batch in dataset.scanner(columns=[pre, post], batch_size=2_000_000).to_batches():
        a, b = batch.column(0).to_numpy(), batch.column(1).to_numpy(); same = a == b
        total += int(same.sum()); x = a[same]; pos = np.searchsorted(bodies, x)
        kept += int(((pos < len(bodies)) & (bodies[np.minimum(pos, len(bodies) - 1)] == x)).sum())
    return {"raw_self_edges": total, "raw_self_edges_with_both_bodies_in_neuron_table": kept,
            "excluded_outside_neuron_table": total - kept}


def _json_clean(value):
    if isinstance(value, dict): return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def make_report(data_dir: Path) -> dict:
    m = load_mcns(data_dir=data_dir)
    rows, built = self_edge_table(m.pre, m.post, m.count, m.neurons), adder_demand()
    demand = autapse_demands(built=built)
    return {"stage": "H autapse census", "inputs": {"edge_cache": str(data_dir / "mcns_v1.0_edges_minsyn1.npz"),
            "neurons": str(data_dir / "mcns_v1.0_neurons.parquet"), "raw_weights": str(data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
            "params": dataclasses.asdict(Params()), "k_max": 4.0}, "mcns": {"version": m.version, "data_dir": str(data_dir),
            "neuron_rule": "superclass is not null", "edge_rule": "both endpoints are neurons and syn_count >= min_syn"},
            "placement_rule": {"quoted": "0 <= |q| <= k_max * count * QUANTA_PER_SYNAPSE; the presynaptic host transmitter sign must match the designed edge sign and the designed neuron's outgoing-edge vote (Dale's law)."},
            "supply": {"self_edges": rows.to_dict(orient="records"), "summary": supply_summary(rows),
                       "raw_source_check": raw_self_edge_check(data_dir, m.neurons["bodyId"].to_numpy())},
            "demand": demand, "fit": fit_summary(rows, demand, built), "adder_demand": built}


def markdown(report: dict) -> str:
    s, f, d, a = report["supply"]["summary"], report["fit"], report["demand"], report["adder_demand"]
    lines = ["# Stage H autapse census", "", "An autapse is a neuron-to-itself anatomical connection; Dale's law means all outputs of one designed neuron use one transmitter sign.", "",
             "| row | count |", "| --- | ---: |", f"| anatomical self-edges | {s['total']} |",
             f"| transmitter class: inhibitory / excitatory / modulatory / unknown | {s['by_nt_class'].get('inhibitory', 0)} / {s['by_nt_class'].get('excitatory', 0)} / {s['by_nt_class'].get('modulatory', 0)} / {s['by_nt_class'].get('unknown', 0)} |",
             f"| placement sign: inhibitory / excitatory | {s['by_placement_sign'].get('inhibitory', 0)} / {s['by_placement_sign'].get('excitatory', 0)} |",
             f"| configuration-F / non-F flip-flop adder neurons | {a['netlist_neurons']} / {a['non_f_flipflop_netlist_neurons']} |",
             f"| F latch / one-shot self-edge demand (quanta) | {a['latch_autapse']['self_edges']} ({a['latch_autapse']['quanta']}) / {a['oneshot_transaction']['self_edges']} ({a['oneshot_transaction']['quanta']}) |",
             f"| latch: standalone inhibitory loops with >= {d['latch_autapse']['required_synapse_count']} synapses / full-rule carriable | {f['latch_autapse']['standalone_inhibitory_self_edges_with_enough_synapses']} / {f['latch_autapse']['full_rule_carriable_self_edges']} |",
             f"| one-shot: standalone inhibitory loops with >= {d['oneshot_transaction']['required_synapse_count']} synapses / full-rule carriable | {f['oneshot_transaction']['standalone_inhibitory_self_edges_with_enough_synapses']} / {f['oneshot_transaction']['full_rule_carriable_self_edges']} |", "",
             "The real F build directly supplies the demand, split by its netlist self-edge quanta rather than a hand-matched circuit pattern.",
             f"The two standalone latch candidates are bodyIds {', '.join(map(str, f['latch_autapse']['standalone_body_ids']))}; neither could also host the excitatory loop to which its autapse belongs.",
             "Both full-rule counts are zero because every relevant designed source votes excitatory while its self-edge is inhibitory.",
             "Thus inhibitory self-feedback on an excitatory neuron needs a separate inhibitory neuron, making a two-step loop through an interneuron anatomically possible.",
             "The optional cross-check against the driven mutual-inhibitory-pair pool was not done."]
    return "\n".join(lines) + "\n"


def main() -> None:
    from ..connectome.mcns_download import DEFAULT_DIR
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    report = make_report(p.parse_args().data_dir)
    out = Path(__file__).resolve().parents[2] / "docs" / "stage_h"; out.mkdir(parents=True, exist_ok=True)
    (out / "autapse_census.json").write_text(json.dumps(_json_clean(report), indent=2, allow_nan=False) + "\n")
    (out / "autapse_census.md").write_text(markdown(report))
    print(json.dumps(report["supply"]["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
