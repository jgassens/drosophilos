"""Stage H1: the placed 4-bit adder inside the whole fly brain.

docs/h1_placement.md measured the placed adder in isolation: condition A (every designed edge
present -- 704 carried on real MCNS neurons at rescaled anatomical weights, 442 added as
labelled Profile 3 edges -- and the 17,845 parasitic anatomical edges among the hosts zeroed)
computes 50 / 50 additions with only the circuit's 614 neurons simulated. This driver embeds
that same image in the whole MCNS topology (`embed_image.full_graph_topology`: 166,700 neurons,
25.6 M edges, the image's edits applied in place) and runs N additions with a surround driven
exactly as Stage H0 drove it (`connectome/h0_run.py`):

  silent    nothing outside the circuit is driven (the brain still reacts to the circuit);
  poisson   every sensory neuron fires at --rate Hz (H0: 2 Hz on 17,937 sensory neurons);
  burst     1,000 random cholinergic neurons fire together 5 ms after the operands are loaded.

Two switches make the boundary explicit: --outputs-zeroed silences every edge from a host to
the rest of the brain (H0's outputs-zeroed condition), --inputs-zeroed every edge from the
brain into a host (the isolated case embedded; the control that must reproduce the isolated
run spike for spike).  --cut-outputs-frac and --cut-inputs-frac instead cut the strongest
deterministic fraction of their respective boundary edges, for a cluster cut ladder.  Per
addition it records correct / wrong / fault / no ACCEPT / timeout,
the ACCEPT latency, the circuit's spikes against the same word run in isolation, the boundary
envelope (spikes arriving into circuit neurons from outside, per ms, by transmitter sign) and
the wall seconds. One addition is ~12,000 RefSim steps of 166,700 neurons: tens of seconds on
a cluster node. Do not run it on a laptop.

    python -m drosophilos.bench.h1_fullgraph --n-cases 50 --seed 0 --surround silent --out data/h1_fullgraph/silent.json
    python -m drosophilos.bench.h1_fullgraph --mapping data/stage_h/place/all_w0_mapping.json --surround silent --cut-inputs-frac .5 --out data/h1_fullgraph/inputs-50.json

Fault attribution (cluster only; run from the repository root). First submit the live,
uncut all_w0 attribution runs, 50 additions in each surround:

    for s in silent poisson; do
      slurm/submit.sh --cluster juno --script juno-cpu --time=2-00:00:00 -- drosophilos.bench.h1_fullgraph \\
        --mapping data/stage_h/place/all_w0_mapping.json --n-cases 50 --seed 0 \\
        --surround "$s" --rate 2 --max-ms 1200 --attribute 20 \\
        --attribution-out "data/stage_h/fullgraph/all_w0_attr_${s}.npz" \\
        --out "data/stage_h/fullgraph/all_w0_attr_${s}.json"
    done

After both jobs finish, submit attribution-ranked rungs AND synapse-ranked comparisons
(the NPZ is the full aggregate score array; compact JSON is not a cut input):

    for s in silent poisson; do
      for f in 0.05 0.10 0.20 0.30; do
        slurm/submit.sh --cluster juno --script juno-cpu --time=2-00:00:00 -- drosophilos.bench.h1_fullgraph \\
          --mapping data/stage_h/place/all_w0_mapping.json --n-cases 50 --seed 0 \\
          --surround "$s" --rate 2 --max-ms 1200 --cut-inputs-frac "$f" \\
          --cut-inputs-by attribution --attribution-in "data/stage_h/fullgraph/all_w0_attr_${s}.npz" \\
          --out "data/stage_h/fullgraph/all_w0_attr_cutin${f}_${s}_n50.json"
        slurm/submit.sh --cluster juno --script juno-cpu --time=2-00:00:00 -- drosophilos.bench.h1_fullgraph \\
          --mapping data/stage_h/place/all_w0_mapping.json --n-cases 50 --seed 0 \\
          --surround "$s" --rate 2 --max-ms 1200 --cut-inputs-frac "$f" \\
          --out "data/stage_h/fullgraph/all_w0_syn_cutin${f}_${s}_n50.json"
      done
    done

Only failed additions are scored (valid expected decode plus READY is success). Failure
time is the earliest consumer FAULT, watchdog TIMEOUT, wrong consumer rail tap, or tap
earlier than its first isolated reference spike relative to load. Rail checks stop at
ACCEPT/reset. Without such a spike, use terminal READY or the exclusive harness deadline
as the stall detection time. Deliveries at source_step + embedded_delay in [time-W, time)
contribute spikes * abs(embedded quanta). See h_fault_attr.failure_point for tie rules.
These scores describe temporal association; testing the cut ladder measures shielding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..connectome.embed_h0 import Policy
from ..connectome.embed_image import (adder_cases, all_missing, boundary_envelope_ms, build_image, circuit_sim,
                                      full_graph_topology, load_mapping, load_mapping_meta, run_words, surround_events, surround_populations,
                                      tally, trace_since, underlying, power_on)
from ..connectome.mcns import load_mcns
from ..lib.adder import build_adder_channel
from ..protocol.latch import DALE_BIASED_VERSION, DALE_FEEDBACK_VERSION
from ..sim.model import Params
from .h_fault_attr import IncomingAttribution, failure_point

DEFAULT_MAPPING = Path(__file__).resolve().parents[2] / "docs" / "h1_placement_mapping.json"
BURST_N = 1000  # h0_run: rng.choice(exc_all, 1000)
BURST_AT = 50  # steps after the load (h0_run: tx.t_data + 50)

ADDER_META_KEYS = ("builder", "width", "ordered", "watchdog_hops", "storage", "rate_robust",
                   "experimental_autapse", "ordered_xor_oneshot")


def adder_build_kwargs(mapping_meta: dict, *, width_override: int | None = None,
                       storage_override: str | None = None) -> dict:
    """Resolve a mapping's Stage-H builder contract to ``build_adder_channel`` keyword args.

    Pre-contract mappings retain the original H1 build (4-bit ordered latch, watchdog 100),
    except that the older flip-flop mapping's top-level ``storage`` is intentionally honoured.
    An explicit CLI value can only confirm, never silently replace, a saved contract.
    """
    meta = dict(mapping_meta)
    contract = "builder" in meta
    if contract:
        missing = [key for key in ADDER_META_KEYS if key not in meta]
        if missing:
            raise ValueError(f"mapping Stage-H metadata is incomplete; missing {', '.join(missing)}")
        if meta["builder"] != "build_adder_channel":
            raise ValueError(f"mapping builder {meta['builder']!r} is not build_adder_channel")
        if width_override is not None and width_override != meta["width"]:
            raise ValueError(f"--width {width_override} disagrees with mapping width {meta['width']}")
        if storage_override is not None and storage_override != meta["storage"]:
            raise ValueError(f"--storage {storage_override!r} disagrees with mapping storage {meta['storage']!r}")
        options = {key: meta[key] for key in ADDER_META_KEYS if key != "builder"}
        latch_feedback = meta.get("latch_feedback", "autapse")
        if latch_feedback not in ("autapse", "dale_private", "dale_biased"):
            raise ValueError(f"unsupported mapping latch_feedback: {latch_feedback!r}")
        if latch_feedback != "autapse" and not options["experimental_autapse"]:
            raise ValueError(f"unsupported mapping latch_feedback: {latch_feedback} requires experimental_autapse=True")
        if latch_feedback == "dale_private":
            dale_feedback_version = meta.get("dale_feedback_version")
            if type(dale_feedback_version) is not int or dale_feedback_version != DALE_FEEDBACK_VERSION:
                raise ValueError("unsupported mapping Dale-feedback circuit version")
            options.update(latch_feedback=latch_feedback, dale_feedback_version=dale_feedback_version)
        elif latch_feedback == "dale_biased":
            dale_biased_version = meta.get("dale_biased_version")
            if type(dale_biased_version) is not int or dale_biased_version != DALE_BIASED_VERSION:
                raise ValueError("unsupported mapping Dale-biased circuit version")
            options.update(latch_feedback=latch_feedback, dale_biased_version=dale_biased_version)
        return options

    # Legacy files did not record a whole builder contract.  This exactly reproduces the old
    # invocation, while allowing h1_ffadder_mapping.json to select its flip-flop output store.
    latch_feedback = meta.get("latch_feedback", "autapse")
    if latch_feedback != "autapse":
        raise ValueError("legacy mapping cannot specify non-default latch_feedback")
    saved_storage = meta.get("storage", "latch")
    if storage_override is not None and storage_override != saved_storage:
        raise ValueError(f"--storage {storage_override!r} disagrees with mapping storage {saved_storage!r}")
    return {"width": 4 if width_override is None else width_override, "ordered": True,
            "watchdog_hops": 100, "storage": saved_storage, "rate_robust": False,
            "experimental_autapse": False, "ordered_xor_oneshot": False}


def top_recruited(m, trace, circuit: np.ndarray, k: int = 8) -> list:
    ev = trace.events
    out = ev["neuron"][~np.isin(ev["neuron"], circuit) & (ev["neuron"] < m.n)]
    if not len(out):
        return []
    ids, cnts = np.unique(out, return_counts=True)
    top = np.argsort(-cnts)[:k]
    return [{"type": str(m.neurons["type"].iat[int(ids[j])]), "superclass": str(m.neurons["superclass"].iat[int(ids[j])]),
             "spikes": int(cnts[j])} for j in top]


def make_sims(ch, isolated_topology, full_topology, full_index_map, params: Params):
    """Make the paired H1 simulators, including their one-time channel power-on pulse.

    Kept separate so the small synthetic regression can exercise the same fresh and chained
    simulator setup as the whole-brain driver.
    """
    sim_iso = circuit_sim(isolated_topology, params)
    sim_full = circuit_sim(full_topology, params, full_index_map)
    power_on(ch, sim_iso)
    power_on(ch, sim_full)
    return sim_iso, sim_full


def run(args) -> dict:
    params = Params()
    max_steps = int(round(args.max_ms / params.dt))
    t_start = time.time()
    mapping_meta = load_mapping_meta(args.mapping)
    build_options = adder_build_kwargs(mapping_meta, width_override=args.width, storage_override=args.storage)
    ch = build_adder_channel(params, **build_options)
    width = build_options["width"]
    m = load_mcns()
    pl = load_mapping(args.mapping, ch.net, m)
    img = build_image(ch.net, m, pl, Policy(), profile3=all_missing)  # condition A: every designed edge, parasitic zeroed
    fg = full_graph_topology(m, params, img, zero_outputs=args.outputs_zeroed, zero_inputs=args.inputs_zeroed,
                             cut_outputs_frac=args.cut_outputs_frac, cut_inputs_frac=args.cut_inputs_frac,
                             cut_inputs_by=getattr(args, "cut_inputs_by", "synapses"),
                             attribution_in=getattr(args, "attribution_in", None))
    attr_window = getattr(args, "attribute", None)
    attribution = IncomingAttribution(m, fg, ch.net.roles, attr_window, params) if attr_window is not None else None
    hosts = fg.index_map[img.real >= 0]
    pops = surround_populations(m, exclude=hosts)
    driven = {"silent": np.empty(0, np.int64), "poisson": pops["sensory"], "burst": pops["cholinergic"]}[args.surround]
    print(f"[h1fg] image: {img.counts['carried']} carried, {img.counts['profile3_edges']} Profile 3, "
          f"{img.counts['parasitic_zeroed']} parasitic zeroed, {img.counts['synthetic_neurons']} synthetic neurons; "
          f"full graph: {fg.counts['n']} neurons, {fg.counts['topology_edges']} edges, {fg.counts['nonzero_edges']} non-zero, "
          f"outputs cut {fg.counts['outputs_cut_edges']} ({fg.counts['outputs_cut_synapses']} synapses), "
          f"inputs cut {fg.counts['inputs_cut_edges']} ({fg.counts['inputs_cut_synapses']} synapses); "
          f"surround {args.surround}: {len(driven)} driven neurons (hosts excluded: {pops['excluded_hosts']}); "
          f"setup {time.time() - t_start:.1f}s", flush=True)

    cases, words, expected = adder_cases(width, args.n_cases, args.seed)
    rng = np.random.default_rng(args.seed)
    sim_iso = sim_full = None
    recs, reached, additions, spikes = [], [], [], 0
    for k, ((a, b, c), w, x) in enumerate(zip(cases, words, expected)):
        if sim_iso is None or not args.chained:
            # run_words receives made simulators and therefore does not add this itself.
            # Each simulator gets exactly one designed-index power-on pulse.
            sim_iso, sim_full = make_sims(ch, img.topology, fg.topology, fg.index_map, params)
        usim = underlying(sim_full)
        t0 = time.time()
        r_iso, re_iso, _ = run_words(ch, sim_iso, [w], [x], params, max_steps)
        wall_iso = time.time() - t0
        # the surround, scheduled over this addition's window (the harness loads at sim.step_index)
        load = sim_full.step_index
        steps, neurons, quanta = surround_events(args.surround, driven, params, load, max_steps, rng, rate_hz=args.rate,
                                                 burst_n=BURST_N, burst_at=BURST_AT)
        if len(steps):
            usim.add_events(0, steps, neurons, quanta)
        b0 = len(usim._spk_step)
        t0 = time.time()
        r_full, re_full, st = run_words(ch, sim_full, [w], [x], params, max_steps)
        wall = time.time() - t0
        rec, ri = r_full[0], r_iso[0]
        end = re_full[0]["end_step"]
        tr_full = trace_since(usim, b0)
        env = boundary_envelope_ms(fg.topology, tr_full, fg.index_map, rec.load_step, end + 1, params)
        recruited = top_recruited(m, tr_full, fg.index_map)
        # the circuit's own spikes against the isolated run of the same word, relative to each load
        ev_f, ev_i = sim_full.trace.events, sim_iso.trace.events
        wf = ev_f[(ev_f["step"] >= rec.load_step) & (ev_f["step"] <= end)]
        wi = ev_i[(ev_i["step"] >= ri.load_step) & (ev_i["step"] <= re_iso[0]["end_step"])]
        identical = len(wf) == len(wi) and bool(np.array_equal(wf["neuron"], wi["neuron"])) and \
            bool(np.array_equal(wf["step"] - rec.load_step, wi["step"] - ri.load_step))
        if args.chained:
            sim_full.drop_underlying_spikes()
        outcome = ("correct" if rec.status == "valid" and rec.decoded == rec.word else
                   "wrong" if rec.status == "valid" and rec.decoded is not None else
                   "no_accept" if rec.accept_step is None and rec.status not in ("fault", "timeout") else rec.status)
        accept_ms = (rec.accept_step - rec.load_step) * params.dt if rec.accept_step is not None else None
        cycle_ms = (rec.ready_step - rec.load_step) * params.dt if rec.ready_step is not None else None
        row = {"k": k, "a": a, "b": b, "cin": c, "expected": x, "decoded": rec.decoded, "status": rec.status, "outcome": outcome,
               "accept_ms": accept_ms, "cycle_ms": cycle_ms, "load_step": int(rec.load_step), "end_step": int(end),
               "fault_spikes": int(rec.fault_spikes), "timeout_spikes": int(rec.timeout_spikes),
               "circuit_spikes": re_full[0]["circuit_spikes"], "isolated": {
                   "status": ri.status, "decoded": ri.decoded, "circuit_spikes": re_iso[0]["circuit_spikes"],
                   "accept_ms": (ri.accept_step - ri.load_step) * params.dt if ri.accept_step is not None else None,
                   "wall_s": round(wall_iso, 2)},
               "identical_to_isolated": identical, "reached": re_full[0],
               "envelope": env, "top_recruited": recruited, "wall_s": round(wall, 2)}
        if attribution is not None:
            failure = failure_point(ch, rec, sim_full.trace, sim_iso.trace, ri, end)
            row["attribution"] = attribution.observe(tr_full, failure, end_step=sim_full.step_index,
                                                     fresh=not args.chained or k == 0)
            if failure is not None:
                row["attribution"]["ms_since_load"] = (failure["step"] - rec.load_step) * params.dt
        additions.append(row)
        recs.append(rec); reached.append(re_full[0]); spikes += st["total_spikes"]
        print(f"[h1fg] {k + 1:3d}/{args.n_cases} {a:2d}+{b:2d}+{c} = {x:2d}: {outcome:9s} decoded {rec.decoded} "
              f"accept {accept_ms if accept_ms is None else round(accept_ms)} ms cycle {cycle_ms if cycle_ms is None else round(cycle_ms)} ms | "
              f"circuit spikes {re_full[0]['circuit_spikes']} (isolated {re_iso[0]['circuit_spikes']}{', identical' if identical else ''}) | "
              f"surround spikes {env['surround_spikes']} from {env['surround_neurons_recruited']} neurons, "
              f"in: exc {env['exc_spikes_in']} inh {env['inh_spikes_in']} (max/ms {env['max_exc_per_ms']}/{env['max_inh_per_ms']}) | "
              f"wall {wall:.1f}s", flush=True)

    summary = tally(recs, reached, args.n_cases, params, spikes)
    summary.pop("records"); summary.pop("reached")
    summary["outcomes"] = {o: sum(1 for r in additions if r["outcome"] == o) for o in ("correct", "wrong", "fault", "no_accept", "timeout", "incomplete")}
    summary["identical_to_isolated"] = sum(1 for r in additions if r["identical_to_isolated"])
    summary["wall_s_per_addition"] = [r["wall_s"] for r in additions]
    summary["wall_s_total"] = round(time.time() - t_start, 1)
    summary["surround_spikes"] = [r["envelope"]["surround_spikes"] for r in additions]
    summary["max_exc_per_ms"] = max((r["envelope"]["max_exc_per_ms"] for r in additions), default=0)
    summary["max_inh_per_ms"] = max((r["envelope"]["max_inh_per_ms"] for r in additions), default=0)
    out = {"args": vars(args) | {"mapping": str(args.mapping), "out": str(args.out)},
           "mapping_meta": mapping_meta, "build_options": {"builder": "build_adder_channel", **build_options}, "params": params.__dict__,
           "image_counts": img.counts, "full_graph_counts": fg.counts,
           "surround": {"kind": args.surround, "rate_hz": args.rate, "driven_neurons": int(len(driven)),
                        "burst_n": BURST_N, "burst_at_steps": BURST_AT, "excluded_hosts": pops["excluded_hosts"],
                        "sensory_neurons": int(len(pops["sensory"])), "cholinergic_neurons": int(len(pops["cholinergic"]))},
           "additions": additions, "summary": summary}
    if attribution is not None:
        out["attribution"] = attribution.report(getattr(args, "attribution_top", 50))
        if getattr(args, "attribution_out", None) is not None:
            attribution.save(args.attribution_out)
    print(f"[h1fg] summary ({args.surround}, outputs zeroed {args.outputs_zeroed}, inputs zeroed {args.inputs_zeroed}, "
          f"{'chained' if args.chained else 'fresh'}): {summary['outcomes']}; correct {summary['correct']}/{args.n_cases}, "
          f"identical to isolated {summary['identical_to_isolated']}/{args.n_cases}, accept ms {summary['accept_ms'][:5]}..., "
          f"max spikes in per ms exc {summary['max_exc_per_ms']} inh {summary['max_inh_per_ms']}, "
          f"wall {summary['wall_s_total']}s ({np.mean(summary['wall_s_per_addition']) if additions else 0:.1f}s per addition)", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, default=str))
        print(f"[h1fg] wrote {args.out}", flush=True)
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surround", choices=("silent", "poisson", "burst"), default="silent")
    ap.add_argument("--rate", type=float, default=2.0, help="Hz per sensory neuron for --surround poisson (H0: 2)")
    ap.add_argument("--inputs-zeroed", action="store_true", help="zero every edge from the brain into a host")
    ap.add_argument("--outputs-zeroed", action="store_true", help="zero every edge from a host into the brain")
    ap.add_argument("--cut-inputs-frac", type=float, default=0.0,
                    help="cut strongest fraction [0, 1] of anatomical edges into hosts")
    ap.add_argument("--cut-outputs-frac", type=float, default=0.0,
                    help="cut strongest fraction [0, 1] of anatomical edges from hosts")
    # Suppressed defaults preserve the legacy args JSON exactly when no new flags are used.
    ap.add_argument("--attribute", type=float, default=argparse.SUPPRESS, metavar="WINDOW_MS",
                    help="attribute failed additions to incoming deliveries in the preceding window")
    ap.add_argument("--attribution-top", type=int, default=argparse.SUPPRESS, metavar="N",
                    help="edges in compact attribution JSON (default: 50)")
    ap.add_argument("--attribution-out", type=Path, default=argparse.SUPPRESS,
                    help="write full aggregate per-edge scores to compressed NPZ, without spike traces")
    ap.add_argument("--cut-inputs-by", choices=("synapses", "attribution"), default=argparse.SUPPRESS,
                    help="rank incoming cuts by synapses (default) or attribution score")
    ap.add_argument("--attribution-in", type=Path, default=argparse.SUPPRESS,
                    help="full score NPZ from --attribution-out, for --cut-inputs-by attribution")
    ap.add_argument("--chained", action="store_true", help="all additions through one simulator (default: fresh per addition)")
    ap.add_argument("--max-ms", type=float, default=1200.0, help="neural time allowed per addition")
    ap.add_argument("--width", type=int, default=None, help="override legacy mapping width (default: 4; metadata mappings set it)")
    ap.add_argument("--storage", choices=("latch", "flipflop"), default=None,
                    help="confirm a mapping's output-store choice (legacy flip-flop mappings record this)")
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if hasattr(args, "attribute") and (not np.isfinite(args.attribute) or args.attribute <= 0):
        ap.error("--attribute must be finite and positive")
    if (hasattr(args, "attribution_out") or hasattr(args, "attribution_top")) and not hasattr(args, "attribute"):
        ap.error("--attribution-out/--attribution-top require --attribute")
    if getattr(args, "attribution_top", 50) < 0:
        ap.error("--attribution-top must be nonnegative")
    if (getattr(args, "cut_inputs_by", "synapses") == "attribution") != hasattr(args, "attribution_in"):
        ap.error("--cut-inputs-by attribution requires --attribution-in (and vice versa)")
    return args


if __name__ == "__main__":
    run(parse())
