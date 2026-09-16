"""The dual-rail register on flip-flop storage (build_channel(storage="flipflop")): the
consumer's eight rail latches are flip-flops (docs/a1_flipflop.md), set by the producer's
rail pulse through a per-rail set chain, cleared by the reset train aimed at their u
members, and READ through their excitatory proxies p (the valid ORs, the fault ANDs and the
decode taps all take `.p`; u is inhibitory and cannot drive a reader on the connectome).
The default build is untouched (tests/test_handshake.py etc. run on it).

Fast: 200 random transfers with zero decoded errors; the CLEAR at every phase; a stray
pulse that puts a parked rail into lockstep is healed by the next reset; a stray pulse into
a proxy while CLEAR gives no false completion; a 20-transfer mix-B smoke of the campaign
loop below. Slow (~10 min on the laptop): 10,000 transfers at mix B.

    PYTHONPATH=. python tests/test_ffregister.py     # re-measures docs/contracts/ffregister.yaml
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from drosophilos.lib.campaign import Perturbation, _decode_node, clopper_pearson_upper
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.flipflop import FlipFlop, flipflop_taps, power_on_events
from drosophilos.protocol.handshake import Register, add_register, build_channel
from drosophilos.protocol.run import run_transactions
from drosophilos.protocol.token import decode_at, rails_for
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim

PARAMS = Params()
MIX_B = Perturbation(0.04, 0.2, 0.2, 5.0, 150, 100)  # drosophilos/bench/a2_campaigns.py MIXES["B"]
WATCHDOG_HOPS = 55  # the frozen 4-bit channel build


def ff_channel(width: int = 4, **kw):
    return build_channel(PARAMS, width, watchdog_hops=WATCHDOG_HOPS, storage="flipflop", **kw)


def powered_sim(ch):
    """A RefSim with the channel's power-on pulses at step 0 (what load_image would inject)."""
    sim = RefSim(ch.net.topology(), PARAMS)
    for s, n, q in ch.power_on_events():
        sim.add_events(0, [s], [n], [q])
    return sim


def _spikes(sim, n, lo, hi):
    e = sim.trace.events
    st = e["step"][e["neuron"] == n]
    return st[(st >= lo) & (st < hi)]


def test_default_storage_is_a_latch_register():
    ch = build_channel(PARAMS, 4)
    assert ch.consumer.storage == "latch" and not ch.consumer.all_flipflops() and ch.power_on_events() == []
    assert ch.net.topology().bias is None  # no biased neuron: the simulators see the old topology


def test_flipflop_register_200_random_transfers():
    """200 transfers, zero decoded errors. Four channels of 50: run_transactions rebuilds the
    spike trace once per transaction (quadratic in the run length, and the eight CLEAR-side
    v members double the trace), so one 200-word run takes ~60 s and four 50-word runs ~25 s."""
    rng = np.random.default_rng(20260915)
    t0 = time.time()
    n_done, n_ok, lat = 0, 0, []
    for k in range(4):
        ch = ff_channel()
        words = [int(w) for w in rng.integers(0, 16, size=50)]
        recs, sim, st = run_transactions(ch, PARAMS, words, sim=powered_sim(ch))
        assert st["completed"] == 50, st
        bad = [(r.word, r.decoded, r.status) for r in recs if r.decoded != r.word or r.status != "valid"]
        assert not bad, bad
        assert all(r.fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)
        assert all(r.accept_step < r.cleared_step < r.ready_step for r in recs)
        # every set chain fires exactly once per active rail, and no consumer rail's u or p
        # (the value) fires between READY and the next load; the v members legitimately do
        Q = ch.consumer
        assert Q.rail_taps == [[ff.p for ff in pair] for pair in Q.rails]  # read through the proxies
        ev = sim.trace.events
        n_set = sum(int((ev["neuron"] == t).sum()) for pair in Q.rail_inputs for t in pair)
        assert n_set == 4 * 50, n_set
        us = [ff.u for pair in Q.rails for ff in pair] + [ff.p for pair in Q.rails for ff in pair]
        for j, r in enumerate(recs[:-1]):
            gap = (ev["step"] > r.ready_step) & (ev["step"] < recs[j + 1].load_step)
            assert not np.isin(ev["neuron"][gap], us).any(), (k, j)
        n_done += 50
        n_ok += st["correct"]
        lat += st["accept_latency_ms"]
    assert n_done == 200 and n_ok == 200
    print(f"flip-flop register: 200/200 in {time.time() - t0:.0f} s, accept {np.mean(lat):.1f} ms mean, {np.max(lat):.1f} max")


def test_register_clears_at_every_phase_and_holds_two_seconds():
    """A consumer register alone: loaded through its set triggers, it holds 2 s at the latch's
    rate; the reset train (4 x 1.5x loop into u only) clears every rail at all 47 phases."""
    drive = Drive.from_params(PARAMS)
    net = Netlist(PARAMS)
    R = add_register(net, drive, "Q", 4, with_completion=True, storage="flipflop")
    assert len(R.all_flipflops()) == 8 and len(flipflop_taps(net)) == 8 and len(flipflop_taps(net, ".p")) == 8
    word = 0b1010
    load = [(500, R.rail_inputs[i][(word >> i) & 1], drive.ignite) for i in range(4)]

    def run(events, ms):
        sim = RefSim(net.topology(), PARAMS)
        for s, n, q in power_on_events(net, drive) + events:
            sim.add_events(0, [s], [n], [q])
        sim.run(int(ms / PARAMS.dt))
        return sim

    sim = run(load, 2600)
    value, status = decode_at(sim.trace, R.rail_taps, 26000, 2 * drive.loop_period_steps)
    assert (value, status) == (word, "valid")
    for n in (R.rails[1][1].u, R.rails[1][1].p):  # u and its proxy: the same train, step for step
        train = _spikes(sim, n, 6000, 26000)
        assert abs(len(train) / 2.0 - 213) < 5 and set(np.diff(train).tolist()) == {drive.loop_period_steps}
    tu, tp = _spikes(sim, R.rails[1][1].u, 6000, 26000), _spikes(sim, R.rails[1][1].p, 6000, 26000)
    k = min(len(tu), len(tp))  # p lags u by a few ms, so the window's edge can hold one u spike whose p spike is outside it
    assert abs(len(tu) - len(tp)) <= 1 and len(set((tp[:k] - tu[:k]).tolist())) == 1 and 0 <= int(tp[0] - tu[0]) < drive.loop_period_steps
    for ph in range(0, 47, 2):
        t = 3000 + ph
        sim = run(load + [(t, R.reset_trigger, drive.ignite)], 420)
        for pair in R.rails:
            for ff in pair:
                assert len(_spikes(sim, ff.u, t + 300, 4200)) == 0, ph  # SET rails cleared, silent ones still silent
                assert len(_spikes(sim, ff.p, t + 300, 4200)) == 0, ph  # ...and their proxies parked (p's last spike <= 27 ms after the trigger)
                assert len(_spikes(sim, ff.v, t + 300, 4200)) > 0, ph  # every v running: CLEAR, not lockstep


def test_stray_lockstep_is_healed_by_the_next_reset():
    """A stray excitatory pulse >= 1.15x ignite into a parked rail's u does not SET it but
    drops the pair into lockstep (both members ~105 Hz; docs/a1_flipflop.md). The reset train
    into u resolves lockstep to CLEAR, so the damage lasts one transaction at most."""
    drive = Drive.from_params(PARAMS)
    net = Netlist(PARAMS)
    R = add_register(net, drive, "Q", 4, with_completion=True, storage="flipflop")
    ff = R.rails[0][1]
    sim = RefSim(net.topology(), PARAMS)
    for s, n, q in power_on_events(net, drive) + [(2000, ff.u, int(round(1.5 * drive.ignite))), (5000, R.reset_trigger, drive.ignite)]:
        sim.add_events(0, [s], [n], [q])
    sim.run(7000)
    u, v = _spikes(sim, ff.u, 3000, 5000), _spikes(sim, ff.v, 3000, 5000)
    assert 15 < len(u) < 30 and 15 < len(v) < 30, (len(u), len(v))  # lockstep: both at ~105 Hz for 200 ms
    assert len(_spikes(sim, ff.u, 5500, 7000)) == 0 and len(_spikes(sim, ff.v, 5500, 7000)) > 25  # CLEAR again (v at 213 Hz)


def _register_readers(net: Netlist, R: Register) -> dict[str, list[int]]:
    ors = [i for i, r in enumerate(net.roles) if r.startswith("Q.valid") and r.endswith(".or")]
    return {"or": ors, "valid": [l.u for l in R.valid], "tree": [l.u for l in R.internal],
            "completion": [R.completion.u], "fault": list(R.fault)}


def test_stray_into_proxy_while_clear_gives_no_false_completion():
    """A stray excitatory pulse into a parked rail's proxy p (docs/contracts/flipflop.yaml,
    noise_margin: 1.1x ignite never fires p; 1.15x fires it once, 1.8x twice; the pair is
    untouched). Through the register's readers: one or two p spikes are 766 quanta each into
    the bit's valid OR (single need 2,586), so no OR spike, no valid latch, no tree latch, no
    completion and no fault gate follows - measured to 4.0x ignite (three p spikes) into one
    proxy. Only the same stray into BOTH proxies of a bit at once (four OR-input spikes) makes
    a false valid on that bit from 1.8x up; completion would need it on every bit."""
    drive = Drive.from_params(PARAMS)
    net = Netlist(PARAMS)
    R = add_register(net, drive, "Q", 4, with_completion=True, storage="flipflop")
    topo = net.topology()
    readers = _register_readers(net, R)
    ff = R.rails[2][1]
    for x, p_spikes in ((1.1, 0), (1.15, 1), (1.8, 2), (4.0, 3)):  # max p spikes over the phases
        for ph in range(0, 47, 4):
            t = 2000 + ph
            sim = RefSim(topo, PARAMS)
            for s, n, q in power_on_events(net, drive) + [(t, ff.p, int(round(x * drive.ignite)))]:
                sim.add_events(0, [s], [n], [q])
            sim.run(4000)
            assert len(_spikes(sim, ff.p, t, 4000)) <= p_spikes, (x, ph)
            assert len(_spikes(sim, ff.u, t, 4000)) == 0 and len(_spikes(sim, ff.v, t, 4000)) > 35, (x, ph)  # pair untouched
            for cls, ns in readers.items():
                assert all(len(_spikes(sim, n, t, 4000)) == 0 for n in ns), (x, ph, cls)
    # both proxies of the bit at once, at the reader's own margin: still nothing at 1.15x
    for ph in range(0, 47, 4):
        t = 2000 + ph
        sim = RefSim(topo, PARAMS)
        for s, n, q in power_on_events(net, drive) + [(t, f.p, int(round(1.15 * drive.ignite))) for f in R.rails[2]]:
            sim.add_events(0, [s], [n], [q])
        sim.run(4000)
        for cls, ns in readers.items():
            assert all(len(_spikes(sim, n, t, 4000)) == 0 for n in ns), (ph, cls)


# ---------------------------------------------------------------------------------------
# Mix-B campaign on the flip-flop channel. lib/campaign.run_campaign cannot run this build
# as it stands: it hands TorchSim its own bias drift, which REPLACES the topology's flip-flop
# biases (Topology.sim_bias returns the caller's array when given), and it classifies every
# consumer spike after READY as late_activity, which the CLEAR-side v trains always are.
# The loop below is run_campaign with those two lines changed and the power-on pulses added.

def run_ff_campaign(build_fn, params: Params, n_transactions: int, *, batch: int = 200, tx_per_chunk: int = 20,
                    tx_period_steps: int = 4500, pert: Perturbation = MIX_B, seed: int = 0, device: str = "cpu",
                    verbose: bool = True) -> dict:
    import torch

    from drosophilos.sim.lif_torch import TorchSim

    ch = build_fn()
    net, drive = ch.net, ch.drive
    topo = net.topology()
    P, Q = ch.producer, ch.consumer
    taps = Q.rail_taps
    comp, cleared, ready = Q.completion.u, P.ready, Q.ready
    faults = np.array(Q.fault, dtype=np.int64)
    v_members = {ff.v for ff in Q.all_flipflops()} | ({Q.fault_latch.v} if isinstance(Q.fault_latch, FlipFlop) else set())
    consumer_set = {x for x, role in enumerate(net.roles)
                    if role.startswith("Q.") and not role.startswith(("Q.ready", "Q.mon.")) and x not in v_members}
    window = 2 * drive.loop_period_steps
    rng = np.random.default_rng(seed)
    base_q = topo.quanta.astype(np.float64)
    topo_bias = topo.sim_bias() if topo.bias is not None else np.zeros(topo.n)
    power_on = ch.power_on_events()
    totals = {"ok": 0, "ok_stale_retry": 0, "wrong_value": 0, "wrong_value_after_failstop": 0, "fault": 0, "timeout": 0,
              "no_accept": 0, "no_cleared": 0, "no_ready": 0, "late_activity": 0, "wrong_master": 0, "master_fault": 0,
              "no_commit": 0, "cascade": 0}
    timeout_n = P.watchdog.timeout.u if P.watchdog is not None else -1
    lat_all, done, chunk_id, t_start = [], 0, 0, time.time()
    while done < n_transactions:
        B = min(batch, int(np.ceil((n_transactions - done) / tx_per_chunk)))
        K = tx_per_chunk
        q = np.rint(base_q[None, :] * np.exp(rng.normal(0, pert.weight_sigma, size=(B, topo.nnz)))).astype(np.int32)
        vth = params.V_th + rng.normal(0, pert.th_sigma_mv, size=(B, topo.n))
        bias = topo_bias[None, :] + rng.normal(0, pert.bias_sigma_mv, size=(B, topo.n))  # drift ON TOP of the flip-flop bias
        sim = TorchSim(topo, params, n_nodes=B, V_th=vth, bias=bias, quanta=q, device=device, dtype=torch.float64,
                       stray_rate_hz=pert.stray_rate_hz, stray_quanta=pert.stray_quanta, stray_seed=int(rng.integers(2**31 - 1)))
        n_steps = K * tx_period_steps + 200
        loads = np.arange(K) * tx_period_steps + 50
        words = rng.integers(0, 1 << ch.width, size=(B, K))
        for b in range(B):
            st_, ne_, qu_ = [list(x) for x in zip(*power_on)] if power_on else ([], [], [])
            for k in range(K):
                for i, r in rails_for(int(words[b, k]), ch.width):
                    off = int(rng.integers(0, pert.arrival_jitter_steps + 1)) if pert.arrival_jitter_steps else 0
                    st_.append(int(loads[k]) + off); ne_.append(P.rails[i][r].u); qu_.append(drive.ignite)
            sim.add_events(b, st_, ne_, qu_)
        t0 = time.time()
        sim.run(n_steps)
        wall = time.time() - t0
        ev = sim.trace.events
        ev = ev[np.lexsort((ev["step"], ev["node"]))]
        bounds = np.searchsorted(ev["node"], np.arange(B + 1))
        chunk_counts = dict.fromkeys(totals, 0)
        for b in range(B):
            sl = slice(bounds[b], bounds[b + 1])
            res = _decode_node(ev["step"][sl], ev["neuron"][sl], taps, comp, cleared, ready, faults, consumer_set,
                               loads.tolist(), words[b].tolist(), window, None, timeout_n)
            prev_ok = True
            for cls, lat, _ in res:
                # a wrong word right after a non-ok transfer of the same node was loaded before that
                # transfer's READY (the harness loads on a fixed period; the protocol's upstream waits):
                # the stale producer rails, not a misread, make the word. Counted apart from a fresh
                # misread of a clean transfer (docs/a1_flipflop.md, Reading through the proxies).
                if cls == "wrong_value" and not prev_ok:
                    cls = "wrong_value_after_failstop"
                prev_ok = cls in ("ok", "ok_stale_retry")
                chunk_counts[cls] += 1
                if lat is not None:
                    lat_all.append(lat)
        done += B * K
        for k_, v in chunk_counts.items():
            totals[k_] += v
        if verbose:
            errs = {k_: v for k_, v in chunk_counts.items() if k_ != "ok" and v}
            print(f"[ff-campaign] chunk {chunk_id}: {B * K} tx in {wall:.0f} s, errors {errs or 'none'}, done {done}/{n_transactions}", flush=True)
        chunk_id += 1
    n_err = sum(v for k_, v in totals.items() if k_ not in ("ok", "ok_stale_retry"))
    n_wrong = totals["wrong_value"] + totals["wrong_value_after_failstop"]
    return {"transactions": done, "counts": totals, "errors": n_err,
            "detected_by": {"neural": totals["fault"] + totals["timeout"], "harness_only": n_err - totals["fault"] - totals["timeout"],
                            "neural_recovered_retries": totals["ok_stale_retry"]},
            "observed_non_ok_rate": n_err / done, "non_ok_upper_95": clopper_pearson_upper(n_err, done),
            "silent_wrong_value_upper_95": clopper_pearson_upper(n_wrong, done),
            "fresh_wrong_value_upper_95": clopper_pearson_upper(totals["wrong_value"], done),
            "accept_latency_ms": {"mean": float(np.mean(lat_all) * params.dt), "max": float(np.max(lat_all) * params.dt),
                                  "p99": float(np.percentile(lat_all, 99) * params.dt)} if lat_all else None,
            "neurons": net.n, "perturbation_text": pert.describe(), "tx_period_ms": tx_period_steps * params.dt,
            "wall_minutes": (time.time() - t_start) / 60}


def test_mix_b_smoke_20_transfers():
    """The campaign loop above on 4 nodes x 5 transfers: exercises the perturbed TorchSim path
    (bias drift on top of the flip-flop bias, stray input, jitter) in seconds."""
    s = run_ff_campaign(ff_channel, PARAMS, 20, batch=4, tx_per_chunk=5, seed=1, verbose=False)
    assert s["transactions"] == 20 and s["errors"] == 0, s["counts"]


@pytest.mark.slow
def test_mix_b_10000_transfers():
    """10,000 transfers at mix B (w~LN(0,0.04), Vth+-0.2 mV, bias+-0.2 mV, stray 5 Hz x 150 q,
    arrival jitter <= 100 steps), 200 nodes x 20 transfers per chunk. ~15 min on the laptop."""
    s = run_ff_campaign(ff_channel, PARAMS, 10000, seed=0)
    print(s)
    assert s["transactions"] == 10000
    assert s["counts"]["wrong_value"] == 0, s["counts"]  # a clean transfer is never misread
    # measured 2026-09-16 on the four-pulse SET train (Juno 408918): 0 non-ok, 0 wrong in 10,000.
    # Before the lockstep fix — u readout (first build): 48 non-ok in 10,000 — 43 timeouts, 4 no ACCEPT,
    # 1 no READY, all fail-stop, 0 wrong values; proxy readout (this build): 37 non-ok — 21
    # timeouts, 10 no ACCEPT, 1 no CLEARED, 3 no READY, and 2 wrong values, both the transfer
    # after a timeout in the same node, loaded by the fixed-period harness before that
    # transfer's READY (a double reset: the late completion of a lockstepped bit lands 54 ms
    # after the FAULT-ACCEPT through p, 46 through u, past the reset trigger's ~50 ms re-arm).
    # The latch channel's rate is ~1e-5. The fail-stop rate and the after-failstop wrong values
    # are recorded in docs/contracts/ffregister.yaml, not asserted: closing that gap (the
    # lockstep entry of a perturbed pair on SET) is the flip-flop register's next design step,
    # and this test must keep measuring it.
    print("[ff-campaign] non-ok", s["errors"], "of", s["transactions"], s["counts"], flush=True)
    assert s["errors"] < 500, s["counts"]  # a regression beyond 5 % is a broken build, not a margin


# ---------------------------------------------------------------------------------------
# Contract measurement (docs/contracts/ffregister.yaml)

def measure_ffregister_contract(rail_proxy: bool = True) -> dict:
    """`rail_proxy=True`: the register read through the proxies p (the build); False: the
    2026-09-16 first build read at u, kept at the top of the contract for the before/after."""
    from drosophilos.lib.contracts import measure_contract

    drive = Drive.from_params(PARAMS)
    words = [0b1010, 0b0101, 0b1111, 0b0000, 0b0110, 0b1001]  # the channel_4bit contract's words
    ch = ff_channel(rail_proxy=rail_proxy)
    Q = ch.consumer
    tap_name = "p" if rail_proxy else "u"
    v_members = [ff.v for ff in Q.all_flipflops()]
    sim_holder = {}

    def runner():
        sim = powered_sim(ch)
        out = run_transactions(ch, PARAMS, words, sim=sim, max_steps_per_tx=20000)
        sim_holder["sim"], sim_holder["recs"] = sim, out[0]
        return out

    c = measure_contract("ffregister_4bit", ch, PARAMS, words, runner=runner,
                         timing_assumptions=[
                             "READY/CLEARED = 15-hop delay chain (~80 ms) after the reset trigger; must exceed reset settling (~21 ms, 4 pulses) plus the flip-flop's u recovery from 4 x 1.5x loop of inhibition on top of v's park",
                             "edge relays: relay fires ~1.8 ms after the source's first spike, its inhibitor's pulse lands ~5.3 ms after; a relay re-arms only after ~50 ms of source silence",
                             "SET is a train: data relay -> set trigger -> two relays, three ignite pulses 5.3 ms apart into u; u's first spike ~6 ms after the first pulse"
                             + (", p's (what the readers see) ~19-23 ms after it: one park recovery (docs/contracts/flipflop.yaml)" if rail_proxy else ""),
                             "power-on: one loop-strength inhibitory pulse into every flip-flop's u" + (" and p" if rail_proxy else "")
                             + " with the image (Channel.power_on_events); without it a pair fires in lockstep for ever",
                             "loads (upstream DATA) arrive only after READY",
                         ] + (["readers of a rail (valid OR, fault AND, decode taps) take the proxy p, never u: u is inhibitory and no host carries an excitatory read of it; the reset controller still clears through u (inhibitory into inhibitory: sign-correct)",
                               "no veto in this channel reads a consumer rail (the watchdog reads the producer's latch rails; READY has no veto), so p's 22 ms veto lead is not exercised and no delay chain was widened"] if rail_proxy else []),
                         notes=["4-bit transport channel, consumer rails on flip-flops (storage='flipflop')" + (" read through their excitatory proxies p (rail_proxy=True)" if rail_proxy else " read at u (rail_proxy=False, the first build)")
                                + ", flags (valid, tree, completion, fault) on latches by measurement, watchdog 55 hops",
                                "producer is the latch register: every harness loads it with one ignition pulse per rail, which cannot set a flip-flop",
                                "activity counts include the CLEAR-side v members: one member of every flip-flop fires at 213 Hz at all times, so an idle register costs 8 x 213 Hz"]
                               + (["lib/contracts.measure_contract classes a role 'Q.b*.p' as control, not latch (its rule is '.u'/'.v'), so the by-class split moves the proxies' spikes (8 x 213 Hz while SET) from latch to control"] if rail_proxy else [])
                               + ["status: measured 2026-09-16, not frozen; the mix-B campaign line below is carried over from the last slow run (tests/test_ffregister.py::test_mix_b_10000_transfers)"])
    sim, recs = sim_holder["sim"], sim_holder["recs"]
    ev = sim.trace.events
    # quiescence with the v members separated out
    quiet_v = quiet_other = 0
    for k, r in enumerate(recs[:-1]):
        gap = (ev["step"] > r.ready_step) & (ev["step"] < recs[k + 1].load_step)
        isv = np.isin(ev["neuron"], v_members)
        quiet_v += int((gap & isv).sum())
        quiet_other += int((gap & ~isv).sum())
    c["activity"]["quiescent_spikes_between_ready_and_next_load"] = quiet_other
    c["activity"]["v_member_spikes_between_ready_and_next_load"] = f"{quiet_v} (the harness loads at READY, so the gap is empty; a CLEAR rail's v fires at 213 Hz whenever the register is idle)"
    # per-bit cost against the latch register (consumer only, completion included)
    lat_ch = build_channel(PARAMS, 4, watchdog_hops=WATCHDOG_HOPS)
    def consumer_neurons(chan):
        return sum(1 for r in chan.net.roles if r.startswith("Q.") and not r.startswith(("Q.ready", "Q.reset", "Q.faultL", "Q.mon", "Q.wd")))
    def rail_neurons(chan):
        return sum(1 for r in chan.net.roles if r.startswith("Q.b"))
    c["resources"]["consumer_neurons_excluding_reset_ready"] = consumer_neurons(ch)
    c["resources"]["rail_storage_neurons_per_bit"] = rail_neurons(ch) / 4
    c["resources"]["latch_register_neurons"] = lat_ch.net.n
    c["resources"]["latch_register_rail_storage_neurons_per_bit"] = rail_neurons(lat_ch) / 4
    c["resources"]["latch_register_consumer_neurons_excluding_reset_ready"] = consumer_neurons(lat_ch)
    # reset latency: Q reset trigger -> last spike of any rail u (CLEAR done), -> last spike of
    # any rail tap (what a reader sees: p when proxied, u otherwise) and -> READY
    us = [ff.u for pair in Q.rails for ff in pair]
    taps = [t for pair in Q.rail_taps for t in pair]
    def last_after_reset(ns):
        return [((ev["step"][np.isin(ev["neuron"], ns) & (ev["step"] >= r.reset_step) & (ev["step"] < r.ready_step)].max()) - r.reset_step) * PARAMS.dt for r in recs]
    clear, clear_tap = last_after_reset(us), last_after_reset(taps)
    c["reset_latency"] = {"clear_ms_mean": round(float(np.mean(clear)), 1), "clear_ms_max": round(float(np.max(clear)), 1),
                          f"reader_clear_ms_mean_last_{tap_name}_spike": round(float(np.mean(clear_tap)), 1),
                          f"reader_clear_ms_max_last_{tap_name}_spike": round(float(np.max(clear_tap)), 1),
                          "ready_after_reset_trigger_ms": round(float(np.mean([(r.ready_step - r.reset_step) * PARAMS.dt for r in recs])), 1),
                          "set_train_ms": "u's first spike 6.0-6.5 ms after the set trigger's pulse" + ("; p's 18.5-23.1 ms" if rail_proxy else "") + " (docs/a1_flipflop.md)"}
    # retention and clear-at-every-phase (tests above), arrival jitter
    rng = np.random.default_rng(3)
    jit_words = [0b1010, 0b0111, 0b1100, 0b0001, 0b1111]
    jitter = {}
    for span in (100, 400, 1000):
        offs = [[int(x) for x in rng.integers(0, span, size=4)] for _ in jit_words]
        cj = ff_channel()
        rj, _, sj = run_transactions(cj, PARAMS, jit_words, bit_offsets=offs, sim=powered_sim(cj), max_steps_per_tx=20000)
        jitter[f"bits_up_to_{span * PARAMS.dt:.0f}_ms_apart"] = {"transfers": len(jit_words), "correct": sj["correct"],
                                                                 "accept_after_last_bit": all(r.accept_step - r.load_step > max(o) for r, o in zip(rj, offs))}
    c["arrival_jitter_tolerance"] = jitter
    c["retention"] = {"hold_s": 2.0, "decoded_after_hold": "the loaded word, valid", "rate_hz_while_holding": 212.5,
                      "period_steps": drive.loop_period_steps, "measured_by": "tests/test_ffregister.py::test_register_clears_at_every_phase_and_holds_two_seconds"}
    c["reset_at_every_phase"] = {"phases": "all 47 measured; the fast test checks every second one", "train": "4 x 1.5x loop into u only (the reset controller's inh)",
                                 "lockstep_resolved": "yes: tests/test_ffregister.py::test_stray_lockstep_is_healed_by_the_next_reset"}
    src = np.asarray(ch.net.src)
    c["fan_in_out"] = {"rail_u_fan_out": sorted({int((src == u).sum()) for u in us}),
                       f"rail_{tap_name}_fan_out_readers": sorted({int((src == t).sum()) for t in taps}),
                       "set_trigger_fan_in": 1, "reset_inh_fan_out": int((src == Q.reset_inh).sum()),
                       "fan_out_tolerated": c.pop("fan_out_tolerated")}
    c["stray_input_tolerance"] = {
        "single_excitatory_pulse_into_parked_rail_u": "1.1x ignite (5,120 q) never changes it; >= 1.15x drops the pair into LOCKSTEP (both ~105 Hz) at every phase, healed by the next reset; 2.0x SETs it at ~50 % of phases",
        "single_excitatory_pulse_into_set_trigger": "0.55x ignite (2,560 q, ~1.0x single need) never fires it; 0.6x fires the chain and SETs the rail at every phase: the register's stray margin is the trigger's, the same as a latch's u (1.1x need)",
        "single_inhibitory_pulse_into_a_set_rail_u": "2.25x loop (8,147 q) never clears it; 2.5x clears 4 of 12 phases, 3.0x 8 of 12",
        "measured_on": "the consumer register alone, RefSim, phases 0..44 step 4 (12 phases) unless stated",
    }
    if rail_proxy:
        c["stray_input_tolerance"]["single_excitatory_pulse_into_parked_rail_p"] = (
            "1.1x ignite (5,120 q) never fires p; 1.15x fires it once, 1.8x twice, 4.0x three times, the pair untouched; "
            "through the readers nothing follows at any of them (no valid-OR spike, no valid / tree / completion latch, no fault gate): "
            "one p spike is 766 q into the OR against a 2,586 q single need. Only the same stray into BOTH proxies of a bit at once "
            "makes a false valid on that bit, from 1.8x (four OR-input spikes); a false completion would need that on every bit "
            "(measured: 1.8x into all eight proxies at once completes). tests/test_ffregister.py::test_stray_into_proxy_while_clear_gives_no_false_completion")
    c["error_rate"] = {"fast": {"transfers": 200, "decoded_errors": 0, "upper_95": clopper_pearson_upper(0, 200),
                                "test": "tests/test_ffregister.py::test_flipflop_register_200_random_transfers"},
                       "mix_B_10000": "pending: tests/test_ffregister.py::test_mix_b_10000_transfers (slow, ~10 min on the laptop)"}
    c["campaign"] = None
    return c


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    from drosophilos.lib.contracts import write_contract

    path = Path("docs/contracts/ffregister.yaml")
    old = yaml.safe_load(path.read_text()) if path.exists() else {}
    old_proxy = old.get("proxy_readout") or {}
    c = measure_ffregister_contract(rail_proxy=False)  # the first build (read at u): the top of the file, for the before/after
    c["notes"].insert(0, "the u-readout build (rail_proxy=False), kept for the before/after; the shipped build is under proxy_readout")
    cp = measure_ffregister_contract(rail_proxy=True)
    cp["notes"].insert(0, "the shipped build (rail_proxy=True): every reader of a rail takes the proxy p")
    for dst, src_ in ((c, old), (cp, old_proxy)):  # the campaign lines are hand-recorded from the slow test: carry them over
        if src_.get("campaign"):
            dst["campaign"] = src_["campaign"]
        if isinstance(src_.get("error_rate"), dict) and src_["error_rate"].get("mix_B_10000"):
            dst["error_rate"]["mix_B_10000"] = src_["error_rate"]["mix_B_10000"]
    c["proxy_readout"] = cp
    write_contract(c, path)
    print("wrote docs/contracts/ffregister.yaml: u-readout", c["resources"]["neurons"], "neurons; accept", c["latency"], "cycle", c["initiation_interval"])
    print("  proxy readout:", cp["resources"]["neurons"], "neurons; accept", cp["latency"], "cycle", cp["initiation_interval"], "reset", cp["reset_latency"])
