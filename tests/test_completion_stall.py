"""Stage D completion stalls (seed 108): an old master rail survives the master reset.

The Stage D netlists are rebuilt from the repository with the records' options
(base 29,375 / 52,808; rate_robust v2 30,643 / 55,936). `make_perturbed_sim`'s
static draws are then reconstructed for seed 108, B=100 and mix B, in its host
order: weights in topology edge order, then V_th, then bias. They are mapped by
role onto the failing cell rebuilt from the real primitives: stage register and
fault latch, master register, commit/guard/grant, COPY and its selected-rail arms,
and DONE. The mapping is checked edge for edge against the kernel.

The CUDA stray stream is not replayed. Strays here follow the mix-B law: 5 Hz x
150 q on every neuron. Old and new words are the reference values of each failing
transaction; bits 8/9/10 are C/Z/V. Stimuli are the old master word, the new stage
word and one commit pulse. Steps are 0.1 ms.
"""

import collections
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from drosophilos.bench.stage_d import PROGRAM, build, load_kernel
from drosophilos.lib.kernel import _NoProducer, _fault_latch
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.lib.staged import add_staged_commit
from drosophilos.protocol.handshake import add_register, compact_register_resets
from drosophilos.protocol.latch import add_latch, add_reset
from drosophilos.protocol.token import rails_for
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim
from drosophilos.sim.trace import SpikeTrace


PARAMS = Params()
DRIVE = Drive.from_params(PARAMS)
WIDTH = 11  # 8 data bits, C, Z, V
# copy: (rate_robust, cell, state cell with power-up veto, old master word, new stage word)
FAILS = {
    17: (False, "c4_sel", True, 0b01000000000, 0b00000110000),   # tick 728: px 0 -> 48
    74: (False, "c5_sub", False, 0b00011000101, 0b00010111011),  # tick 323: 197 -> 187; C/Z/V = 0
    11: (True, "c10_xor", False, 0b00000000110, 0b00000111011),  # tick 864: 6 -> 59
    24: (True, "c2_sel", False, 0b01000000000, 0b00000110000),   # tick 654: 0 -> 48
}
LOAD, COMMIT = 6000, 10300  # stage load; commit_in event before per-node jitter


@lru_cache(maxsize=None)
def _kernel(rate_robust):
    params = Params()
    pl = build(load_kernel(PROGRAM), params, "generic", rate_robust=rate_robust,
               relight_requests=True, act_hops=11, idle_hops=20, watchdog_hops=170,
               in_watchdog_hops=None, powerup_veto=True, commit_reignite=True,
               retry_clear=False, start_relight_hops=5, relight_repair_delay=True,
               copy_requires_rail=True, request_clear_pulses=4, kernel_kill_pulses=4,
               true_guards=True, drive=replace(Drive.from_params(params), kill_strength=0.75))
    topo = pl.net.topology()
    rng = np.random.default_rng(108)
    q = np.rint(topo.quanta[None, :] * np.exp(rng.normal(0, 0.04, (100, topo.nnz)))).astype(np.int64)
    vth = PARAMS.V_th + rng.normal(0, 0.2, (100, topo.n))
    bias = rng.normal(0, 0.2, (100, topo.n)) + (0.0 if topo.bias is None else np.asarray(topo.bias))
    return pl.net.roles, topo, q, vth, bias


def _map(copy, net, *, rate_robust=None):
    """Copy `copy`'s draws for every edge/neuron of `net`, matched by role and delay."""
    roles, kt, kq, kvth, kbias = _kernel(FAILS[copy][0] if rate_robust is None else rate_robust)
    index = {r: i for i, r in enumerate(roles)}
    gid = np.array([index[r] for r in net.roles])
    members = np.zeros(len(roles), bool)
    members[gid] = True
    inside = np.flatnonzero(members[kt.src] & members[kt.dst])
    pool = collections.defaultdict(list)
    for e in inside:
        pool[(int(kt.src[e]), int(kt.dst[e]), int(kt.delay[e]))].append(e)
    topo = net.topology()
    q = np.empty(topo.nnz, np.int64)
    for e in range(topo.nnz):
        k = pool[(int(gid[topo.src[e]]), int(gid[topo.dst[e]]), int(topo.delay[e]))].pop(0)
        assert kt.quanta[k] == topo.quanta[e]
        q[e] = kq[copy, k]
    assert not any(pool.values()), "kernel edges among these neurons that the primitives lack"
    return q, kvth[copy, gid], kbias[copy, gid], gid


def _cell(copy, nominal=False, compact=False):
    rate_robust, name, state, old, new = FAILS[copy]
    drive = replace(DRIVE, kill_pulses=4, rate_robust=rate_robust)
    net = Netlist(PARAMS)
    Q = add_register(net, drive, f"{name}.Q", WIDTH, with_completion=True)
    _fault_latch(net, drive, name, Q)
    reg = add_staged_commit(net, drive, name, Q, _NoProducer(), ordered_grant=True,
                            copy_requires_rail=True)
    if state:  # lib/kernel.py's power-up veto: lit by the image, dark after the first commit
        root = net.roles[reg.master.completion.u][: -len(".L.u")]
        veto0 = add_latch(net, drive, f"{name}.M.comp.veto0")
        net.synapse(veto0.u, net.roles.index(f"{root}.ign.edge"), -int(round(2.2 * drive.loop)))
        for x in veto0.members:
            net.synapse(reg.master.reset_inh, x, -int(round(0.75 * drive.loop)))
    q, vth, bias, gid = _map(copy, net)
    if nominal:
        q, vth, bias = net.topology().quanta.astype(np.int64), np.full(net.n, PARAMS.V_th), np.asarray(net.bias)
    if compact:
        before = net.topology()
        compact_register_resets(net, drive, [Q, reg.master])
        after = net.topology()
        assert np.array_equal(before.src, after.src) and np.array_equal(before.dst, after.dst)
        q = np.rint(q * after.quanta / before.quanta).astype(np.int64)
    return dict(net=net, Q=Q, reg=reg, M=reg.master, q=q, vth=vth, bias=bias, old=old, new=new,
                name=name, gid=gid)


def _observed_run(sim, steps, neurons):
    """Keep only observed spikes, without changing RefSim's integration/delivery.

    RefSim appends in (step, node, neuron) order already. Avoid sorting millions
    of irrelevant spikes and bound retained recording memory during the survey.
    """
    wanted = np.zeros(sim.n, bool)
    wanted[list(neurons)] = True
    # add_events stores one tuple per node/step. Deliver the same integer events
    # together, so a batched noisy run does not dispatch thousands of tiny arrays.
    for step, groups in sim._events.items():
        if len(groups) > 1:
            sim._events[step] = [tuple(np.concatenate(column) for column in zip(*groups))]
    columns = [[], [], []]
    for start in range(0, steps, 256):
        sim.run(min(256, steps - start))
        if sim._spk_step:
            s, b, n = map(np.concatenate, (sim._spk_step, sim._spk_node, sim._spk_neuron))
            keep = wanted[n]
            for out, values in zip(columns, (s, b, n)):
                out.append(values[keep])
        sim._spk_step.clear(); sim._spk_node.clear(); sim._spk_neuron.clear()
    if not columns[0]:
        return SpikeTrace.empty()
    return SpikeTrace(np.rec.fromarrays([np.concatenate(c) for c in columns],
                                        names="step,node,neuron"))


def _run(c, seeds, *, strays=True, quanta=None, steps=COMMIT + 7100):
    """Node b replays default_rng(seeds[b]): commit jitter, stage-load offsets, strays."""
    net, M, Q = c["net"], c["M"], c["Q"]
    topo = net.topology()
    q = np.broadcast_to(c["q"], (len(seeds), topo.nnz)) if quanta is None else quanta
    sim = RefSim(topo, PARAMS, n_nodes=len(seeds), V_th=c["vth"], bias=c["bias"], quanta=q)
    commits = []
    for b, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        tc = COMMIT + int(rng.integers(0, 600))
        ev = [(10, M.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["old"], WIDTH)]
        ev += [(LOAD + int(rng.integers(0, 101)), Q.rails[i][r].u, DRIVE.ignite)
               for i, r in rails_for(c["new"], WIDTH)]
        ev.append((tc, c["reg"].commit_in, DRIVE.ignite))
        st, ne, qu = (list(x) for x in zip(*ev))
        if strays:
            s_, n_ = np.nonzero(rng.random((steps, net.n)) < 5.0 * PARAMS.dt / 1000)
            st += s_.tolist(); ne += n_.tolist(); qu += [150] * len(s_)
        sim.add_events(b, st, ne, qu)
        commits.append(tc)
    watched = [M.ready, c["reg"].done_relay, *M.fault, *(v.u for v in M.valid)]
    watched += [l.u for pair in M.rails for l in pair]
    return _observed_run(sim, steps, watched), commits


def _outcome(c, trace, node, tc):
    """DONE after the rewrite, old rails still firing at master READY, fault-gate spikes."""
    # Slice once: scanning every other node for every bit made the survey quadratic.
    trace, node = trace.node(node), 0
    M, old = c["M"], dict(rails_for(c["old"], WIDTH))
    ready = trace.neuron_steps(M.ready, node)
    ready = int(ready[ready > tc][0])
    def live(n, lo, hi):
        s = trace.neuron_steps(n, node)
        return bool(np.any((s >= lo) & (s < hi)))
    faults = {i: trace.neuron_steps(M.fault[i], node) - ready for i in range(WIDTH)}
    return dict(
        done=bool(np.any(trace.neuron_steps(c["reg"].done_relay, node) > tc + 1000)),
        ready=ready - tc,
        survivors=[i for i in range(WIDTH) if live(M.rails[i][old[i]].u, ready - 300, ready)],
        valid_dark=[i for i in range(WIDTH) if not live(M.valid[i].u, ready + 1500, ready + 3500)],
        faults={i: f[f > 0] for i, f in faults.items() if np.any(f > 0)},
    )


def _immune(c, bits):
    """Per node, drop the master reset's fan-out onto one old rail: a guaranteed survivor."""
    topo, M, old = c["net"].topology(), c["M"], dict(rails_for(c["old"], WIDTH))
    q = np.broadcast_to(c["q"], (len(bits), topo.nnz)).copy()
    for b, i in enumerate(bits):
        if i is not None:
            q[b, (topo.src == M.reset_inh) & np.isin(topo.dst, M.rails[i][old[i]].members)] = 0
    return q


# ---- reduced primitive: one master rail latch and the real 4-tap master reset ------------
def _reset_latch(copy, bit, *, nominal=False):
    name, old = FAILS[copy][1], FAILS[copy][3]
    rail = (old >> bit) & 1
    net = Netlist(PARAMS)
    latch = add_latch(net, DRIVE, f"{name}.M.b{bit}r{rail}")
    trigger, inh, _ = add_reset(net, DRIVE, f"{name}.M", [latch])
    q, vth, bias, _ = _map(copy, net)
    topo = net.topology()
    if nominal:
        q, vth, bias = topo.quanta.astype(np.int64), np.full(net.n, PARAMS.V_th), np.zeros(net.n)
    return topo, latch, trigger, inh, q, vth, bias


def _strayed_survival(copy, bit, n, **kw):
    topo, latch, trigger, inh, q, vth, bias = _reset_latch(copy, bit, **kw)
    rng = np.random.default_rng(108)
    sim = RefSim(topo, PARAMS, n_nodes=n, V_th=vth, bias=bias, quanta=np.broadcast_to(q, (n, topo.nnz)))
    resets = rng.integers(3000, 5000, n)
    for b, t in enumerate(resets):
        s_, n_ = np.nonzero(rng.random((6800, topo.n)) < 5.0 * PARAMS.dt / 1000)
        sim.add_events(b, [10, int(t)] + s_.tolist(), [latch.u, trigger] + n_.tolist(),
                       [DRIVE.ignite, DRIVE.relay_in] + [150] * len(s_))
    trace = _observed_run(sim, 6800, [latch.u])
    events = trace.events
    live = (events["neuron"] == latch.u) & (events["step"] > resets[events["node"]] + 1500)
    return len(np.unique(events["node"][live]))


# ---- tests ---------------------------------------------------------------------------------
@pytest.mark.parametrize("copy", sorted(FAILS))
def test_isolated_path_is_the_kernel_subgraph(copy):
    """Every primitive edge has its kernel edge with the same nominal quanta (asserted in
    _map); outside inputs reach only the stage rails and commit_in, never M/COPY/DONE."""
    c = _cell(copy)
    roles, kt, *_ = _kernel(FAILS[copy][0])
    inside = np.zeros(len(roles), bool)
    inside[c["gid"]] = True
    entering = {roles[d] for s, d in zip(kt.src, kt.dst) if inside[d] and not inside[s]}
    stage_rails = {f"{c['name']}.Q.b{i}r{r}.u" for i in range(WIDTH) for r in (0, 1)}
    assert f"{c['name']}.commit_in" in entering
    assert entering - {f"{c['name']}.commit_in"} <= stage_rails


@pytest.mark.parametrize("copy, spikes", [(17, 4), (74, 4), (24, 4), (11, 5), (None, 5)])
def test_master_reset_inhibitor_spike_count(copy, spikes):
    """Four configured taps nominally emit five spikes; three failing copies' draws drop the
    residual fifth (copy 28's request clear did the same). Copy 11 keeps five."""
    topo, latch, trigger, inh, q, vth, bias = _reset_latch(copy or 74, 3, nominal=copy is None)
    sim = RefSim(topo, PARAMS, V_th=vth, bias=bias, quanta=q[None, :])
    sim.add_events(0, [10, 3000], [latch.u, trigger], [DRIVE.ignite, DRIVE.relay_in])
    sim.run(4000)
    assert len(sim.trace.neuron_steps(inh)) == spikes


def test_noise_free_cell_commits_at_failing_draws():
    """Without strays copy 74's path completes on the capture's schedule (commit -> master
    READY 1,608 steps in the capture; READY -> DONE 2,652 in its previous transaction)."""
    c = _cell(74)
    trace, (tc,) = _run(c, [0], strays=False)
    out = _outcome(c, trace, 0, tc)
    assert out["done"] and not out["survivors"] and not out["faults"]
    assert 1550 < out["ready"] < 1650
    done = trace.neuron_steps(c["reg"].done_relay)
    assert 2400 < int(done[-1]) - tc - out["ready"] < 2800


def test_surviving_rail_blocks_completion_and_reproduces_copy74_fault3():
    """A bit-3 old rail kept alive through the reset (copy 74's draws) leaves valid3 dark,
    so W_M and DONE never return, while the new b3r1 makes fault3 fire as captured: first
    spike 528 steps after master READY, then every 382-385 steps. At nominal draws the
    valid relay re-fires after the reset, and the same survivor completes."""
    c = _cell(74)
    trace, commits = _run(c, [0, 0], strays=False, quanta=_immune(c, [3, None]))
    blocked, control = (_outcome(c, trace, b, commits[b]) for b in (0, 1))
    assert control["done"] and not control["faults"]
    assert not blocked["done"] and blocked["survivors"] == [3] and blocked["valid_dark"] == [3]
    assert list(blocked["faults"]) == [3]
    f3 = blocked["faults"][3]
    assert 450 < f3[0] < 650 and np.all((np.diff(f3[:6]) > 360) & (np.diff(f3[:6]) < 410))
    n = _cell(74, nominal=True)
    trace, commits = _run(n, [0], strays=False, quanta=_immune(n, [3]))
    out = _outcome(n, trace, 0, commits[0])
    assert out["done"] and out["survivors"] == [3] and out["valid_dark"] == []


def test_replayed_realization_stalls_only_at_copy74_draws():
    """One mix-B stray realization (seed 23 of the 1,600-realization campaign): at copy
    74's draws the old b3r0 survives the shipped reset and the cell stalls with fault3;
    the identical stimulus and strays at nominal draws commit normally."""
    c = _cell(74)
    trace, (tc,) = _run(c, [23])
    out = _outcome(c, trace, 0, tc)
    assert not out["done"] and out["survivors"] == [3] and out["valid_dark"] == [3]
    assert list(out["faults"]) == [3]
    n = _cell(74, nominal=True)
    trace, (tc,) = _run(n, [23])
    out = _outcome(n, trace, 0, tc)
    assert out["done"] and not out["survivors"] and not out["faults"]


def test_copy74_rail_survives_the_real_reset_only_at_its_draws():
    """2,000 strayed resets at random phases, real add_reset controller: copy 74's b3r0
    (loop +6/+13 %, u V_th -0.46 mV, four inhibitor spikes) survives; the nominal latch
    and controller never do. Compact-policy surveys live in test_robust_reset."""
    assert _strayed_survival(74, 3, 2000) == 8
    assert _strayed_survival(74, 3, 2000, nominal=True) == 0


@pytest.mark.slow
@pytest.mark.parametrize("copy,expected,surviving", [(74, 24, 50), (17, 4, 4), (24, 1, 1), (11, 0, 0)])
@pytest.mark.parametrize("setting", ["draws", "nominal", "compact"])
def test_1600_seed_completion_survey(copy, expected, surviving, setting):
    """Exact committed survey, including healthy double-railed copy-74 outcomes.

    Static draws and each node's RNG stream are unchanged by batching or compacting.
    This is a conditional RefSim experiment, not a replay of the CUDA stray stream.
    """
    c = _cell(copy, nominal=setting == "nominal", compact=setting == "compact")
    stalls = survivors = 0
    for lo in range(0, 1600, 100):
        trace, commits = _run(c, range(lo, lo + 100))
        for b, tc in enumerate(commits):
            out = _outcome(c, trace, b, tc)
            survivors += bool(out["survivors"])
            if not out["done"]:
                stalls += 1
                assert len(out["survivors"]) == 1 and out["valid_dark"] == out["survivors"]
        print(dict(copy=copy, setting=setting, seeds=lo + 100, stalls=stalls,
                   survivors=survivors), flush=True)
    assert stalls == (expected if setting == "draws" else 0)
    assert survivors == (surviving if setting == "draws" else 0)
