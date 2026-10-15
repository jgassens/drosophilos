"""Experimental operand one-shot: bounded local probes and separate cluster cases.

No campaign qualification is claimed by these laptop tests. Prescribed trains
are synaptic-boundary inputs; receivers, spent memory and resets are real LIFs.
"""

import os
from time import perf_counter, process_time

import numpy as np
import pytest

from drosophilos.lib.gates import Gates, Rail2, Tap
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.latch import add_experimental_autapses, add_reset
from drosophilos.sim.model import Topology
from drosophilos.sim.ref64 import RefSim
from test_ordered_xor import P, SurveySim, arithmetic_survey, prescribe, replay_maps


SCENARIOS = ("second-pulse", "order-15ms", "order-100ms", "mix-B", "copy85",
             "addsub4-mix-B", "addsub8-mix-B", "addsub16-nominal")
# All distinct copies present in the cluster logs' capped failure_details.
# (width, copy, tx, first reported wrong bit, a, b, sub)
REPLAY_CASES = {
    "replay8-copy175": (8, 175, 0, 2, 14, 161, 1),
    "replay8-copy391": (8, 391, 0, 2, 238, 1, 1),
    "replay8-copy418": (8, 418, 1, 4, 103, 10, 1),
    "replay8-copy497": (8, 497, 1, 8, 131, 137, 0),
    "replay8-copy536": (8, 536, 0, 7, 72, 218, 0),
    "replay8-copy699": (8, 699, 0, 2, 233, 198, 1),
    "replay4-copy75": (4, 75, 1, 3, 5, 2, 1),
    "replay4-copy140": (4, 140, 0, 4, 4, 6, 0),
    "replay4-copy338": (4, 338, 1, 4, 9, 10, 0),
}
SCENARIOS += (*REPLAY_CASES, "replay85-clear2174")
# Every distinct copy in the v2 logs' capped survivor_details. Counts are
# latch MEMBERS, over all four transactions (one for copy85).
# scenario, copy, v2 survivors, sum-only F survivors, v2 role, surviving txs
SURVIVOR_CASES = {
    "survivor-second-pulse-copy406": ("second-pulse", 406, 8, 8, "c0", (0, 1, 2, 3)),
    "survivor-order-100ms-copy406": ("order-100ms", 406, 2, 6, "c0", (1,)),
    "survivor-order-100ms-copy1098": ("order-100ms", 1098, 6, 0, "s.once0.spent", (0, 1, 3)),
    "survivor-order-15ms-copy406": ("order-15ms", 406, 8, 8, "c0", (0, 1, 2, 3)),
    "survivor-copy85-copy3725": ("copy85", 3725, 2, 2, "s.y0", (0,)),
    "survivor-copy85-copy4082": ("copy85", 4082, 2, 2, "s.y1", (0,)),
    "survivor-copy85-copy11001": ("copy85", 11001, 2, 0, "s.y1", (0,)),
    "survivor-copy85-copy16939": ("copy85", 16939, 2, 0, "s.y0", (0,)),
}
SCENARIOS += tuple(SURVIVOR_CASES)
FAILURES = ("double", "missing", "wrong", "repeated", "ignition_repeat", "survivors")


class OperandV1Gates(Gates):
    """Frozen v1 wiring via the unchanged default ripple and explicit x one-shots.

    Diagnostic fixture only: production record replay rejects v1.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operand_xor_oneshot = False

    def xor2_ordered(self, name, early, late, *, oneshot=False):
        return super().xor2_ordered(name, early, late, oneshot=oneshot or name.endswith(".x"))


class OperandV2Gates(Gates):
    """Frozen v2: shared carry pulses, but only operand storage is strengthened."""

    def finalize_operand_reset(self, inhibitor):
        members = self._operand_reset_members
        self._operand_reset_members = {m for m in members if ".x." in self.net.roles[m]}
        try:
            super().finalize_operand_reset(inhibitor)
        finally:
            self._operand_reset_members = members


def operand_fixture(enabled=True, gates_type=Gates):
    net, drive = Netlist(P), Drive.from_params(P)
    early = Rail2(Tap(net.neuron("early0")), Tap(net.neuron("early1")))
    late = Rail2(Tap(net.neuron("late0")), Tap(net.neuron("late1")))
    cin = Rail2(Tap(net.neuron("cin0")), Tap(net.neuron("cin1")))
    g = gates_type(net, drive, ordered_xor_oneshot=True, operand_xor_oneshot=enabled)
    _, _, xs, _ = g.ripple_adder_ordered("add", [late], [early], cin)
    reset = add_reset(net, drive, "reset", g.latches, g.gates)
    g.finalize_operand_reset(reset[1])
    add_experimental_autapses(net, drive)
    return net, drive, early, late, xs[0], reset, g


def copy85_receiver(*, nominal=False, first_only=False):
    """Frozen reconstructed receiving draws and captured arrivals, origin 198000.

    The original sorted topology has 123323 neurons / 249940 edges, mix B,
    seed 110, B=100, row 85. Values are preserved from the diagnostic report,
    independent of changed topology IDs or newly allocated random draws.
    """
    sim = RefSim(Topology.empty(1), P, V_th=[-45.17450109365213],
                 bias=[.022141032716653797])
    sim.V[:] = P.E_L + sim.bias
    excitation, veto, source_inhibition = ((4655, -1810, -7966) if nominal else
                                           (4919, -1792, -7663))
    times = [566] if first_only else [566, 607, 645]
    sim.add_events(0, times, [0] * len(times), [excitation] * len(times))
    sim.add_events(0, [480, 548, 614], [0] * 3, [veto] * 3)
    sim.add_events(0, [624, 671], [0] * 2, [source_inhibition] * 2)
    sim.run(700)
    return sim.trace.events["step"].tolist()


def carry_escape_receiver(copy, *, first_only=False, background=True):
    """Cold receiving-neuron controls from the exact arithmetic replays.

    Origins are 1000 (copy 175) and 6000 (copy 75). Copy 75 includes its
    captured 150-q background arrival at 6371; removing it delays the escape
    one step in this reduced experiment. These are sufficient local mechanisms,
    not claims that a cold receiver reconstructs every previous membrane state.
    """
    if copy == 175:
        threshold, bias = -44.98305837509053, -.34870083385449524
        excitation, inhibition, veto = 4491, -7631, -1674
        veto_times, inhibition_time = [418, 492], 536
    else:
        assert copy == 75
        threshold, bias = -45.18410708610765, -.046470447175844365
        excitation, inhibition, veto = 4489, -7403, -1622
        veto_times, inhibition_time = [435, 508], 532
    sim = RefSim(Topology.empty(1), P, V_th=[threshold], bias=[bias])
    sim.V[:] = P.E_L + bias
    times = [478] if first_only else [478, 522]
    sim.add_events(0, times, [0] * len(times), [excitation] * len(times))
    sim.add_events(0, veto_times, [0, 0], [veto, veto])
    sim.add_events(0, [inhibition_time], [0], [inhibition])
    if copy == 75 and background:
        sim.add_events(0, [371], [0], [150])
    sim.run(550)
    return sim.trace.events["step"].tolist()


def operand_survey(scenario, trials, *, enabled=True, replay_copies=None, gates_type=Gates,
                   reference_type=None, reference_enabled=None, rearm=False):
    net, drive, early, late, out, reset, g = operand_fixture(enabled, gates_type)
    topo = net.topology()
    replay_nodes = replay_edges = None
    draw_topo = topo
    if reference_type is not None or reference_enabled is not None:
        reference = operand_fixture(enabled if reference_enabled is None else reference_enabled,
                                    reference_type or Gates)[0]
        draw_topo = reference.topology()
        if reference_enabled is None or reference_enabled == enabled:
            replay_nodes, replay_edges, _ = replay_maps(net, reference)
        else:
            # Option-off control: keep every surviving role/receiving-edge draw
            # from the v2 log, including carry/sum storage and the controller.
            assert not enabled and reference_enabled
            roles = {r: i for i, r in enumerate(reference.roles)}
            replay_nodes = np.array([roles[r] for r in net.roles])
            keys = {(reference.roles[s], reference.roles[d], int(dt)): e
                    for e, (s, d, dt) in enumerate(zip(draw_topo.src, draw_topo.dst, draw_topo.delay))}
            edges = []
            for s, d, dt, q in zip(topo.src, topo.dst, topo.delay, topo.quanta):
                key = (net.roles[s], net.roles[d], int(dt))
                if key in keys:
                    edges.append(keys[key])
                else:
                    matches = [e for e, (rd, rq, rdt) in enumerate(
                        zip(draw_topo.dst, draw_topo.quanta, draw_topo.delay))
                        if reference.roles[rd] == net.roles[d] and rq > 0 and q > 0 and rdt == dt]
                    assert len(matches) == 1, key
                    edges.append(matches[0])
            replay_edges = np.array(edges)
            assert len(set(replay_nodes)) == net.n
            assert len(set(replay_edges)) == topo.nnz
    rng = np.random.default_rng(8585)
    copies = trials if replay_copies is None else len(replay_copies)
    result = dict(copies=copies, trials=copies * (1 if scenario == "copy85" else 4),
                  double=0, missing=0, wrong=0, repeated=0,
                  ignition_repeat=0, survivors=0, min_latency=10000, max_latency=0,
                  survivor_details=[])
    for start in range(0, trials, 256):
        if replay_copies is not None and start > max(replay_copies):
            break
        size = min(256, trials - start)
        gaussian = scenario == "mix-B"
        w = (rng.normal(0, .04, (size, draw_topo.nnz)) if gaussian else
             rng.choice([-.12, .12], (size, draw_topo.nnz)))
        th = (rng.normal(0, .2, (size, draw_topo.n)) if gaussian else
              rng.choice([-.6, .6], (size, draw_topo.n)))
        bias = (rng.normal(0, .2, (size, draw_topo.n)) if gaussian else
                rng.choice([-.6, .6], (size, draw_topo.n)))
        if replay_nodes is not None:
            w, th, bias = w[:, replay_edges], th[:, replay_nodes], bias[:, replay_nodes]
        # Draw all transactions' jitter even in skipped blocks. Copy indices
        # refer to the original 40,000-copy run, never to fresh RNG seeds.
        if scenario == "copy85":
            jitters = [rng.integers(-5, 6, (size, 1))]
            jitters[0][0] = 0
        else:
            jitters = [rng.integers(0, 6, (size, 7)) for _ in range(4)]
        if replay_copies is not None:
            selected = [c - start for c in replay_copies if start <= c < start + size]
            if not selected:
                continue
            w, th, bias = (x[selected] for x in (w, th, bias))
            jitters = [j[selected] for j in jitters]
            size = len(selected)
        sim = SurveySim(topo, P, n_nodes=size, quanta=np.rint(topo.quanta * np.exp(w)),
                        V_th=P.V_th + th, bias=np.asarray(net.bias) + bias)
        # Each physical copy visits every truth-table row, changes polarities,
        # holds consumption over long gaps and reloads after real reset.
        for tx in range(4 if scenario != "copy85" else 1):
            origin = sim.step_index
            ev, lv = tx // 2, tx % 2
            first = 1000 if scenario == "order-100ms" else 200 if scenario == "second-pulse" else 150
            spacing = 41 if scenario == "second-pulse" else 30
            lt = np.array([first, first + spacing, first + 2 * spacing, first + 3 * spacing,
                           first + 1000, first + 1042, first + 2300])
            if scenario == "copy85":
                ev = lv = 0
                # Captured delayed acc6 emissions. Three real captured pulses
                # suffice for the escape; added late pulses probe retention.
                lt = np.array([548, 589, 627, 1600, 1642, 2700])[None, :] + jitters[tx]
                wrong = net.roles.index("add.fa0.x.y1.a.edge")
                sim.V_th[:, wrong] = -45.17450109365213
                sim.bias[:, wrong] = .022141032716653797
                weights = {".edge": 4919, ".edge_inh": -7663, ".veto": -1792}
                for e, (src, dst) in enumerate(zip(topo.src, topo.dst)):
                    if dst == wrong:
                        role = net.roles[src]
                        sim.quanta[:, e] = (weights[".veto"] if role.endswith(".veto") else
                                            weights[".edge_inh"] if role.endswith(".edge_inh") else
                                            weights[".edge"])
                veto = net.roles.index("add.fa0.x.y1.a.veto")
                sim.silenced[:, veto] = True
                prescribe(sim, veto, [462, 530, 596] + list(range(662, 3200, 66)))
                if not enabled:
                    inh = net.roles.index("add.fa0.x.y1.a.edge_inh")
                    sim.silenced[:, inh] = True
                    prescribe(sim, inh, [606, 653] + list(range(700, 3200, 47)))
                prescribe(sim, early.taps[ev], np.arange(404, 3200, 60))
            else:
                # Minimum 15-ms EARLY lead, with independent pulse jitter, and
                # a much earlier settled EARLY. Reversed EARLY/LATE arrival is
                # outside the ordered primitive's contract, not qualified here.
                lt = lt[None, :] + jitters[tx]
                prescribe(sim, early.taps[ev], origin + np.arange(0, 4000, 60))
            if tx == 0:
                sim.V[:] = P.E_L + sim.bias
            before = sim.counts.copy()
            sim.first.fill(-1)
            prescribe(sim, late.taps[lv], origin + lt)
            sim.run(4200)
            counts = sim.counts - before
            seen = counts[:, out.taps] > 0
            result["double"] += int(seen.all(axis=1).sum())
            result["missing"] += int((~seen.any(axis=1)).sum())
            result["wrong"] += int(seen[:, 1 - (ev ^ lv)].sum())
            edges = [i for i, r in enumerate(net.roles) if r.startswith("add.fa0.x.y") and r.endswith(".edge")]
            result["ignition_repeat"] += int((counts[:, edges].sum(axis=1) > 1).sum())
            if enabled:
                once = [net.roles.index(f"add.fa0.x.once{r}.edge") for r in (0, 1)]
                result["repeated"] += int((counts[:, once].sum(axis=1) != 1).sum())
            lat = sim.first[:, out.taps[ev ^ lv]] - origin - lt[:, 0]
            lat = lat[sim.first[:, out.taps[ev ^ lv]] >= 0]
            if len(lat):
                result["min_latency"] = min(result["min_latency"], int(lat.min()))
                result["max_latency"] = max(result["max_latency"], int(lat.max()))
            for node in range(size):
                sim.add_events(node, [origin + 4200], [reset[0]], [drive.ignite])
            sim.run(2000)
            members = [m for latch in g.latches for m in latch.members]
            result["survivors"] += int((sim.last[:, members] >= origin + 5900).sum())
            for node, member in zip(*np.nonzero(sim.last[:, members] >= origin + 5900)):
                if len(result["survivor_details"]) < 8:
                    result["survivor_details"].append(dict(
                        copy=start + (int(node) if replay_copies is None else selected[node]), tx=tx,
                        role=net.roles[members[member]], last=int(sim.last[node, members[member]]),
                        reset_count=int(sim.counts[node, reset[1]]), reset_last=int(sim.last[node, reset[1]])))
            if rearm:
                assert scenario == "copy85" and enabled
                # On the frozen clear draw, reload the opposite EARLY polarity
                # after the same 2000-step recovery; the old captured veto has
                # drained and is correctly silent for this new truth-table row.
                reload_origin = sim.step_index
                before = sim.counts.copy()
                sim.first.fill(-1)
                prescribe(sim, early.r1.u, reload_origin + np.arange(0, 4000, 60))
                prescribe(sim, late.r0.u, reload_origin + np.array([200, 242, 284, 2500, 2542, 3500]))
                sim.run(4200)
                counts = sim.counts - before
                seen = counts[:, out.taps] > 0
                result["trials"] += size
                result["double"] += int(seen.all(axis=1).sum())
                result["wrong"] += int(seen[:, 0].sum())
                result["missing"] += int((~seen[:, 1]).sum())
                result["repeated"] += int((counts[:, once].sum(axis=1) != 1).sum())
                result["ignition_repeat"] += int((counts[:, edges].sum(axis=1) > 1).sum())
                latency = sim.first[:, out.r1.u] - reload_origin - 200
                latency = latency[sim.first[:, out.r1.u] >= 0]
                if len(latency):
                    result["min_latency"] = min(result["min_latency"], int(latency.min()))
                    result["max_latency"] = max(result["max_latency"], int(latency.max()))
                for node in range(size):
                    sim.add_events(node, [reload_origin + 4200], [reset[0]], [drive.ignite])
                sim.run(2000)
                result["survivors"] += int((sim.last[:, members] >= reload_origin + 5900).sum())
    return result


def test_copy85_reconstruction_and_suppression():
    assert copy85_receiver() == [621]
    assert copy85_receiver(nominal=True) == []
    assert copy85_receiver(first_only=True) == []
    legacy = operand_survey("copy85", 1, enabled=False)
    assert legacy["double"] == legacy["wrong"] == 1, legacy
    fixed = operand_survey("copy85", 1)
    assert all(fixed[key] == 0 for key in FAILURES), fixed


def test_operand_spent_latches_are_in_stage_reset_domain():
    net, _, _, _, _, reset, g = operand_fixture()
    spent = {m for latch in g.latches if ".x.once" in net.roles[latch.u] for m in latch.members}
    targets = {d for s, d, q in zip(net.src, net.dst, net.quanta) if s == reset[1] and q < 0}
    assert len(spent) == 4
    assert spent <= targets
    protected = {m for latch in g.latches for m in latch.members}
    assert len(protected) == 20  # operand (8), carry (4), sum/spent (8)
    for s, d, q in zip(net.src, net.dst, net.quanta):
        if s == reset[1] and d in protected:
            assert q == -round(1.5 * g.drive.loop)
    assert not any(".bx" in r and ".once" in r for r in net.roles)


def test_operand_probe_clear_matches_kernel_stage():
    from drosophilos.lib.kernel import build_pipeline
    pl = build_pipeline(P, 1, [dict(name="out", op="ADD", a="input", b=("const", "k"))],
                        consts={"k": 0}, rate_robust=True, experimental_autapse=True,
                        ordered_xor_oneshot=True)
    net, drive, _, _, _, reset, _ = operand_fixture(False)
    def controller_edges(net, trigger):
        prefix = net.roles[trigger].removesuffix(".reset") + "."
        nodes = {i for i, r in enumerate(net.roles)
                 if r.startswith(prefix) and r.removeprefix(prefix).startswith("reset")}
        return {(net.roles[s].removeprefix(prefix), net.roles[d].removeprefix(prefix), q, dt)
                for s, d, q, dt in zip(net.src, net.dst, net.quanta, net.delay)
                if s in nodes and d in nodes}
    stage = pl.cells[0].stage
    # The real stage controller comes from add_register -> add_reset (four
    # configured taps); extend_reset claims its carry/sum latches at .75 loop.
    assert controller_edges(net, reset[0]) == controller_edges(pl.net, stage.reset_trigger)
    assert {r for s, r, q, dt in controller_edges(net, reset[0]) if "relay" in r} == {
        "reset_relay1", "reset_relay2", "reset_relay3"}
    for suffix in (".fa0.c0.u", ".fa0.s.y0.u", ".fa0.s.once0.spent.u"):
        member, = [i for i, r in enumerate(pl.net.roles) if r.endswith(suffix)]
        weights = [pl.net.quanta[e] for e in pl.net.incoming[member]
                   if pl.net.src[e] == stage.reset_inh]
        assert weights == [-round(.75 * drive.loop)]


def test_operand_v3_changes_only_adder_storage_clear_weights():
    old, _, _, _, _, reset, _ = operand_fixture(gates_type=OperandV2Gates)
    new, drive, *_ = operand_fixture()
    for field in ("roles", "src", "dst", "delay", "bias"):
        assert getattr(old, field) == getattr(new, field)
    changed = [(s, new.roles[d], q0, q1) for s, d, q0, q1 in
               zip(new.src, new.dst, old.quanta, new.quanta) if q0 != q1]
    assert len(changed) == 12
    assert {role for s, role, q0, q1 in changed} == {
        f"add.fa0.{part}.{member}" for part in
        ("c0", "c1", "s.y0", "s.y1", "s.once0.spent", "s.once1.spent") for member in ("u", "v")}
    assert all(s == reset[1] and q0 == -round(.75 * drive.loop) and
               q1 == -round(1.5 * drive.loop) for s, role, q0, q1 in changed)


@pytest.mark.parametrize("mirror_before_reset", [False, True])
def test_operand_reset_strengthens_reader_mirror(mirror_before_reset):
    net, drive = Netlist(P), Drive.from_params(P)
    g = Gates(net, drive, ordered_xor_oneshot=True, operand_xor_oneshot=True)
    inputs = [Rail2(Tap(net.neuron(f"input{k}r0")), Tap(net.neuron(f"input{k}r1")))
              for k in range(3)]
    g.ripple_adder_ordered("add", [inputs[0]], [inputs[1]], inputs[2])
    member = net.roles.index("add.fa0.s.y0.u")
    reader = net.neuron("sum.reader")
    gain = 1.7
    if mirror_before_reset:
        net.mirror_inhibition(member, reader, gain)
    reset = add_reset(net, drive, "reset", g.latches, g.gates)
    if not mirror_before_reset:
        net.mirror_inhibition(member, reader, gain)
    def clear_weight(target):
        return [net.quanta[e] for e in net.incoming[target] if net.src[e] == reset[1]]
    ordinary = -round(.75 * drive.loop)
    assert clear_weight(reader) == [round(ordinary * gain)]
    g.finalize_operand_reset(reset[1])
    assert clear_weight(member) == [-round(1.5 * drive.loop)]
    assert clear_weight(reader) == [round(-round(1.5 * drive.loop) * gain)]


def test_nominal_operand_latency():
    latencies = []
    for enabled in (False, True):
        net, _, early, late, out, _, _ = operand_fixture(enabled)
        sim = SurveySim(net.topology(), P, n_nodes=4)
        for node in range(4):
            ev, lv = node // 2, node % 2
            prescribe(sim, early.taps[ev], np.arange(0, 800, 60), [node])
            prescribe(sim, late.taps[lv], [200, 241, 282, 700], [node])
        sim.run(1000)
        assert not (sim.counts[:, out.taps] > 0).all(axis=1).any()
        first = [sim.first[node, out.taps[(node // 2) ^ (node % 2)]] for node in range(4)]
        assert min(first) > 200
        latencies.append(np.asarray(first) - 200)
    print("nominal operand XOR latency steps off/on", [lat.tolist() for lat in latencies])
    assert (latencies[1] - latencies[0] == 42).all()


def test_operand_and_carry_branches_share_consumed_pulses():
    from test_ordered_xor import arithmetic_fixture
    net, _, _, *_ = arithmetic_fixture(2, operand_oneshot=True)
    topo = net.topology()
    for bit in range(2):
        prefix = f"add.fa{bit}"
        expected = {"g": f"{prefix}.x.once1.edge", "k": f"{prefix}.x.once0.edge",
                    "p1": f"{prefix}.s.once1.edge",
                    "p0": f"{prefix}.s.once{0 if bit else 1}.edge"}
        for branch, source in expected.items():
            for target in (f"{prefix}.{branch}.edge", f"{prefix}.{branch}.edge_inh"):
                inputs = [net.roles[s] for s, d, q in zip(topo.src, topo.dst, topo.quanta)
                          if net.roles[d] == target and q > 0]
                assert inputs == [source]


@pytest.mark.parametrize("datapath", ["generic", "specialized"])
def test_pipeline_rearms_with_master_ignition_and_zero_once(datapath):
    from test_ordered_xor import test_pipeline_combinations_rearm
    test_pipeline_combinations_rearm(True, datapath, True, operand=True, ignition=True)


@pytest.mark.skipif(os.environ.get("DROSO_OPXOR_SMOKE") != "1",
                    reason="operand-XOR laptop frontier requires DROSO_OPXOR_SMOKE=1")
def test_noisy16_carry_escape_frontier():
    """The rejected v1 noisy-16 draw is a frozen regression, not qualification."""
    legacy = arithmetic_survey(16, 12, True, operand_oneshot=True,
                               gates_type=OperandV1Gates)
    assert legacy["double"] == legacy["wrong"] == 1, legacy
    assert legacy["missing"] == legacy["repeated"] == legacy["survivors"] == 0, legacy
    failure, = legacy["failure_details"]
    assert {key: failure[key] for key in ("copy", "tx", "bit", "a", "b", "sub")} == dict(
        copy=9, tx=1, bit=12, a=0xaaaa, b=0x5555, sub=1)
    probes = failure["probes"]
    assert probes["add.fa11.x.once1.edge"][0] == 1
    assert probes["add.fa11.x.y1.u"][0] == 0
    assert probes["add.fa11.x.y0.u"][0] > 0
    assert probes["add.fa11.p0.edge"] == (1, 11527)
    assert probes["add.fa11.c0.u"][0] > 0 and probes["add.fa11.c1.u"][0] > 0
    fixed = arithmetic_survey(16, 12, True, operand_oneshot=True,
                              reference_type=OperandV1Gates)
    assert all(fixed[key] == 0 for key in FAILURES if key in fixed), fixed
    print("noisy16 frozen v1 frontier: v1 one double/wrong, current circuit zero failures in 24 transactions",
          flush=True)


def run_frozen_replay(scenario):
    if scenario == "replay85-clear2174":
        kwargs = dict(replay_copies=[2174])
        legacy = operand_survey("copy85", 40_000, gates_type=OperandV1Gates, **kwargs)
        assert legacy["copies"] == legacy["trials"] == 1
        assert legacy["survivors"] == 2, legacy
        assert {d["role"] for d in legacy["survivor_details"]} == {
            "add.fa0.x.y0.u", "add.fa0.x.y0.v"}
        assert all(d["reset_count"] == 3 and d["reset_last"] == 4461
                   for d in legacy["survivor_details"]), legacy
        class SharedPulsesOnly(Gates):
            def finalize_operand_reset(self, inhibitor):
                pass
        unhardened = operand_survey("copy85", 40_000, gates_type=SharedPulsesOnly,
                                    reference_type=OperandV1Gates, **kwargs)
        assert unhardened["survivors"] == 2, unhardened
        fixed = operand_survey("copy85", 40_000, reference_type=OperandV1Gates, rearm=True, **kwargs)
        assert fixed["trials"] == 2
        assert all(fixed[key] == 0 for key in FAILURES), fixed
    else:
        width, copy, tx, bit, a, b, sub = REPLAY_CASES[scenario]
        trials = 512 if width == 4 else 2048
        kwargs = dict(operand_oneshot=True, replay_copies=[copy])
        legacy = arithmetic_survey(width, trials, True, gates_type=OperandV1Gates, **kwargs)
        assert legacy["trials"] == 2
        assert legacy["wrong"] == legacy["double"] > 0, legacy
        assert any({k: f[k] for k in ("copy", "tx", "bit", "a", "b", "sub")} ==
                   dict(copy=copy, tx=tx, bit=bit, a=a, b=b, sub=sub)
                   for f in legacy["failure_details"]), scenario
        if copy in (175, 75):
            p = legacy["failure_details"][0]["probes"]
            culprit, step = (1, 1536) if copy == 175 else (2, 6530)
            assert p[f"add.fa{culprit}.p0.edge"] == (1, step)
            assert p[f"add.fa{culprit}.x.y1.u"][0] == 0
            assert p[f"add.fa{culprit}.x.y0.u"][0] > 0
            replay = legacy["replay_details"][tx]
            origin = 1000 if copy == 175 else 6000
            assert [s + 18 - origin for s in
                    replay["spikes"][f"add.fa{culprit}.cinr0.d4"][:2]] == [478, 522]
            assert carry_escape_receiver(copy) == [step - origin]
            assert carry_escape_receiver(copy, first_only=True) == []
            if copy == 75:
                assert carry_escape_receiver(copy, background=False) == [531]
        fixed = arithmetic_survey(width, trials, True, reference_type=OperandV1Gates, **kwargs)
        assert all(fixed[key] == 0 for key in FAILURES if key in fixed), {
            k: v for k, v in fixed.items() if k != "replay_details"}
    print("frozen replay", scenario, "v1 wrong/double/survivors",
          [legacy[k] for k in ("wrong", "double", "survivors")],
          "current circuit", {k: fixed[k] for k in FAILURES if k in fixed}, flush=True)


def run_survivor_replay(case):
    scenario, copy, v2_count, off_count, role, txs = SURVIVOR_CASES[case]
    kwargs = dict(replay_copies=[copy], reference_type=OperandV2Gates)
    # Control first, using precisely the v2 static draws/controller/timing.
    off = operand_survey(scenario, 40_000, enabled=False, reference_enabled=True, **kwargs)
    assert off["survivors"] == off_count, off
    legacy = operand_survey(scenario, 40_000, gates_type=OperandV2Gates, **kwargs)
    assert legacy["survivors"] == v2_count, legacy
    assert all(legacy[k] == 0 for k in FAILURES if k != "survivors"), legacy
    assert {(d["copy"], d["tx"], d["role"], d["reset_count"])
            for d in legacy["survivor_details"]} == {
        (copy, tx, f"add.fa0.{role}.{member}", 3 * (tx + 1))
        for tx in txs for member in ("u", "v")}
    fixed = operand_survey(scenario, 40_000, rearm=scenario == "copy85", **kwargs)
    assert all(fixed[k] == 0 for k in FAILURES), fixed
    print("frozen survivor", case, "F sum-only", off, "v2", legacy,
          "v3", fixed, flush=True)


def run_scenario(scenario, *, smoke=False):
    wall, cpu = perf_counter(), process_time()
    if scenario in SURVIVOR_CASES:
        run_survivor_replay(scenario)
        print(scenario, f"wall_s={perf_counter() - wall:.3f}", flush=True)
        return
    if scenario.startswith("replay"):
        run_frozen_replay(scenario)
        print(scenario, f"wall_s={perf_counter() - wall:.3f}", flush=True)
        return
    if scenario.startswith("addsub"):
        width = int(scenario.split("-")[0].removeprefix("addsub"))
        result = arithmetic_survey(width, 12 if smoke else 512 if width == 4 else 2048,
                                   scenario.endswith("mix-B"),
                                   operand_oneshot=True)
        keys = ("wrong", "double", "missing", "repeated", "survivors")
    else:
        result = operand_survey(scenario, 8 if smoke else 40_000)
        keys = FAILURES
    print("operand XOR", scenario, "smoke" if smoke else "cluster", result,
          f"wall_s={perf_counter() - wall:.3f}", f"cpu_s={process_time() - cpu:.3f}", flush=True)
    assert all(result[key] == 0 for key in keys), result


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_OPXOR_CLUSTER") != "1",
                    reason="operand-XOR cluster surveys require DROSO_OPXOR_CLUSTER=1")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_operand_xor_cluster(scenario):
    run_scenario(scenario)


@pytest.mark.skipif(os.environ.get("DROSO_OPXOR_SMOKE") != "1",
                    reason="operand-XOR laptop smoke requires DROSO_OPXOR_SMOKE=1")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_operand_xor_smoke(scenario):
    run_scenario(scenario, smoke=True)
