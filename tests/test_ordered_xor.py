"""RefSim qualification of transaction-scoped ordered XOR ignition."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from drosophilos.lib.gates import Gates, Rail2, Tap
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.latch import add_experimental_autapses, add_latch, add_reset
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


P = Params()


class SurveySim(RefSim):
    """Use the reference step unchanged; retain counts/first times instead of huge traces."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = np.zeros((self.B, self.n), dtype=int)
        self.first = np.full((self.B, self.n), -1, dtype=int)
        self.last = np.full((self.B, self.n), -1, dtype=int)

    def step(self):
        super().step()
        if self._spk_node:
            b, n = self._spk_node[-1], self._spk_neuron[-1]
            self.counts[b, n] += 1
            self.last[b, n] = self.step_index - 1
            new = self.first[b, n] < 0
            self.first[b[new], n[new]] = self.step_index - 1
            self._spk_step.clear()
            self._spk_node.clear()
            self._spk_neuron.clear()


def prescribe(sim, source, times, nodes=None):
    """Prescribe source spikes at the synaptic boundary; all receivers are real LIFs."""
    nodes = np.arange(sim.B) if nodes is None else np.asarray(nodes)
    times = np.asarray(times)
    if times.ndim == 1:
        times = np.broadcast_to(times, (len(nodes), len(times)))
    for edge in range(sim.topo.indptr[source], sim.topo.indptr[source + 1]):
        weights = (np.full(len(nodes), sim.topo.quanta[edge]) if sim.quanta is None
                   else sim.quanta[nodes, edge])
        for k in range(times.shape[1]):
            arrivals = times[:, k] + sim.topo.delay[edge]
            for step in np.unique(arrivals):
                mask = arrivals == step
                sim._events[int(step)].append((nodes[mask],
                    np.full(mask.sum(), sim.topo.dst[edge]), weights[mask]))


def xor_fixture(candidate="oneshot", autapse=True):
    net = Netlist(P)
    drive = Drive.from_params(P)
    early = Rail2(Tap(net.neuron("early0")), Tap(net.neuron("early1")))
    late = Rail2(Tap(net.neuron("late0")), Tap(net.neuron("late1")))
    gates = Gates(net, drive)
    out = gates.xor2_ordered("xor", early, late, oneshot=candidate.startswith("oneshot"))
    if candidate == "fast":
        for e, d in enumerate(net.dst):
            if net.roles[d].endswith(".edge_inh"):
                net.quanta[e] = drive.relay_in
    if candidate in ("oneshot_light", "oneshot_fast"):
        for e, (s, d) in enumerate(zip(net.src, net.dst)):
            if net.roles[s].startswith("xor.once") and s == d:
                net.quanta[e] = -round((1 if candidate == "oneshot_light" else 2.2) * drive.loop)
            if candidate == "oneshot_fast" and net.roles[d] in ("xor.once0.edge_inh", "xor.once1.edge_inh"):
                net.quanta[e] = drive.relay_in
    reset = add_reset(net, drive, "reset", gates.latches, gates.gates)
    if autapse:
        add_experimental_autapses(net, drive)
    return net, drive, early, late, out, reset


@lru_cache(maxsize=1)
def copy82_draws():
    """Reconstruct the original B=100 draw order, never the modified topology's IDs."""
    from drosophilos.bench.kernel_campaign import block
    _, pl, _, _ = block("tick", P, rate_robust=True, experimental_autapse=True)
    topo = pl.net.topology()
    assert (topo.n, topo.nnz) == (30643, 62366)
    rng = np.random.default_rng(110)
    q = np.rint(topo.quanta * np.exp(rng.normal(0, .04, (100, topo.nnz))))[82]
    th = P.V_th + rng.normal(0, .2, (100, topo.n))[82]
    bias = np.asarray(pl.net.bias) + rng.normal(0, .2, (100, topo.n))[82]
    edges = {(pl.net.roles[s], pl.net.roles[d], int(delay)): int(weight)
             for s, d, delay, weight in zip(topo.src, topo.dst, topo.delay, q)
             if pl.net.roles[d].startswith("c11_sub.fa6.s.")}
    nodes = {r: (th[i], bias[i]) for i, r in enumerate(pl.net.roles)
             if r.startswith("c11_sub.fa6.s.")}
    return nodes, edges


def apply_copy82(sim, net):
    nodes, edges = copy82_draws()
    def role(r):
        aliases = {"early0": "c11_sub.fa6.x.y0.u", "early1": "c11_sub.fa6.x.y1.u",
                   "late0": "c11_sub.fa6.cinr0.d4", "late1": "c11_sub.fa6.cinr1.d4",
                   "xor.once0.edge": "c11_sub.fa6.cinr0.d4",
                   "xor.once1.edge": "c11_sub.fa6.cinr1.d4",
                   "reset.reset_inh": "c11_sub.Q.reset_inh"}
        return aliases.get(r, r.replace("xor.", "c11_sub.fa6.s."))
    for i, r in enumerate(net.roles):
        if role(r) in nodes:
            sim.V_th[:, i], sim.bias[:, i] = nodes[role(r)]
    for e, (s, d, delay) in enumerate(zip(sim.topo.src, sim.topo.dst, sim.topo.delay)):
        key = (role(net.roles[s]), role(net.roles[d]), int(delay))
        if key in edges:
            sim.quanta[:, e] = edges[key]


def xor_survey(scenario, trials, candidate="oneshot", seed=882):
    net, drive, early, late, out, reset = xor_fixture(candidate)
    topo = net.topology()
    rng = np.random.default_rng(seed)
    result = dict(trials=trials, double=0, missing=0, wrong=0, repeated=0,
                  sum_doublet=0,
                  min_latency=10000, max_latency=0)
    for start in range(0, trials, 512):
        size = min(512, trials - start)
        if scenario in ("corners", "copy82"):
            w = rng.choice([-.12, .12], (size, topo.nnz))
            th = rng.choice([-.6, .6], (size, topo.n))
            bias = rng.choice([-.6, .6], (size, topo.n))
        elif scenario == "nominal":
            w = np.zeros((size, topo.nnz))
            th = bias = np.zeros((size, topo.n))
        else:
            w = rng.normal(0, .04, (size, topo.nnz))
            th = rng.normal(0, .2, (size, topo.n))
            bias = rng.normal(0, .2, (size, topo.n))
        sim = SurveySim(topo, P, n_nodes=size,
                        quanta=np.rint(topo.quanta * np.exp(w)),
                        V_th=P.V_th + th, bias=np.asarray(net.bias) + bias)
        evalue = rng.integers(0, 2, size)
        lvalue = rng.integers(0, 2, size)
        # EARLY leads by >= 15 ms, then remains asserted; LATE includes a
        # refractory-limit burst and a long gap beyond feed-forward recovery.
        early_times = np.arange(0, 1600, 60)[None, :] + rng.integers(0, 8, (size, 27))
        late_times = np.array([200, 230, 260, 290, 320, 700, 730, 760, 1300, 1330])[None, :]
        late_times = late_times + rng.integers(-5, 6, (size, 10))
        if scenario == "copy82":
            apply_copy82(sim, net)
            evalue.fill(0)
            lvalue.fill(1)
            jitter = rng.integers(-10, 11, (size, 1))
            jitter[0] = 0
            late_times = np.array([629, 671, 713, 757, 1100, 1142, 1500])[None, :] + jitter
            prescribe(sim, late.r1.u, late_times)
            veto = net.roles.index("xor.y0.a.veto")
            sim.silenced[:, veto] = True
            vt = np.array([492, 562, 628, 691, 752, 815, 878, 941, 1004,
                           1067, 1130, 1193, 1256, 1319, 1382, 1445, 1508, 1571])
            veto_jitter = rng.integers(-5, 6, (size, 1))
            veto_jitter[0] = 0
            prescribe(sim, veto, vt[None, :] + veto_jitter)
            if candidate == "legacy":
                inh = net.roles.index("xor.y0.a.edge_inh")
                sim.silenced[:, inh] = True
                prescribe(sim, inh, [689, 738, 784])
        else:
            for r in (0, 1):
                ns = np.flatnonzero(evalue == r)
                prescribe(sim, early.taps[r], early_times[ns], ns)
                ns = np.flatnonzero(lvalue == r)
                prescribe(sim, late.taps[r], late_times[ns], ns)
        sim.run(1700)
        seen = sim.counts[:, out.taps] > 0
        sum_edges = [i for i, r in enumerate(net.roles) if r.startswith("xor.y") and r.endswith(".edge")]
        result["sum_doublet"] += int((sim.counts[:, sum_edges].sum(axis=1) > 1).sum())
        expected = evalue ^ lvalue
        result["double"] += int(seen.all(axis=1).sum())
        result["missing"] += int((~seen.any(axis=1)).sum())
        result["wrong"] += int(seen[np.arange(size), 1 - expected].sum())
        first = sim.first[np.arange(size), np.asarray(out.taps)[expected]]
        lat = first[first >= 0] - late_times[first >= 0, 0]
        if len(lat):
            result["min_latency"] = min(result["min_latency"], int(lat.min()))
            result["max_latency"] = max(result["max_latency"], int(lat.max()))
        if candidate.startswith("oneshot"):
            edges = [i for i, r in enumerate(net.roles) if r in ("xor.once0.edge", "xor.once1.edge")]
            result["repeated"] += int((sim.counts[:, edges].sum(axis=1) != 1).sum())
    return result


@pytest.mark.parametrize("scenario", ["nominal", "mix_b", "corners", "copy82"])
def test_ordered_xor_fast_survey(scenario):
    result = xor_survey(scenario, 128)
    print(scenario, result)
    assert result["double"] == result["missing"] == result["wrong"] == result["repeated"] == result["sum_doublet"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["mix_b", "corners", "copy82"])
def test_ordered_xor_40000(scenario):
    result = xor_survey(scenario, 40_000)
    print(scenario, result)
    assert result["double"] == result["missing"] == result["wrong"] == result["repeated"] == result["sum_doublet"] == 0


def test_copy82_reconstruction_and_legacy_control():
    nodes, edges = copy82_draws()
    assert nodes["c11_sub.fa6.s.y0.a.edge"] == (-45.235698207123605, .38859703129223355)
    assert edges[("c11_sub.fa6.cinr1.d4", "c11_sub.fa6.s.y0.a.edge", 18)] == 4919
    assert xor_survey("copy82", 1, "legacy")["double"] == 1
    assert xor_survey("copy82", 1)["double"] == 0


def test_rejected_candidates_preserve_the_frontier():
    light = xor_survey("corners", 128, "oneshot_light")
    fast = xor_survey("copy82", 128, "oneshot_fast")
    assert light["repeated"] > 0  # transaction memory alone misses the initial doublet
    assert fast["missing"] > 0  # fast feed-forward inhibition can beat the first pulse


@pytest.mark.parametrize("autapse", [False, True])
def test_transaction_memory_holds_across_long_gaps_and_resets(autapse):
    net, drive, early, late, out, reset = xor_fixture(autapse=autapse)
    topo = net.topology()
    rng = np.random.default_rng(912)
    size = 64
    sim = SurveySim(topo, P, n_nodes=size,
                    quanta=np.rint(topo.quanta * np.exp(rng.choice([-.12, .12], (size, topo.nnz)))),
                    V_th=P.V_th + rng.choice([-.6, .6], (size, topo.n)),
                    bias=np.asarray(net.bias) + rng.choice([-.6, .6], (size, topo.n)))
    once = [net.roles.index(f"xor.once{r}.edge") for r in (0, 1)]
    failures = 0
    for tx in range(4):
        origin = tx * 6000
        # Exercise every truth-table row on each physical copy, including rail flips.
        ev, lv = tx // 2, tx % 2
        prescribe(sim, early.taps[ev], origin + np.arange(0, 4000, 60))
        prescribe(sim, late.taps[lv], origin + np.array([200, 242, 284, 2500, 2542, 3500]))
        sim.counts.fill(0)
        sim.first.fill(-1)
        sim.run(4200)
        failures += int((sim.counts[:, once].sum(axis=1) != 1).sum())
        failures += int((sim.counts[:, out.taps[ev ^ lv]] == 0).sum())
        failures += int((sim.counts[:, out.taps[1 - (ev ^ lv)]] > 0).sum())
        for node in range(size):
            sim.add_events(node, [origin + 4200], [reset[0]], [drive.ignite])
        sim.run(1800)
        failures += int((sim.last[:, out.taps] >= origin + 5000).sum())
    # This is why the builder requires experimental_autapse, rather than
    # advertising a reset qualification for the old two-neuron storage loop.
    assert failures == 0 if autapse else failures > 0


def arithmetic_survey(width, trials, noisy=False, autapse=True, oneshot=True, recovery=2000,
                      *, operand_oneshot=False):
    net, drive = Netlist(P), Drive.from_params(P)
    gates = Gates(net, drive, ordered_xor_oneshot=oneshot, operand_xor_oneshot=operand_oneshot)
    def word(name, n):
        return [Rail2(gates.latch(f"{name}{i}r0"), gates.latch(f"{name}{i}r1")) for i in range(n)]
    a, b = word("A", width), word("B", width)
    # A kernel cell has a fixed op: SUB is image state, outside its stage reset.
    sub = Rail2(add_latch(net, drive, "sub0"), add_latch(net, drive, "sub1"))
    # Same SUB/B^d/A^d ordering and carry delay as add_alu_logic_tokens.
    bd = [gates.delayed(f"b{i}", x, 6) for i, x in enumerate(b)]
    bx = [gates.xor2_ordered(f"bx{i}", sub, x) for i, x in enumerate(bd)]
    ad = [gates.delayed(f"a{i}", x, 17) for i, x in enumerate(a)]
    sums, carries, _, _ = gates.ripple_adder_ordered("add", ad, bx, sub)
    reset = add_reset(net, drive, "clear", gates.latches, gates.gates)
    if autapse:
        add_experimental_autapses(net, drive)
    topo = net.topology()
    rng = np.random.default_rng(718)
    result = dict(trials=2 * trials, wrong=0, double=0, missing=0, repeated=0, survivors=0,
                  min_latency=10000, max_latency=0)
    if operand_oneshot:
        result["failure_details"] = []
    for start in range(0, trials, 64):
        size = min(64, trials - start)
        kwargs = {}
        if noisy:
            kwargs = dict(quanta=np.rint(topo.quanta * np.exp(rng.normal(0, .04, (size, topo.nnz)))),
                          V_th=P.V_th + rng.normal(0, .2, (size, topo.n)),
                          bias=np.asarray(net.bias) + rng.normal(0, .2, (size, topo.n)))
        sim = SurveySim(topo, P, n_nodes=size, **kwargs)
        if width == 4 and not noisy:
            ids = np.arange(start, start + size)
            av, bv, sv = ids % 16, ids // 16 % 16, ids // 256 % 2
        else:
            av, bv = rng.integers(0, 1 << width, (2, size))
            sv = rng.integers(0, 2, size)
        if operand_oneshot and width == 4 and trials >= 512:
            ids = np.arange(start, start + size)
            av, bv, sv = ids // 32 % 16, ids // 2 % 16, ids % 2
        elif operand_oneshot:
            # Long carry/borrow, zero, overflow and alternating rails, for both
            # fixed ADD and SUB images. Every copy reloads swapped operands.
            mask = (1 << width) - 1
            directed = [(mask, 1), (0, 1), (0, 0), (mask >> 1, 1),
                        (mask // 3, 2 * (mask // 3)), (mask, mask)]
            for k in range(min(size, max(0, 2 * len(directed) - start))):
                index = start + k
                av[k], bv[k] = directed[index // 2]
                sv[k] = index % 2
        # Two different words through the same physical state, including the
        # complete delayed-input drain and inhibitory recovery after reset.
        duration = 3000 if width <= 4 else 5000 if width <= 8 else 8000
        for tx, origin in enumerate((0, duration + recovery)):
            if tx:
                av, bv = bv, av
            for node in range(size):
                if not tx:
                    sim.add_events(node, [origin], [sub.latches[sv[node]].u], [drive.ignite])
                taps = [rail.latches[(value >> i) & 1].u
                        for value, rails in ((av[node], a), (bv[node], b))
                        for i, rail in enumerate(rails)]
                jitter = rng.integers(0, 101, len(taps)) if noisy else np.zeros(len(taps), int)
                sim.add_events(node, origin + 200 + jitter, taps, [drive.ignite] * len(taps))
            sim.counts.fill(0)
            sim.first.fill(-1)
            # Mix B's 5-Hz, 150-q independent background, plus 0..100-step load jitter.
            while sim.step_index < origin + duration:
                if noisy:
                    count = rng.binomial(sim.B * sim.n, .0005)
                    flat = rng.choice(sim.B * sim.n, count, replace=False)
                    sim._events[sim.step_index].append((flat // sim.n, flat % sim.n, np.full(count, 150)))
                sim.step()
            raw = av + np.where(sv, ((1 << width) - 1) ^ bv, bv) + sv
            for bit, rail in enumerate(sums + [carries[-1]]):
                seen = sim.counts[:, rail.taps] > 0
                expected = (raw >> bit) & 1
                result["double"] += int(seen.all(axis=1).sum())
                result["missing"] += int((~seen.any(axis=1)).sum())
                result["wrong"] += int(seen[np.arange(size), 1 - expected].sum())
                if operand_oneshot:
                    for node in np.flatnonzero(seen[np.arange(size), 1 - expected]):
                        if len(result["failure_details"]) >= 8:
                            break
                        probes = {r: (int(sim.counts[node, i]), int(sim.first[node, i]))
                                  for i, r in enumerate(net.roles)
                                  if any(r.startswith(f"add.fa{j}.")
                                         for j in range(max(0, bit - 2), min(width, bit + 1)))}
                        result["failure_details"].append(dict(copy=start + int(node), tx=tx, bit=bit,
                            a=int(av[node]), b=int(bv[node]), sub=int(sv[node]), probes=probes))
                times = sim.first[np.arange(size), np.asarray(rail.taps)[expected]]
                lat = times[times >= 0] - origin - 200
                if len(lat):
                    result["min_latency"] = min(result["min_latency"], int(lat.min()))
                    result["max_latency"] = max(result["max_latency"], int(lat.max()))
            once = [i for i, r in enumerate(net.roles) if ".once" in r and r.endswith(".edge")]
            result["repeated"] += int((sim.counts[:, once] > 1).sum())
            if operand_oneshot:
                # Check first-pulse acceptance as well as repeat suppression in
                # both the operand and sum XORs, one of two rails per bit.
                for kind in ("x", "s"):
                    for bit in range(width):
                        edges = [net.roles.index(f"add.fa{bit}.{kind}.once{r}.edge") for r in (0, 1)]
                        result["missing"] += int((sim.counts[:, edges].sum(axis=1) == 0).sum())
            if not tx:
                for node in range(size):
                    sim.add_events(node, [duration], [reset[0]], [drive.ignite])
                sim.run(recovery)
                if operand_oneshot:
                    members = [m for latch in gates.latches for m in latch.members]
                    result["survivors"] += int((sim.last[:, members] >= duration + recovery - 300).sum())
    return result


@pytest.mark.parametrize("noisy,autapse", [(False, False), (False, True), (True, True)])
def test_add_sub_and_reset(noisy, autapse):
    result = arithmetic_survey(4, 16, noisy, autapse)
    print("arithmetic", noisy, autapse, result)
    assert result["double"] == result["missing"] == result["wrong"] == result["repeated"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("width,trials,noisy", [(4, 512, False), (8, 2048, True)])
def test_add_sub_large_survey(width, trials, noisy):
    result = arithmetic_survey(width, trials, noisy)
    print("arithmetic", width, noisy, result)
    assert result["double"] == result["missing"] == result["wrong"] == result["repeated"] == 0


@pytest.mark.parametrize("zero,datapath,noisy", [
    (False, "generic", False), (False, "generic", True),
    (True, "generic", True), (True, "specialized", True),
])
def test_pipeline_combinations_rearm(zero, datapath, noisy, *, operand=False, ignition=False):
    from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
    from test_experimental_autapse import _NoisyRefSim
    spec = [{"name": "add", "op": "ADD", "a": "input", "b": ("const", "one")},
            {"name": "sub", "op": "SUB", "a": "add", "b": ("const", "three")},
            {"name": "out", "op": "XOR", "a": "sub", "b": ("const", "one")}]
    pl = build_pipeline(P, 2, spec, consts={"one": 1, "three": 3},
                        rate_robust=True, experimental_autapse=True, zero_once=zero,
                        datapath=datapath, ordered_xor_oneshot=True,
                        operand_xor_oneshot=operand, master_ignition=ignition)
    sim = _NoisyRefSim(pl.net, copies=2) if noisy else RefSim(pl.net.topology(), P, n_nodes=2)
    outputs, _, stats = run_pipeline_batched(pl, P, [[0, 1, 3]] * 2,
                                             max_ms=9000, sim=sim, progress=0)
    assert [[v for _, v in copy["out"]] for copy in outputs] == [[3, 2, 0]] * 2
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    # Every spent latch is owned by the corresponding stage's reset domain.
    for cell in pl.cells:
        targets = {d for s, d, q in zip(pl.net.src, pl.net.dst, pl.net.quanta)
                   if s == cell.stage.reset_inh and q < 0}
        spent = {i for i, r in enumerate(pl.net.roles)
                 if r.startswith(cell.name + ".fa") and ".spent." in r}
        assert spent <= targets
        assert spent or (datapath == "specialized" and cell.op == "XOR")


@pytest.mark.parametrize("rate,autapse", [(False, False), (True, False), (False, True)])
def test_requires_qualified_storage_and_readers(rate, autapse):
    from drosophilos.lib.kernel import build_pipeline
    with pytest.raises(ValueError, match="ordered_xor_oneshot requires"):
        build_pipeline(P, 1, [], rate_robust=rate, experimental_autapse=autapse,
                       ordered_xor_oneshot=True)


@pytest.mark.parametrize("options", [
    {"n": 9}, {"params": replace(P, dt=.2)}, {"act_hops": 10}, {"idle_hops": 19},
    {"robust_request_clear": True},
    {"experimental_register_reset": True}, {"kernel_kill_pulses": 3},
    {"retry_clear": True},
])
def test_unqualified_shapes_rejected(options):
    from drosophilos.lib.kernel import build_pipeline
    args = dict(params=P, n=2, spec=[{"name": "out", "op": "ADD", "a": "input", "b": ("const", "k")}],
                consts={"k": 1}, ordered_xor_oneshot=True, rate_robust=True, experimental_autapse=True)
    with pytest.raises(ValueError, match="ordered_xor_oneshot supports only"):
        build_pipeline(**(args | options))


@pytest.mark.parametrize("version", [None, 0, 2])
def test_rebuild_requires_identified_oneshot_version(version):
    from drosophilos.bench.stall_diag import build_tick_pipeline
    from drosophilos.bench.stage_d import recheck_record
    record = {"ordered_xor_oneshot": True, "ordered_xor_oneshot_version": version}
    for rebuild in (build_tick_pipeline, recheck_record):
        with pytest.raises(ValueError, match="ordered-XOR one-shot circuit version"):
            rebuild(record)


@pytest.mark.parametrize("version", [None, -1, 2, "1", True])
def test_wide_array_version_rejects_unknown_circuits(version):
    with pytest.raises(ValueError, match="unsupported ordered_xor_oneshot_mul_version"):
        Gates(Netlist(P), Drive.from_params(P), ordered_xor_oneshot=True,
              ordered_xor_oneshot_mul_version=version)
