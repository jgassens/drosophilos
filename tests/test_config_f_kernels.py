"""Configuration F on the wider kernel primitives (float64 RefSim, never Torch).

Small tests exercise actual reset/re-arm cycles. Slow surveys expand operands,
widths and physical mix-B draws; neither is a game-kernel campaign exit.
"""

from dataclasses import replace
import hashlib
import json
import os
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim
from test_experimental_autapse import _NoisyRefSim
from test_ordered_xor import SurveySim


P = Params()
F = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)
SHAPES = ("rom", "ram", "ram_shared", "MUL", "MULP", "streams", "each", "wrap")
# Ordered netlist/image digests collected by executing caf043e's kernel.py,
# including biases, reader and reset-mirror registries (not just edge counts).
OFF_DIGESTS = dict(zip(SHAPES, (
    "260d54b72b6836e472f4f404b8b38cccd5715371b11613640157d1753d212a72",
    "a63f967b10ed020ac3bd7d6d508d20cc093135052eafcaef863657eef89f2c31",
    "985bdd91c57b6e5d0000c427f95dc463f0db02896ef795464bb1cfeda26442cf",
    "a4f6da262c3e4be49bfe16a1a8e9046cf3fdef1339c3cb1ce5c847222b941084",
    "5b4f0c0b3e6f1c4eaf3abc7c7effa68b140ad94bb2c0cc5f55b86ccd7a575b8c",
    "9ff43bf4419378e519d8977494f6e433f5699494a6d0f9de6b25f75ce53d58f2",
    "41570fd545cbae54471636c5a6974057c6449c81ccce555bfba372adac868805",
    "79e155f1dec7b3d896632b40bddd0e907c54bebbd32abfca1137734ac4374800",
)))
# Captured from the unmodified builder before the wide-array v1 repair.
F_CAMPAIGN_DIGESTS = {
    "tick-generic": "86f32cd954b39a59f57109675dd7cf7b6dbc6da32c81a14c95ff042d0045b0fd",
    "tick-specialized": "2ebac22b72725602c02952495c787e429a42dc54ca0d6e4a9f95589438920a5f",
    "render": "d0e713368a042c8a5486666535d5f1edb5c45b4ddff59d8e515e589954614e75",
}
F_SMALL_DIGESTS = dict(zip(SHAPES, (
    "b7e3982fc5f74ecb2f34b01eb9dbd0de3261706607d71ce5ed1a8d91a8a78303",
    "459c40ae1912c0e41a7f9df2b5cb45f50dbf3d98b513f84d0f978c7c070a14c6",
    "ef59584b1c4bd3e5832940f449e43e3cefa095dc82669c3d59a051ed155e01f7",
    "f57017d7f7f4570649f52a5a187ed2e660a3401981dfdca793726800e25d7bcf",
    "3a0621f83f0270459f99fde11030644ab94854939f3ae60ae8dd30c2da1b823f",
    "440096d116a3070e56647db65b0ecdcb9b3c9417c9ed310b7919a371c7cdb9d6",
    "c895f24c528bea4e0d496e10a69abcf335438f96f76544d5318e426d4e5b04f7",
    "c295a1ec3d5944427b3edff7a95bd5ae00cdd4b24f04e5708726d511db05560f",
)))


def pipeline_digest(pl):
    net = pl.net
    data = [getattr(net, k) for k in ("roles", "src", "dst", "quanta", "delay", "bias",
                                     "rate_readouts", "inhibition_mirrors")]
    data.append([(l.u, l.v) for l in pl.image_latches])
    return hashlib.sha256(json.dumps(data, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("name", F_CAMPAIGN_DIGESTS)
def test_f_campaigns_without_wide_array_stay_byte_identical(name):
    from drosophilos.bench import kernel_campaign, stage_d

    pl = (kernel_campaign.block("render", P, **F)[1] if name == "render" else
          stage_d.build(stage_d.load_kernel(), P, name.removeprefix("tick-"), **F))
    assert pipeline_digest(pl) == F_CAMPAIGN_DIGESTS[name]
    assert "ordered_xor_oneshot_mul_version" not in pl.build_options


@pytest.mark.parametrize("options", [{}, {"rate_robust": True},
                                     {"rate_robust": True, "zero_once": True}])
def test_non_f_wide_array_stays_byte_identical(options, monkeypatch):
    from functools import partial
    from drosophilos.lib import kernel
    from drosophilos.lib.gates import Gates

    flags = dict.fromkeys(F, False) | options
    current, _, _ = small_shape("MUL", 4, **flags)
    monkeypatch.setattr(kernel, "Gates", partial(Gates, ordered_xor_oneshot_mul_version=0))
    legacy, _, _ = small_shape("MUL", 4, **flags)
    assert pipeline_digest(current) == pipeline_digest(legacy)
    assert current.build_options == legacy.build_options
    assert "ordered_xor_oneshot_mul_version" not in current.build_options


def small_shape(shape, width=2, **options):
    """Real compiler-level shapes; expected values are independent integer arithmetic."""
    mask = (1 << width) - 1
    kwargs = dict(consts={"one": 1, "mask": mask})
    tokens = [0, 1, mask, 0]
    if shape == "rom":
        spec = [dict(name="out", op="LOAD", a="input", mem="mem")]
        kwargs["mems"] = {"mem": (4, {0: 3, 1: 0, 2: 2, 3: 1})}
        tokens = [0, 1, 3, 2, 0]
        expected = {"out": [3, 0, 1, 2, 3]}
    elif shape in ("ram", "ram_shared"):
        spec = [dict(name="data", op="XOR", a="input", b=("const", "mask")),
                dict(name="store", op="STORE", a="input", b="data", mem="mem"),
                dict(name="out", op="LOAD", a="input", mem="mem", trigger=["store"])]
        kwargs["mems"] = {"mem": (4, {0: 0, 1: 0, 2: 0, 3: 0}, "ram")}
        tokens = [0, 1, 3, 2, 0]
        expected = {"out": [x ^ mask for x in tokens]}
        if shape == "ram_shared":
            spec.insert(1, dict(name="first", op="STORE", a="input", b="input", mem="mem"))
            spec[2]["trigger"] = ["first"]
    elif shape in ("MUL", "MULP"):
        spec = [dict(name="out", op=shape, a="input", b="input:b")]
        kwargs["streams"] = ["input", "b"]
        operands = [(0, mask), (mask, mask), (mask, 1), (1, 0)]
        tokens = [t for a, b in operands for t in (("input", a), ("b", b))]
        expected = {"out": [(a * b) & mask for a, b in operands]}
    elif shape in ("streams", "each", "wrap"):
        spec = [dict(name="out", op="ADD", a="input", b=("const", "one")),
                dict(name="other", op="MOV", a="input:b", b="input:b")]
        kwargs.update(streams=["input", "b"], outputs=["out", "other"])
        count = 2 if shape == "wrap" else 1
        if shape == "wrap":
            spec.append(dict(name="ring", op="RING", k=count, stream="input", trigger=["out"]))
        if shape != "streams":
            kwargs["phases"] = [("input", "ring" if shape == "wrap" else "out",
                                 "wrap" if shape == "wrap" else "each"),
                                ("b", "other", "each")]
        tokens = []
        for frame in range(3):
            # The existing two-phase host contract: do not deal the next frame
            # into the still-open, draining first phase (test_kernel.py).
            tokens += [("input", (frame * count + i) & mask,
                        frame * (count + 1) if i == 0 else 0) for i in range(count)]
            tokens += [("b", frame, 0)]
        expected = {"out": [(i + 1) & mask for i in range(3 * count)], "other": [0, 1, 2]}
    else:
        raise ValueError(shape)
    pl = build_pipeline(P, width, spec, **(kwargs | F | options))
    return pl, tokens, expected


def run_shape(shape, *, noisy=False, width=2, datapath="generic"):
    pl, tokens, expected = small_shape(shape, width, datapath=datapath)
    copies = 2 if noisy else 1
    sim = KernelMixSim(pl, copies=copies, seed=110) if noisy else RefSim(pl.net.topology(), P)
    outputs, _, stats = run_pipeline_batched(
        pl, P, [tokens] * copies, max_ms=25000, sim=sim, progress=0,
        expect_outputs=[sum(map(len, expected.values()))] * copies)
    got = [{name: [v for _, v in values] for name, values in row.items()} for row in outputs]
    assert got == [expected] * copies, (got, expected, stats)
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0, stats
    assert stats.get("refusals", 0) == stats.get("retries", 0) == 0, stats
    return pl, outputs, stats


@pytest.mark.parametrize("shape", SHAPES)
def test_small_kernel_nominal(shape):
    pl, outputs, stats = run_shape(shape)
    print(shape, pl.net.n, pl.net.nnz, stats["neural_ms"])
    if shape in ("each", "wrap"):
        count = 2 if shape == "wrap" else 1
        order = [name for _, name in sorted((s, name) for name, seq in outputs[0].items() for s, _ in seq)]
        assert order == (["out"] * count + ["other"]) * 3


@pytest.mark.parametrize("shape", SHAPES)
def test_off_byte_identical_and_f_reset_ownership(shape):
    off, _, _ = small_shape(shape, **dict.fromkeys(F, False))
    assert pipeline_digest(off) == OFF_DIGESTS[shape]
    pl, _, _ = small_shape(shape)
    assert pipeline_digest(pl) == F_SMALL_DIGESTS[shape]
    net = pl.net
    # All wider latch populations, including RAM words/COPY/marks and pacing
    # pairs, receive v1 local feedback. Readers never acquire self feedback.
    for i, role in enumerate(net.roles):
        if role.endswith((".u", ".v")):
            feedback = [(q, d) for s, q, d in
                        ((net.src[e], net.quanta[e], net.delay[e]) for e in net.incoming[i]) if s == i]
            assert feedback == [(-round(pl.drive.loop * .2), 0)], role
    for source, mirrors in net.inhibition_mirrors.items():
        for target, _ in mirrors:
            assert not any(net.src[e] == source and net.quanta[e] < 0 for e in net.incoming[target])
    assert pl.build_options["experimental_autapse_version"] == 1
    assert pl.build_options["ordered_xor_oneshot_version"] == 1
    for cell in pl.cells:
        spent = {i for i, r in enumerate(net.roles) if r.startswith(cell.name + ".") and ".spent." in r
                 and not any(other.name.startswith(cell.name + ".") and r.startswith(other.name + ".")
                             for other in pl.cells)}
        targets = {d for s, d, q in zip(net.src, net.dst, net.quanta)
                   if s == cell.stage.reset_inh and q < 0}
        assert spent <= targets


@pytest.mark.parametrize("shape", SHAPES)
def test_wider_shapes_require_complete_f(shape):
    with pytest.raises(ValueError, match="experimental_autapse supports only"):
        small_shape(shape, ordered_xor_oneshot=False)


def test_non_f_zero_once_render_remains_buildable():
    """The fda6528-compatible non-F build is outside the F survey gate."""
    from drosophilos.bench import kernel_campaign

    _, pl, _, _ = kernel_campaign.block("render", P, zero_once=True)
    assert pl.build_options["zero_once"] is True
    assert not pl.build_options["experimental_autapse"]
    assert not pl.build_options["ordered_xor_oneshot"]
    with pytest.raises(ValueError, match="zero_once is not surveyed"):
        kernel_campaign.block("render", P, zero_once=True, **F)


@pytest.mark.parametrize("shape", [
    pytest.param("rom", id="ROM"),
    pytest.param("ram", id="RAM"),
    pytest.param("ram_shared", id="RAM-shared"),
    pytest.param("MUL", id="MUL"),
    pytest.param("MULP", id="MULP"),
    pytest.param("streams", id="streams"),
    pytest.param("each", id="each"),
    pytest.param("wrap", id="wrap"),
])
def test_zero_once_rejects_unsurveyed_wider_shapes(shape):
    with pytest.raises(ValueError, match="zero_once is not surveyed"):
        small_shape(shape, zero_once=True)


@pytest.mark.parametrize("datapath", ["generic", "specialized"])
def test_zero_once_tick2_remains_allowed(datapath):
    from drosophilos.bench.stage_d import build, load_kernel

    build(load_kernel(), P, datapath, zero_once=True, **F)


@pytest.mark.parametrize("options", [dict(width=16), dict(act_hops=10), dict(kernel_kill_pulses=3),
                                    dict(experimental_register_reset=True), dict(robust_request_clear=True)])
def test_unevaluated_timing_and_width_remain_rejected(options):
    with pytest.raises(ValueError, match="ordered_xor_oneshot supports only"):
        small_shape("MULP", **options)


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("datapath", ["generic", "specialized"])
def test_small_kernel_mix_b(shape, datapath):
    run_shape(shape, noisy=True, datapath=datapath)


class KernelMixSim(_NoisyRefSim):
    def __init__(self, pl, **kwargs):
        self.host_inputs = {l.u for _, producer in pl.inputs.values() for pair in producer.rails for l in pair}
        super().__init__(pl.net, **kwargs)

    def add_events(self, node, steps, neurons, quanta):
        steps = np.asarray(steps)
        if all(int(n) in self.host_inputs for n in neurons):
            steps = steps + self.rng.integers(0, 101, len(steps))
        super().add_events(node, steps, neurons, quanta)


@pytest.mark.slow
@pytest.mark.parametrize("width,copies,noisy", [(2, 16, False), (4, 8, True), (8, 4, True)])
def test_pipelined_multiplier_operands(width, copies, noisy):
    pl, _, _ = small_shape("MULP", width)
    mask = (1 << width) - 1
    sim = KernelMixSim(pl, copies=copies, seed=113) if noisy else RefSim(pl.net.topology(), P, n_nodes=copies)
    rng = np.random.default_rng(113)
    operands = ([(i % 4, i // 4) for i in range(copies)] if not noisy else
                rng.integers(0, mask + 1, (copies, 2)).tolist())
    schedules = [[("input", a), ("b", b), ("input", a ^ mask), ("b", b ^ mask)] for a, b in operands]
    outputs, _, stats = run_pipeline_batched(pl, P, schedules, max_ms=25000, sim=sim,
                                             progress=0, expect_outputs=[2] * copies)
    want = [[a * b & mask, (a ^ mask) * (b ^ mask) & mask] for a, b in operands]
    assert [[v for _, v in row["out"]] for row in outputs] == want
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0, stats


class MixSurvey(SurveySim):
    """The campaign's independent quenched draws and 5 Hz / 150 q background.

    Input jitter is applied only to host operand loads, never internal events.
    Reset, copy, completion, kill trains and rate readers are real neurons.
    """

    def __init__(self, net, copies, seed, noisy):
        self.rng = np.random.default_rng(seed)
        self.noisy = noisy
        topo = net.topology()
        kwargs = {}
        if noisy:
            kwargs = dict(quanta=np.rint(topo.quanta * np.exp(self.rng.normal(0, .04, (copies, topo.nnz)))),
                          V_th=P.V_th + self.rng.normal(0, .2, (copies, topo.n)),
                          bias=np.asarray(net.bias) + self.rng.normal(0, .2, (copies, topo.n)))
        super().__init__(topo, P, n_nodes=copies, **kwargs)

    def step(self):
        if self.noisy:
            count = self.rng.binomial(self.B * self.n, 5 * P.dt / 1000)
            flat = self.rng.choice(self.B * self.n, count, replace=False)
            self._events[self.step_index].append((flat // self.n, flat % self.n, np.full(count, 150)))
        super().step()

    def load(self, rails, values, at, drive):
        for node, value in enumerate(values):
            taps = [pair[(int(value) >> i) & 1].u for i, pair in enumerate(rails)]
            jitter = self.rng.integers(0, 101, len(taps)) if self.noisy else np.zeros(len(taps), int)
            self.add_events(node, at + jitter, taps, [drive.ignite] * len(taps))

    def pulse(self, neuron, at, drive):
        for node in range(self.B):
            self.add_events(node, [at], [neuron], [drive.ignite])


def score_rails(sim, rails, values, result, *, since=None):
    for bit, pair in enumerate(rails):
        seen = (sim.counts[:, [p.u for p in pair]] > 0 if since is None else
                sim.last[:, [p.u for p in pair]] >= since)
        want = (np.asarray(values) >> bit) & 1
        result["double"] += int(seen.all(axis=1).sum())
        result["missing"] += int((~seen.any(axis=1)).sum())
        result["wrong"] += int(seen[np.arange(sim.B), 1 - want].sum())


def multiplier_fixture(width, *, oneshot=True, mul_version=1):
    """Actual kernel ordering, retiming trees and ordinary stage reset."""
    from drosophilos.lib.gates import Gates, Rail2
    from drosophilos.lib.netlist import Drive, Netlist
    from drosophilos.protocol.latch import add_experimental_autapses, add_reset

    net = Netlist(P)
    drive = replace(Drive.from_params(P), rate_robust=True, kill_pulses=4)
    g = Gates(net, drive, ordered_xor_oneshot=oneshot,
              ordered_xor_oneshot_mul_version=mul_version)
    a = [Rail2(g.latch(f"a{i}r0"), g.latch(f"a{i}r1")) for i in range(width)]
    b = [Rail2(g.latch(f"b{i}r0"), g.latch(f"b{i}r1")) for i in range(width)]
    zero = Rail2(g.latch("zero0"), g.latch("zero1"))
    # Exact kernel B/SUB XOR, A delay, and carry ordering.
    bd = [g.delayed(f"bd{i}", r, 6) for i, r in enumerate(b)]
    bx = [g.xor2_ordered(f"bx{i}", zero, r) for i, r in enumerate(bd)]
    ad = [g.delayed(f"ad{i}", r, 17) for i, r in enumerate(a)]
    out = g.multiplier_ordered("mul", ad, bx, zero)
    reset = add_reset(net, drive, "stage", g.latches, g.gates)
    add_experimental_autapses(net, drive)
    return net, drive, g, a, b, zero, out, reset


def multiplier_survey(width, trials, noisy=False, *, operands=None, oneshot=True, mul_version=1):
    """All small nominal operand pairs, then complemented reloads after 200 ms clear.

    Keep the original evaluation deadline: wide-array v1 must finish before
    that clear, rather than merely hiding the latency regression with a longer run.
    """
    net, drive, g, a, b, zero, out, reset = multiplier_fixture(
        width, oneshot=oneshot, mul_version=mul_version)
    result = dict(trials=2 * trials, wrong=0, double=0, missing=0, repeated=0, survivors=0)
    once = [i for i, role in enumerate(net.roles) if ".once" in role and role.endswith(".edge")]
    members = sorted({m for latch in g.latches for m in latch.members})
    mask = (1 << width) - 1
    duration = 1000 + width * width * 500
    for start in range(0, trials, 16):
        size = min(16, trials - start)
        sim = MixSurvey(net, size, 4100 + start, noisy)
        ids = np.arange(start, start + size)
        av, bv = ((ids % (mask + 1), ids // (mask + 1)) if not noisy and width <= 4 else
                  sim.rng.integers(0, mask + 1, (2, size)))
        if operands is not None:
            av, bv = np.asarray(operands[start:start + size]).T
        for tx in range(2):
            origin = sim.step_index
            sim.counts.fill(0)
            sim.load([r.latches for r in a], av, origin + 200, drive)
            sim.load([r.latches for r in b], bv, origin + 200, drive)
            sim.pulse(zero.r0.u, origin, drive)
            sim.run(duration)
            score_rails(sim, [r.latches for r in out], av * bv & mask, result)
            result["repeated"] += int((sim.counts[:, once] > 1).sum())
            sim.pulse(reset[0], sim.step_index, drive)
            sim.run(2000)
            result["survivors"] += int((sim.last[:, members] >= sim.step_index - 500).sum())
            av, bv = av ^ mask, bv ^ mask
    return result


@pytest.mark.parametrize("noisy", [False, True])
def test_multiplier_fast_survey(noisy):
    result = multiplier_survey(2, 16, noisy)
    print("mul survey", noisy, result)
    assert all(v == 0 for k, v in result.items() if k != "trials"), result


@pytest.mark.slow
@pytest.mark.parametrize("width,trials,noisy", [(3, 64, False), (3, 128, True),
                                               (4, 256, False), (4, 128, True)])
def test_multiplier_large_survey(width, trials, noisy):
    result = multiplier_survey(width, trials, noisy)
    print("mul survey", width, noisy, result)
    assert result == dict(trials=2 * trials, wrong=0, double=0, missing=0, repeated=0, survivors=0)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_WIDE_MUL") != "1",
                    reason="width-8 cluster qualification requires DROSO_WIDE_MUL=1")
@pytest.mark.parametrize("trials,noisy", [(256, False), (128, True)])
def test_multiplier_width8_cluster_survey(trials, noisy):
    # Primitive qualification only; passed on Juno 452953/452954 (2026-10-12), which lifted the width-8 builder gate.
    result = multiplier_survey(8, trials, noisy)
    print("mul cluster survey", 8, noisy, result)
    assert result == dict(trials=2 * trials, wrong=0, double=0, missing=0, repeated=0, survivors=0)


@pytest.mark.slow
@pytest.mark.parametrize("width", [4, 8])
def test_qualified_array_kernel_nominal(width):
    # Exercise real operands, output mux, completion, commit and stage/master
    # reset ownership after the primitive has met its wider qualification.
    run_shape("MUL", width=width)


def test_retimed_array_deadline_regression_and_legacy_control():
    result = multiplier_survey(4, 1, operands=[(7, 1)], mul_version=0)
    assert result["missing"] > 0, result
    assert result["wrong"] == result["double"] == result["survivors"] == 0
    repaired = multiplier_survey(4, 1, operands=[(7, 1)])
    assert all(v == 0 for k, v in repaired.items() if k != "trials"), repaired
    control = multiplier_survey(4, 1, operands=[(7, 1)], oneshot=False)
    assert all(v == 0 for k, v in control.items() if k != "trials"), control


@pytest.mark.parametrize("width,records_version", [(1, False), (2, False), (4, True), (8, True)])
def test_qualified_array_builder_records_version_and_reset_ownership(width, records_version):
    pl, _, _ = small_shape("MUL", width)
    if records_version:
        assert pl.build_options["ordered_xor_oneshot_mul_version"] == 1
        spent = {i for i, role in enumerate(pl.net.roles) if ".mul.row" in role and ".spent." in role}
        targets = {d for s, d, q in zip(pl.net.src, pl.net.dst, pl.net.quanta)
                   if s == pl.cells[0].stage.reset_inh and q < 0}
        assert spent and spent <= targets
        for j in range(1, width):
            for i in range(width):
                assert f"out.mul.rt{j}.v{i}.or" in pl.net.roles
                for rail in (0, 1):
                    assert (f"out.mul.rt{j}.o{i}r{rail}.u" in pl.net.roles) == (i >= j)
    else:
        assert "ordered_xor_oneshot_mul_version" not in pl.build_options


def test_retime_suffix_preserves_whole_word_completion():
    from drosophilos.lib.gates import Gates, Rail2
    from drosophilos.lib.netlist import Drive, Netlist

    snapshots = []
    for output_start in (0, 2):
        net = Netlist(P)
        g = Gates(net, Drive.from_params(P))
        word = [Rail2(g.latch(f"a{i}r0"), g.latch(f"a{i}r1")) for i in range(4)]
        out = g.retime("rt", word, output_start=output_start)
        assert len(out) == 4 - output_start
        # Everything constructed before the output latches is identical,
        # including each bit-valid input, the completion tree and its delay.
        boundary = net.roles.index(f"rt.o{output_start}r0.u")
        snapshots.append((net.roles[:boundary], net.bias[:boundary],
                          [(s, d, q, delay) for s, d, q, delay in
                           zip(net.src, net.dst, net.quanta, net.delay)
                           if s < boundary and d < boundary]))
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("width", [3, 5, 6, 7, 9, 16])
def test_other_array_widths_remain_rejected(width):
    with pytest.raises(ValueError, match="rejects unqualified or unsupported array MUL width"):
        small_shape("MUL", width)


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_campaign_replay_checks_wide_array_version(version, monkeypatch):
    from drosophilos.bench import kernel_campaign

    pl, _, _ = small_shape("MUL", 4)
    monkeypatch.setattr(kernel_campaign, "block", lambda *a, **k: (None, pl, [], []))
    record = {"block": "wide-array", **pl.build_options}
    assert kernel_campaign.rebuild_record(record)[1] is pl
    key = "ordered_xor_oneshot_mul_version"
    if version is None:
        record.pop(key)
    else:
        record[key] = version
    with pytest.raises(ValueError, match="unsupported campaign build option ordered_xor_oneshot_mul_version"):
        kernel_campaign.rebuild_record(record)


def multiplier_witness_trace(*, mul_version=0, oneshot=True):
    """Retain real RefSim spikes beyond the old 900-ms observation/clear boundary."""
    net, drive, _, a, b, zero, out, _ = multiplier_fixture(
        4, oneshot=oneshot, mul_version=mul_version)
    sim = RefSim(net.topology(), P)
    sim.add_events(0, [0], [zero.r0.u], [drive.ignite])
    for word, value in ((a, 7), (b, 1)):
        sim.add_events(0, [200] * 4, [r.latches[value >> i & 1].u for i, r in enumerate(word)],
                       [drive.ignite] * 4)
    sim.run(10000)
    return net, sim.trace, out


def test_retimed_array_neuron_trace_disproves_lost_first_edge():
    net, trace, out = multiplier_witness_trace()
    expected = {
        "mul.row3.fa3.x.y0.u": 7929,
        "mul.row3.fa2.c0.u": 8711,
        "mul.row3.fa3.cinr0.d4": 8976,
        "mul.row3.fa3.s.once0.edge": 9018,
        "mul.row3.fa3.s.once0.edge_inh": 9029,
        "mul.row3.fa3.s.once0.spent.u": 9060,
        "mul.row3.fa3.s.y0.b.edge": 9060,
        "mul.row3.fa3.s.y0.b.edge_inh": 9071,
        "mul.row3.fa3.s.y0.u": 9102,
    }
    for role, first in expected.items():
        times = trace.neuron_steps(net.roles.index(role))
        assert times[0] == first, (role, times[:5])
        if role.endswith(".edge"):
            assert len(times) == 1
    assert len(trace.before(9000).neuron_steps(out[3].r0.u)) == 0
    assert len(trace.neuron_steps(out[3].r0.u)) > 0
    assert len(trace.neuron_steps(out[3].r1.u)) == 0
    legacy_net, legacy_trace, legacy_out = multiplier_witness_trace(oneshot=False)
    assert legacy_trace.neuron_steps(legacy_out[3].r0.u)[0] == 8976
    assert legacy_trace.neuron_steps(legacy_net.roles.index("mul.row3.fa3.s.y0.b.edge")).tolist() == [8934]
    _, fixed_trace, fixed_out = multiplier_witness_trace(mul_version=1)
    for bit, rail in enumerate(fixed_out):
        assert len(fixed_trace.before(9000).neuron_steps(rail.latches[7 >> bit & 1].u)) > 0
        assert len(fixed_trace.neuron_steps(rail.latches[1 - (7 >> bit & 1)].u)) == 0


def ram_survey(width, copies, noisy=False, cycles=8, *, shared=False):
    """Real word masters, exact address vetoes, COPY/READY and read registers.

    Each address is written, read, then overwritten with the opposite rails.
    Reads of *all* words after each write check that decoding preserves peers.
    """
    from drosophilos.lib.netlist import Drive, Netlist
    from drosophilos.lib.ram import add_memory, add_read_port, add_write_port
    from drosophilos.protocol.handshake import add_register
    from drosophilos.protocol.latch import add_experimental_autapses, add_latch, add_reset

    net = Netlist(P)
    drive = replace(Drive.from_params(P), rate_robust=True, kill_pulses=4)
    mem = add_memory(net, drive, "mem", 4, width)
    addr = [[add_latch(net, drive, f"a{i}r{r}") for r in (0, 1)] for i in range(2)]
    data = [[add_latch(net, drive, f"d{i}r{r}") for r in (0, 1)] for i in range(width)]
    read_addr = [[add_latch(net, drive, f"ra{i}r{r}") for r in (0, 1)] for i in range(2)]
    trigger, read = net.neuron("write"), net.neuron("read")
    wp = add_write_port(net, drive, "port", mem, trigger,
                        [[l.u for l in p] for p in addr], [[l.u for l in p] for p in data])
    ports = [wp]
    other_data = []
    if shared:
        from drosophilos.lib.kernel import _share_write_ports

        other_trigger = net.neuron("other_write")
        other_data = [[add_latch(net, drive, f"otherd{i}r{r}") for r in (0, 1)]
                      for i in range(width)]
        ports.append(add_write_port(net, drive, "otherport", mem, other_trigger,
                                   [[l.u for l in p] for p in addr],
                                   [[l.u for l in p] for p in other_data]))
        _share_write_ports(net, drive, "shared", mem,
                           [SimpleNamespace(name=f"store{i}", wport=port)
                            for i, port in enumerate(ports)])
    out = add_register(net, drive, "out", width, with_completion=True)
    add_read_port(net, drive, "readport", mem, read, [[l.u for l in p] for p in read_addr], out.rails)
    clear = add_reset(net, drive, "portclear", [l for p in addr + data + other_data for l in p] +
                      [l for port in ports for l in port.domain_latches])
    read_clear = add_reset(net, drive, "readclear", [l for p in read_addr for l in p])
    add_experimental_autapses(net, drive)
    sim = MixSurvey(net, copies, 6110, noisy)
    result = dict(writes=cycles * copies, reads=cycles * 4 * copies,
                  wrong=0, double=0, missing=0, survivors=0, rearm_missing=0,
                  min_copy_after_clear_steps=100000, max_copy_after_clear_steps=0)
    mask = (1 << width) - 1
    state = np.zeros((copies, 4), dtype=int)
    for word in mem.words:
        sim.load(word.rails, np.zeros(copies, int), 0, drive)
    # At 16 bits initial completion is step 2277 nominally. Its shared-port
    # mark clear must also drain before the first write (initialized-read /
    # non-overlapping-write contract); retain the original <=8 startup/draws.
    sim.run(2000 if width <= 8 else 4000)
    for tx in range(cycles):
        wp = ports[(tx // 4) % len(ports)]
        write_trigger = other_trigger if shared and wp is ports[1] else trigger
        word_idx = tx % 4
        values = (np.arange(copies) + word_idx) & mask
        if tx // 4 % 2:
            values ^= mask
        origin = sim.step_index
        sim.load(addr, np.full(copies, word_idx), origin, drive)
        sim.load(data, values if wp is ports[0] else values ^ mask, origin, drive)
        if shared:
            # The inactive port carries the opposite word: missing exclusion
            # marks must be observable as wrong/double rails, not a same-data copy.
            sim.load(other_data, values if wp is ports[1] else values ^ mask, origin, drive)
        sim.pulse(write_trigger, origin + 400, drive)
        sim.counts.fill(0)
        sim.first.fill(-1)
        sim.run(6000)
        state[:, word_idx] = values
        result["rearm_missing"] += int((sim.counts[:, wp.done[word_idx]] == 0).sum())
        margin = sim.first[:, wp.copies[word_idx].u] - sim.last[:, mem.words[word_idx].reset_inh]
        result["min_copy_after_clear_steps"] = min(result["min_copy_after_clear_steps"], int(margin.min()))
        result["max_copy_after_clear_steps"] = max(result["max_copy_after_clear_steps"], int(margin.max()))
        for j, word in enumerate(mem.words):
            score_rails(sim, word.rails, state[:, j], result, since=sim.step_index - 300)
        # End of the STORE stage: COPY and its source tokens use ordinary clears.
        sim.pulse(clear[0], sim.step_index, drive)
        sim.run(2000)
        members = [m for port in ports for latch in port.domain_latches for m in latch.members]
        if shared:
            members += [i for i, role in enumerate(net.roles)
                        if role.startswith("store") and role.endswith((".sel.u", ".sel.v"))]
        result["survivors"] += int((sim.last[:, members] >= sim.step_index - 500).sum())
        for j in range(4):
            origin = sim.step_index
            sim.load(read_addr, np.full(copies, j), origin, drive)
            sim.pulse(read, origin + 400, drive)
            sim.counts.fill(0)
            # The deeper completion tree needs a wider observation window;
            # preserve every original <=8-bit deadline and RNG step count.
            sim.run(2500 if width <= 8 else 4000)
            score_rails(sim, out.rails, state[:, j], result)
            result["rearm_missing"] += int((sim.counts[:, out.completion.u] == 0).sum())
            sim.pulse(out.reset_trigger, sim.step_index, drive)
            sim.pulse(read_clear[0], sim.step_index, drive)
            sim.run(2000)
            members = [m for latch in out.all_latches() for m in latch.members]
            result["survivors"] += int((sim.last[:, members] >= sim.step_index - 500).sum())
    return result


@pytest.mark.parametrize("noisy", [False, True])
def test_ram_fast_survey(noisy):
    result = ram_survey(2, 4, noisy)
    print("ram survey", noisy, result)
    assert all(result[k] == 0 for k in ("wrong", "double", "missing", "survivors", "rearm_missing")), result
    assert result["min_copy_after_clear_steps"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("width,noisy", [(4, False), (4, True), (8, True)])
def test_ram_large_survey(width, noisy):
    result = ram_survey(width, 16, noisy, cycles=16)
    print("ram survey", width, noisy, result)
    assert all(result[k] == 0 for k in ("wrong", "double", "missing", "survivors", "rearm_missing")), result
    assert result["min_copy_after_clear_steps"] > 0


def pacing_survey(copies=8, noisy=True, sources=2, count=3, cycles=9):
    from drosophilos.lib.kernel import add_pacing_ring
    from drosophilos.lib.netlist import Drive, Netlist
    from drosophilos.protocol.latch import add_experimental_autapses

    net = Netlist(P)
    drive = replace(Drive.from_params(P), rate_robust=True, kill_pulses=4)
    advance = [net.neuron(f"done{i}") for i in range(sources)]
    image = []
    rings, wrap = add_pacing_ring(net, drive, "ring", count, advance, image, true_guards=True)
    add_experimental_autapses(net, drive)
    sim = MixSurvey(net, copies, 7110, noisy)
    for latch in image:
        sim.pulse(latch.u, 0, drive)
    sim.run(2000)
    result = dict(advances=copies * sources * cycles, wrong=0, double=0, missing=0, bad_wraps=0)
    for tx in range(cycles):
        origin = sim.step_index
        sim.counts.fill(0)
        for j, source in enumerate(advance):
            # Several producer completions per token, with a 100 ms skew.
            sim.pulse(source, origin + j * 1000, drive)
        sim.run(4000 + sources * 1000)
        for lines in rings:
            for k, pair in enumerate(lines):
                score_rails(sim, [pair], np.full(copies, int(k == (tx + 1) % count)),
                            result, since=sim.step_index - 300)
        result["bad_wraps"] += int((sim.counts[:, wrap] != int((tx + 1) % count == 0)).sum())
    return result


@pytest.mark.parametrize("count,sources", [(1, 1), (2, 1), (3, 2)])
def test_pacing_fast_survey(count, sources):
    result = pacing_survey(count=count, sources=sources, cycles=3 * count)
    print("pacing survey", count, sources, result)
    assert all(v == 0 for k, v in result.items() if k != "advances"), result


@pytest.mark.slow
def test_pacing_large_survey():
    result = pacing_survey(copies=64, sources=3, count=8, cycles=24)
    print("pacing survey", result)
    assert all(v == 0 for k, v in result.items() if k != "advances"), result


@pytest.mark.slow
def test_render_f_nominal():
    from drosophilos.bench.kernel_campaign import block
    ks, pl, tokens, reference = block("render", P, **F)
    outputs, _, stats = run_pipeline_batched(pl, P, [tokens * 2], sim=RefSim(pl.net.topology(), P),
                                             progress=0, max_ms=40000, expect_outputs=[16])
    assert [[v for _, v in outputs[0][name]] for name in ks.outputs] == [
        [row[j] for row in reference * 2] for j in range(len(ks.outputs))]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    steps = [s for s, _ in outputs[0][ks.outputs[0]]]
    print("render F", pl.net.n, pl.net.nnz, stats["neural_ms"],
          "first completion ms", steps[0] * P.dt, "mean interval ms", (steps[-1] - steps[0]) * P.dt / 15)


# Width-16 evidence is deliberately built from primitives, never by relaxing or
# monkeypatching build_pipeline's width gate. These fixtures use the real ALU,
# ROM relays, MULP row ordering, staged commits and four-tap reset domains.
WIDE16_SHAPES = (
    "ADD", "SUB", "AND", "OR", "XOR", "MOV",
    "AND-specialized", "OR-specialized", "XOR-specialized", "MOV-specialized",
    "rom", "rom-specialized", "ram", "ram-shared", "MULP",
    "streams", "each", "wrap",
    "streams-specialized", "each-specialized", "wrap-specialized", "MUL",
)


def wide16_fixture(shape):
    from drosophilos.lib.adder import extend_reset
    from drosophilos.lib.alu import wire_alu, wire_outputs
    from drosophilos.lib.gates import Gates, Rail2
    from drosophilos.lib.kernel import (
        _NoProducer, _fault_latch, _unit_rails, add_alu_logic_tokens,
        add_pacing_ring, add_specialized_alu_logic_tokens,
    )
    from drosophilos.lib.netlist import Drive, Netlist
    from drosophilos.lib.ram import address_vetoes
    from drosophilos.lib.staged import add_staged_commit
    from drosophilos.protocol.celement import add_delay_chain, add_veto_neuron, add_veto_relay
    from drosophilos.protocol.handshake import add_register
    from drosophilos.protocol.latch import add_experimental_autapses, add_latch

    width = 16
    net = Netlist(P)
    drive = replace(Drive.from_params(P), rate_robust=True, kill_pulses=4)
    inputs = [add_register(net, drive, f"input{i}", width, with_completion=True)
              for i in range(2)]
    image, stages, masters, commits = [], [], [], []
    domains = [latch for reg in inputs for latch in reg.all_latches()]
    acts, rings, wrap = [], [], None
    op = shape.removesuffix("-specialized")
    stream_shape = op in ("streams", "each", "wrap")
    rows = range(width) if shape == "MULP" else range(2 if stream_shape else 1)
    rom_values = (0, 0x7fff, 0x2aaa, 0x4001)  # bit 15 exercises the constant-bit shortcut
    for j in rows:
        name = f"row{j}"
        g = Gates(net, drive, ordered_xor_oneshot=True,
                  ordered_xor_oneshot_mul_version=1)
        def word(label):
            return [Rail2(g.latch(f"{name}.{label}{i}r0"),
                          g.latch(f"{name}.{label}{i}r1")) for i in range(width)]
        def sample(label, rails):
            tokens = word(label)
            for i, pair in enumerate(rails):
                for r in (0, 1):
                    g.veto(f"{name}.{label}{i}r{r}.g", act, [pair[1 - r].u], tokens[i].latches[r])
            return tokens
        act = net.neuron(f"{name}.act")
        acts.append(act)
        if op in ("each", "wrap") and j:
            start = add_delay_chain(net, drive, f"{name}.phase", commits[-1].done_relay, 11)
            net.synapse(start, act, drive.ignite)
        if shape == "MULP" and j:
            start = add_delay_chain(net, drive, f"{name}.actd", commits[-1].done_relay, 11)
            net.synapse(start, act, drive.ignite)
            previous = masters[-1].rails
            a = sample("a", previous[width + 3:2 * width + 3])
            b = sample("b", previous[2 * width + 3:])
        else:
            a = sample("a", inputs[0].rails)
            b = sample("b", inputs[1].rails)
        sub = Rail2(add_latch(net, drive, f"{name}.sub0"),
                    add_latch(net, drive, f"{name}.sub1"))
        image.append(sub.latches[int(op == "SUB")])
        stage = add_register(net, drive, f"{name}.stage",
                             3 * width + 3 if shape == "MULP" else width + 3,
                             with_completion=True)
        _fault_latch(net, drive, name, stage)
        sr = add_staged_commit(net, drive, name, stage, _NoProducer(), copy_requires_rail=True)
        net.synapse(act, sr.commit_in, drive.ignite)
        if shape == "MULP":
            acc = (sample("acc", masters[-1].rails[:width]) if j else word("acc"))
            if not j:
                for i, rail in enumerate(acc):
                    g.veto(f"{name}.acc{i}.z", act, [], rail.r0)
            bd = [g.delayed(f"{name}.bd{i}", r, 6) for i, r in enumerate(b)]
            ad = [g.delayed(f"{name}.ad{i}", r, 17) for i, r in enumerate(a)]
            accd = [g.delayed(f"{name}.accd{i}", r, 23) for i, r in enumerate(acc)]
            zero = Rail2(g.latch(f"{name}.zero0"), g.latch(f"{name}.zero1"))
            g.veto(f"{name}.zero.g", act, [], zero.r0)
            pp = word("pp")
            for i, rail in enumerate(pp):
                g.veto(f"{name}.pp{i}.g", ad[i].r1.u, [bd[j].r0.u], rail.r1)
                g.veto(f"{name}.pp{i}.a", ad[i].r0.u, [], rail.r0)
                g.veto(f"{name}.pp{i}.b", ad[i].r1.u, [bd[j].r1.u], rail.r0)
            result, _, _, _ = g.ripple_adder_ordered(name, accd, [zero] * j + pp[:width - j], zero, 5)
            c = Rail2(g.latch(f"{name}.c0"), g.latch(f"{name}.c1"))
            v = Rail2(g.latch(f"{name}.v0"), g.latch(f"{name}.v1"))
            g.veto(f"{name}.c.g", act, [], c.r0)
            g.veto(f"{name}.v.g", act, [], v.r0)
            wire_outputs(net, drive, ad + bd, stage, list(range(width + 3, 3 * width + 3)))
        else:
            fixed_op = ("ADD" if j == 0 else "MOV") if stream_shape else op
            if op == "rom":
                # Same constant-bit optimization, full-word high-address veto,
                # exact decoder and token-to-PASSB path as kernel LOAD.
                # LOAD has an all-zero A token. The sampled address remains
                # in the stage domain and tests full-width clearing.
                a = word("zeroa")
                for i, rail in enumerate(a):
                    g.veto(f"{name}.zeroa{i}.g", act, [], rail.r0)
                b = word("romdata")
                taps = [[l.u for l in pair] for pair in inputs[0].rails[:2]]
                high = [pair[1].u for pair in inputs[0].rails[2:]]
                same = {i: rom_values[0] >> i & 1 for i in range(width)
                        if len({value >> i & 1 for value in rom_values}) == 1}
                for i, r in same.items():
                    add_veto_relay(net, drive, f"rom.b{i}", act, [], b[i].latches[r])
                for address, value in enumerate(rom_values):
                    veto = add_veto_neuron(net, drive, f"rom.w{address}", address_vetoes(taps, address) + high)
                    for i, rail in enumerate(b):
                        if i not in same:
                            add_veto_relay(net, drive, f"rom.w{address}.b{i}", act, [],
                                           rail.latches[value >> i & 1], veto_neurons=[veto])
                fixed_op = "MOV"
            if shape.endswith("-specialized") and fixed_op in ("AND", "OR", "XOR", "MOV"):
                result, c, v = add_specialized_alu_logic_tokens(g, name, a, b, sub, act, fixed_op)
            else:
                unit = _unit_rails(net, drive, f"{name}.unit", fixed_op, g)
                image.extend(rail.latches[int(k == g.unit_rails[-1][1])] for k, rail in enumerate(unit))
                result, c, v = add_alu_logic_tokens(g, name, a, b, unit, sub, act)
        wire_alu(net, drive, result, c, v, stage)
        extend_reset(net, drive, stage, g.latches, g.gates)
        stages.append(stage)
        masters.append(sr.master)
        commits.append(sr)
        domains += stage.all_latches() + sr.master.all_latches() + g.latches
        domains += [stage.fault_latch, sr.commit, sr.granted, sr.copy]
    if op in ("each", "wrap"):
        rings, wrap = add_pacing_ring(net, drive, "pacing", 2 if op == "wrap" else 1,
                                     [sr.done_relay for sr in commits], image, true_guards=True)
    add_experimental_autapses(net, drive)
    return SimpleNamespace(net=net, drive=drive, inputs=inputs, image=image, stages=stages,
                           masters=masters, acts=acts, domains=domains, rings=rings, wrap=wrap,
                           rom_values=rom_values)


def wide16_datapath_survey(shape, trials, noisy=False):
    """Repeated physical copies, complemented reloads and real staged COPY.

    MULP chains all sixteen rows through the preceding master's done pulse.
    Host-paced streams inspect independent words and completion ordering; rings
    consume actual commit-done pulses rather than synthetic host completions.
    This is a primitive composition, not the compiler's full request/watchdog FSM.
    """
    from drosophilos.lib.alu import alu_reference

    fixture = wide16_fixture(shape)
    base_shape = shape.removesuffix("-specialized")
    net, drive = fixture.net, fixture.drive
    result = dict(trials=2 * trials, wrong=0, double=0, missing=0,
                  repeated=0, survivors=0, rearm_missing=0, faults=0, bad_wraps=0, bad_order=0)
    once = [i for i, role in enumerate(net.roles) if ".once" in role and role.endswith(".edge")]
    members = sorted({m for latch in fixture.domains for m in latch.members} |
                     {i for i, role in enumerate(net.roles) if role.endswith((".commit2.u", ".commit2.v"))})
    # Include stage -> grant -> master READY/COPY -> completion, after the
    # 16-bit worst-case carry. Ordered phases also wait for the second stream.
    duration = (16000 * 16 if shape == "MULP" else
                24000 if base_shape in ("each", "wrap") else 16000)
    for start in range(0, trials, 16):
        size = min(16, trials - start)
        sim = MixSurvey(net, size, 16100 + start, noisy)
        av, bv = sim.rng.integers(0, 65536, (2, size))
        # Directed long carry, signed overflow, zero and alternating high rails
        # lead each condition; the remaining pairs are deterministic random draws.
        directed = [(65535, 1), (32767, 1), (0, 65535), (0xaaaa, 0x5555)]
        for k in range(min(size, max(0, len(directed) - start))):
            av[k], bv[k] = directed[start + k]
        for latch in fixture.image:
            sim.pulse(latch.u, 0, drive)
        sim.run(2000)
        for tx in range(2):
            origin = sim.step_index
            sim.counts.fill(0)
            sim.first.fill(-1)
            addr = ((np.arange(start, start + size) + tx) % 4 if shape.startswith("rom") else av)
            sim.load(fixture.inputs[0].rails, addr, origin, drive)
            sim.load(fixture.inputs[1].rails, bv, origin, drive)
            scheduled_acts = fixture.acts[:1] if base_shape in ("MULP", "each", "wrap") else fixture.acts
            for j, act in enumerate(scheduled_acts):
                sim.pulse(act, origin + 400 + j * 1000, drive)
            sim.run(duration)
            for j, master in enumerate(fixture.masters):
                if shape == "MULP":
                    acc = av * (bv & ((1 << (j + 1)) - 1)) & 65535
                    want = acc | ((acc == 0).astype(np.int64) << 17) | (av << 19) | (bv << 35)
                else:
                    op = shape.removesuffix("-specialized")
                    if op == "rom":
                        values = np.asarray(fixture.rom_values)[addr]
                        want = values | ((values == 0).astype(np.int64) << 17)
                    else:
                        if base_shape in ("streams", "each", "wrap"):
                            op = "ADD" if j == 0 else "MOV"
                        want = [alu_reference(int(a), int(b), op, 16)["word"] for a, b in zip(av, bv)]
                score_rails(sim, master.rails, want, result, since=sim.step_index - 300)
                result["rearm_missing"] += int((sim.counts[:, master.completion.u] == 0).sum())
            result["repeated"] += int((sim.counts[:, once] > 1).sum())
            for stage in fixture.stages:
                result["faults"] += int((sim.counts[:, stage.fault_latch.u] > 0).sum())
            if base_shape in ("each", "wrap"):
                times = sim.first[:, [master.completion.u for master in fixture.masters]]
                result["bad_order"] += int(((times[:, 0] < 0) | (times[:, 1] <= times[:, 0])).sum())
            if fixture.rings:
                count = 2 if base_shape == "wrap" else 1
                for lines in fixture.rings:
                    for k, pair in enumerate(lines):
                        score_rails(sim, [pair], np.full(size, int(k == (tx + 1) % count)),
                                    result, since=sim.step_index - 300)
                result["bad_wraps"] += int((sim.counts[:, fixture.wrap] != int((tx + 1) % count == 0)).sum())
            # Clear masters too so survivors cannot masquerade as a valid reload.
            # An explicit master clear raises READY and hence COPY again.
            # Drain that real READY path before clearing the stage domain;
            # simultaneous clears would leave a newly ignited COPY behind.
            for reg in fixture.masters:
                sim.pulse(reg.reset_trigger, sim.step_index, drive)
            sim.run(2000)
            for reg in fixture.inputs + fixture.stages:
                sim.pulse(reg.reset_trigger, sim.step_index, drive)
            sim.run(2000)
            result["survivors"] += int((sim.last[:, members] >= sim.step_index - 500).sum())
            av, bv = av ^ 65535, bv ^ 65535
    return result


def run_wide16_survey(shape, noisy, *, smoke=False):
    """One shape/physical condition per cluster job, with visible timing/counters."""
    wall_start, cpu_start = perf_counter(), process_time()
    if shape in ("ram", "ram-shared"):
        result = ram_survey(16, 1 if smoke else 16, noisy,
                            cycles=8 if smoke else 16, shared=shape == "ram-shared")
        failure_keys = ("wrong", "double", "missing", "survivors", "rearm_missing")
    elif shape == "MUL":
        trials = 1 if smoke else (256 if not noisy else 128)
        result = multiplier_survey(16, trials, noisy,
                                   operands=[(65535, 1)] if smoke else None, mul_version=1)
        failure_keys = ("wrong", "double", "missing", "repeated", "survivors")
    else:
        # ADD/SUB match the largest width-8 arithmetic survey (2,048 pairs /
        # 4,096 transactions). Other datapaths use 16 pairs / 32 transactions,
        # >= the small kernel and width-8 MULP exposures per condition.
        trials = ((4 if shape.startswith("rom") else 1) if smoke else
                  (2048 if shape in ("ADD", "SUB") else 16))
        result = wide16_datapath_survey(shape, trials, noisy)
        failure_keys = ("wrong", "double", "missing", "repeated", "survivors",
                        "rearm_missing", "faults", "bad_wraps", "bad_order")
    print("wide16 survey", shape, "mix-B" if noisy else "nominal", "smoke" if smoke else "cluster",
          result, f"wall_s={perf_counter() - wall_start:.3f}", f"cpu_s={process_time() - cpu_start:.3f}",
          flush=True)
    assert all(result[key] == 0 for key in failure_keys), result
    if shape in ("ram", "ram-shared"):
        assert result["min_copy_after_clear_steps"] > 0, result


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_WIDE16") != "1",
                    reason="width-16 cluster evidence requires DROSO_WIDE16=1")
@pytest.mark.parametrize("noisy", [pytest.param(False, id="nominal"), pytest.param(True, id="mix-B")])
@pytest.mark.parametrize("shape", WIDE16_SHAPES)
def test_width16_cluster_survey(shape, noisy):
    run_wide16_survey(shape, noisy)


@pytest.mark.skipif(os.environ.get("DROSO_WIDE16_SMOKE") != "1",
                    reason="width-16 laptop smoke requires DROSO_WIDE16_SMOKE=1")
@pytest.mark.parametrize("shape", WIDE16_SHAPES)
def test_width16_smoke_survey(shape):
    run_wide16_survey(shape, False, smoke=True)
