"""Experimental campaign opt-in; correctness does not imply latch qualification."""

from itertools import product
from collections import Counter

import numpy as np
import pytest

from drosophilos.bench import kernel_campaign, stage_d
from drosophilos.bench.stall_diag import build_tick_pipeline
from drosophilos.lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from drosophilos.protocol.latch import EXPERIMENTAL_AUTAPSE_VERSION, Latch
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim
from test_build_options import _pipeline_fingerprint
from test_stable_latch import AUTAPSE, _modify


P = Params()
SPEC = [{"name": "sum", "op": "ADD", "a": "input", "b": ("const", "one")},
        {"name": "out", "op": "XOR", "a": "sum", "b": ("const", "one")}]
OPTION_NAMES = ("zero_once", "robust_request_clear", "experimental_register_reset")


def _build(**options):
    return build_pipeline(P, 2, SPEC, consts={"one": 1}, rate_robust=True,
                          experimental_autapse=True, **options)


def test_tick_build_equals_committed_probe_edge_for_edge():
    _, probe, _, _ = kernel_campaign.block("tick", P, rate_robust=True)
    net = probe.net
    before = (net.n, net.nnz)
    edges = {(s, d): q for s, d, q in zip(net.src, net.dst, net.quanta)}
    latches = [Latch(u, v) for (u, v), q in edges.items()
               if q == probe.drive.loop and edges.get((v, u)) == probe.drive.loop
               and net.roles[u].endswith(".u")
               and net.roles[v] == net.roles[u].removesuffix(".u") + ".v"]
    for latch in latches:
        _modify(net, latch, AUTAPSE)
    _, actual, _, _ = kernel_campaign.block("tick", P, rate_robust=True,
                                            experimental_autapse=True)
    for field in ("roles", "src", "dst", "quanta", "delay", "bias", "incoming",
                  "groups", "rate_readouts", "inhibition_mirrors"):
        assert getattr(actual.net, field) == getattr(net, field)
    assert _pipeline_fingerprint(actual) == _pipeline_fingerprint(probe)
    assert actual.net.n == before[0]
    assert actual.net.nnz - before[1] == 2 * len(latches)
    assert len(latches) == 3215
    assert (actual.net.n, actual.net.nnz) == (30643, 62366)
    assert all(s == d and q == -round(actual.drive.loop * .2) and delay == 0
               for s, d, q, delay in zip(actual.net.src[before[1]:], actual.net.dst[before[1]:],
                                          actual.net.quanta[before[1]:], actual.net.delay[before[1]:]))
    # Feedback remains confined to storage; genuine reader/qualifier clears survive.
    for source, mirrors in actual.net.inhibition_mirrors.items():
        for target, gain in mirrors:
            expected = sorted((actual.net.src[e], round(actual.net.quanta[e] * gain), actual.net.delay[e])
                              for e in actual.net.incoming[source]
                              if actual.net.quanta[e] < 0 and actual.net.src[e] != source)
            observed = sorted((actual.net.src[e], actual.net.quanta[e], actual.net.delay[e])
                              for e in actual.net.incoming[target] if actual.net.quanta[e] < 0)
            # Qualifiers can also have their own controller/veto inhibitors.
            assert not (Counter(expected) - Counter(observed))


@pytest.mark.parametrize("values", list(product((False, True), repeat=3)))
def test_all_allowed_combinations_compute_on_multi_cell_kernel(values):
    options = dict(zip(OPTION_NAMES, values))
    pl = _build(**options)
    assert all(pl.build_options[key] == value for key, value in options.items())
    outs, _, stats = run_pipeline(pl, P, [0, 1, 3], max_ms=6000)
    assert [v for _, v in outs] == [0, 3, 1]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0


class _NoisyRefSim(RefSim):
    """Small copy count under full mix B, including independent background strays."""

    def __init__(self, net, copies=2, seed=109):
        topo = net.topology()
        rng = np.random.default_rng(seed)
        super().__init__(topo, P, n_nodes=copies,
                         quanta=np.rint(topo.quanta * np.exp(rng.normal(0, .04, (copies, topo.nnz)))),
                         V_th=P.V_th + rng.normal(0, .2, (copies, topo.n)),
                         bias=np.asarray(net.bias) + rng.normal(0, .2, (copies, topo.n)))
        self.rng = rng

    def step(self):
        count = self.rng.binomial(self.B * self.n, 5 * P.dt / 1000)
        flat = self.rng.choice(self.B * self.n, count, replace=False)
        self._events[self.step_index].append(
            (flat // self.n, flat % self.n, np.full(count, 150)))
        super().step()


@pytest.mark.parametrize("other_options", [False, True])
def test_small_mix_b_multi_cell_kernel(other_options):
    pl = _build(**dict.fromkeys(OPTION_NAMES, other_options))
    outputs, _, stats = run_pipeline_batched(pl, P, [[0, 1, 3]] * 2, max_ms=6000,
                                             sim=_NoisyRefSim(pl.net), progress=0)
    assert [[v for _, v in copy["out"]] for copy in outputs] == [[0, 3, 1]] * 2
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0


def test_recorded_flag_rebuild_and_explicit_false():
    _, enabled, _, _ = kernel_campaign.block("tick", P, rate_robust=True,
                                              experimental_autapse=True)
    assert enabled.build_options["experimental_autapse"] is True
    assert enabled.build_options["experimental_autapse_version"] == EXPERIMENTAL_AUTAPSE_VERSION == 1
    _, _, rebuilt = build_tick_pipeline({"block": "tick", **enabled.build_options})
    assert _pipeline_fingerprint(enabled) == _pipeline_fingerprint(rebuilt)
    record = {"block": "tick", **enabled.build_options}
    record.pop("experimental_autapse")
    _, _, old = build_tick_pipeline(record)
    _, explicit_off, _, _ = kernel_campaign.block("tick", P, rate_robust=True,
                                                   experimental_autapse=False)
    assert old.build_options["experimental_autapse"] is False
    assert _pipeline_fingerprint(old) == _pipeline_fingerprint(explicit_off)


@pytest.mark.parametrize("entry", ["build", "campaign", "recheck", "diagnostic"])
def test_rate_readers_required(entry):
    with pytest.raises(ValueError, match="experimental_autapse requires rate_robust=True"):
        if entry == "build":
            build_pipeline(P, 2, SPEC, consts={"one": 1}, experimental_autapse=True)
        elif entry == "campaign":
            kernel_campaign.block("tick", P, experimental_autapse=True)
        elif entry == "recheck":
            stage_d.recheck_record({"build_options": {"experimental_autapse": True}})
        else:
            build_tick_pipeline({"block": "tick", "experimental_autapse": True})


@pytest.mark.parametrize("module", [stage_d, kernel_campaign])
def test_cli_help_states_qualification_limits(module):
    help_text = module.parser().format_help()
    assert "--experimental-autapse" in help_text
    assert "44-step" in help_text and "reload margin" in help_text


@pytest.mark.parametrize("datapath", ["generic", "specialized"])
def test_every_role_named_latch_and_only_latches_have_autapses(datapath):
    # Enumerate roles first, independently of the transform's edge/weight predicate.
    _, pl, _, _ = kernel_campaign.block("tick", P, datapath=datapath,
                                        rate_robust=True, experimental_autapse=True)
    net = pl.net
    roles = {role: i for i, role in enumerate(net.roles)}
    pairs = {}
    for role, member in roles.items():
        if role.endswith((".u", ".v")):
            pairs.setdefault(role[:-2], {})[role[-1]] = member
    assert pairs
    members = set()
    for name, pair in pairs.items():
        assert set(pair) == {"u", "v"}, name
        u, v = pair["u"], pair["v"]
        # Every role pair really is a two-neuron excitatory storage loop.
        for src, dst in ((u, v), (v, u)):
            weights = [net.quanta[e] for e in net.incoming[dst] if net.src[e] == src]
            assert weights == [pl.drive.loop], name
        members.update((u, v))
    self_edges = [(src, q, delay) for src, dst, q, delay
                  in zip(net.src, net.dst, net.quanta, net.delay) if src == dst]
    assert Counter(src for src, _, _ in self_edges) == Counter(dict.fromkeys(members, 1))
    # Literal circuit identity: changing both production constants and selection
    # must not silently make this independent v1 assertion pass.
    assert all(q == -round(pl.drive.loop * 0.2) and delay == 0
               for _, q, delay in self_edges)
    if datapath == "generic":
        assert len(pairs) == 3215


@pytest.mark.parametrize("shape", [
    {"mems": {"ram": (2, {})}},
    {"phases": [("input", None, "each")]},
    {"streams": ["input", "second"]},
    *({"spec": [{"name": "out", "op": op, "a": "input", "b": ("const", "one")}]}
      for op in ("MUL", "MULP", "MULP_ROW")),
])
def test_unevaluated_shapes_rejected(shape):
    options = dict(shape)
    spec = options.pop("spec", SPEC)
    with pytest.raises(ValueError, match="experimental_autapse supports only single-stream"):
        build_pipeline(P, 2, spec, consts={"one": 1}, rate_robust=True,
                       experimental_autapse=True, **options)


@pytest.mark.parametrize("stage", [False, True])
def test_version_enforced_by_rebuild_and_recheck(stage):
    if stage:
        k = stage_d.load_kernel(stage_d.PROGRAM)
        pl = stage_d.build(k, P, "generic", rate_robust=True, experimental_autapse=True)
        record = {"stage": "D", "build_options": dict(pl.build_options)}
    else:
        _, pl, _, _ = kernel_campaign.block("tick", P, rate_robust=True,
                                            experimental_autapse=True)
        record = {"block": "tick", **pl.build_options}
    options = record["build_options"] if stage else record
    assert options["experimental_autapse_version"] == 1
    _, _, rebuilt = build_tick_pipeline(record)
    assert _pipeline_fingerprint(rebuilt) == _pipeline_fingerprint(pl)
    if stage:
        stage_d.recheck_record(record)

    # Pre-version captures built exactly v1, including when the opt-in was on.
    options.pop("experimental_autapse_version")
    _, _, old = build_tick_pipeline(record)
    assert _pipeline_fingerprint(old) == _pipeline_fingerprint(pl)
    if stage:
        stage_d.recheck_record(record)

    for version in (0, 2, None, "1"):
        options["experimental_autapse_version"] = version
        with pytest.raises(ValueError, match="unsupported experimental-autapse circuit"):
            build_tick_pipeline(record)
        with pytest.raises(ValueError, match="unsupported experimental-autapse circuit"):
            stage_d.recheck_record(record)


def test_nested_version_takes_precedence():
    record = {"experimental_autapse_version": 2,
              "build_options": {"experimental_autapse_version": 1}}
    build_tick_pipeline(record)
    stage_d.recheck_record(record)
    record["experimental_autapse_version"] = 1
    record["build_options"]["experimental_autapse_version"] = 2
    with pytest.raises(ValueError, match="unsupported experimental-autapse circuit"):
        build_tick_pipeline(record)
    with pytest.raises(ValueError, match="unsupported experimental-autapse circuit"):
        stage_d.recheck_record(record)
