"""Stage D exit harness (drosophilos/bench/stage_d.py, docs/stage_d.md)."""

import json
import shutil

import pytest

from drosophilos.bench import stage_d as sd
from drosophilos.sim.model import Params

K = sd.load_kernel()
PARAMS = Params()
HAS_CC = any(shutil.which(x) for x in ("clang", "gcc", "cc"))


def _fake_outs(states, k=K, step0=1000, per=100):
    """Committed words shaped like the runner's: cell -> [(step, value)], one per tick."""
    return {k.cells[f]: [(step0 + per * t, s[f]) for t, s in enumerate(states)] for f in k.fields}


def _fake_load_events(n, step0=1000, per=100):
    return [{"schedule_index": t, "event_step": step0 + per * t, "retry": False} for t in range(n)]


def test_state_cells_and_canonical_order():
    assert K.fields == ("px", "mx", "health")
    assert K.cells == {"px": "c4_sel", "mx": "c9_sel", "health": "c12_sel"}
    assert sd.reference_start(K) == {"px": 20, "mx": 90, "health": 100}
    assert [x["offset"] for x in sd.canonical_layout(8)] == [0, 1, 2]
    assert [x["offset"] for x in sd.canonical_layout(16)] == [0, 2, 4]


def test_canonical_state_is_fixed_width_and_order_stable():
    a = sd.canonical_state({"px": 1, "mx": 2, "health": 255})
    b = sd.canonical_state({"health": 255, "vel": 9, "mx": 2, "px": 1})  # dict order and extra keys ignored
    assert a == b == bytes([1, 2, 255])
    assert sd.canonical_state({"px": 0, "mx": 0, "health": 0}) == b"\0\0\0"
    assert sd.canonical_state({"px": 0x1234, "mx": 1, "health": 0xFFFF}, width=16) == bytes.fromhex("12340001ffff")
    assert sd.decode_canonical(bytes([1, 2, 255])) == {"px": 1, "mx": 2, "health": 255}
    for bad in ({"px": 256, "mx": 0, "health": 0}, {"px": -1, "mx": 0, "health": 0},
                {"px": None, "mx": 0, "health": 0}, {"px": True, "mx": 0, "health": 0}, {"px": 1, "mx": 2}):
        with pytest.raises(ValueError):
            sd.canonical_state(bad)


def test_tokens_scripted_prefix_then_seeded_random():
    t = sd.tokens_for(1000, 7)
    assert len(t) == 1000 and t[:8] == sd.CAMPAIGN_TOKENS and t[:len(sd.SCRIPTED_TOKENS)] == sd.SCRIPTED_TOKENS
    assert t == sd.tokens_for(1000, 7) and t != sd.tokens_for(1000, 8)
    assert all(0 <= x < 256 for x in t) and sd.tokens_for(6, 0) == sd.CAMPAIGN_TOKENS[:6]


def test_scripted_prefix_reaches_the_walls_and_contact():
    ref = sd.reference_states(K, sd.SCRIPTED_TOKENS)
    health = [100] + [s["health"] for s in ref]
    assert sum(1 for a, b in zip(health, health[1:]) if a != b) >= 10  # contact costs health
    assert 0 in health and 246 in health  # through zero: the unsigned wrap
    assert sum(1 for s in ref if s["px"] == 0) >= 5 and max(s["px"] for s in ref) == 127  # the west wall


def test_reference_matches_the_ir_interpreter_and_the_c_program():
    tokens = sd.SCRIPTED_TOKENS
    ref = sd.reference_states(K, tokens)
    assert sd.ir_states(K, tokens) == ref
    if not HAS_CC:
        pytest.skip("no C compiler")
    assert sd.c_states(K, tokens) == ref
    # and the rewrite used for the per-tick states changes nothing: the unmodified tick2.c
    # (8 ticks, px and mx per tick, health at the end) agrees with the same reference
    from drosophilos.compiler.golden import run_golden
    g = run_golden(K.source, sd.CAMPAIGN_TOKENS, list(K.fields), 8)
    assert g["outs"] == [v for s in ref[:8] for v in (s["px"], s["mx"])] + [ref[7]["health"]]
    assert dict(g["state"]) == ref[7]
    chk = sd.three_point_check(K, tokens, ref, 20)
    assert chk["ir_equal"] and chk["c_equal"] and chk["c_ticks"] == 20


def test_compare_counts_and_first_mismatch_on_synthetic_commits():
    tokens = sd.tokens_for(12, 3)
    ref = sd.reference_states(K, tokens)
    ok = sd.compare_copy(K, ref, tokens, _fake_outs(ref), t_load0=0, dt=0.1)
    assert ok["status"] == "matched" and ok["matched"] == 12 and ok["first_mismatch"] is None
    assert ok["tick_ms"][:2] == [100.0, 110.0]
    bad = [dict(s) for s in ref]
    bad[5]["health"] ^= 1
    c = sd.compare_copy(K, ref, tokens, _fake_outs(bad), t_load0=0, dt=0.1)
    fm = c["first_mismatch"]
    assert (c["status"], c["matched"], c["wrong"], c["unscored"]) == ("mismatch", 5, 1, 6)
    assert (fm["tick"], fm["field"], fm["expected"], fm["got"], fm["token"]) == (5, "health", ref[5]["health"], bad[5]["health"], tokens[5])
    # unfinished: the last 3 ticks never committed, the run hit its neural-time limit
    short = sd.compare_copy(K, ref, tokens, _fake_outs(ref[:9]), t_load0=0, dt=0.1, limit="max_ms")
    assert (short["status"], short["completed"], short["missing"]) == ("truncated", 9, 3)
    assert sd.verdict([ok, c, short], 12) == "exit not met: 1 mismatch, 1 truncated, refusals 0, retries 0"
    assert sd.verdict([ok, ok], 12) == "exit met"


def test_faults_block_exit_met_even_when_every_copy_matches():
    tokens = sd.tokens_for(4, 2)
    ref = sd.reference_states(K, tokens)
    ok = sd.compare_copy(K, ref, tokens, _fake_outs(ref), t_load0=0, dt=0.1)
    assert sd.verdict([ok], 4, faults=1) == "exit not met: 1 faults (run-level), refusals 0, retries 0"


def test_a_copy_stalled_before_a_max_ms_cut_is_not_truncated():
    tokens = sd.tokens_for(12, 3)
    ref = sd.reference_states(K, tokens)
    # Its last state word was at step 1,800, long before this max-ms run ended at 5,000.
    c = sd.compare_copy(K, ref, tokens, _fake_outs(ref[:9]), t_load0=1000, dt=0.1,
                        limit="max_ms", run_end_step=5000, stall_ms=100.0)
    assert (c["status"], c["stopped_at_tick"]) == ("stalled", 8)


def test_commit_counts_detect_missing_plus_extra_on_a_constant_field():
    tokens = sd.tokens_for(6, 8)
    ref = sd.reference_states(K, tokens)
    outs = _fake_outs(ref)
    health = outs[K.cells["health"]]
    # health remains 100 here.  Dropping tick 1 and adding a second tick-2 commit preserves
    # both the observed list length and its values, so the old positional comparison passed.
    del health[1]
    health.append((1220, ref[2]["health"]))
    health.sort()
    c = sd.compare_copy(K, ref, tokens, outs, t_load0=1000, dt=0.1,
                        load_events=_fake_load_events(len(tokens)))
    fm = c["first_mismatch"]
    assert c["status"] == "mismatch" and c["commit_counts_checkable"] is True
    assert (fm["field"], fm["tick"], fm["expected_commits"], fm["got_commits"]) == ("health", 1, 1, 0)


def test_a_word_applied_twice_is_named():
    tokens = sd.tokens_for(10, 4)
    ref = sd.reference_states(K, tokens)
    # tick 4's word applied a second time: every later tick runs one word late
    doubled = ref[:5] + sd.reference_states(K, tokens[4:9], ref[4])
    c = sd.compare_copy(K, ref, tokens, _fake_outs(doubled), t_load0=0, dt=0.1,
                        load_events=[{"schedule_index": 4, "retry": True}])
    assert c["first_mismatch"]["tick"] == 5 and c["first_mismatch"]["class"] == "previous word applied twice"
    assert c["applied_twice"] and c["retried_ticks"] == [4] and c["retried_ticks_matched"] is True
    lag = ref[:3] + [ref[2]] + ref[3:9]  # the state lags a tick (seed 109 copy 61's class)
    assert sd.compare_copy(K, ref, tokens, _fake_outs(lag), t_load0=0, dt=0.1)["first_mismatch"]["class"] == "state lags a tick"
    extra = _fake_outs(ref)
    extra[K.cells["px"]].append((99999, 0))  # a commit beyond the requested ticks
    assert sd.compare_copy(K, ref, tokens, extra, t_load0=0, dt=0.1)["duplicates"] == 1


def test_cli_writes_the_record(tmp_path, monkeypatch):
    """The record's shape, with the neural run replaced by the reference (copy 1 corrupted):
    the neural path itself is the slow test below."""
    calls = {}

    def fake_run(k, pl, params, tokens, *, copies=1, **kw):
        calls.update(kw, copies=copies)
        ref = sd.reference_states(k, tokens)
        bad = [dict(s) for s in ref]
        bad[3]["mx"] = (bad[3]["mx"] + 2) % 256
        outs = [_fake_outs(ref), _fake_outs(bad)][:copies]
        st = {"load_steps": [[1000]] * copies, "load_events": [[] for _ in range(copies)], "refused": [[] for _ in range(copies)],
              "faults": 0, "timeouts": 0, "bad_outputs": 0, "blocked_nodes": [], "host_stalls": False, "truncated": False,
              "stopped_on_stall": False, "simulator": "Fake", "neural_ms": 1000.0, "run_started_perf": 0.0}
        return {"outs": outs, "stats": st, "wall": [[(c, s_, 1.0) for c, l in o.items() for s_, _ in l] for o in outs]}

    monkeypatch.setattr(sd, "run_neural", fake_run)
    out = tmp_path / "stage_d" / "rec.json"
    sd.main(["--ticks", "10", "--seed", "5", "--copies", "2", "--mix", "B", "--max-ms", "90000",
             "--c-ticks", "10", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert calls["mix"] == "B" and calls["max_ms"] == 90000 and calls["copies"] == 2
    assert rec["build_options"]["relight_repair_delay"] is True and "true_guard_version" in rec["build_options"]
    assert rec["tokens"] == sd.tokens_for(10, 5) and rec["max_ms_source"] == "given"
    assert rec["per_copy"][0]["status"] == "matched"
    assert rec["per_copy"][1]["first_mismatch"]["tick"] == 3 and rec["per_copy"][1]["first_mismatch"]["field"] == "mx"
    assert rec["counters"] == {"requested": 20, "completed": 20, "matched": 13, "wrong": 1, "unscored": 6, "missing": 0,
                               "duplicates": 0, "invalid": 0, "refusals": 0, "retries": 0}
    assert rec["verdict"] == "exit not met: 1 mismatch, refusals 0, retries 0" and rec["retried_commit_applied_once"] is True
    assert rec["canonical_format"] == sd.FORMAT and len(rec["reference_canonical_hex"]) == 10
    assert rec["three_point_check"]["ir_equal"] and len(rec["tick_ms"][0]) == 10
    rechecked = sd.main(["--recheck", str(out)])
    assert rechecked["per_copy"][1]["status"] == "mismatch"
    assert rechecked["recheck"]["commit_counts_checkable"] is False


@pytest.fixture(scope="module")
def neural6():
    """Six ticks of the campaign tokens on one nominal RefSim copy (~40 s neural: several
    minutes of wall on a laptop core)."""
    tokens = sd.tokens_for(6, 0)
    pl = sd.build(K, PARAMS)
    r = sd.run_neural(K, pl, PARAMS, tokens, backend="ref", max_ms=60000, stall_ms=30000)
    return tokens, pl, r


@pytest.mark.slow
def test_neural_six_ticks_match_the_reference_every_tick(neural6):
    tokens, pl, r = neural6
    assert [o.name for o in pl.outputs] == ["c4_sel", "c9_sel", "c12_sel"]
    ref = sd.reference_states(K, tokens)
    st = r["stats"]
    c = sd.compare_copy(K, ref, tokens, r["outs"][0], t_load0=st["load_steps"][0][0], dt=PARAMS.dt,
                        load_events=st["load_events"][0], refused=st["refused"][0])
    assert c["status"] == "matched" and c["matched"] == 6 and c["duplicates"] == 0, c
    assert st["faults"] == 0 and st["timeouts"] == 0 and not c["applied_twice"]
    print("stage D, 6 ticks:", {x: c[x] for x in ("tick_ms", "refusals", "retries")},
          "wall", round(st["run_wall_s"]), "neural", st["neural_ms"])


@pytest.mark.slow
def test_a_corrupted_reference_gives_a_first_mismatch_record(neural6):
    tokens, _, r = neural6
    ref = sd.reference_states(K, tokens)
    ref[2] = dict(ref[2], px=(ref[2]["px"] + 1) % 256)
    c = sd.compare_copy(K, ref, tokens, r["outs"][0], t_load0=0, dt=PARAMS.dt)
    fm = c["first_mismatch"]
    assert (c["status"], c["matched"], fm["tick"], fm["field"]) == ("mismatch", 2, 2, "px")
    assert fm["expected"] == ref[2]["px"] and fm["got"] == (ref[2]["px"] - 1) % 256 and fm["token"] == tokens[2]
