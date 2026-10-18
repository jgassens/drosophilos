"""Perturbation campaign on resident kernels: B copies of a kernel at mix B (weight noise,
threshold and bias drift, stray input), every copy fed the same tokens, every output compared
with the kernel reference. Counts ok / wrong / missing (a stalled copy) per token and the
exact 95 % upper limits, like the block campaigns of A2.

    python -m drosophilos.bench.kernel_campaign render --copies 100 --device cuda
    blocks: render (4 cells, ROM tables), fanout (a value read by two cells: the multi-pair guard),
            tick (the 13-cell state kernel, fresh input per tick), perspective (16-bit, MULP),
            mul8 (8-bit array MUL, configuration F)
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_kernel, kernel_outputs, loop_body
from ..lib.campaign import CampaignCheckpoint, atomic_write, make_perturbed_sim
from ..lib.kernel import TRUE_GUARD_VERSION, build_pipeline, run_pipeline_batched
from ..protocol.latch import DALE_BIASED_VERSION, DALE_FEEDBACK_VERSION
from ..sim.model import Params
from .a2_campaigns import MIXES
from .stall_diag import write_compact_dump


def _upper95(errors: int, n: int) -> float:
    from scipy.stats import beta
    return float(beta.ppf(0.95, errors + 1, n - errors)) if n else float("nan")


def block(name: str, params: Params, *, datapath: str = "generic", rate_robust: bool = False,
          zero_once: bool = False, robust_request_clear: bool = False,
          robust_register_reset: bool = False, verified_register_reset: bool = False,
          experimental_register_reset: bool = False, experimental_autapse: bool = False,
          ordered_xor_oneshot: bool = False, f_width16_experimental: bool = False,
          master_ignition: bool = False, operand_xor_oneshot: bool = False,
          latch_feedback: str = "autapse", dale_feedback_version: int | None = None,
          dale_biased_version: int | None = None):
    if name == "render":
        prog = compile_c(open("examples/render.c").read())
        ks = compile_kernel(prog, loop_body(prog, "main", "loop3"), "col", params={"heading": 3})
        tokens = list(range(8))
    elif name == "fanout":
        spec = [{"name": "c1", "op": "ADD", "a": "input", "b": ("const", "k1")},
                {"name": "c2", "op": "AND", "a": "c1", "b": ("const", "k7")},
                {"name": "c3", "op": "XOR", "a": "c1", "b": ("const", "k3")},
                {"name": "c4", "op": "ADD", "a": "c2", "b": "c3"}]
        from ..compiler.kernel import KernelSpec
        ks = KernelSpec(spec, {"k1": 1, "k7": 7, "k3": 3}, {}, "input", 8); ks.outputs = ["c4"]
        tokens = [0, 5, 9, 14, 14, 200, 77, 3]
        prog = None
    elif name == "mul8":
        spec = [{"name": "c1", "op": "MUL", "a": "input", "b": ("const", "k5")},
                {"name": "c2", "op": "ADD", "a": "c1", "b": ("const", "k3")},
                {"name": "c3", "op": "MUL", "a": "c2", "b": "input"}]
        from ..compiler.kernel import KernelSpec
        ks = KernelSpec(spec, {"k5": 5, "k3": 3}, {}, "input", 8); ks.outputs = ["c3"]
        tokens = [0, 1, 7, 13, 51, 128, 200, 255]
        prog = None
    elif name == "tick":
        prog = compile_c(open("examples/tick2.c").read())
        ks = compile_kernel(prog, loop_body(prog), "i")
        tokens = [5, 5, 250, 3, 0, 40, 40, 40]
    elif name == "perspective":
        prog = compile_c(open("examples/render2.c").read())
        ks = compile_kernel(prog, loop_body(prog), "col", params={"heading": 3}, mul="pipelined")
        tokens = list(range(8))
    else:
        raise SystemExit(name)
    width = prog.width if prog is not None else ks.width
    pl = build_pipeline(params, width, ks.cells, consts=ks.consts, mems=ks.mems,
                        outputs=ks.outputs, datapath=datapath, rate_robust=rate_robust,
                        zero_once=zero_once, robust_request_clear=robust_request_clear,
                        robust_register_reset=robust_register_reset,
                        verified_register_reset=verified_register_reset,
                        experimental_register_reset=experimental_register_reset,
                        experimental_autapse=experimental_autapse, ordered_xor_oneshot=ordered_xor_oneshot,
                        f_width16_experimental=f_width16_experimental, master_ignition=master_ignition,
                        operand_xor_oneshot=operand_xor_oneshot, latch_feedback=latch_feedback,
                        **({} if dale_feedback_version is None else
                           {"dale_feedback_version": dale_feedback_version}),
                        **({} if dale_biased_version is None else
                           {"dale_biased_version": dale_biased_version}))
    ref = kernel_outputs(ks, tokens)
    return ks, pl, tokens, ref


def rebuild_record(record: dict):
    """Rebuild a current campaign, enforcing every effective option/version.

    The historical tick diagnostic remains responsible for legacy records.
    Wider F kernels first became buildable with complete option records; a
    missing version must not silently select a different circuit on replay.
    """
    flags = ("rate_robust", "zero_once", "robust_request_clear", "robust_register_reset",
             "verified_register_reset", "experimental_register_reset", "experimental_autapse",
             "ordered_xor_oneshot", "f_width16_experimental", "master_ignition", "operand_xor_oneshot")
    from ..protocol.latch import MASTER_IGNITION_VERSION, validate_latch_feedback
    from ..lib.gates import validate_operand_xor_record
    validate_operand_xor_record(record)
    if "master_ignition" in record and type(record["master_ignition"]) is not bool:
        raise ValueError("unsupported campaign build option master_ignition: expected a bool")
    if record.get("master_ignition", False) and (
            type(record.get("master_ignition_version")) is not int or
            record["master_ignition_version"] != MASTER_IGNITION_VERSION):
        raise ValueError("unsupported master-ignition circuit version")
    if "f_width16_experimental" in record and type(record["f_width16_experimental"]) is not bool:
        raise ValueError("unsupported campaign build option f_width16_experimental: expected a bool")
    latch_feedback = record.get("latch_feedback", "autapse")
    dale_feedback_version = record.get("dale_feedback_version", DALE_FEEDBACK_VERSION)
    dale_biased_version = record.get("dale_biased_version")
    try:
        validate_latch_feedback(latch_feedback, record.get("experimental_autapse", False),
                                dale_feedback_version, dale_biased_version)
    except ValueError as exc:
        raise ValueError(f"unsupported campaign latch_feedback: {exc}") from exc
    built = block(record["block"], Params(), datapath=record.get("datapath", "generic"),
                  latch_feedback=latch_feedback, dale_feedback_version=dale_feedback_version,
                  dale_biased_version=dale_biased_version,
                  **{key: record.get(key, False) for key in flags})
    pl = built[1]
    for key, value in pl.build_options.items():
        recorded = record.get(key)
        if key not in record or type(recorded) is not type(value) or recorded != value:
            raise ValueError(f"unsupported campaign build option {key}: recorded {recorded!r}, current {value!r}")
    for key, value in (("neurons", pl.net.n), ("edges", pl.net.nnz)):
        if key in record and record[key] != value:
            raise ValueError(f"campaign {key} differs from the rebuilt kernel")
    return built


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--copies", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the block's token sequence; recompute the stateful oracle for the full sequence")
    ap.add_argument("--mix", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ms", type=float, default=30000, help="ceiling on the neural time (the run stops when every copy has delivered)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="torch", choices=("torch", "torch-fast"),
                    help="batched simulator (torch is the comparison/default backend)")
    ap.add_argument("--observe-every", type=int, default=None,
                    help="FastSim observation transfer interval in simulation steps")
    ap.add_argument("--datapath", choices=("generic", "specialized"), default="generic")
    ap.add_argument("--rate-robust", action="store_true",
                    help="use refractory-limited rate readers (opt-in; changes the netlist)")
    ap.add_argument("--ordered-xor-oneshot", action="store_true",
                    help=("consume sum-XOR carry once per stage reset; requires --rate-robust "
                          "and --experimental-autapse (default timing, widths 1..8; "
                          "wider kernel qualification: docs/stable_latch.md)"))
    ap.add_argument("--operand-xor-oneshot", action="store_true",
                    help="UNQUALIFIED v3: share operand/sum transaction pulses with carry branches and strengthen operand/carry/sum clears; requires complete configuration F")
    ap.add_argument("--experimental-autapse", action="store_true",
                    help="EXPERIMENTAL: requires --rate-robust; unqualified on the 44-step rate floor and reload margin")
    ap.add_argument("--latch-feedback", choices=("autapse", "dale_private", "dale_biased"), default="autapse",
                    help=("dale_private and dale_biased are EXPERIMENTAL and NOT autapse replacements: "
                          "dale_private fails the three-sigma corners and reload; dale_biased passes the "
                          "512-corner and reload bar (0/0/0, 118/128) but FAILS the separate 44-step "
                          "rate floor (corner minimum interval 33 steps)"))
    ap.add_argument("--master-ignition", action="store_true",
                    help="EXPERIMENTAL: stagger selected master partner ignition from qualified COPY; requires rate readers and experimental autapses")
    ap.add_argument("--f-width16-experimental", action="store_true",
                    help="EXPERIMENTAL: allow width 16 for the kernel campaign; requires all three configuration F flags")
    ap.add_argument("--zero-once", action="store_true",
                    help="ignite Z0 once per word through a shared stage-reset OR/relay")
    ap.add_argument("--robust-register-reset", action="store_true",
                    help="withdrawn: rejected until full mix-B reset/reload qualification passes")
    ap.add_argument("--verified-register-reset", action="store_true",
                    help="unqualified closed-loop prototype: rejected; see docs/stage_d_completion_stall.md")
    ap.add_argument("--experimental-register-reset", action="store_true",
                    help="EXPERIMENTAL, known full-domain failures: 1.75-loop reset / eight delayed READY links")
    ap.add_argument("--robust-request-clear", action="store_true",
                    help="use the opt-in compact clear on DONE's false request rails")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None, help="atomically save/resume a trusted local checkpoint")
    ap.add_argument("--checkpoint-every-s", type=float, default=1800,
                    help="wall seconds between checkpoints (default: 1800); signals also save and exit 75")
    ap.add_argument("--fp32", action="store_true", help="single precision (Apple GPU always; GeForce cards are slow at float64)")
    ap.add_argument("--dump-node", type=int, default=None, help="copy whose matching neural spikes to retain")
    ap.add_argument("--dump-roles", default=None, help="regular expression matched against neuron roles")
    ap.add_argument("--dump-out", default=None, help="atomically write compact step/neuron and uniq/role_of arrays")
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.repeats < 1:
        ap.error("--repeats must be positive")
    if not math.isfinite(a.checkpoint_every_s) or a.checkpoint_every_s <= 0:
        ap.error("--checkpoint-every-s must be finite and positive")
    if a.checkpoint and a.out and Path(a.checkpoint).resolve() == Path(a.out).resolve():
        ap.error("--checkpoint and --out must be different paths")
    if a.checkpoint:
        with CampaignCheckpoint(a.checkpoint, a.checkpoint_every_s) as checkpoint:
            try:
                return _main(a, ap, checkpoint)
            except ValueError as exc:
                ap.error(str(exc))
    return _main(a, ap)


def _fingerprint(a, pl, params):
    import torch
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    source = hashlib.sha256()
    for directory in (root / "drosophilos", root / "examples"):
        for path in sorted(directory.rglob("*")):
            if path.suffix in (".py", ".c"):
                source.update(str(path.relative_to(root)).encode())
                source.update(path.read_bytes())
    topology = pl.net.topology()
    net_hash = hashlib.sha256()
    for field in ("src", "dst", "delay", "quanta", "bias"):
        value = getattr(topology, field)
        if value is not None:
            net_hash.update(np.asarray(value).tobytes())
    config = {k: v for k, v in vars(a).items()
              if k not in ("checkpoint", "checkpoint_every_s", "out", "dump_out")}
    return dict(commit=commit, source=source.hexdigest(), config=config,
                build_options=pl.build_options, params=asdict(params),
                neurons=pl.net.n, edges=pl.net.nnz, topology=net_hash.hexdigest(),
                torch=str(torch.__version__), numpy=np.__version__,
                device_name=torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else a.device)


def _main(a, ap, checkpoint=None):
    P = Params()
    ks, pl, tokens, ref = block(a.block, P, datapath=a.datapath, rate_robust=a.rate_robust,
                              zero_once=a.zero_once, robust_request_clear=a.robust_request_clear,
                              robust_register_reset=a.robust_register_reset,
                              verified_register_reset=a.verified_register_reset,
                              experimental_register_reset=a.experimental_register_reset,
                              experimental_autapse=a.experimental_autapse, ordered_xor_oneshot=a.ordered_xor_oneshot,
                              f_width16_experimental=a.f_width16_experimental, master_ignition=a.master_ignition,
                              operand_xor_oneshot=a.operand_xor_oneshot, latch_feedback=a.latch_feedback)
    if a.repeats != 1:
        tokens = tokens * a.repeats
        ref = kernel_outputs(ks, tokens)
    B = a.copies
    dump_given = (a.dump_node is not None, a.dump_roles is not None, a.dump_out is not None)
    if any(dump_given) and not all(dump_given):
        ap.error("--dump-node, --dump-roles and --dump-out must be supplied together")
    if a.dump_node is not None and not 0 <= a.dump_node < B:
        ap.error(f"--dump-node must be in [0, {B})")
    capture = None
    if a.dump_node is not None:
        try:
            role_re = re.compile(a.dump_roles)
        except re.error as exc:
            ap.error(f"invalid --dump-roles expression: {exc}")
        capture = (a.dump_node, [i for i, role in enumerate(pl.net.roles) if role_re.search(role)])
    pert = MIXES[a.mix]
    rng = np.random.default_rng(a.seed)
    n_steps = int(a.max_ms / P.dt) + 5000
    t0 = time.time()
    import torch
    dtype = torch.float32 if a.fp32 else torch.float64
    if checkpoint is not None:
        checkpoint.configure(_fingerprint(a, pl, P), vars(a), rng)
        if checkpoint.saved and checkpoint.saved.get("finished"):
            rec = checkpoint.saved["record"]
            if a.out:
                encoded = json.dumps(rec, indent=1).encode()
                if not Path(a.out).exists() or Path(a.out).read_bytes() != encoded:
                    atomic_write(a.out, lambda stream: stream.write(encoded))
            return
        if a.out and Path(a.out).exists():
            try:
                rec = json.loads(Path(a.out).read_text())
            except (ValueError, OSError):
                rec = {}
            expected = dict(block=a.block, copies=B, seed=a.seed, repeats=a.repeats,
                            mix=a.mix, backend=a.backend, neurons=pl.net.n, edges=pl.net.nnz,
                            tokens_list=tokens, **pl.build_options)
            if (all(rec.get(k) == v for k, v in expected.items()) and
                    all(k in rec for k in ("outputs", "per_node", "neural_s", "missing", "ok", "wrong")) and
                    len(rec["per_node"]) == B and len(rec["outputs"]) == B and
                    rec.get("outputs_expected") == len(tokens) * len(ks.outputs) * B and
                    rec["ok"] + rec["wrong"] + rec["missing"] == rec["outputs_expected"]):
                checkpoint.finish(rec)
                return
    sim = make_perturbed_sim(pl.net.topology(), P, B, pert, rng, n_steps, device=a.device, dtype=dtype,
                             backend=a.backend)
    if checkpoint is not None:
        checkpoint.restore_rng(sim)
        sim.campaign_checkpoint = checkpoint
    outs, sim, st = run_pipeline_batched(pl, P, [list(tokens) for _ in range(B)], max_ms=a.max_ms, device=a.device,
                                         expect_outputs=[len(tokens) * len(pl.outputs)] * B, sim=sim, dtype=dtype,
                                         capture_spikes=capture, backend=a.backend, observe_every=a.observe_every)
    captured = st.pop("captured_spikes", None)
    dump_errors = {}
    dump_paths = {}
    if captured is not None:
        steps, neurons = captured
        try:
            write_compact_dump(a.dump_out, steps, neurons, pl.net.roles, capture_ids=capture[1])
            dump_paths[str(a.dump_node)] = str(a.dump_out)
        except Exception as exc:
            dump_errors[str(a.dump_node)] = {"path": str(a.dump_out), "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
    ok = wrong = missing = extra = 0
    per_node = []
    for b in range(B):
        node_ok = node_wrong = node_missing = 0
        for j, o in enumerate(ks.outputs):
            got = [v for _, v in outs[b][o]]
            want = [r[j] for r in ref]
            extra += max(0, len(got) - len(want))
            for k, w in enumerate(want):
                if k < len(got) and got[k] == w:
                    node_ok += 1
                elif k < len(got):
                    node_wrong += 1
                else:
                    node_missing += 1
        ok += node_ok; wrong += node_wrong; missing += node_missing
        per_node.append((node_ok, node_wrong, node_missing))
    n = ok + wrong + missing
    rec = {"block": a.block, "datapath": pl.datapath, "backend": a.backend,
           "observe_every": a.observe_every,
           "simulator": st.get("simulator"), "mix": a.mix, "perturbation": str(pert), "copies": B,
           "seed": a.seed, "repeats": a.repeats, "tokens": len(tokens), "neurons": pl.net.n, "edges": pl.net.nnz,
           "outputs_expected": n, "ok": ok, "wrong": wrong, "missing": missing, "extra": extra,
           "faults": st["faults"], "timeouts": st["timeouts"],
           "kill_train": [pl.build_options["kernel_kill_pulses"], pl.build_options["kill_strength"]],
           "true_guard_version": TRUE_GUARD_VERSION,
           "refusals": st.get("refusals", 0), "retries": st.get("retries", 0),
           "per_node_refusals": [len(r) for r in st.get("refused", [])], "blocked_nodes": st.get("blocked_nodes", []),
           "bad_outputs": st["bad_outputs"], "wrong_upper_95": _upper95(wrong, n), "non_ok_upper_95": _upper95(wrong + missing, n),
           "nodes_with_errors": sum(1 for x in per_node if x[1] or x[2]), "neural_s": st["neural_ms"] / 1000, "wall_s": round(time.time() - t0),
           "per_node": per_node, "reference": ref, "tokens_list": list(tokens), "output_cells": list(ks.outputs),
           "outputs": [{o: [[int(s_), (None if v is None else int(v))] for s_, v in outs[b][o]] for o in ks.outputs} for b in range(B)],
           "first_output_ms": st.get("first_output_ms"), "per_token_ms": st.get("per_token_ms")}
    rec.update(pl.build_options)
    rec["latch_feedback"] = a.latch_feedback
    if a.latch_feedback == "dale_biased":
        rec["dale_biased_version"] = pl.build_options.get("dale_biased_version", DALE_BIASED_VERSION)
    else:
        rec["dale_feedback_version"] = pl.build_options.get("dale_feedback_version", DALE_FEEDBACK_VERSION)
    if capture is not None:
        rec.update(spike_dumps=dump_paths, spike_dump_errors=dump_errors)
    if a.out:
        if checkpoint is not None:
            atomic_write(a.out, lambda stream: stream.write(json.dumps(rec, indent=1).encode()))
        else:
            with open(a.out, "w") as stream:
                json.dump(rec, stream, indent=1)
    if checkpoint is not None:
        checkpoint.finish(rec)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
