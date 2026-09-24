"""Perturbation campaign on resident kernels: B copies of a kernel at mix B (weight noise,
threshold and bias drift, stray input), every copy fed the same tokens, every output compared
with the kernel reference. Counts ok / wrong / missing (a stalled copy) per token and the
exact 95 % upper limits, like the block campaigns of A2.

    python -m drosophilos.bench.kernel_campaign render --copies 100 --device cuda
    blocks: render (4 cells, ROM tables), fanout (a value read by two cells: the multi-pair guard),
            tick (the 13-cell state kernel, fresh input per tick), perspective (16-bit, MULP)
"""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from ..lib import control

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_kernel, kernel_outputs, loop_body
from ..lib.campaign import make_perturbed_sim
from ..lib.kernel import TRUE_GUARD_VERSION, build_pipeline, run_pipeline_batched
from ..sim.model import Params
from .a2_campaigns import MIXES


def _upper95(errors: int, n: int) -> float:
    from scipy.stats import beta
    return float(beta.ppf(0.95, errors + 1, n - errors)) if n else float("nan")


def block(name: str, params: Params, *, datapath: str = "generic"):
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
                        outputs=ks.outputs, datapath=datapath)
    ref = kernel_outputs(ks, tokens)
    return ks, pl, tokens, ref


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--copies", type=int, default=100)
    ap.add_argument("--mix", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ms", type=float, default=30000, help="ceiling on the neural time (the run stops when every copy has delivered)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="torch", choices=("torch", "torch-fast"),
                    help="batched simulator (torch is the comparison/default backend)")
    ap.add_argument("--datapath", choices=("generic", "specialized"), default="generic")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fp32", action="store_true", help="single precision (Apple GPU always; GeForce cards are slow at float64)")
    ap.add_argument("--dump-node", type=int, default=None, help="copy whose matching neural spikes to retain")
    ap.add_argument("--dump-roles", default=None, help="regular expression matched against neuron roles")
    ap.add_argument("--dump-out", default=None, help="write the selected step/neuron/role arrays to this .npz")
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    P = Params()
    ks, pl, tokens, ref = block(a.block, P, datapath=a.datapath)
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
    sim = make_perturbed_sim(pl.net.topology(), P, B, pert, rng, n_steps, device=a.device, dtype=dtype,
                             backend=a.backend)
    outs, sim, st = run_pipeline_batched(pl, P, [list(tokens) for _ in range(B)], max_ms=a.max_ms, device=a.device,
                                         expect_outputs=[len(tokens) * len(pl.outputs)] * B, sim=sim, dtype=dtype,
                                         capture_spikes=capture, backend=a.backend)
    captured = st.pop("captured_spikes", None)
    if captured is not None:
        steps, neurons = captured
        dump_path = Path(a.dump_out)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        roles = np.asarray([pl.net.roles[int(i)] for i in neurons], dtype=str)
        np.savez(dump_path, step=steps, neuron=neurons, role=roles)
    ok = wrong = missing = 0
    per_node = []
    for b in range(B):
        node_ok = node_wrong = node_missing = 0
        for j, o in enumerate(ks.outputs):
            got = [v for _, v in outs[b][o]]
            want = [r[j] for r in ref]
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
           "simulator": st.get("simulator"), "mix": a.mix, "perturbation": str(pert), "copies": B, "tokens": len(tokens), "neurons": pl.net.n,
           "outputs_expected": n, "ok": ok, "wrong": wrong, "missing": missing, "faults": st["faults"], "timeouts": st["timeouts"],
           "kill_train": [control.KILL_PULSES, control.KILL_STRENGTH], "powerup_veto": True, "commit_reignite": True, "true_guards": True, "retry_clear": False, "start_relight_hops": 0, "request_clear_pulses": 3,
           "true_guard_version": TRUE_GUARD_VERSION,
           "refusals": st.get("refusals", 0), "retries": st.get("retries", 0),
           "per_node_refusals": [len(r) for r in st.get("refused", [])], "blocked_nodes": st.get("blocked_nodes", []),
           "bad_outputs": st["bad_outputs"], "wrong_upper_95": _upper95(wrong, n), "non_ok_upper_95": _upper95(wrong + missing, n),
           "nodes_with_errors": sum(1 for x in per_node if x[1] or x[2]), "neural_s": st["neural_ms"] / 1000, "wall_s": round(time.time() - t0),
           "per_node": per_node, "reference": ref, "tokens_list": list(tokens), "output_cells": list(ks.outputs),
           "outputs": [{o: [[int(s_), (None if v is None else int(v))] for s_, v in outs[b][o]] for o in ks.outputs} for b in range(B)],
           "first_output_ms": st.get("first_output_ms"), "per_token_ms": st.get("per_token_ms")}
    print(json.dumps(rec, indent=1), flush=True)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
