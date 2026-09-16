"""Stage H1: the 4-bit adder with its output register on flip-flops, placed and simulated.

The placement note found the flip-flop the easier motif to place on the fly's wiring (32 / 32
pairs on real driven inhibitory pairs, hosts with 42 % fewer external inputs than the latches')
but only as a toy; this bench places the real thing -- `build_adder_channel(storage="flipflop")`,
the ordered adder whose output register's ten rails are flip-flops read through their proxies
(664 neurons) -- beside the latch-register adder (614 neurons) it replaces, on real MCNS at
`k_max = 4`, and runs both images through the H1 harness: condition A (every designed edge,
missing ones as Profile 3; the netlist-equivalent) and B (carried edges only). Writes
docs/h1_ffadder.json and the flip-flop mapping docs/h1_ffadder_mapping.json;
`python -m drosophilos.bench.h1_ffadder` reproduces it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..connectome.embed_image import h1_conditions, run_h1_conditions, save_mapping
from ..connectome.embed_netlist import Policy, place_netlist
from ..connectome.mcns import MCNS, load_mcns
from ..lib.adder import build_adder_channel
from ..sim.model import Params

K_MAX = 4.0
RESTARTS = 4
TIME_LIMIT_S = 600.0


def run_one(storage: str, m: MCNS, n_cases: int, restarts: int, time_limit_s: float, conditions: str,
            mapping_path: Path | None = None) -> dict:
    policy = Policy(k_max=K_MAX)
    ch = build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100, storage=storage)
    net = ch.net
    t0 = time.time()
    pl = place_netlist(net, m, policy=policy, restarts=restarts, time_limit_s=time_limit_s, seed=0, verbose=False)
    out = {"storage": storage, "n_cases": n_cases, "restarts": restarts, "time_limit_s": time_limit_s, "seed": 0,
           "k_max": K_MAX, "neurons": net.n, "synapses": len(net.src), "placement": pl.summary(net),
           "motifs": pl.motifs, "ffpair_drivers": pl.ffpair_drivers, "proxy_readout": pl.proxy_readout,
           "exposure": pl.exposure, "search_seconds": round(time.time() - t0, 1)}
    print(f"[ffadder] {storage}: {net.n} neurons, carried {pl.carried} / {pl.edges}, unplaced {len(pl.unplaced)}, "
          f"motifs {pl.motifs}, {out['search_seconds']} s", flush=True)
    if mapping_path is not None:
        save_mapping(mapping_path, net, m, pl, {"storage": storage, "k_max": K_MAX, "restarts": restarts})
    conds = {k: v for k, v in h1_conditions(policy).items() if k in conditions}
    assert conds, f"no condition among {conditions!r} (A-F)"  # an empty dict would run all six
    out["conditions"] = run_h1_conditions(ch, m, pl, width=4, n_cases=n_cases, seed=0, policy=policy, conditions=conds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT_S)
    ap.add_argument("--conditions", default="AB")
    ap.add_argument("--storage", default="flipflop,latch")
    a = ap.parse_args()
    m = load_mcns()
    docs = Path(__file__).resolve().parents[2] / "docs"
    path = docs / "h1_ffadder.json"
    out = json.loads(path.read_text()) if path.exists() else {}  # a partial rerun keeps the other storage's entry
    for storage in a.storage.split(","):
        out[storage] = run_one(storage, m, a.cases, a.restarts, a.time_limit, a.conditions,
                               docs / "h1_ffadder_mapping.json" if storage == "flipflop" else None)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
