"""A netlist must not depend on PYTHONHASHSEED.

The kernel compiler used to walk sets of variable names (the loop-carried state, and the variables
an if/else merges with SEL cells), so the cell order -- and the doom4/doom2 netlists built from it --
changed from one Python process to the next.  Each build runs in its own subprocess, because the
hash seed is fixed when the interpreter starts.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Four state variables that an if/else rewrites in different orders on its two arms: under the
# old set iteration, four hash seeds gave four different cell lists.
SMALL_KERNEL = r'''
import hashlib
from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.kernel import compile_kernel, loop_body
SRC = """static u8 a, b, c, d, i;
int main(void) { a = 1; b = 2; c = 3; d = 4; i = 4;
  while (i != 0) { if (a == 0) { a = b + 1; b = c + 2; c = d + 3; d = a + 4; } else { d = d + 5; c = c + 6; b = b + 7; a = a + 8; }
    out_pixel(a + b + c + d); i = i - 1; } return 0; }"""
prog = compile_c(SRC)
ks = compile_kernel(prog, loop_body(prog), "i")
print(hashlib.sha256(repr((ks.cells, ks.outputs, ks.state_cells, ks.consts)).encode()).hexdigest())
'''

# The full doom2 netlist, fingerprinted by role (independent of neuron insertion indices) and by
# repro.circuit_hash (dependent on them: what a benchmark record stores).
DOOM2_NETLIST = r'''
import hashlib, json
from drosophilos.bench.repro import circuit_hash
from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.kernel import compile_program
from drosophilos.lib.kernel import build_pipeline, load_pipeline_image
from drosophilos.sim.model import Params

class Rec:
    def __init__(self):
        self.neurons = []
    def add_events(self, node, steps, neurons, quanta):
        self.neurons.extend(int(i) for i in neurons)

prog = compile_c(open("examples/doom2.c").read())
ks = compile_program(prog, params=None, pacing="host", mul="pipelined")
pl = build_pipeline(Params(), prog.width, ks.cells, consts=ks.consts, mems=ks.mems,
                    outputs=ks.outputs, streams=ks.streams, phases=ks.phases)
rec = Rec()
load_pipeline_image(rec, pl)
net = pl.net
edges = sorted((net.roles[s], net.roles[d], int(q), int(dl))
               for s, d, q, dl in zip(net.src, net.dst, net.quanta, net.delay))
biases = sorted((r, float(b)) for r, b in zip(net.roles, net.bias))
image = sorted(net.roles[i] for i in rec.neurons)
roles = hashlib.sha256(json.dumps([net.n, net.nnz, edges, biases, image]).encode()).hexdigest()
print(net.n, net.nnz, roles, circuit_hash(pl))
'''


def _run(code: str, seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True,
                         capture_output=True, text=True)
    return out.stdout.strip()


def _under_seeds(code: str, seeds) -> dict:
    with ThreadPoolExecutor(len(seeds)) as pool:
        return dict(zip(seeds, pool.map(lambda s: _run(code, s), seeds)))


def test_kernel_compiler_cell_order_ignores_the_hash_seed():
    got = _under_seeds(SMALL_KERNEL, [0, 1, 2, 3])
    assert len(set(got.values())) == 1, got


@pytest.mark.slow
def test_doom2_netlist_ignores_the_hash_seed():
    got = _under_seeds(DOOM2_NETLIST, [0, 1])
    assert len(set(got.values())) == 1, got
