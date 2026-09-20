# Track A: a simulator step without host interaction

Implements `docs/perf_campaign.md` §4 Track A (A1–A4). The circuit is untouched; only the
simulator changes. `TorchSim` stays as the comparison backend.

## What the H200 profile said, and what this changes

Juno job 412204 (`docs/perf/juno-h200-profile.md`) timed one 0.1 ms step at 0.36–0.46 ms for
every workload from 3k to 880k neurons, flat in the neuron count: the step was paying for the
*number* of small kernel launches (≈15 in `deliver`, ≈9 in `integrate`) and for two host
synchronisations (`torch.nonzero`, whose output shape is data-dependent, and the `.cpu()` copy
of the spike indices), not for data volume. The host-side copy itself was only 6.5 %.

`TorchSim` step, per region (H200, one cell, 3,172 neurons):

| region | µs | what forces it |
|---|---|---|
| deliver | 176 | `nonzero` → `repeat_interleave`, `cumsum`, `arange`, gathers, `index_put_` — all shaped by how many neurons spiked |
| integrate | 88 | nine `torch.where`/arithmetic launches with fresh `full_like` allocations |
| threshold | 36 | compare + `nonzero` (sync) |
| observe | 24 | `.cpu()` of the index list (sync) |
| reset | 20 | two `where` + `full_like` |

`FastSim` (`drosophilos/sim/lif_fast.py`) removes the shape dependence and the syncs:

| region | FastSim | launches |
|---|---|---|
| integrate | in-place on preallocated buffers; `E_L + bias` and `w_unit · gain` precomputed | 12 elementwise |
| threshold | dense `(B, n)` bool compare, no index list | 2 |
| observe | `index_select` of the watched columns into a device ring `obs[K, B, W]` | 1 |
| deliver | per distinct delay: one CSR product `W_dᵀ @ spkᵀ` (CUDA) or one gather + `scatter_add_` (CPU/MPS), then a tensor-indexed `index_add_` into the ring | ~6 + 6 |
| reset | two `masked_fill_`, slot counter advanced on the device | 3 |
| host | one `.cpu()` of the observation block every K steps | 1 / K |

All shapes are static, every index that changes from step to step (`step % L`) lives in a
one-element device tensor, so a K-step block can be captured as a CUDA graph (A4).

## A1 — observation is separate from propagation (`sim/observe.py`)

`Observer(watch_ids, B, n_neurons=n, observe_every=K, full_trace=False, retain_steps=…,
capture=(node, ids))` is what the runners read spikes through. The pipeline runners build it
with `_pipeline_observation_ids`: READY of each input stream, completion and R-bit rail taps of
each output master, every fault latch, the producers' watchdog timeouts, plus any
`capture_spikes` ids (24 neurons of 10,744 for the render kernel; 13 of 960 for a 4-bit MOV
cell). Nothing about propagation changes; only what is copied to the host.

- `fired_at(step) -> (node_ids, neuron_ids)`; `active_in(step, window, node) -> set`;
  `capture_spikes()`; `trace` (full-trace mode only); `watched_trace()`.
- `protocol/token.py` is untouched: the observer keeps `_spk_step/_spk_node/_spk_neuron`
  chunk lists for the watched set in step order, so `decode_recent(observer, …)`,
  `recent_active(observer, …)` and `recent_spikes(observer, …)` work when handed the observer.
  The runners now call `decode_recent(observer, …)`.
- `RefSim`/`TorchSim` (and their subclasses in tests and campaigns) feed the same observer
  through `feed_legacy(sim, trim_chunks)` after each step, which also applies the old runner's
  memory bound (keep 4 windows of chunks once 8 accumulate). A caller-supplied `sim=` keeps
  working, including the campaigns' perturbed RefSims.
- Retention: `full_trace` keeps everything (reference/debug runs; `sim.trace`). Otherwise
  8 decode windows of the watched set, and the `capture` subset for the whole run.
- Overflow never drops spikes silently: writing past the device buffer, reading a step that
  is not observed yet or already trimmed, `trace` on a filtered observer, and a `capture`
  outside the watched set all raise.

## A2 — device-resident observation buffer

`FastSim` gathers `spk[:, watch_ids]` into `obs[k]` each step (static `(B, W)` gather). The
block is copied with one `.cpu()` when it is full (`observe_every=K`, default: the decode
window `2·loop_period_steps` = 94 steps for the default drive) or when a K-step block ends.

**K-step observation lag.** The runner sees READY at the end of the block it fired in, so a
token may be loaded up to K steps later than with per-step observation. Measured on the 4-bit
MOV kernel, 2 copies × 3 tokens (`tests/test_sim_fast.py::test_batched_runner_backends_agree_on_a_kernel`):

| observation | load steps (node 0) | decoded outputs | output steps |
|---|---|---|---|
| TorchSim, per step | 3005, 9484, 16009 | 0, 0, 0 | 18210, 29306, 40404 |
| FastSim, `observe_every=1` | same | same | same (trace bit-identical) |
| FastSim, `observe_every=10` | 3005, 9485, 16015 | same | same |
| FastSim, `observe_every=94` | 3005, 9491, 16071 | same | same |
| FastSim, `graph_steps=94` | — | same | same |

The protocol is host-paced (a token goes in after READY, whenever the host gets to it), so a
late load only shifts the host schedule; the decoded transactions are the same. The output
steps did not move here because the next transaction is gated by the cell's own period, not by
the load. `observe_every=1` reproduces TorchSim's schedule exactly and is what the trace-identity
tests use; the default K is the decode window.

## A3 — the fused step, and why it is bit-identical

Same expressions in the same floating-point order as `TorchSim`:

- `Vn = E_L + bias + (V − E_L − bias)·a + g·k` is evaluated as
  `((V − E_L) − bias)·a + (E_L + bias) + (g·k)` with the product `g·k` formed separately
  (`add_(g, alpha=k)` would fuse it into an FMA with one rounding instead of two — that would
  differ in the last bit).
- Delivery sums integer quanta. The CSR product is formed in the float dtype, but every partial
  sum is an integer below 2^53 (float64) or 2^24 (float32) — checked at construction from the
  largest per-target Σ|q| — so it is exact in any order, cast to int64 and added into the same
  int64 ring `TorchSim` uses; `g += (w_unit·gain)·due` then rounds once, as before. The
  edge-wise path accumulates in int64 directly.
- Stray input draws the same `torch.rand((B, n), generator)` once per step in the same place,
  so a seeded stray stream is identical (5 seeds tested, eager and block mode).
- Refractory hold, silencing, per-node `V_th`/`bias`/`gain`/`silenced`, per-node quanta,
  delayed events and external events follow schedule.md §5 order.

Verified (`tests/test_sim_fast.py`, CPU): spike traces and recorded V/g trajectories identical
to `RefSim` and `TorchSim` in float64 and to `TorchSim` in float32; on the `small_circuit`
kernels, on a two-delay topology (grouped path), with per-node quanta (scatter path), with
stray input, through snapshot/restore, and for a compiled MOV kernel through
`run_pipeline_batched` on both backends (same `(step, value)` outputs, same load steps, same
full trace). **No floating-point or RNG difference from `TorchSim` was found on this laptop
(CPU float64/float32, MPS float32, torch 2.14).** That is a measurement, not a proof: the CUDA
cuSPARSE product has not been run (see the cluster list).

Delivery paths (`FastSim(delivery=…)`, `--delivery`): `auto` = CSR product on CUDA, edge-wise
`scatter_add_` on CPU and MPS and whenever quanta are per-node. The CPU CSR kernel is
single-threaded and ~10 ns per synapse (190 µs for 19k edges — 10× the scatter); MPS has no
sparse CSR. A float32 CSR request whose fan-in could reach 2^24 is refused with the bound in the
message (`auto` falls back to scatter, which is exact).

Several distinct delays: one product per delay per step, logged at construction
(`sim.delays`). The doom pipelines have one delay (18 steps), so there is one product.

## A4 — K-step blocks and CUDA graphs

`FastSim(graph_steps=K)` (K ≤ L = 101) runs `run(n)` as blocks:

1. `observer.flush()` — nothing pending;
2. `_stage_events(s0, K)` — every external event of steps `s0 … s0+K−1` is `index_put_` into its
   ring slot **before** the block. Safe because slot `s % L` was last cleared at step `s − L`,
   before the block, and the block's own deliveries only add. Events land on their step: the
   ring is read at exactly step `s` (tested with `record=`: `g` jumps at the scheduled step, on
   the first and last step of a block, identical to `RefSim`);
3. the block: on CUDA a captured `torch.cuda.CUDAGraph` replay, otherwise (and for a partial
   tail block) `_block_eager(K)` — the very function that was captured;
4. `observer.commit_block(s0, K)` — one `.cpu()`; `step_index += K`.

The runner (`run_pipeline_batched`) runs its host schedule once per block; a load lands at
`step_index + 5`, i.e. inside the next block, staged before it runs. READY/completion are seen
at the block end (the A2 lag, K steps at most); nothing is polled every hundred steps and called
the same schedule.

**Why `torch.cuda.CUDAGraph` rather than `torch.compile(mode="reduce-overhead")`.** The step is
already what manual capture wants: fixed shapes, in-place writes into preallocated buffers, no
Python branching on tensor values. Capture then costs nothing at runtime and replays exactly the
kernels we wrote. `torch.compile` would have to trace `torch.sparse.mm` on a CSR tensor and the
tensor-indexed ring ops; Dynamo's sparse support is partial and each graph break re-introduces
a launch boundary — the very cost being removed — while adding a compile step whose result
depends on the installed torch. The graph is captured lazily at the first whole block, after two
warm-up blocks on a side stream; the warm-up advances device state, which is cloned before and
copied back after. A capture failure falls back to eager blocks and is recorded in
`sim.graph_fallback_reason` / `stats["graph_fallback_reason"]`.

Refused for capture (eager blocks with the reason logged): `stray_rate_hz > 0` (the generator
must be registered with the graph — TODO below), `record=` (per-step host copies), and while the
profiler is timing regions.

### Review findings (Kimi A–B, openai-sol C–F; fixed in `bcb1f10` and the commit after it)

- `set_observer` after a capture left the graph gathering into the old observer's buffer:
  the graph is now dropped and re-captured.
- A failed capture (any exception during warm-up or capture) fell back to eager blocks from
  the advanced state, with the staged events consumed: the state is now restored on every
  exit of `_capture_graph`.
- `Observer.active_in` clamped a window that reached into trimmed steps and read it silently;
  it now raises, and `protocol.token.recent_active` reads an observer through `active_in`, so
  `decode_recent(observer, …)` cannot decode from a partial window.
- Both runners count faults and timeouts during the 3,000-step settle and on the terminal
  step (the single-node runner had lost both in the refactor).
- **Neural pacing under K-step observation** (`render_doom --pacing neural`: no host barrier,
  tokens dealt after READY): the host sees READY at the end of the block it rose in, so a
  token is loaded up to K steps (9.4 ms at K = 94) later than with per-step observation, and
  at most one token per stream per block. A cell's cycle is ~1 s, so the throttle is below
  1 % of the token period; a run bounded by `max_ms` can nonetheless end with work
  outstanding that a per-step run finished. Not a value change; a schedule shift. The doom
  small-render twin (`--backend torch-fast`) measures it against the `torch` run.

### TODO(cluster) — what only the H200 can verify

1. **Capture works**: `stats["graph_active"] is True` on a `--backend torch-fast --graph-steps 94`
   run; if `graph_fallback_reason` names cuSPARSE or the allocator, try
   `--delivery scatter` (no cuSPARSE call inside the capture) and report which path captured.
2. **Traces identical on CUDA**: `run_pipeline_batched(..., full_trace=True, observe_every=1)`
   with `backend="torch"` and `"torch-fast"` on the cells-mov block, compare `sim.trace`; then
   `delivery="sparse"` vs `"scatter"`. cuSPARSE SpMM sums in float — exactness holds by the
   2^53 argument, but confirm it, and confirm float32 on a kernel whose fan-in bound is below 2^24.
3. **Per-step time**: the `--profile-steps 2000` regions for `torch-fast` (eager) should show
   `deliver` and `integrate` shrunk and no `observe` sync; with `--graph-steps 94` the profile
   shows one `graph_replay` region per block — compare ms/step from the unprofiled `wall_s`.
4. **CSR vs scatter on the H200** at 47k×8 and 110k×8 (perspective, pipelined): the CSR
   product reads all nnz once per step; the scatter path reads B·nnz and contends on atomics.
5. **Stray input under capture**: register the stray generator with the graph
   (`torch.cuda.CUDAGraph.register_generator_state` / the graph-safe RNG API of the installed
   torch) and re-run the seeded stray parity test on CUDA; until then stray runs use eager blocks.
6. **Memory**: `obs[K, B, W]` bool is K·B·W bytes (94 × 8 × 24 for a render kernel; with
   `capture_spikes=(0, range(n))` as in the primitive campaign, W = n: 94 × 8 × 110k = 83 MB).

## Laptop measurements (the cluster numbers are what count)

`tests/test_kernel.py` render kernel (10,744 neurons, 19,488 edges, one delay), image loaded,
1,500 settle steps, then ms per step over the next steps; Apple laptop, torch 2.14; CPU float64,
MPS float32. `TorchSim` copies every spike to the host each step (the only mode it has); the
`FastSim` rows watch the runner's 24 neurons. Single run each, so ±10 % is noise.

| configuration | B = 1, CPU float64 | B = 1, MPS float32 | B = 8, CPU float64 | B = 8, MPS float32 |
|---|---|---|---|---|
| `TorchSim` (index list to host every step) | 0.229 | 1.542 | 2.143 | 1.503 |
| `FastSim` eager, watched set, transfer every step | 0.228 | 0.800 | 2.166 | 1.068 |
| `FastSim` eager, watched set, transfer every 94 steps | 0.222 | 0.233 | 2.153 | 0.479 |
| `FastSim` 94-step blocks, eager loop (`graph_steps=94`) | 0.217 | **0.217** | 2.059 | **0.469** |
| `FastSim` eager, full trace transferred every step | 0.289 | 0.765 | 3.363 | 1.445 |

ms per step; 1,500 timed steps; the delivery path is the edge-wise scatter on both devices
(`auto`). MPS speed-up over `TorchSim`: 7.1× at B = 1, 3.2× at B = 8.

Reading: on the GPU-shaped device (MPS) the per-step host round trip is most of the step —
the "transfer every step" row shows the sync alone costing 0.6 ms — and removing it is worth
3–7×. On the CPU there is no launch cost and no sync to remove, the dense step does the same
arithmetic on every neuron, and the outcome is parity (the K-step loop is a few per cent
faster because Python bookkeeping runs once per block). The H200 sits at the MPS end of this
picture (its step was launch- and sync-bound at every size), and A4 removes the launches as
well, which no laptop device can show.

## H200 results (Juno 413148 clean A/B; 413020 first A/B, 413021 profile, 413022 eager/scatter; `docs/perf/`)

Wall ÷ neural on one H200 NVL, identical circuits and inputs, `--no-spike-count` (the runner
watches only its own neurons, as a render does), every row 0 wrong / 0 missing / 0 faults:

| block | neurons | `TorchSim` | `FastSim` graph + sparse | gain |
|---|---|---|---|---|
| one cell (ADD / AND / XOR / MOV) | 3,172 | 5.5–5.6× | 0.93–0.95× | 5.9× |
| fan-out | 10,888 | 5.8× | 1.01× | 5.7× |
| tick | 28,439 | 5.9× | 1.08× | 5.5× |
| perspective | 47,507 | 6.0× | 1.13× | 5.3× |
| perspective, pipelined multiplier | 110,024 | 6.0× | 1.21× | 4.9× |

Every kernel from 3k to 110k neurons now simulates within ~20 % of real time; the one-cell
blocks run faster than the neurons they simulate. Capture worked (`graph_active` true, the
cuSPARSE product inside the graph, no fallback). The first A/B (413020) had shown the gain
shrinking with size (2.2× at 110k): that was the primitive level's own all-neuron spike
capture, whose host unpack was 45–84 % of the measured step (profile 413021) — the graph
replay itself is 0.09–0.23 ms per step at every size. The eager/scatter twin (413022) shows
the graph is worth ~⅓ of the step and the scatter path collapses at 8 copies (5.0× on tick).

At frame scale (small-render level, `docs/perf/juno-h200-small*.md`): doom2 8 × 5 × 3 frames,
664k neurons, 1 copy — 9,126 s wall on `TorchSim`, 2,997 s on `FastSim` (3.0×), pixels
identical; at 8 copies (5.3 M neurons) 3,775 s vs 3,196 s (1.18×). Above ~1 M neurons the
step is bound by the arithmetic, which both simulators pay; the launch overhead `FastSim`
removes is then a small share. The next simulator lever is that arithmetic (the product's
memory traffic, float32), not the host.

## Wiring

- `run_pipeline_batched(..., backend="torch" | "torch-fast", graph_steps=0, observe_every=None,
  delivery="auto", full_trace=False)`; `run_pipeline(..., full_trace=False, observe_every=None)`
  reads any simulator through the observer. `stats` carries `simulator`, `observer`,
  `observe_every`, `graph_steps`, `graph_active`, `graph_fallback_reason`, `delivery`, `delays`.
- `bench/render_doom.py --backend torch-fast [--graph-steps K] [--observe-every K]
  [--delivery auto|sparse|scatter]`; the flags are rejected with other backends; the
  reproducibility record names the effective simulator class (`simulator_effective`) and its
  mode (`simulator_mode`).
- `bench/perf_campaign.py --backend torch-fast` for a torch-fast baseline, or
  `--variant-backend torch-fast` to add a one-key `backend-torch-fast` variant next to the
  `torch` baseline (a simulator-only comparison); `--graph-steps/--observe-every/--delivery`
  apply to the torch-fast configurations.

## Cluster commands

Same suite, both simulators, one job each (Juno, one H200; size `--time` from the 1 h 16 min the
`torch` primitive level took in job 412136):

```
slurm/submit.sh --time=3:00:00 -- drosophilos.bench.perf_campaign --levels primitive \
  --device cuda --backend torch-fast --graph-steps 94 --profile-steps 2000 --out data/perf/juno-h200-fast
slurm/submit.sh --time=3:00:00 -- drosophilos.bench.perf_campaign --levels primitive \
  --device cuda --backend torch --profile-steps 2000 --out data/perf/juno-h200-torch
```

or in one job as a one-key variant:

```
slurm/submit.sh --time=5:00:00 -- drosophilos.bench.perf_campaign --levels primitive \
  --device cuda --backend torch --variant-backend torch-fast --graph-steps 94 --out data/perf/juno-h200-ab
```

and the eager (no graph) and scatter twins for items 1, 3 and 4 above: drop `--graph-steps 94`,
add `--delivery scatter`.
