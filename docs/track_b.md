# Track B: fixed-operation cell datapaths

This implements B1 and the safe part of B2 from `docs/perf_campaign.md` §4. The default is
still `build_pipeline(..., datapath="generic")`; existing callers therefore build the same
circuit and image. `datapath="specialized"` selects a single-unit datapath for AND, OR, XOR,
MOV, and both ROM and RAM LOAD cells. ADD, SUB, MUL, SEL, shifts, multiplier rows, and STORE
remain generic or keep their existing dedicated implementations. The requested option is
recorded on `Pipeline`; each `Cell.datapath` records the implementation it actually received.

## Circuit choice and timing

The specialized logic path retains B delayed by 6 hops, `bx = SUB xor B^d`, and A delayed by
17 hops. AND/OR/XOR still use the corresponding ordered gate with `bx` EARLY and A^d LATE.
The documented margin consequently remains at least 24 ms. MOV also keeps the
ordered `bx` gate rather than taking the optional one-hop-shorter B^d path, so this patch
does not introduce a second timing experiment.

There is no adder, overflow network, unused logic unit, six-way unit-select image, or mux in
a specialized cell. The selected ordered unit's output rails are already latches registered
in the cell's `Gates` collection. They join the same stage-reset domain as the removed mux
latches and hold until the stage clears, so they connect directly to `wire_alu`; no extra
relay or latch per rail is needed. C and V are fresh dual-rail zero tokens on every
transaction: ACT^d drives their rail-0 veto relays. They are not image-lit levels. Z is still
computed by `add_zero_flag` in the consuming stage.

LOAD uses this direct PASSB path after either read-port implementation. STORE deliberately
does not. The existing STORE comment records the write landing about 130 ms after ACT^d and
the cell done pulse about 500 ms after ACT^d. That separation currently comes from the long
generic ALU path, not an acknowledgement edge from the RAM write port. Specializing it would
turn a measured margin into a shorter unguarded interval, so a pipeline requested as
specialized reports its STORE cell as generic and retains the unit rails and full ALU.

`load_pipeline_image` uses each cell's actual datapath. It lights unit-select rails only for
generic ALU/LOAD/STORE cells and never looks up the absent `cell.u{k}r{r}.u` roles of a
specialized cell.

## Laptop measurements

These are clean `RefSim`, 8-bit, one-cell pipelines built on this laptop. Each run used eight
tokens. Spike counts include image loading and its settling interval, as the primitive
campaign's whole-run capture does, divided by eight. LOAD-ROM and LOAD-RAM use one initialized
word to isolate the read path; STORE uses one RAM word. Times are neural milliseconds.

| op | requested datapath | actual cell datapath | neurons | edges | spikes/token | first output (ms) | interval (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| ADD (control) | generic | generic | 3,172 | 5,769 | 253,646.4 | 1,680.5 | 1,265.7 |
| ADD (control) | specialized | generic | 3,172 | 5,769 | 253,646.4 | 1,680.5 | 1,265.7 |
| AND | generic | generic | 3,172 | 5,769 | 238,687.5 | 1,777.8 | 1,168.7 |
| AND | specialized | specialized | 2,166 | 3,847 | 157,893.4 | 1,769.7 | 1,160.5 |
| OR | generic | generic | 3,172 | 5,769 | 228,087.4 | 1,645.8 | 1,130.8 |
| OR | specialized | specialized | 2,166 | 3,847 | 151,511.2 | 1,637.8 | 1,122.5 |
| XOR | generic | generic | 3,172 | 5,769 | 227,775.5 | 1,645.8 | 1,131.3 |
| XOR | specialized | specialized | 2,198 | 3,911 | 153,294.2 | 1,637.8 | 1,122.8 |
| MOV | generic | generic | 3,172 | 5,769 | 218,204.4 | 1,719.4 | 1,073.3 |
| MOV | specialized | specialized | 2,070 | 3,655 | 139,834.6 | 1,710.9 | 1,064.5 |
| LOAD-ROM | generic | generic | 3,077 | 5,617 | 207,342.2 | 1,587.8 | 1,072.6 |
| LOAD-ROM | specialized | specialized | 1,975 | 3,503 | 129,951.4 | 1,579.4 | 1,064.1 |
| LOAD-RAM | generic | generic | 3,261 | 5,998 | 226,178.8 | 1,587.8 | 1,072.6 |
| LOAD-RAM | specialized | specialized | 2,159 | 3,884 | 148,651.1 | 1,579.4 | 1,064.1 |
| STORE | generic | generic | 3,433 | 6,289 | 240,206.1 | 1,719.4 | 1,073.3 |
| STORE | specialized | generic | 3,433 | 6,289 | 240,206.1 | 1,719.4 | 1,073.3 |

Neural latency did change, but only slightly: the specialized cells produced their first
output 8.0–8.5 ms earlier and reduced their mean interval by 8.2–8.9 ms. The staged commit,
completion, reset, and request protocol still dominates latency. The material result is the
component and spike reduction: 974–1,102 neurons and 1,858–2,114 edges per supported 8-bit
cell, with roughly 31–38% fewer whole-run spikes/token in these short runs. STORE is exactly
unchanged.

## H200 result (Juno 413149; `docs/perf/juno-h200-trackB-prim.md`)

`FastSim` (graph + sparse), one H200, `--no-spike-count`, generic baseline against the
one-key `datapath=specialized` variant, 16 tokens, every row 0 wrong / 0 missing / 0 faults:

| cell | neurons generic → specialized | first output (s) | interval (s) | wall ÷ neural |
|---|---|---|---|---|
| ADD (control, stays generic) | 3,172 → 3,172 | 1.681 → 1.681 | 1.191 → 1.191 | 1.00× → 0.94× |
| AND | 3,172 → 2,166 (−32 %) | 1.778 → 1.770 | 1.157 → 1.149 | 0.94× → 0.93× |
| XOR | 3,172 → 2,198 (−31 %) | 1.646 → 1.638 | 1.131 → 1.123 | 0.94× → 0.95× |
| MOV | 3,172 → 2,070 (−35 %) | 1.719 → 1.711 | 1.073 → 1.064 | 0.94× → 0.93× |
| fan-out kernel | 10,888 → 8,908 (−18 %) | 3.858 → 3.850 | 1.272 → 1.272 | 1.02× → 1.01× |
| tick kernel | 28,439 → 24,479 (−14 %) | 6.449 → 6.433 | 5.947 → 5.930 | 1.08× → 1.08× |

The result the spec anticipated, stated plainly: **a third fewer neurons per supported cell,
no change in speed.** Neural latency moves by 8–16 ms in ~1,150–5,950 (the handshake, not
the ALU, sets it), and wall time does not move because `FastSim`'s step at these sizes does
not scale with the neuron count. The saving is a capacity result — fewer components per
cell for Stage H's placement budget and for runs large enough that arithmetic binds — not a
frame-time result on its own. The 100-copy noisy campaign (`kernel_campaign --datapath
specialized`) is still to run before the option can be a default.

## Cluster handoff

Run the matched generic baseline and one-key specialized circuit variant with:

```sh
uv run python -m drosophilos.bench.perf_campaign \
  --levels primitive,small,historical --historical \
  --backend torch --device cuda --dtype float64 --copies 1,8 \
  --variant-datapath specialized --out data/perf/track-b
```

Run the noisy 100-copy composition blocks with the specialized option (STORE remains generic
inside them):

```sh
uv run python -m drosophilos.bench.kernel_campaign render --datapath specialized --copies 100 --device cuda
uv run python -m drosophilos.bench.kernel_campaign fanout --datapath specialized --copies 100 --device cuda
uv run python -m drosophilos.bench.kernel_campaign tick --datapath specialized --copies 100 --device cuda
uv run python -m drosophilos.bench.kernel_campaign perspective --datapath specialized --copies 100 --device cuda
```

The wall-time and noisy-campaign results are intentionally left to those cluster runs. This
patch does not combine width reduction, retained intermediates, predication, larger blocks,
or any request/commit protocol change with the fixed-unit comparison.
