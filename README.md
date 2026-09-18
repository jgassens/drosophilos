# DrosophilOS

An asynchronous neural-computing project targeting the *Drosophila* male central
nervous system connectome (MCNS v1.0, 166,700 neurons). The successful rendering
workloads currently use freely synthesized fly-model spiking networks (Profile 3),
not the anatomical wiring (Profile 2). The target program is a Doom-like engine whose game update
**and** renderer both execute in the neural substrate; the host only integrates neuron
equations, transports spike events, transduces input, loads the program image, and
displays already-computed pixels.

Design document: `docs/spec.md`. Plan and milestone order: `docs/plan.md`.
Results record (what was done, what worked, what did not): `RESULTS.md`.

## Current status and proposed priorities (2026-09-17)

The [campaign report](docs/campaign_report_2026-09-17.md) records correct neural
world-update and rendering demonstrations, including textures and a chasing sprite.
The present labels are **Profile 3 / isolated / hybrid orchestration / external
compilation**. This is a feasibility result, not an interactive game or full-workload
execution on fly wiring.

The [performance-first roadmap addendum](docs/performance_roadmap_2026-09-17.md)
proposes measurement, semantics-preserving simulator acceleration, reduction of exact
neural work, and then measured E2 population acceleration. The original milestone
requirements remain in `docs/plan.md`; no incomplete stage is declared complete.

Saved two-frame H200 runs average **3.94 h per 40 x 25 textured frame** and
**8.99 h per 24 x 15 sprite frame**. These are aggregate simulation-run times divided
by requested frame count, not measured first-frame or input-to-display latency.
This review adds reporting/tests, not a measured runtime speedup.

```bash
python -m drosophilos.bench.render_budget \
  docs/a2/doom2_40_h200.json docs/a2/doom4_24na_h200.json
```

Sub-systems: FlyISA (instruction set), FlyASM (assembler), FlyLink (inter-node transport),
DrosoC (restricted C subset compiled to neural circuits), `minidoom` (the workload).

## Layout

```
drosophilos/
  sim/         reference and production spiking simulators, snapshots, execution modes
  connectome/  MCNS download, loader, manifests, Stage H0 embedding
  protocol/    four-phase self-timed spike protocol: latches, C-elements, completion
  lib/         circuit primitives with measured contracts
  isa/         FlyISA, arithmetic semantics, assembler, IR interpreter
  compiler/    DrosoC front end, IR, resource planning, expansion, placement, image
  cluster/     FlyLink, nodes, pods, synchronization, TMR, commit log, control planes
  minidoom/    the workload (DrosoC source, assets, golden reference)
  display/     pixel-record sinks
  bench/       benchmark reports
tests/
docs/
```

## Development

```bash
uv sync
uv run pytest
```

## Historical status snapshot (2026-09-14)

The entries below are retained as the earlier record. Use the September 17 campaign
report and addendum above for current results, qualifications and priorities.

- **Stage 0 done.** Normative model in `drosophilos/sim/schedule.md`; float64 reference
  simulator and PyTorch production simulator agree spike-for-spike with each other and with
  Brian2 (`tests/`); arithmetic semantics in `drosophilos/isa/semantics.md` with Python and
  C implementations checked against 7,989 shared vectors under UBSan; MCNS v1.0 loader
  reproduces the published counts (166,700 neurons; 25,582,938 edges at ≥ 1 synapse;
  6,242,118 at ≥ 5) under stated rules.
- **Stage H0 done.** A dual-rail AND with register, completion, and reset embedded in real
  MCNS wiring (Profile 2): works isolated and inside the full graph with its outputs
  silenced; an unshielded circuit ignites 12–18 % of the brain. Search and harness in
  `drosophilos/connectome/embed_h0.py` and `h0_run.py`; data tables in `docs/h0_report.md`,
  conclusions in `docs/h0_findings.md`. Rerun: `uv run python -m drosophilos.connectome.h0_run`
  then `uv run python -m drosophilos.bench.h0_report`.

- **M1 (Stage A1) done.** Four-phase dual-rail token protocol (spec, executable abstract
  machine, exhaustive checker) and its neural implementation: latches, latched completion
  tree, reset train, READY chain, edge relays, latched FAULT path; gates, 1-bit full adder,
  4-bit ripple adder, all exact. 10⁶ perturbed 4-bit transactions: 0 wrong values consumed,
  2.2 × 10⁻⁵ detected refusals/hangs. Report `docs/m1_report.md`, contracts `docs/contracts/`.

- **A2 opening done.** Neural watchdog (timeout → FAULT-ACCEPT); stale monitor rejected on
  measurement; adder composition campaign fixed three defects (reset strength, ignition
  doublets, AND window). Final build at mix B: channel 10⁵ with 0 wrong values; 4-bit adder
  10⁵ random additions with 0 wrong sums, 6.4 × 10⁻⁴ refusals/hangs (half raised neurally).
  Channel contract frozen. Notes: `docs/a2_liveness.md`.

- **A2 step 6 done (ALU, staged-commit register, accumulator).** 4-bit ALU (ADD/SUB/AND/OR/
  XOR/MOV, flags C Z V, exact against the ISA semantics), word register with staged commit
  (a COMMIT token applies exactly once; faults and timeouts discard with the master untouched),
  and the first closed loop (ALU → register → operand). Two new primitives: the veto relay (an
  AND with no exposure window) and the light relay hold. Mix B campaigns: ALU 5,000, register
  5,000, accumulator 2,000 random instructions, all correct, no refusals. Notes:
  `docs/a2_alu_register.md`; contracts `docs/contracts/{alu,staged_register,accumulator}_4bit.yaml`.

- **Ordered datapath done.** The adder and the whole ALU datapath run on veto relays only,
  with arrival order fixed by delay chains (operand gate, delayed B, delayed carries; Z from
  the consumer's completion). Mix B campaigns: ordered adder 30,000 random additions, ALU 5,000, accumulator
  2,000 instructions, all correct, no refusals (the rate-mode adder had 6.4 × 10⁻⁴ refusals/hangs). Notes: `docs/a2_alu_register.md` §4.

- **Data RAM and control machine done.** Word masters with veto-relay decoding (exact, no
  threshold margin); a one-hot sequencer that executes an 8-word program held in neural
  memory (MOV/ADD/SUB/AND/OR/XOR, LOAD, STORE, JZ/JNZ, HALT) with the accumulator and the
  RAM, checked instruction by instruction against a Python reference. Safe-point interrupts; a compiler path (DrosoC → IR → interpreter → machine,
  checked against a clang/UBSan golden reference) with a compiled program running on the
  neural machine; a capacity report.
  Notes: `docs/a2_ram_control.md`.

- **Stage C started.** FlyLink transport between simulated nodes; two machines: A computes
  and sends, B's arrival interrupt computes and emits the pixel (`docs/stage_c_flylink.md`).
  The exact channel (alternating bit, immediate ACK, timeout and retransmit, duplicate
  rejection) passes its clean-link and dropped-event scenarios on two neural machines.
- **A frame rendered in the substrate.** `examples/frame.c`: a 160 × 100 Doom-like view, one
  token per pixel, on 128 copies of a 16-cell pixel kernel (65k neurons each) on one H200 —
  all 16,000 pixels equal to the reference in 411 s of neural time; a three-frame slideshow
  with a tick kernel turning the view likewise (`RESULTS.md`, `docs/img/frame160_neural.png`).
- **Neural pacing (Stage F2, first step).** The frame's phase order lives in the substrate
  (phase gates on the input registers, wrapping counters from the compiler); the host only
  deals tokens in program order (`docs/a3_kernels.md` §11).
- **Profile 2 feasibility measured.** The kernels above are free synthesis (Profile 3). MCNS
  holds 1,223 reciprocal cholinergic pairs strong enough for a latch at the H0 weight
  policy (12,315 at four times the scale), 5 of them in the visual system; the render
  kernel needs ~5,300. A first greedy placement of the 4-bit adder carries 10 % of its
  edges (`docs/capacity_doom.md` §5, `connectome/embed_netlist.py`).
- **Resident kernels.** A loop body compiles to a spatial dataflow pipeline of ALU and
  memory-read cells with a neural handshake between them; the toy renderer's column loop
  runs one column per 1.18 s, ~27× the sequencer, on 13.7k neurons; the toy world update
  (`examples/tick.c`: calls, ifs as select cells, loop-carried state) runs as a 13-cell state
  kernel at one tick per 6 s with every output correct; a 16-bit perspective column
  (reciprocal table and multiply) renders at 6.6 s per column; a two-loop program (frame
  loop around a column loop) runs as a tick kernel and a column kernel with the host pacing
  the frames, two frames of eight columns correct (`docs/a3_kernels.md`).

Data: `uv run python -m drosophilos.connectome.mcns_download` (1.1 GB, Janelia, CC-BY, no login).
