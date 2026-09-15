# DrosophilOS

A connectome-constrained, asynchronous neural computer. The substrate is a simulated
spiking network built on the *Drosophila* male central nervous system connectome
(MCNS v1.0, 166,700 neurons). The target program is a Doom-like engine whose game update
**and** renderer both execute in the neural substrate; the host only integrates neuron
equations, transports spike events, transduces input, loads the program image, and
displays already-computed pixels.

Design document: `docs/spec.md`. Plan and milestone order: `docs/plan.md`.
Results record (what was done, what worked, what did not): `RESULTS.md`.

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

## Status (2026-09-14)

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
- **Resident kernels.** A loop body compiles to a spatial dataflow pipeline of ALU and
  memory-read cells with a neural handshake between them; the toy renderer's column loop
  runs one column per 1.18 s, ~27× the sequencer, on 13.7k neurons (`docs/a3_kernels.md`).

Data: `uv run python -m drosophilos.connectome.mcns_download` (1.1 GB, Janelia, CC-BY, no login).
