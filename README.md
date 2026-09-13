# DrosophilOS

A connectome-constrained, asynchronous neural computer. The substrate is a simulated
spiking network built on the *Drosophila* male central nervous system connectome
(MCNS v1.0, 166,700 neurons). The target program is a Doom-like engine whose game update
**and** renderer both execute in the neural substrate; the host only integrates neuron
equations, transports spike events, transduces input, loads the program image, and
displays already-computed pixels.

Design document: `docs/spec.md`. Plan and milestone order: `docs/plan.md`.

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

## Status (2026-09-13)

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

Data: `uv run python -m drosophilos.connectome.mcns_download` (1.1 GB, Janelia, CC-BY, no login).
