# DrosophilOS — results record, Stages 0 and H0

**Date:** 2026-09-13. **Repository:** https://github.com/jgassens/drosophilos (commit `5e1869e` and
this file). **Hardware:** Apple M1 Pro, 16 GB, macOS 26; Python 3.12, PyTorch 2.14, NumPy 2.5,
Brian2 2.10, SciPy. **Data:** MCNS v1.0 (male *Drosophila* central nervous system connectome,
HHMI Janelia, CC BY 4.0), bulk tables `body-annotations`, `body-neurotransmitters`,
`connectome-weights` (min confidence 0.5).

This is a record of what was built, what was measured, what worked, and what did not,
including the dead ends. Every number below is reproducible from the repository with the
commands in §7.

---

## 1. Claim boundary

DrosophilOS treats a simulated spiking network built on a real fly connectome as computer
hardware. The long-term target is a Doom-like engine whose game update **and** renderer both
execute in the neural substrate. Nothing in this record claims that. What is claimed:

1. A spiking simulator whose model is written down normatively and certified three ways
   (against a float64 reference, against Brian2, and by internal consistency tests).
2. An arithmetic specification implemented identically in Python and C, checked on 7,989
   shared vectors under Clang's undefined-behaviour sanitizer.
3. A loader for MCNS v1.0 whose stated rules reproduce the published neuron and connection
   counts exactly.
4. **Stage H0:** a minimal circuit (dual-rail 1-bit AND with a register, a completion
   detector, and a reset path) embedded in real MCNS neuron IDs under "Profile 2" rules
   (anatomical graph preserved; weights scaled within a bound; documented zeroing allowed;
   no added or rerouted edges). It computes correctly when isolated or when its outputs are
   silenced, and fails when the surrounding brain is live.

All benchmarks carry four labels: substrate profile (**2**), isolated **and** full-graph
execution, **hybrid** orchestration (the host injects input tokens and reads spikes), and
**external** compilation (the circuit was designed by hand and by a search program, not by
the machine).

---

## 2. Methods

### 2.1 Neuron and synapse model (normative text: `drosophilos/sim/schedule.md`)

Current-based leaky integrate-and-fire neuron with an exponential synaptic kernel:

```
tau_m dV/dt = (E_L + b − V) + g        tau_s dg/dt = −g
```

Defaults are the Shiu et al. (2024) whole-brain model values: `E_L = −52 mV`,
`V_th = −45 mV`, `V_reset = −52 mV`, `tau_m = 20 ms`, `tau_s = 5 ms`, refractory `2.2 ms`,
one anatomical synapse `= 0.275 mV`, delay `1.8 ms` on every edge, `dt = 0.1 ms`.
Sign comes from the presynaptic transmitter (GABA, glutamate, histamine negative; all
others including "unclear" positive). Integration is the exact closed-form solution of the
linear system over one step; the within-step order is fixed to Brian2's default slot order
(integrate → threshold → synaptic delivery → reset). Synaptic weights are **integers**
(16 quanta per anatomical synapse, `0.0171875 mV` per quantum) so that delivery sums are
order-independent and spike traces are bit-identical across backends and batch sizes.

### 2.2 Simulators

- `sim/ref64.py`: NumPy float64 reference. The oracle.
- `sim/lif_torch.py`: PyTorch, batched over nodes, CPU/CUDA/MPS.
- Comparison objects: membrane trajectories, the sorted spike-event trace `(step, node,
  neuron)`, and decoded transactions. Snapshots (state, delay queue, pending events) restore
  bit-exactly.

### 2.3 Arithmetic semantics (`drosophilos/isa/semantics.md`)

Wrapping and saturating integer arithmetic at 8/16/32 bits, Q16.16 fixed point matching
Doom's `FixedMul` (64-bit product, arithmetic shift, floor for negatives), division with
truncation toward zero and defined division-by-zero, shifts with defined out-of-range
behaviour, width conversions, and a four-bit status word (SAT, DIVZ, SHIFT, OVF).
Implemented in `isa/semantics.py` (Python integers) and `minidoom/fx/fx.h` (C, no signed
overflow, no shifts of negative values).

### 2.4 Connectome loader (`drosophilos/connectome/mcns.py`)

Rules, recorded verbatim in every manifest:

| rule | statement |
|---|---|
| neuron | a body is a neuron iff its `superclass` annotation is non-null |
| edge | both endpoints are neurons and synapse count ≥ `min_syn` |
| sign | presynaptic `consensus_nt` ∈ {gaba, glutamate, histamine} → −1, else +1 |
| delay | every edge 1.8 ms |

### 2.5 The H0 circuit

Dual-rail encoding: each bit has two rails; a spike on rail 1 means "valid one", on rail 0
"valid zero"; silence is "not here yet"; both is a fault.

| role | construction |
|---|---|
| register (four rails a1, a0, b1, b0) | two-neuron mutual excitatory loop; a circulating spike holds the value |
| y1 = a1 AND b1 | one neuron receiving sub-threshold drive from each of two latches; fires only on both |
| y0 = a0 OR b0 | one neuron receiving supra-threshold drive from either latch |
| completion | one neuron driven by y1 or y0 (single pulse suffices) |
| reset / return path | completion → inhibitory neurons → both members of every latch |

Design targets, all derived from the model's measured physics rather than assumed:

| quantity | value | how obtained |
|---|---|---|
| membrane peak per mV of synaptic jump | 0.1575, at 9.2 ms | simulated impulse response |
| single synchronous input that just reaches threshold | 2,586 quanta (161.6 synapses) | 7 mV gap ÷ peak |
| latch drive (1.4×) | 3,621 quanta | margin |
| latch loop period at that drive | 4.7 ms (213 Hz), first spike after 3.5 ms | simulated isolated loop |
| sustained-train drive that just reaches threshold | 383 quanta per pulse | mean of periodic drive |
| AND input per latch (0.65×) | 249 quanta | one latch 4.6 mV, two 9.1 mV |
| OR input (2×) | 766 quanta | margin |
| completion and reset-drive edges | 3,621 quanta (single-pulse mode) | see §4.3 |
| reset per latch member (1.5× loop drive) | −5,432 quanta | measured, see §3.4 |

Profile 2 policy: `0 ≤ |q| ≤ 4 × count × 16` per anatomical edge (k_max = 4), sign
preserved; anatomical edges among circuit neurons that are not part of the design
("parasitic") may be zeroed and are listed; outgoing edges to the surround may be zeroed
in a separately labeled condition. Required anatomical synapse counts follow from the
targets: latch ≥ 57 both directions, AND ≥ 4, OR ≥ 12, completion ≥ 57, reset ≥ 85 per
member (several inhibitory neurons may add up).

### 2.6 Search (`drosophilos/connectome/embed_h0.py`)

Sparse-matrix enumeration: candidate latches = mutual cholinergic pairs with both
directions ≥ 57 synapses; completion candidates = cholinergic neurons with ≥ 2
cholinergic inputs ≥ 57; reset feasibility precomputed as a sparse product (completion →
inhibitory partner → latch member) so that latches are filtered before gate assignment;
gate candidates from the completion neuron's strong inputs; four pairwise-disjoint latches;
solutions scored by weight margin, then isolation cost.

### 2.7 Experimental conditions (`drosophilos/connectome/h0_run.py`)

Each transaction: inject one DATA event per active rail at 5 ms (b optionally delayed),
run 150 ms, read the y1/y0/completion/reset spikes, judge correctness (exactly the expected
rail fired and completion fired), reset (all latches silent for the last 30 ms), latencies,
surround spikes and recruited neurons, and the boundary-input envelope (quanta delivered
into circuit neurons from non-circuit spikes, per neuron per step).

| condition | topology | surround |
|---|---|---|
| isolated, parasitic zeroed | 15 circuit neurons only | — |
| isolated, parasitic anatomical | 15 neurons + their 107 mutual anatomical edges | — |
| full graph, surround live | all 166,700 neurons, 25.58 M edges, designed edits applied | silent / 2 Hz on the 17,937 sensory neurons / one synchronous burst of 1,000 cholinergic neurons |
| full graph, outputs zeroed | as above, plus every circuit→surround edge zeroed (4,830 edges) | same three |

---

## 3. Results

### 3.1 Simulator certification (Stage 0)

| test | result |
|---|---|
| PyTorch float64 CPU vs NumPy reference, random 60-neuron recurrent circuit, 2 nodes, 3,000 steps | identical spike traces and identical recorded trajectories |
| batch independence (node simulated alone vs in a batch) | identical |
| snapshot at step 1,000, run 500, restore, continue | identical to uninterrupted run |
| Brian2 (`method='exact'`, matched scheduling), same circuit, 3,000 steps | identical spike trace, spike for spike |
| refractory semantics, delay semantics, silenced neurons, zero-delay self-consistency | pass |
| arithmetic: 7,989 vectors, Python spec vs C helpers under `-fsanitize=undefined` | 0 failures |
| MCNS loader vs published counts | 166,700 neurons; 25,582,938 edges at ≥ 1 synapse; 6,242,118 at ≥ 5; 124,177,617 contacts |

The loader result resolves an open point from the plan review: Codex's displayed
"6,242,118 connections" is the ≥ 5-synapse graph, and the "25.5 M" figure is the ≥ 1-synapse
graph at confidence 0.5, both between the 166,700 annotated neurons.

Physics of the default model that everything downstream depends on: a synaptic jump moves
the membrane by only 0.49 % of its size in the first 0.1 ms step and peaks 9.2 ms later at
15.75 % of it. Crossing the 7 mV threshold gap therefore needs about 160 synapse-equivalents
arriving together, and about 5,150 to fire on the very next step. Gate latencies in this
substrate are milliseconds, not steps.

### 3.2 Motif availability in MCNS v1.0

| motif class | count |
|---|---|
| cholinergic edges ≥ 24 / ≥ 41 / ≥ 57 / ≥ 80 / ≥ 120 synapses | 486,764 / 187,456 / 99,501 / 50,524 / 20,622 |
| GABA/glutamate edges ≥ 24 / ≥ 57 / ≥ 120 | 297,742 / 75,486 / 15,417 |
| mutual cholinergic pairs, both directions ≥ 24 / ≥ 41 / ≥ 57 / ≥ 80 | 5,793 / 2,354 / 1,223 / 589 |
| cholinergic 3-cycles, all edges ≥ 57 | 8,402 |
| neurons with ≥ 2 cholinergic inputs ≥ 57 | 10,782 (5,004 of them cholinergic) |
| inhibitory neurons with ≥ 4 strong (≥ 57) outputs and a strong cholinergic input | 3,796 |
| completion candidates with four resettable latches (final policy) | 141 |
| completion candidates with usable AND + OR gate pairs | 610 |
| complete embeddings found | 50 in 1.8 s (search stopped at 50; not exhausted) |

### 3.3 The chosen embedding

| role | bodyId | type | superclass | transmitter | side |
|---|---|---|---|---|---|
| latch a1 driver / partner | 32461 / 10864 | GNG457 / GNG014 | cb_intrinsic | ACh | R / R |
| latch b1 driver / partner | 523040 / 11127 | GNG108 / DNge059 | cb_intrinsic / descending | ACh | L / L |
| latch a0 driver / partner | 26764 / 10643 | GNG108 / DNge059 | cb_intrinsic / descending | ACh | R / R |
| latch b0 driver / partner | 37111 / 555296 | GNG457 / GNG014 | cb_intrinsic | ACh | L / L |
| y1 (AND) | 10881 | GNG120 | cb_intrinsic | ACh | R |
| y0 (OR) | 512079 | GNG169 | cb_intrinsic | ACh | L |
| completion | 11429 | GNG236 | cb_intrinsic | ACh | L |
| reset ×4 | 14210, 520816, 14160, 523983 | GNG048, GNG298, (untyped), GNG048 | cb_intrinsic | GABA | R, M, L, L |

The search is agnostic to biology; it picked gnathal-ganglion local neurons and, for two of
the latch partners, the descending neuron DNge059 on each side. Minimum weight margin is
0.27 against the 0.25 floor (the tightest designed edge is scaled 3.65× its anatomical
count). 107 parasitic anatomical edges among the 15 neurons were zeroed. Full neuron and
edge tables: `docs/h0_report.md`; manifest with every edit: `docs/h0/manifest_full_graph_surround_live.json`.

### 3.4 Reset requirement (measured on an isolated latch, 50 phases across one period)

| inhibitory drive per pulse | one member, 1 pulse | both members, 1 pulse | one member, 4 pulses 8 ms apart | both members, 4 pulses |
|---|---|---|---|---|
| 1,811 quanta (28 syn at k=4) | 0/50 | 0/50 | 0/50 | 50/50 |
| 3,621 (57) | 0/50 | 0/50 | 50/50 | 50/50 |
| 5,432 (85) | 0/50 | **50/50** | 50/50 | 50/50 |
| 7,242 (113) | 0/50 | 50/50 | 50/50 | 50/50 |
| 10,863 (170) | 41/50 | 50/50 | 50/50 | 50/50 |

Policy adopted: 5,432 quanta on both members, since a gate that barely exceeds threshold
fires only every ≈ 30 ms and cannot be relied on for a pulse train.

### 3.5 Transactions

Isolated, parasitic edges zeroed:

| a | b | y1 | y0 | completion | correct | completion latency | ready latency |
|---|---|---|---|---|---|---|---|
| 0 | 0 | – | ✓ | ✓ | ✓ | 22.8 ms | 26.9 ms |
| 0 | 1 | – | ✓ | ✓ | ✓ | 36.9 ms | 41.7 ms |
| 1 | 0 | – | ✓ | ✓ | ✓ | 36.9 ms | 41.7 ms |
| 1 | 1 | ✓ | – | ✓ | ✓ | 56.3 ms | 60.8 ms |

Delaying operand b by 0, 1, 2, 3, 4, 5, 6, 8, 10, or 12 ms: all correct. The gates
integrate sustained trains, so operand arrival phase does not matter.

Isolated, parasitic edges at anatomical strength: (0,0), (0,1), (1,0) correct; **(1,1)
fails** (y0 fires instead of y1) at every operand offset.

Full graph:

| condition | correct | reset | surround neurons recruited | surround spikes / 150 ms | envelope into circuit, max per step (quanta) |
|---|---|---|---|---|---|
| surround live, silent | 0/4 | 2/4 | 19,680 – 23,917 | 98,908 – 217,118 | +5,104 / −6,176 |
| surround live, 2 Hz sensory background | 1/4 | 2/4 | 29,862 – 30,045 | 457,642 – 469,185 | +6,784 / −5,920 |
| surround live, 1,000-neuron burst | 1/4 | 2/4 | 26,694 – 27,835 | 400,878 – 419,051 | +6,656 / −7,008 |
| outputs zeroed, silent | **4/4** | **4/4** | 0 | 0 | 0 |
| outputs zeroed, background | 1/4 | 1/4 | 29,730 – 30,172 | 456,967 – 468,281 | +6,784 / −7,168 |
| outputs zeroed, burst | 1/4 | 2/4 | 26,393 – 27,163 | 402,754 – 420,968 | +6,848 / −7,552 |

With outputs zeroed and a silent surround, the spike times equal the isolated run's
exactly, so that configuration is classified *equivalent under stated isolation
assumptions* (the assumption being a silent surround). Total stray input per 150 ms
transaction in the live conditions: 1.2 – 5.2 million quanta. Most active recruited
neurons: octopaminergic VUM neurons (OA-VUMa1/a8), ExR6, olfactory local neurons
(lLN2, il3LN6), APL.

Isolation cost of the chosen circuit: 4,830 anatomical edges out to the surround
(72,274 synapses; 697 edges ≥ 24 synapses) and 5,493 edges in from it (71,364 synapses;
714 ≥ 24).

Wall-clock: one 150 ms transaction takes 0.02 s isolated, 2.9 s in the full graph with a
quiet surround, 4.4 – 7.8 s with the brain erupting. The complete H0 protocol (search plus
36 transactions) runs in about five minutes on the laptop.

---

## 4. What did not work, in the order it happened

### 4.1 First search design: zero embeddings in 398 s
Reset was specified as one inhibitory neuron with a single edge of ≥ 170 synapses onto a
latch member (a 3× reset factor from a back-of-envelope estimate), and the enumeration
tested reset only after choosing four latches. 263 completion candidates had usable gate
pairs and 757,718 latch assignments were tried; none had a reset. Both the requirement and
the search order were wrong.

### 4.2 Gates designed for single pulses
The original AND assumed two coincident single spikes. A latch is not a single spike: it
emits a 213 Hz train. Under a train the mean synaptic drive is `q × w_unit × tau_s / period`,
so a single latch alone (≈ 29 mV) would have fired an AND designed for coincidence. The
gate inputs were redesigned in rate mode (one latch 4.6 mV, two 9.1 mV against a 7 mV gap).

### 4.3 Rate-mode chain starvation
With every downstream edge in rate mode, the AND (which sits barely above threshold) fired
only every ≈ 34 ms, the completion neuron received too little drive and fired twice in
150 ms, and the reset neurons never fired. Fix: completion and reset-drive edges were moved
to single-pulse mode (3,621 quanta), where each gate spike fires its target alone.

### 4.4 A misleading reset probe
The first reset probe forced the inhibitory neuron with a 100,000-quanta injection. That
much drive made it burst seven times in 18 ms, so even −1,811 quanta on one member appeared
to stop the latch at every phase. Inside the circuit the same neuron fires once per
completion spike and the latch survived sixteen of them. Re-probing with realistic
single-spike drive gave the table in §3.4. Lesson recorded as a harness rule: never
characterise a primitive with unrealistic injection.

### 4.5 Search grind from an over-strict constraint
The reset selection forbade reusing an inhibitory neuron across latches. Assignments that
passed the vectorised feasibility check then failed late, and the enumeration ran for
15 minutes at 100 % CPU without output. Allowing reuse (a strong inhibitory neuron serving
several latches is exactly what one wants) and capping tries per gate pair brought the
search to 1.8 s.

### 4.6 A column-indexing bug caught by the synthetic test
Reading a column of a CSR sparse matrix returns column indices of a one-column matrix (all
zero), not row indices. The reset candidate lookup silently found nothing until the
synthetic-connectome test (a planted motif in a 55-neuron graph) failed. The synthetic test
is now the gate before any run on the real data.

### 4.7 Manifest validation crashed the first full run
The sign-flip check compared a negative inhibitory target against the unsigned anatomical
synapse count. The run had finished all conditions and crashed while saving the manifest,
before writing results. Fixed by storing signed anatomical quanta and writing results before
the manifest. All conditions were rerun.

### 4.8 Test drives too weak for the model
The first simulator tests injected "eight synapses' worth" of drive and expected spikes.
Nothing fired: the physics in §3.1 was not yet understood. The tests were rewritten around
measured requirements, and the physics was written into the normative model file.

### 4.9 What the surround does on its own
This is the finding that matters most and it is not a bug. Under the default parameters
the whole-brain model has two states: silence and runaway. A 2 Hz drive on the sensory
neurons, or one synchronous burst of 1,000 cholinergic neurons, recruits ≈ 30,000 neurons
into sustained firing with no circuit involved. Uniform 0.275 mV synapses without
adaptation, synaptic depression, or tonic inhibition make the connectome bistable. The
review's demand — "operate within a largely active surround" — cannot be tested for any
circuit until the surround model has a stable active regime. That is a modelling task, now
on the Stage H list, and it applies to every published whole-brain LIF model with these
parameters, not just to this project.

---

## 5. What the results mean for the machine

| finding | consequence |
|---|---|
| Latches broadcast at 213 Hz through thousands of anatomical edges | Profile 2 circuits need documented silencing of their outputs, or placement on neurons with few strong partners. Isolation cost is now a search objective and a manifest line. |
| Stray input of thousands of quanta per step destroys any designed drive | Incoming surround edges must be silenced too unless the surround is provably quiet; stray-input tolerance becomes a primitive contract field. |
| Gates on latch trains work in rate mode | Stage latency is 20 – 60 ms, not the 2 ms assumed in the architecture document; a 100-stage critical path is seconds, not 200 ms. Single-pulse coincidence gates on quiet inputs and bounded threshold edits are the levers to measure in Stage A1. |
| Operand arrival phase does not matter | The self-timed protocol gets timing slack for free, paid for in latency. |
| Reset needs 1.5× loop drive on both members from a single spike | 85-synapse inhibitory edges, or several inhibitory neurons summing to it; plentiful (75,486 inhibitory edges ≥ 57). |
| The register bits landed in descending neurons | Placement policy should carry a no-touch list (sensory, motor, descending) for Profile 1/2 hygiene. |

---

## 6. Limitations and things not done

- The circuit is 15 neurons and one bit wide. Nothing here shows that thousands of such
  circuits compose; Stage A1/A2 will measure composition contracts.
- "Works in real wiring" means: with 30 designed edges rescaled up to 3.65×, 107 parasitic
  edges zeroed, and (in the only working full-graph condition) 4,830 outgoing edges zeroed.
  Preserving an adjacency matrix while zeroing inconvenient edges is the weaker of the two
  achievements the review distinguished; the stronger one was not achieved and, given §4.9,
  is not currently testable.
- Non-spiking cells (APL and others) are simulated as LIF neurons in full-graph mode; the
  declared model for them is "same as everything else", which is a documented gap.
- All delays are 1.8 ms; MCNS provides no per-synapse delay data.
- The k_max = 4 weight bound is a policy choice, not a physiological measurement.
- The search was stopped at 50 embeddings; the count of all embeddings is unknown.
- No matched-shuffled-graph control has been run (Stage H), so nothing here says the fly
  wiring is better or worse than a random graph with the same degree structure.
- Simulator certification on CUDA/MPS at the transaction level, and the timestep-refinement
  test, are specified but not yet run (no primitives exist yet to run them on).
- Brian2 parity was checked on one random circuit family; the circuit-library harness will
  extend it.

---

## 7. Reproduction

```bash
git clone https://github.com/jgassens/drosophilos && cd drosophilos
uv sync                                                    # Python 3.12 environment
uv run pytest                                              # 22 tests, ~2 min (Brian2 included)
uv run python -m drosophilos.connectome.mcns_download      # 1.1 GB from Janelia, no login
uv run python -m drosophilos.connectome.mcns               # builds the cache, prints the counts
uv run python -m drosophilos.connectome.h0_run             # search + 36 transactions, ~5 min
uv run python -m drosophilos.bench.h0_report               # regenerates docs/h0_report.md
uv run python -m drosophilos.isa.semantics                 # regenerates the arithmetic vectors
```

Files: normative model `drosophilos/sim/schedule.md`; semantics
`drosophilos/isa/semantics.md`; plan `docs/plan.md`; architecture specification
`docs/spec.md`; the review that shaped the plan `docs/review-2026-09-13.md`; H0 data
tables `docs/h0_report.md`; H0 conclusions `docs/h0_findings.md`; H0 raw results and manifest
`docs/h0/`.

## 8. Attribution

Architecture specification and review: Jeremiah Gassensmith. Implementation, experiments,
and this record: Claude Fable 5.1 (Anthropic) working in Claude Code, directed by the
author. Connectome: MCNS v1.0, HHMI Janelia FlyEM project team and collaborators
(CC BY 4.0). Model parameters: Shiu et al., *Nature* 2024.
