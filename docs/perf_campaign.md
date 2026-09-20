# Performance campaign — orders and ledger

Adopted 2026-09-17 from the owner's campaign instructions (written against the 17 September
campaign report and the repository audit). The instructions are reproduced below unchanged;
the ledger at the top records what has been done against them and where the evidence is.

## Ledger

| order | status | evidence |
|---|---|---|
| §2 backend selection, timing, reproducibility record, profiling regions | done 2026-09-17 (openai-sol, cleanup openai-terra) | `fe2f07a`, `f95bcdc`; `bench/render_doom.py`, `bench/repro.py`, `sim/profile.py` |
| §3 fixed comparison suite (`bench/perf_campaign.py`) | built 2026-09-17 (openai-terra); sized for the cluster 2026-09-18 (claude-sonnet, openai-luna) after a default run consumed 14 h of laptop CPU; primitive baseline measured on Juno (412136, 1 h 16 min on one H200; `docs/perf/juno-h200-primitive.md`): wall/neural 5.1–5.3× at 3k neurons, 5.9× at 28k, 6.2–8.0× at 47k–110k; 8 copies cost the same wall time as 1 below ~50k neurons; the pipelined multiplier is 2.1× faster per token in isolation. Small level: 412137 crashed after a correct 2.5 h doom2 render on a duplicates-counter bug (fixed, `2cb03db`); rerun 412442 running (3/4 configurations) | `docs/perf_campaign_suite.md`; results land in `data/perf/` when the jobs run |
| §4 Track A (simulator) | **done 2026-09-19**: profile (Juno 412204) showed a launch- and sync-bound step, flat 0.36–0.46 ms from 3k to 880k neurons; FastSim (static-shape step, sparse-product delivery, Observer, K-step CUDA graphs) merged `70f7267`; H200 A/B (413020): 5.3× at 3k neurons, 3.6× at 28k, 2.2× at 110k, all outputs identical; the profiled twin (413021) shows the graph replay at 0.09–0.23 ms/step at every size, the rest being the primitive level's own all-neuron spike capture (`--no-spike-count` added); reviews: Kimi (A–B) and openai-sol (C–F), five findings fixed `bcb1f10`, `dd82449`; frame-scale twin running (413097 vs 412442) | `docs/track_a.md`, `docs/perf/juno-h200-trackA-ab.md`, `docs/perf/juno-h200-fast-profile.md` |
| §4 Track B (specialized cells) | **B1+B2 merged and measured 2026-09-20** (`f1806a0`; Juno 413149): −31–35 % neurons per AND/XOR/MOV cell, −14–18 % per kernel, latency −8–16 ms, wall time unchanged (FastSim's step does not scale with neurons at these sizes); STORE stays generic; 100-copy noisy campaign not yet run | `docs/track_b.md`, `docs/perf/juno-h200-trackB-prim.md` |
| §5 multiplier diagnostic | **done 2026-09-20** (`docs/mul_diag.md`): the pixel pass runs one pixel at a time — early values read by late cells hold their producers' commits (operand retention), so the period is the path latency (array 26.2 s, pipelined 42.4 s per pixel; the pipelined rows overlap 0 of 40 times inside the pass against 15 of 16 standalone); latency, not throughput, is the cost. Reproducer `tests/test_pipeline_retention.py` (relaying the value: 2.3× throughput). Fix = relaying long-lived values (the 'retained intermediates' experiment), not a multiplier change | `docs/perf/muldiag_*.json` |
| §6 tick-failure diagnosis, 1,000-tick Stage D exit | tool merged 2026-09-20 (`bench/stall_diag.py`, openai-sol): rebuilds the campaign kernel, validates a dump's roles, reconstructs every cell's request/repair/clear/go/START/ACT^d/stage/commit/DONE per token and names the first blocked phase; reproduces the seed-107 precedent (c2_sel, latch 11828, relight 11970 @ 37,674) with a passing RefSim regression of the mechanism; campaigns take `--backend torch-fast` with identical perturbation draws. **Seed-108 replay does not reproduce the recorded copies** (Track B's index reorder changed the noise realization; the build itself is deterministic, 100/100 copies identical across reruns) and shows **5 wrong values on copy 77** in 1,600 outputs — the report's '0 wrong in 4,000' was one realization; **copy 77 read (413584): one refused input word, not five wrong values** — bit 0 of token 4 never latched, the producer watchdog refused the word after 459 ms, READY re-rose and the host loaded the next token; outputs match the 7-token reference exactly. Fixed on the host side 2026-09-20: both runners count refusals per TIMEOUT rise and resend the refused word (`retry_refused`, `max_retries=3`, blocked nodes), `tests/test_input_refusal.py`; **copy 8 read (413616): a request's false rail running at 34 steps (mix-B drift + weight draw) survived its 3 × 0.75 kill train** — the train has no margin against a fast latch; measured in RefSim, stronger kill trains measured and **rejected** (4 × 1.5 stops the control machine; 3 × 1.5 stalls 85/100 mix-B copies, Juno 413672) — train stays 3 × 0.75, the margin problem is recorded in `tests/test_kill_margin.py` as an open item, `tests/test_kill_margin.py`; review (sol) → resend gated on the word's commit, next word only after the previous commit, READY over an uncommitted word = lost word (resent), and the stage reset now clears the producer's TIMEOUT latch (`protocol/handshake.py`); old-build baseline seeds 109/110 = 4 + 2 failing copies incl. **a new silent wrong-value mechanism (seed 109 copy 61: `c9_sel` commits its initial value early, state lags a token; capture 413685 queued)**; new-build seeds 108–110 running; 1,000-tick exit not yet run | `docs/tick_stalls.md`, `tests/test_stall_diag.py` |
| §7 report recommendation and speed wording | done 2026-09-17 | `docs/campaign_report_2026-09-17.md` §6, §3 |
| §8 four-configuration final comparison | not started | — |

---

# DrosophilOS Performance Campaign — Specific Instructions for the Next Campaign

**Source note:** These recommendations are based on the 17 September 2026 DrosophilOS campaign report and the repository audit discussed in this conversation.

Replace the current “write it up, then build E2” recommendation with a performance campaign that has two implementation tracks: faster simulation of the existing circuits, and smaller exact circuits. Keep E2 as a measured follow-on rather than assuming it is the only route to acceptable speed.

## 1. Change the immediate objective and limit the scope

Use this as the next campaign’s objective:

> Reduce the wall-clock cost of the existing neural world-update and rendering workloads while preserving their computed results. Measure simulator throughput separately from neural circuit latency. Establish which changes improve complete-frame performance, which only reduce circuit size, and which introduce correctness or reliability regressions.

Keep the current neuron model, timestep, arithmetic semantics, scenes, and reference outputs unchanged for the initial comparisons. Keep the current array multiplier and request-repair mechanism as the baseline.

Defer additional gameplay features, higher-resolution demonstrations, multi-GPU expansion, and self-hosted compilation. Those remain roadmap items, but they should not consume the primary performance effort. Continue the feasibility write-up and tick-failure diagnosis alongside this work. Stage D’s 1,000-tick requirement and the missing C/F/F2 components remain incomplete; a speed improvement does not complete them.

## 2. First implementation: repair the benchmark and record the missing timing evidence

**Files:** `drosophilos/bench/render_doom.py` and the runners in `drosophilos/lib/kernel.py`.

### Make backend selection explicit

Add a backend choice independent of copy count: reference simulator or Torch simulator. A Torch run with one copy must still use Torch on the requested device. Reject unsupported backend/device combinations rather than silently falling back. Record the effective simulator class, device, and dtype in both the console output and saved report.

This fixes a concrete defect: the current one-copy branch calls `run_pipeline` without passing the requested device or dtype, while its summary can still print the requested device.

### Separate timing into distinct measurements

Record:

- compilation and circuit construction;
- image loading and startup;
- time to the first completed frame;
- subsequent frame-completion intervals;
- total run time;
- for each runtime input, when it was injected;
- when the resulting state committed;
- when the first frame using that state completed.

Do not use the periodic progress callback to timestamp frame completion. Add an observer callback that fires when the runner has collected the last required valid pixel. Record the neural completion step and the wall time at which the result becomes available to the host separately.

Report any final tick after the last displayed frame as a separate tail, rather than treating it as rendering latency.

### Add a reproducibility record

Each result must identify:

- source and commit hashes;
- compiled circuit hash;
- inputs and initial state;
- pacing mode;
- multiplier;
- copy count;
- resolution;
- neuron and edge counts;
- model parameters;
- perturbation configuration;
- seed;
- effective backend and dtype;
- hardware and software versions.

Save requested and completed frame counts separately.

Preserve these outcomes separately:

- wrong values;
- missing values;
- duplicates;
- invalid output words;
- neural faults;
- host-detected stalls;
- externally truncated runs.

Missing counters are unknown, not zero.

### Use profiling without turning every benchmark into a trace dump

Add named profiling regions for:

- numerical integration;
- spike delivery;
- device-to-host observation;
- host scheduling;
- output decoding;
- image writing.

Profile short representative intervals. Obtain headline timings from separate unprofiled runs.

Use synchronized measurement boundaries or CUDA events for GPU timings rather than timing asynchronous submissions alone.

### Required output

Produce a reproducible baseline that shows where wall time is spent, with actual frame-completion timestamps — not another aggregate JSON containing only total run time.

## 3. Replace repeated long renders with a small, fixed comparison suite

Add a campaign driver, for example:

`drosophilos/bench/perf_campaign.py`

It should run explicitly selected configurations and write one comparison report.

Use three levels:

| Test level | Proposed workload | Purpose |
|---|---|---|
| Primitive and composition | Fixed-operation cells, a fan-out kernel, a multiplier kernel, and the existing tick kernel; enough inputs to observe repeated operation | Measure latency, sustained throughput, reset behavior, and backpressure |
| Small complete render | Start with three-frame, 8 × 5 versions of `doom2.c` and `doom4.c`, using a fixed input sequence | Exercise complete phase transitions and expose end-to-end regressions without immediately repeating the largest runs |
| Historical-size confirmation | The existing 40 × 25 textured workload and 24 × 15 sprite workload | Confirm whether a promising local improvement survives the original workload scale |

The proposed small samples must first pass a coverage check:

- the selected inputs must change state;
- later frames must reflect those changes;
- the sprite test must actually contain visible sprite pixels.

Adjust the sampling or input sequence when it does not.

Within each comparison, change only one implementation choice. Keep the following identical:

- source;
- inputs;
- copy count;
- precision;
- pacing;
- sampled pixels.

Do not compare a smaller render or lower-precision run against the old configuration and call the difference an implementation speedup.

Start with one and eight copies on the same effective backend. Add more copies only to answer a specific scaling question.

Repeat the short unprofiled measurements and report their spread. Reserve historical-size runs for configurations that have already passed the smaller tests.

### Required output

Provide separate comparisons for:

1. unchanged-circuit simulator speed;
2. changed-circuit performance;
3. their combined effect.

## 4. Run two optimization tracks after establishing that baseline

# Track A: remove avoidable host interaction from the simulator

**Files:** `drosophilos/sim/lif_torch.py`, a separate optimized backend under `drosophilos/sim/`, and `run_pipeline_batched`.

The current timestep extracts spikes with `torch.nonzero`, copies spike indices to NumPy, reads a GPU-derived scalar into Python, and constructs variable-sized delivery arrays. These are concrete optimization targets, although their contribution to total runtime still needs measurement.

Make the changes in this order.

### A1. Separate simulation from observation

Introduce an explicit observation interface instead of requiring runners to inspect:

- `_spk_step`;
- `_spk_node`;
- `_spk_neuron`.

Retain full tracing for reference/debug runs.

Add a filtered mode that retains only the spikes needed for:

- READY detection;
- output completion;
- output-rail decoding;
- fault/timeout observation;
- any requested diagnostic capture.

Do not disable internal spike propagation. Only reduce what is copied and retained for observation.

The current runner relies on those lists for both control and decoding, so deleting the copies without replacing that interface would break execution.

### A2. Prototype device-resident delivery and compact observation buffers

Avoid extracting variable-sized spike lists and scalar counts to Python every step.

Use preallocated device-side storage with explicit overflow detection.

An overflow must produce either:

- a reported failure; or
- a controlled fallback.

It must never silently discard spikes.

### A3. Combine numerical operations into fewer GPU kernels

Preserve the existing:

- integration order;
- threshold order;
- synaptic-delivery order;
- reset order;
- delayed events;
- integer synaptic accumulation;
- refractory behavior;
- silencing;
- perturbation semantics.

Keep the existing simulators available as comparison backends.

### A4. Only then consider multi-step execution or CUDA graphs

Multi-step execution must preserve:

- external-input timing;
- observation semantics;
- READY behavior;
- completion behavior.

Do not simply poll READY every hundred steps and describe the result as the same execution schedule.

CUDA graphs are not a wrapper that automatically fixes the present loop. CPU synchronization and dynamic shapes must first be addressed.

### Validation for Track A

For this track, the circuit must remain identical.

Compare:

- voltage/current trajectories where applicable;
- spike events under the applicable backend contract;
- decoded transactions;
- complete outputs.

Any floating-point or random-stream differences must be documented rather than described as bit-identical.

### Expected outcome

Lower wall-time cost for the same neural work.

If profiling shows that memory traffic or active-synapse delivery dominates instead, optimize that measured component rather than continuing a Python-overhead hypothesis.

# Track B: specialize fixed-operation cells before redesigning the protocol

**Files:** `drosophilos/lib/kernel.py`, particularly `add_alu_logic_tokens`, `_unit_rails`, their callers, and `load_pipeline_image`.

The current ALU builder constructs the adder, overflow logic, AND, OR, XOR, and pass-through paths, then selects a result — even though a resident cell’s operation is fixed.

STORE’s pass-through path also calls this general builder.

Add an experimental generic-versus-specialized datapath option and retain the generic implementation for comparison.

### B1. Start with AND, OR, XOR, and MOV

Instantiate only:

- the required operation;
- the required flag outputs.

Remove:

- unused datapaths;
- unnecessary unit-selection hardware.

Update image initialization so it does not assume removed unit-select neurons still exist.

Keep the external:

- request;
- sampling;
- staged commit;
- completion;
- reset;
- fault behavior

unchanged in the first version.

In particular, generate valid flags on every transaction. Do not replace a per-transaction zero flag with a constant level.

### B2. Then test direct LOAD/STORE data paths

Do not shorten STORE completion until a neural dependency guarantees the RAM write has finished.

The current implementation explicitly relies on the longer cell completion occurring after the write. Removing a general ALU could otherwise remove timing that currently protects the memory operation.

### B3. Test specialized cells rigorously

Test:

- exhaustive small-width values;
- wider boundary cases;
- wider random cases;
- repeated identical values;
- alternating values;
- stalled consumers;
- fan-out;
- reset/restart behavior.

Follow these with the existing noisy composition campaign.

Measure separately:

- neuron count;
- edge count;
- spike traffic;
- neural latency;
- initiation interval;
- wall time.

A smaller circuit that leaves neural latency unchanged can still be useful, but report that result accurately.

Only after this comparison should the campaign attempt:

- range-proven width reduction;
- retained column intermediates;
- neural predication;
- larger exact blocks with fewer handshakes.

Those are separate experiments. Do not combine them into the first specialization patch.

### Expected outcome

Fewer simulated components without initially changing the transaction protocol.

The magnitude of any frame-time improvement remains unmeasured until the matched campaign is run.

## 5. Turn the multiplier regression into a bounded diagnostic task

Keep the array multiplier as the default.

The current results show a standalone throughput improvement from the pipelined multiplier but worse full-workload behavior. The cause is not identified.

Run array and pipelined variants on an identical:

- input stream;
- surrounding kernel;
- source;
- backend;
- copy count;
- dtype;
- pacing mode.

Capture selected events from:

- the multiplier;
- its immediate producers;
- its immediate consumers.

Capture:

- request arrival;
- START;
- operand sampling;
- stage completion;
- commit;
- DONE;
- consumer readiness.

Separate four quantities:

1. time to the first result;
2. intervals between subsequent results;
3. time waiting for downstream consumers;
4. time draining the phase.

Use enough tokens to distinguish pipeline fill from sustained throughput.

Do not change timing constants while collecting this comparison.

First determine whether the loss comes from:

- producer starvation;
- downstream backpressure;
- operand retention;
- phase boundaries;
- the multiplier itself.

Correct the campaign report’s reference to inter-commit intervals in `doom4_24na_h200.json`: that saved file contains aggregate counters, not the event history needed for this diagnosis.

### Required output

Produce:

- an event-level explanation;
- a minimal reproducer.

A further long run that merely confirms “pipelined is slower” does not answer the open question.

## 6. Preserve the reliability campaign, but distinguish correctness from throughput

Run tick-failure diagnosis in parallel with the performance work.

Replay a recorded failing seed/copy and capture only the relevant:

- request;
- repair;
- clear;
- completion circuitry.

Turn the localized failure into a regression test before changing its mechanism.

Keep the established 100-copy mix-B comparison for handshake changes.

Add independent seeds after the original reproducer passes.

Report separately:

- requested transactions;
- completed transactions;
- wrong values;
- duplicates;
- detected refusals;
- unfinished transactions.

A faster run that silently drops work is not faster execution.

Classify a run stopped by its resource limit as **truncated**, not automatically stalled.

Establish stall thresholds from the tested contract and workload.

Complete the 1,000-tick canonical-state comparison on the explicitly named world-update program.

A toy tick result must remain labeled as such rather than being generalized to `doom4`.

Keep timestep refinement and CUDA transaction-level certification as explicit outstanding validation work.

## 7. Revise the campaign recommendation and final report

Replace Section 6’s recommendation with:

> Write up the demonstrated feasibility result now. In parallel, establish an instrumented performance baseline, improve simulator execution without changing the neural computation, and test operation-specific exact datapaths. Diagnose the multiplier regression and remaining tick failures with bounded traces. Use the resulting complete-frame measurements to determine the scope of E2 and any later cluster expansion.

Change the speed discussion from:

> wall time is GPU throughput

to a distinction between:

- **measured neural workload cost**; and
- **not-yet-profiled simulator cost**.

The current aggregate timings do not establish their detailed attribution.

Keep the E2 estimate of 10–50× clearly labeled as a proposed gain, not a measured result.

Begin E2 with one bounded primitive and include:

- encoding;
- settling;
- decoding;
- uncertainty detection;
- neural fallback

in its benchmark.

Do not replace the exact renderer until the complete-path comparison supports doing so.

## 8. Required final comparison

The campaign’s final comparison should contain four configurations:

| Configuration | Purpose |
|---|---|
| Original simulator + original circuits | Baseline |
| Optimized simulator + original circuits | Measures simulator-only improvement |
| Original simulator + specialized circuits | Measures circuit-only improvement |
| Optimized simulator + specialized circuits | Measures combined end-to-end improvement |

For each configuration, report:

- first-frame latency;
- subsequent frame intervals;
- neural time;
- wall time;
- wall/neural ratio;
- circuit neuron count;
- circuit edge count;
- spike count or bounded spike-traffic metric;
- wrong outputs;
- missing outputs;
- duplicates;
- invalid outputs;
- faults;
- timeouts;
- truncation status.

## 9. Immediate implementation order

The first code changes should be:

1. **Fix benchmark/backend selection.**
2. **Add instrumentation and frame-completion timing.**
3. **Add a fixed small performance campaign.**
4. **Profile the current Torch simulator.**
5. **Implement the observation abstraction.**
6. **Prototype device-resident spike delivery/observation.**
7. **Implement specialized fixed-operation ALU datapaths.**
8. **Diagnose the pipelined multiplier with event-level timing.**
9. **Re-run the matched small complete-render suite.**
10. **Only then run the historical-size confirmation workloads.**
11. **Use those results to decide the scope and priority of E2.**

The next result should be an attributable before-and-after speed measurement — not another larger image or a new gameplay feature.
