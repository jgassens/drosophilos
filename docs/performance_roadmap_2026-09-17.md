# Performance-first roadmap addendum — 17 September 2026

**Status: proposed priorities, not a measured acceleration.** Audit baseline:
`c6b9aa88118b1f26f58a7b414945447cb35d56b2`. This review inspected source code and
recorded results; it did not rerun the H200 campaigns. The existing
[campaign report](campaign_report_2026-09-17.md) is already committed.

This addendum changes the proposed order of work, not the scientific requirements in
[the plan](plan.md) or [specification](spec.md). Game update and rendering must remain
neural. The host may integrate neuron equations, transport spikes, transduce input,
load the image, and display already-computed pixels. Host arithmetic that replaces a
neural operation is not a simulator optimization under this contract.

## 1. Progress against the original plan

| Stage | Evidence-supported status | Remaining requirement |
| --- | --- | --- |
| 0 / A1 | Reference/simulator agreement and primitive/channel campaigns reported | Timestep refinement and backend-specific transaction certification remain explicit work |
| A2 / B | Arithmetic, memory, control and compiled input-dependent demonstrations exist | Preserve their semantics and reliability while optimizing; demonstrations do not establish the entire proposed language/runtime |
| C | Two-node channel demonstrations | Epochs, credits, corruption, backpressure and broader recovery |
| D | State-update kernels and short correct sequences | 1,000-tick canonical-state campaign, protected voting and idempotent commit log |
| E1 | Exact pixels for the tested examples, including runtime ray casting, textures and a chasing sprite | This is not the full planned minidoom feature set or an interactive renderer |
| F | Single-GPU batched copies | Actual multi-GPU execution, architectural checkpoints and recovery |
| F2 | Neural phase order | Token dispatch and the remaining host control plane |
| E2 / G | Not started in the campaign report | Population acceleration / self-hosted compilation |
| H0 / H | Constrained motifs and interference measured; negative live-surround results | Full-workload Profile 2 execution has not been demonstrated |

The successful large workloads currently carry four labels: **Profile 3, isolated
execution, hybrid orchestration, external compilation**. The measured placement and
interference failures concern these circuits, the embedding method, and the stated
model parameters. They are not a proof that all computation on fly wiring is impossible.
See the campaign report, `a3_kernels.md`, and `RESULTS.md` for the underlying experiments.

## 2. Establish two separate performance budgets

These figures are arithmetic on the saved JSON reports, not new runs:

| Workload | Frames / size / copies | Neural seconds per requested frame | Run wall time per requested frame | Wall seconds / neural second |
| --- | --- | ---: | ---: | ---: |
| `doom2_40_h200.json` | 2 / 40 x 25 / 8 | 991.01 | 14,179.88 s = 3.94 h | 14.31 |
| `doom4_24na_h200.json` | 2 / 24 x 15 / 8 | 1,355.36 | 32,373.43 s = 8.99 h | 23.89 |

Sources: [doom2](a2/doom2_40_h200.json), [doom4](a2/doom4_24na_h200.json).
All five saved error counters are zero in these two reports. These are different
workloads and sizes, not a controlled comparison of optimization variants.

`render_doom.py` starts its wall timer after compilation, reference generation and
circuit construction. Its run includes scheduled updates and may include progress-time
image writes. Dividing this aggregate by frame count does **not** measure first-frame,
steady-state frame, or input-to-display latency. Report those separately in future runs.

For these aggregate timings:

    run wall time = simulated neural time x simulator wall/neural ratio

The first factor includes circuit critical paths, pipeline initiation intervals,
workload partitioning and orchestration. The second includes numerical integration,
spike delivery, synchronization, Python observation and other timed work. Both need
measurement; neither can be inferred from GPU model alone.

For scale only, 60 seconds per frame would require aggregate reductions of approximately
236x and 540x for these runs; one second would require approximately 14,180x and 32,373x.
These are illustrative planning comparisons, not agreed requirements or predictions.
Even an idealized 50x reduction of the *entire* current run would leave about 4.7 and
10.8 minutes per frame. E2's proposed 10–50x per-pixel gain is not a measured end-to-end gain.
The original 35 Hz tick ambition is unchanged, not asserted achievable by this addendum.

The report's roughly one-minute neural-time floor is an estimate for the current
architecture under parallelization. It is neither a universal neuronal limit nor an
unavoidable minute of wall time: a different simulator can, in principle, run faster
than the simulated clock. More copies alone do not remove sequential dependencies.

## 3. Concrete findings from the current execution path

### A. The production simulator synchronizes with the CPU in its inner loop

In `drosophilos/sim/lif_torch.py`, `TorchSim.step()` calls `torch.nonzero(spk)`,
copies spike node and neuron indices with `.cpu().numpy()`, extracts
`int(lens.sum())`, and builds variable-sized delivery arrays using
`repeat_interleave`. This happens at a 0.1 ms model timestep. The CUDA behavior of
`nonzero` is documented as host-device synchronization; `repeat_interleave` documents
that a supplied output size can avoid synchronization for shape calculation.

`drosophilos/lib/kernel.py::run_pipeline_batched` then visits schedules and observed
spikes in Python on every step, including `np.isin` over the observed spike arrays.
It explicitly instantiates `TorchSim`; the inspected rendering path does not use an
already-fused CUDA simulator. The old full-trace sorting problem is already fixed and
must not be presented as a new discovery.

These are concrete optimization candidates, **not a measured attribution of the total
runtime**. A profiler must distinguish launch overhead, synchronization, memory traffic,
active-synapse delivery, and host bookkeeping. CUDA graphs are not a one-line remedy:
CPU-dependent control and dynamic allocation/shapes must first be addressed.

Primary implementation references:
- [PyTorch nonzero](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html)
- [PyTorch repeat_interleave](https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html)
- [PyTorch CUDA graphs and constraints](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

### B. Current saved reports cannot diagnose the multiplier regression

The report proposes inspecting inter-commit intervals in
`docs/a2/doom4_24na_h200.json`, but that file contains aggregate counters and timings,
not commit timestamps. The runner has `(step, value)` output lists in memory and an
optional narrow spike capture, but `render_doom.py` does not save the timing evidence
needed to separate pipeline fill, steady-state throughput, backpressure and phase tails.

Keep the array multiplier as the baseline. Compare array and pipelined variants on an
identical source, state, input schedule, resolution, copy count, backend and dtype.
Trace selected cell starts, operand sampling, commits and downstream readiness. A
standalone throughput improvement is not evidence of a frame-time improvement.

### C. The one-copy benchmark takes a different backend than its flags suggest

`render_doom.py` calls `run_pipeline` when `B == 1`, without passing `--device` or
`--fp32`; that path uses the reference simulator. It can still print the requested
`a.device` in its summary. For `B > 1`, it calls the batched Torch path. Before using
one-copy measurements to project GPU scaling, make backend selection explicit and save
both requested and effective backend/dtype. This addendum records the issue; it does
not change the runner.

### D. Exact neural computation has avoidable work worth investigating

The current compiler turns conditional arms into computation followed by a select.
Consequently an expensive invisible-sprite arm can still be computed. `doom4.c` also
uses 16-bit values for quantities with smaller value ranges and reloads column-derived
quantities in the pixel pass. These are candidates for neural predication, proven
width reduction, retained column state and coarser exact circuit blocks. Their net
benefit must include extra storage, control, decoding and reset costs.

## 4. Proposed implementation order

### P0 — Make the benchmark explain where time goes

Before another large frame campaign:

- Record source and commit hashes, source parameters, inputs, effective backend, dtype,
  multiplier, pacing, neuron/edge counts, copy count, model parameters and timing scope.
- Save build/startup time, time to first completed frame, each subsequent frame completion,
  input-to-state commit and input-to-visible-frame latency. Keep wall and neural clocks.
- Record compact per-stream/cell commit intervals and selected handshake states. Bound
  capture memory and report dropped records explicitly; do not dump all spikes by default.
- Profile matched short kernel and render samples. Fix the one-copy backend labeling and
  retain stalled/incorrect runs in the report instead of treating missing counters as zero.

Expected outcome: a reproducible baseline separating simulator cost from neural critical
paths and a localized multiplier regression. If overhead is small, move effort to measured
integration/delivery or circuit costs rather than assuming a Python rewrite will suffice.

### P1 — A faster simulator with unchanged model semantics

Retain `RefSim` and current `TorchSim` as comparison backends. Prototype a fused device
step and GPU-resident spike-delivery/observation path. Transfer compact observed events
rather than every internal spike. Use bounded queues with visible overflow failure.
Only batch multiple timesteps when external-input scheduling and observation semantics
are preserved; account for control timing explicitly rather than hiding it in a chunk.

Preserve integration/threshold/delivery/reset order, delayed events, refractory behavior,
integer synaptic accumulation, silencing, bias and the chosen perturbation contract.
Validate deterministic trajectories/spikes where promised and transaction outcomes on
all supported dtype/backend combinations. Any changed random-number stream needs a
stated coupling or statistical comparison, not an unsupported bit-identical claim.

Expected outcome: lower wall/neural ratio at identical neural work. If memory or synaptic
traffic dominates, optimize that bottleneck. If gains are modest, report them; do not
change neuron time constants or drop recurrent activity to manufacture acceleration.

### P2 — Reduce exact neural work in both rendering and world update

Use matched benchmarks to evaluate:

- Range-proven widths, constant propagation and narrower ROM/data paths. Preserve
  wrapping arithmetic and high-bit sign tests; values cannot simply be truncated.
- Column-invariant calculations retained in neural registers, neural lookup/recurrence
  alternatives, and reduced repeated work. Count the added storage and read controls.
- Neural branch predication with explicit skip/completion behavior, so omitted work does
  not leave a join waiting for a token that will never arrive.
- Coarser exact neural blocks with one external transaction contract and correctly
  sequenced internal computation, instead of paying a full storage/reset handshake at
  every trivial operation. This is circuit synthesis, not host evaluation of the block.

Measure latency, initiation interval, neuron/edge/spike counts, reset and backpressure
cost, and noisy correctness. Include the tick critical path: a faster pixel kernel alone
cannot make a 30–60 second neural update interactive. Do not enable the pipelined
multiplier globally unless matched end-to-end runs support it.

Expected outcome: lower neural work and/or smaller resident circuits. If added control
or storage cancels a local gain, retain the measured alternative rather than assuming
fewer arithmetic operations implies a faster neural program.

### P3 — E2 as a measured accelerator, not the only proposed solution

Begin with a bounded transform or interpolation primitive. Include encoding/decoding,
settling time, population size, noise, error propagation, boundary detection and fallback
in the budget. State, addresses, branch decisions and commits remain exact. Any runtime
fallback required by the neural-execution claim must also execute neurally.

Keep the exact E1 path as the reference. Approximate-output experiments must say so and
report image errors; exact-output experiments must establish a valid decision margin or
fall back. Report fallback frequency and end-to-end speed, not only primitive throughput.
A 10–50x gain remains a hypothesis until demonstrated on matched workloads.

## 5. Work that remains required but should not lead the speed effort

Complete Stage D's 1,000-tick test and resolve its visible fail-stops as correctness work;
perform it alongside performance work when affordable. Retain C/F2 protocol obligations.
Defer substantial multi-GPU expansion and new gameplay features until a capacity and
latency budget justifies them. Defer self-hosted compilation as a separate result, not a
rendering remedy. Preserve the feasibility write-up and constrained-wiring measurements.

Further full-graph claims require a documented stable active regime and a model for
non-spiking classes. Adaptation, depression or inhibition would be new model variants
requiring their own evidence, not silently equivalent versions of the present model.

The reported zero silent errors in 4,000 kernel outputs and the much larger channel
campaign have different scopes. Neither establishes a 10^-10 kernel error rate. No
performance change bypasses the existing perturbation and exact-reference comparisons.

## 6. Changes delivered with this review

This branch updates navigation/current status and adds a standard-library audit tool:

```bash
python -m drosophilos.bench.render_budget \
  docs/a2/doom2_40_h200.json docs/a2/doom4_24na_h200.json \
  --target-seconds 60
python -m unittest discover -s tests -p 'test_render_budget.py' -v
```

The helper validates counts/timings, reports aggregate budgets, preserves unknown error
counters, and marks target ratios as arithmetic rather than predictions. Its nine unit
tests passed in a local isolated workspace. No simulator, neural circuit, benchmark
runner, historical result, or original milestone requirement is changed. The full
repository suite and GPU performance/correctness campaigns were not run in this review.
