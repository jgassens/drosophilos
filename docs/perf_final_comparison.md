# §8 — the four-configuration comparison

`docs/perf_campaign.md` §8 asks for one table: original simulator + original circuits,
optimized simulator + original circuits, original simulator + specialized circuits, optimized
simulator + specialized circuits. All numbers are one H200 NVL on Juno, identical inputs,
`--no-spike-count`, 16 tokens per block; every row 0 wrong / 0 missing / 0 invalid / 0 faults /
not truncated. Wall ÷ neural, with the wall seconds in brackets.

## Primitive level (Juno 413148 = Q1 + Q2, 413419 = Q3, 413149 = Q4; `docs/perf/`)

| block | Q1 `TorchSim` + generic | Q2 `FastSim` + generic | Q3 `TorchSim` + specialized | Q4 `FastSim` + specialized | neurons generic → specialized |
|---|---|---|---|---|---|
| ADD cell (stays generic) | 5.58× (111 s) | 0.95× (19 s) | 5.65× (112 s) | 0.94× (19 s) | 3,172 → 3,172 |
| AND cell | 5.55× (108 s) | 0.93× (18 s) | 5.60× (108 s) | 0.93× (18 s) | 3,172 → 2,166 |
| XOR cell | 5.54× (105 s) | 0.93× (18 s) | 5.54× (104 s) | 0.95× (18 s) | 3,172 → 2,198 |
| MOV cell | 5.53× (100 s) | 0.93× (17 s) | 5.61× (101 s) | 0.93× (17 s) | 3,172 → 2,070 |
| fan-out kernel | 5.76× (134 s) | 1.01× (23 s) | 5.84× (136 s) | 1.01× (23 s) | 10,888 → 8,908 |
| tick kernel | 5.93× (589 s) | 1.08× (107 s) | 6.07× (601 s) | 1.08× (107 s) | 28,439 → 24,479 |

First-output latency and per-token interval are the same to within 8–16 ms across all four
columns (`docs/track_b.md`). Reading: the simulator change is the whole speed-up (5.5–5.9×);
the circuit change removes a third of the neurons of the supported cells and changes neither
neural time nor wall time at these sizes; the two are independent, so their combination equals
the simulator gain alone.

## Render level (small: 8 × 5 × 3 frames, doom2 664k neurons, doom4 1.44 M; `docs/perf/juno-h200-small*.md`)

| workload, copies | Q1 `TorchSim` + generic | Q2 `FastSim` + generic | Q4 `FastSim` + specialized (Juno 413470) | Q2 → Q4 |
|---|---|---|---|---|
| doom2, 1 | 9,126 s (first frame 2,934 s) | 2,997 s (first frame 969 s) | 2,801 s (906 s) | 1.07× |
| doom2, 8 | 3,775 s (1,053 s) | 3,196 s (898 s) | 2,837 s (794 s) | 1.13× |
| doom4, 1 | 27,520 s (10,193 s), cut by the cap at 3,600 neural s (22 pixels unfinished) | 13,105 s (4,899 s), same cut | 14,260 s (4,547 s), **complete**: 4,215 neural s, all pixels | 1.08× on the first frame |
| doom4, 8 (11.5 M neurons) | 16,522 s (4,224 s) | 16,764 s (4,309 s) | 15,230 s (3,903 s) | 1.10× |

Q4 (`--backend torch-fast --datapath specialized`, 10 h 25 min on one H200): the specialized
datapath takes doom2 from 664k to 582k neurons and doom4 from 1.44 M to 1.30 M (−12 % and
−10 %), and the wall time follows the neuron count and no more: 1.07–1.13× over Q2, the
primitive columns' "specialization does not move wall time" seen at render scale. Its doom4
single-copy run is the first to finish every pixel (the render cap became `max(1 h, 4 ×
estimate)` after the earlier runs), so its 14,260 s covers 4,215 neural s against Q2's
3,600; on the first frame, the like-for-like number, it is 1.08× faster. Neural time per
frame is unchanged by the simulator or the datapath (~480 s doom2, ~1,400 s doom4 single
copy): the frame rate is the kernel's, and §5 says why.

Pixels identical wherever both delivered them; frame completion stamped at the last valid
pixel by the `on_output` observer, not by a progress poll. Q3 at render level is not run: the
primitive columns show specialization does not move wall time, and a 16-hour `TorchSim`
render would only repeat that.

Above ~1 M neurons the step is bound by the arithmetic, which both simulators pay, so the
gain falls from 5.5× (kernels) to 3.0× (664k), 2.1× (1.44 M), 1.2× (5.3 M) and 1.0× (11.5 M,
doom4 × 8 copies: the two simulators are equal); the next simulator lever is that arithmetic —
the sparse product's memory traffic, float32 — not the host.
