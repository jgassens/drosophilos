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

| workload, copies | Q1 `TorchSim` + generic | Q2 `FastSim` + generic | gain |
|---|---|---|---|
| doom2, 1 | 9,126 s (first frame 2,934 s) | 2,997 s (first frame 969 s) | 3.0× |
| doom2, 8 | 3,775 s (1,053 s) | 3,196 s (898 s) | 1.2× |
| doom4, 1 — same 3,600 s of neural work, both cut by the cap (22 pixels unfinished in each) | 27,520 s (10,193 s) | 13,105 s (4,899 s) | 2.1× |
| doom4, 8 (11.5 M neurons) | 16,522 s (4,224 s) | 16,764 s (4,309 s) | 1.0× |

Pixels identical wherever both delivered them; frame completion stamped at the last valid
pixel by the `on_output` observer, not by a progress poll. Q3 at render level is not run: the
primitive columns show specialization does not move wall time, and a 16-hour `TorchSim`
render would only repeat that. Q4 at render level (`FastSim` + specialized) is queued.

Above ~1 M neurons the step is bound by the arithmetic, which both simulators pay, so the
gain falls from 5.5× (kernels) to 3.0× (664k), 2.1× (1.44 M), 1.2× (5.3 M) and 1.0× (11.5 M,
doom4 × 8 copies: the two simulators are equal); the next simulator lever is that arithmetic —
the sparse product's memory traffic, float32 — not the host.
