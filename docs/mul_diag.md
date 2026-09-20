# §5 — why the pipelined multiplier doubles the pixel cost (event-level)

`docs/perf_campaign.md` §5 asked for an event-level explanation and a minimal reproducer of
the multiplier regression: the pipelined multiplier is 2.1× faster per token on the
perspective kernel alone (Juno 412136, 3.2 s against 6.9 s), yet a doom4 pixel cost 63 s with
it against 30 s with the array multiplier (a3_kernels §11.2). Tool: `bench/mul_diag.py`
(captures every multiplier cell's request → START → ACT^d → completion → commit → DONE and
its consumers' requests on node 0; `tests/test_mul_diag.py`). Runs: Juno 413027
(perspective, pipelined, 16 tokens), 413025 (doom4 8 × 5 × 1 frame, array), 413026 (same,
pipelined). Records: `docs/perf/muldiag_*.json`.

## 1. Standalone (perspective kernel, pipelined multiplier, 16 tokens)

Each of the 16 rows cycles in ~1.5 s: go 72–114 ms, compute 590 ms, commit wait 80 ms,
commit + reset ("drain") 529 ms, ~70–320 ms waiting for its input. 15 of 16 tokens entered
row 0 before the previous token left row 15: the rows genuinely overlap, which is where the
standalone 2.1× comes from. A third of every row's cycle is the handshake (drain).

## 2. Inside doom4's pixel pass (array multiplier, 40 pixels per multiplier)

| phase of a pixel token in `c2_9_mul` / `c2_40_mul` | ms | share |
|---|---|---|
| waiting for its input to arrive (previous DONE → last request) | ~20,000 | 76 % |
| go (request → START) | 148 | < 1 % |
| compute (16-bit array multiply) | 5,300–5,500 | 21 % |
| waiting for the consumer (completion → commit) | 80 | < 1 % |
| commit + reset (commit → DONE) | 467 | 2 % |
| **period per pixel** | **26,200** | |

Every cell of the pixel pass — the multipliers, their producers (`c2_7_sub`, `c2_8_load`,
`c2_38_sub`, `c2_39_load`) and consumers — has the same 26.2 s period, and the two operands of
`c2_9_mul` arrive 25.5 s apart for the same token (`c2_8_load` first, `c2_7_sub` last). The
pass is not throughput-bound by any cell; it is running **one pixel at a time**: the period
equals the path latency of the whole pass.

## 3. Mechanism: a late reader holds its producer's commit (operand retention)

A producer commits its next value only when every consumer has started on the previous one
(`lib/kernel.py`, `gate_commit`: "commit once every consumer has started on the previous
value"). In the compiled pixel pass (55 cells, depth 19) early values are read late:
`c2_29_load` (depth 1) by `c2_54_sel` (depth 19, the output) and `c2_44_sel` (depth 9);
`c2_1_shr` (depth 0) by cells at depths 3–5; `c2_4_load` (depth 1) at depths 8–9;
`c2_28_sel` (depth 13) at depths 16–19. So `c2_29_load` cannot accept pixel k+1 until the
last cell has started pixel k, `c2_0_and` behind it cannot either, and the pixel input
register waits for the whole pass. One token in flight; period = latency.

That also explains the multiplier regression without any multiplier defect: when the period
is the path latency, **latency is the cost and throughput is irrelevant**. Confirmed by the
pipelined capture (Juno 413026, same 8 × 5 × 1 frame, 2,195 s neural against 1,407 s):

| pipelined multiplier inside the pixel pass (each of 16 rows, 40 pixels) | ms |
|---|---|
| waiting for its input | ~41,000 |
| go / compute / consumer wait / commit + reset | 72 / 580–760 / 80 / 529 |
| **period per pixel (every row, both multipliers)** | **42,400** (array: 26,200) |
| tokens entering row 0 before the previous left row 15 | **0 of 40** (`c2_40_mulp`), 1 of 40 (`c2_9_mulp`); standalone: 15 of 16 |

The rows never overlap inside the pass, so the 16-row multiplier is pure latency: ~1.4 s per
row × 16 ≈ 23 s from request to DONE against ~6 s for the array. The two multipliers sit on
parallel branches (`c2_9_mul` at depth 4, `c2_40_mul` at depth 5, converging at the output
selects), so the path grows by one multiplier's difference, 26.2 → 42.4 s per pixel (+16 s,
1.62×). The 63 s against 30 s of a3_kernels §11.2 was the 24 × 15 render, whose longer pass
multiplies the same per-token latency by more pixels per column.

## 4. Minimal reproducer (`tests/test_pipeline_retention.py`, RefSim, 4-bit)

Chain A → B → C → D with D also reading A directly ("held"), against the same function with
A carried to D through two MOV copies ("relayed"). Same outputs. Measured 2026-09-20:

| | held | relayed |
|---|---|---|
| neurons | 6,472 | 9,400 |
| A's compute | 365 ms | 365 ms |
| A's commit wait | **1,591 ms** | 157 ms |
| period (A and D) | **2,550 ms** | **1,117 ms** |

The wait is at A's commit — its value is ready after 365 ms and sits until D has started on
the previous token; relaying A's value alongside the chain restores overlap and gives 2.3× the
throughput for the same arithmetic, at the cost of one copy cell per hop.

## 5. What follows (not done here — the orders keep it a separate experiment)

The pixel pass is limited by retention, not by the multiplier: relaying long-lived values
(a MOV chain per hop, or a register between a producer and a distant reader) would let pixels
overlap, with a ceiling of period ≈ slowest cell (~6 s with the array multiplier, ~1.5 s per
row with the pipelined one — at which point the pipelined multiplier wins again). That is the
"retained column intermediates" item of `docs/perf_campaign.md` §4 Track B's later list; it
needs the compiler to insert the copies and the matched campaign to measure it. The array
multiplier stays the default until then.
