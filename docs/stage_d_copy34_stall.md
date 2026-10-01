# Tick stall diagnostic: copy 34

## Classification

Captured window: steps **1,611,372–4,261,465**, inclusive. Transaction ordinals count events in this window; earlier history is unavailable.

**`c5_sub`** STARTed transaction 6 at step 1,909,328, but its captured stage completion never rose: **datapath**.

## Artifact and threshold

- Dump: `data/stage_d/mixB_s108_c100_replay40_copy34.npz` (compact/role-filtered; 41,628,289 spikes, 5,175 firing neuron ids; steps 1,611,372–4,261,465).
- Rebuilt kernel: 29,375 neurons / 52,808 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 24,826 steps (2.4826 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Blocked cell and neighbour timeline

The rows retain the transaction before the failure, the failed transaction, and the immediately adjacent producer/consumer work visible in the dump.

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c4_sel` / 4 | c2_sel 1,832,172; c3_xor 1,841,802 | 1,843,602 | 1,844,183 | 1,846,650 | 1,848,855 | 1,852,980 |
| `c4_sel` / 5 | c2_sel 1,886,517; c3_xor 1,896,220 | 1,898,013 | 1,898,591 | 1,901,028 | 1,903,231 | 1,907,396 |
| `c4_sel` / 6 | c2_sel 1,951,970; c3_xor 1,961,668 | 1,963,478 | 1,964,063 | 1,966,520 | — | — |
| `c5_sub` / 4 | c9_sel 1,770,515; c4_sel 1,787,712 | 1,789,551 | 1,790,140 | 1,795,321 | 1,796,086 | 1,800,210 |
| `c5_sub` / 5 | c9_sel 1,821,936; c4_sel 1,853,064 | 1,854,909 | 1,855,492 | 1,862,377 | 1,863,148 | 1,867,261 |
| `c5_sub` / 6 | c9_sel 1,890,297; c4_sel 1,907,480 | 1,909,328 | 1,909,917 | — | — | — |
| `c6_and` / 3 | c5_sub 1,747,640 | 1,748,329 | 1,748,901 | 1,753,650 | 1,754,424 | 1,758,569 |
| `c6_and` / 4 | c5_sub 1,800,292 | 1,800,979 | 1,801,551 | 1,805,028 | 1,805,802 | 1,809,932 |
| `c6_and` / 5 | c5_sub 1,867,343 | 1,868,035 | 1,868,603 | 1,873,332 | 1,874,104 | 1,878,289 |
| `c9_sel` / 3 | c7_add 1,713,372; c8_sub 1,714,437; c6_and 1,758,656 | 1,760,475 | 1,761,061 | 1,763,363 | 1,766,341 | 1,770,429 |
| `c9_sel` / 4 | c7_add 1,781,654; c8_sub 1,782,214; c6_and 1,810,018 | 1,811,823 | 1,812,403 | 1,814,682 | 1,817,662 | 1,821,850 |
| `c9_sel` / 5 | c7_add 1,832,970; c8_sub 1,834,090; c6_and 1,878,376 | 1,880,203 | 1,880,787 | 1,883,075 | 1,886,045 | 1,890,212 |

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 120 | 94 | 0 | 0 | 1 (runner: 0) | 26 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
