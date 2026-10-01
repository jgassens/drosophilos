# Tick stall diagnostic: copy 18

## Classification

Captured window: steps **0–2,473,209**, inclusive. Transaction ordinals count events in this window; earlier history is unavailable.

**`c3_xor`** STARTed transaction 2 at step 101,602, but its captured stage completion never rose: **datapath**.

## Artifact and threshold

- Dump: `data/stage_d/mixB_s108_c100_replay40_copy18.npz` (compact/role-filtered; 40,161,515 spikes, 5,277 firing neuron ids; steps 23–2,473,209).
- Rebuilt kernel: 29,375 neurons / 52,808 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 21,354 steps (2.1354 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Blocked cell and neighbour timeline

The rows retain the transaction before the failure, the failed transaction, and the immediately adjacent producer/consumer work visible in the dump.

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c2_sel` / 1 | c0_add 23,934; c1_and 35,019 | 36,858 | 37,438 | 40,010 | 41,513 | 45,574 |
| `c2_sel` / 2 | c0_add 79,123; c1_and 90,222 | 92,059 | 92,633 | 95,227 | 96,745 | 100,805 |
| `c3_xor` / 1 | c2_sel 45,660 | 46,379 | 46,948 | 50,679 | 51,453 | 55,558 |
| `c3_xor` / 2 | c2_sel 100,893 | 101,602 | 102,170 | — | — | — |
| `c4_sel` / 1 | c2_sel 45,657; c3_xor 55,640 | 57,474 | 58,057 | 60,542 | 62,780 | 66,878 |

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 120 | 3 | 0 | 0 | 0 (runner: 0) | 117 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
