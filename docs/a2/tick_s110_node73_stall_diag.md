# Tick stall diagnostic: copy 73

## Classification

The first blocked cell is **`c2_sel`**, at its **1st START**. This is a **request / repair / clear** failure, not a completion failure: false request latch **11845, `c2_sel.req.c1_andr0.u`** remains lit after the next `c1_and` request's clear. No preceding repair pulse is present in the captured repair roles. Both request rails are consequently live; false vetoes the go chain, so START, ACT^d, stage completion, commit, and DONE do not occur.

## Artifact and threshold

- Dump: `data/a2/tick_s110_node73.npz` (role-filtered; 24,402,821 spikes, 3,343 firing neuron ids; steps 23–899,999).
- Rebuilt kernel: 28,669 neurons / 51,324 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 20,424 steps (2.0424 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Request / repair / clear evidence

| Signal | Neuron | Observed step(s) |
|---|---:|---:|
`c2_sel.req.c1_and` false u | 11845 | live through 899,999 |
`c2_sel.req.c1_and` false v | 11846 | partner of the stuck latch |
`c2_sel.req.c1_and` true u | 11847 | 35,002 onward |
repair edge | 11,979 | **—** |
received / clear trigger | 11850 | 35,003 |
clear inhibitory train | 11853 | 35,103, 35,153, 35,197 |
START | 2422 | no 1st pulse |
ACT^d | 11972 | no pulse after the blockage |
stage completion | 2048 | no completion after the blockage |
commit | 27319 | no commit after the blockage |
DONE | 2384 | no DONE after the blockage |

False-u train intervals are `first–last (spikes)`: 24–899,993 (21943).

## Blocked cell and neighbour timeline

The rows retain the transaction before the failure, the failed transaction, and the immediately adjacent producer/consumer work visible in the dump.

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c0_add` / 1 | c4_sel 25; input 11,936 | 13,815 | 14,404 | 18,329 | 19,864 | 23,911 |
| `c1_and` / 1 | c0_add 23,992 | 24,710 | 25,283 | 30,142 | 30,923 | 34,922 |
| **`c2_sel` / 1 (blocked)** | c0_add 23,992; c1_and 35,002 | **—** | — | — | — | — |

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 0 | 0 | 0 | 0 (runner: 0) | 16 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
