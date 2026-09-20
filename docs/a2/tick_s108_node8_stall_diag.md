# Tick stall diagnostic: copy 8

## Classification

The first blocked cell is **`c4_sel`**, at its **3rd START**. This is a **request / repair / clear** failure, not a completion failure: false request latch **14087, `c4_sel.req.c2_selr0.u`** remains lit after the next `c2_sel` request's clear. No preceding repair pulse is present in the captured repair roles. Both request rails are consequently live; false vetoes the go chain, so START, ACT^d, stage completion, commit, and DONE do not occur.

## Artifact and threshold

- Dump: `data/a2/tick_s108_node8.npz` (role-filtered; 15,655,765 spikes, 4,874 firing neuron ids; steps 22–899,999).
- Rebuilt kernel: 28,439 neurons / 51,069 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 21,424 steps (2.1424 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Request / repair / clear evidence

| Signal | Neuron | Observed step(s) |
|---|---:|---:|
`c4_sel.req.c2_sel` false u | 14087 | live through 899,999 |
`c4_sel.req.c2_sel` false v | 14088 | partner of the stuck latch |
`c4_sel.req.c2_sel` true u | 14089 | 164,436 onward |
repair edge | 14,229 | **—** |
received / clear trigger | 14092 | 164,439 |
clear inhibitory train | 14095 | 164,535, 164,581, 164,624 |
START | 3524 | no 3rd pulse |
ACT^d | 14225 | no pulse after the blockage |
stage completion | 3150 | no completion after the blockage |
commit | 27432 | no commit after the blockage |
DONE | 3486 | no DONE after the blockage |

False-u train intervals are `first–last (spikes)`: 24–45,935 (1204); 57,489–101,189 (1166); 112,847–899,983 (20919).

## Blocked cell and neighbour timeline

The rows retain the transaction before the failure, the failed transaction, and the immediately adjacent producer/consumer work visible in the dump.

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c0_add` / 1 | c4_sel 24; input 11,876 | 13,757 | 14,335 | 18,242 | 19,771 | 23,921 |
| `c0_add` / 2 | c4_sel 67,010; input 21,620 | 68,448 | 69,023 | 73,486 | 75,008 | 79,160 |
| `c0_add` / 3 | c4_sel 122,311; input 130,089 | 131,970 | 132,551 | 136,755 | 138,288 | 142,433 |
| `c2_sel` / 1 | c0_add 24,005; c1_and 35,170 | 37,070 | 37,658 | 40,042 | 41,597 | 45,687 |
| `c2_sel` / 2 | c0_add 79,244; c1_and 90,405 | 92,305 | 92,893 | 95,277 | 96,829 | 100,954 |
| `c2_sel` / 3 | c0_add 142,517; c1_and 153,780 | 155,683 | 156,272 | 158,694 | 160,243 | 164,355 |
| `c3_xor` / 1 | c2_sel 45,773 | 46,497 | 47,074 | 50,516 | 51,303 | 55,478 |
| `c3_xor` / 2 | c2_sel 101,040 | 101,770 | 102,356 | 105,860 | 106,649 | 110,839 |
| `c3_xor` / 3 | c2_sel 164,441 | 165,169 | 165,753 | 169,233 | 170,025 | 174,192 |
| `c4_sel` / 1 | c2_sel 45,768; c3_xor 55,557 | 57,447 | 58,018 | 60,533 | 62,802 | 66,925 |
| `c4_sel` / 2 | c2_sel 101,035; c3_xor 110,919 | 112,806 | 113,385 | 115,876 | 118,150 | 122,225 |
| **`c4_sel` / 3 (blocked)** | c2_sel 164,436; c3_xor 174,272 | **—** | — | — | — | — |
| `c5_sub` / 1 | c9_sel 26; c4_sel 67,011 | 68,884 | 69,465 | 74,204 | 75,001 | 79,177 |
| `c5_sub` / 2 | c9_sel 102,420; c4_sel 122,311 | 124,197 | 124,775 | 129,176 | 129,975 | 134,140 |
| `c10_xor` / 1 | c9_sel 102,419; c4_sel 67,008 | 103,880 | 104,466 | 107,938 | 108,748 | 112,821 |
| `c10_xor` / 2 | c9_sel 157,251; c4_sel 122,307 | 158,716 | 159,295 | 162,776 | 163,568 | 167,588 |

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 4 | 0 | 0 | 0 | 12 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
