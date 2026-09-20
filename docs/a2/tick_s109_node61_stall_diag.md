# Tick stall diagnostic: copy 61

## Classification

**`c9_sel`** completed its stage at step 478,539, but commit never fired: **consumer holds / commit gating**.

## Artifact and threshold

- Dump: `data/a2/tick_s109_node61.npz` (role-filtered; 14,684,358 spikes, 4,874 firing neuron ids; steps 22–899,999).
- Rebuilt kernel: 28,439 neurons / 51,071 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 135,764 steps (13.5764 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Blocked cell and neighbour timeline

The rows retain the transaction before the failure, the failed transaction, and the immediately adjacent producer/consumer work visible in the dump.

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c5_sub` / 6 | c9_sel 320,752; c4_sel 341,179 | 343,069 | 343,634 | 347,694 | 348,474 | 352,637 |
| `c5_sub` / 7 | c9_sel 375,534; c4_sel 395,732 | 397,632 | 398,201 | 402,043 | 402,816 | 406,986 |
| `c5_sub` / 8 | c9_sel 428,773; c4_sel 451,412 | 453,304 | 453,867 | 457,770 | 458,549 | 462,725 |
| `c6_and` / 6 | c5_sub 352,720 | 353,439 | 354,022 | 358,655 | 359,432 | 363,409 |
| `c6_and` / 7 | c5_sub 407,070 | 407,782 | 408,369 | 411,859 | 412,634 | 416,677 |
| `c6_and` / 8 | c5_sub 462,809 | 463,532 | 464,118 | 468,780 | 469,556 | 473,591 |
| `c7_add` / 6 | c9_sel 266,159; input 259,843 | 267,594 | 268,152 | 272,019 | 311,868 | 316,026 |
| `c7_add` / 7 | c9_sel 320,747; input 314,610 | 322,182 | 322,742 | 326,842 | 366,511 | 370,672 |
| `c7_add` / 8 | c9_sel 375,530; input 369,332 | 376,972 | 377,533 | 381,610 | 419,797 | 423,864 |
| `c8_sub` / 6 | c9_sel 266,165; input 259,846 | 267,624 | 268,175 | 272,241 | 311,837 | 315,885 |
| `c8_sub` / 7 | c9_sel 320,753; input 314,615 | 322,206 | 322,760 | 326,780 | 366,507 | 370,591 |
| `c8_sub` / 8 | c9_sel 375,535; input 369,335 | 376,984 | 377,542 | 381,557 | 419,771 | 423,806 |
| `c9_sel` / 6 | c7_add 316,110; c8_sub 315,968; c6_and 363,493 | 365,363 | 365,934 | 368,351 | 371,417 | 375,448 |
| `c9_sel` / 7 | c7_add 370,756; c8_sub 370,674; c6_and 416,761 | 418,636 | 419,199 | 421,590 | 424,657 | 428,686 |
| `c9_sel` / 8 | c7_add 423,947; c8_sub 423,889; c6_and 473,675 | 475,554 | 476,126 | 478,539 | — | — |
| `c10_xor` / 6 | c9_sel 320,754; c4_sel 341,172 | 343,048 | 343,639 | 347,148 | 347,946 | 352,023 |
| `c10_xor` / 7 | c9_sel 375,534; c4_sel 395,725 | 397,612 | 398,200 | 401,780 | 402,592 | 406,653 |
| `c10_xor` / 8 | c9_sel 428,775; c4_sel 451,409 | 453,289 | 453,877 | 457,452 | 458,255 | 462,328 |

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 16 | 8 | 0 | 0 | 0 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
