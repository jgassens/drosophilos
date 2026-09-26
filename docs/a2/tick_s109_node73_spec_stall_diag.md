# Tick stall diagnostic: copy 73

## Classification

**`c9_sel`** STARTed transaction 4 at step 262,456, but its captured stage completion never rose: **datapath**.

## Artifact and threshold

- Dump: `data/a2/tick_s109_node73_spec.npz` (role-filtered; 22,157,010 spikes, 6,379 firing neuron ids; steps 22–899,999).
- Rebuilt kernel: 25,113 neurons / 44,644 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 21,312 steps (2.1312 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Blocked cell and neighbour timeline

The rows retain the transaction before the failure, the failed transaction, and the immediately adjacent producer/consumer work visible in the dump.

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c5_sub` / 2 | c9_sel 101,280; c4_sel 121,217 | 123,069 | 123,649 | 128,099 | 128,884 | 132,885 |
| `c5_sub` / 3 | c9_sel 155,746; c4_sel 183,901 | 185,753 | 186,334 | 190,459 | 191,244 | 195,248 |
| `c5_sub` / 4 | c9_sel 218,072; c4_sel 238,347 | 240,180 | 240,763 | 244,882 | 245,669 | 249,626 |
| `c6_and` / 2 | c5_sub 132,973 | 133,668 | 134,240 | 138,970 | 139,759 | 143,840 |
| `c6_and` / 3 | c5_sub 195,334 | 196,035 | 196,604 | 201,220 | 202,009 | 206,110 |
| `c6_and` / 4 | c5_sub 249,710 | 250,407 | 250,982 | 255,691 | 256,476 | 260,555 |
| `c7_add` / 2 | c9_sel 101,284; input 21,777 | 102,711 | 103,296 | 107,389 | 108,179 | 112,418 |
| `c7_add` / 3 | c9_sel 155,750; input 129,106 | 157,176 | 157,770 | 161,780 | 162,571 | 166,797 |
| `c7_add` / 4 | c9_sel 218,077; input 183,579 | 219,493 | 220,078 | 224,276 | 225,063 | 229,294 |
| `c8_sub` / 2 | c9_sel 101,277; input 21,775 | 102,685 | 103,255 | 107,182 | 107,956 | 112,263 |
| `c8_sub` / 3 | c9_sel 155,745; input 129,103 | 157,156 | 157,725 | 161,764 | 162,534 | 166,818 |
| `c8_sub` / 4 | c9_sel 218,071; input 183,576 | 219,475 | 220,049 | 224,012 | 224,770 | 228,994 |
| `c9_sel` / 2 | c7_add 112,507; c8_sub 112,348; c6_and 143,919 | 145,737 | 146,313 | 148,686 | 151,627 | 155,664 |
| `c9_sel` / 3 | c7_add 166,885; c8_sub 166,902; c6_and 206,189 | 208,013 | 208,592 | 211,017 | 213,964 | 217,991 |
| `c9_sel` / 4 | c7_add 229,382; c8_sub 229,079; c6_and 260,633 | 262,456 | 263,035 | — | — | — |
| `c10_xor` / 1 | c9_sel 101,287; c4_sel 66,553 | 102,721 | 103,314 | 106,779 | 107,567 | 111,678 |
| `c10_xor` / 2 | c9_sel 155,753; c4_sel 121,222 | 157,195 | 157,787 | 161,227 | 162,027 | 166,170 |
| `c10_xor` / 3 | c9_sel 218,078; c4_sel 183,905 | 219,514 | 220,103 | 223,579 | 224,378 | 228,569 |

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 7 | 0 | 0 | 1 (runner: 0) | 9 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
