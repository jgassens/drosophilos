# Stage H0 report — a minimal circuit in real MCNS wiring (Profile 2)

Conclusions and consequences are written up in `docs/h0_findings.md`.

Generated from `data/h0/results.json` (copy versioned at `docs/h0/results.json`; manifest at `docs/h0/manifest_full_graph_surround_live.json`). Labels: **profile 2**, isolated **and** full-graph, hybrid orchestration (host injects DATA and reads spikes), external compilation (hand-designed circuit).

## Circuit

Dual-rail 1-bit AND (the carry of a half adder) with register, completion detector, and reset return path. Values are held by two-neuron excitatory loops (a circulating spike); the AND is a threshold on the sustained drive of two loops; OR, completion, and reset work in single-pulse mode.

| role | bodyId | type | superclass | transmitter | side |
|---|---|---|---|---|---|
| loop_a0_partner | 10643 | DNge059 | descending_neuron | acetylcholine | R |
| loop_a1_partner | 10864 | GNG014 | cb_intrinsic | acetylcholine | R |
| and_y1 | 10881 | GNG120 | cb_intrinsic | acetylcholine | R |
| loop_b1_partner | 11127 | DNge059 | descending_neuron | acetylcholine | L |
| completion | 11429 | GNG236 | cb_intrinsic | acetylcholine | L |
| reset_b0_1 | 14160 | nan | cb_intrinsic | gaba | L |
| reset_a0_1 | 14210 | GNG048 | cb_intrinsic | gaba | R |
| loop_a0_driver | 26764 | GNG108 | cb_intrinsic | acetylcholine | R |
| loop_a1_driver | 32461 | GNG457 | cb_intrinsic | acetylcholine | R |
| loop_b0_driver | 37111 | GNG457 | cb_intrinsic | acetylcholine | L |
| or_y0 | 512079 | GNG169 | cb_intrinsic | acetylcholine | L |
| reset_a0_0 | 520816 | GNG298 | cb_intrinsic | gaba | M |
| loop_b1_driver | 523040 | GNG108 | cb_intrinsic | acetylcholine | L |
| reset_b0_0 | 523983 | GNG048 | cb_intrinsic | gaba | L |
| loop_b0_partner | 555296 | GNG014 | cb_intrinsic | acetylcholine | L |

### Designed edges (all anatomical; weight = quanta, bound 0 ≤ q ≤ k_max·count·16)

| role | pre → post | anatomical synapses | quanta | scale vs anatomical |
|---|---|---|---|---|
| loop_a1 | 32461 → 10864 | 161 | 3621 | 1.41× |
| loop_a1 | 10864 → 32461 | 107 | 3621 | 2.12× |
| loop_b1 | 523040 → 11127 | 66 | 3621 | 3.43× |
| loop_b1 | 11127 → 523040 | 122 | 3621 | 1.86× |
| loop_a0 | 26764 → 10643 | 98 | 3621 | 2.31× |
| loop_a0 | 10643 → 26764 | 211 | 3621 | 1.07× |
| loop_b0 | 37111 → 555296 | 107 | 3621 | 2.12× |
| loop_b0 | 555296 → 37111 | 92 | 3621 | 2.46× |
| and_a1 | 32461 → 10881 | 27 | 249 | 0.58× |
| and_b1 | 523040 → 10881 | 151 | 249 | 0.1× |
| or_a0 | 26764 → 512079 | 64 | 766 | 0.75× |
| or_b0 | 37111 → 512079 | 24 | 766 | 1.99× |
| completion_y1 | 10881 → 11429 | 65 | 3621 | 3.48× |
| completion_y0 | 512079 → 11429 | 70 | 3621 | 3.23× |
| reset_drive_a1 | 11429 → 14160 | 82 | 3621 | 2.76× |
| reset_drive_a1 | 11429 → 14210 | 218 | 3621 | 1.04× |
| reset_a1 | 14160 → 10864 | 163 | -5432 | -2.08× |
| reset_a1 | 14210 → 32461 | 254 | -5432 | -1.34× |
| reset_drive_b1 | 11429 → 520816 | 63 | 3621 | 3.59× |
| reset_drive_b1 | 11429 → 523983 | 62 | 3621 | 3.65× |
| reset_b1 | 520816 → 11127 | 166 | -5432 | -2.05× |
| reset_b1 | 523983 → 523040 | 207 | -5432 | -1.64× |
| reset_drive_a0 | 11429 → 520816 | 63 | 3621 | 3.59× |
| reset_drive_a0 | 11429 → 14210 | 218 | 3621 | 1.04× |
| reset_a0 | 520816 → 10643 | 127 | -5432 | -2.67× |
| reset_a0 | 14210 → 26764 | 230 | -5432 | -1.48× |
| reset_drive_b0 | 11429 → 523983 | 62 | 3621 | 3.65× |
| reset_drive_b0 | 11429 → 14160 | 82 | 3621 | 2.76× |
| reset_b0 | 523983 → 37111 | 205 | -5432 | -1.66× |
| reset_b0 | 14160 → 555296 | 113 | -5432 | -3.0× |

Parasitic anatomical edges among circuit neurons (zeroed, documented): 107; minimum weight margin (count·16 / |q|, ≥ 1/k_max required): 0.27.

## Motif availability

- Physics targets (quanta): loop 3621, AND input 249 (rate mode), OR input 766, completion / reset drive 3621 (single pulse), reset -5432 per hit member; loop period 47 steps.
- Required anatomical synapse counts at k_max = 4.0: loop ≥ 57, AND ≥ 4, OR ≥ 12, completion ≥ 57, reset ≥ 5432 quanta per member.
- Candidate latches (mutual cholinergic pairs, both directions ≥ 57): **1223**.
- Completion candidates (cholinergic, ≥ 2 cholinergic inputs ≥ 57): **5004**; of these, with four resettable latches: **141**; with usable AND+OR gate pairs: **610**.
- Complete embeddings found: **50** in 1.8 s (search truncated).
- Relay overhead: reset uses 8 inhibitory edges from 4 inhibitory neurons; no relay neurons were needed beyond the designed roles.

## Isolation cost (what Profile 2 silencing must zero to stop the circuit broadcasting)

- Anatomical edges from the 15 circuit neurons to the surround: **4830** (72274 synapses; 697 edges of ≥ 24 synapses, each strong enough to fire its target on its own under a 213 Hz latch train).
- Anatomical edges from the surround into circuit neurons: **5493** (71364 synapses; 714 of ≥ 24).

## Results by condition

### isolated_parasitic_zeroed

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | True | True | ✓ | 22.8 ms | ✓ | 26.9 ms | 0 | 0 |
| 0 | 1 | False | True | True | ✓ | 36.9 ms | ✓ | 41.7 ms | 0 | 0 |
| 1 | 0 | False | True | True | ✓ | 36.9 ms | ✓ | 41.7 ms | 0 | 0 |
| 1 | 1 | True | False | True | ✓ | 56.3 ms | ✓ | 60.8 ms | 0 | 0 |

Timing tolerance (a=1, b=1 with b delayed):

| b offset | y1 fired | correct | completion latency | reset |
|---|---|---|---|---|
| 0 ms | True | ✓ | 56.3 ms | ✓ |
| 1 ms | True | ✓ | 55.4 ms | ✓ |
| 2 ms | True | ✓ | 54.4 ms | ✓ |
| 3 ms | True | ✓ | 53.5 ms | ✓ |
| 4 ms | True | ✓ | 52.8 ms | ✓ |
| 5 ms | True | ✓ | 52.3 ms | ✓ |
| 6 ms | True | ✓ | 52.0 ms | ✓ |
| 8 ms | True | ✓ | 51.8 ms | ✓ |
| 10 ms | True | ✓ | 51.5 ms | ✓ |
| 12 ms | True | ✓ | 49.7 ms | ✓ |

### isolated_parasitic_anatomical

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | True | True | ✓ | 47.3 ms | ✓ | 56.6 ms | 0 | 0 |
| 0 | 1 | False | True | True | ✓ | 21.5 ms | ✓ | 30.6 ms | 0 | 0 |
| 1 | 0 | False | True | True | ✓ | 55.0 ms | ✓ | 66.0 ms | 0 | 0 |
| 1 | 1 | False | True | True | ✗ | 64.6 ms | ✓ | 74.7 ms | 0 | 0 |

Timing tolerance (a=1, b=1 with b delayed):

| b offset | y1 fired | correct | completion latency | reset |
|---|---|---|---|---|
| 0 ms | False | ✗ | 64.6 ms | ✓ |
| 1 ms | False | ✗ | 62.9 ms | ✓ |
| 2 ms | False | ✗ | 62.1 ms | ✓ |
| 3 ms | False | ✗ | 54.3 ms | ✓ |
| 4 ms | False | ✗ | 53.2 ms | ✓ |
| 5 ms | False | ✗ | 51.9 ms | ✓ |
| 6 ms | False | ✗ | 50.9 ms | ✓ |
| 8 ms | False | ✗ | 48.9 ms | ✓ |
| 10 ms | False | ✗ | 47.0 ms | ✓ |
| 12 ms | False | ✗ | 48.6 ms | ✓ |

### full_graph_surround_live_silent

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | True | False | True | ✗ | 80.3 ms | ✓ | 18.5 ms | 204727 | 23717 |
| 0 | 1 | True | True | True | ✗ | 24.7 ms | ✓ | 27.9 ms | 217118 | 23917 |
| 1 | 0 | False | False | False | ✗ | — | ✗ | — | 98908 | 19680 |
| 1 | 1 | False | False | False | ✗ | — | ✗ | — | 185605 | 23147 |

Boundary-input envelope (quanta into circuit neurons from non-circuit spikes):

| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |
|---|---|---|---|---|---|
| 0 | 0 | 3408 | -6176 | 2403088 | 1219 |
| 0 | 1 | 5104 | -4720 | 2705136 | 1278 |
| 1 | 0 | 3696 | -5792 | 1232320 | 797 |
| 1 | 1 | 3440 | -5872 | 2148720 | 1184 |

Most recruited surround neurons for a=0, b=0: OA-VUMa8 (cb_intrinsic, 48 spikes), OA-VUMa1 (cb_intrinsic, 47 spikes), OA-AL2i1 (visual_centrifugal, 47 spikes), OA-VUMa1 (cb_intrinsic, 47 spikes), ExR6 (cb_intrinsic, 46 spikes), ExR6 (cb_intrinsic, 46 spikes), GNG003 (cb_intrinsic, 45 spikes), GLNO (cb_intrinsic, 45 spikes)

Most recruited surround neurons for a=0, b=1: OA-VUMa8 (cb_intrinsic, 51 spikes), OA-AL2i1 (visual_centrifugal, 49 spikes), OA-VUMa1 (cb_intrinsic, 49 spikes), OA-VUMa1 (cb_intrinsic, 49 spikes), ExR6 (cb_intrinsic, 49 spikes), GNG003 (cb_intrinsic, 48 spikes), LT33 (ol_intrinsic, 48 spikes), ExR6 (cb_intrinsic, 48 spikes)

Most recruited surround neurons for a=1, b=0: Cm34 (ol_intrinsic, 32 spikes), OA-AL2i1 (visual_centrifugal, 31 spikes), ExR6 (cb_intrinsic, 31 spikes), ExR6 (cb_intrinsic, 31 spikes), GNG003 (cb_intrinsic, 31 spikes), LT33 (ol_intrinsic, 31 spikes), ExR4 (cb_intrinsic, 30 spikes), GLNO (cb_intrinsic, 30 spikes)

Most recruited surround neurons for a=1, b=1: OA-VUMa8 (cb_intrinsic, 46 spikes), OA-VUMa1 (cb_intrinsic, 44 spikes), ExR6 (cb_intrinsic, 44 spikes), OA-VUMa1 (cb_intrinsic, 44 spikes), ExR6 (cb_intrinsic, 44 spikes), ExR4 (cb_intrinsic, 43 spikes), OA-AL2i1 (visual_centrifugal, 43 spikes), GLNO (cb_intrinsic, 43 spikes)

### full_graph_surround_live_background

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | True | True | ✓ | 16.7 ms | ✓ | 12.7 ms | 461436 | 30045 |
| 0 | 1 | True | False | True | ✗ | 21.7 ms | ✓ | 14.7 ms | 462061 | 29870 |
| 1 | 0 | False | False | False | ✗ | — | ✗ | — | 457642 | 29862 |
| 1 | 1 | False | False | False | ✗ | — | ✗ | — | 469185 | 29964 |

Boundary-input envelope (quanta into circuit neurons from non-circuit spikes):

| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |
|---|---|---|---|---|---|
| 0 | 0 | 6768 | -5392 | 3696448 | 1370 |
| 0 | 1 | 6784 | -5424 | 4408416 | 1369 |
| 1 | 0 | 3488 | -5792 | 3642688 | 1353 |
| 1 | 1 | 6768 | -5920 | 5134944 | 1386 |

Most recruited surround neurons for a=0, b=0: il3LN6 (cb_intrinsic, 65 spikes), lLN2P_a (cb_intrinsic, 64 spikes), il3LN6 (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), lLN2P_b (cb_intrinsic, 63 spikes), VA6_adPN (cb_intrinsic, 63 spikes), lLN2X04 (cb_intrinsic, 63 spikes)

Most recruited surround neurons for a=0, b=1: lLN2P_a (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), il3LN6 (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), il3LN6 (cb_intrinsic, 64 spikes)

Most recruited surround neurons for a=1, b=0: il3LN6 (cb_intrinsic, 66 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), il3LN6 (cb_intrinsic, 65 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2X04 (cb_intrinsic, 64 spikes), lLN2P_c (cb_intrinsic, 64 spikes)

Most recruited surround neurons for a=1, b=1: il3LN6 (cb_intrinsic, 66 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), il3LN6 (cb_intrinsic, 65 spikes), lLN1_bc (cb_intrinsic, 64 spikes), lLN1_bc (cb_intrinsic, 64 spikes), lLN2T_a (cb_intrinsic, 64 spikes), v2LN30 (cb_intrinsic, 64 spikes)

### full_graph_surround_live_burst

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | False | False | ✗ | — | ✗ | — | 400878 | 27680 |
| 0 | 1 | True | False | True | ✗ | 13.8 ms | ✓ | 11.9 ms | 419051 | 26694 |
| 1 | 0 | False | False | False | ✗ | — | ✗ | — | 418196 | 27835 |
| 1 | 1 | True | False | True | ✓ | 47.5 ms | ✓ | 15.9 ms | 418505 | 27093 |

Boundary-input envelope (quanta into circuit neurons from non-circuit spikes):

| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |
|---|---|---|---|---|---|
| 0 | 0 | 3872 | -5840 | 3766976 | 1337 |
| 0 | 1 | 6656 | -5568 | 3491008 | 1339 |
| 1 | 0 | 4560 | -7008 | 3981456 | 1347 |
| 1 | 1 | 4128 | -5264 | 3488256 | 1343 |

Most recruited surround neurons for a=0, b=0: APL (cb_intrinsic, 63 spikes), APL (cb_intrinsic, 62 spikes), il3LN6 (cb_intrinsic, 61 spikes), lLN2F_b (cb_intrinsic, 61 spikes), lLN2F_b (cb_intrinsic, 61 spikes), lLN2F_b (cb_intrinsic, 61 spikes), lLN2F_b (cb_intrinsic, 61 spikes), MBON11 (cb_intrinsic, 61 spikes)

Most recruited surround neurons for a=0, b=1: APL (cb_intrinsic, 63 spikes), APL (cb_intrinsic, 63 spikes), lLN2F_b (cb_intrinsic, 62 spikes), v2LN30 (cb_intrinsic, 62 spikes), lLN2F_b (cb_intrinsic, 62 spikes), lLN2P_a (cb_intrinsic, 62 spikes), lLN2X04 (cb_intrinsic, 62 spikes), il3LN6 (cb_intrinsic, 62 spikes)

Most recruited surround neurons for a=1, b=0: lLN2X11 (cb_intrinsic, 64 spikes), APL (cb_intrinsic, 63 spikes), APL (cb_intrinsic, 63 spikes), lLN2T_a (cb_intrinsic, 62 spikes), lLN2X12 (cb_intrinsic, 62 spikes), lLN2F_b (cb_intrinsic, 62 spikes), il3LN6 (cb_intrinsic, 62 spikes), lLN2T_a (cb_intrinsic, 62 spikes)

Most recruited surround neurons for a=1, b=1: lLN2X04 (cb_intrinsic, 64 spikes), APL (cb_intrinsic, 63 spikes), lLN2T_c (cb_intrinsic, 62 spikes), lLN2P_a (cb_intrinsic, 62 spikes), lLN1_bc (cb_intrinsic, 62 spikes), lLN1_bc (cb_intrinsic, 62 spikes), lLN2T_e (cb_intrinsic, 62 spikes), DP1m_adPN (cb_intrinsic, 62 spikes)

### full_graph_outputs_zeroed_silent

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | True | True | ✓ | 22.8 ms | ✓ | 29.4 ms | 0 | 0 |
| 0 | 1 | False | True | True | ✓ | 36.9 ms | ✓ | 41.7 ms | 0 | 0 |
| 1 | 0 | False | True | True | ✓ | 36.9 ms | ✓ | 41.7 ms | 0 | 0 |
| 1 | 1 | True | False | True | ✓ | 56.3 ms | ✓ | 63.1 ms | 0 | 0 |

Boundary-input envelope (quanta into circuit neurons from non-circuit spikes):

| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 | 0 | 0 |

Outgoing anatomical edges from circuit neurons to the surround zeroed (documented silencing): 4830.

### full_graph_outputs_zeroed_background

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | True | True | ✓ | 16.2 ms | ✓ | 12.5 ms | 461675 | 30172 |
| 0 | 1 | False | False | False | ✗ | — | ✗ | — | 460676 | 29754 |
| 1 | 0 | False | False | False | ✗ | — | ✗ | — | 456967 | 29730 |
| 1 | 1 | False | False | False | ✗ | — | ✗ | — | 468281 | 29743 |

Boundary-input envelope (quanta into circuit neurons from non-circuit spikes):

| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |
|---|---|---|---|---|---|
| 0 | 0 | 6784 | -4912 | 3744400 | 1367 |
| 0 | 1 | 6672 | -7168 | 4454976 | 1365 |
| 1 | 0 | 6688 | -4976 | 3659904 | 1349 |
| 1 | 1 | 6768 | -6944 | 5203088 | 1383 |

Most recruited surround neurons for a=0, b=0: il3LN6 (cb_intrinsic, 65 spikes), il3LN6 (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), lLN2P_c (cb_intrinsic, 63 spikes), lLN2P_b (cb_intrinsic, 63 spikes), lLN2X04 (cb_intrinsic, 63 spikes)

Most recruited surround neurons for a=0, b=1: lLN2P_a (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes), il3LN6 (cb_intrinsic, 64 spikes), il3LN6 (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes), lLN2F_b (cb_intrinsic, 64 spikes)

Most recruited surround neurons for a=1, b=0: il3LN6 (cb_intrinsic, 66 spikes), il3LN6 (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2T_a (cb_intrinsic, 64 spikes), lLN2P_c (cb_intrinsic, 64 spikes), lLN2T_a (cb_intrinsic, 64 spikes)

Most recruited surround neurons for a=1, b=1: il3LN6 (cb_intrinsic, 66 spikes), il3LN6 (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN2F_b (cb_intrinsic, 65 spikes), lLN1_bc (cb_intrinsic, 64 spikes), lLN2X05 (cb_intrinsic, 64 spikes), lLN1_bc (cb_intrinsic, 64 spikes), lLN2P_a (cb_intrinsic, 64 spikes)

Outgoing anatomical edges from circuit neurons to the surround zeroed (documented silencing): 4830.

### full_graph_outputs_zeroed_burst

| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | False | True | False | ✗ | — | ✗ | — | 405869 | 26393 |
| 0 | 1 | False | False | False | ✗ | — | ✗ | — | 420968 | 27163 |
| 1 | 0 | True | True | True | ✗ | 28.4 ms | ✓ | 18.4 ms | 406289 | 26616 |
| 1 | 1 | True | False | True | ✓ | 18.1 ms | ✓ | 15.6 ms | 402754 | 26819 |

Boundary-input envelope (quanta into circuit neurons from non-circuit spikes):

| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |
|---|---|---|---|---|---|
| 0 | 0 | 5648 | -7072 | 4557520 | 1336 |
| 0 | 1 | 6672 | -4992 | 4490080 | 1330 |
| 1 | 0 | 6752 | -7552 | 5185376 | 1341 |
| 1 | 1 | 6848 | -5792 | 4180032 | 1336 |

Most recruited surround neurons for a=0, b=0: APL (cb_intrinsic, 63 spikes), lLN2T_b (cb_intrinsic, 63 spikes), APL (cb_intrinsic, 63 spikes), lLN2F_b (cb_intrinsic, 62 spikes), v2LN30 (cb_intrinsic, 62 spikes), lLN2P_c (cb_intrinsic, 62 spikes), lLN2P_c (cb_intrinsic, 62 spikes), lLN2P_c (cb_intrinsic, 62 spikes)

Most recruited surround neurons for a=0, b=1: lLN2X12 (cb_intrinsic, 64 spikes), lLN2X12 (cb_intrinsic, 64 spikes), APL (cb_intrinsic, 63 spikes), il3LN6 (cb_intrinsic, 63 spikes), APL (cb_intrinsic, 63 spikes), il3LN6 (cb_intrinsic, 63 spikes), lLN2X11 (cb_intrinsic, 62 spikes), lLN2X05 (cb_intrinsic, 62 spikes)

Most recruited surround neurons for a=1, b=0: APL (cb_intrinsic, 63 spikes), lLN2F_b (cb_intrinsic, 62 spikes), il3LN6 (cb_intrinsic, 62 spikes), lLN1_bc (cb_intrinsic, 62 spikes), lLN2T_a (cb_intrinsic, 62 spikes), il3LN6 (cb_intrinsic, 62 spikes), lLN2X05 (cb_intrinsic, 62 spikes), lLN2F_b (cb_intrinsic, 62 spikes)

Most recruited surround neurons for a=1, b=1: lLN2F_b (cb_intrinsic, 62 spikes), APL (cb_intrinsic, 62 spikes), APL (cb_intrinsic, 62 spikes), lLN2F_b (cb_intrinsic, 62 spikes), v2LN30 (cb_intrinsic, 62 spikes), lLN2P_a (cb_intrinsic, 61 spikes), il3LN6 (cb_intrinsic, 61 spikes), lLN2X04 (cb_intrinsic, 61 spikes)

Outgoing anatomical edges from circuit neurons to the surround zeroed (documented silencing): 4830.

