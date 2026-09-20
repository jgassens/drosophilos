# Tick stall diagnostic: copy 55

## Classification

The captured roles do not establish a request, datapath, consumer-hold, completion, or producer anomaly beyond the workload-derived stall threshold.

## Artifact and threshold

- Dump: `data/a2/tick_s110_node55.npz` (role-filtered; 28,179,407 spikes, 6,252 firing neuron ids; steps 22–899,999).
- Rebuilt kernel: 28,439 neurons / 51,071 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 113,060 steps (11.306 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 16 | 2 | 0 | 0 (runner: 0) | 0 |

Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture `wd\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's own count in the campaign record. A refused input word that the host does not resend shifts every later output by one and scores as wrong values (copy 77, seed 108); the runner resends it since 2026-09-20.

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
