# Tick stall diagnostic: copy 77

## Classification

The captured roles do not establish a request, datapath, consumer-hold, completion, or producer anomaly beyond the workload-derived stall threshold.

## Artifact and threshold

- Dump: `data/a2/tick_s108_node77.npz` (role-filtered; 13,513,510 spikes, 4,874 firing neuron ids; steps 23–899,999).
- Rebuilt kernel: 28,439 neurons / 51,069 synapses. Every recorded role string was asserted against the rebuilt netlist at the recorded neuron id.
- A latch train is continuous across gaps ≤ 141 steps (three nominal loop periods). A phase is called stalled only after 21,584 steps (2.1584 s): twice the longest completed START→DONE in this workload, with a 2 s floor.
- The analysis is read-only. It does not attempt a single-copy replay; the campaign's device-drawn stray stream depends on the original batch size.

## Campaign accounting for this copy

| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 14 | 5 | 0 | 0 | 2 |

A capture or campaign stopped by its neural-time or wall-time resource limit is **truncated**, not automatically stalled. The stall label above rests on a specific handshake phase remaining blocked past the stated workload threshold; unfinished counts alone are not that evidence.
