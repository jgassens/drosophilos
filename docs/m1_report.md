# M1 (Stage A1) report — reliable neural word transport and arithmetic

**Date:** 2026-09-13/14. **Substrate profile:** 3 (free synthesis, default neuron parameters,
1.8 ms delays, integer weights). **Labels:** isolated execution, hybrid orchestration (the host
loads operands only after READY and reads spikes), external compilation (hand-designed
circuits). Code: `drosophilos/protocol/`, `drosophilos/lib/`; tests: `tests/test_protocol_exhaustive.py`,
`test_handshake.py`, `test_faults.py`, `test_adder.py`; contracts: `docs/contracts/`.

## 1. What M1 set out to establish

A digital substrate in a spiking network: exact words move between circuits through a
self-timed four-phase protocol, hold in registers, pass through gates and a 4-bit adder,
and do so repeatedly under perturbation with zero silent errors. Everything above this in
the plan (DrosoC, RAM, FlyLink, minidoom) assumes it.

## 2. The protocol, before any neuron

`protocol/spec.md` defines the channel: DATA (each bit on one of two rails) → ACCEPT (the
consumer's completion element holds "complete" and the value is consumed once) → CLEARED
(the producer has emptied its register) → READY (the consumer has emptied its register).
Silence is never a value; both rails on one bit is a FAULT.

`protocol/machine.py` is the same protocol as an executable abstract machine and
`protocol/explore.py` enumerates every interleaving of events, with fault injection:

| check | result |
|---|---|
| no deadlock, exactly-once consume, no illegal transition (3 words, 2 bits) | holds, 34 states |
| duplicated tokens (every token may be delivered twice) | holds, 99,359 states |
| late opposite-rail spike after completion | flagged, never consumed as a value |
| stale DATA from the previous word, no protection | **stale acceptance, 5 deadlocks, 8 illegal transitions** |
| stale DATA with the timing bound or with phase rails | holds |
| a lost ACCEPT / CLEARED / READY / DATA bit | deadlock (timeouts are needed and not yet built) |

The neural implementation uses the timing-bound variant and, on top of it, the fault
detector as a safety net (§5).

## 3. The neural implementation

Physics that shaped every choice (from Stage 0 and H0): one synchronous input of ≈160
synapse-equivalents fires a neuron; a two-neuron loop holding a spike runs at 213 Hz; a
membrane pushed below rest by inhibition recovers with τ_m = 20 ms.

| element | construction | why |
|---|---|---|
| bit latch | two-neuron excitatory loop, 1.4× single-pulse drive | H0's register; the cheapest, most robust store in this model (§7) |
| gate | one neuron reading only latch taps at rate-mode weights (AND 0.75×, OR 2× of the sustained need; 2-of-3 majority at the AND weight) | a latch train is the only standardised signal; a gate's own rate is not |
| latched gate | gate → edge relay → latch | so the next gate again sees a standard train, and the reset can clear it |
| completion element | binary tree of latched 2-input ANDs over latched bit-valid ORs; the root latch is set when all inputs agree and cleared only by RESET | a state-holding C-element with explicit return-to-empty |
| edge relay | relay driven at 1.8× need, its inhibitor at 1.4×, inhibition 2.2× loop per spike | one ignition per source activation; the ordering is settled by drive, not luck |
| reset controller | edge-detected trigger (same head start and inhibitor) → 4-pulse relay chain → one inhibitory neuron → both members of every latch and every gate, 0.75× loop per pulse | kills every latch at every phase with the loop +15 % and the reset −15 %; smallest total inhibitory charge that does |
| READY / CLEARED | 11-hop pure delay chain (~58 ms) from the reset trigger | must exceed reset settling (~21 ms) plus membrane recovery from ~−45 mV total inhibition |
| DATA path | producer latch → edge relay → consumer latch at 2× ignition | the producer's train must not drive the consumer's loop continuously |
| FAULT | per-bit AND of the two rails at 0.55× → one shared fault latch, which holds the completion latch, gate and relay down and raises FAULT-ACCEPT once | a faulted word is never consumed; the channel still completes its four phases |

Measured on the reference simulator (contracts in `docs/contracts/`):

| circuit | neurons | synapses | ACCEPT latency | cycle (initiation interval) | spikes / transaction |
|---|---|---|---|---|---|
| 1-bit channel | 50 | 79 | 38–51 ms | 154–168 ms | ~190 |
| 4-bit channel | 125 | 250 | 112 ms | 247 ms | 1,221 (latch 77 %, control 20 %, gate 5 %) |
| 8-bit channel | 211 | 415 | 158–174 ms | 275–290 ms | ~3,200 |
| 1-bit full adder channel | 153 | ~310 | 169 ms | 305 ms | ~1,300 |
| 4-bit ripple adder channel | 444 | 953 | 244–249 ms | 380–384 ms | 5,288 |

All decode exactly: the 1-bit full adder on all eight inputs, the 4-bit adder on random
operands including 15 + 15 + 1. Operand bits may arrive up to 40 ms apart (tested) — the
rate-mode gates integrate sustained trains, so arrival phase does not matter.

Reset energy is the price of latched everything: 4 × 105,924 quanta per consumer reset
for the 4-bit channel (7,282 mV-equivalent), 4 × 437,164 for the 4-bit adder.

## 4. Fault behaviour of the neural channel (`tests/test_faults.py`)

| injected | outcome |
|---|---|
| same-rail duplicate DATA | absorbed, value correct |
| corrupted bit (both rails at load) | fault latch set, completion blocked, FAULT-ACCEPT, four phases complete, next word clean |
| late opposite-rail spike after ACCEPT | flagged, consumed value stands, next word clean |
| stale full-strength spike 5–45 ms after the consumer's reset trigger | rides through or ignites after the reset train; always detected as a fault, never consumed wrong, channel recovers on the next word |
| lost ACCEPT | channel stalls (no timeout yet) |

## 5. Perturbation campaign

### 5.1 Stress sweep (`data/m1/stress_sweep_4bit.json`)

Each level changes one knob from mix B (weights log-normal σ = 4 %, threshold and bias
±0.2 mV, stray 5 Hz × 2.6 mV pulses on every neuron, arrival jitter ≤ 10 ms).

| level (others at mix B) | ok / 10,000 | silent wrong values | false faults | hangs (no accept / cleared / ready) | stale activity | error rate |
|---|---|---|---|---|---|---|
| weights 6 % | 9983 | 0 | 1 | 4 | 12 | 1.7e-03 |
| weights 8 % | 9827 | 1 | 2 | 85 | 85 | 1.7e-02 |
| weights 10 % | 9297 | 3 | 10 | 432 | 258 | 7.0e-02 |
| threshold ±0.5 mV | 9974 | 0 | 0 | 11 | 15 | 2.6e-03 |
| threshold ±1.0 mV | 7114 | 33 | 196 | 2173 | 484 | 2.9e-01 |
| stray 20 Hz × 2.6 mV | 9998 | 0 | 2 | 0 | 0 | 2.0e-04 |
| stray 50 Hz × 2.6 mV | 9913 | 0 | 67 | 20 | 0 | 8.7e-03 |
| arrival jitter ≤ 40 ms | 10000 | 0 | 0 | 0 | 0 | 0.0e+00 |

Reading: arrival timing is free (the gates integrate); stray input is tolerated to ~20 Hz
per neuron and then produces false faults (detected refusals) before anything else; weight
noise is the real limit, with silent wrong values appearing only at ≥ 8 % and threshold
drift only at ±1 mV. The transition is sharp, as expected of margins set at ±10–15 %.

### 5.2 Million-transaction campaign at mix B (`data/m1/campaign_4bit_B_summary.json`)

(filled in when the run completes)

## 6. What did not work, in the order it happened

1. **READY vetoed by latch activity.** The veto's early pulses hyperpolarised the READY
   neuron itself; 40 ms later it was still 4 mV below rest and the chain pulse missed.
   Replaced by a pure delay chain.
2. **A self-inhibiting one-shot trigger.** Its −280 mV pulse left the trigger unable to
   fire for the next transaction. Replaced by feed-forward edge detection.
3. **One strong reset pulse.** It blocks re-ignition for only ~5 ms; a latch → gate → latch
   chain settles over 10–15 ms and tree latches re-ignited. Replaced by a 4-pulse train.
4. **Half-strength reset pulses.** Fine for a nominal loop, but a loop with +10 % weights
   survived (campaign). 0.75× kills at ±15 %.
5. **Continuous DATA drive.** The producer's train pushed the consumer's latches above the
   standard rate and tripped fault gates at 8 bits. Replaced by per-rail edge relays.
6. **Gates driving latches directly.** A gate's residual firing during the reset train
   re-ignited valid and tree latches. Every gate → latch path is now an edge relay.
7. **Rate-mode AND at 0.65×.** Two inputs at −10 % weights and −10 % rate no longer fired
   (campaign class `no_accept`). Raised to 0.75×; the fault gate, which must *not* fire on one
   rail plus noise, went the other way, to 0.55×.
8. **Edge relays racing their own inhibitor.** With equal drives, 4 % weight noise made the
   relay lose ~0.1 % of the time and drop a bit. Head start: 1.8× need (largest doublet-free
   drive) against a 1.4× inhibitor input.
9. **Reset triggers racing their own inhibitor.** Same race, same fix; then the inhibitor
   had to be raised to 2.2× because the head-started trigger otherwise fired at the
   completion train's rate in ~1 node per 1,000 and CLEARED became a train.
10. **Unlatched fault gates.** Their sparse spikes (~every 36 ms) re-armed the edge
    detector and fired a second reset that wiped the next word. FAULT is now a latch.

Every one of these was found by a test or the campaign, not by inspection, and each
fix was measured before it was kept.

## 7. Storage primitive comparison

See `docs/m1_latches.md`. In this neuron model there is no low-activity storage: the
two-neuron loop at 1.4× is the production register (44 spikes per 100 ms held); weaker
drive lowers activity but loses robustness; longer rings and long delays hold several
circulating spikes and broadcast more.

## 8. Contract fields the model cannot exercise

Fan-out costs a presynaptic neuron nothing here (no load), so "fan-out tolerated" is
unbounded by the model and is measured on the real connectome as isolation cost instead.
Per-synapse delay jitter is not supported by the shared-topology backend and is listed as
not measured. Both are stated in every contract file.
