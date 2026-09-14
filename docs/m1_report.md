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

### 5.1 Stress sweep (`docs/m1/stress_sweep_4bit.json`)

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

### 5.2 Million-transaction campaign at mix B (`docs/m1/campaign_4bit_B_summary.json`)

1,000,000 four-bit transactions, 250 independent channel copies at a time, each copy with
its own weight noise (log-normal σ = 4 %), threshold and bias drift (±0.2 mV), stray input
(5 Hz × 2.6 mV on every neuron), random words and per-bit arrival jitter (≤ 10 ms); one
word every 320 ms; every transaction decoded from the spike trace after the fact.
Reference-exact float64 CPU backend; 5.5 h wall on the M1 Pro.

| class | count | meaning |
|---|---|---|
| ok | 999,978 | correct value consumed, CLEARED and READY seen, consumer quiet until the next load |
| **wrong value consumed** | **0** | the only silent failure class |
| false fault (detected refusal) | 3 | fault gate tripped without corruption; word refused, channel recovered |
| no accept (hang) | 11 | completion never fired |
| no cleared / no ready | 0 / 0 | |
| stale activity | 8 | a consumer latch survived the reset train |
| total non-ok | 22 | observed 2.2 × 10⁻⁵; exact one-sided 95 % upper limit (Clopper–Pearson) 3.1 × 10⁻⁵ |

**Who noticed.** Only the 3 false faults were raised by the neural machine itself (the
fault latch). The 11 no-accept hangs and the 8 stale-activity cases were inferred by the
benchmark harness from the spike trace (completion never appeared; consumer spikes after
READY). DrosophilOS has no neural timeout yet, so those 19 are *harness-detected*, not
*architecturally detected*. Converting them into neural fault/recovery events is the first
A2 work (§9).

ACCEPT latency: mean 109.5 ms, p99 119.4 ms, max 130.9 ms.
Spikes per transaction: mean 1198, max 1853.

**Against the M1 exit criterion** ("a four-bit word completes a full four-phase handshake
10⁶ times under perturbation with zero decoded errors"):

- Zero wrong values were ever consumed, in 10⁶ transactions at mix B and in the 10⁴-per-level
  sweep up to 6 % weight noise and ±0.5 mV threshold drift. With zero observed silent
  failures in 10⁶ trials the exact 95 % upper limit on the silent-error rate is 3.0 × 10⁻⁶ —
  an observed bound, not the 10⁻¹⁰ objective, which remains analytical and unproven, and
  which further brute-force simulation will not reach; it has to come from architectural
  protection, detection, retry, redundancy and analysis.
- The channel is not yet fully available at this perturbation level: an observed
  2.2 × 10⁻⁵ (95 % upper limit 3.1 × 10⁻⁵) of transactions end in a refusal or a hang, and
  only the refusals are detected neurally. A second 2 × 10⁵ run at another seed
  gave 2 in 2 × 10⁵, the same order. The one hang caught with a timeline was a completion
  latch still firing late in the reset train, i.e. a reset-margin case, not a protocol
  case. Timeouts (which the abstract machine already needs for lost tokens) would convert
  hangs into detected refusals; the reset margin can be raised at the cost of longer
  recovery. Both are recorded as A2 work, not hidden.
- At mix A (3 % / ±0.15 mV / 5 Hz / ≤ 10 ms) every sample so far has been clean; a 10⁶ run
  there was not made because it would only move the observed bound, not the mechanism.
- The campaign ran on the 4-bit transport channel, not on the adder. The adders are built
  from the same primitives and inherit their measured margins, but a 444-neuron network can
  expose interactions the primitive margins do not predict; a 10⁵ composition campaign on
  the ripple adder precedes freezing the channel contract (§9).

**Verdict:** M1 is closed for functional correctness and safety (no silent errors), with a
residual liveness/availability defect carried forward into A2/B — not "fully reliable".

## 9. Opening sequence for A2/B (from the M1 review)

1. Neural timeout / watchdog behaviour.
2. Convert no-ACCEPT / no-READY into an explicit, recoverable, neurally raised transaction fault.
3. Harden reset so that stale state after the train is either eliminated or detected
   neurally before READY is issued.
4. A 10⁵ random-addition campaign at mix B on the complete 4-bit adder.
5. Freeze the M1 channel contract.
6. Then the ALU, word register with staged commit, RAM, ROM and control machine.

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

See `docs/m1_latches.md` (data `docs/m1/latch_alternatives.json`). In this neuron model there is no low-activity storage: the
two-neuron loop at 1.4× is the production register (44 spikes per 100 ms held); weaker
drive lowers activity but loses robustness; longer rings and long delays hold several
circulating spikes and broadcast more.

## 8. Contract fields the model cannot exercise

Fan-out costs a presynaptic neuron nothing here (no load), so "fan-out tolerated" is
unbounded by the model and is measured on the real connectome as isolation cost instead.
Per-synapse delay jitter is not supported by the shared-topology backend and is listed as
not measured. Both are stated in every contract file.
