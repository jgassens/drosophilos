# A2 opening: liveness — what the machine now detects by itself

Follows the M1 review: the 22 non-ok transactions in the 10⁶ campaign were noticed by the
benchmark harness, not by the neural machine (no timeout existed). This note records the
two mechanisms built to change that, what each measured, and which one survived.

## 1. Watchdog (kept)

`protocol/watchdog.py::add_watchdog`. A relay chain (40 hops ≈ 212 ms on the channel,
80 ≈ 425 ms on the adder) is started once per producer activation (an OR over the
producer's rail taps through an edge relay) and cancelled by ACCEPT or FAULT-ACCEPT (the
completion latch's or the fault latch's train drives an inhibitory neuron that holds every
hop down, so the pulse in flight dies). If it reaches the end it ignites a TIMEOUT latch,
which is a FAULT-ACCEPT source through the producer's edge-detected reset trigger: the
producer clears, CLEARED runs, the consumer resets, READY is issued, the next word is
accepted. A transaction that can never complete is therefore *refused by the machine*.

| test | result |
|---|---|
| data path for one rail severed, word using that rail | TIMEOUT latch at 227 ms, CLEARED at 306 ms, READY at 384 ms, next two words correct |
| clean transactions | no TIMEOUT spike |
| 6,000 transactions at mix B | 6,000 ok, 0 false timeouts |

Cost on the 4-bit channel: 125 → 179 neurons.

## 2. Stale-state monitor (rejected on measurement)

`protocol/watchdog.py::add_stale_monitor`, available behind `build_channel(..., monitor=True)`.
Design: an ENABLE latch ignited two hops after the reset train's last relay (a fifth,
trailing inhibitory pulse at +27 ms killed it when armed one hop after); one two-input
rate-mode AND per latch (ENABLE + that latch's tap); an OR over the detectors ignites a
STALE latch that holds READY and the chain tail down and re-triggers the reset; READY
clears ENABLE with a two-pulse train.

Nominal behaviour is as designed. Under mix B it fails:

| mechanism | measured |
|---|---|
| a detector's tap input alone sits at 75 % of the sustained need for ~100 ms per transaction, across 12–14 detectors; one spurious spike is enough because the path to STALE is single-spike ignition | ~2 % of transactions raised STALE with ENABLE never having fired (confirmed by timeline: ENABLE 0 spikes, detector 1 spike) |
| consequences | false retries, READY held (`no_ready`), and, when the re-triggered reset landed mid-transaction, **wrong values** (21 in 4,000) |
| the safe alternative (0.55 + 0.55) | fires ~45 ms after arming; READY comes ~48 ms after arming at 15 hops: no margin |

A second unconditional reset train was probed as the "eliminate" option: at loop +20 %
/ reset −20 % one train kills 49/50 phases and two kill 50/50; at +25 % / −25 % neither
kills any. The 0.75× train's margin edge is ~±20 %, and a second train does not move it.

Conclusion: stale state after a reset stays harness-observed. It is rare (8 in 10⁶ at mix
B) and, when it happens, it either hangs the next transaction (now converted to a neural
TIMEOUT by the watchdog) or is caught by the fault gates. A neurally *clean* "register is
empty" signal needs a primitive this model does not offer cheaply: a fast AND with a large
one-input rejection margin. That is recorded for A2's gate work, not forced here.

## 3. What the adder composition campaign exposed (your step 4)

The first 10⁵-addition attempt at mix B failed ~25 % of transactions (both-rail faults,
missing completions, timeouts, wrong sums) while the transport channel stayed clean. Three
separate causes, each found from failure anatomy and fixed on measurement:

| cause | evidence | fix |
|---|---|---|
| adder-internal latches were wired into the consumer's reset at 0.5× per pulse (a leftover default) while every other latch gets 0.75×; 0.5× lets +10 % loops survive | stale carries and XOR stages across transactions | `extend_reset` at 0.75× |
| the 2× ignition pulse is a doublet; a latch that briefly carries two spikes reads as up to 259 Hz (+22 %) to its gates, and a one-input AND at 0.75 then crosses threshold. Long one-input exposure in the adder (carries ripple slowly) made this visible where the channel's completion ANDs (inputs arrive together) hid it | latch-rate p99 259 Hz with 2× ignition, 226 Hz with 1.8× need; the false-firing ANDs each had exactly one live input | ignition = 1.8× need, the largest doublet-free drive; the 15-hop READY chain makes the post-reset hangover negligible, so 2× is no longer needed |
| with rates at 195–236 Hz the two-input rate-mode AND has a real two-sided window; 0.75 still leaks in long-exposure gates | 2,000 perturbed additions: 0.75 → 4.4 % non-ok, 0.70 → 7, 0.68 → 5, 0.65 → 0; channel at 0.65 → 0 | AND input fraction 0.65 (two inputs 1.3×) |

Cost: ACCEPT latency 112 → 155 ms on the 4-bit channel, 245 → 358 ms on the 4-bit adder.
The margins are now consistent across gates (fault gates stay at 0.55: they must not fire on
one rail even with a stale second spike).

Two-input rate-mode ANDs are the weakest primitive in the library: their usable window is
set by the latch-rate spread on one side and by weight noise on the other, and a wider
window needs a different construction (an inhibition-sharpened AND, or inputs that are
released together by a completion stage), which is recorded for the A2 gate work.

## 4. Campaigns with the final build (mix B, 10⁵ each; `docs/a2/`)

| run | transactions | ok | wrong value | fault (neural) | timeout (neural) | hangs (harness) | stale activity (harness) | non-ok observed / 95 % upper | ACCEPT p99 |
|---|---|---|---|---|---|---|---|---|---|
| 4-bit channel, M1 build + watchdog (AND 0.75, 2× ignition) | 100,000 | 99,997 | 0 | 0 | 0 | 2 | 1 | 3.0e-05 / 7.8e-05 | 120 ms |
| 4-bit channel, final build (AND 0.65, 1.8× ignition, watchdog 55 hops) | 100,000 | 99,999 | 0 | 0 | 0 | 0 | 1 | 1.0e-05 / 4.7e-05 | 170 ms |

| run | transactions | ok | wrong value | fault (neural) | timeout (neural) | hangs | stale | non-ok observed / 95 % upper |
|---|---|---|---|---|---|---|---|---|
| 4-bit ripple adder, final build, random operands | 100,002 | 99,938 | 0 | 30 | 1 | 24 | 9 | 6.4e-04 / 7.9e-04 |

Reading: no wrong value was ever consumed in 3 × 10⁵ channel transactions and 10⁵
random additions (exact 95 % upper limits 3.0 × 10⁻⁵ each). The watchdog never fired
falsely. On the adder, ACCEPT p99 388 ms, max 410 ms, 558 neurons; its residual non-ok
rate is 6.4 × 10⁻⁴, an order of magnitude above the channel's, and about half of it is now
raised by the machine (30 faults, 1 timeout) while 24 missing completions and 9 stale-
activity cases are still harness-observed. The faults are the fault gates doing their job
on internal false positives that reach the output rails; they cost a refused transaction,
not a wrong sum. The missing completions are the class the watchdog was built for and
did not catch: an addition whose producer register activated (the watchdog started) but
where cancellation or the chain itself failed, or a hang that lasted beyond the 700 ms
window but resolved before the watchdog's 530 ms deadline had expired in the run's own
timeline — separating these needs a per-case dump and is the first item of A2's gate work. The hangs the watchdog is designed to catch (a transaction
that starts and never completes) did not occur in these runs; the two `no_accept` cases in
the M1-build run were not converted to timeouts either, which means they were not
"started and stalled" — most likely the producer register never activated (a failed load
ignition), which is outside the watchdog's window by construction and belongs to the
upstream's own timeout. Recorded as a scoped limitation.

