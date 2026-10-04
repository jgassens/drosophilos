# Stage D: the request/clear stall of seed 108, `rate_robust` copy 28

**Mechanism established.** The false ("consumed") rail of `c4_sel`'s request from
`c2_sel` (latch 15701/15702) survived the four-pulse DONE clear at step 30,423,365. It was
not re-lit afterwards. This copy's noise draws make the latch asymmetric: its `u → v`
edge is +11.8 % and `v`'s threshold is −0.50 mV. Single background strays therefore keep
throwing it into a recurring fast excursion, about three times a second. In each excursion
the two members slide into near-coincident firing, with a 35–38-step period. The clear
arrived at the one phase of that excursion where a 4 × 0.75 train fails. Each pulse landed
0–5 steps after both members had fired, delayed them by roughly the inter-pulse spacing,
and so kept them just ahead of the next pulse: the latch was **entrained, not killed**.
Both request rails stayed live, the false rail vetoed `c4_sel`'s go chain, and the cell
never started again.

No repair pulse is expected in this situation. The ACT^d repair is driven by START,
re-lights a *dark* false rail, and is vetoed by a live one. `rate_robust` does not touch
this circuit; it matters here only because it gives the copy a different set of noise
draws.

This is the known kill-margin class (seed-108 copy 8, seed-110 copy 73), now against the
shipped 4 × 0.75 train. Reconstructing copy 28's own draws reproduces it in an isolated
RefSim latch: the excursion pattern, its rate, and the failing phase. With strays, the
survival probability is **0.15 % per clear at four pulses and 0 / 60,000 at five**.

**Recommended fix:** a fifth pulse on the request clear only (`request_clear_pulses=5` in
`lib/kernel.py build_pipeline`). On `tick2.c` it costs one relay per request pair:
+25 neurons and +50 synapses, with no latency on the handshake path. The fix is not
applied here, because this task changes no netlist code. It needs a seed 108–110 Stage D
campaign before it ships.

## Inputs and rebuild

All read-only, under `/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:
`mixB_s108_c100_rr_replay870.json`, `mixB_s108_c100_rr_replay870_copy28.npz` and
`reports/rr_copy28.md`. The record's options are: generic datapath, true guards v2,
`start_relight_hops=5`, `relight_repair_delay`, `copy_requires_rail`, request and kernel
clears 4 × 0.75, `rate_robust` version 2 and no `retry_clear`. It rebuilds
(`stall_diag.build_tick_pipeline`) to **30,643 neurons / 55,936 synapses**, which matches
the record. The dump window is 30,138,019–32,262,185 (steps of 0.1 ms). The last commit is
at 30,438,019.

| Neuron | Role | In dump |
|---:|---|---|
| 15701 / 15702 | `c4_sel.req.c2_selr0.u` / `.v` (false, "consumed") | yes |
| 15703 / 15704 | `c4_sel.req.c2_selr1.u` / `.v` (true, "pending") | yes |
| 15706 | `c4_sel.req.c2_sel.received` | yes |
| 15709 | `c4_sel.req.c2_sel.k1.inh` (the clear train) | yes |
| 15868 / 15869 / 15870 | `c4_sel.relight.c2_sel.edge` / `.edge_inh` / `.veto` (ACT^d repair) | yes |
| 29297 / 15780 | `…r0.u.rate` / `…r1.u.rate` (`rate_robust` readouts) | yes |
| 15840 | `c4_sel.start.fd.d4` (START's delayed re-light) | **not captured** |

Wiring into the false latch (netlist, delay 18 steps on every edge):

| Edge | Nominal q | Copy 28 q |
|---|---:|---:|
| `u → v` (loop) | 3,621 | **4,050 (+11.8 %)** |
| `v → u` (loop) | 3,621 | 3,672 (+1.4 %) |
| `k1.inh → u` | −2,716 | −2,893 (+6.5 %) |
| `k1.inh → v` | −2,716 | −2,741 (+0.9 %) |
| `start.fd.d4 → u` (re-light) | 4,655 | 4,522 |
| `relight.edge → u` (repair) | 4,655 | 4,565 |

The thresholds are `u` −44.897 mV and `v` **−45.504 mV** (nominal −45.0); the biases are
−0.242 and +0.074 mV. There are no other inputs. Copy 28's draws can be reconstructed
exactly because `make_perturbed_sim` draws on the host from `np.random.default_rng(seed)`
in a fixed order: first the weights `(100, nnz)` in topology edge order, then V_th
`(100, n)`, then bias `(100, n)`. Only the stray stream is device-drawn.

## What the dump shows

### The latch's history in the window

The false `u` trains (gap ≤ 141 steps):

| Train | Lit by | Ended by |
|---|---|---|
| 30,148,856–30,198,089 | START (before the table) | clear 30,197,984 |
| 30,209,437–30,251,598 | START | clear 30,251,449 |
| 30,262,882–30,309,446 | START 30,262,573, +309 | clear 30,309,343 |
| 30,320,796–30,363,097 | START 30,320,486, +310 | clear 30,362,971 |
| **30,374,396–32,262,157** | START 30,374,085, +311 | **clear 30,423,365 fails** |

Every re-light came 309–311 steps after START, as the five-hop re-light is meant to. The
last train is continuous through the end of the dump: 45,474 `u` spikes, with no gap
longer than 104 steps. **The latch survived; it was not re-lit.** Nothing could have re-lit
it: START never fired again, and the repair edge 15868 has no spike in the window.

Every train settles to a **43-step period with `v` firing 18 steps after `u`**. It then
repeatedly leaves that orbit along the same path. The `u → v` offset runs 17, 16, 15, …, 2,
1; `v` passes `u`; the offset returns as 32, 29, 26, 24, 22, 20, 19, 18. Meanwhile the
period drops to 35 and comes back to 43. Excursion onsets come at 3.05, 3.79, 3.44, 3.07 and
**2.99 per second** in the five trains. Over the last train, the `u` intervals are
distributed as follows (steps: count):

> 34: 6, 35: 2,320, 36: 2,143, 37: 1,175, 38: 1,672, 39: 1,136, 40: 1,702, 41: 2,137,
> 42: 5,392, 43: 27,233, 44: 528

### The failed clear against the four successful ones

The clear train's `k1.inh` spikes were at 30,423,456, 502, 551 and 600, which is +91, +137,
+186 and +235 after `received`. The spacing is 46/49/49 steps, the same in all five clears
of the window. Each pulse reaches both members 18 steps later. The table counts steps
relative to `received` R = 30,423,365.

| | `u` spikes | `v` spikes | pulse arrivals |
|---|---|---|---|
| before the clear | −36, +0, +36, +72 (intervals 39, 38, 38, 37 → 36) | −30, +6, +41, +76 | |
| during the train | +107, +150, +204 | +112, +152, +199, +253 | **+109, +155, +204, +253** |
| after the train | +308, +400, then intervals 92, 62, 53, 49, 47, 45 … back to 43 | +361 … | |

- Before the clear, the latch was in an excursion, at offsets 6, 6, 5, 4 with a 35–37-step
  period.
- **Pulse 1** arrives 2 steps after `u` and 3 steps before `v`.
- **Pulse 2** arrives 5 and 3 steps after the pair.
- **Pulse 3** arrives 5 steps after `v`, on `u`'s spike step.
- **Pulse 4** arrives on `v`'s spike step.

Every pulse lands in the members' post-spike refractory period (n_ref = 22 steps), where
it can only delay them. The delays grow from about 7 to about 20 steps, so the pair's
period stretches to 43–54 steps. That stretch locks the pair to the train's 46–49-step
spacing, and the pair keeps firing just ahead of each next pulse. After the fourth pulse
the latch has slowed to intervals of about 100 steps, then recovers.

The healthy clears look different. At 30,309,343, for example, the latch sat on the
43-step orbit (`v` −6, `u` +19, `v` +36, `u` +61, `v` +78, `u` +103, `v` +126). Pulse 1
arrived at +111, 8 steps after `u`. It delayed `v`, and pulses 1 and 2 together stopped `u`
before `v`'s spike could re-fire it. The other healthy clears end the same way, within the
second pulse. The 30,251,449 clear also met the latch in an excursion, at offset ≈ 20 on
the return leg, and killed it on the third pulse.

### Why no repair fired, and what held

- **The repair cannot fire here, by design.** The repair (`kernel.py`, §10.5) starts from
  START's ACT^d chain. It re-*lights* a dark false rail and is vetoed by a live one
  (`relight.veto` ← false `u` and true `u`, −1,810 q onto the edge). Its delayed tap fired
  on schedule after each START: `edge_inh` 15869 at START + 1,140, + 1,131 and + 1,130
  steps (113 ms). The edge stayed silent each time because the false rail had re-lit, which
  is correct behaviour. After the failed clear no START occurred, so no tap occurred.
  Firing it would not have helped either, since it ignites the same rail that is stuck.
- **The fault stayed a fail-stop, not a wrong value.** The go chain's vetoes
  (`go.g0.pb.veto`, `go.g1.pa.veto`, `go.g1.pb.veto`) read false `u` directly, at 3,621 q.
  The producer `c2_sel` cannot overwrite the unconsumed value either. Its commit gate
  `c2_sel.cg.g1.pa` *requires* the false readout 29297, but it is *vetoed* by
  `c4_sel.req.c2_selr1.u` (pending), and that rail is live. No commit-gate edge fired.
  Accounting: 0 wrong, 0 duplicates, 1,005 outputs unfinished.

### `rate_robust`'s part: none in the mechanism

`rate_robust` leaves the request latch, its k1 train, the START re-light and the repair
relay unchanged; `kernel.py` pass 2 does not consult it. The netlist has **no edge from
any `.rate` readout into any request-latch member**. The readouts are feed-forward taps that
inherit k1's inhibition:

- 29297 drives only `c2_sel.cg.g1.pa.require0`, at 155 q.
- 15780 drives `c4_sel`'s go `require1` and `pb.edge`, also at 155 q.

The vetoes that block START, and the veto that keeps the repair away from a live rail, read
the latch member at full pulse drive. The guard timing that matters here is therefore the
default build's. The repair tap's +113 ms matches the `relight_repair_delay` design.
`rate_robust` matters only as a different netlist. Edge indices shift, so copy 28 of the
default build is a different set of draws, and that copy completed.

## Reproduction from copy 28's own draws (RefSim, isolated latch)

These are scratch probes, not committed. Each one is one `add_latch` with the table's
copy-28 weights, thresholds and biases. It is ignited by 4,522 q on `u`. The clear is
injected as `k1.inh` arrivals at the recorded offsets 0/46/95/144 (+49 per extra pulse),
with −2,893 q on `u` and −2,741 q on `v`.

1. **Free run, no strays.** The latch has a 44-step period with `v` 19 steps after `u`. Its
   ignition transient (intervals 73, 56, 50, 48, 46, 45, 45, 44) matches the dump's train
   starts step for step. Without noise, no excursion occurs.
2. **Excursion trigger.** A single 150 q kick never starts an excursion, at any of 44
   phases, on either member (175 q: 0 / 44). Kicks on `v` do start it: 200 q at 7 / 44
   phases, 225 q at 22, 250 q at 31, 300 q at 40. Two 150 q strays on `v` 5–40 steps apart
   start it at 41–44 / 44 phases. Kicks on `u` never start it, up to 300 q. Once started,
   the excursion follows the dump's trajectory exactly: offsets 17, 16, …, 1, then 32, 29,
   26, 24, 22, 20, 19.
3. **With mix-B strays** (5 Hz × 150 q on both members, 40 runs × 4.8 s), the steady
   period becomes 43. Excursions start at **2.80/s** (dump: 2.99/s), and the interval
   histogram reproduces the dump's: 35: 2,221, 36: 2,052, 37: 1,144, 38: 1,586, 39: 1,084,
   40: 1,626, 41: 2,031, 42: 5,013, 43: 28,365, 44: 978. Strays near threshold are what turn
   single strays into triggers.
4. **The failing phase.** A 300 q kick on `v` is followed by the recorded 4 × 0.75 train,
   started at each of 1,500 consecutive steps.
   - On the steady orbit: 0 / 1,500 survive.
   - On the excursion orbit: **5 / 1,500 survive.** All five have a `u → v` offset of 3–5,
     a preceding interval of 35–36, and the first pulse arriving within 3 steps of a `u`
     spike (1–3 steps after it, or 0–2 steps before the next one).
   - Copy 28 at its failure: offset 5, interval 35, lag 2, which is inside that window.
5. **Monte Carlo with strays and a uniformly random clear time.** Each run is ignited, gets
   strays throughout, and is cleared at a random step in 3,000–5,000. It counts as a
   survivor if `u` still fires 150 ms later.

   | clear | survivors / clears |
   |---|---:|
   | 3 × 0.75 | 1,866 / 40,000 (4.7 %) |
   | **4 × 0.75 (shipped)** | **59 / 40,000 (0.15 %)**; a second, independent run gives 28 / 20,000 |
   | 5 × 0.75 | **0 / 40,000** and 0 / 20,000 (95 % upper bound 5.0 × 10⁻⁵ per clear) |

   The pair is cleared once per `c4_sel` transaction. Over the ≈535 clears before the
   stall, 0.15 % gives a 1 − (1 − 0.0015)^535 ≈ **55 %** chance of at least one survival.
   A 1,000-tick run gives ≈ 77 %. A stall at tick 535 is what this predicts.

### How rare such a latch is

All 25 request false latches of each of the 100 copies were taken from this record's draws,
2,500 latches in total. Their noise-free periods have minimum 43, p1 44 and median 47.

- Only **6 latches** enter an excursion after a 300 q kick at any of 16 phases. They belong
  to copies 19, 24, 28, 59, 87 and 88. Copy 28's latch is the most susceptible: 15 / 16
  phases, against 1–13 for the others. Every one of the six has one member clearly favoured
  (an input edge +8 to +13 % and/or a threshold 0.4–0.7 mV low).
- At 200 strayed clears per latch, 4 × 0.75 left **0 / 500,000** survivors. At 20,000
  clears each, on the six susceptible latches plus the six fastest of the rest, **only copy
  28's latch survives** (28 / 20,000). Each of the other eleven gives 0 / 20,000.

So this mechanism accounts for copy 28 and for no other copy of this realization. Copy 24's
latch, `c11_sub.req.input`, is susceptible to the excursion but is killed in every trial,
so copy 24's stall needs its own explanation.

## Relation to the kill-margin class

The class shares its shape with the earlier cases. A request's false rail outlives its DONE
clear, both rails stay live, the false rail vetoes go, and no repair is involved. The cases
differ as follows.

| | Seed-108 copy 8 (base, 2026-09-20) | Seed-110 copy 73 | **Stage D rr copy 28** |
|---|---|---|---|
| clear | 3 × 0.75 | 3 × 0.75 | **4 × 0.75** |
| latch period | 33–41, wandering; 34 at the failure | 41 | **43 steady**; 35–38 in stray-triggered excursions |
| failure | slipped between pulses | survived the first clear | **entrained**: every pulse 0–5 steps after both members fired |

The fourth pulse (Juno 422309–11) did what was measured for it. In every strayed census
trial it killed every latch of this realization except copy 28's, and it kills copy 28's
in all but 0.15 % of clears. The residue is not a statically fast loop, which `test_kill_margin.py` and the 10,000-copy
survey cover. It is a **dynamic** fast state: an asymmetric latch whose two members drift
into near-coincidence, at a period (35–38) at the edge of what four pulses kill. The
surveys missed it because they draw a symmetric loop scale and threshold offset, and their
kills start on the latch's steady orbit.

The START re-light delay and the delayed ACT^d repair are not involved. The re-light worked
on all five STARTs, and the repair was correctly vetoed by a live false rail. The true
guards turned the fault into a fail-stop, as intended.

## Recommendation

**Use five pulses on the request clear only:** `request_clear_pulses=5` in
`lib/kernel.py build_pipeline`, the k1 train of each request pair in pass 2. Leave the
other kernel kill trains (`kernel_kill_pulses=4`) and the machine's (3 × 0.75) alone. The
knob already exists and is recorded in `build_options`.

- **Cost:** one relay neuron and two synapses per request pair. On `tick2.c` that is +25
  neurons / +50 synapses: 29,375 / 52,808 → 29,400 / 52,858 by default, and
  30,643 / 55,936 → 30,668 / 55,986 with `rate_robust`. The added edges carry ordinary
  pulse drive, and the existing `k1.inh` mirrors cover the readouts.
- **Latency:** nothing on the handshake path waits for the end of the clear. The train ends
  49 steps (4.9 ms) later, at R + ~302 instead of R + ~253.
- **Ordering, the hazard behind the 2026-09-20 wrong values:** the false rail's earliest
  re-light is START + 5 hops. In this copy, receipt → START was at least 686 steps (over
  129 receipts; median 1,834). Ignition reaches `u` about 25 steps before its first spike,
  at START + ~285, so ≥ ~970 steps after receipt. The fifth pulse arrives ~670 steps before
  that. The old race of a clear tail killing a fresh re-light needed the two within a few
  milliseconds.
- **Reload:** the extra charge delays a killed rail's earliest reliable re-ignition by
  about 50 steps. This was measured with an isolated, nominal kill and a 4,655 q ignition at
  11 phases. "All phases reload from" is counted after the last pulse arrives:

  | latch | 4 pulses | 5 pulses |
  |---|---:|---:|
  | copy 28 | 200 | 300 |
  | nominal | 350 | 400 |
  | loop −8 % / V_th +0.4 mV | 500 | 550 |
  | loop −12 % / V_th +0.6 mV | 600 | 650 |

  The slowest corner keeps a margin of about 20 steps against the ~670 available. A failed
  re-ignition there is the case the delayed ACT^d repair handles
  (`tests/test_relight_repair.py`).
- **Interactions:** with true guards, a dark pair still blocks rather than passes. The
  repair's live-false veto is unchanged. `copy_requires_rail` is unaffected because it is
  not a request path. `rate_robust` readouts and `require` neurons already mirror every
  `k1.inh` spike, so the fifth pulse clears them too. A rebuilt netlist is a new noise
  realization, so the pinned tick SHA-256 regressions in `tests/test_build_options.py` move
  and campaigns must be compared by per-seed totals.
- **Rejected alternatives:**
  - `retry_clear` also cures it in isolation (0 / 1,500 on the kicked orbit). It costs more
    per pair: a 12-hop delay chain, a gate and a 3 × 1.5 train. It also has two campaign
    records of turning stalls into wrong values.
  - 4 × 1.0, or the four pulses at 24-step spacing, also give 0 / 1,500. Both change the
    clear's charge or timing more than one extra pulse does, and stronger trains have
    failed in the kernel before.
  - Making the latch symmetric is not a fix at this depth: the asymmetry is the noise model.

**Validation required before shipping:**

1. Stage D on Juno with `request_clear_pulses=5`. `stage_d` has no CLI flag for it, so add
   one, or change the build default in the fix commit:
   `python -m drosophilos.bench.stage_d --ticks 1000 --seed {108,109,110} --backend torch-fast --device cuda --copies 100 --mix B [--rate-robust]`,
   for both builds.
2. The machine and multiplier suites (`test_machine.py`, `test_mul_diag.py`), and the
   relight-repair and true-guard suites, run with the new default.

## Regression test

Proposed `tests/test_request_clear_entrainment.py`, in RefSim with no kernel:

- **Fast.**
  1. Build one `add_latch` with copy 28's draws: `u→v` 4,050, `v→u` 3,672, V_th
     −44.897 / −45.504 mV, bias −0.242 / +0.074 mV.
  2. Drive its clear with the real `add_kill_train` at `pulses=4` and `pulses=5`. Use
     copy 28's relay and inhibitor weights (h1–h3 → inh 3,441 / 3,422 / 3,469;
     inh → u / v −2,893 / −2,741), or inject arrivals at the recorded 0/46/95/144 offsets.
  3. Ignite at step 10, kick `v` with 300 q at 5,020, and start the train at each of 1,500
     steps from 5,000.
  4. Assert: 4 pulses leave ≥ 1 survivor, all with a `u → v` offset ≤ 6 (measured 5 /
     1,500); 5 pulses leave 0. Also assert 0 / 1,500 at 4 pulses without the kick (steady
     orbit). Each sweep is a single batched RefSim of a few seconds.
- **Slow.** 40,000 clears with 5 Hz × 150 q strays and a random clear time. Assert 4 pulses
  > 0 (measured 59), 5 pulses 0, and an excursion-onset rate of 2–4/s for the free-running
  latch.
- **Reload.** A 5-pulse clear followed by a START-like 4,655 q ignition 650 steps after the
  last arrival. Assert it reloads at all 11 phases at the −12 % / +0.6 mV corner.

A kernel-level replay of the exact stall is not needed. The device stray stream depends on
batch size, and the isolated probes above reproduce the dump's statistics and its failing
phase from the recorded draws. No recapture is required.
